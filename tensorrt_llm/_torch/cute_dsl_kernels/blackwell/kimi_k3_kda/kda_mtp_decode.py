# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL device kernel for KDA multi-token speculative verify (conv-MTP).

Source-integrated from the ``KDA_decode_mtp`` kernel drop ("readable KDA
conv-MTP") and restructured for the latency-bound production regime (one CTA
per (head, request); TEP8 launches only ``H * N = 24`` CTAs on a 212-SM Rubin).
Specialized for the benchmarked contract: conv enabled, no bias, optional BF16
gated RMSNorm output, lower_bound gate, Q/K L2 norm, beta sigmoid, W=4,
K == V == 128, HV == H, ``TILE_V == V`` (one V pass), 512 threads.

Per request ``n`` the kernel processes an accepted count plus ``1 +
NUM_SPEC`` steps: it first *replays* the accepted draft tokens from the
``qkg/v/beta`` caches (raw conv inputs from the extended tail columns of
``cs_q/cs_k/cs_v``), then processes the ``1 + NUM_SPEC`` new tokens. The
recurrent state and the base conv windows are committed in place after the
first new (golden) token; the new spec tokens are cached for the next
round's replay. The pool invariant is therefore "state after the last
golden token; accepted drafts pending in the replay caches".

Structure (all stage boundaries are CTA barriers):

1. **Load phase.** Every warp issues its global loads up front, so the whole
   CTA waits for one DRAM round trip: the recurrent state ``h0`` straight into
   registers (the conv windows are read from the staged raw sequence, so the
   state load no longer has to wait for precompute), the raw conv inputs of
   every token (``x_q/x_k/x_v/g/beta`` for new tokens, the replay caches and
   ``cs_*`` tail columns for accepted drafts) into smem staging buffers, the
   conv weights, and the gated-RMSNorm operands for the epilogue.
2. **Per-token-parallel precompute.** Warp ``j`` computes the Q/K conv + SiLU +
   L2 norm of new token ``j`` and warp ``num_warps/2 + j`` its gate, beta and
   V conv, reading the width-4 windows from the staged raw sequence (rows
   ``0..W-2`` hold the committed window). Replay tokens were written to
   ``sQ/sK/sG/sBeta/sVall`` directly by the load phase. Conv-window commits
   and replay-cache writes are done by the warp that owns the token.
3. **Recurrence.** 16 warps x 8 value rows cover all 128 rows in one pass.
   Per token the eight ``S . k`` partial sums are reduced together with one
   multi-value butterfly (9 shuffles, depth 5: each lane keeps half of its
   rows per xor step and ends up with one complete row sum), ``vn`` is
   broadcast back with 8 shuffles, the state is updated, and ``S . q`` is
   reduced the same way -- 26 shuffles per token per warp instead of 88,
   with the same pairwise-add tree as the original row-pair butterflies, so
   results are bit-identical.
4. **Epilogue.** Gated RMSNorm and MXFP8 quantization read their operands from
   smem.

Measured on Rubin at N=2/H=12/num_spec=7 with the SSM state cold in L2 (the
E2E condition): 22.8 us -> 9.5 us per launch; with the state L2-resident
16.2 us -> 8.6 us (nsys). All outputs, the committed state, the conv pools
and the three replay caches are bit-exact against the previous kernel.

Vendoring delta vs the drop: the ``v_row_a``/``v_row_b`` readout indices in
the output stage are hoisted above the ``if/elif`` on ``USE_ZERO_ACCEPTED``
/ ``i_t >= commit_len``. In the drop they were first assigned inside the
*dynamic* ``elif`` branch, which the CuTe DSL cannot trace (names must
pre-exist before dynamic assignment), so every non-``USE_ZERO_ACCEPTED``
compile -- i.e. any replay round, and any non-benchmark shape -- failed with
``DSLRuntimeError: v_row_a is None``. The hoist is semantics-preserving.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

# 16 warps: the full 128-row recurrent state is processed in one pass (eight
# rows per warp) and the per-token precompute runs one (token, role) per warp.
NUM_THREADS = 512
TILE_K = 128


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the SM global timer for optional in-kernel stage profiling."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def kda_decode_mtp_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    o: cute.Tensor,
    output_scale: cute.Tensor,
    ht: cute.Tensor,
    qkg_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    smem_qk_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    precompute_control: cute.Tensor,
    TILE_V: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    scale_leading_dim: cutlass.Constexpr[int],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    USE_SETMAXREG: cutlass.Constexpr[bool],
    USE_REGULAR_METADATA: cutlass.Constexpr[bool],
    USE_PACKED_TOKEN_LAYOUT: cutlass.Constexpr[bool],
    USE_REG_Q_WEIGHTS: cutlass.Constexpr[bool],
    USE_ZERO_ACCEPTED: cutlass.Constexpr[bool],
    FUSE_PRECOMPUTE: cutlass.Constexpr[bool],
    RUNTIME_PRECOMPUTE_FLAG: cutlass.Constexpr[bool],
    FUSE_OUTPUT_NORM: cutlass.Constexpr[bool],
    QUANTIZE_OUTPUT: cutlass.Constexpr[bool],
    stage_timing: cute.Tensor,
    PROFILE_STAGES: cutlass.Constexpr[bool],
):
    """KDA MTP decode — SMEM pre-compute + register-resident state.

    With ``PROFILE_STAGES=True``, ``stage_timing`` must be an int64 tensor
    with at least ``HV * N * 4`` elements, indexed as
    ``(i_hv * grid_n + i_n) * 4``. With profiling off it is never accessed
    and the host may pass any placeholder tensor.
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, _ = cute.arch.block_idx()
    i_h = i_hv
    if cutlass.const_expr(PROFILE_STAGES):
        t_stage0 = read_globaltimer()
        # Pre-declare so the dynamic `run_precompute` branch below only
        # reassigns (first assignment inside a dynamic branch is untraceable;
        # see the module docstring on v_row_a/v_row_b).
        t_stage1 = Int64(0)
    if cutlass.const_expr(USE_REGULAR_METADATA):
        bos = i_n * (2 * NUM_SPEC + 1)
        eos = bos + (2 * NUM_SPEC + 1)
        slot = i_n
    elif cutlass.const_expr(USE_PACKED_TOKEN_LAYOUT):
        # Runtime verification packs exactly 1 + NUM_SPEC new rows per
        # request. Accepted drafts live in slot-indexed replay pools and do
        # not occupy rows in the projection outputs.
        bos = cutlass.Int32(0)
        eos = cutlass.Int32(0)
        slot = ssm_state_indices[i_n]
    else:
        bos = cu_seqlens[i_n]
        eos = cu_seqlens[i_n + 1]
        slot = ssm_state_indices[i_n]
    h0_idx = slot * HV + i_hv
    hk_off = i_h * K
    hv_off = i_hv * V
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        commit_len = 0
    else:
        if cutlass.const_expr(USE_PACKED_TOKEN_LAYOUT):
            commit_len = num_accepted_tokens[slot]
        else:
            commit_len = num_accepted_tokens[i_n]
        # Only NUM_SPEC drafts can be pending from the previous round. Clamp
        # so a malformed count cannot drive T_loop past the t_max-sized SMEM
        # buffers or the num_spec extents of the replay caches.
        if commit_len > NUM_SPEC:
            commit_len = cutlass.Int32(NUM_SPEC)
    if cutlass.const_expr(USE_PACKED_TOKEN_LAYOUT):
        # Existing body addresses every new row as bos + i_t, where i_t is
        # offset by commit_len after replay. Shift bos so those accesses map
        # to request-major packed rows without materialized cu_seqlens.
        bos = i_n * (1 + NUM_SPEC) - commit_len
        eos = bos + commit_len + 1 + NUM_SPEC
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        T_loop = 1 + NUM_SPEC
        t_max = 1 + NUM_SPEC
    else:
        T_loop = commit_len + 1 + NUM_SPEC
        t_max = 2 * NUM_SPEC + 1
    vec_size = TILE_K // 32
    num_v_tiles = V // TILE_V
    NUM_V_ROWS = TILE_V // (NUM_THREADS // 32)
    # The restructured kernel assumes one V pass with eight rows per warp,
    # smem-resident conv weights and no setmaxreg register juggling.
    assert num_v_tiles == 1 and NUM_V_ROWS == 8, (
        f"kda_decode_mtp_kernel expects TILE_V == V and 8 rows per warp, got TILE_V={TILE_V}, V={V}"
    )
    assert not USE_REG_Q_WEIGHTS and not USE_SETMAXREG, (
        "kda_decode_mtp_kernel reads conv weights from smem and does not use setmaxreg"
    )
    q_weight_elems = KERNEL_WIDTH * K
    v_weight_elems = KERNEL_WIDTH * V
    k_weight_base = q_weight_elems
    v_weight_base = q_weight_elems + KERNEL_WIDTH * K
    conv_weight_elems = q_weight_elems + KERNEL_WIDTH * K + v_weight_elems
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    # Preserve the offsets and bank mapping of the shared-memory buffers from
    # the source kernel. The fused norm below reduces within one warp and does
    # not need this legacy CTA-reduction scratch.
    smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * V,)), 16)
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((conv_weight_elems,)),
        16,
    )
    if cutlass.const_expr(FUSE_OUTPUT_NORM):
        # Epilogue operands are staged in smem during the load phase.
        sOnG = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((1 + NUM_SPEC) * V,)), 16)
        sOnW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)
    # Staged per-token inputs (raw conv sequence with the committed window
    # prepended in rows 0..W-2, raw gates, raw betas).
    sXQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXK = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * V,)), 16)
    sGin = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * K,)), 16)
    sBetaRaw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_decay = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_bk = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    # Multi-value warp-reduction scratch (per-row partials and reduction stages).
    r_part = cute.make_rmem_tensor(cute.make_layout((NUM_V_ROWS,), stride=(1,)), cutlass.Float32)
    r_red4 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    r_red2 = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), cutlass.Float32)
    r_state = cute.make_rmem_tensor(
        cute.make_layout((NUM_V_ROWS * vec_size,), stride=(1,)), cutlass.Float32
    )
    r_exp_A = cutlass.Float32(0.0)
    # Issue the recurrent-state loads first so their DRAM latency overlaps the
    # whole precompute stage (one V pass: TILE_V == V).
    for row in range(NUM_V_ROWS):
        v_row = warp_idx * NUM_V_ROWS + row
        for i in range(vec_size):
            if cutlass.const_expr(USE_FLAT_LAYOUT):
                r_state[row * vec_size + i] = cutlass.Float32(
                    h0[h0_idx, v_row, i * 32 + in_warp_tid]
                )
            else:
                r_state[row * vec_size + i] = cutlass.Float32(
                    h0[slot, i_hv, v_row, i * 32 + in_warp_tid]
                )
    if cutlass.const_expr(USE_REGULAR_METADATA or USE_PACKED_TOKEN_LAYOUT) or eos > bos:
        if cutlass.const_expr(FUSE_PRECOMPUTE or RUNTIME_PRECOMPUTE_FLAG):
            if cutlass.const_expr(RUNTIME_PRECOMPUTE_FLAG):
                run_precompute = precompute_control[0] != 0
            else:
                run_precompute = True
            if run_precompute:
                if tidx < K:
                    for w in range(KERNEL_WIDTH - 1):
                        sXQ[w * K + tidx] = cutlass.Float32(cs_q[slot, hk_off + tidx, w])
                        sXK[w * K + tidx] = cutlass.Float32(cs_k[slot, hk_off + tidx, w])
                for w in range(KERNEL_WIDTH):
                    if tidx < K:
                        sConvW[w * K + tidx] = cutlass.Float32(w_q[hk_off + tidx, w])
                        sConvW[k_weight_base + w * K + tidx] = cutlass.Float32(
                            w_k[hk_off + tidx, w]
                        )
                for ld in range(max(1, V * KERNEL_WIDTH // NUM_THREADS)):
                    flat = ld * NUM_THREADS + tidx
                    if flat < V * KERNEL_WIDTH:
                        sConvW[v_weight_base + flat] = cutlass.Float32(
                            w_v[hv_off + flat % V, flat // V]
                        )
                # Load phase: one round trip for every per-token input. Warp w
                # stages tokens w, w + num_warps, ...; replay tokens go straight
                # into sQ/sK/sG/sBeta/sVall, new tokens stage their raw inputs.
                if tidx < V:
                    for _w in range(KERNEL_WIDTH - 1):
                        sXV[_w * V + tidx] = cutlass.Float32(cs_v[slot, hv_off + tidx, _w])
                for _rep in range((t_max + NUM_THREADS // 32 - 1) // (NUM_THREADS // 32)):
                    _t = _rep * (NUM_THREADS // 32) + warp_idx
                    if _t < T_loop:
                        if _t < commit_len:
                            for i in range(vec_size):
                                k_idx = i * 32 + in_warp_tid
                                sQ[_t, k_idx] = cutlass.Float32(
                                    qkg_cache[slot, _t, 0, hk_off + k_idx]
                                )
                                sK[_t, k_idx] = cutlass.Float32(
                                    qkg_cache[slot, _t, 1, hk_off + k_idx]
                                )
                                sG[_t, k_idx] = cute.math.exp(
                                    cutlass.Float32(qkg_cache[slot, _t, 2, hk_off + k_idx]),
                                    fastmath=True,
                                )
                                sXQ[(_t + 3) * K + k_idx] = cutlass.Float32(
                                    cs_q[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + _t]
                                )
                                sXK[(_t + 3) * K + k_idx] = cutlass.Float32(
                                    cs_k[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + _t]
                                )
                                sXV[(_t + 3) * V + k_idx] = cutlass.Float32(
                                    cs_v[slot, hv_off + k_idx, KERNEL_WIDTH - 1 + _t]
                                )
                                sVall[_t * V + k_idx] = cutlass.Float32(
                                    v_cache[slot, _t, hv_off + k_idx]
                                )
                            if in_warp_tid == 0:
                                sBeta[_t] = cutlass.Float32(beta_cache[slot, _t, i_hv])
                        else:
                            _tok = bos + _t
                            for i in range(vec_size):
                                k_idx = i * 32 + in_warp_tid
                                if cutlass.const_expr(USE_FLAT_LAYOUT):
                                    sXQ[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_q[0, _tok, hk_off + k_idx]
                                    )
                                    sXK[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_k[0, _tok, hk_off + k_idx]
                                    )
                                    sXV[(_t + 3) * V + k_idx] = cutlass.Float32(
                                        x_v[0, _tok, hv_off + k_idx]
                                    )
                                else:
                                    sXQ[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_q[0, _tok, i_h, k_idx]
                                    )
                                    sXK[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_k[0, _tok, i_h, k_idx]
                                    )
                                    sXV[(_t + 3) * V + k_idx] = cutlass.Float32(
                                        x_v[0, _tok, i_hv, k_idx]
                                    )
                                sGin[_t * K + k_idx] = cutlass.Float32(g[0, _tok, i_hv, k_idx])
                            if in_warp_tid == 0:
                                sBetaRaw[_t] = cutlass.Float32(beta[0, _tok, i_hv])
                if cutlass.const_expr(FUSE_OUTPUT_NORM):
                    # Threads 224+ stage the gated-RMSNorm operands for the epilogue.
                    if tidx >= 224:
                        _pf_tid = tidx - 224
                        for _e in range(
                            ((1 + NUM_SPEC) * V + (NUM_THREADS - 224) - 1) // (NUM_THREADS - 224)
                        ):
                            _flat = _e * (NUM_THREADS - 224) + _pf_tid
                            if _flat < (1 + NUM_SPEC) * V:
                                _tok = _flat // V
                                _vv = _flat % V
                                sOnG[_flat] = cutlass.Float32(
                                    onorm_g[0, i_n * (1 + NUM_SPEC) + _tok, i_hv, _vv]
                                )
                        for _e in range((V + (NUM_THREADS - 224) - 1) // (NUM_THREADS - 224)):
                            _flat = _e * (NUM_THREADS - 224) + _pf_tid
                            if _flat < V:
                                sOnW[_flat] = cutlass.Float32(onorm_weight[_flat])
                cute.arch.barrier()
                # ---- per-token-parallel precompute ----
                num_warps = NUM_THREADS // 32
                half = num_warps // 2
                role = warp_idx // half
                r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_h]), fastmath=True)
                for _rep in range((1 + NUM_SPEC + half - 1) // half):
                    j = _rep * half + warp_idx % half
                    if j < 1 + NUM_SPEC:
                        t = commit_len + j
                        if role == 0:
                            for i in range(vec_size):
                                k_idx = i * 32 + in_warp_tid
                                r_conv_0 = cutlass.Float32(0.0)
                                for w in range(KERNEL_WIDTH):
                                    r_conv_0 += sXQ[(t + w) * K + k_idx] * sConvW[w * K + k_idx]
                                e0 = cute.math.exp(-r_conv_0, fastmath=True)
                                r_q[i] = r_conv_0 * cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                            sum_q = 0.0
                            for i in range(vec_size):
                                sum_q += r_q[i] * r_q[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_q += cute.arch.shuffle_sync_bfly(
                                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                            for i in range(vec_size):
                                r_q[i] = r_q[i] * rnorm_q_scaled
                            for i in range(vec_size):
                                k_idx = i * 32 + in_warp_tid
                                sQ[t, k_idx] = r_q[i]
                            for i in range(vec_size):
                                k_idx = i * 32 + in_warp_tid
                                r_conv = sXK[t * K + k_idx] * sConvW[k_weight_base + 0 * K + k_idx]
                                for w in range(1, KERNEL_WIDTH):
                                    r_conv += (
                                        sXK[(t + w) * K + k_idx]
                                        * sConvW[k_weight_base + w * K + k_idx]
                                    )
                                r_conv = r_conv * cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                                )
                                r_k[i] = r_conv
                            sum_k = 0.0
                            for i in range(vec_size):
                                sum_k += r_k[i] * r_k[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_k += cute.arch.shuffle_sync_bfly(
                                    sum_k, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                            for i in range(vec_size):
                                r_k[i] = r_k[i] * rnorm_k
                            for i in range(vec_size):
                                k_idx = i * 32 + in_warp_tid
                                sK[t, k_idx] = r_k[i]
                            if j == 0:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    for w in range(KERNEL_WIDTH - 1):
                                        cs_q[slot, hk_off + k_idx, w] = sXQ[(t + 1 + w) * K + k_idx]
                                        cs_k[slot, hk_off + k_idx, w] = sXK[(t + 1 + w) * K + k_idx]
                            else:
                                cache_pos = j - 1
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    qkg_cache[slot, cache_pos, 0, hk_off + k_idx] = r_q[i]
                                    qkg_cache[slot, cache_pos, 1, hk_off + k_idx] = r_k[i]
                                    cs_q[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = sXQ[
                                        (t + 3) * K + k_idx
                                    ]
                                    cs_k[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = sXK[
                                        (t + 3) * K + k_idx
                                    ]
                        else:
                            for i in range(vec_size):
                                k_idx = i * 32 + in_warp_tid
                                r_g_raw = sGin[t * K + k_idx]
                                r_g_raw = r_g_raw + cutlass.Float32(dt_bias[i_h * K + k_idx])
                                exp_A_x = r_exp_A * r_g_raw
                                sigmoid_val = cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
                                )
                                r_gk = lower_bound * sigmoid_val
                                sG[t, k_idx] = cute.math.exp(r_gk, fastmath=True)
                                r_decay[i] = r_gk
                            r_beta_t = cute.arch.rcp_approx(
                                cutlass.Float32(1.0) + cute.math.exp(-sBetaRaw[t], fastmath=True)
                            )
                            if in_warp_tid == 0:
                                sBeta[t] = r_beta_t
                            for i in range(vec_size):
                                v_idx = i * 32 + in_warp_tid
                                _v_conv = 0.0
                                for w in range(KERNEL_WIDTH):
                                    _v_conv += (
                                        sXV[(t + w) * V + v_idx]
                                        * sConvW[v_weight_base + w * V + v_idx]
                                    )
                                _v_conv = _v_conv * cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-_v_conv, fastmath=True)
                                )
                                sVall[t * V + v_idx] = _v_conv
                                r_bk[i] = _v_conv
                            if j == 0:
                                for i in range(vec_size):
                                    v_idx = i * 32 + in_warp_tid
                                    for w in range(KERNEL_WIDTH - 1):
                                        cs_v[slot, hv_off + v_idx, w] = sXV[(t + 1 + w) * V + v_idx]
                            else:
                                cache_pos = j - 1
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    qkg_cache[slot, cache_pos, 2, hk_off + k_idx] = r_decay[i]
                                    v_cache[slot, cache_pos, hv_off + k_idx] = r_bk[i]
                                    cs_v[slot, hv_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = sXV[
                                        (t + 3) * V + k_idx
                                    ]
                                if in_warp_tid == 0:
                                    beta_cache[slot, cache_pos, i_hv] = r_beta_t
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
            else:
                cute.arch.barrier()
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
        else:
            cute.arch.barrier()
            if cutlass.const_expr(PROFILE_STAGES):
                cute.arch.barrier()
                t_stage1 = read_globaltimer()
        cute.arch.barrier()
        for i_v in range(num_v_tiles):
            v_base = i_v * TILE_V
            if i_v > 0:
                for row in range(NUM_V_ROWS):
                    v_row = warp_idx * NUM_V_ROWS + row
                    for i in range(vec_size):
                        if cutlass.const_expr(USE_FLAT_LAYOUT):
                            r_state[row * vec_size + i] = cutlass.Float32(
                                h0[h0_idx, v_base + v_row, i * 32 + in_warp_tid]
                            )
                        else:
                            r_state[row * vec_size + i] = cutlass.Float32(
                                h0[slot, i_hv, v_base + v_row, i * 32 + in_warp_tid]
                            )
            i_t = 0
            while i_t < T_loop:
                r_beta_val = sBeta[i_t]
                for i_pair in range(vec_size // 2):
                    i0 = i_pair * 2
                    i1 = i_pair * 2 + 1
                    k_idx0 = i0 * 32 + in_warp_tid
                    k_idx1 = i1 * 32 + in_warp_tid
                    r_q[i0] = sQ[i_t, k_idx0]
                    r_q[i1] = sQ[i_t, k_idx1]
                    _k0 = sK[i_t, k_idx0]
                    _k1 = sK[i_t, k_idx1]
                    r_decay[i0] = sG[i_t, k_idx0]
                    r_decay[i1] = sG[i_t, k_idx1]
                    r_bk[i0], r_bk[i1] = cute.arch.mul_packed_f32x2(
                        (r_beta_val, r_beta_val), (_k0, _k1)
                    )
                    r_k[i0], r_k[i1] = cute.arch.mul_packed_f32x2(
                        (r_decay[i0], r_decay[i1]), (_k0, _k1)
                    )
                # ---- multi-value warp reduction over the eight rows ----
                assert NUM_V_ROWS == 8
                b4 = (in_warp_tid // 16) % 2
                b3 = (in_warp_tid // 8) % 2
                b2 = (in_warp_tid // 4) % 2
                my_row = in_warp_tid // 4
                for r in range(NUM_V_ROWS):
                    pa = cutlass.Float32(0.0)
                    pb = cutlass.Float32(0.0)
                    for _pi in range(vec_size // 2):
                        _p = _pi * 2
                        pa, pb = cute.arch.fma_packed_f32x2(
                            src_a=(r_state[r * vec_size + _p], r_state[r * vec_size + _p + 1]),
                            src_b=(r_k[_p], r_k[_p + 1]),
                            src_c=(pa, pb),
                        )
                    r_part[r] = pa + pb
                for j in range(4):
                    keep = r_part[j]
                    send = r_part[4 + j]
                    if b4 == 1:
                        keep = r_part[4 + j]
                        send = r_part[j]
                    r_red4[j] = keep + cute.arch.shuffle_sync_bfly(
                        send, offset=16, mask=-1, mask_and_clamp=31
                    )
                for j in range(2):
                    keep = r_red4[j]
                    send = r_red4[2 + j]
                    if b3 == 1:
                        keep = r_red4[2 + j]
                        send = r_red4[j]
                    r_red2[j] = keep + cute.arch.shuffle_sync_bfly(
                        send, offset=8, mask=-1, mask_and_clamp=31
                    )
                keep = r_red2[0]
                send = r_red2[1]
                if b2 == 1:
                    keep = r_red2[1]
                    send = r_red2[0]
                shk = keep + cute.arch.shuffle_sync_bfly(send, offset=4, mask=-1, mask_and_clamp=31)
                shk = shk + cute.arch.shuffle_sync_bfly(shk, offset=2, mask=-1, mask_and_clamp=31)
                shk = shk + cute.arch.shuffle_sync_bfly(shk, offset=1, mask=-1, mask_and_clamp=31)
                r_v_my = sVall[i_t * V + v_base + warp_idx * NUM_V_ROWS + my_row]
                vn_my = r_v_my - shk
                for r in range(NUM_V_ROWS):
                    vn_r = cute.arch.shuffle_sync(vn_my, r * 4)
                    qa = cutlass.Float32(0.0)
                    qb = cutlass.Float32(0.0)
                    for _pi in range(vec_size // 2):
                        _p = _pi * 2
                        vnbk0, vnbk1 = cute.arch.mul_packed_f32x2(
                            (vn_r, vn_r), (r_bk[_p], r_bk[_p + 1])
                        )
                        r_state[r * vec_size + _p], r_state[r * vec_size + _p + 1] = (
                            cute.arch.fma_packed_f32x2(
                                src_a=(r_decay[_p], r_decay[_p + 1]),
                                src_b=(r_state[r * vec_size + _p], r_state[r * vec_size + _p + 1]),
                                src_c=(vnbk0, vnbk1),
                            )
                        )
                        qa, qb = cute.arch.fma_packed_f32x2(
                            src_a=(r_state[r * vec_size + _p], r_state[r * vec_size + _p + 1]),
                            src_b=(r_q[_p], r_q[_p + 1]),
                            src_c=(qa, qb),
                        )
                    r_part[r] = qa + qb
                for j in range(4):
                    keep = r_part[j]
                    send = r_part[4 + j]
                    if b4 == 1:
                        keep = r_part[4 + j]
                        send = r_part[j]
                    r_red4[j] = keep + cute.arch.shuffle_sync_bfly(
                        send, offset=16, mask=-1, mask_and_clamp=31
                    )
                for j in range(2):
                    keep = r_red4[j]
                    send = r_red4[2 + j]
                    if b3 == 1:
                        keep = r_red4[2 + j]
                        send = r_red4[j]
                    r_red2[j] = keep + cute.arch.shuffle_sync_bfly(
                        send, offset=8, mask=-1, mask_and_clamp=31
                    )
                keep = r_red2[0]
                send = r_red2[1]
                if b2 == 1:
                    keep = r_red2[1]
                    send = r_red2[0]
                shq = keep + cute.arch.shuffle_sync_bfly(send, offset=4, mask=-1, mask_and_clamp=31)
                shq = shq + cute.arch.shuffle_sync_bfly(shq, offset=2, mask=-1, mask_and_clamp=31)
                shq = shq + cute.arch.shuffle_sync_bfly(shq, offset=1, mask=-1, mask_and_clamp=31)
                if in_warp_tid % 4 == 0:
                    v_row_my = warp_idx * NUM_V_ROWS + my_row
                    if cutlass.const_expr(USE_ZERO_ACCEPTED):
                        if cutlass.const_expr(FUSE_OUTPUT_NORM):
                            sVall[i_t * V + v_base + v_row_my] = cutlass.Float32(
                                cutlass.BFloat16(shq)
                            )
                        else:
                            o[0, bos + i_t, i_hv, v_base + v_row_my] = cutlass.BFloat16(shq)
                    elif i_t >= commit_len:
                        if cutlass.const_expr(FUSE_OUTPUT_NORM):
                            new_token = i_t - commit_len
                            sVall[new_token * V + v_base + v_row_my] = cutlass.Float32(
                                cutlass.BFloat16(shq)
                            )
                        else:
                            o[0, bos + i_t, i_hv, v_base + v_row_my] = cutlass.BFloat16(shq)
                if i_t == commit_len:
                    for row in range(NUM_V_ROWS):
                        v_row = warp_idx * NUM_V_ROWS + row
                        for i in range(vec_size):
                            if cutlass.const_expr(USE_FLAT_LAYOUT):
                                ht[h0_idx, v_base + v_row, i * 32 + in_warp_tid] = r_state[
                                    row * vec_size + i
                                ]
                            else:
                                ht[slot, i_hv, v_base + v_row, i * 32 + in_warp_tid] = r_state[
                                    row * vec_size + i
                                ]
                i_t = i_t + 1
        if cutlass.const_expr(FUSE_OUTPUT_NORM):
            # All 16 warps produced disjoint value rows above. One CTA
            # barrier makes the BF16-rounded rows visible before assigning
            # one complete token to each warp.
            cute.arch.barrier()
            token_idx = warp_idx
            while token_idx < 1 + NUM_SPEC:
                rms_partial = cutlass.Float32(0.0)
                for i in range(V // 32):
                    v_idx = i * 32 + in_warp_tid
                    value = sVall[token_idx * V + v_idx]
                    rms_partial += value * value
                for offset in [16, 8, 4, 2, 1]:
                    rms_partial += cute.arch.shuffle_sync_bfly(
                        rms_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                inv_rms = cute.math.rsqrt(
                    rms_partial / cutlass.Float32(V) + cutlass.Float32(onorm_eps),
                    fastmath=True,
                )
                output_row = i_n * (1 + NUM_SPEC) + token_idx
                normalized_values = cute.make_rmem_tensor(
                    cute.make_layout((V // 32,), stride=(1,)), cutlass.Float32
                )
                output_amax = cutlass.Float32(0.0)
                for i in range(V // 32):
                    v_idx = i * 32 + in_warp_tid
                    value = sVall[token_idx * V + v_idx]
                    gate = sOnG[token_idx * V + v_idx]
                    sigmoid_gate = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.math.exp(-gate)
                    )
                    normalized = value * inv_rms * sOnW[v_idx] * sigmoid_gate
                    if cutlass.const_expr(QUANTIZE_OUTPUT):
                        # Preserve the old standalone RMSNorm's BF16 output
                        # boundary in registers. The value goes straight to
                        # MXFP8 below; it is never written to or reloaded from
                        # HBM as BF16.
                        normalized = cutlass.Float32(cutlass.BFloat16(normalized))
                    normalized_values[i] = normalized
                    output_abs = cute.arch.fmax(normalized, -normalized)
                    output_amax = cute.arch.fmax(output_amax, output_abs)
                if cutlass.const_expr(QUANTIZE_OUTPUT):
                    output_amax = cute.arch.warp_reduction_max(output_amax)
                    output_amax = cute.arch.fmax(output_amax, cutlass.Float32(1.0e-10))
                    output_sf = (output_amax * cutlass.Float32(1.0 / 448.0)).to(
                        cutlass.Float8E8M0FNU
                    )
                    quant_scale = cute.arch.rcp_approx(cutlass.Float32(output_sf))
                    for i in range(V // 32):
                        v_idx = i * 32 + in_warp_tid
                        o[output_row, i_hv * V + v_idx] = cutlass.Float8E4M3FN(
                            normalized_values[i] * quant_scale
                        )
                    if in_warp_tid == 0:
                        scale_bytes = cute.make_tensor(
                            cute.recast_ptr(
                                output_scale.iterator,
                                dtype=cutlass.Float8E8M0FNU,
                            ),
                            cute.make_layout(
                                (output_scale.shape[1], scale_leading_dim, 4),
                                stride=(scale_leading_dim * 4, 4, 1),
                            ),
                        )
                        scale_bytes[i_hv // 4, output_row, i_hv % 4] = output_sf
                        if i_hv == HV - 1:
                            # The packed int32 ABI reserves four scale bytes
                            # per group. Make tail bytes deterministic when H
                            # is not divisible by four without a zero-fill
                            # kernel or pre-initialized output tensor.
                            for pad_head in range(HV, (HV + 3) // 4 * 4):
                                scale_bytes[pad_head // 4, output_row, pad_head % 4] = (
                                    cutlass.Float8E8M0FNU(0.0)
                                )
                else:
                    for i in range(V // 32):
                        v_idx = i * 32 + in_warp_tid
                        o[0, output_row, i_hv, v_idx] = cutlass.BFloat16(normalized_values[i])
                token_idx = token_idx + NUM_THREADS // 32
        if cutlass.const_expr(PROFILE_STAGES):
            cute.arch.barrier()
            t_stage2 = read_globaltimer()
            if tidx == 0:
                _, grid_n, _ = cute.arch.grid_dim()
                timing_base = (i_hv * grid_n + i_n) * 4
                stage_timing[timing_base + 0] = t_stage1 - t_stage0
                stage_timing[timing_base + 1] = t_stage2 - t_stage1
                stage_timing[timing_base + 2] = t_stage2 - t_stage0
                stage_timing[timing_base + 3] = t_stage0

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

"""Latency-scheduled CuTe DSL kernel for KDA multi-token speculative verify (packed rows).

Computes exactly what ``kda_mtp_decode.kda_decode_mtp_kernel`` computes on its runtime
contract -- packed token rows, fused gated RMSNorm with a bf16 output, 3xTF32 tcgen05 chunk
recurrence, fp32 or bf16 state, K = V = 128, conv width 4, NUM_SPEC <= 7 -- with the same
arithmetic in the same order, so outputs, committed state, conv pools and replay caches are
bitwise identical. See that module for the math (replay of accepted drafts, chunk WY form,
GEMM1/GEMM3/forward substitution/GEMM2, in-place state commit after the golden token, the
cluster-pair ``SPLIT_V == 2`` form). It also runs NUM_SPEC = 8, which that kernel does not
(see "NUM_SPEC = 8" below). The difference is the schedule of the latency-bound CTA:

* **Load phase.** The state slot is the first load. The slot-independent operands (new-token
  projections, conv weights, per-head constants, output-norm operands) are issued under it; the
  slot-dependent ones (the S0 TMA, the committed conv windows and every pending replay entry,
  accepted or not) as soon as it arrives. Each lane moves four consecutive channels of a row
  (8 B / 16 B vector loads) and keeps the raw words in registers until the accepted count fixes
  the row's smem position, so the CTA waits for one round trip per dependency level (slot, then
  operands) instead of one per staging step.
* **Precompute** works in smem only, with the q and k conv / SiLU / L2-norm chains of a token
  interleaved in one warp.
* **Tensor-core ordering.** The GEMM3 operand rows lie clear of the S0 boxes, so GEMM3 is issued
  before S0 moves into TMEM and runs under that copy; GEMM1 follows once its rows are written over
  the read boxes. While the tensor cores run, warps 1..15 stage Kbar, seed the GEMM2
  accumulator, build the B2 tiles and write the new tokens' replay-cache rows and conv-window
  commits (vectorized, from smem).

Optional (compile-time) variants, both bitwise identical to the plain kernel fed by
``k3_kda_fab``:

* **PDL.** Launched as a programmatic dependent of the projection GEMM: the slot, accepted count,
  S0 TMA, replay entries, conv windows and weights are loaded before ``griddepcontrol.wait``,
  only the new-token rows (and the TMEM allocation) after it.
* **Fused forget gate.** The kernel reads ``f_a`` (``k3_kda_fab`` in f_a mode) instead of the
  gate and computes ``bf16(f_a @ W_fb[h]^T)`` itself: warps 8..15 each own 16 ``W_fb`` rows
  (TMA-loaded into free scratch words up front) and run eight ``mma.sync`` k16 steps on f_a
  fragments loaded with the other new-token rows, before the precompute barrier.
* **MXFP8 output.** The norm epilogue also stores the MXFP8 form of the BF16 output for the
  o_proj low-M GEMM (``k3_fp8w_lowm_mxfp8``): E4M3 codes ``[T, HV * V]`` and one UE8M0 exponent
  per 1x32 group ``[T, HV * V / 32]``. A group is one 32-channel column block of a token row, one
  value per lane of the token's warp; amax, exponent and E4M3 conversion are the low-M GEMM's own,
  so the codes are bitwise what it would quantize the BF16 output to.

**NUM_SPEC = 8** (nine new tokens, up to 17 chunk tokens; every NUM_SPEC <= 7 instantiation is
unchanged). The work of the ninth new token and of the eighth replay entry goes to warps that hold
the register arrays of that role anyway: the TMA warp loads new token 8, warp 7 replay entry 7; the
f_b MMA warps run a second n8 tile for token 8 (same column, operands and k order as
``k3_kda_fab``'s tail, so the gate stays bitwise the fab kernel's; token 8's f_a row arrives in
shared memory by one bulk copy on the W_fb mbarrier); in the precompute token 8 rides along on warp
0 (q / k) and warp 9 (gate, beta, v), its dependency chains interleaved with the warp's own token,
and the norm epilogue walks the tokens in steps of its eight warps. The chunk stage runs tokens
``cb .. T_loop - 1``, whose q / k / gate / v / beta rows are staged at their chunk index ``t - cb``
(the raw conv inputs keep their sequence rows), so it is the NUM_SPEC <= 7 code on static
shared-memory addresses: ``cb = 0`` while at most 7 drafts were accepted (``T_loop <= 16``); after
a fully accepted round (``T_loop = 17``) ``cb = 1``, the first replayed draft is staged in the
spare row 16 and a pre-step folds it into the initial state with the exact single-token recurrence
in fp32 on CUDA cores (``S1 = S0 diag(e^g0) + beta0 (v0 - S0 diag(e^g0) k0) k0^T``, one row of S
per thread group, partial row sums exchanged through shared memory), so the 16-token chunk then
starts from ``S1``. With a bf16 state pool, GEMM1 reads ``S1`` truncated to tf32 (no lo sweep;
relative error below 2^-10, under the 2^-9 rounding of the bf16 state that the round commits); the
GEMM2 seed uses it in fp32.

The NUM_SPEC = 8 paths keep their code small (a dynamic loop over a warpgroup's output tokens, only
a thread's own four chunk tokens in the gate stage): the kernel is straight-line code, and with
every token unrolled it outgrew the instruction cache (warps then stall on instruction fetch in
every phase). The conv weights arrive by one load per thread from the quad's row (three loads
predicated on the quad may share their registers and wait for each other).
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import k3_fp8w_lowm_mxfp8 as mx_ops

NUM_THREADS = 512
#: TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER=1 (PDL builds): release the dependent grid (the o_proj GEMM)
#: right after ``griddepcontrol.wait`` instead of at exit, so it can become resident on the SMs this
#: grid leaves free and stream its weights during the recurrence.
#:
#: Why it is safe in the Kimi-K3 decode graph. The trigger follows this grid's own wait, so the
#: projection GEMM, ``k3_kda_fab`` and everything before them have completed when a dependent starts.
#: The grids that can start before this one completes are the o_proj GEMM (low-M or mid-M FP8
#: kernel), which before its wait only streams its weights and weight scales into shared memory (its
#: weight warp triggers at once), and behind it the o_proj MNNVL all-reduce, which calls
#: ``cudaGridDependencySynchronize`` before it reads or writes anything and triggers only after it.
#: Neither writes global memory this grid reads (state pool, conv windows, replay rows, new-token
#: rows) before its wait, and this grid's outputs (o, the committed state and conv windows) are read
#: only after a wait: A_mid, which reads the prefix and snapshot rows before its own wait, starts
#: only after the all-reduce's wait, i.e. after this grid completed.
EARLY_TRIGGER = os.environ.get("TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER", "0") == "1"
TILE_K = 128
# Chunk length padding (MMA N granularity); NUM_SPEC <= 7 -> T_loop <= 15. NUM_SPEC = 8 -> T_loop
# <= 17: the chunk takes the last 16 tokens, a pre-step the first one (T_loop = 17).
T_PAD = 16
#: Largest draft count (NUM_SPEC) the kernel supports.
MAX_NUM_SPEC = 8
# TMEM columns: D4 (16) + D2 halves (2 x 64) + S0 hi (128) + S0 lo (128) + D1 (48) + D3 (16)
TMEM_COLS = 512
# One 114 KB scratch region hosts, at different times, the TMA boxes of S0 / the committed state
# and the swizzled MMA operand tiles. Offsets in floats, all multiples of 256 (1024 B) for the
# 128B-swizzle atoms.
SCRATCH_FLOATS = 29184
OFF_B4HI = 10240  # 16 x 16    Bcoef hi, immediately followed by Kbar hi -> N=80 GEMM2 half-0 tile
OFF_B4LO = 12544  # 16 x 16    Bcoef lo, immediately followed by Kbar lo
OFF_B1K = 6144  # 16 x 128   K~hi (duplicate, N=16 tile for the S0_lo sweep; A2 reuses it)
OFF_B1 = 0  # 48 x 128   [K~hi; Q~; K~lo]
OFF_A2HI = 6144  # 128 x 16   V'^T hi
OFF_A2LO = 8192  # 128 x 16   V'^T lo
OFF_B2HI = 10496  # 128 x 16   Kbar hi
OFF_B2LO = 12800  # 128 x 16   Kbar lo
OFF_A3 = 16384  # 64 x 128   [Kf; Qf hi; Qf lo; -] (clear of the S0 boxes)
OFF_B3HI = 24576  # 16 x 128   Kb hi
OFF_B3LO = 26624  # 16 x 128   Kb lo
TP_STRIDE = 17  # padded row stride of the transposed (v|k, s) staging buffers
# FUSED_FB: this head's W_fb rows (128 x 128 BF16 = 32 KB, two 128B-swizzled TMA boxes of 64
# columns) in scratch words that stay unused until the cumulative-gate stage writes the GEMM3
# tiles (OFF_A3 onwards).
OFF_FB = 16384
FB_BOX_COLS = 64


def swz64_k16(n, k):
    """Element offset in a K-major 64B-swizzled fp32 tile with 16 columns."""
    off = n * 16 + k
    return off ^ (((off >> 5) & 3) << 2)


@dsl_user_op
def cvt_bf16x2(hi: cutlass.Float32, lo: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Pack two fp32 values into one bf16x2 word (round to nearest even, `lo` in the low half)."""
    w = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
    return w.bitcast(cutlass.Float32)


@dsl_user_op
def _bf16_bits(x: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Int32:
    """BF16 bits of ``x`` (round to nearest even) in the low half of a word, zero high half."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f32 z;\nmov.f32 z, 0f00000000;\ncvt.rn.bf16x2.f32 $0, z, $1;\n}",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3_of_bf16_lo(word, inv_scale, *, loc=None, ip=None) -> cutlass.Int32:
    """E4M3 code of the low BF16 half of ``word`` times the packed power-of-two ``inv_scale``:
    the per-element arithmetic of ``k3_fp8w_lowm_gemm._quant_bf16x8_to_e4m3`` (exact BF16
    multiply, satfinite round-to-nearest conversion)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(word).ir_value(loc=loc, ip=ip),
                cutlass.Int32(inv_scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b32 p;\n.reg .b16 h;\n"
            "mul.rn.bf16x2 p, $1, $2;\n"
            "cvt.rn.satfinite.e4m3x2.bf16x2 h, p;\n"
            "cvt.u32.u16 $0, h;\n"
            "and.b32 $0, $0, 255;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def st_global_u8(addr, v):
    _st_asm(
        "st.global.u8 [$0], $1;",
        "l,r",
        [Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
    )


def st_global_u32(addr, v):
    _st_asm(
        "st.global.b32 [$0], $1;",
        "l,r",
        [Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
    )


@dsl_user_op
def tcgen05_fence_before_thread_sync(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::before_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_fence_after_thread_sync(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def round_tf32(x: cutlass.Float32) -> cutlass.Float32:
    """Round to the tf32 grid (rna on mantissa bit 13) so hi/lo splits are exact."""
    b = x.bitcast(cutlass.Int32)
    b = (b + cutlass.Int32(0x1000)) & cutlass.Int32(-8192)
    return b.bitcast(cutlass.Float32)


def _asm_words(n, asm, constraints, operands, *, loc, ip):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * n),
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)]


@dsl_user_op
def ld_global_b32x2(addr: Int64, *, loc=None, ip=None):
    """Two 32-bit words (8 B, 8-byte aligned) from global memory, unconverted: the load has no
    consumer until its value is placed, so a warp keeps all of its loads in flight."""
    return _asm_words(
        2,
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_b32x4(addr: Int64, *, loc=None, ip=None):
    """Four 32-bit words (16 B, 16-byte aligned) from global memory, unconverted."""
    return _asm_words(
        4,
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_f32x4(addr: cutlass.Int32, *, loc=None, ip=None):
    """Four consecutive fp32 (16-byte aligned) from shared memory."""
    words = _asm_words(
        4,
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    return [w.bitcast(cutlass.Float32) for w in words]


@dsl_user_op
def _st_asm(asm, constraints, operands, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def st_global_b32x2(addr, a, b):
    _st_asm(
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        [Int64(addr).ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()],
    )


def st_global_b32x4(addr, a, b, c, d):
    _st_asm(
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        [Int64(addr).ir_value()] + [cutlass.Int32(x).ir_value() for x in (a, b, c, d)],
    )


def st_shared_f32x4(addr, a, b, c, d):
    _st_asm(
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        [cutlass.Int32(addr).ir_value()] + [cutlass.Float32(x).ir_value() for x in (a, b, c, d)],
    )


@dsl_user_op
def bulk_g2s(dst, src, nbytes, mbar, *, loc=None, ip=None):
    """``nbytes`` (a multiple of 16) from global ``src`` to this CTA's shared ``dst``, completing
    on the mbarrier at shared address ``mbar`` (``cp.async.bulk``)."""
    _st_asm(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        f"[$0], [$1], {nbytes}, [$2];",
        "r,l,r",
        [
            cutlass.Int32(dst).ir_value(loc=loc, ip=ip),
            Int64(src).ir_value(loc=loc, ip=ip),
            cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None):
    _st_asm("griddepcontrol.wait;", "", [], loc=loc, ip=ip)


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _st_asm("griddepcontrol.launch_dependents;", "", [], loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4(addr: cutlass.Int32, *, loc=None, ip=None):
    """Four 8x8 b16 matrices from shared memory (``ldmatrix.sync.aligned.m8n8.x4``)."""
    return _asm_words(
        4,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_b32(addr: Int64, *, loc=None, ip=None):
    """One 32-bit word (4-byte aligned) from global memory, unconverted."""
    return _asm_words(
        1, "ld.global.b32 $0, [$1];", "=r,l", [Int64(addr).ir_value(loc=loc, ip=ip)], loc=loc, ip=ip
    )[0]


@dsl_user_op
def mma_bf16_16816(a, b0, b1, c, *, loc=None, ip=None):
    """``d[16x8] = a[16x16] @ b[16x8] + c`` (``mma.sync`` m16n8k16, BF16 inputs, FP32 accumulate)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (*a, b0, b1)]
        + [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


def round_bf16(x: cutlass.Float32) -> cutlass.Float32:
    """``x`` rounded to BF16 (nearest even, ``cvt.rn``) and widened back to FP32."""
    w = cvt_bf16x2(x, x).bitcast(cutlass.Int32)
    return (w & cutlass.Int32(-65536)).bitcast(cutlass.Float32)


def _load_fb_b_fragments(fa, i_n, NEW_TOK, lane, r_fb_b):
    """B fragments of the f_b MMA (``m16n8k16``, K = 128 in eight k16 steps): column ``n`` holds
    the f_a row of this request's token whose packed row index is ``n`` mod 8 (the column
    k3_kda_fab's tail gives it); lane ``l`` holds rows ``k = 16 kk + 2 (l % 4) + {0, 1}`` and
    ``+ 8`` of column ``l / 4``. Columns without a token of this request read the request's
    last row here and are zeroed before the MMA."""
    jt = (lane // 4 - (i_n * NEW_TOK) % 8 + 8) % 8
    over = cutlass.Int32(jt >= NEW_TOK)
    tok = i_n * NEW_TOK + jt - over * (jt - (NEW_TOK - 1))
    kq = 2 * (lane % 4)
    for kk in range(8):
        r_fb_b[2 * kk] = ld_global_b32(gaddr(fa, (0, tok, 16 * kk + kq)))
        r_fb_b[2 * kk + 1] = ld_global_b32(gaddr(fa, (0, tok, 16 * kk + 8 + kq)))


def _load_new_token_row(
    x_q,
    x_k,
    x_v,
    g,
    onorm_g,
    beta,
    tok_row,
    i_h,
    i_hv,
    hk_off,
    hv_off,
    k4,
    r_new,
    r_newb,
    use_flat_layout,
    fused_fb,
):
    """Issue one lane's loads of a new token's projections: q / k / v / gate (unless the gate is
    computed in the kernel) / output gate four channels each into ``r_new``, beta into
    ``r_newb``."""
    if cutlass.const_expr(use_flat_layout):
        ld_row4(x_q, (0, tok_row, hk_off + k4), r_new, 0)
        ld_row4(x_k, (0, tok_row, hk_off + k4), r_new, 4)
        ld_row4(x_v, (0, tok_row, hv_off + k4), r_new, 8)
    else:
        ld_row4(x_q, (0, tok_row, i_h, k4), r_new, 0)
        ld_row4(x_k, (0, tok_row, i_h, k4), r_new, 4)
        ld_row4(x_v, (0, tok_row, i_hv, k4), r_new, 8)
    if cutlass.const_expr(not fused_fb):
        ld_row4(g, (0, tok_row, i_hv, k4), r_new, 12)
    ld_row4(onorm_g, (0, tok_row, i_hv, k4), r_new, 16)
    r_newb[0] = cutlass.Float32(beta[0, tok_row, i_hv])


def _precompute_qk(ts, rows, lane, sXQ, sXK, sConvW, sQ, sK, k_weight_base, K, W, scale):
    """q and k of the new tokens whose raw conv inputs start at sequence rows ``ts``, into chunk
    rows ``rows``: conv + SiLU + L2 norm (the arithmetic of the inline precompute), the chains
    of all tokens interleaved so that one warp overlaps their latencies."""
    vec = K // 32
    n = len(ts)
    r_q = [[None] * vec for _ in range(n)]
    r_k = [[None] * vec for _ in range(n)]
    for i in range(vec):
        k_idx = i * 32 + lane
        for a in range(n):
            t = ts[a]
            r_conv_0 = cutlass.Float32(0.0)
            for w in range(W):
                r_conv_0 += sXQ[(t + w) * K + k_idx] * sConvW[w * K + k_idx]
            r_conv = sXK[t * K + k_idx] * sConvW[k_weight_base + 0 * K + k_idx]
            for w in range(1, W):
                r_conv += sXK[(t + w) * K + k_idx] * sConvW[k_weight_base + w * K + k_idx]
            e0 = cute.math.exp(-r_conv_0, fastmath=True)
            r_q[a][i] = r_conv_0 * cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
            r_k[a][i] = r_conv * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
            )
    sum_q = [cutlass.Float32(0.0)] * n
    sum_k = [cutlass.Float32(0.0)] * n
    for i in range(vec):
        for a in range(n):
            sum_q[a] += r_q[a][i] * r_q[a][i]
            sum_k[a] += r_k[a][i] * r_k[a][i]
    for offset in [16, 8, 4, 2, 1]:
        for a in range(n):
            sum_q[a] += cute.arch.shuffle_sync_bfly(
                sum_q[a], offset=offset, mask=-1, mask_and_clamp=31
            )
            sum_k[a] += cute.arch.shuffle_sync_bfly(
                sum_k[a], offset=offset, mask=-1, mask_and_clamp=31
            )
    for a in range(n):
        rnorm_q_scaled = cute.math.rsqrt(sum_q[a] + 1e-06, fastmath=True) * scale
        rnorm_k = cute.math.rsqrt(sum_k[a] + 1e-06, fastmath=True)
        for i in range(vec):
            k_idx = i * 32 + lane
            sQ[rows[a], k_idx] = r_q[a][i] * rnorm_q_scaled
            sK[rows[a], k_idx] = r_k[a][i] * rnorm_k


def _precompute_gbv(
    ts,
    rows,
    lane,
    sGin,
    sDtBias,
    sAlog,
    sG,
    sBetaRaw,
    sBeta,
    sXV,
    sConvW,
    sVall,
    v_weight_base,
    K,
    V,
    W,
    lower_bound,
):
    """Gate, beta and v of the new tokens at sequence rows ``ts`` into chunk rows ``rows`` (the
    arithmetic of the inline precompute, the tokens interleaved). Every lane stores the token's
    beta (the same value)."""
    vec = K // 32
    n = len(ts)
    r_exp_A = cute.math.exp(sAlog[0], fastmath=True)
    for i in range(vec):
        k_idx = i * 32 + lane
        for a in range(n):
            r_g_raw = sGin[ts[a] * K + k_idx]
            r_g_raw = r_g_raw + sDtBias[k_idx]
            exp_A_x = r_exp_A * r_g_raw
            sigmoid_val = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
            )
            sG[rows[a], k_idx] = lower_bound * sigmoid_val
    for a in range(n):
        sBeta[rows[a]] = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-sBetaRaw[ts[a]], fastmath=True)
        )
    for i in range(vec):
        v_idx = i * 32 + lane
        for a in range(n):
            _v_conv = cutlass.Float32(0.0)
            for w in range(W):
                _v_conv += sXV[(ts[a] + w) * V + v_idx] * sConvW[v_weight_base + w * V + v_idx]
            sVall[rows[a] * V + v_idx] = _v_conv * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-_v_conv, fastmath=True)
            )


def gaddr(tensor, coord):
    """Global address of `tensor[coord]`."""
    return (tensor.iterator + cute.crd2idx(coord, tensor.layout)).toint()


def saddr(tensor, offset):
    """Shared-memory address of element `offset` of an smem tensor (row-major offset)."""
    return (tensor.iterator + offset).toint()


def ld_row4(tensor, coord, words, base):
    """Issue the load of four consecutive elements of `tensor` (16- or 32-bit) at `coord` into the
    raw-word buffer `words[base : base + 4]` (two packed words for a 16-bit type)."""
    if cutlass.const_expr(tensor.element_type.width == 16):
        w = ld_global_b32x2(gaddr(tensor, coord))
        words[base] = w[0]
        words[base + 1] = w[1]
    else:
        w = ld_global_b32x4(gaddr(tensor, coord))
        for i in range(4):
            words[base + i] = w[i]


def unpack4(dtype, words, base):
    """Four fp32 values from the raw words of `ld_row4` (bf16 -> fp32 is exact)."""
    if cutlass.const_expr(dtype.width == 16):
        assert dtype == cutlass.BFloat16, "16-bit operands are bf16"
        w0 = words[base]
        w1 = words[base + 1]
        return [
            (w0 << 16).bitcast(cutlass.Float32),
            (w0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
            (w1 << 16).bitcast(cutlass.Float32),
            (w1 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
        ]
    return [words[base + i].bitcast(cutlass.Float32) for i in range(4)]


def st_row4(tensor, coord, v):
    """Store four fp32 values `v` as consecutive elements of `tensor` (rounded to nearest even
    for bf16, like a scalar `.to(BFloat16)`)."""
    if cutlass.const_expr(tensor.element_type.width == 16):
        assert tensor.element_type == cutlass.BFloat16, "16-bit caches are bf16"
        st_global_b32x2(
            gaddr(tensor, coord),
            cvt_bf16x2(v[1], v[0]).bitcast(cutlass.Int32),
            cvt_bf16x2(v[3], v[2]).bitcast(cutlass.Int32),
        )
    else:
        st_global_b32x4(gaddr(tensor, coord), *[x.bitcast(cutlass.Int32) for x in v])


@cute.kernel
def kda_decode_mtp_v2_kernel(
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
    k_cache: cute.Tensor,
    g_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    smem_qk_layout: cute.Layout,
    tiled_mma1: cute.TiledMma,
    tiled_mma3: cute.TiledMma,
    tiled_mma1k: cute.TiledMma,
    smem_layout_b1: cute.ComposedLayout,
    smem_layout_a2: cute.ComposedLayout,
    smem_layout_a3: cute.ComposedLayout,
    smem_layout_b3: cute.ComposedLayout,
    smem_layout_b1k: cute.ComposedLayout,
    tiled_mma4: cute.TiledMma,
    tma_atom_t: cute.CopyAtom,
    mT_tma: cute.Tensor,
    box_layout: cute.ComposedLayout,
    tma_atom_l: cute.CopyAtom,
    mS_tma: cute.Tensor,
    tiled_mma2h: cute.TiledMma,
    smem_layout_b2h: cute.ComposedLayout,
    box_layout_t: cute.ComposedLayout,
    tiled_mma2q: cute.TiledMma,
    smem_layout_b2q: cute.ComposedLayout,
    tiled_mma_f32a: cute.TiledMma,
    fa: cute.Tensor,
    tma_atom_fb: cute.CopyAtom,
    mFb_tma: cute.Tensor,
    fb_layout: cute.ComposedLayout,
    xq_addr: Int64,
    xs_addr: Int64,
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    SPLIT_V: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[bool],
    FUSED_FB: cutlass.Constexpr[bool],
    MXFP8_OUT: cutlass.Constexpr[bool] = False,
):
    """KDA MTP verify, packed-token / fused-norm contract (see the module docstring).

    Grid ``(HV, N, SPLIT_V)``: one (head, request) tile per CTA, or per cluster pair of CTAs
    with ``SPLIT_V == 2`` (each CTA seeds and stores half of the state rows).

    ``PDL``: launched as a programmatic dependent; everything that does not depend on this
    layer's projections (slot, accepted count, state TMA, replay entries, conv windows, weights)
    is loaded before ``griddepcontrol.wait``, the new-token rows and TMEM after it.
    ``FUSED_FB``: ``fa`` holds the BF16 ``f_a`` rows ``[1, T, 128]`` instead of the forget gate;
    the CTA computes its head's gate ``bf16(f_a @ W_fb[h]^T)`` with ``mma.sync`` (``W_fb`` rows
    TMA-loaded via ``tma_atom_fb`` before ``griddepcontrol.wait``, f_a fragments loaded with the
    other new-token rows): FP32 accumulation, the operands, positions and k order of
    ``k3_kda_fab``'s tail, so bitwise its forget gate.
    ``MXFP8_OUT``: also store the MXFP8 form of ``o`` at ``xq_addr`` (E4M3 ``[T, HV * V]``) and
    ``xs_addr`` (UE8M0 ``[T, HV * V / 32]``); the addresses are unused otherwise.
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    i_hv, i_n, vh = cute.arch.block_idx()
    i_h = i_hv
    # The request's state slot is the first load of the kernel; its round trip overlaps the
    # slot-independent loads issued below.
    slot32 = ssm_state_indices[i_n]
    hk_off = i_h * K
    hv_off = i_hv * V
    # Recurrent-state element type (fp32 or bf16). The TMA boxes have 128 B swizzled rows,
    # i.e. BOX_COLS = 32 fp32 or 64 bf16 columns; the state tile is NUM_SLABS such boxes.
    STATE_DTYPE = h0.element_type
    STATE_BF16 = STATE_DTYPE == cutlass.BFloat16
    # 3xTF32 recurrence (hi/lo splits on every product that feeds the state); S0_LO: a fp32
    # state needs the lo part of S0 (a bf16 state is exact in tf32).
    S0_LO = not STATE_BF16
    N1 = 3 * T_PAD  # GEMM1 N: [K~hi; Q~; K~lo]
    CONV_DTYPE = cs_q.element_type
    K_CACHE_DTYPE = k_cache.element_type
    V_CACHE_DTYPE = v_cache.element_type
    BETA_CACHE_DTYPE = beta_cache.element_type
    BOX_COLS = 128 // (STATE_DTYPE.width // 8)
    NUM_SLABS = K // BOX_COLS
    t_max = 2 * NUM_SPEC + 1
    NEW_TOK = 1 + NUM_SPEC
    vec_size = TILE_K // 32
    # Load-phase warp roles: warp j < NEW_TOK stages new token j, warp REPLAY_WARP0 + r the
    # pending draft r (all NUM_SPEC entries are loaded, the accepted ones placed), the last warp
    # the committed conv windows and the S0 TMA. WIDE (NUM_SPEC = 8): new token 8 also goes to
    # the last warp and replay entry 7 to warp 7.
    REPLAY_WARP0 = NUM_THREADS // 64
    TMA_WARP = NUM_THREADS // 32 - 1
    WIDE = NUM_SPEC == MAX_NUM_SPEC
    if cutlass.const_expr(WIDE):
        assert NEW_TOK == REPLAY_WARP0 + 1 and REPLAY_WARP0 + NUM_SPEC - 1 == TMA_WARP
    else:
        assert NUM_SPEC <= 7 and t_max <= T_PAD, "chunk form pads the token axis to 16"
        assert NEW_TOK <= REPLAY_WARP0 and REPLAY_WARP0 + NUM_SPEC <= TMA_WARP
    # Bound of the staged token rows inside the 16-row chunk loops (all 16 exist when WIDE).
    t_lim = T_PAD if WIDE else t_max
    assert K == 128 and V == 128 and KERNEL_WIDTH == 4 and NUM_THREADS == 512, (
        "the thread mappings assume K = V = 128, conv width 4 and 512 threads"
    )
    k_weight_base = KERNEL_WIDTH * K
    v_weight_base = 2 * KERNEL_WIDTH * K
    conv_weight_elems = 2 * KERNEL_WIDTH * K + KERNEL_WIDTH * V
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * V,)), 16)
    sConvW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((conv_weight_elems,)), 16)
    sOnG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NEW_TOK * V,)), 16)
    sOnW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)
    # Staged per-token inputs (raw conv sequence with the committed window
    # prepended in rows 0..W-2, raw gates, raw betas).
    sXQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXK = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * V,)), 16)
    sGin = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * K,)), 16)
    sBetaRaw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    sDtBias = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sAlog = smem.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), 16)
    tmem_holder = smem.allocate_array(cutlass.Int32, 1)
    mbar1 = smem.allocate_array(cutlass.Int64, 1)
    mbar2 = smem.allocate_array(cutlass.Int64, 1)
    mbar_ld = smem.allocate_array(cutlass.Int64, 1)
    mbar2b = smem.allocate_array(cutlass.Int64, 1)
    mbar_fb = smem.allocate_array(cutlass.Int64, 1)
    sScratch = smem.allocate_tensor(cutlass.Float32, cute.make_layout((SCRATCH_FLOATS,)), 1024)
    sB1 = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B1, smem_layout_b1.inner, dtype=cutlass.Float32),
        smem_layout_b1.outer,
    )
    sA2hi = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_A2HI, smem_layout_a2.inner, dtype=cutlass.Float32),
        smem_layout_a2.outer,
    )
    sA2lo = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_A2LO, smem_layout_a2.inner, dtype=cutlass.Float32),
        smem_layout_a2.outer,
    )
    sA3 = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_A3, smem_layout_a3.inner, dtype=cutlass.Float32),
        smem_layout_a3.outer,
    )
    sB3hi = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B3HI, smem_layout_b3.inner, dtype=cutlass.Float32),
        smem_layout_b3.outer,
    )
    sB3lo = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B3LO, smem_layout_b3.inner, dtype=cutlass.Float32),
        smem_layout_b3.outer,
    )
    sB1k = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B1K, smem_layout_b1k.inner, dtype=cutlass.Float32),
        smem_layout_b1k.outer,
    )
    sB1kf = cute.make_tensor(sScratch.iterator + OFF_B1K, cute.make_layout((T_PAD * 128,)))
    # Flat (unswizzled-pointer) views for element writes with closed-form swizzle offsets.
    sB1f = cute.make_tensor(sScratch.iterator + OFF_B1, cute.make_layout((48 * 128,)))
    sA2hif = cute.make_tensor(sScratch.iterator + OFF_A2HI, cute.make_layout((128 * 16,)))
    sA2lof = cute.make_tensor(sScratch.iterator + OFF_A2LO, cute.make_layout((128 * 16,)))
    sB2hif = cute.make_tensor(sScratch.iterator + OFF_B2HI, cute.make_layout((128 * 16,)))
    sB2lof = cute.make_tensor(sScratch.iterator + OFF_B2LO, cute.make_layout((128 * 16,)))
    sB4hif = cute.make_tensor(sScratch.iterator + OFF_B4HI, cute.make_layout((T_PAD * T_PAD,)))
    sB4lof = cute.make_tensor(sScratch.iterator + OFF_B4LO, cute.make_layout((T_PAD * T_PAD,)))
    sA3f = cute.make_tensor(sScratch.iterator + OFF_A3, cute.make_layout((64 * 128,)))
    sB3hif = cute.make_tensor(sScratch.iterator + OFF_B3HI, cute.make_layout((16 * 128,)))
    sB3lof = cute.make_tensor(sScratch.iterator + OFF_B3LO, cute.make_layout((16 * 128,)))
    # S0 / committed-state TMA boxes at the start of the scratch region (fp32 and bf16 views)
    sBoxf = cute.make_tensor(sScratch.iterator, cute.make_layout((V * K,)))
    sBoxh = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator, dtype=cutlass.BFloat16), cute.make_layout((V * K,))
    )
    # GEMM3 result [L; B hi; B lo; -] as a plain (64 x 16) row-major matrix.
    sLB = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((4 * T_PAD, T_PAD), stride=(T_PAD, 1)), 16
    )
    sEc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    # V'^T (v, s) and Kbar^T (k, s) staged with a padded row stride (conflict-free for both
    # the row-per-thread producers and the 16-lanes-per-row tile writers).
    sVpT = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V * TP_STRIDE,)), 16)
    sKbT = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K * TP_STRIDE,)), 16)
    # Cumulative gates G_t (per k) reuse the raw-gate staging buffer (no longer needed after the
    # precompute).
    sE = cute.make_tensor(sGin.iterator, cute.make_layout((t_max * K,)))

    # ---- load phase. Every load is issued before any of its consumers, so the CTA waits for its
    #      round trips in parallel: the slot-independent operands (new-token projections, conv
    #      weights, per-head constants, output-norm operands) under the slot load, the
    #      slot-dependent ones (S0 TMA, committed conv windows, every replay entry) as soon as the
    #      slot arrives; only the smem placement of the new tokens and accepted drafts waits for
    #      the accepted count. Each lane moves four consecutive channels (8 B for bf16, 16 B for
    #      fp32) per row, held as raw words until the row's smem position is known.
    ch = tidx % 128
    quad = tidx // 128
    k4 = 4 * in_warp_tid
    # (1) slot-independent: conv weights ([ch][w] rows of 16 B: threads [0, 384) one row each),
    #     dt_bias / output-norm weight / A_log (threads [384, 512)), new token j = warp
    #     (warps < NEW_TOK)
    r_w = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Int32)
    if cutlass.const_expr(WIDE):
        # one load from the quad's weight row: three loads predicated on the quad can share
        # their destination registers and then wait for each other
        w_addr = gaddr(w_q, (hk_off + ch, 0))
        if quad == 1:
            w_addr = gaddr(w_k, (hk_off + ch, 0))
        if quad == 2:
            w_addr = gaddr(w_v, (hv_off + ch, 0))
        if quad < 3:
            if cutlass.const_expr(w_q.element_type.width == 16):
                w2 = ld_global_b32x2(w_addr)
                r_w[0] = w2[0]
                r_w[1] = w2[1]
            else:
                w4 = ld_global_b32x4(w_addr)
                for i in cutlass.range_constexpr(4):
                    r_w[i] = w4[i]
    else:
        if quad == 0:
            ld_row4(w_q, (hk_off + ch, 0), r_w, 0)
        if quad == 1:
            ld_row4(w_k, (hk_off + ch, 0), r_w, 0)
        if quad == 2:
            ld_row4(w_v, (hv_off + ch, 0), r_w, 0)
    r_c = cute.make_rmem_tensor(cute.make_layout((3,)), cutlass.Float32)
    if quad == 3:
        r_c[0] = cutlass.Float32(dt_bias[hk_off + ch])
        r_c[1] = cutlass.Float32(onorm_weight[ch])
        r_c[2] = cutlass.Float32(A_log[i_h])
    r_new = cute.make_rmem_tensor(cute.make_layout((5 * 4,)), cutlass.Int32)
    r_newb = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
    if cutlass.const_expr(WIDE):
        # new token tok_j of this warp: warps 0..7 tokens 0..7, the TMA warp token 8
        is_tok = (warp_idx < REPLAY_WARP0) | (warp_idx == TMA_WARP)
        tok_j = warp_idx
        if warp_idx == TMA_WARP:
            tok_j = cutlass.Int32(NEW_TOK - 1)
    else:
        is_tok = warp_idx < NEW_TOK
        tok_j = warp_idx
    tok_row = i_n * NEW_TOK + tok_j
    new_rows = (x_q, x_k, x_v, g, onorm_g, beta)
    # FUSED_FB: warp REPLAY_WARP0 + mi computes the forget gate of W_fb rows [16 mi, 16 mi + 16)
    # of this head; its f_a B fragments are loaded with the other new-token rows. WIDE: the TMA
    # warp also loads token 8's f_a row (four channels per lane into the gate words of r_new, which
    # FUSED_FB leaves unused) for the second f_b tile.
    r_fb_b = cute.make_rmem_tensor(cute.make_layout((16,)), cutlass.Int32)
    if cutlass.const_expr(not PDL):
        if is_tok:
            _load_new_token_row(
                *new_rows,
                tok_row,
                i_h,
                i_hv,
                hk_off,
                hv_off,
                k4,
                r_new,
                r_newb,
                USE_FLAT_LAYOUT,
                FUSED_FB,
            )
        if cutlass.const_expr(FUSED_FB):
            if warp_idx >= REPLAY_WARP0:
                _load_fb_b_fragments(fa, i_n, NEW_TOK, in_warp_tid, r_fb_b)
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar1, 1)
            cute.arch.mbarrier_init(mbar2, 1)
            cute.arch.mbarrier_init(mbar2b, 1)
            cute.arch.mbarrier_init(mbar_ld, 1)
            if cutlass.const_expr(FUSED_FB):
                cute.arch.mbarrier_init(mbar_fb, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    if cutlass.const_expr(FUSED_FB):
        # This head's W_fb rows into scratch (independent of the previous grid).
        sFb = cute.make_tensor(
            cute.recast_ptr(sScratch.iterator + OFF_FB, fb_layout.inner, dtype=cutlass.BFloat16),
            fb_layout.outer,
        )
        gFb = cute.local_tile(mFb_tma, (TILE_K, FB_BOX_COLS), (None, None))
        tFbs, tFbg = cpasync.tma_partition(
            tma_atom_fb,
            0,
            cute.make_layout(1),
            cute.group_modes(sFb, 0, 2),
            cute.group_modes(gFb, 0, 2),
        )
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_fb, TILE_K * TILE_K * 2 + (TILE_K * 2 if WIDE else 0)
                )
            for b in cutlass.range_constexpr(TILE_K // FB_BOX_COLS):
                cute.copy(tma_atom_fb, tFbg[(None, i_hv, b)], tFbs[(None, b)], tma_bar_ptr=mbar_fb)
        if cutlass.const_expr(WIDE):
            # token 8's f_a row (128 raw BF16 = 64 words) for the second f_b tile, one bulk copy
            # into the Kbar staging words (unused until the chunk stage) on mbar_fb; with PDL it
            # is issued after griddepcontrol.wait (it is the previous grid's output)
            sFa8w = cute.make_tensor(
                cute.recast_ptr(sKbT.iterator, dtype=cutlass.Int32),
                cute.make_layout((TILE_K // 2,)),
            )
            fa8_src = gaddr(fa, (0, i_n * NEW_TOK + (NEW_TOK - 1), 0))
            if cutlass.const_expr(not PDL):
                if warp_idx == TMA_WARP:
                    with cute.arch.elect_one():
                        bulk_g2s(saddr(sKbT, 0), fa8_src, TILE_K * 2, mbar_fb.toint())
    # V2 state views may place many layer pages between adjacent slots. Widen
    # the slot before indexing: slot * state stride can exceed INT32_MAX.
    slot = Int64(slot32)
    h0_idx = slot * HV + i_hv
    # (2) slot-dependent: replay entry r = warp - REPLAY_WARP0 (every pending entry, accepted or
    #     not; WIDE: entry 7 on warp 7), the committed conv windows (TMA warp: three columns of q,
    #     k and v)
    rep = warp_idx - REPLAY_WARP0
    if cutlass.const_expr(WIDE):
        is_rep = ((warp_idx >= REPLAY_WARP0) & (warp_idx < TMA_WARP)) | (
            warp_idx == REPLAY_WARP0 - 1
        )
        if warp_idx == REPLAY_WARP0 - 1:
            rep = cutlass.Int32(NUM_SPEC - 1)
    else:
        is_rep = (warp_idx >= REPLAY_WARP0) & (rep < NUM_SPEC)
    r_rep = cute.make_rmem_tensor(cute.make_layout((6 * 4,)), cutlass.Int32)
    r_repb = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
    if is_rep:
        ld_row4(k_cache, (slot, rep, hk_off + k4), r_rep, 0)
        ld_row4(g_cache, (slot, rep, hk_off + k4), r_rep, 4)
        ld_row4(v_cache, (slot, rep, hv_off + k4), r_rep, 8)
        ld_row4(cs_q, (slot, hk_off + k4, KERNEL_WIDTH - 1 + rep), r_rep, 12)
        ld_row4(cs_k, (slot, hk_off + k4, KERNEL_WIDTH - 1 + rep), r_rep, 16)
        ld_row4(cs_v, (slot, hv_off + k4, KERNEL_WIDTH - 1 + rep), r_rep, 20)
        r_repb[0] = cutlass.Float32(beta_cache[slot, rep, i_hv])
    r_win = cute.make_rmem_tensor(cute.make_layout((3 * (KERNEL_WIDTH - 1) * 4,)), cutlass.Int32)
    if warp_idx == TMA_WARP:
        for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
            ld_row4(cs_q, (slot, hk_off + k4, w), r_win, 4 * w)
            ld_row4(cs_k, (slot, hk_off + k4, w), r_win, 4 * (KERNEL_WIDTH - 1 + w))
            ld_row4(cs_v, (slot, hv_off + k4, w), r_win, 4 * (2 * (KERNEL_WIDTH - 1) + w))
    commit_len = num_accepted_tokens[slot]
    # ---- TMA bulk load of S0 into the boxes; completes under the token staging /
    #      precompute and is consumed (mbar_ld) at the start of the chunk stage.
    # Issued by the whole of one warp (a TMA load issued from a single thread never
    # completes on this DSL build; the copy elects its issuing lane internally).
    if warp_idx == TMA_WARP:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ld, V * K * (STATE_DTYPE.width // 8))
    for sl_h in cutlass.range_constexpr(NUM_SLABS):
        sTl_h = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + sl_h * (V * 32), box_layout.inner, dtype=STATE_DTYPE
            ),
            box_layout.outer,
        )
        gS_h_all = cute.local_tile(mS_tma, (V, BOX_COLS, 1), (0, sl_h, None))
        if cutlass.const_expr(USE_FLAT_LAYOUT):
            gS_h = gS_h_all[(None, None, 0, h0_idx)]
        else:
            gS_h = gS_h_all[(None, None, 0, (i_hv, slot))]
        tSl_h, tGl_h = cpasync.tma_partition(
            tma_atom_l,
            0,
            cute.make_layout(1),
            cute.group_modes(sTl_h, 0, 2),
            cute.group_modes(gS_h, 0, 2),
        )
        if warp_idx == TMA_WARP:
            cute.copy(tma_atom_l, tGl_h, tSl_h, tma_bar_ptr=mbar_ld)
    if cutlass.const_expr(PDL):
        # Everything above only reads this layer's pools, metadata and weights; the new-token
        # rows are the previous grids' outputs.
        griddepcontrol_wait()
        if cutlass.const_expr(EARLY_TRIGGER):
            griddepcontrol_launch_dependents()
        if cutlass.const_expr(WIDE and FUSED_FB):
            if warp_idx == TMA_WARP:
                with cute.arch.elect_one():
                    bulk_g2s(saddr(sKbT, 0), fa8_src, TILE_K * 2, mbar_fb.toint())
        if is_tok:
            _load_new_token_row(
                *new_rows,
                tok_row,
                i_h,
                i_hv,
                hk_off,
                hv_off,
                k4,
                r_new,
                r_newb,
                USE_FLAT_LAYOUT,
                FUSED_FB,
            )
        if cutlass.const_expr(FUSED_FB):
            if warp_idx >= REPLAY_WARP0:
                _load_fb_b_fragments(fa, i_n, NEW_TOK, in_warp_tid, r_fb_b)
    # TMEM allocation overlaps the in-flight loads; the pointer is retrieved after the
    # precompute barrier (first TMEM use is the S0 staging in the chunk stage). With PDL it
    # follows griddepcontrol.wait, so a waiting CTA never holds tensor memory.
    if warp_idx == 0:
        cute.arch.alloc_tmem(TMEM_COLS, tmem_holder)
    # (3) placement: the slot-independent operands first, the token rows once the accepted count
    #     is in. Smem rows keep the [row][channel] fp32 layout of the precompute.
    if quad < 3:
        wv = unpack4(w_q.element_type, r_w, 0)
        for w in cutlass.range_constexpr(KERNEL_WIDTH):
            sConvW[quad * (KERNEL_WIDTH * K) + w * K + ch] = wv[w]
    if quad == 3:
        sDtBias[ch] = r_c[0]
        sOnW[ch] = r_c[1]
        if ch == 0:
            sAlog[0] = r_c[2]
    if is_tok:
        st_shared_f32x4(saddr(sOnG, tok_j * V + k4), *unpack4(onorm_g.element_type, r_new, 16))
    if warp_idx == TMA_WARP:
        for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
            st_shared_f32x4(saddr(sXQ, w * K + k4), *unpack4(CONV_DTYPE, r_win, 4 * w))
            st_shared_f32x4(
                saddr(sXK, w * K + k4), *unpack4(CONV_DTYPE, r_win, 4 * (KERNEL_WIDTH - 1 + w))
            )
            st_shared_f32x4(
                saddr(sXV, w * V + k4),
                *unpack4(CONV_DTYPE, r_win, 4 * (2 * (KERNEL_WIDTH - 1) + w)),
            )
    # Only NUM_SPEC drafts can be pending from the previous round. Clamp
    # so a malformed count cannot drive T_loop past the t_max-sized SMEM
    # buffers or the num_spec extents of the replay caches.
    if commit_len > NUM_SPEC:
        commit_len = cutlass.Int32(NUM_SPEC)
    T_loop = commit_len + NEW_TOK
    # Chunk window: the chunk stage runs tokens cb .. T_loop - 1 as chunk tokens 0 .. Tc - 1 with
    # the golden token at chunk index cl. The q / k / gate / v / beta rows of the precompute and
    # of the replayed drafts are staged at their chunk index (WIDE: cb = 1 after a fully accepted
    # round, T_loop = 17; the first replayed draft then goes to the spare row T_PAD, which the
    # pre-step folds into S0); the raw conv inputs keep their sequence positions.
    if cutlass.const_expr(WIDE):
        cb = cutlass.Int32(0)
        if T_loop > T_PAD:
            cb = T_loop - T_PAD
        cl = commit_len - cb
        Tc = T_loop - cb
    else:
        cl = commit_len
        Tc = T_loop
    if is_tok:
        t_new = commit_len + tok_j
        st_shared_f32x4(saddr(sXQ, (t_new + 3) * K + k4), *unpack4(x_q.element_type, r_new, 0))
        st_shared_f32x4(saddr(sXK, (t_new + 3) * K + k4), *unpack4(x_k.element_type, r_new, 4))
        st_shared_f32x4(saddr(sXV, (t_new + 3) * V + k4), *unpack4(x_v.element_type, r_new, 8))
        if cutlass.const_expr(not FUSED_FB):
            st_shared_f32x4(saddr(sGin, t_new * K + k4), *unpack4(g.element_type, r_new, 12))
        if in_warp_tid == 0:
            sBetaRaw[t_new] = r_newb[0]
    if cutlass.const_expr(WIDE):
        place_rep = is_rep & (rep < commit_len)
    else:
        place_rep = (warp_idx >= REPLAY_WARP0) & (rep < commit_len)
    if place_rep:
        # replayed drafts produce no output: their q rows are unused
        if cutlass.const_expr(WIDE):
            crow = rep - cb
            if crow < 0:
                crow = cutlass.Int32(T_PAD)
        else:
            crow = rep
        z = cutlass.Float32(0.0)
        st_shared_f32x4(saddr(sQ, crow * K + k4), z, z, z, z)
        st_shared_f32x4(saddr(sK, crow * K + k4), *unpack4(K_CACHE_DTYPE, r_rep, 0))
        st_shared_f32x4(saddr(sG, crow * K + k4), *unpack4(g_cache.element_type, r_rep, 4))
        st_shared_f32x4(saddr(sVall, crow * V + k4), *unpack4(V_CACHE_DTYPE, r_rep, 8))
        st_shared_f32x4(saddr(sXQ, (rep + 3) * K + k4), *unpack4(CONV_DTYPE, r_rep, 12))
        st_shared_f32x4(saddr(sXK, (rep + 3) * K + k4), *unpack4(CONV_DTYPE, r_rep, 16))
        st_shared_f32x4(saddr(sXV, (rep + 3) * V + k4), *unpack4(CONV_DTYPE, r_rep, 20))
        if in_warp_tid == 0:
            sBeta[crow] = r_repb[0]
    if cutlass.const_expr(FUSED_FB):
        # ---- forget gate of the new tokens, gate[t, row] = bf16(f_a[t] @ W_fb[h * 128 + row]^T)
        #      (MMA M = this warp's 16 W_fb rows, N = the token columns, K = 128 in eight
        #      sequential k16 steps from zero) into the raw-gate rows read by the precompute.
        if warp_idx >= REPLAY_WARP0:
            col0 = (i_n * NEW_TOK) % 8
            if cutlass.const_expr(not WIDE):
                if (in_warp_tid // 4 - col0 + 8) % 8 >= NEW_TOK:
                    for i in cutlass.range_constexpr(16):
                        r_fb_b[i] = cutlass.Int32(0)
            # A fragments: W_fb rows 16 mi + {0..7, 8..15} x k16 chunk kk through ldmatrix.x4 on
            # the 128B-swizzled boxes (row r & 7 == lane & 7 selects the 16 B unit XOR)
            mi = warp_idx - REPLAY_WARP0
            mat = in_warp_tid >> 3
            r8 = in_warp_tid & 7
            a_row = saddr(sScratch, OFF_FB) + (mi * 16 + (mat & 1) * 8 + r8) * 128
            cute.arch.mbarrier_wait(mbar_fb, 0)
            acc = [cutlass.Float32(0.0)] * 4
            if cutlass.const_expr(WIDE):
                # WIDE: a second n8 tile holds token 8 in column col0 (its packed row i_n * 9 + 8
                # is col0 mod 8, the column k3_kda_fab's tail gives it); the lanes of that column
                # take its f_a words from shared memory (bulk-copied on mbar_fb with W_fb), the
                # others zero.
                fb_mask = cutlass.Int32(0) - cutlass.Int32((in_warp_tid >> 2) == col0)
                acc_b = [cutlass.Float32(0.0)] * 4
            for kk in cutlass.range_constexpr(TILE_K // 16):
                a = ldmatrix_x4(
                    a_row + (kk // 4) * (TILE_K * 128) + ((((kk % 4) * 2 + (mat >> 1)) ^ r8) << 4)
                )
                acc = mma_bf16_16816(a, r_fb_b[2 * kk], r_fb_b[2 * kk + 1], acc)
                if cutlass.const_expr(WIDE):
                    b8_0 = sFa8w[8 * kk + (in_warp_tid & 3)] & fb_mask
                    b8_1 = sFa8w[8 * kk + 4 + (in_warp_tid & 3)] & fb_mask
                    acc_b = mma_bf16_16816(a, b8_0, b8_1, acc_b)
            for e in cutlass.range_constexpr(4):
                jt = (2 * (in_warp_tid & 3) + (e & 1) - col0 + 8) % 8
                if jt < NEW_TOK:
                    row = mi * 16 + (in_warp_tid >> 2) + (e >> 1) * 8
                    sGin[(commit_len + jt) * K + row] = round_bf16(acc[e])
            if cutlass.const_expr(WIDE):
                for e in cutlass.range_constexpr(4):
                    if 2 * (in_warp_tid & 3) + (e & 1) == col0:
                        row = mi * 16 + (in_warp_tid >> 2) + (e >> 1) * 8
                        sGin[(commit_len + NEW_TOK - 1) * K + row] = round_bf16(acc_b[e])
    # Padding rows [T_loop, t_max) of the gates, values and betas are zeroed so the chunk
    # stage's r_Y / cumulative-gate / V' loops are branch-free.
    if cutlass.const_expr(WIDE):
        pad_lo = Tc
        pad_hi = T_PAD
    else:
        pad_lo = T_loop
        pad_hi = t_max
    if tidx < K:
        for _z in cutlass.range(pad_lo, pad_hi, 1):
            sG[_z, tidx] = cutlass.Float32(0.0)
            sVall[_z * V + tidx] = cutlass.Float32(0.0)
            if tidx == 0:
                sBeta[_z] = cutlass.Float32(0.0)
    wg = warp_idx // 4
    wg_tidx = tidx % 128
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,)), cutlass.Float32)
    r_state = cute.make_rmem_tensor(cute.make_layout((32,)), cutlass.Float32)
    cute.arch.barrier()

    # ---- per-token-parallel precompute (smem only; the new tokens' cache and conv-window writes
    #      are issued from smem while the tensor cores run GEMM1/GEMM3): warp j < NEW_TOK computes
    #      the q and k conv + SiLU + L2 norm of new token j (the two chains interleaved), warp 8 + j
    #      its gate, beta and v conv. WIDE: token 8 rides along on warp 0 (q / k) and warp 9 (gate,
    #      beta, v), its dependency chains interleaved with the warp's own token; the q / k / gate
    #      / v / beta rows go to the tokens' chunk rows.
    half = NUM_THREADS // 64
    role = warp_idx // half
    j = warp_idx % half
    if cutlass.const_expr(WIDE):
        pc_t8 = commit_len + (NEW_TOK - 1)
        pc_r8 = cl + (NEW_TOK - 1)
        qk_args = (in_warp_tid, sXQ, sXK, sConvW, sQ, sK, k_weight_base, K, KERNEL_WIDTH, scale)
        gbv_args = (
            in_warp_tid,
            sGin,
            sDtBias,
            sAlog,
            sG,
            sBetaRaw,
            sBeta,
            sXV,
            sConvW,
            sVall,
            v_weight_base,
            K,
            V,
            KERNEL_WIDTH,
            lower_bound,
        )
        if role == 0:
            if j == 0:
                _precompute_qk([commit_len + j, pc_t8], [cl + j, pc_r8], *qk_args)
            else:
                _precompute_qk([commit_len + j], [cl + j], *qk_args)
        else:
            if j == 1:
                _precompute_gbv([commit_len + j, pc_t8], [cl + j, pc_r8], *gbv_args)
            else:
                _precompute_gbv([commit_len + j], [cl + j], *gbv_args)
    else:
        if j < NEW_TOK:
            t = commit_len + j
            if role == 0:
                for i in cutlass.range(vec_size, unroll_full=True):
                    k_idx = i * 32 + in_warp_tid
                    r_conv_0 = cutlass.Float32(0.0)
                    for w in cutlass.range(KERNEL_WIDTH, unroll_full=True):
                        r_conv_0 += sXQ[(t + w) * K + k_idx] * sConvW[w * K + k_idx]
                    r_conv = sXK[t * K + k_idx] * sConvW[k_weight_base + 0 * K + k_idx]
                    for w in cutlass.range(1, KERNEL_WIDTH, unroll_full=True):
                        r_conv += sXK[(t + w) * K + k_idx] * sConvW[k_weight_base + w * K + k_idx]
                    e0 = cute.math.exp(-r_conv_0, fastmath=True)
                    r_q[i] = r_conv_0 * cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                    r_conv = r_conv * cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                    )
                    r_k[i] = r_conv
                sum_q = 0.0
                sum_k = 0.0
                for i in cutlass.range(vec_size, unroll_full=True):
                    sum_q += r_q[i] * r_q[i]
                    sum_k += r_k[i] * r_k[i]
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                for i in cutlass.range(vec_size, unroll_full=True):
                    k_idx = i * 32 + in_warp_tid
                    sQ[t, k_idx] = r_q[i] * rnorm_q_scaled
                    sK[t, k_idx] = r_k[i] * rnorm_k
            else:
                r_exp_A = cute.math.exp(sAlog[0], fastmath=True)
                for i in cutlass.range(vec_size, unroll_full=True):
                    k_idx = i * 32 + in_warp_tid
                    r_g_raw = sGin[t * K + k_idx]
                    r_g_raw = r_g_raw + sDtBias[k_idx]
                    exp_A_x = r_exp_A * r_g_raw
                    sigmoid_val = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
                    )
                    sG[t, k_idx] = lower_bound * sigmoid_val
                r_beta_t = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-sBetaRaw[t], fastmath=True)
                )
                if in_warp_tid == 0:
                    sBeta[t] = r_beta_t
                for i in cutlass.range(vec_size, unroll_full=True):
                    v_idx = i * 32 + in_warp_tid
                    _v_conv = 0.0
                    for w in cutlass.range(KERNEL_WIDTH, unroll_full=True):
                        _v_conv += sXV[(t + w) * V + v_idx] * sConvW[v_weight_base + w * V + v_idx]
                    _v_conv = _v_conv * cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-_v_conv, fastmath=True)
                    )
                    sVall[t * V + v_idx] = _v_conv
    cute.arch.barrier()

    # =================================================================
    # Chunk (WY-form) recurrence on tensor cores.
    # =================================================================
    tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_holder)
    pool = cutlass.utils.TmemBufferPool(tmem_ptr, TMEM_COLS)
    tD4_frag = tiled_mma4.make_fragment_C(tiled_mma4.partition_shape_C((V, T_PAD)))
    tD4 = pool.allocate_tensor(tD4_frag.layout, cutlass.Float32)  # outputs (columns 0..15)
    tD2h_frag = tiled_mma2h.make_fragment_C(tiled_mma2h.partition_shape_C((V, K // 2)))
    tD2h0 = pool.allocate_tensor(tD2h_frag.layout, cutlass.Float32)  # GEMM2 half 0: k cols 0..63
    tD2h1 = pool.allocate_tensor(tD2h_frag.layout, cutlass.Float32)  # GEMM2 half 1: k cols 64..127
    tD2q_frag = tiled_mma2q.make_fragment_C(tiled_mma2q.partition_shape_C((V, T_PAD + K // 2)))
    tD2q = cute.make_tensor(tD4.iterator, tD2q_frag.layout)  # [D4 | dS half 0] accumulator view
    tA_frag = tiled_mma_f32a.make_fragment_A(tiled_mma_f32a.partition_shape_A((V, K)))
    tAhi = pool.allocate_tensor(tA_frag.layout, cutlass.Float32)
    tAlo = pool.allocate_tensor(tA_frag.layout, cutlass.Float32)
    tD1_frag = tiled_mma1.make_fragment_C(tiled_mma1.partition_shape_C((V, N1)))
    tD1 = pool.allocate_tensor(tD1_frag.layout, cutlass.Float32)
    tD3_frag = tiled_mma3.make_fragment_C(tiled_mma3.partition_shape_C((4 * T_PAD, T_PAD)))
    tD3 = pool.allocate_tensor(tD3_frag.layout, cutlass.Float32)
    tD2 = cute.make_tensor(tD2h0.iterator, tA_frag.layout)  # A-view layout (all 128 columns)

    st32 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    thr_stA = tcgen05.make_tmem_copy(st32, tD2).get_slice(wg_tidx)
    tD2_st = thr_stA.partition_D(tD2)
    rA_shape = thr_stA.partition_S(tD2).shape
    tAhi_st = thr_stA.partition_D(tAhi)
    tAlo_st = thr_stA.partition_D(tAlo)
    thr_ldD2 = tcgen05.make_tmem_copy(ld32, tD2).get_slice(wg_tidx)
    tD2_ld = thr_ldD2.partition_S(tD2)
    rS = cute.make_rmem_tensor(thr_ldD2.partition_D(tD2).shape[:-1], cutlass.Float32)
    rS_flat = cute.make_tensor(rS.iterator, cute.make_layout((32,)))
    # TMA store partitions for the NUM_SLABS boxes of this CTA's committed-state tile.
    tS_list = []
    tG_list = []
    for sl_h in cutlass.range_constexpr(NUM_SLABS):
        sT_h = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + sl_h * ((V // SPLIT_V) * 32),
                box_layout_t.inner,
                dtype=STATE_DTYPE,
            ),
            box_layout_t.outer,
        )
        gT_h_all = cute.local_tile(mT_tma, (V // SPLIT_V, BOX_COLS, 1), (vh, sl_h, None))
        if cutlass.const_expr(USE_FLAT_LAYOUT):
            gT_h = gT_h_all[(None, None, 0, h0_idx)]
        else:
            gT_h = gT_h_all[(None, None, 0, (i_hv, slot))]
        tS_h, tG_h = cpasync.tma_partition(
            tma_atom_t,
            0,
            cute.make_layout(1),
            cute.group_modes(sT_h, 0, 2),
            cute.group_modes(gT_h, 0, 2),
        )
        tS_list.append(tS_h)
        tG_list.append(tG_h)
    r_hi = cute.make_rmem_tensor(rA_shape[:-1], cutlass.Float32)
    r_lo = cute.make_rmem_tensor(rA_shape[:-1], cutlass.Float32)
    r_hi_flat = cute.make_tensor(r_hi.iterator, cute.make_layout((32,)))
    r_lo_flat = cute.make_tensor(r_lo.iterator, cute.make_layout((32,)))

    if cutlass.const_expr(WIDE):
        # pre-step operands in the V'^T staging words (unused until the forward substitution):
        # e^{g} of the pre-step token per k column, and the four partial row sums of
        # S0 diag(e^g0) k0
        sE0 = cute.make_tensor(sVpT.iterator, cute.make_layout((K,)))
        sPart = cute.make_tensor(sVpT.iterator + K, cute.make_layout((4 * V,)))

    # ---- cumulative gates per k column (thread kq, quarter q owns tokens t = q mod 4):
    #      G_t -> sE, the T x T coupling factors, and the operand rows written with closed-form
    #      128B-swizzle offsets (lanes = consecutive k: conflict-free). The GEMM3 rows (A3 =
    #      [Kf; Qf hi; Qf lo], B3 = Kb hi / lo) lie clear of the S0 boxes and are written first,
    #      so GEMM3 runs while S0 moves into TMEM; the GEMM1 rows (K~ hi / lo, Q~) stay in
    #      registers until the S0 boxes have been read.
    kq = tidx % K
    quarter = tidx // K
    t_mid = Tc // 2
    r_G = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
    g_run = cutlass.Float32(0.0)
    for t in cutlass.range(T_PAD, unroll_full=True):
        if t < t_lim:
            g_run = g_run + sG[t, kq]
        r_G[t] = g_run
    if cutlass.const_expr(WIDE):
        if (cb > 0) & (quarter == 0):
            sE0[kq] = cute.math.exp(sG[T_PAD, kq], fastmath=True)
    g_mid = cutlass.Float32(0.0)
    for t in cutlass.range(T_PAD, unroll_full=True):
        if t == t_mid:
            g_mid = r_G[t]
    # Factors of the T x T coupling: e^{G_t - G_mid} (t side) and e^{G_mid - G_s} (s side),
    # clamped at e^87 so a chunk whose decay exceeds the fp32 range yields ~0 instead of
    # inf/NaN (the affected products are < e^{-|g|} and negligible either way).
    exp_clamp = cutlass.Float32(87.0)
    ka = kq // 32
    kc = kq % 32
    b1_base = ka * (3 * T_PAD * 32)
    a3_base = ka * (4 * T_PAD * 32)
    b3_base = ka * (T_PAD * 32)
    b1k_base = ka * (T_PAD * 32)
    # GEMM1 rows of this thread's tokens t = quarter + 4 i: (K~ hi, Q~, K~ lo) per token
    r_b1 = cute.make_rmem_tensor(cute.make_layout((3 * (T_PAD // 4),)), cutlass.Float32)
    if cutlass.const_expr(WIDE):
        # the thread's four tokens only (G_t picked from r_G by quarter): a quarter of the code
        for i in cutlass.range_constexpr(T_PAD // 4):
            tw = quarter + 4 * i
            g_t_w = r_G[4 * i]
            for qq_w in cutlass.range_constexpr(1, 4):
                if quarter == qq_w:
                    g_t_w = r_G[4 * i + qq_w]
            kt_hi_w = cutlass.Float32(0.0)
            kt_lo_w = cutlass.Float32(0.0)
            qt_r_w = cutlass.Float32(0.0)
            kf_w = cutlass.Float32(0.0)
            qf_hi_w = cutlass.Float32(0.0)
            qf_lo_w = cutlass.Float32(0.0)
            kb_hi_w = cutlass.Float32(0.0)
            kb_lo_w = cutlass.Float32(0.0)
            if tw < Tc:
                d_t_w = g_t_w - g_mid
                e_t_w = cute.math.exp(-cute.arch.fmax(-d_t_w, -exp_clamp), fastmath=True)
                e_bt_w = cute.math.exp(-cute.arch.fmax(d_t_w, -exp_clamp), fastmath=True)
                eg_t_w = cute.math.exp(g_t_w, fastmath=True)
                sE[tw * K + kq] = g_t_w
                if tw == cl:
                    sEc[kq] = eg_t_w
                k_t_w = sK[tw, kq]
                q_t_w = sQ[tw, kq]
                kt_w = k_t_w * eg_t_w
                kt_hi_w = round_tf32(kt_w)
                qt_r_w = round_tf32(q_t_w * eg_t_w)
                kf_w = round_tf32(k_t_w * e_t_w)
                qf_w = q_t_w * e_t_w
                qf_hi_w = round_tf32(qf_w)
                kb_w = k_t_w * e_bt_w
                kb_hi_w = round_tf32(kb_w)
                kt_lo_w = kt_w - kt_hi_w
                qf_lo_w = qf_w - qf_hi_w
                kb_lo_w = kb_w - kb_hi_w
            # rows tw, 16+tw, 32+tw share tw % 8 -> the same 16-byte-chunk XOR
            xc_w = kc ^ (4 * (tw % 8))
            sA3f[a3_base + 32 * tw + xc_w] = kf_w
            sA3f[a3_base + 32 * (T_PAD + tw) + xc_w] = qf_hi_w
            sA3f[a3_base + 32 * (2 * T_PAD + tw) + xc_w] = qf_lo_w
            sB3hif[b3_base + 32 * tw + xc_w] = kb_hi_w
            sB3lof[b3_base + 32 * tw + xc_w] = kb_lo_w
            r_b1[3 * i] = kt_hi_w
            r_b1[3 * i + 1] = qt_r_w
            r_b1[3 * i + 2] = kt_lo_w
    else:
        for t in cutlass.range(T_PAD, unroll_full=True):
            if quarter == t % 4:
                kt_hi = cutlass.Float32(0.0)
                kt_lo = cutlass.Float32(0.0)
                qt_r = cutlass.Float32(0.0)
                kf = cutlass.Float32(0.0)
                qf_hi = cutlass.Float32(0.0)
                qf_lo = cutlass.Float32(0.0)
                kb_hi = cutlass.Float32(0.0)
                kb_lo = cutlass.Float32(0.0)
                if t < Tc:
                    d_t = r_G[t] - g_mid
                    e_t = cute.math.exp(-cute.arch.fmax(-d_t, -exp_clamp), fastmath=True)
                    e_bt = cute.math.exp(-cute.arch.fmax(d_t, -exp_clamp), fastmath=True)
                    eg_t = cute.math.exp(r_G[t], fastmath=True)
                    sE[t * K + kq] = r_G[t]
                    if t == cl:
                        sEc[kq] = eg_t
                    k_t = sK[t, kq]
                    q_t = sQ[t, kq]
                    kt = k_t * eg_t
                    kt_hi = round_tf32(kt)
                    qt_r = round_tf32(q_t * eg_t)
                    kf = round_tf32(k_t * e_t)
                    qf = q_t * e_t
                    qf_hi = round_tf32(qf)
                    kb = k_t * e_bt
                    kb_hi = round_tf32(kb)
                    kt_lo = kt - kt_hi
                    qf_lo = qf - qf_hi
                    kb_lo = kb - kb_hi
                # rows t, 16+t, 32+t share t % 8 -> the same 16-byte-chunk XOR
                xc = kc ^ (4 * (t % 8))
                sA3f[a3_base + 32 * t + xc] = kf
                sA3f[a3_base + 32 * (T_PAD + t) + xc] = qf_hi
                sA3f[a3_base + 32 * (2 * T_PAD + t) + xc] = qf_lo
                sB3hif[b3_base + 32 * t + xc] = kb_hi
                sB3lof[b3_base + 32 * t + xc] = kb_lo
                r_b1[3 * (t // 4)] = kt_hi
                r_b1[3 * (t // 4) + 1] = qt_r
                r_b1[3 * (t // 4) + 2] = kt_lo
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()
    # ---- GEMM3: D3 = A3 Kb_hi^T + A3 Kb_lo^T (warp 0; completion is tracked with GEMM1's)
    tCrA3 = tiled_mma3.make_fragment_A(
        cute.make_tensor(
            cute.recast_ptr(sA3.iterator, smem_layout_a3.inner, dtype=cutlass.TFloat32),
            sA3.layout,
        )
    )
    tCrB3hi = tiled_mma3.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB3hi.iterator, smem_layout_b3.inner, dtype=cutlass.TFloat32),
            sB3hi.layout,
        )
    )
    tCrB3lo = tiled_mma3.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB3lo.iterator, smem_layout_b3.inner, dtype=cutlass.TFloat32),
            sB3lo.layout,
        )
    )
    if warp_idx == 0:
        tiled_mma3.set(tcgen05.Field.ACCUMULATE, False)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(
                tiled_mma3, tD3, tCrA3[(None, None, kb_i, 0)], tCrB3hi[(None, None, kb_i, 0)], tD3
            )
            tiled_mma3.set(tcgen05.Field.ACCUMULATE, True)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(
                tiled_mma3, tD3, tCrA3[(None, None, kb_i, 0)], tCrB3lo[(None, None, kb_i, 0)], tD3
            )

    # ---- S0 row v = wg_tidx, columns [32 wg, +32), from the TMA-loaded swizzled box
    #      (16 B chunks: conflict-free LDS.128 under the 128B swizzle) into TMEM (hi / lo)
    cute.arch.mbarrier_wait(mbar_ld, 0)
    # in-place state update: the peer CTA must not store its half before our S0 load
    # completed -> arrive here, wait right before the state store.
    if cutlass.const_expr(SPLIT_V == 2):
        cute.arch.cluster_arrive_relaxed()
    vx = 4 * (wg_tidx % 8)
    if cutlass.const_expr(STATE_BF16):
        # bf16 box: 64 columns per 128 B row, so this warpgroup's 32 columns are the 16 B
        # chunks [4 (wg % 2), +4) of box wg // 2. A bf16 value is exact in tf32: no lo part,
        # and the S0_lo sweep of GEMM1 is skipped.
        for c in cutlass.range(4, unroll_full=True):
            chunk = (4 * (wg % 2) + c) ^ (wg_tidx % 8)
            off_l = (wg // 2) * (V * 64) + 64 * wg_tidx + 8 * chunk
            for i in cutlass.range(8, unroll_full=True):
                r_state[8 * c + i] = cutlass.Float32(sBoxh[off_l + i])
    else:
        for c in cutlass.range(8, unroll_full=True):
            off_l = wg * (V * 32) + 32 * wg_tidx + ((4 * c) ^ vx)
            for i in cutlass.range(4, unroll_full=True):
                r_state[4 * c + i] = sBoxf[off_l + i]
    if cutlass.const_expr(WIDE):
        if cb > 0:
            # pre-step (T_loop = 17): fold the first replayed draft (staged in row T_PAD) into the
            # state,
            # S1 = S0 diag(e^g0) + beta0 (v0 - S0 diag(e^g0) k0) k0^T, for row v = wg_tidx and
            # columns [32 wg, +32); the row sum over all 128 columns adds the four warpgroups'
            # partial sums (exchanged through shared memory) in a fixed order.
            pre_part = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(8):
                pre_e = ld_shared_f32x4(saddr(sE0, 32 * wg + 4 * c))
                pre_k = ld_shared_f32x4(saddr(sK, T_PAD * K + 32 * wg + 4 * c))
                for i in cutlass.range_constexpr(4):
                    pre_d = r_state[4 * c + i] * pre_e[i]
                    r_state[4 * c + i] = pre_d
                    pre_part = pre_part + pre_d * pre_k[i]
            sPart[wg * V + wg_tidx] = pre_part
            cute.arch.barrier()
            pre_u = (sPart[wg_tidx] + sPart[V + wg_tidx]) + (
                sPart[2 * V + wg_tidx] + sPart[3 * V + wg_tidx]
            )
            pre_vn = sBeta[T_PAD] * (sVall[T_PAD * V + wg_tidx] - pre_u)
            for c in cutlass.range_constexpr(8):
                pre_k = ld_shared_f32x4(saddr(sK, T_PAD * K + 32 * wg + 4 * c))
                for i in cutlass.range_constexpr(4):
                    r_state[4 * c + i] = r_state[4 * c + i] + pre_vn * pre_k[i]
    if cutlass.const_expr(STATE_BF16):
        # (WIDE pre-step: S1 is fp32; the tf32 MMA reads it truncated to tf32)
        for i in cutlass.range(32, unroll_full=True):
            r_hi_flat[i] = r_state[i]
        for jj in cutlass.range_constexpr(4):
            if wg == jj:
                cute.copy(st32, r_hi, tAhi_st[(None, None, None, jj)])
    else:
        for i in cutlass.range(32, unroll_full=True):
            s_hi = round_tf32(r_state[i])
            r_hi_flat[i] = s_hi
            r_lo_flat[i] = r_state[i] - s_hi
        for jj in cutlass.range_constexpr(4):
            if wg == jj:
                cute.copy(st32, r_hi, tAhi_st[(None, None, None, jj)])
                cute.copy(st32, r_lo, tAlo_st[(None, None, None, jj)])
    cute.arch.fence_view_async_tmem_store()
    # the S0 boxes have been read: the GEMM1 operand rows go over them
    cute.arch.barrier()
    if cutlass.const_expr(WIDE):
        for i in cutlass.range_constexpr(T_PAD // 4):
            tw = quarter + 4 * i
            xc_w = kc ^ (4 * (tw % 8))
            sB1f[b1_base + 32 * tw + xc_w] = r_b1[3 * i]
            if cutlass.const_expr(S0_LO):
                sB1kf[b1k_base + 32 * tw + xc_w] = r_b1[3 * i]
            sB1f[b1_base + 32 * (T_PAD + tw) + xc_w] = r_b1[3 * i + 1]
            sB1f[b1_base + 32 * (2 * T_PAD + tw) + xc_w] = r_b1[3 * i + 2]
    else:
        for t in cutlass.range(T_PAD, unroll_full=True):
            if quarter == t % 4:
                xc = kc ^ (4 * (t % 8))
                sB1f[b1_base + 32 * t + xc] = r_b1[3 * (t // 4)]
                if cutlass.const_expr(S0_LO):
                    sB1kf[b1k_base + 32 * t + xc] = r_b1[3 * (t // 4)]
                sB1f[b1_base + 32 * (T_PAD + t) + xc] = r_b1[3 * (t // 4) + 1]
                sB1f[b1_base + 32 * (2 * T_PAD + t) + xc] = r_b1[3 * (t // 4) + 2]
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()

    # ---- operand fragments of GEMM1 (D1 = S0_hi B1^T + S0_lo K~hi^T)
    tCrB1 = tiled_mma1.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB1.iterator, smem_layout_b1.inner, dtype=cutlass.TFloat32),
            sB1.layout,
        )
    )
    tCrB1k = tiled_mma1k.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB1k.iterator, smem_layout_b1k.inner, dtype=cutlass.TFloat32),
            sB1k.layout,
        )
    )
    tD1k = cute.make_tensor(
        tD1.iterator,
        tiled_mma1k.make_fragment_C(tiled_mma1k.partition_shape_C((V, T_PAD))).layout,
    )
    tAhi_tf = cute.make_tensor(cute.recast_ptr(tAhi.iterator, dtype=cutlass.TFloat32), tAhi.layout)
    tAlo_tf = cute.make_tensor(cute.recast_ptr(tAlo.iterator, dtype=cutlass.TFloat32), tAlo.layout)
    # ---- warp 0 issues GEMM1 right away; warps 1..15 stage Kbar^T (sKbT[k, s], lanes =
    #      consecutive k: conflict-free), seed the GEMM2 accumulator, build the B2 tiles and write
    #      the new tokens' cache rows while the tensor cores run.
    if warp_idx > 0:
        for i in cutlass.range(
            (K * T_PAD + NUM_THREADS - 33) // (NUM_THREADS - 32), unroll_full=True
        ):
            e_kb = i * (NUM_THREADS - 32) + (tidx - 32)
            if e_kb < K * T_PAD:
                k_kb = e_kb % K
                s_idx = e_kb // K
                b_val = cutlass.Float32(0.0)
                if s_idx <= cl:
                    b_val = sK[s_idx, k_kb] * cute.math.exp(
                        sE[cl * K + k_kb] - sE[s_idx * K + k_kb], fastmath=True
                    )
                sKbT[k_kb * TP_STRIDE + s_idx] = b_val
    if warp_idx == 0:
        tiled_mma1.set(tcgen05.Field.ACCUMULATE, False)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(
                tiled_mma1, tD1, tAhi_tf[(None, None, kb_i)], tCrB1[(None, None, kb_i, 0)], tD1
            )
            tiled_mma1.set(tcgen05.Field.ACCUMULATE, True)
        if cutlass.const_expr(S0_LO):
            tiled_mma1k.set(tcgen05.Field.ACCUMULATE, True)
            for kb_i in cutlass.range_constexpr(K // 8):
                cute.gemm(
                    tiled_mma1k,
                    tD1k,
                    tAlo_tf[(None, None, kb_i)],
                    tCrB1k[(None, None, kb_i, 0)],
                    tD1k,
                )
        # tracks GEMM3 and GEMM1
        with cute.arch.elect_one():
            tcgen05.commit(mbar1)
    else:
        # sKbT complete before the B2 tiles read it
        cute.arch.barrier(barrier_id=5, number_of_threads=NUM_THREADS - 32)
    # ---- GEMM2 accumulator seed (overlaps the GEMM1/GEMM3 execution)
    # D4 (output-combination) columns start from zero: GEMM2 half 0 accumulates into [D4 | dS]
    st16 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    thr_stD4 = tcgen05.make_tmem_copy(st16, tD4).get_slice(wg_tidx)
    tD4_st = thr_stD4.partition_D(tD4)
    r_zero = cute.make_rmem_tensor(thr_stD4.partition_S(tD4).shape, cutlass.Float32)
    for i in cutlass.range(cute.size(r_zero), unroll_full=True):
        r_zero[i] = cutlass.Float32(0.0)
    if wg == 0:
        cute.copy(st16, r_zero, tD4_st)
    # only this CTA's state rows (TMEM lanes [vh * V/2, +V/2)) are seeded and stored
    if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
        for i in cutlass.range(32, unroll_full=True):
            r_hi_flat[i] = r_state[i] * sEc[wg * 32 + i]
        for jj in cutlass.range_constexpr(4):
            if wg == jj:
                cute.copy(st32, r_hi, tD2_st[(None, None, None, jj)])
    cute.arch.fence_view_async_tmem_store()
    # ---- B2 tiles (k rows x s cols, K-major SW64) from sKbT while the MMAs run:
    #      lanes 0..15 -> row 2w, lanes 16..31 -> row 2w+1, four 32-row rounds
    s_col = in_warp_tid % T_PAD
    if warp_idx > 0:
        for r in cutlass.range(4, unroll_full=True):
            k_row = 32 * r + 2 * warp_idx + in_warp_tid // T_PAD
            b_val2 = sKbT[k_row * TP_STRIDE + s_col]
            b_hi = round_tf32(b_val2)
            o_b2 = swz64_k16(k_row, s_col)
            sB2hif[o_b2] = b_hi
            sB2lof[o_b2] = b_val2 - b_hi
        # warp 0 is issuing the MMAs: warps 1..4 cover its rows 32 r + {0, 1}
        if warp_idx <= 4:
            k_row0 = 32 * (warp_idx - 1) + in_warp_tid // T_PAD
            b_val0 = sKbT[k_row0 * TP_STRIDE + s_col]
            b_hi0 = round_tf32(b_val0)
            o_b20 = swz64_k16(k_row0, s_col)
            sB2hif[o_b20] = b_hi0
            sB2lof[o_b20] = b_val0 - b_hi0
    # ---- the new tokens' replay-cache rows and conv-window commits, from smem: warp 1 + p
    #      writes draft p's k / gate / v cache rows, its raw conv columns and beta, warps
    #      CONV_WARP0 + {0, 1, 2} commit the q / k / v conv windows (the raw inputs after the
    #      golden token). One CTA of a pair writes them.
    CONV_WARP0 = NUM_SPEC + 1 if WIDE else REPLAY_WARP0
    if vh == 0:
        if (warp_idx >= 1) & (warp_idx <= NUM_SPEC):
            p = warp_idx - 1
            t_c = commit_len + 1 + p
            # the chunk row of draft p (its sequence row t_c holds the raw conv inputs)
            if cutlass.const_expr(WIDE):
                q_c = cl + 1 + p
            else:
                q_c = t_c
            st_row4(k_cache, (slot, p, hk_off + k4), ld_shared_f32x4(saddr(sK, q_c * K + k4)))
            st_row4(g_cache, (slot, p, hk_off + k4), ld_shared_f32x4(saddr(sG, q_c * K + k4)))
            st_row4(v_cache, (slot, p, hv_off + k4), ld_shared_f32x4(saddr(sVall, q_c * V + k4)))
            st_row4(
                cs_q,
                (slot, hk_off + k4, KERNEL_WIDTH - 1 + p),
                ld_shared_f32x4(saddr(sXQ, (t_c + 3) * K + k4)),
            )
            st_row4(
                cs_k,
                (slot, hk_off + k4, KERNEL_WIDTH - 1 + p),
                ld_shared_f32x4(saddr(sXK, (t_c + 3) * K + k4)),
            )
            st_row4(
                cs_v,
                (slot, hv_off + k4, KERNEL_WIDTH - 1 + p),
                ld_shared_f32x4(saddr(sXV, (t_c + 3) * V + k4)),
            )
            if in_warp_tid == 0:
                beta_cache[slot, p, i_hv] = sBeta[q_c].to(BETA_CACHE_DTYPE)
        if warp_idx == CONV_WARP0:
            for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                st_row4(
                    cs_q,
                    (slot, hk_off + k4, w),
                    ld_shared_f32x4(saddr(sXQ, (commit_len + 1 + w) * K + k4)),
                )
        if warp_idx == CONV_WARP0 + 1:
            for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                st_row4(
                    cs_k,
                    (slot, hk_off + k4, w),
                    ld_shared_f32x4(saddr(sXK, (commit_len + 1 + w) * K + k4)),
                )
        if warp_idx == CONV_WARP0 + 2:
            for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                st_row4(
                    cs_v,
                    (slot, hv_off + k4, w),
                    ld_shared_f32x4(saddr(sXV, (commit_len + 1 + w) * V + k4)),
                )
    # ---- wait for GEMM1/GEMM3
    cute.arch.mbarrier_wait(mbar1, 0)
    tcgen05_fence_after_thread_sync()
    thr_ld1 = tcgen05.make_tmem_copy(ld16, tD1).get_slice(wg_tidx)
    rD1 = cute.make_rmem_tensor(thr_ld1.partition_D(tD1).shape, cutlass.Float32)
    cute.copy(ld16, thr_ld1.partition_S(tD1), rD1)
    cute.arch.fence_view_async_tmem_load()
    rD1_flat = cute.make_tensor(rD1.iterator, cute.make_layout((N1,)))
    v_row = wg_tidx
    # warpgroup 0: D3 (64 x 16) -> smem [L; B_hi; B_lo; -]
    ld3 = cute.make_copy_atom(
        tcgen05.Ld16x128bOp(tcgen05.Repetition(T_PAD * 32 // 128)), cutlass.Float32
    )
    thr_ld3 = tcgen05.make_tmem_copy(ld3, tD3).get_slice(wg_tidx)
    tD3_ld = thr_ld3.partition_S(tD3)
    sLB3 = cute.make_tensor(
        sLB.iterator,
        cute.make_layout(((4 * T_PAD, T_PAD), 1, 1), stride=((T_PAD, 1), 0, 0)),
    )
    tD3_sm = thr_ld3.partition_D(sLB3)
    rD3 = cute.make_rmem_tensor(tD3_sm.shape, cutlass.Float32)
    if wg == 0:
        cute.copy(ld3, tD3_ld, rD3)
        cute.arch.fence_view_async_tmem_load()
        cute.autovec_copy(rD3, tD3_sm)
    cute.arch.barrier()
    # ---- warpgroup 0: V' by forward substitution (row v per thread, four accumulators);
    #      warpgroups 1-2: output-combination operand Bcoef[t, s] = B_hi + B_lo (rows t < T_loop,
    #      s <= t; zero elsewhere), one element per thread, K-major SW64 16 x 16 tiles (hi / lo)
    #      that head the GEMM2 half-0 B tile
    if wg == 0:
        r_Y = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
        for s in cutlass.range(T_PAD, unroll_full=True):
            y = cutlass.Float32(0.0)
            if s < t_lim:
                u0 = rD1_flat[s] + rD1_flat[2 * T_PAD + s]
                y = sBeta[s] * (sVall[s * V + v_row] - u0)
            r_Y[s] = y
        r_V = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
        for t in cutlass.range(T_PAD, unroll_full=True):
            vp = cutlass.Float32(0.0)
            if t < t_lim:
                acc0 = cutlass.Float32(0.0)
                acc1 = cutlass.Float32(0.0)
                acc2 = cutlass.Float32(0.0)
                acc3 = cutlass.Float32(0.0)
                for s in range(t):
                    if s % 4 == 0:
                        acc0 = acc0 + sLB[t, s] * r_V[s]
                    elif s % 4 == 1:
                        acc1 = acc1 + sLB[t, s] * r_V[s]
                    elif s % 4 == 2:
                        acc2 = acc2 + sLB[t, s] * r_V[s]
                    else:
                        acc3 = acc3 + sLB[t, s] * r_V[s]
                vp = r_Y[t] - sBeta[t] * ((acc0 + acc1) + (acc2 + acc3))
            r_V[t] = vp
        for t in cutlass.range(T_PAD, unroll_full=True):
            sVpT[v_row * TP_STRIDE + t] = r_V[t]
    else:
        if tidx < 128 + T_PAD * T_PAD:
            t4 = (tidx - 128) // T_PAD
            s4 = (tidx - 128) % T_PAD
            bc = cutlass.Float32(0.0)
            if (t4 < Tc) & (s4 <= t4):
                bc = sLB[T_PAD + t4, s4] + sLB[2 * T_PAD + t4, s4]
            bc_hi = round_tf32(bc)
            o_b4 = swz64_k16(t4, s4)
            sB4hif[o_b4] = bc_hi
            sB4lof[o_b4] = bc - bc_hi
    cute.arch.barrier()
    # ---- A2 tiles (v rows x s cols) from sVpT, same 16-lanes-per-row mapping; s > c -> 0
    for r in cutlass.range(4, unroll_full=True):
        v_row2 = 32 * r + 2 * warp_idx + in_warp_tid // T_PAD
        a_val = sVpT[v_row2 * TP_STRIDE + s_col]
        a_hi = round_tf32(a_val)
        o_a2 = swz64_k16(v_row2, s_col)
        sA2hif[o_a2] = a_hi
        sA2lof[o_a2] = a_val - a_hi
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()
    sA2hi_tf = cute.make_tensor(
        cute.recast_ptr(sA2hi.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32),
        sA2hi.layout,
    )
    sA2lo_tf = cute.make_tensor(
        cute.recast_ptr(sA2lo.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32),
        sA2lo.layout,
    )
    tCrA2hi = tiled_mma2h.make_fragment_A(sA2hi_tf)
    tCrA2lo = tiled_mma2h.make_fragment_A(sA2lo_tf)
    # B2 halves: k rows 0..63 / 64..127 of the SW64 K-major tiles (a 4 KB shift keeps the swizzle)
    tCrB2hi_h1 = tiled_mma2h.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B2HI + 64 * T_PAD,
                smem_layout_b2h.inner,
                dtype=cutlass.TFloat32,
            ),
            smem_layout_b2h.outer,
        )
    )
    tCrB2lo_h1 = tiled_mma2h.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B2LO + 64 * T_PAD,
                smem_layout_b2h.inner,
                dtype=cutlass.TFloat32,
            ),
            smem_layout_b2h.outer,
        )
    )
    tCrA2hi_q = tiled_mma2q.make_fragment_A(sA2hi_tf)
    tCrA2lo_q = tiled_mma2q.make_fragment_A(sA2lo_tf)
    tCrB2q_hi = tiled_mma2q.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B4HI, smem_layout_b2q.inner, dtype=cutlass.TFloat32
            ),
            smem_layout_b2q.outer,
        )
    )
    tCrB2q_lo = tiled_mma2q.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B4LO, smem_layout_b2q.inner, dtype=cutlass.TFloat32
            ),
            smem_layout_b2q.outer,
        )
    )
    if warp_idx == 0:
        # GEMM2 half 0 fused with the output combination: [D4 | dS_0..63] += A2 x [Bcoef; Kbar_0..63]
        tiled_mma2q.set(tcgen05.Field.ACCUMULATE, True)
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2q,
                tD2q,
                tCrA2hi_q[(None, None, kb_i, 0)],
                tCrB2q_hi[(None, None, kb_i, 0)],
                tD2q,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2q,
                tD2q,
                tCrA2hi_q[(None, None, kb_i, 0)],
                tCrB2q_lo[(None, None, kb_i, 0)],
                tD2q,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2q,
                tD2q,
                tCrA2lo_q[(None, None, kb_i, 0)],
                tCrB2q_hi[(None, None, kb_i, 0)],
                tD2q,
            )
        with cute.arch.elect_one():
            tcgen05.commit(mbar2)
        tiled_mma2h.set(tcgen05.Field.ACCUMULATE, True)
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2h,
                tD2h1,
                tCrA2hi[(None, None, kb_i, 0)],
                tCrB2hi_h1[(None, None, kb_i, 0)],
                tD2h1,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2h,
                tD2h1,
                tCrA2hi[(None, None, kb_i, 0)],
                tCrB2lo_h1[(None, None, kb_i, 0)],
                tD2h1,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2h,
                tD2h1,
                tCrA2lo[(None, None, kb_i, 0)],
                tCrB2hi_h1[(None, None, kb_i, 0)],
                tD2h1,
            )
        with cute.arch.elect_one():
            tcgen05.commit(mbar2b)

    # ---- outputs of the new tokens: o_t = O0_t + D4[v, t] (columns 0..15 of GEMM2 half 0),
    #      warpgroup w writes new tokens j = w mod 4, bf16-rounded, into sVall rows j
    cute.arch.mbarrier_wait(mbar2, 0)
    tcgen05_fence_after_thread_sync()
    thr_ld4 = tcgen05.make_tmem_copy(ld16, tD4).get_slice(wg_tidx)
    rD4 = cute.make_rmem_tensor(thr_ld4.partition_D(tD4).shape, cutlass.Float32)
    cute.copy(ld16, thr_ld4.partition_S(tD4), rD4)
    cute.arch.fence_view_async_tmem_load()
    rD4_flat = cute.make_tensor(rD4.iterator, cute.make_layout((T_PAD,)))
    if cutlass.const_expr(WIDE):
        for j_w in cutlass.range(wg, NEW_TOK, 4):
            t_out_w = cl + j_w
            o_val_w = cutlass.Float32(0.0)
            for tt_w in cutlass.range(T_PAD, unroll_full=True):
                if tt_w == t_out_w:
                    o_val_w = rD1_flat[T_PAD + tt_w] + rD4_flat[tt_w]
            sVall[j_w * V + v_row] = cutlass.Float32(cutlass.BFloat16(o_val_w))
    else:
        for j_new in cutlass.range(NEW_TOK, unroll_full=True):
            if wg == j_new % 4:
                t_out = cl + j_new
                o_val = cutlass.Float32(0.0)
                for tt in cutlass.range(T_PAD, unroll_full=True):
                    if tt == t_out:
                        o_val = rD1_flat[T_PAD + tt] + rD4_flat[tt]
                sVall[j_new * V + v_row] = cutlass.Float32(cutlass.BFloat16(o_val))
    # ---- committed-state store through TMA: D2 slab -> swizzled smem box (row per thread,
    #      16 B chunks land in distinct banks under the 128B swizzle) -> one bulk tensor store
    if cutlass.const_expr(SPLIT_V == 2):
        cute.arch.cluster_wait()
    # All 16 warps produced disjoint value rows above; the barrier makes them visible to the
    # norm epilogue, which assigns one complete token to each warp.
    cute.arch.barrier()
    # The state-owning warps of warpgroups 0/1 stage k columns 0..63 (GEMM2 half 0, mbar2), those
    # of warpgroups 2/3 columns 64..127 (half 1, mbar2b) into the swizzled smem boxes; the first
    # staging thread of each pair issues the TMA stores once its half is visible (named barriers
    # 6 / 7) and waits for the bulk groups at the very end. The warps without state rows run the
    # norm epilogue.
    if wg < 2:
        cute.arch.mbarrier_wait(mbar2, 0)
    else:
        cute.arch.mbarrier_wait(mbar2b, 0)
    tcgen05_fence_after_thread_sync()
    for sl in cutlass.range_constexpr(4):
        if wg == sl:
            if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
                cute.copy(ld32, tD2_ld[(None, None, None, sl)], rS)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(STATE_BF16):
                    # bf16 box: chunks [4 (sl % 2), +4) of box sl // 2, round to nearest
                    for c in cutlass.range(4, unroll_full=True):
                        chunk = (4 * (sl % 2) + c) ^ (wg_tidx % 8)
                        off_b = (
                            (sl // 2) * ((V // SPLIT_V) * 64)
                            + 64 * (wg_tidx - (V // SPLIT_V) * vh)
                            + 8 * chunk
                        )
                        for i in cutlass.range(8, unroll_full=True):
                            sBoxh[off_b + i] = cutlass.BFloat16(rS_flat[8 * c + i])
                else:
                    for c in cutlass.range(8, unroll_full=True):
                        off_b = (
                            sl * ((V // SPLIT_V) * 32)
                            + 32 * (wg_tidx - (V // SPLIT_V) * vh)
                            + ((4 * c) ^ vx)
                        )
                        for i in cutlass.range(4, unroll_full=True):
                            sBoxf[off_b + i] = rS_flat[4 * c + i]
    # last TMEM reads of the kernel: warps 1..15 signal barrier 8, warp 0 frees TMEM below
    tcgen05_fence_before_thread_sync()
    if warp_idx > 0:
        cute.arch.barrier_arrive(barrier_id=8, number_of_threads=NUM_THREADS)
    cute.arch.fence_view_async_shared()
    st_tid0 = (V // 2) * vh
    if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
        if wg < 2:
            cute.arch.barrier(barrier_id=6, number_of_threads=256 // SPLIT_V)
            if tidx == st_tid0:
                for s in cutlass.range_constexpr(NUM_SLABS // 2):
                    cute.copy(tma_atom_t, tS_list[s], tG_list[s])
                cute.arch.cp_async_bulk_commit_group()
        else:
            cute.arch.barrier(barrier_id=7, number_of_threads=256 // SPLIT_V)
            if tidx == 256 + st_tid0:
                for s in cutlass.range_constexpr(NUM_SLABS // 2, NUM_SLABS):
                    cute.copy(tma_atom_t, tS_list[s], tG_list[s])
                cute.arch.cp_async_bulk_commit_group()
    if warp_idx == 0:
        # all TMEM reads are done (barrier 8): free the columns while the store drains
        cute.arch.barrier(barrier_id=8, number_of_threads=NUM_THREADS)
        tcgen05_fence_after_thread_sync()
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)
    # ---- gated RMSNorm epilogue (CTA 0 of a pair) on the warps (warp % 4) in {2, 3}: one token
    #      per warp, p_v = o_v * w_v * sigmoid(g_v), out = bf16(p_v * rsqrt(mean(o_v^2) + eps))
    token_idx = (warp_idx // 4) * 2 + (warp_idx % 4 - 2)
    if (vh == 0) & ((warp_idx % 4) // 2 == 1):
        while token_idx < NEW_TOK:
            pre_values = cute.make_rmem_tensor(cute.make_layout((V // 32,)), cutlass.Float32)
            rms_partial = cutlass.Float32(0.0)
            for i in cutlass.range(V // 32, unroll_full=True):
                v_idx = i * 32 + in_warp_tid
                value = sVall[token_idx * V + v_idx]
                gate = sOnG[token_idx * V + v_idx]
                sigmoid_gate = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
                )
                pre_values[i] = value * sOnW[v_idx] * sigmoid_gate
                rms_partial += value * value
            for offset in [16, 8, 4, 2, 1]:
                rms_partial += cute.arch.shuffle_sync_bfly(
                    rms_partial, offset=offset, mask=-1, mask_and_clamp=31
                )
            inv_rms = cute.math.rsqrt(
                rms_partial / cutlass.Float32(V) + cutlass.Float32(onorm_eps),
                fastmath=True,
            )
            output_row = i_n * NEW_TOK + token_idx
            for i in cutlass.range(V // 32, unroll_full=True):
                v_idx = i * 32 + in_warp_tid
                o[0, output_row, i_hv, v_idx] = cutlass.BFloat16(pre_values[i] * inv_rms)
            if cutlass.const_expr(MXFP8_OUT):
                # Channels [32 i, 32 i + 32) of this head form 1x32 group i of the o_proj input
                # row, one value per lane: the group amax is a warp reduction. Groups (0, 1) and
                # (2, 3) share the BF16 halves of a word, so two interleaved chains reduce all four.
                bits = [_bf16_bits(pre_values[i] * inv_rms) for i in range(V // 32)]
                words = [bits[2 * j] | (bits[2 * j + 1] << 16) for j in range(V // 64)]
                amax = list(words)
                for offset in [16, 8, 4, 2, 1]:
                    for j in cutlass.range_constexpr(V // 64):
                        amax[j] = mx_ops.amax_pair(
                            amax[j], cute.arch.shuffle_sync_bfly(amax[j], offset=offset)
                        )
                row_q = xq_addr + Int64(output_row) * (HV * V) + i_hv * V + in_warp_tid
                packed_e = cutlass.Int32(0)
                for i in cutlass.range_constexpr(V // 32):
                    shift = 16 * (i % 2)
                    e_a = mx_ops.ue8m0_of_amax((amax[i // 2] >> shift) & 0xFFFF)
                    word = (words[i // 2] >> shift) & 0xFFFF
                    code = _e4m3_of_bf16_lo(word, mx_ops.bf16x2_inv_scale(e_a))
                    st_global_u8(row_q + i * 32, code)
                    packed_e = packed_e | (e_a << (8 * i))
                if in_warp_tid == 0:
                    row_s = xs_addr + Int64(output_row) * (HV * V // 32) + i_hv * (V // 32)
                    st_global_u32(row_s, packed_e)
            token_idx = token_idx + (NUM_THREADS // 64 if WIDE else NUM_THREADS // 32)
    # the state stores must have read their smem boxes before the CTA exits
    if (tidx == st_tid0) | (tidx == 256 + st_tid0):
        cute.arch.cp_async_bulk_wait_group(0, read=True)

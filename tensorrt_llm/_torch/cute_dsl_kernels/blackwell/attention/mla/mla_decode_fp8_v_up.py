# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Balanced-split FP8 MLA decode whose reduction also applies the V up-projection (Helix).

:class:`BlackwellMlaDecodeFP8VUp` runs the split kernel of
:class:`BlackwellMultiHeadLatentAttentionForwardFP8` unchanged and replaces its balanced-split
reduction by one kernel that, per head ``h``, combines the partial outputs of the rows
``(b, t, h)`` and multiplies them by that head's V up-projection::

    v_out[b * S_q + t, h, :] = bf16(O[b, t, h, :] @ w_uv[h].T)

where ``O[b, t, h, :]`` is the BF16 row the reduction would store and ``w_uv`` is the BF16
``[H, v_head_dim, kv_lora_rank]`` weight. It writes the Helix softmax statistics of the rows it
combines as the reduction does; the latent O is not stored.

One CTA per head: before ``griddepcontrol.wait`` its MMA warp loads ``w_uv[h]`` (128 x 512 BF16,
the A operand of a swap-AB ``tcgen05.mma`` kind::f16 with FP32 accumulation in tensor memory);
after it, the combine warps reduce the rows (the slot-ordered, LSE-weighted sum of
``balanced_reduction_kernel``, element for element) into the B operand in shared memory, rows
whose M tile one cluster covered alone are read from the split kernel's O, and rows of a request
without KV are zero. Every output element is the K-ordered chain of 16-wide MMA steps over the
512 latent columns. The combine warps first stage the M tile of every partial slot in shared
memory; warp ``w`` then reduces rows ``w, w + 16, ...``. Neither the latent O of the combined
rows nor their LSE is stored.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cutlass_dsl import dsl_user_op

from .mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8
from .mla_helpers import LOG2_E

#: V up-projection outputs per head (UMMA M).
V_HEAD_DIM = 128
#: BF16 elements per 128-byte swizzled shared-memory row: the K extent of one operand stage.
K_BLOCK = 64
#: Largest number of rows per head (batch * seq_len_q) the fused reduction serves: each combine
#: warp reduces its rows one after the other, so past two rows per warp (batch 8 with MTP6) the
#: separate reduction and BMM are faster.
MAX_ROWS = 32
#: Warps that combine rows (warps 0-3 also read the accumulator); the next warp loads the weight
#: and issues the MMA.
COMBINE_WARPS = 16
#: Partial slots a combining warp loads at once (``vec`` values per lane each): BF16 / FP32 partials.
V_UNROLL_BF16 = 8
V_UNROLL_FP32 = 4


@dsl_user_op
def _sts_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(w0).ir_value(loc=loc, ip=ip),
            Int32(w1).ir_value(loc=loc, ip=ip),
            Int32(w2).ir_value(loc=loc, ip=ip),
            Int32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def v_up_rows_bucket(rows: int) -> int:
    """UMMA N (padded rows per head) of the fused reduction for ``rows`` = batch * seq_len_q."""
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"rows must be in [1, {MAX_ROWS}], got {rows}")
    return max(16, -(-rows // 16) * 16)


class BlackwellMlaDecodeFP8VUp(BlackwellMultiHeadLatentAttentionForwardFP8):
    """The balanced-split FP8 decode with the reduction fused with the V up-projection.

    ``v_rows``: UMMA N, the padded number of rows per head (:func:`v_up_rows_bucket` of the
    largest batch * seq_len_q the compiled kernel serves). Needs the balanced split, the folded
    query layout and the Helix softmax statistics.
    """

    def __init__(self, *args, v_rows: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.balanced_split and self.emit_softmax_stats and self.fold_sq):
            raise ValueError(
                "The fused V up-projection needs the balanced split, the folded query layout "
                "and the softmax statistics."
            )
        if v_rows % 16 or not 16 <= v_rows <= MAX_ROWS:
            raise ValueError(f"v_rows must be a multiple of 16 in [16, {MAX_ROWS}], got {v_rows}")
        self.v_rows = v_rows
        self.v_tmem_cols = max(32, 1 << (v_rows - 1).bit_length())
        self.v_stages = self.latent_dim // K_BLOCK
        self.v_mma_tiler = (V_HEAD_DIM, v_rows, K_BLOCK)
        self.v_mma_warp = COMBINE_WARPS
        self.v_threads = 32 * (COMBINE_WARPS + 1)

    @cute.jit
    def run_with_softmax_stats_v_up(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        softmax_stats: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        kv_bounds: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale: cute.Tensor,
        output_scale: cute.Tensor,
        v_weight: cute.Tensor,
        v_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """``run_with_softmax_stats`` plus ``v_weight`` (BF16 ``[H, 128, 512]``) and ``v_out``
        (BF16 ``[B * S_q, H, 128]``); ``o`` then holds only the rows of M tiles one cluster
        covered alone."""
        self._v_weight = v_weight
        self._v_out = v_out
        self._run(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            o,
            lse,
            softmax_stats,
            workspace,
            split_kv,
            cache_seqs,
            kv_bounds,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        )

    @cute.jit
    def _launch_balanced_reduction(
        self,
        o,
        lse,
        softmax_stats,
        acc_o,
        acc_lse,
        part_desc,
        cache_seqs,
        kv_bounds,
        ws_rows,
        batch_size,
        stream,
    ):
        w = self._v_weight
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.v_mma_tiler[:2],
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.v_mma_tiler, cutlass.BFloat16, self.v_stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.v_mma_tiler, cutlass.BFloat16, self.v_stages
        )
        # w_uv as (M = v dim, K = latent, L = head)
        mW = cute.make_tensor(
            w.iterator,
            cute.make_layout(
                (V_HEAD_DIM, self.latent_dim, self.num_heads),
                stride=(self.latent_dim, 1, V_HEAD_DIM * self.latent_dim),
            ),
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mW,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.v_mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        self.v_a_bytes = cute.size_in_bytes(cutlass.BFloat16, a_layout)
        num_slots = self.balanced_num_slots

        @cute.struct
        class VUpStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_layout)], 1024
            ]
            slots: cute.struct.MemRange[Int32, COMBINE_WARPS * (num_slots + 1)]
            desc: cute.struct.MemRange[Int32, num_slots]
            a_full: cutlass.Int64
            acc_full: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        self.v_storage = VUpStorage
        self.reduce_v_up_kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            o,
            softmax_stats,
            acc_o,
            acc_lse,
            part_desc,
            cache_seqs,
            kv_bounds,
            self._v_out,
            a_layout,
            b_layout,
            batch_size * self.seq_len_q,
        ).launch(
            grid=(self.num_heads, 1, 1),
            block=[self.v_threads, 1, 1],
            smem=VUpStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def reduce_v_up_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        mO: cute.Tensor,
        mSoftmaxStats: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        mDesc: cute.Tensor,
        cache_seqs: cute.Tensor,
        kv_bounds: Optional[cute.Tensor],
        mV: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        num_rows: Int32,
    ):
        """CTA ``h``: rows ``(b, t, h)`` for ``j = b * S_q + t < num_rows``; see the module
        docstring."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        head, _, _ = cute.arch.block_idx()
        num_slots = self.balanced_num_slots

        storage = utils.SmemAllocator().allocate(self.v_storage)
        a_full = storage.a_full
        acc_full = storage.acc_full
        if warp_idx == self.v_mma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(a_full, 1)
                cute.arch.mbarrier_init(acc_full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx == self.v_mma_warp:
            # The weight does not depend on the split kernel: load it before the wait.
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            gA = cute.local_tile(
                mA, cute.slice_(self.v_mma_tiler, (None, 0, None)), (None, None, None)
            )
            tCgA = tiled_mma.get_slice(0).partition_A(gA)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(a_full, self.v_a_bytes)
            for kt in cutlass.range_constexpr(self.v_stages):
                cute.copy(
                    tma_atom_a, tAgA[(None, 0, kt, head)], tAsA[(None, kt)], tma_bar_ptr=a_full
                )
            cute.arch.alloc_tmem(self.v_tmem_cols, storage.tmem_holding_buf)
            cute.arch.relinquish_tmem_alloc_permit()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        if warp_idx < COMBINE_WARPS:
            # the M tile of every partial slot, once for all rows
            sDesc = storage.desc.get_tensor(cute.make_layout(num_slots))
            for i in cutlass.range(tidx, num_slots, COMBINE_WARPS * 32, unroll=1):
                sDesc[i] = mDesc[i, 0]
            cute.arch.barrier(barrier_id=1, number_of_threads=COMBINE_WARPS * 32)
            tile_slots = cute.make_tensor(
                storage.slots.data_ptr() + warp_idx * (num_slots + 1),
                cute.make_layout(num_slots + 1),
            )
            b_base = storage.sB.data_ptr().toint()
            vec = self.latent_dim // self.threads_per_warp
            # lane l holds latent columns [vec * l, vec * l + vec): K stage, 16-byte chunks
            stage = lane * vec // K_BLOCK
            chunk0 = (lane * vec % K_BLOCK) // 8
            rAcc = cute.make_rmem_tensor(cute.make_layout(vec), self.acc_dtype)
            rO = cute.make_rmem_tensor(cute.make_layout(vec), cutlass.BFloat16)
            for j in cutlass.range(warp_idx, self.v_rows, COMBINE_WARPS, unroll=1):
                rO.fill(0.0)
                if j < num_rows:
                    b = j // self.seq_len_q
                    t = j - b * self.seq_len_q
                    g = t * self.num_heads + head
                    c = g // self.fold_rows
                    r = g - c * self.fold_rows
                    self._gather_tile_slots_smem(sDesc, b * self.num_m_chunks + c, tile_slots, lane)
                    cute.arch.sync_warp()
                    if tile_slots[0] > 0:
                        sum_lse = self._combine_row(mAccO, mAccLSE, tile_slots, r, lane, rAcc)
                        rO.store(rAcc.load().to(cutlass.BFloat16))
                        if lane == 0:
                            self._store_row_stats(
                                mSoftmaxStats, cache_seqs, kv_bounds, sum_lse, r, c, b, t
                            )
                    elif cache_seqs[b] <= 0:
                        if lane == 0:
                            self._store_row_stats(
                                mSoftmaxStats, cache_seqs, kv_bounds, self.lse_dtype.inf, r, c, b, t
                            )
                    else:
                        # one cluster covered the M tile: the split kernel stored the row
                        cute.autovec_copy(cute.local_tile(mO[r, None, c, b], (vec,), (lane,)), rO)
                    cute.arch.sync_warp()
                words = cute.recast_tensor(rO, Int32)
                row_addr = b_base + stage * (self.v_rows * 128) + j * 128
                swz = j & 7
                for u in cutlass.range_constexpr(vec // 8):
                    _sts_v4(
                        row_addr + (((chunk0 + u) ^ swz) << 4),
                        words[4 * u],
                        words[4 * u + 1],
                        words[4 * u + 2],
                        words[4 * u + 3],
                    )
            cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.v_mma_tiler[:2])
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        if warp_idx == self.v_mma_warp:
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kphases = cute.size(tCrA, mode=[2])
            cute.arch.mbarrier_wait(a_full, 0)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kt in cutlass.range_constexpr(self.v_stages):
                for kp in cutlass.range_constexpr(num_kphases):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, kt)],
                        tCrB[(None, None, kp, kt)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(acc_full)
        elif warp_idx < 4:
            tAcc = tCtAcc[((None, None), 0, 0)]
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(self.v_rows)), Float32
            )
            tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc)
            thr_t2r = tiled_t2r.get_slice(tidx)
            tTR_tAcc = thr_t2r.partition_S(tAcc)
            tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((V_HEAD_DIM, self.v_rows)))
            rV = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)
            cute.arch.mbarrier_wait(acc_full, 0)
            cute.copy(tiled_t2r, tTR_tAcc, rV)
            cute.arch.fence_view_async_tmem_load()
            for i in cutlass.range_constexpr(cute.size(rV)):
                v, j = tTR_cAcc[i]
                if j < num_rows:
                    mV[j, head, v] = rV[i].to(cutlass.BFloat16)
        cute.arch.barrier()
        if warp_idx == self.v_mma_warp:
            cute.arch.dealloc_tmem(acc_tmem_ptr, self.v_tmem_cols)

    @cute.jit
    def _combine_row(self, mAccO, mAccLSE, tile_slots, r, lane, rAcc):
        """One warp combines row ``r`` of the M tile whose partial slots ``tile_slots`` lists,
        all ``latent_dim`` columns, into ``rAcc`` (``vec`` per lane) and returns its global LSE.

        The arithmetic of ``_combine_tile_rows`` for one row: the first ``reduction_unroll``
        partials load with the LSEs, the global LSE is the warp-parallel max / sum over the
        partials' LSEs, and every element accumulates the partials in slot order."""
        vec = self.latent_dim // self.threads_per_warp
        unroll = V_UNROLL_BF16 if self.partial_dtype == cutlass.BFloat16 else V_UNROLL_FP32
        n_parts = tile_slots[0]
        rLSE = cute.make_rmem_tensor(cute.make_layout(1), self.lse_dtype)
        lse_max = cute.make_rmem_tensor(cute.make_layout(1), self.lse_dtype)
        sum_lse = cute.make_rmem_tensor(cute.make_layout(1), self.lse_dtype)
        rPartLSE = cute.make_rmem_tensor(cute.make_layout(unroll), self.lse_dtype)
        rPartIn = cute.make_rmem_tensor(cute.make_layout((vec, unroll)), self.partial_dtype)
        # one partial widened at a time (fewer registers than widening all of them)
        rPart = cute.make_rmem_tensor(cute.make_layout(vec), self.acc_dtype)
        rLSE.fill(-self.lse_dtype.inf)
        if lane < n_parts:
            rLSE[0] = mAccLSE[r, tile_slots[1 + lane], 0, 0]
        for u in cutlass.range_constexpr(unroll):
            if u < n_parts:
                q = tile_slots[1 + u]
                cute.autovec_copy(
                    cute.local_tile(mAccO[r, q, None, 0, 0], (vec,), (lane,)), rPartIn[None, u]
                )
                rPartLSE[u] = mAccLSE[r, q, 0, 0]
        lse_max[0] = rLSE[0]
        for j0 in range(self.threads_per_warp, n_parts, self.threads_per_warp):
            if j0 + lane < n_parts:
                lse_max[0] = cute.arch.fmax(lse_max[0], mAccLSE[r, tile_slots[1 + j0 + lane], 0, 0])
        m = cute.arch.warp_reduction_max(lse_max[0])
        lse_max[0] = m if m != -self.lse_dtype.inf else 0.0
        sum_lse[0] = cute.math.exp2(rLSE[0] - lse_max[0], fastmath=True)
        for j0 in range(self.threads_per_warp, n_parts, self.threads_per_warp):
            if j0 + lane < n_parts:
                sum_lse[0] += cute.math.exp2(
                    mAccLSE[r, tile_slots[1 + j0 + lane], 0, 0] - lse_max[0], fastmath=True
                )
        sm = cute.arch.warp_reduction_sum(sum_lse[0])
        sum_lse[0] = (
            lse_max[0] + cute.math.log2(sm, fastmath=True)
            if not sm == self.lse_dtype(0.0) or sm != sm
            else self.lse_dtype.inf
        )
        rAcc.fill(0.0)
        for u in cutlass.range_constexpr(unroll):
            if u < n_parts:
                scale = cute.math.exp2(rPartLSE[u] - sum_lse[0], fastmath=True)
                rPart.store(rPartIn[None, u].load().to(self.acc_dtype))
                for e in cutlass.range_constexpr(vec):
                    rAcc[e] += rPart[e] * scale
        for j0 in range(unroll, n_parts, unroll):
            for u in cutlass.range_constexpr(unroll):
                if j0 + u < n_parts:
                    q = tile_slots[1 + j0 + u]
                    cute.autovec_copy(
                        cute.local_tile(mAccO[r, q, None, 0, 0], (vec,), (lane,)),
                        rPartIn[None, u],
                    )
                    rPartLSE[u] = mAccLSE[r, q, 0, 0]
            for u in cutlass.range_constexpr(unroll):
                if j0 + u < n_parts:
                    scale = cute.math.exp2(rPartLSE[u] - sum_lse[0], fastmath=True)
                    rPart.store(rPartIn[None, u].load().to(self.acc_dtype))
                    for e in cutlass.range_constexpr(vec):
                        rAcc[e] += rPart[e] * scale
        return sum_lse[0]

    @cute.jit
    def _gather_tile_slots_smem(self, sDesc, tile, tile_slots, lane):
        """``_gather_tile_slots`` on the slots' M tiles staged in shared memory: one warp lists
        the partial slots of M tile ``tile`` in slot order (``tile_slots[0]`` their count)."""
        num_slots = self.balanced_num_slots
        count = Int32(0)
        for g in cutlass.range_constexpr((num_slots + 31) // 32):
            q = g * 32 + lane
            match = False
            if q < num_slots:
                match = sDesc[q] == tile
            ballot = cute.arch.vote_ballot_sync(match)
            if match:
                below = cute.arch.popc(ballot & Int32(cute.arch.lanemask_lt()))
                tile_slots[1 + count + below] = q
            count = count + cute.arch.popc(ballot)
        if lane == 0:
            tile_slots[0] = count

    @cute.jit
    def _store_row_stats(self, mSoftmaxStats, cache_seqs, kv_bounds, global_lse, r, c, b, t):
        """The Helix softmax statistics ``_store_reduced_row`` stores for a combined row."""
        has_local_kv = cache_seqs[b] > 0
        if cutlass.const_expr(kv_bounds is not None):
            has_local_kv = kv_bounds[b * self.seq_len_q + t] > 0
        if has_local_kv:
            mSoftmaxStats[r, c, b, 0] = global_lse / LOG2_E
            mSoftmaxStats[r, c, b, 1] = 1.0
        else:
            mSoftmaxStats[r, c, b, 0] = -self.lse_dtype.inf
            mSoftmaxStats[r, c, b, 1] = 0.0


__all__ = ["BlackwellMlaDecodeFP8VUp", "MAX_ROWS", "V_HEAD_DIM", "v_up_rows_bucket"]

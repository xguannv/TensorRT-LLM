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
"""Split-KV FP8 MLA decode whose reduction also applies the V up-projection (Helix).

:class:`BlackwellMlaDecodeFP8VUp` runs the split kernel of
:class:`~.mla_decode_fp8.BlackwellMultiHeadLatentAttentionForwardFP8` unchanged (with its FP8 P @ V on
probabilities biased by ``2^P_EXP2_BIAS``, so the softmax tail below e4m3's normal range is kept) and
replaces its split-KV reduction by one kernel that, per head ``h``, combines the split partials of the
rows ``(b, t, h)`` and multiplies them by that head's V up-projection::

    v_out[b * S_q + t, h, :] = bf16(O[b, t, h, :] @ w_uv[h].T)

where ``O[b, t, h, :]`` is the BF16 row the reduction would store and ``w_uv`` is the BF16
``[H, v_head_dim, kv_lora_rank]`` weight. It writes the Helix softmax statistics of every row as the
reduction does; neither the latent O nor the LSE is stored.

One CTA per head: before ``griddepcontrol.wait`` its MMA warp loads ``w_uv[h]`` (128 x 512 BF16, the A
operand of a swap-AB ``tcgen05.mma`` kind::f16 with FP32 accumulation in tensor memory); after it, warp
``w`` of the combine warps reduces rows ``w, w + 16, ...``: lane ``i`` holds the LSE of split ``i``, the
global LSE and the split weights are the reduction kernel's warp-0 butterflies, and lane ``l`` accumulates
latent columns ``[16 l, 16 l + 16)`` over the splits in split order with the reduction's expression, then
rounds them to BF16 into the B operand in shared memory. Every output element is the K-ordered chain of
16-wide MMA steps over the 512 latent columns.

Needs the split-KV workspace (``split_kv > 1``), the fixed query layout and the Helix softmax statistics.

With ``is_var_split_kv`` (``TRTLLM_K3LL_MLA_VAR_SPLIT``; constructed with ``var_split_bounded``) request ``b`` is
split ``block_split_kvs[b]`` ways (at most the grid's ``split_kv`` <= 32): the split kernel's clusters past a
request's count have no K tiles, and the reduction combines that request's partials only.

``prewait`` (``TRTLLM_K3LL_MLA_PREWAIT_L2``, with the split kernel's PDL): before its ``griddepcontrol.wait`` the
split kernel L2-prefetches ``w_uv`` (one warp per CTA, the chunks spread over the grid), and after the wait it
releases the fused reduction, whose CTAs then take SMs as they free up and load their ``w_uv[h]`` from L2 while the
split kernel still runs. Prefetch hints and trigger placement only: the bytes are the same.

``partial_f16`` (``TRTLLM_K3LL_MLA_PARTIAL_F16``): the split kernel stores its normalized partial O as FP16 instead
of FP32 (half the bytes its epilogue writes and the combine reads; the LSE stays FP32), and the combine widens each
partial to FP32 before the same multiply-adds. Not the same bits: each partial is rounded once to FP16 (11-bit
significand) before the combine, below the BF16 rounding of the result.

``combine_early`` (``TRTLLM_K3LL_MLA_COMBINE_EARLY``): a combine warp issues its row's LSE load (lanes below the grid's
split count) and the partial loads of its first ``combine_early`` splits (one of ``COMBINE_EARLY_SPLITS``) before it
reads the request's split count and K length, so those loads share one round trip; the LSE beyond the request's splits
is then replaced by ``-inf`` and only the request's splits are added, the same multiply-adds in split order, so the
output is the same bits.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cutlass_dsl import dsl_user_op

from ...k3_l2_prefetch import CHUNK_BYTES as L2_CHUNK_BYTES
from ...k3_l2_prefetch import _bulk_prefetch_l2
from .mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8
from .mla_helpers import LOG2_E

#: V up-projection outputs per head (UMMA M).
V_HEAD_DIM = 128
#: BF16 elements per 128-byte swizzled shared-memory row: the K extent of one operand stage.
K_BLOCK = 64
#: Largest number of rows per head (batch * seq_len_q) the fused reduction serves (UMMA N <= 32: each
#: combine warp reduces at most two rows one after the other).
MAX_ROWS = 32
#: Warps that combine rows (warps 0-3 also read the accumulator); the next warp loads the weight and
#: issues the MMA.
COMBINE_WARPS = 16
#: The values of ``combine_early`` (splits whose partials a combine warp loads before the split count is known; 0 off).
COMBINE_EARLY_SPLITS = (2, 4, 8)


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
    """The split-KV FP8 decode with the reduction fused with the V up-projection.

    ``v_rows``: UMMA N, the padded number of rows per head (:func:`v_up_rows_bucket` of the largest
    batch * seq_len_q the compiled kernel serves). Needs ``emit_softmax_stats``, a fixed query layout and a
    reducer capacity of at most 32 splits (one split per lane).
    """

    def __init__(
        self,
        *args,
        v_rows: int = 16,
        prewait: bool = False,
        partial_f16: bool = False,
        combine_early: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # The LSE and first partial loads issued before the split count is read (``TRTLLM_K3LL_MLA_COMBINE_EARLY``).
        if combine_early and combine_early not in COMBINE_EARLY_SPLITS:
            raise ValueError(
                f"combine_early must be 0 or one of {COMBINE_EARLY_SPLITS}, got {combine_early}"
            )
        self.combine_early = combine_early > 0
        # The split partials' O as FP16 in the workspace (``TRTLLM_K3LL_MLA_PARTIAL_F16``); the LSE stays FP32.
        self.partial_o_dtype = cutlass.Float16 if partial_f16 else None
        self.combine_load_dtype = cutlass.Float16 if partial_f16 else cutlass.Float32
        self.combine_early_splits = combine_early
        if not self.emit_softmax_stats or self.is_var_q:
            raise ValueError(
                "The fused V up-projection needs the Helix softmax statistics and a fixed query layout."
            )
        if self.reducer_max_splits > 32:
            raise ValueError(
                f"The fused V up-projection reduces one split per lane, got reducer_max_splits="
                f"{self.reducer_max_splits}"
            )
        if v_rows % 16 or not 16 <= v_rows <= MAX_ROWS:
            raise ValueError(f"v_rows must be a multiple of 16 in [16, {MAX_ROWS}], got {v_rows}")
        self.v_rows = v_rows
        # The fused reduction always launches with programmatic dependent launch: it loads its weight before
        # ``griddepcontrol.wait`` (the split kernel's ``use_pdl`` is the base class's).
        self.v_up_use_pdl = True
        self.v_tmem_cols = max(32, 1 << (v_rows - 1).bit_length())
        self.v_stages = self.latent_dim // K_BLOCK
        self.v_mma_tiler = (V_HEAD_DIM, v_rows, K_BLOCK)
        self.v_mma_warp = COMBINE_WARPS
        self.v_threads = 32 * (COMBINE_WARPS + 1)
        # The split kernel's pre-wait W_UV L2 prefetch and early release of the fused reduction (needs its PDL wait).
        self.split_prewait = bool(prewait and self.use_pdl)

    def initialize_workspace(self, H, D, S, B, split_kv, acc_dtype, workspace):
        """The base class's workspace views; with ``partial_o_dtype`` the partial O is stored in that type and the
        LSE (``acc_dtype``) starts right after it."""
        if cutlass.const_expr(self.partial_o_dtype is None or workspace is None):
            return super().initialize_workspace(H, D, S, B, split_kv, acc_dtype, workspace)
        workspace_rows = cutlass.max(H, cutlass.Int32(128))
        align = 256 // self.q_dtype.width
        acc_o_layout = cute.make_layout(
            (workspace_rows, split_kv, D, S, B),
            stride=(
                cute.assume(split_kv * D, align),
                cute.assume(D, align),
                1,
                cute.assume(split_kv * workspace_rows * D, align),
                cute.assume(workspace_rows * split_kv * S * D, align),
            ),
        )
        acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=self.partial_o_dtype)
        acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
        acc_lse_layout = cute.make_layout(
            (workspace_rows, split_kv, S, B),
            stride=(split_kv, 1, workspace_rows * split_kv, workspace_rows * split_kv * S),
        )
        # D is a multiple of 2, so the LSE (4-byte elements) starts 4-byte aligned after the 2-byte partials.
        acc_lse_iter = cute.recast_ptr(acc_o_iter + cute.cosize(acc_o_layout), dtype=acc_dtype)
        acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

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
        kv_bounds: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        v_weight: cute.Tensor,
        v_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """``run_with_softmax_stats`` (fixed query lengths) plus ``v_weight`` (BF16 ``[H, 128, 512]``) and ``v_out``
        (BF16 ``[B * S_q, H, 128]``); ``o`` and ``lse`` are not written. ``block_split_kvs``: with
        ``is_var_split_kv``, each request's split count (``[B]`` int32, at most ``split_kv``, the grid's); else
        None (the scalar ``split_kv``)."""
        self._v_weight = v_weight
        self._v_out = v_out
        self._v_prefetch = None
        if cutlass.const_expr(self.split_prewait):
            self._v_prefetch = v_weight
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
            None,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        )

    @cute.jit
    def _split_prewait_loads(self, mW: cute.Tensor):
        """In the split kernel, before its PDL wait (one warp per CTA): an L2 prefetch of ``w_uv``, the fused
        reduction's weight, in ``L2_CHUNK_BYTES`` chunks spread over every thread of that warp across the grid."""
        lane = cute.arch.lane_idx()
        grid_x, grid_y, grid_z = cute.arch.grid_dim()
        blk_x, blk_y, blk_z = cute.arch.block_idx()
        cta = Int32(blk_x) + Int32(grid_x) * (Int32(blk_y) + Int32(grid_y) * Int32(blk_z))
        step = Int32(grid_x) * Int32(grid_y) * Int32(grid_z) * self.threads_per_warp
        base = mW.iterator.toint()
        nbytes = Int64(cute.size(mW)) * (cutlass.BFloat16.width // 8)
        chunks = Int32((nbytes + (L2_CHUNK_BYTES - 1)) // L2_CHUNK_BYTES)
        for c in cutlass.range(cta * self.threads_per_warp + lane, chunks, step, unroll=1):
            off = Int64(c) * L2_CHUNK_BYTES
            size = nbytes - off
            if size > L2_CHUNK_BYTES:
                size = Int64(L2_CHUNK_BYTES)
            _bulk_prefetch_l2(base + off, Int32(size))

    @cute.jit
    def _launch_reduction(
        self,
        o_unpacked,
        lse_unpacked,
        softmax_stats_unpacked,
        acc_o,
        acc_lse,
        split_kv,
        cache_seqs,
        kv_bounds,
        cum_seq_lens_q,
        block_split_kvs,
        runtime_batch_size,
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

        @cute.struct
        class VUpStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_layout)], 1024
            ]
            a_full: cutlass.Int64
            acc_full: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        self.v_storage = VUpStorage
        self.reduce_v_up_kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            softmax_stats_unpacked,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            kv_bounds,
            block_split_kvs,
            self._v_out,
            a_layout,
            b_layout,
            runtime_batch_size * self.seq_len_q,
        ).launch(
            grid=(self.num_heads, 1, 1),
            block=[self.v_threads, 1, 1],
            smem=VUpStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.v_up_use_pdl,
        )

    @cute.kernel
    def reduce_v_up_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        mSoftmaxStats: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: Int32,
        cache_seqs: cute.Tensor,
        kv_bounds: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        mV: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        num_rows: Int32,
    ):
        """CTA ``h``: rows ``(b, t, h)`` for ``j = b * S_q + t < num_rows``; see the module docstring."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        head, _, _ = cute.arch.block_idx()

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

        if cutlass.const_expr(self.v_up_use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        if warp_idx < COMBINE_WARPS:
            b_base = storage.sB.data_ptr().toint()
            vec = self.latent_dim // self.threads_per_warp
            # lane l holds latent columns [vec * l, vec * l + vec): K stage, 16-byte chunks
            stage = lane * vec // K_BLOCK
            chunk0 = (lane * vec % K_BLOCK) // 8
            rAcc = cute.make_rmem_tensor(cute.make_layout(vec), self.acc_dtype)
            rPart = cute.make_rmem_tensor(cute.make_layout(vec), self.acc_dtype)
            rO = cute.make_rmem_tensor(cute.make_layout(vec), cutlass.BFloat16)
            for j in cutlass.range(warp_idx, self.v_rows, COMBINE_WARPS, unroll=1):
                rO.fill(0.0)
                if j < num_rows:
                    b = j // self.seq_len_q
                    t = j - b * self.seq_len_q
                    self._combine_row(
                        mSoftmaxStats,
                        mAccO,
                        mAccLSE,
                        split_kv,
                        cache_seqs,
                        kv_bounds,
                        block_split_kvs,
                        b,
                        t,
                        head,
                        lane,
                        rAcc,
                        rPart,
                    )
                    rO.store(rAcc.load().to(cutlass.BFloat16))
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
    def _combine_row(
        self,
        mSoftmaxStats,
        mAccO,
        mAccLSE,
        split_kv,
        cache_seqs,
        kv_bounds,
        block_split_kvs,
        b,
        t,
        head,
        lane,
        rAcc,
        rPart,
    ):
        """One warp combines row ``(b, t, head)`` over its splits into ``rAcc`` (``vec`` columns per lane)
        and stores its Helix softmax statistics: the arithmetic of ``reduction_kernel`` for that row (its
        warp-0 LSE butterflies with split ``i`` on lane ``i``, the split weights ``exp2(lse_i - global)``,
        and ``acc += partial * weight`` over the splits in order)."""
        vec = self.latent_dim // self.threads_per_warp
        flat_q_row = t * self.num_heads + head
        q_tile = flat_q_row >> 7
        r = flat_q_row & 127
        gLSE = mAccLSE[r, None, q_tile, b]
        gAccO = mAccO[r, None, None, q_tile, b]
        early = self.combine_early_splits
        early_lse = self.lse_dtype(0.0)
        early_parts = [
            cute.make_rmem_tensor(cute.make_layout(vec), self.combine_load_dtype)
            for _ in range(early)
        ]
        if cutlass.const_expr(self.combine_early):
            # Before the split count: every split of the grid has an LSE slot and a partial row, stale where the
            # request has fewer splits; those are masked below and never added.
            grid_split_kv = split_kv
            early_lse = gLSE[lane] if cute.elem_less(lane, grid_split_kv) else -self.lse_dtype.inf
            for u in cutlass.range_constexpr(early):
                if u < grid_split_kv:
                    cute.autovec_copy(
                        cute.local_tile(gAccO[u, None], (vec,), (lane,)), early_parts[u]
                    )
        has_local_kv = cache_seqs[b] > 0
        if cutlass.const_expr(kv_bounds is not None):
            has_local_kv = kv_bounds[b * self.seq_len_q + t] > 0
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[b]
        k_tile_total = cute.ceil_div(cache_seqs[b], self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        if cutlass.const_expr(kv_bounds is not None):
            k_tile_per_cta = cutlass.max(k_tile_per_cta, 1)
        local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

        local_lse = cute.make_rmem_tensor(cute.make_layout(1), self.lse_dtype)
        if cutlass.const_expr(self.combine_early):
            local_lse[0] = (
                early_lse if cute.elem_less(lane, local_split_kv) else -self.lse_dtype.inf
            )
        else:
            local_lse[0] = (
                gLSE[lane] if cute.elem_less(lane, local_split_kv) else -self.lse_dtype.inf
            )
        lse_max = -self.lse_dtype.inf
        lse_max = cute.arch.fmax(lse_max, local_lse[0])
        lse_max = cute.arch.warp_reduction_max(lse_max)
        lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
        sum_lse = 0.0
        sum_lse += cute.math.exp2(local_lse[0] - lse_max, fastmath=True)
        sum_lse = cute.arch.warp_reduction_sum(sum_lse)
        global_lse = (
            lse_max + cute.math.log2(sum_lse, fastmath=True)
            if not sum_lse == self.lse_dtype(0.0) or sum_lse != sum_lse
            else self.lse_dtype.inf
        )
        if lane == 0:
            if has_local_kv:
                mSoftmaxStats[head, t, b, 0] = global_lse / LOG2_E
                mSoftmaxStats[head, t, b, 1] = 1.0
            else:
                mSoftmaxStats[head, t, b, 0] = -self.lse_dtype.inf
                mSoftmaxStats[head, t, b, 1] = 0.0
        weight = cute.math.exp2(local_lse[0] - global_lse, fastmath=True)

        rAcc.fill(0.0)
        for u in cutlass.range_constexpr(early):
            if u < local_split_kv:
                scale = cute.arch.shuffle_sync(weight, u)
                if cutlass.const_expr(self.partial_o_dtype is None):
                    for e in cutlass.range_constexpr(vec):
                        rAcc[e] += early_parts[u][e] * scale
                else:
                    rPart.store(early_parts[u].load().to(self.acc_dtype))
                    for e in cutlass.range_constexpr(vec):
                        rAcc[e] += rPart[e] * scale
        if cutlass.const_expr(self.partial_o_dtype is None):
            for i in range(early, local_split_kv):
                scale = cute.arch.shuffle_sync(weight, i)
                cute.autovec_copy(cute.local_tile(gAccO[i, None], (vec,), (lane,)), rPart)
                for e in cutlass.range_constexpr(vec):
                    rAcc[e] += rPart[e] * scale
        else:
            rHalf = cute.make_rmem_tensor(cute.make_layout(vec), self.partial_o_dtype)
            for i in range(early, local_split_kv):
                scale = cute.arch.shuffle_sync(weight, i)
                cute.autovec_copy(cute.local_tile(gAccO[i, None], (vec,), (lane,)), rHalf)
                rPart.store(rHalf.load().to(self.acc_dtype))
                for e in cutlass.range_constexpr(vec):
                    rAcc[e] += rPart[e] * scale


__all__ = ["BlackwellMlaDecodeFP8VUp", "MAX_ROWS", "V_HEAD_DIM", "v_up_rows_bucket"]

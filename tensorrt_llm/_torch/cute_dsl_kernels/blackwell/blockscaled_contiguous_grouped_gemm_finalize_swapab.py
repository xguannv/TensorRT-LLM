# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Swap-AB NVFP4 grouped GEMM + MoE finalize (scatter-add) for few tokens per expert (SM100/SM103).

Computes the same result as ``Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel``::

    out[token(r), h] += bf16(alpha[e] * token_final_scale(r) * sum_k A[r, k] * B[e][h, k])

for every valid permuted row ``r`` of a ``tile_size`` (128) row tile owned by expert ``e``, with the
operands swapped: the expert weights ``B[e]`` are the MMA A operand (blocks of 128 rows of the output
dimension ``h``) and the permuted activations are the MMA B operand (``n_tok`` = 64 or 128 rows of
one permuted tile). The accumulator is ``h x token`` in TMEM, so an epilogue thread owns one output
column ``h`` and reads only as many accumulator columns as the tile has valid tokens. The regular
kernel's epilogue reads the whole ``token x h`` accumulator whatever the number of valid tokens, which
dominates when the GEMM K is small (the down projection of a TP-split expert, K = intermediate / TP)
and an expert sees one or two tokens.

Each partial ``bf16(alpha * scale * acc)`` is bit-identical to the regular kernel's partial (same FP32
accumulation, same FP32 scale product, round-to-nearest BF16). Partials are added to ``out`` with
16-byte ``red.global.add.bf16x2`` operations; as in the regular kernel (bulk-reduce adds), the order in
which the top-k partials of a token are added is not fixed.

Work decomposition: a tile is (permuted tile, ``128 * h_blocks`` rows of ``h``); tiles are linearized
in weight-memory order (``h`` fastest) and dealt to the persistent CTAs in round-robin chunks of
``sched_chunk`` tiles, so concurrently running CTAs stream neighbouring weight rows while each CTA
still sees runs of the same permuted tile: the token operand and the per-row metadata (token index,
``alpha * final_scale``) are loaded once per run. Warp roles: ``num_epi_wg`` epilogue warpgroups
(consecutive tiles rotate between them), then MMA, TMA, scheduler and metadata warps.

Requirements: NVFP4 (Float4E2M1FN data, Float8E4M3FN scales, vector 16), K-major operands,
``n % (128 * h_blocks) == 0``, ``k % 64 == 0``, ``k <= 256`` (one K stage), ``tile_size == 128``;
``n_tok`` = 64 only when no permuted tile can hold more than 64 valid rows (num_tokens <= 64: a token
selects an expert at most once).
"""

from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op

from .utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait


@dsl_user_op
def red_add_v8_bf16_from_smem(dst_ptr, src_smem_addr, *, loc=None, ip=None):
    """``dst[0:8] += src[0:8]``: 8 BF16 (16 bytes) read from shared memory, one vector red to global."""
    llvm.inline_asm(
        None,
        [dst_ptr, src_smem_addr],
        "{\n.reg .b32 a, b, c, d;\nld.shared.v4.b32 {a, b, c, d}, [$1];\n"
        "red.global.add.noftz.v4.bf16x2 [$0], {a, b, c, d};\n}",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


class Sm100BlockScaledContiguousGroupedGemmFinalizeSwapABKernel:
    """Swap-AB variant of the NVFP4 contiguous grouped GEMM with fused MoE finalize.

    :param sf_vec_size: Scale-factor vector size (16 for NVFP4).
    :param n_tok: Token rows of a permuted tile fed to the MMA (64 or 128).
    :param tile_size: Rows of a permuted tile (moe_sort ``tile_tokens_dim``); must be 128.
    :param num_epi_wg: Epilogue warpgroups (1-4); consecutive tiles rotate between them.
    :param num_tile_stage: Depth of the scheduler -> warps tile-info ring.
    :param h_blocks: 128-row blocks of ``h`` per tile (1 or 2).
    :param sched_chunk: Consecutive tiles per round-robin chunk (0: one contiguous range per CTA).
    :param num_b_stage: Shared-memory slots for the token operand of a run.
    :param num_meta_stage: Shared-memory slots for the per-row metadata of a run.
    """

    def __init__(
        self,
        sf_vec_size: int,
        n_tok: int = 64,
        tile_size: int = 128,
        num_epi_wg: int = 3,
        num_tile_stage: int = 16,
        h_blocks: int = 2,
        sched_chunk: int = 4,
        num_b_stage: int = 3,
        num_meta_stage: int = 4,
    ):
        assert tile_size == 128, "swap-AB finalize supports tile_size 128 only"
        assert n_tok in (64, 128)
        assert num_epi_wg in (1, 2, 3, 4)
        assert h_blocks in (1, 2)
        # 128-row MMA blocks of the output dimension per tile (one TMA box of 128 * h_blocks rows)
        self.h_blocks = h_blocks
        # 0: every CTA takes one contiguous range of tiles; c > 0: chunks of c consecutive tiles are
        # dealt round-robin to the CTAs (concurrent CTAs then stream neighbouring weight rows)
        self.sched_chunk = sched_chunk
        self.sf_vec_size = sf_vec_size
        self.n_tok = n_tok
        self.tile_size = tile_size
        self.num_epi_wg = num_epi_wg
        self.num_tile_stage = num_tile_stage
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        # (h rows, token rows, K): K is set in _setup_attributes
        self.mma_tiler = (128, n_tok, 1)
        self.occupancy = 1
        self.threads_per_warp = 32
        self.num_epi_warps = 4 * num_epi_wg
        self.mma_warp_id = self.num_epi_warps
        self.tma_warp_id = self.num_epi_warps + 1
        self.sched_warp_id = self.num_epi_warps + 2
        self.meta_load_warp_id = self.num_epi_warps + 3
        self.threads_per_cta = self.threads_per_warp * (self.num_epi_warps + 4)
        # every warp but the scheduler consumes the tile-info ring
        self.threads_wo_sched = self.threads_per_warp * (self.num_epi_warps + 3)
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_warp * (self.num_epi_warps + 1)
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.threads_per_warp
        )
        self.epi_all_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=self.threads_per_warp * self.num_epi_warps
        )
        self.epi_wg_barriers = tuple(
            pipeline.NamedBarrier(barrier_id=5 + i, num_threads=128) for i in range(num_epi_wg)
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512
        # epilogue: 8 accumulator (token) columns per TMEM load, 16 token rows staged per pass
        self.epi_cols = 8
        self.stage_rows = 16
        self.num_b_stage = num_b_stage
        self.num_meta_stage = num_meta_stage

    def _setup_attributes(self):
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (self.mma_tiler[0], cute.round_up(self.mma_tiler[1], 128))
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn_sfb,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * mma_inst_tile_k)
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,)
        )

        self.mma_tiler_a = (self.mma_tiler[0] * self.h_blocks, self.mma_tiler[1], self.mma_tiler[2])
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.mma_tiler[0] // sf_atom_mn) * mma_inst_tile_k * self.h_blocks
        self.num_sfb_tmem_cols = (self.mma_tiler_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_acc_stage = min(
            8, (self.num_tmem_alloc_cols - self.num_sf_tmem_cols) // self.n_tok
        )
        self.num_accumulator_tmem_cols = self.n_tok * self.num_acc_stage
        # A warpgroup skips the accumulator stages of the other warpgroups' tiles without waiting on
        # them, so its next wait is num_epi_wg * h_blocks stages ahead: that must stay within one lap
        # of the ring or the barrier parity of the previous lap would satisfy it.
        assert self.num_epi_wg * self.h_blocks <= self.num_acc_stage, (
            f"num_epi_wg * h_blocks ({self.num_epi_wg * self.h_blocks}) > accumulator stages ({self.num_acc_stage})"
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler_a, self.a_dtype, 1
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, 1
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.h_blocks
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        a_bytes_per_stage = cute.size_in_bytes(
            self.a_dtype, self.a_smem_layout_staged
        ) + cute.size_in_bytes(self.sf_dtype, self.sfa_smem_layout_staged)
        b_bytes_per_stage = cute.size_in_bytes(
            self.b_dtype, self.b_smem_layout_staged
        ) + cute.size_in_bytes(self.sf_dtype, self.sfb_smem_layout_staged)
        meta_bytes = self.n_tok * 8 * self.num_meta_stage
        c_bytes = self.num_epi_wg * self.stage_rows * self.mma_tiler[0] * 2
        reserved = 3072 + meta_bytes + c_bytes + 1024 + self.num_b_stage * b_bytes_per_stage
        self.num_ab_stage = (
            self.num_smem_capacity // self.occupancy - reserved
        ) // a_bytes_per_stage

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler_a, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_b_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage * self.h_blocks
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_b_stage
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        out: cute.Tensor,
        sfx: cute.Tensor,
        sfw: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
    ):
        """x: permuted activations (m, k, 1); w: expert weights (n, k, l); out: (num_tokens, n, 1)."""
        self.a_dtype: Type[cutlass.Numeric] = w.element_type
        self.b_dtype: Type[cutlass.Numeric] = x.element_type
        self.out_dtype: Type[cutlass.Numeric] = out.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfw.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(w).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(x).mma_major_mode()
        self.topK = token_final_scales.shape[1]
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.out_dtype != cutlass.BFloat16):
            raise TypeError("swap-AB finalize writes BFloat16 output only")

        self._setup_attributes()

        sfw = cute.make_tensor(
            sfw.iterator, blockscaled_utils.tile_atom_to_shape_SF(w.shape, self.sf_vec_size)
        )
        sfx = cute.make_tensor(
            sfx.iterator, blockscaled_utils.tile_atom_to_shape_SF(x.shape, self.sf_vec_size)
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn_sfb,
        )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w, a_smem_layout, self.mma_tiler_a, tiled_mma, self.cluster_layout_vmnk.shape
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfw,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, x, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfx,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.num_tma_load_bytes_a = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + self.h_blocks * cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.num_tma_load_bytes_b = cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)

        @cute.struct
        class SharedStorage:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage], 16
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            meta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_meta_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                1024,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], 1024
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, self.num_epi_wg * self.stage_rows * self.mma_tiler[0]
                ],
                128,
            ]
            meta_token_idx: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.n_tok * self.num_meta_stage], 16
            ]
            meta_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.n_tok * self.num_meta_stage], 16
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            out,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.topK,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
        ).launch(
            grid=(max_active_clusters, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    @cute.jit
    def _wg_barrier(self, wg: cutlass.Int32):
        """Named barrier of epilogue warpgroup ``wg``."""
        for i in cutlass.range_constexpr(self.num_epi_wg):
            if wg == i:
                self.epi_wg_barriers[i].arrive_and_wait()

    def mainloop_s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_nkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_mk1: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_nkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_mk1: cute.Tensor,
        out: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        topK: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        # an accumulator stage is drained by the one epilogue warpgroup that owns its tile
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_wo_sched),
        )
        meta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.meta_mbar_ptr.data_ptr(),
            num_stages=self.num_meta_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp * self.num_epi_warps
            ),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sInfo = storage.sInfo.get_tensor(cute.make_layout((5, self.num_tile_stage), stride=(1, 5)))
        meta_layout = cute.make_layout((self.n_tok, self.num_meta_stage), stride=(1, self.n_tok))
        sMetaTokenIdx = storage.meta_token_idx.get_tensor(meta_layout)
        sMetaScale = storage.meta_scale.get_tensor(meta_layout)
        # epilogue staging of scaled BF16 rows: (h, staged token row, warpgroup), h contiguous
        sC = storage.sC.get_tensor(
            cute.make_layout(
                (self.mma_tiler[0], self.stage_rows, self.num_epi_wg),
                stride=(1, self.mma_tiler[0], self.mma_tiler[0] * self.stage_rows),
            )
        )

        # (bM, bK, loopN_h, loopK, loopL)
        gA_nkl = cute.local_tile(
            mA_nkl, cute.slice_(self.mma_tiler_a, (None, 0, None)), (None, None, None)
        )
        # (bN_tok, bK, loopM, loopK, 1)
        gB_mk1 = cute.local_tile(
            mB_mk1, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_nkl = cute.local_tile(
            mSFA_nkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_mk1 = cute.local_tile(
            mSFB_mk1, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )

        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        tCgA = thr_mma.partition_A(gA_nkl)
        tCgB = thr_mma.partition_B(gB_mk1)
        tCgSFA = thr_mma.partition_A(gSFA_nkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_mk1)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            0,
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        self.cta_sync_barrier.arrive_and_wait()
        griddepcontrol_wait()

        #
        # Scheduler warp: this CTA's contiguous range of (permuted tile, h block) tiles, h fastest.
        # sInfo[3]: bit 0 = valid, bit 1 = first tile of a run of the same permuted tile.
        #
        if warp_idx == self.sched_warp_id:
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )
            bid, _, _ = cute.arch.block_idx()
            num_ctas, _, _ = cute.arch.grid_dim()
            num_h_tiles = cutlass.Int32(cute.size(gA_nkl, mode=[2]))
            total = cutlass.Int32(num_non_exiting_tiles[0]) * num_h_tiles
            if cutlass.const_expr(self.sched_chunk == 0):
                idx = (cutlass.Int32(bid) * total) // cutlass.Int32(num_ctas)
                end = ((cutlass.Int32(bid) + 1) * total) // cutlass.Int32(num_ctas)
                chunk_left = end - idx
                stride = cutlass.Int32(0)
            else:
                idx = cutlass.Int32(bid) * self.sched_chunk
                end = total
                chunk_left = cutlass.Int32(self.sched_chunk)
                stride = (cutlass.Int32(num_ctas) - 1) * self.sched_chunk
            prev_tok = cutlass.Int32(-1)
            expert_idx = cutlass.Int32(0)
            mn_limit = cutlass.Int32(0)
            while idx < end:
                tok_tile = idx // num_h_tiles
                h_tile = idx - tok_tile * num_h_tiles
                new_tok = cutlass.Int32(0)
                if tok_tile != prev_tok:
                    new_tok = cutlass.Int32(2)
                    expert_idx = tile_idx_to_expert_idx[tok_tile]
                    mn_limit = tile_idx_to_mn_limit[tok_tile]
                    prev_tok = tok_tile
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = tok_tile
                    sInfo[(1, tile_info_producer_state.index)] = h_tile
                    sInfo[(2, tile_info_producer_state.index)] = expert_idx
                    sInfo[(3, tile_info_producer_state.index)] = 1 + new_tok
                    sInfo[(4, tile_info_producer_state.index)] = mn_limit
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                idx += 1
                chunk_left -= 1
                if chunk_left == 0:
                    idx += stride
                    chunk_left = cutlass.Int32(self.sched_chunk)
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # TMA warp: A = weights[expert][h block], B = rows of the permuted tile
        #
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_b_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(4):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] & 1) == 1
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                tok_tile = tile_info[0]
                h_tile = tile_info[1]
                expert_idx = tile_info[2]
                if (tile_info[3] & 2) != 0:
                    # first tile of a run: load the token operand of the permuted tile once
                    b_pipeline.producer_acquire(b_producer_state)
                    b_bar = b_pipeline.producer_get_barrier(b_producer_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, tok_tile * (self.tile_size // self.n_tok), 0, 0)],
                        tBsB[(None, b_producer_state.index)],
                        tma_bar_ptr=b_bar,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB[(None, tok_tile, 0, 0)],
                        tBsSFB[(None, b_producer_state.index)],
                        tma_bar_ptr=b_bar,
                    )
                    b_producer_state.advance()

                # K fits one stage: one A box of 128 * h_blocks rows plus one SFA box per 128 rows
                ab_pipeline.producer_acquire(ab_producer_state)
                tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, h_tile, 0, expert_idx)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=tma_bar,
                )
                for mb in cutlass.range_constexpr(self.h_blocks):
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA[(None, h_tile * self.h_blocks + mb, 0, expert_idx)],
                        tAsSFA[(None, ab_producer_state.index * self.h_blocks + mb)],
                        tma_bar_ptr=tma_bar,
                    )
                ab_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(4):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] & 1) == 1
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)
            b_pipeline.producer_tail(b_producer_state)

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            # one SFA region per 128-row block of a tile
            sfa_cols_per_block = self.num_sfa_tmem_cols // self.h_blocks
            tCtSFA_blocks = [
                cute.make_tensor(
                    cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + mb * sfa_cols_per_block,
                        dtype=self.sf_dtype,
                    ),
                    tCtSFA_layout,
                )
                for mb in range(self.h_blocks)
            ]
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            s2t_sfa = [
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA_blocks[mb])
                for mb in range(self.h_blocks)
            ]
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_b_stage
            )
            holds_b = cutlass.Boolean(False)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_flags = sInfo[(3, tile_info_consumer_state.index)]
            is_valid_tile = (tile_flags & 1) == 1
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                if (tile_flags & 2) != 0:
                    if holds_b:
                        b_pipeline.consumer_release(b_consumer_state)
                        b_consumer_state.advance()
                    b_pipeline.consumer_wait(b_consumer_state)
                    holds_b = cutlass.Boolean(True)
                    cute.copy(
                        tiled_copy_s2t_sfb,
                        tCsSFB_compact_s2t[(None, None, None, None, b_consumer_state.index)],
                        tCtSFB_compact_s2t,
                    )
                # K fits one stage: one A stage per tile, one accumulator stage per 128-row block
                ab_pipeline.consumer_wait(ab_consumer_state)
                num_kblocks = cute.size(tCrA, mode=[2])
                for mb in cutlass.range_constexpr(self.h_blocks):
                    tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = s2t_sfa[mb]
                    cute.copy(
                        tiled_copy_s2t_sfa,
                        tCsSFA_compact_s2t[
                            (None, None, None, None, ab_consumer_state.index * self.h_blocks + mb)
                        ],
                        tCtSFA_compact_s2t,
                    )
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_blocks[mb][sf_kblock_coord].iterator
                        )
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc[(None, 0, 0)],
                            tCrA[(None, mb, kblock_idx, ab_consumer_state.index)],
                            tCrB[(None, 0, kblock_idx, b_consumer_state.index)],
                            tCtAcc[(None, 0, 0)],
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_flags = sInfo[(3, tile_info_consumer_state.index)]
                is_valid_tile = (tile_flags & 1) == 1
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            if holds_b:
                b_pipeline.consumer_release(b_consumer_state)
                b_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Metadata warp: token index and alpha * token_final_scale of the valid rows, once per run
        #
        if warp_idx == self.meta_load_warp_id:
            meta_lane = tidx % self.threads_per_warp
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_meta_stage
            )
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(5):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] & 1) == 1
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                if (tile_info[3] & 2) != 0:
                    tile_m_start = tile_info[0] * self.tile_size
                    alpha_val = alpha[tile_info[2]]
                    meta_pipeline.producer_acquire(meta_producer_state)
                    meta_stage = meta_producer_state.index
                    for j in cutlass.range(self.n_tok // self.threads_per_warp, unroll_full=True):
                        r = meta_lane + j * self.threads_per_warp
                        permuted_row = tile_m_start + r
                        if permuted_row < tile_info[4]:
                            expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                            token_idx = expanded_idx // topK
                            topk_idx = expanded_idx % topK
                            sMetaTokenIdx[(r, meta_stage)] = token_idx
                            sMetaScale[(r, meta_stage)] = (
                                alpha_val * token_final_scales[(token_idx, topk_idx)]
                            )
                    meta_pipeline.producer_commit(meta_producer_state)
                    meta_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(5):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] & 1) == 1
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            meta_pipeline.producer_tail(meta_producer_state)

        #
        # Epilogue warpgroups: tile k of this CTA goes to warpgroup k % num_epi_wg; thread t of the
        # warpgroup owns output column h = h_block * 128 + t (TMEM lane t)
        #
        if warp_idx < self.num_epi_warps:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            epi_tidx = tidx % 128
            wg = tidx // 128

            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(self.epi_cols)), self.acc_dtype
            )
            # (128, EPI_COLS, 1, N_TOK / EPI_COLS, STAGE)
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], (128, self.epi_cols)
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_cAcc = thr_copy_t2r.partition_D(cute.make_identity_tensor((128, self.epi_cols)))
            tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, self.acc_dtype)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_meta_stage
            )
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(5):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] & 1) == 1
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            holds_meta = cutlass.Boolean(False)
            tile_k = cutlass.Int32(0)
            chunks_per_row = self.mma_tiler[0] // 8
            while is_valid_tile:
                if (tile_info[3] & 2) != 0:
                    if holds_meta:
                        meta_pipeline.consumer_release(meta_consumer_state)
                        meta_consumer_state.advance()
                    meta_pipeline.consumer_wait(meta_consumer_state)
                    holds_meta = cutlass.Boolean(True)
                if tile_k % self.num_epi_wg == wg:
                    meta_stage = meta_consumer_state.index
                    n_valid = cutlass.min(
                        cutlass.max(tile_info[4] - tile_info[0] * self.tile_size, cutlass.Int32(0)),
                        cutlass.Int32(self.n_tok),
                    )
                    for mb in cutlass.range_constexpr(self.h_blocks):
                        h0 = (tile_info[1] * self.h_blocks + mb) * self.mma_tiler[0]
                        acc_pipeline.consumer_wait(acc_consumer_state)
                        row0 = cutlass.Int32(0)
                        while row0 < n_valid:
                            rows = cutlass.min(n_valid - row0, cutlass.Int32(self.stage_rows))
                            # TMEM -> scaled BF16 rows in sC[:, 0:rows, wg]
                            for s in cutlass.range(
                                (rows + self.epi_cols - 1) // self.epi_cols, unroll=1
                            ):
                                cute.copy(
                                    tiled_copy_t2r,
                                    tTR_tAcc[
                                        (
                                            None,
                                            None,
                                            None,
                                            0,
                                            row0 // self.epi_cols + s,
                                            acc_consumer_state.index,
                                        )
                                    ],
                                    tTR_rAcc,
                                )
                                for c in cutlass.range_constexpr(self.epi_cols):
                                    j = s * self.epi_cols + c
                                    if j < rows:
                                        v = tTR_rAcc[c] * sMetaScale[(row0 + j, meta_stage)]
                                        sC[(epi_tidx, j, wg)] = v.to(cutlass.BFloat16)
                            self._wg_barrier(wg)
                            # 16-byte reds: chunk q covers h0 + 8 * (q % 16) of staged row q // 16
                            q = epi_tidx
                            while q < rows * chunks_per_row:
                                jr = q // chunks_per_row
                                col = (q % chunks_per_row) * 8
                                tok = sMetaTokenIdx[(row0 + jr, meta_stage)]
                                src = sC[(None, jr, wg)].iterator + col
                                dst = cute.domain_offset((tok, h0 + col, 0), out)
                                red_add_v8_bf16_from_smem(
                                    dst.iterator.llvm_ptr, src.toint().ir_value()
                                )
                                q += 128
                            self._wg_barrier(wg)
                            row0 += self.stage_rows
                        cute.arch.fence_view_async_tmem_load()
                        acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                else:
                    for mb in cutlass.range_constexpr(self.h_blocks):
                        acc_consumer_state.advance()
                tile_k += 1

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(5):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] & 1) == 1
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            if holds_meta:
                meta_pipeline.consumer_release(meta_consumer_state)
                meta_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epi_all_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Same pointer interface as the regular finalize kernel's wrapper (a = permuted activations)."""
        scale_k = k // scaling_vector_size
        num_tiles = m // tile_size
        x = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        sfx = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, scale_k // 4, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        out = cute.make_tensor(
            c_ptr, layout=cute.make_layout((num_tokens, n, 1), stride=(n, 1, num_tokens * n))
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
        w = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        sfw = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
        )
        return self(
            x,
            w,
            out,
            sfx,
            sfw,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
        )

    @staticmethod
    def can_implement(
        n: int, k: int, num_tokens: int, n_tok: int, tile_size: int, h_blocks: int = 2
    ) -> bool:
        # the whole K of a tile is one 256-wide stage
        if tile_size != 128 or n % (128 * h_blocks) != 0 or k % 64 != 0 or k > 256:
            return False
        if n_tok == 64:
            return num_tokens <= 64
        return n_tok == 128


__all__: Tuple[str, ...] = ("Sm100BlockScaledContiguousGroupedGemmFinalizeSwapABKernel",)

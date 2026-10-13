# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-M FP8-weight GEMM for the Kimi-K3 decode projections (SM100/SM103).

Computes ``out[M, N] = x[M, K] @ dequant(W[N, K])^T`` for an FP8 E4M3 weight with one UE8M0
scale per 128x128 block (the resmoothed weight and DeepGEMM-packed scale that
``fp8_swap_ab_gemm`` call sites hold) and a BF16 activation, optionally followed by the MoE
residual update ``residual + bf16(bf16(out) + shared)``.

* ``K3Fp8Quant1x128Kernel`` (:mod:`k3_latent_fp8_gemm`, ``sf_m_major=True``) quantizes the
  activation to E4M3 with the exact UE8M0 1x128 scale of ``fp8_quantize_1x128`` and writes the
  scale replicated over the four K32 slots of the UMMA scale atom in the CUTLASS layout
  ``[ceil(M/128)][K/128][512 B]``.
* :class:`K3Fp8MidGemmKernel` is a persistent swap-AB MXFP8 GEMM: the weight is the UMMA ``A``
  operand (``tcgen05.mma.cta_group::2`` with 256 weight rows per CTA pair) and the tokens are
  ``B`` (``tn`` tokens per tile, chosen per shape and M by :func:`tile_for`). The weight scale is
  expanded once per weight into the per-row scale atoms (:func:`expand_weight_scale`), so both
  scale operands arrive by TMA.
  There is no split-K: every output is the FP32 accumulation chain over K of DeepGEMM, with the
  same MMA kind and K order, so the BF16 output is bit-identical to ``fp8_swap_ab_gemm``.
* With PDL the TMA warp streams the first weight stages of its first tile before
  ``griddepcontrol.wait``; only the activation and the epilogue inputs wait on the previous grid.
* The ``residual`` epilogue TMA-loads the shared-expert and residual sub-tiles next to the
  accumulator sub-tile and stores ``bf16(r + bf16(bf16(acc) + s))`` (each add in FP32), the
  rounding sequence of ``k3_latent_up_residual``.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.warp import (
    LdMatrix8x8x16bOp,
    LdMatrix16x8x8bOp,
    StMatrix8x8x16bOp,
    StMatrix16x8x8bOp,
)
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from . import k3_latent_fp8_gemm as _lat
from .custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from .dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

#: K extent of one pipeline stage and of the activation / weight scale blocks.
CTA_K = 128
#: Weight rows per scale block (and per scale-factor atom).
BLOCK = 128
#: tcgen05 MXF8 scale vector length (the 1x128 scales are replicated four times).
SF_VEC = 32
#: Supported token tiles (UMMA N).
SUPPORTED_TN = (64, 128, 192, 256)
#: Largest M the op accepts.
MAX_M = 8192

EPI_BF16 = "bf16"
EPI_RESIDUAL = "residual"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dsl_user_op
def _ldg_b32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class K3Fp8MidGemmKernel(Sm100BlockScaledPersistentDenseGemmKernel):
    """Persistent swap-AB MXFP8 GEMM ``C[Nw, Mtok] = W[Nw, K] x X[Mtok, K]^T`` (C column-major).

    Reuses the tile / pipeline / epilogue partitioning of the CUTLASS persistent block-scaled GEMM
    and differs in three places: the weight stages of the first tile are issued before
    ``griddepcontrol.wait`` (PDL), the epilogue can apply the MoE residual update, and the output
    is always BF16 without an alpha multiply.
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        epilogue: str = EPI_BF16,
        use_pdl: bool = True,
        res_stages: int = 2,
        sfb_packed: bool = False,
    ):
        super().__init__(SF_VEC, mma_tiler_mn, cluster_shape_mn, False, 1, True)
        if epilogue not in (EPI_BF16, EPI_RESIDUAL):
            raise ValueError(f"unknown epilogue {epilogue!r}")
        self.epilogue_kind = epilogue
        self.use_pdl = use_pdl
        # Activation scales as DeepGEMM-packed UE8M0 words (int32 ``[ceil(K/512)][ld]``, byte
        # ``kb % 4`` of word ``(kb // 4, token)``), the layout of the fused all-reduce + RMSNorm +
        # FP8 block quantization: an extra warp expands them into the scale atoms of each stage.
        self.sfb_packed = sfb_packed
        self.sfb_warp_id = self.tma_warp_id + 1
        if sfb_packed:
            self.threads_per_cta += self.threads_per_warp
        # 128-token scale atoms per stage (the 192-token tile reads a 256-token window).
        self.sfb_atoms = _ceil_div(mma_tiler_mn[1], BLOCK)
        # Shared + residual sub-tile buffers: the first ``res_stages`` sub-tiles of a tile are
        # loaded while its mainloop runs, later ones as earlier buffers free up (deeper
        # prefetch measured no faster on GB300: it costs mainloop stages).
        self.res_stages = res_stages

    # ------------------------------------------------------------------ host
    def _setup_with_residual_buffers(self):
        """``_setup_attributes`` with SMEM left for two residual + two shared epilogue stages."""
        # The residual mbarriers and the (1024-aligned) shared / residual buffers, which the
        # base stage computation does not know about.
        self.smem_capacity -= 3 * 1024
        self._setup_attributes()
        if self.epilogue_kind == EPI_RESIDUAL:
            c_stage_bytes = cute.size_in_bytes(
                self.c_dtype, cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            )
            self.smem_capacity -= 2 * self.res_stages * c_stage_bytes + 1024
            self._setup_attributes()
        self.res_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.res_stages
        )

    @cute.jit
    def wrapper(
        self,
        n_w: cutlass.Int64,
        m_tok: cutlass.Int64,
        k: cutlass.Int64,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        wsf_ptr: cute.Pointer,
        xsf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        shared_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        xsf_ld: Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Dynamic-shape entry: W ``[n_w, k]``, X ``[m_tok, k]`` (both K-major FP8), scale atoms
        ``[ceil(rows/128)][k/128][512]`` (UE8M0), C / shared / residual BF16 ``[m_tok, n_w]``.
        With ``sfb_packed``, ``xsf_ptr`` holds the packed activation scale words with leading
        dimension ``xsf_ld`` instead."""
        a_tensor = cute.make_tensor(
            w_ptr, layout=cute.make_ordered_layout((n_w, k, 1), order=(1, 0, 2))
        )
        b_tensor = cute.make_tensor(
            x_ptr, layout=cute.make_ordered_layout((m_tok, k, 1), order=(1, 0, 2))
        )
        c_layout = cute.make_ordered_layout((n_w, m_tok, 1), order=(0, 1, 2))
        c_tensor = cute.make_tensor(c_ptr, layout=c_layout)
        s_tensor = cute.make_tensor(shared_ptr, layout=c_layout)
        r_tensor = cute.make_tensor(residual_ptr, layout=c_layout)
        sf_k = (k + CTA_K - 1) // CTA_K
        sfa_tensor = cute.make_tensor(
            wsf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, (n_w + BLOCK - 1) // BLOCK, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        sfb_tensor = cute.make_tensor(
            xsf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, (m_tok + BLOCK - 1) // BLOCK, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        self.run(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            s_tensor,
            r_tensor,
            xsf_ptr,
            xsf_ld,
            Int32(m_tok),
            max_active_clusters,
            stream,
        )

    @cute.jit
    def run(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        s_tensor: cute.Tensor,
        r_tensor: cute.Tensor,
        xsf_ptr: cute.Pointer,
        xsf_ld: Int32,
        m_tok: Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a_tensor.element_type
        self.b_dtype = b_tensor.element_type
        self.sf_dtype = sfa_tensor.element_type
        self.c_dtype = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        self._setup_with_residual_buffers()

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, cute.make_layout(new_shape, stride=new_stride)
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        if cutlass.const_expr(self.sfb_packed):
            # The scale-expanding warp writes the activation scale atoms, not TMA.
            if cutlass.const_expr(
                cute.cosize(self.sfb_smem_layout_staged) != self.num_ab_stage * self.sfb_atoms * 512
            ):
                raise ValueError("unexpected activation scale SMEM layout")
            sfb_copy_size = 0
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, self.epi_tile
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), s_tensor, epi_smem_layout, self.epi_tile
        )
        tma_atom_r, tma_tensor_r = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), r_tensor, epi_smem_layout, self.epi_tile
        )
        self.epi_stage_bytes = cute.size_in_bytes(self.c_dtype, epi_smem_layout)

        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.swizzle_size,
            self.raster_along_m,
        )
        self.buffer_align_bytes = 1024
        res_cosize = (
            cute.cosize(self.res_smem_layout_staged.outer)
            if self.epilogue_kind == EPI_RESIDUAL
            else 8
        )

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            res_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.res_stages]
            sfb_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sS: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, res_cosize], self.buffer_align_bytes
            ]
            sR: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, res_cosize], self.buffer_align_bytes
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.mid_kernel(
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
            tma_atom_c,
            tma_tensor_c,
            tma_atom_s,
            tma_tensor_s,
            tma_atom_r,
            tma_tensor_r,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.res_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            xsf_ptr,
            xsf_ld,
            m_tok,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    # ---------------------------------------------------------------- device
    @cute.kernel
    def mid_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_mnl: cute.Tensor,
        tma_atom_r: cute.CopyAtom,
        mR_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        res_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        xsf_ptr: cute.Pointer,
        xsf_ld: Int32,
        m_tok: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
                cpasync.prefetch_descriptor(tma_atom_s)
                cpasync.prefetch_descriptor(tma_atom_r)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf
        res_full = storage.res_full_mbar_ptr.data_ptr()
        sfb_full = storage.sfb_full_mbar_ptr.data_ptr()

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
            if warp_idx == self.epilog_warp_id[0]:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(self.res_stages):
                        cute.arch.mbarrier_init(res_full + i, 1)
        if cutlass.const_expr(self.sfb_packed):
            if warp_idx == self.sfb_warp_id:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(self.num_ab_stage):
                        # Arrivals: the scale-expanding warp of each CTA of the MMA pair.
                        cute.arch.mbarrier_init(sfb_full + i, 2 if use_2cta_instrs else 1)
        cute.arch.mbarrier_init_fence()
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed(aligned=True)

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_block_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        #
        # TMA warp: weight stages of the first tile before the PDL wait, then everything else.
        #
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            n_pre = Int32(0)
            if work_tile.is_valid_tile:
                n_pre = _lat._min_s32(Int32(self.num_ab_stage), k_block_cnt)
                coord0 = work_tile.tile_idx
                m0 = coord0[0] // cute.size(tiled_mma.thr_id.shape)
                tAgA0 = tAgA[(None, m0, None, coord0[2])]
                tAgSFA0 = tAgSFA[(None, m0, None, coord0[2])]
                for kb in cutlass.range(n_pre, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA0[(None, kb)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA0[(None, kb)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    ab_producer_state.advance()
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_wait()
            # k blocks of the current tile whose weight stage is already in flight, and the
            # pipeline position of the first of them.
            k_pre = n_pre
            pre_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Activation halves of the prefetched stages (stages 0 .. k_pre - 1).
                for kb in cutlass.range(k_pre, unroll=1):
                    bar = ab_pipeline.producer_get_barrier(pre_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, kb)],
                        tBsB[(None, pre_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    if cutlass.const_expr(not self.sfb_packed):
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, kb)],
                            tBsSFB[(None, pre_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                    pre_state.advance()
                for kb in cutlass.range(k_pre, k_block_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, kb)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, kb)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, kb)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    if cutlass.const_expr(not self.sfb_packed):
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, kb)],
                            tBsSFB[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                    ab_producer_state.advance()
                k_pre = Int32(0)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Scale-expanding warp (packed activation scales)
        #
        if cutlass.const_expr(self.sfb_packed):
            if warp_idx == self.sfb_warp_id:
                self.expand_packed_sfb(
                    storage,
                    sfb_full,
                    tile_sched_params,
                    k_block_cnt,
                    xsf_ptr,
                    xsf_ld,
                    m_tok,
                    use_2cta_instrs,
                    cta_rank_in_cluster,
                )

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
            )
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
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
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = (
                        cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in range(k_block_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        if cutlass.const_expr(self.sfb_packed):
                            cute.arch.mbarrier_wait(
                                sfb_full + ab_consumer_state.index, ab_consumer_state.phase
                            )
                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx, ab_consumer_state.index)
                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kphase_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kphase_coord].iterator)
                            cute.gemm(
                                tiled_mma, tCtAcc, tCrA[kphase_coord], tCrB[kphase_coord], tCtAcc
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            if cutlass.const_expr(self.use_pdl):
                # Shared / residual inputs come from earlier grids.
                griddepcontrol_wait()
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=use_2cta_instrs
                )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
            )
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )
            if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
                sS = storage.sS.get_tensor(
                    res_smem_layout_staged.outer, swizzle=res_smem_layout_staged.inner
                )
                sR = storage.sR.get_tensor(
                    res_smem_layout_staged.outer, swizzle=res_smem_layout_staged.inner
                )
                gS_mnl = cute.local_tile(
                    mS_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
                )
                gR_mnl = cute.local_tile(
                    mR_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
                )
                tCgS = thr_mma.partition_C(gS_mnl)
                tCgR = thr_mma.partition_C(gR_mnl)
                _, bGS_sS, bGS_gS_partitioned = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_s, tCgS, epi_tile, sS
                )
                _, bGS_sR, bGS_gR_partitioned = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_r, tCgR, epi_tile, sR
                )
                # The shared / residual sub-tiles are read back with the transpose of the
                # accumulator's SMEM store (ldmatrix for stmatrix), so every thread gets the
                # elements of its own accumulator values.
                tiled_copy_s2r = cute.make_tiled_copy_D(
                    self._smem_load_atom_for(tiled_copy_t2r), tiled_copy_t2r
                )
                thr_s2r = tiled_copy_s2r.get_slice(epi_tidx)
                tSR_sS = thr_s2r.partition_S(sS)
                tSR_sR = thr_s2r.partition_S(sR)
                rS = cute.make_fragment(tTR_rC.shape, self.c_dtype)
                rR = cute.make_fragment(tTR_rC.shape, self.c_dtype)
                tSR_rS = tiled_copy_s2r.retile(rS)
                tSR_rR = tiled_copy_s2r.retile(rR)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )
            epilog_threads = 32 * len(self.epilog_warp_id)
            res_count = Int32(0)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
                    bGS_gS = bGS_gS_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                    bGS_gS = cute.group_modes(bGS_gS, 1, cute.rank(bGS_gS))
                    bGS_gR = bGS_gR_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                    bGS_gR = cute.group_modes(bGS_gR, 1, cute.rank(bGS_gR))
                    # The first sub-tiles' shared / residual loads overlap the mainloop.
                    if warp_idx == self.epilog_warp_id[0]:
                        for j in cutlass.range_constexpr(min(self.res_stages, subtile_cnt)):
                            sub = Int32(j)
                            if cutlass.const_expr(self.overlapping_accum):
                                if reverse_subtile:
                                    sub = Int32(subtile_cnt - 1 - j)
                            self._issue_residual_load(
                                tma_atom_s,
                                tma_atom_r,
                                bGS_sS,
                                bGS_sR,
                                bGS_gS,
                                bGS_gR,
                                res_full,
                                res_count + j,
                                sub,
                            )

                acc_pipeline.consumer_wait(acc_consumer_state)

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                            )
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
                        buf = res_count % self.res_stages
                        cute.arch.mbarrier_wait(res_full + buf, (res_count // self.res_stages) % 2)
                        cute.copy(tiled_copy_s2r, tSR_sS[(None, None, None, buf)], tSR_rS)
                        cute.copy(tiled_copy_s2r, tSR_sR[(None, None, None, buf)], tSR_rR)
                        s_vec = tiled_copy_r2s.retile(rS).load().to(cutlass.Float32)
                        r_vec = tiled_copy_r2s.retile(rR).load().to(cutlass.Float32)
                        t_vec = acc_vec.to(self.c_dtype).to(cutlass.Float32)
                        u_vec = (t_vec + s_vec).to(self.c_dtype).to(cutlass.Float32)
                        tRS_rC.store((r_vec + u_vec).to(self.c_dtype))
                    else:
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
                    )
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)]
                        )
                        if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
                            # Every thread has read this sub-tile's buffer: refill it.
                            if subtile_idx + self.res_stages < subtile_cnt:
                                nxt = subtile_idx + self.res_stages
                                if cutlass.const_expr(self.overlapping_accum):
                                    if reverse_subtile:
                                        nxt = subtile_cnt - 1 - nxt
                                self._issue_residual_load(
                                    tma_atom_s,
                                    tma_atom_r,
                                    bGS_sS,
                                    bGS_sR,
                                    bGS_gS,
                                    bGS_gR,
                                    res_full,
                                    res_count + self.res_stages,
                                    nxt,
                                )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
                        res_count = res_count + 1
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
                    )
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            c_pipeline.producer_tail()

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()

    @cute.jit
    def expand_packed_sfb(
        self,
        storage,
        sfb_full: cute.Pointer,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        k_block_cnt: Int32,
        xsf_ptr: cute.Pointer,
        xsf_ld: Int32,
        m_tok: Int32,
        use_2cta_instrs: cutlass.Constexpr,
        cta_rank_in_cluster: Int32,
    ):
        """Write each stage's activation scale atoms from the packed UE8M0 words.

        Lane ``l`` owns tokens ``128 a + 32 g + l`` (``g < 4``) of atom ``a``: the 16 bytes at
        ``l * 16`` hold their exponents, each replicated over the four K32 slots. A packed word
        holds four K blocks, so it is loaded once per four stages of a tile. Tokens past ``M``
        get the neutral exponent (their FP8 rows are TMA zero fill).
        """
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
        lane = cute.arch.lane_idx()
        empty = storage.ab_empty_mbar_ptr.data_ptr()
        sfb_base = storage.sSFB.data_ptr().toint()
        stage_bytes = self.sfb_atoms * 512
        leader = (cta_rank_in_cluster // 2) * 2
        rW = cute.make_rmem_tensor((self.sfb_atoms * 4,), Int32)
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        it = Int32(0)
        while work_tile.is_valid_tile:
            atom0 = (work_tile.tile_idx[1] * self.cta_tile_shape_mnk[1]) // BLOCK
            for kb in cutlass.range(k_block_cnt, unroll=1):
                stage = it % self.num_ab_stage
                if it >= self.num_ab_stage:
                    cute.arch.mbarrier_wait(empty + stage, ((it // self.num_ab_stage) % 2) ^ 1)
                if kb % 4 == 0:
                    for i in cutlass.range_constexpr(self.sfb_atoms):
                        for g in cutlass.range_constexpr(4):
                            t = (atom0 + i) * BLOCK + g * 32 + lane
                            w = Int32(0x7F7F7F7F)
                            if t < m_tok:
                                w = _ldg_b32(xsf_ptr.toint() + (Int64(kb // 4) * xsf_ld + t) * 4)
                            rW[i * 4 + g] = w
                shift = (kb % 4) * 8
                for i in cutlass.range_constexpr(self.sfb_atoms):
                    words = []
                    for g in cutlass.range_constexpr(4):
                        e = (rW[i * 4 + g] >> shift) & Int32(0xFF)
                        words.append(e * Int32(0x01010101))
                    _lat._sts_v4_b32(sfb_base + stage * stage_bytes + i * 512 + lane * 16, *words)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    if cutlass.const_expr(use_2cta_instrs):
                        cute.arch.mbarrier_arrive(sfb_full + stage, leader)
                    else:
                        cute.arch.mbarrier_arrive(sfb_full + stage)
                it = it + 1
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    def _smem_load_atom_for(self, tiled_copy_t2r):
        """SMEM -> register atom that inverts the accumulator's SMEM store atom."""
        store_atom = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        op = store_atom.op
        if isinstance(op, StMatrix8x8x16bOp):
            return cute.make_copy_atom(
                LdMatrix8x8x16bOp(op.transpose, op.num_matrices), self.c_dtype
            )
        if isinstance(op, StMatrix16x8x8bOp):
            return cute.make_copy_atom(
                LdMatrix16x8x8bOp(op.transpose, op.num_matrices), self.c_dtype
            )
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

    @cute.jit
    def _issue_residual_load(
        self, tma_atom_s, tma_atom_r, bGS_sS, bGS_sR, bGS_gS, bGS_gR, res_full, count, subtile
    ):
        """TMA-load one shared + residual sub-tile into buffer ``count % 2`` (one thread)."""
        buf = count % self.res_stages
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(res_full + buf, 2 * self.epi_stage_bytes)
        cute.copy(
            tma_atom_s, bGS_gS[(None, subtile)], bGS_sS[(None, buf)], tma_bar_ptr=res_full + buf
        )
        cute.copy(
            tma_atom_r, bGS_gR[(None, subtile)], bGS_sR[(None, buf)], tma_bar_ptr=res_full + buf
        )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@functools.cache
def _max_active_clusters(cluster_size: int) -> int:
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)


def _gmem_ptr(dtype, tensor: torch.Tensor, align: int = 16):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@functools.cache
def _compiled_gemm(
    tiler: Tuple[int, int],
    cluster: Tuple[int, int],
    epilogue: str,
    use_pdl: bool,
    res_stages: int = 2,
    sfb_packed: bool = False,
):
    kernel = K3Fp8MidGemmKernel(tiler, cluster, epilogue, use_pdl, res_stages, sfb_packed)
    dummy = 1 << 20
    f8 = make_ptr(cutlass.Float8E4M3FN, dummy, cute.AddressSpace.gmem, assumed_align=16)
    sf = make_ptr(cutlass.Float8E8M0FNU, dummy, cute.AddressSpace.gmem, assumed_align=16)
    bf = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel.wrapper,
        cutlass.Int64(256),
        cutlass.Int64(256),
        cutlass.Int64(256),
        f8,
        f8,
        sf,
        sf,
        bf,
        bf,
        bf,
        Int32(1),
        _max_active_clusters(cluster[0] * cluster[1]),
        stream,
        options="--opt-level 2",
    )


def expand_weight_scale(w_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """DeepGEMM-packed UE8M0 weight scale -> per-row scale atoms ``[ceil(N/128)][K/128][512]``.

    ``w_scale`` is int32 ``[N, ceil(K/512)]`` MN-major: byte ``kb % 4`` of word
    ``(row, kb // 4)`` holds the exponent of block ``(row // 128, kb)``. Every byte of atom
    ``(nb, kb)`` gets that block's exponent (128 rows x 4 replicated K32 slots).
    """
    kb = k // CTA_K
    nb = _ceil_div(n, BLOCK)
    words = w_scale.t().contiguous()  # [ceil(K/512), N]
    by = words.view(torch.uint8).view(-1, n, 4)  # [K/512, N, 4]
    e = by.permute(1, 0, 2).reshape(n, -1)[:, :kb]  # [N, K/128]
    e_blk = e[::BLOCK]  # [ceil(N/128), K/128]
    assert e_blk.shape[0] == nb
    return e_blk[:, :, None].expand(nb, kb, 512).contiguous().view(-1)


#: Token tile (UMMA N of the 256-row 2-CTA tile) per M bucket for the Kimi-K3 Helix decode
#: projections, from A/B runs of the whole chain (activation quantization + GEMM [+ residual
#: update]) against the production path on GB300 (CUDA graph, cold weights). ``None`` marks the
#: buckets where the production path (DeepGEMM, the low-M kernel or the fused latent chain) is
#: as fast or faster; up to 64 tokens the low-M kernel (activation quantized inside the GEMM)
#: always serves. Keys are (N rounded up to 128, K); bucket ``i`` covers M up to
#: ``_M_BUCKETS[i]``.
_M_BUCKETS = (64, 176, 336, 672, 1120, 1568, MAX_M)
_TUNED_TN = {
    (7168, 3584): (None, 64, None, None, 192, 192, 192),  # latent_up (+ residual update)
    (7168, 768): (None, 64, 128, 64, 128, 192, 128),  # KDA / MLA o_proj
    (18432, 1536): (None, 128, 128, 128, 192, 192, 256),  # MLA q_b
    (2176, 7168): (None, None, None, 64, 128, 192, 256),  # MLA fused q_a / kv_a (2112 rows)
    (768, 7168): (None, None, 64, 64, 64, 64, 128),  # MLA output gate
    (3072, 7168): (None, None, 64, 128, None, 128, 192),  # KDA fused q / k / v / g
}
#: Token tiles when the activation arrives already quantized with DeepGEMM-packed scales (the
#: per-tile scale expansion makes narrow tiles relatively dearer than with scale atoms). A
#: shape missing here uses the scale-atom table.
_TUNED_TN_PACKED = {
    (7168, 3584): (None, 64, 128, 256, 192, 192, 256),  # latent_up (fused all-reduce quant)
    (7168, 768): (None, 128, 128, 128, 128, 192, 192),  # o_proj (KDA verify / MLA gate quant)
}


def _heuristic_tn(m: int, n: int) -> int:
    """Token tile minimizing ``ceil(tiles / pairs) * (tn + 32)`` (ties: the wider tile)."""
    pairs = _max_active_clusters(2)
    w_tiles = _ceil_div(n, 256)
    best, best_cost = None, None
    for tn in SUPPORTED_TN:
        cost = _ceil_div(w_tiles * _ceil_div(m, tn), pairs) * (tn + 32)
        if best_cost is None or cost <= best_cost:
            best, best_cost = tn, cost
    return best


def tile_for(
    m: int, n: int, k: int, packed: bool = False
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """(MMA tiler, cluster) for an ``[M, K] x [N, K]^T`` problem, or ``None`` where the tuned
    table says the production path is faster. Shapes outside the table use a makespan heuristic.
    ``packed`` selects the table for prequantized activations with DeepGEMM-packed scales.
    """
    key = (_ceil_div(n, BLOCK) * BLOCK, k)
    row = (_TUNED_TN_PACKED.get(key) if packed else None) or _TUNED_TN.get(key)
    if row is None:
        return (256, _heuristic_tn(m, n)), (2, 1)
    for max_m, tn in zip(_M_BUCKETS, row):
        if m <= max_m:
            return None if tn is None else ((256, tn), (2, 1))
    return None


def run_k3_fp8_mid_gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    out: torch.Tensor,
    shared: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    *,
    tiler: Optional[Tuple[int, int]] = None,
    cluster: Optional[Tuple[int, int]] = None,
    use_pdl: bool = True,
    res_stages: int = 2,
    xs_packed_ld: int = 0,
) -> None:
    """Launch the GEMM on prequantized operands.

    ``w_sf`` is in the CUTLASS atom layout (:func:`expand_weight_scale`). ``xs`` is either the
    activation scale in the same atom layout (:func:`quantize_act`) or, with ``xs_packed_ld > 0``,
    DeepGEMM-packed int32 words ``[ceil(K/512)][xs_packed_ld]`` as produced by the fused
    all-reduce + RMSNorm + FP8 block quantization.
    """
    m, k = xq.shape
    n = w.shape[0]
    if tiler is None:
        tiler, cluster = tile_for(m, n, k) or ((256, _heuristic_tn(m, n)), (2, 1))
    epilogue = EPI_RESIDUAL if residual is not None else EPI_BF16
    if residual is None:
        shared = residual = out
    packed = xs_packed_ld > 0
    stream = cuda.CUstream(torch.cuda.current_stream(xq.device).cuda_stream)
    _compiled_gemm(tuple(tiler), tuple(cluster), epilogue, use_pdl, res_stages, packed)(
        cutlass.Int64(n),
        cutlass.Int64(m),
        cutlass.Int64(k),
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Float8E8M0FNU, w_sf),
        _gmem_ptr(cutlass.Float8E8M0FNU, xs),
        _gmem_ptr(cutlass.BFloat16, out),
        _gmem_ptr(cutlass.BFloat16, shared),
        _gmem_ptr(cutlass.BFloat16, residual),
        Int32(xs_packed_ld if packed else 1),
        stream,
    )


def quantize_act(x: torch.Tensor, *, use_pdl: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """E4M3 activation + UE8M0 1x128 scale atoms ``[ceil(M/128)][K/128][512]``."""
    m, k = x.shape
    xq = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty(_ceil_div(m, BLOCK) * (k // CTA_K) * 512, dtype=torch.uint8, device=x.device)
    _lat.quantize_act_1x128(x, xq, xs, use_pdl=use_pdl, sf_m_major=True)
    return xq, xs


def k3_fp8_mid_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    shared: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """``out = x @ dequant(w)^T`` (BF16), or the residual update when ``residual`` is given."""
    m = x.shape[0]
    n = w.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    xq, xs = quantize_act(x, use_pdl=kwargs.get("use_pdl", True))
    run_k3_fp8_mid_gemm(xq, xs, w, w_sf, out, shared, residual, **kwargs)
    return out


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3))


def can_run(
    x: torch.Tensor,
    w: torch.Tensor,
    w_sf: Optional[torch.Tensor],
    shared: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
) -> bool:
    """Whether :func:`k3_fp8_mid_gemm` accepts these inputs (non-raising check)."""
    if w_sf is None or x.dim() != 2 or w.dim() != 2 or not x.is_cuda:
        return False
    if x.dtype != torch.bfloat16 or w.dtype != torch.float8_e4m3fn:
        return False
    if not x.is_contiguous() or not w.is_contiguous():
        return False
    m, k = x.shape
    n = w.shape[0]
    if not 1 <= m <= MAX_M or w.shape[1] != k or k % CTA_K or n % 16:
        return False
    if w_sf.dtype != torch.uint8 or w_sf.numel() != _ceil_div(n, BLOCK) * (k // CTA_K) * 512:
        return False
    if x.data_ptr() % 16 or w.data_ptr() % 16:
        return False
    for t in (shared, residual):
        if t is not None and (
            t.dtype != torch.bfloat16
            or tuple(t.shape) != (m, n)
            or not t.is_contiguous()
            or t.data_ptr() % 16
        ):
            return False
    return _is_supported_device(x.device.index)


def _check_bf16_rows(t: Optional[torch.Tensor], m: int, n: int) -> bool:
    return (
        t is not None
        and t.dtype == torch.bfloat16
        and tuple(t.shape) == (m, n)
        and t.is_contiguous()
        and t.data_ptr() % 16 == 0
    )


def can_run_packed(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_sf: Optional[torch.Tensor],
    shared: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
) -> bool:
    """Whether :func:`k3_fp8_mid_gemm_packed_op` (or, with ``shared`` and ``residual``,
    :func:`k3_fp8_mid_residual_packed_op`) accepts these inputs (non-raising check).

    ``xq`` is the E4M3 activation ``[M, K]`` and ``xs`` its DeepGEMM-packed UE8M0 scale, int32
    ``[M, ceil(K/512)]`` with strides ``(1, ld)``, ``ld >= M`` (the fused all-reduce output).
    """
    if w_sf is None or xq.dim() != 2 or xs.dim() != 2 or w.dim() != 2 or not xq.is_cuda:
        return False
    if xq.dtype != torch.float8_e4m3fn or xs.dtype != torch.int32:
        return False
    if w.dtype != torch.float8_e4m3fn or not w.is_contiguous():
        return False
    m, k = xq.shape
    n = w.shape[0]
    if not 1 <= m <= MAX_M or w.shape[1] != k or k % CTA_K or n % 16:
        return False
    if not xq.is_contiguous() or xq.data_ptr() % 16 or w.data_ptr() % 16:
        return False
    if tuple(xs.shape) != (m, _ceil_div(k // CTA_K, 4)) or xs.stride(0) != 1:
        return False
    if xs.shape[1] > 1 and xs.stride(1) < m:
        return False
    if w_sf.dtype != torch.uint8 or w_sf.numel() != _ceil_div(n, BLOCK) * (k // CTA_K) * 512:
        return False
    if (shared is None) != (residual is None):
        return False
    if shared is not None and not (
        _check_bf16_rows(shared, m, n) and _check_bf16_rows(residual, m, n)
    ):
        return False
    return _is_supported_device(xq.device.index)


@torch.library.custom_op("trtllm::k3_fp8_mid_gemm", mutates_args=())
def k3_fp8_mid_gemm_op(x: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor) -> torch.Tensor:
    """BF16 ``x @ dequant(w)^T``; ``w_sf`` from :func:`expand_weight_scale`."""
    return k3_fp8_mid_gemm(x, w, w_sf)


@k3_fp8_mid_gemm_op.register_fake
def _(x: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::k3_fp8_mid_residual", mutates_args=())
def k3_fp8_mid_residual_op(
    x: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """``residual + bf16(bf16(x @ dequant(w)^T) + shared)`` (each add in FP32)."""
    out = torch.empty_like(residual)
    return k3_fp8_mid_gemm(x, w, w_sf, out, shared, residual)


@k3_fp8_mid_residual_op.register_fake
def _(x, w, w_sf, shared, residual):
    return torch.empty_like(residual)


def _packed_ld(xs: torch.Tensor, m: int) -> int:
    return xs.stride(1) if xs.shape[1] > 1 else max(m, 1)


@torch.library.custom_op("trtllm::k3_fp8_mid_gemm_packed", mutates_args=())
def k3_fp8_mid_gemm_packed_op(
    xq: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor
) -> torch.Tensor:
    """BF16 ``xq @ dequant(w)^T`` for an activation quantized upstream (E4M3 ``xq`` +
    DeepGEMM-packed UE8M0 scale ``xs``, see :func:`can_run_packed`)."""
    m, k = xq.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    tiler, cluster = tile_for(m, n, k, packed=True) or ((256, _heuristic_tn(m, n)), (2, 1))
    run_k3_fp8_mid_gemm(
        xq, xs, w, w_sf, out, tiler=tiler, cluster=cluster, xs_packed_ld=_packed_ld(xs, m)
    )
    return out


@k3_fp8_mid_gemm_packed_op.register_fake
def _(xq, xs, w, w_sf):
    return xq.new_empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::k3_fp8_mid_residual_packed", mutates_args=())
def k3_fp8_mid_residual_packed_op(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """``residual + bf16(bf16(xq @ dequant(w)^T) + shared)`` for an activation quantized upstream
    (E4M3 ``xq`` + DeepGEMM-packed UE8M0 scale ``xs``, see :func:`can_run_packed`)."""
    m, k = xq.shape
    n = w.shape[0]
    out = torch.empty_like(residual)
    tiler, cluster = tile_for(m, n, k, packed=True) or ((256, _heuristic_tn(m, n)), (2, 1))
    run_k3_fp8_mid_gemm(
        xq,
        xs,
        w,
        w_sf,
        out,
        shared,
        residual,
        tiler=tiler,
        cluster=cluster,
        xs_packed_ld=_packed_ld(xs, m),
    )
    return out


@k3_fp8_mid_residual_packed_op.register_fake
def _(xq, xs, w, w_sf, shared, residual):
    return torch.empty_like(residual)


__all__ = [
    "can_run",
    "can_run_packed",
    "k3_fp8_mid_gemm_packed_op",
    "k3_fp8_mid_residual_packed_op",
    "k3_fp8_mid_gemm_op",
    "k3_fp8_mid_residual_op",
    "K3Fp8MidGemmKernel",
    "MAX_M",
    "expand_weight_scale",
    "k3_fp8_mid_gemm",
    "quantize_act",
    "run_k3_fp8_mid_gemm",
    "tile_for",
]

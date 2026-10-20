# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) under the Apache-2.0
# license. Original file: flashinfer/norm/kernels/rmsnorm.py (RMSNormKernel.kernel).
# Original copyright: Copyright (c) 2025 by FlashInfer team.
"""FlashInfer's CuTe RMSNorm with the PDL trigger right after the wait.

``flashinfer.norm.kernels.rmsnorm.RMSNormKernel`` waits for the previous grid at its start
and triggers its dependents only after its last store, so a GEMM that follows it (the DSpark
drafter's q_b after q_a_layernorm) cannot become resident and fetch its weight while the norm
runs. :class:`EarlyTriggerRMSNormKernel` is that kernel with ``griddepcontrol.launch_dependents``
moved to right after ``griddepcontrol.wait``: the body is FlashInfer's, with FlashInfer's
helpers, so the values are the same. The dependents read the output only after their own wait
(which returns once this grid completed); before it they may read only data older than this
grid's predecessor (a GEMM's weight), complete by then.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import (
    get_cutlass_dtype,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)


class EarlyTriggerRMSNormKernel(RMSNormKernel):
    """``RMSNormKernel`` triggering its dependents right after its PDL wait."""

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


@functools.cache
def _compiled(dtype_str: str, H: int, sm_version: int, contiguous: bool):
    """FlashInfer's ``_get_compiled_rmsnorm_kernel`` (weight_bias 0, PDL) for this kernel."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = EarlyTriggerRMSNormKernel(dtype, H, 0.0, sm_version=sm_version)
    sym_m = cute.sym_int(64)
    if contiguous:
        tensor_align = math.gcd(128, H * (dtype.width // 8))
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
    else:
        x_fake = cute.runtime.make_fake_tensor(
            dtype,
            (sym_m, H),
            (cute.sym_int64(divisibility=kernel_obj.vec_size), 1),
            assumed_align=16,
        )
        y_fake = cute.runtime.make_fake_tensor(
            dtype,
            (sym_m, H),
            (cute.sym_int64(divisibility=kernel_obj.vec_size), 1),
            assumed_align=16,
        )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int64(1),
        Float32(1e-6),
        True,
        stream_fake,
        options="--enable-tvm-ffi",
    )


_DTYPE_STR = {torch.bfloat16: "bfloat16", torch.float16: "float16"}


def can_run(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`rmsnorm_early_trigger` serves this call (2D BF16/FP16 rows, unit inner
    stride, 16-byte aligned rows, an SM90+ GPU)."""
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype in _DTYPE_STR
        and weight.dtype == x.dtype
        and weight.dim() == 1
        and weight.shape[0] == x.shape[1]
        and x.stride(1) == 1
        and (x.stride(0) * x.element_size()) % 16 == 0
        and x.data_ptr() % 16 == 0
        and get_sm_version(x.device) >= 90
    )


def rmsnorm_early_trigger(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``flashinfer.norm.rmsnorm(x, weight, eps, enable_pdl=True)`` (same values) with the PDL
    trigger right after the wait. ``x`` is ``[M, H]`` with unit inner stride; returns a new
    contiguous tensor."""
    m, h = x.shape
    out = torch.empty((m, h), dtype=x.dtype, device=x.device)
    contiguous = x.is_contiguous() and m * h <= 2**31 - 1
    kernel = _compiled(_DTYPE_STR[x.dtype], h, get_sm_version(x.device), contiguous)
    kernel(x, weight, out, m, eps)
    return out


__all__ = ["EarlyTriggerRMSNormKernel", "can_run", "rmsnorm_early_trigger"]

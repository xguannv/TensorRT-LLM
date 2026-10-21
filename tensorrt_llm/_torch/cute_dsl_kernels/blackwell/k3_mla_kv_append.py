# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MLA decode: KV-side RMSNorm, latent row and paged FP8 KV append in one kernel (SM100/SM103).

The absorbed MLA decode with an FP8 KV cache and the query built by ``mla_fp8_q_absorb_nope``
(NoPE, ``q_rope_applied``) prepares its KV side with three kernels: the ``kv_a_layernorm``
RMSNorm (FlashInfer's CuTe DSL ``RMSNormKernel``) writing into the latent row, a copy of ``k_pe``
beside it, and ``mla_rope_generation`` (``applyMLARopeAndAssignQKVKernelGeneration``), which
rotates ``k_pe``, converts the row to FP8, appends it to the paged cache and fills the decode's
scale and scheduler buffers. :func:`k3_mla_kv_append` does all of it in one launch and writes the
same bytes:

* ``latent[t, :512]`` = ``rmsnorm(compressed_kv[t], weight, eps)`` with the arithmetic of
  FlashInfer's ``RMSNormKernel`` for ``H = 512`` (this kernel subclasses it and keeps its
  thread / value layout, reduction and normalization code), ``latent[t, 512:]`` = ``k_pe[t]``.
* The cache row of token ``t`` (``helix_local_slots[t]`` with speculative verify groups, else
  ``kv_len - seq_len + t % seq_len`` unless the request is inactive on this Helix rank) gets
  ``e4m3(float(latent[t, :512]) * scale)`` followed by the rotated ``k_pe``: each pair
  ``(x, y)`` becomes ``bf16(c * x - s * y), bf16(c * y + s * x)`` with ``(c, s)`` from the
  rotary table at the token's position, then ``e4m3(float(.) * scale)``. Conversions are
  ``cvt.rn.satfinite`` as in the C++ kernel's ``quantCopy``. With K3's identity rotary table
  (``c = 1``, ``s = 0``) every product is exact, so the result does not depend on how the C++
  compiler contracted the rotation.
* ``fmha_scheduler_counter[0] = 0``, ``cu_q_seqlens[b] = num_heads * seq_len * b`` and
  ``cu_kv_seqlens[b] = sum(kv_lens[:b])`` for ``b <= batch``, ``bmm1_scale = (h, h * log2(e))``
  and ``bmm2_scale = 1`` with the host-computed ``h = 1 / (q_scaling * sqrt(qk_head_dim))`` (no
  KV / output quantization scales: ``mla_rope_generation`` is called without them on this
  path).

Launch: ``ceil(tokens / 4)`` CTAs of 128 threads (one warp per token, as the FlashInfer norm).
PDL: before ``griddepcontrol.wait`` the kernel reads only the step's metadata and the rotary
table (see the argument in the kernel), and it releases its dependents right after the wait.
"""

from __future__ import annotations

import functools
import math
import struct
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
    from flashinfer.norm.utils import predicate_k, row_reduce_sum_multirow

    IS_AVAILABLE = True
except ImportError:  # pragma: no cover - FlashInfer without the CuTe DSL norm kernels
    RMSNormKernel = object
    IS_AVAILABLE = False

#: Latent (normalized) and rotary widths of a K3 MLA KV row.
KV_LORA_RANK = 512
ROPE_DIM = 64
ROW_DIM = KV_LORA_RANK + ROPE_DIM
#: Helix write gating: none, per-request inactive flags, per-token local slots.
HELIX_NONE = 0
HELIX_INACTIVE = 1
HELIX_SLOTS = 2

_LOG2E = 1.4426950408889634074


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ldg_s32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.s32 $0, [$1];", "=r,l"))


@dsl_user_op
def _ldg_u8(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.u8 $0, [$1];", "=r,l"))


@dsl_user_op
def _ldg_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ldg_v2_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _stg_v2_b32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None):
    """Four FP32 values -> four E4M3 bytes (``cvt.rn.satfinite``), ``v0`` in the low byte."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n"
            ".reg .b16 lo, hi;\n"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
            "mov.b32 $0, {lo, hi};\n"
            "}",
            "=r,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_lo_hi(word, *, loc=None, ip=None):
    """The two BF16 values of a 32-bit word as FP32 (low half first)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 a, b;\n"
        "shl.b32 a, $2, 16;\n"
        "and.b32 b, $2, 0xFFFF0000;\n"
        "mov.b32 $0, a;\n"
        "mov.b32 $1, b;\n"
        "}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _rotate_bf16x2(word, c, s, *, loc=None, ip=None):
    """GPT-J rotation of one BF16 pair: ``bf16x2(c * x - s * y, c * y + s * x)`` as FP32 values
    (already rounded to BF16), the products formed like the C++ ``rotary_embedding_transform``."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip), _fv(c, loc=loc, ip=ip), _fv(s, loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 x, y;\n"
        ".reg .f32 fx, fy, t, rx, ry;\n"
        ".reg .b16 hx, hy;\n"
        "shl.b32 x, $2, 16;\n"
        "and.b32 y, $2, 0xFFFF0000;\n"
        "mov.b32 fx, x;\n"
        "mov.b32 fy, y;\n"
        "mul.rn.f32 t, $4, fy;\n"
        "neg.f32 t, t;\n"
        "fma.rn.f32 rx, $3, fx, t;\n"
        "mul.rn.f32 t, $4, fx;\n"
        "fma.rn.f32 ry, $3, fy, t;\n"
        "cvt.rn.bf16.f32 hx, rx;\n"
        "cvt.rn.bf16.f32 hy, ry;\n"
        "mov.b32 x, {hx, hx};\n"
        "mov.b32 y, {hy, hy};\n"
        "and.b32 x, x, 0xFFFF0000;\n"
        "and.b32 y, y, 0xFFFF0000;\n"
        "mov.b32 $0, x;\n"
        "mov.b32 $1, y;\n"
        "}",
        "=f,=f,r,f,f",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _bulk_prefetch_l2(addr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(nbytes, loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


class _K3MlaKvAppendKernel(RMSNormKernel):
    """FlashInfer's ``RMSNormKernel`` (H = 512, BF16) whose epilogue also appends the FP8 cache
    row; everything up to the normalized BF16 values is the parent's code.

    ``ld_x`` / ``ld_kpe`` (elements) are the row strides of the latent and rotary inputs, compile
    time constants: both are views of the fused q_a / kv_a projection output.
    """

    def __init__(
        self,
        sm_version: int,
        ld_x: int,
        ld_kpe: int,
        helix_mode: int,
        has_positions: bool,
        prefetch_decode: bool = False,
    ):
        super().__init__(cutlass.BFloat16, KV_LORA_RANK, 0.0, sm_version=sm_version)
        # Block 0 brings the page-table rows of the batch's KV and the KV lengths into L2 for the
        # decode that follows (bulk prefetches stay in L2 until their first read).
        self.prefetch_decode = prefetch_decode
        if self.cluster_n != 1 or self.threads_per_row != 32 or self.vec_size != 8:
            raise ValueError("unexpected FlashInfer RMSNorm configuration for H = 512")
        if ld_x % 8 or ld_kpe % 8:
            raise ValueError(f"row strides must be multiples of 8 elements: {ld_x}, {ld_kpe}")
        self.ld_x = ld_x
        self.ld_kpe = ld_kpe
        self.helix_mode = helix_mode
        self.has_positions = has_positions

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        lat_ptr: cute.Pointer,
        lat_addr: Int64,
        kpe_addr: Int64,
        cos_sin_addr: Int64,
        pos_addr: Int64,
        kv_lens_addr: Int64,
        helix_addr: Int64,
        page_table_addr: Int64,
        page_table_ld: Int32,
        kv_base_addr: Int64,
        page_bytes: Int64,
        token_bytes: Int32,
        page_size: Int32,
        page_offset: Int32,
        counter_addr: Int64,
        bmm1_addr: Int64,
        bmm2_addr: Int64,
        cu_q_addr: Int64,
        cu_kv_addr: Int64,
        num_tokens: Int32,
        seq_len: Int32,
        num_heads: Int32,
        batch: Int32,
        eps: Float32,
        kv_scale: Float32,
        bmm1_scale: Float32,
        stream: cuda.CUstream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row, self.rows_per_block, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        m64 = Int64(num_tokens)
        mX = cute.make_tensor(x_ptr, cute.make_layout((m64, self.H), stride=(self.ld_x, 1)))
        mW = cute.make_tensor(w_ptr, cute.make_layout((self.H,)))
        mY = cute.make_tensor(lat_ptr, cute.make_layout((m64, self.H), stride=(ROW_DIM, 1)))
        self.kernel(
            mX,
            mW,
            mY,
            m64,
            eps,
            tv_layout,
            tiler_mn,
            lat_addr,
            kpe_addr,
            cos_sin_addr,
            pos_addr,
            kv_lens_addr,
            helix_addr,
            page_table_addr,
            page_table_ld,
            kv_base_addr,
            page_bytes,
            token_bytes,
            page_size,
            page_offset,
            counter_addr,
            bmm1_addr,
            bmm2_addr,
            cu_q_addr,
            cu_kv_addr,
            num_tokens,
            seq_len,
            num_heads,
            batch,
            kv_scale,
            bmm1_scale,
        ).launch(
            grid=[cute.ceil_div(num_tokens, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        lat_addr: Int64,
        kpe_addr: Int64,
        cos_sin_addr: Int64,
        pos_addr: Int64,
        kv_lens_addr: Int64,
        helix_addr: Int64,
        page_table_addr: Int64,
        page_table_ld: Int32,
        kv_base_addr: Int64,
        page_bytes: Int64,
        token_bytes: Int32,
        page_size: Int32,
        page_offset: Int32,
        counter_addr: Int64,
        bmm1_addr: Int64,
        bmm2_addr: Int64,
        cu_q_addr: Int64,
        cu_kv_addr: Int64,
        num_tokens: Int32,
        seq_len: Int32,
        num_heads: Int32,
        batch: Int32,
        kv_scale: Float32,
        bmm1_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        # This warp's token: the FlashInfer layout gives warp w of block b row 4 * b + w.
        row = bidx * self.rows_per_block + tidx // self.threads_per_row

        # Before the wait: only the step's metadata (lengths, Helix slots and positions, page
        # table) and the static rotary table, all written before this layer's projection ran.
        # PDL argument: this kernel follows the fused q_a / kv_a GEMM, which releases its
        # dependents at its start, before its own wait; so when this kernel is resident that
        # GEMM's predecessor (the A_pre attn_res kernel, which releases only after its own wait)
        # is running, hence every grid before A_pre is complete, the step's input preparation
        # included. Nothing written by the GEMM or later is read here.
        if cutlass.const_expr(self.prefetch_decode):
            if bidx == 0:
                for pb in cutlass.range(tidx, batch, self.num_threads, unroll=1):
                    pages = (_ldg_s32(kv_lens_addr + Int64(pb) * 4) + page_size - 1) // page_size
                    first = page_table_addr + Int64(pb) * Int64(page_table_ld) * 4
                    lo = first & Int64(-16)
                    hi = (first + Int64(pages) * 4 + 15) & Int64(-16)
                    if hi > lo:
                        _bulk_prefetch_l2(lo, Int32(hi - lo))
                if tidx == 0:
                    lo = kv_lens_addr & Int64(-16)
                    _bulk_prefetch_l2(
                        lo, Int32(((kv_lens_addr + Int64(batch) * 4 + 15) & Int64(-16)) - lo)
                    )
        valid = row < num_tokens
        b = Int32(0)
        local = Int32(0)
        slot = Int32(0)
        page = Int32(0)
        write = cutlass.Boolean(False)
        c0 = s0 = c1 = s1 = c2 = s2 = c3 = s3 = Float32(0.0)
        if valid:
            b = row // seq_len
            local = row - b * seq_len
            kv_len = _ldg_s32(kv_lens_addr + Int64(b) * 4)
            slot = kv_len - seq_len + local
            position = kv_len - seq_len + local
            write = cutlass.Boolean(True)
            if cutlass.const_expr(self.helix_mode == HELIX_SLOTS):
                slot = _ldg_s32(helix_addr + Int64(row) * 4)
                write = slot >= 0
            if cutlass.const_expr(self.helix_mode == HELIX_INACTIVE):
                write = _ldg_u8(helix_addr + Int64(b)) == 0
            if cutlass.const_expr(self.has_positions):
                position = _ldg_s32(pos_addr + Int64(row) * 4)
            if write:
                page = _ldg_s32(
                    page_table_addr
                    + (Int64(b) * Int64(page_table_ld) + Int64(slot // page_size)) * 4
                )
                if lane < ROPE_DIM // 8:
                    coef = cos_sin_addr + (Int64(position) * ROPE_DIM + Int64(lane) * 4) * 8
                    c0, s0 = _ldg_v2_f32(coef)
                    c1, s1 = _ldg_v2_f32(coef + 8)
                    c2, s2 = _ldg_v2_f32(coef + 16)
                    c3, s3 = _ldg_v2_f32(coef + 24)

        _griddepcontrol_wait()
        # Everything this kernel's dependents read is stored after the wait; releasing them now
        # only lets them become resident earlier.
        _griddepcontrol_launch_dependents()

        # ------------------------------------------------------------------------------------
        # FlashInfer RMSNormKernel.kernel (cluster_n == 1), unchanged up to the BF16 output.
        # ------------------------------------------------------------------------------------
        H = self.H
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        mbar_ptr = None

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mY.element_type, num_bits_per_copy=copy_bits
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=copy_bits
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
        sum_sq = row_reduce_sum_multirow(x_sq, threads_per_row, reduction_buffer, mbar_ptr, 1)

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        # ------------------------------------------------------------------------------------
        # KV append: ``applyMLARopeAndAssignQKVKernelGeneration`` with ``q_rope_applied`` (the
        # rotary block and the eight latent-copy blocks of its grid) for this warp's token.
        # ------------------------------------------------------------------------------------
        if valid:
            # latent[t, 512:] = k_pe[t]: lanes 0..7, 16 bytes each.
            k0 = Int32(0)
            k1 = Int32(0)
            k2 = Int32(0)
            k3 = Int32(0)
            if lane < ROPE_DIM // 8:
                k0, k1, k2, k3 = _ldg_v4_b32(
                    kpe_addr + (Int64(row) * self.ld_kpe + Int64(lane) * 8) * 2
                )
                _stg_v4_b32(
                    lat_addr + (Int64(row) * ROW_DIM + KV_LORA_RANK + Int64(lane) * 8) * 2,
                    k0,
                    k1,
                    k2,
                    k3,
                )
            if write:
                dst = (
                    kv_base_addr
                    + Int64(page + page_offset) * page_bytes
                    + Int64(slot % page_size) * Int64(token_bytes)
                )
                # Latent part: this lane's normalized values, 8-value (16-byte) vectors.
                frag = cute.make_fragment_like(tXrO, Float32)
                frag.store(tXrO.load().to(Float32))
                for v in cutlass.range_constexpr(cute.size(tXrO) // 8):
                    col = Int32(tXcX[v * 8][1])
                    w0 = _e4m3x4(
                        frag[v * 8] * kv_scale,
                        frag[v * 8 + 1] * kv_scale,
                        frag[v * 8 + 2] * kv_scale,
                        frag[v * 8 + 3] * kv_scale,
                    )
                    w1 = _e4m3x4(
                        frag[v * 8 + 4] * kv_scale,
                        frag[v * 8 + 5] * kv_scale,
                        frag[v * 8 + 6] * kv_scale,
                        frag[v * 8 + 7] * kv_scale,
                    )
                    _stg_v2_b32(dst + Int64(col), w0, w1)
                # Rotary part: lanes 0..7, four pairs each.
                if lane < ROPE_DIM // 8:
                    r0, r1 = _rotate_bf16x2(k0, c0, s0)
                    r2, r3 = _rotate_bf16x2(k1, c1, s1)
                    r4, r5 = _rotate_bf16x2(k2, c2, s2)
                    r6, r7 = _rotate_bf16x2(k3, c3, s3)
                    w0 = _e4m3x4(r0 * kv_scale, r1 * kv_scale, r2 * kv_scale, r3 * kv_scale)
                    w1 = _e4m3x4(r4 * kv_scale, r5 * kv_scale, r6 * kv_scale, r7 * kv_scale)
                    _stg_v2_b32(dst + KV_LORA_RANK + Int64(lane) * 8, w0, w1)

        # Decode scale and scheduler buffers (block 0, thread 0, as the C++ kernel).
        if bidx == 0:
            if tidx == 0:
                _stg_b32(counter_addr, Int32(0))
                bmm1 = Float32(1.0) * Float32(1.0) * bmm1_scale
                _stg_f32(bmm1_addr, bmm1)
                _stg_f32(bmm1_addr + 4, bmm1 * Float32(_LOG2E))
                _stg_f32(bmm2_addr, Float32(1.0) * Float32(1.0))
                _stg_b32(cu_q_addr, Int32(0))
                _stg_b32(cu_kv_addr, Int32(0))
                acc = Int32(0)
                for i in cutlass.range(batch, unroll=1):
                    acc = acc + _ldg_s32(kv_lens_addr + Int64(i) * 4)
                    _stg_b32(cu_q_addr + Int64(i + 1) * 4, num_heads * seq_len * (i + 1))
                    _stg_b32(cu_kv_addr + Int64(i + 1) * 4, acc)


@functools.cache
def _compiled(
    sm_version: int,
    ld_x: int,
    ld_kpe: int,
    helix_mode: int,
    has_positions: bool,
    prefetch_decode: bool = False,
):
    kernel = _K3MlaKvAppendKernel(
        sm_version, ld_x, ld_kpe, helix_mode, has_positions, prefetch_decode
    )
    dummy = 1 << 20

    def bf16_ptr():
        return make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)

    z64, z32 = Int64(0), Int32(1)
    return cute.compile(
        kernel,
        bf16_ptr(),
        bf16_ptr(),
        bf16_ptr(),
        *([z64] * 7),
        z32,
        z64,
        z64,
        z32,
        z32,
        z32,
        *([z64] * 5),
        z32,
        z32,
        z32,
        z32,
        Float32(1e-6),
        Float32(1.0),
        Float32(1.0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _f32(value: float) -> float:
    """Round a Python float to the nearest FP32 value."""
    return struct.unpack("f", struct.pack("f", value))[0]


def host_bmm1_scale(q_scaling: float, qk_head_dim: int) -> float:
    """``1.f / (q_scaling * sqrt(static_cast<float>(qk_head_dim)))`` as ``mla_rope_generation``
    computes it on the host (``q_scaling`` a double, the square root a float)."""
    return _f32(1.0 / (q_scaling * _f32(math.sqrt(_f32(float(qk_head_dim))))))


def can_run(compressed_kv: torch.Tensor, k_pe: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`k3_mla_kv_append` serves these operands (BF16 rows of a K3 MLA KV half
    with 16-byte aligned rows, on SM100 / SM103)."""
    if not IS_AVAILABLE or not compressed_kv.is_cuda:
        return False
    major, minor = torch.cuda.get_device_capability(compressed_kv.device)
    if (major, minor) not in ((10, 0), (10, 3)):
        return False
    return (
        compressed_kv.dtype == torch.bfloat16
        and k_pe.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and compressed_kv.dim() == 2
        and k_pe.dim() == 2
        and compressed_kv.shape[-1] == KV_LORA_RANK
        and k_pe.shape[-1] == ROPE_DIM
        and k_pe.shape[0] == compressed_kv.shape[0]
        and weight.numel() == KV_LORA_RANK
        and weight.is_contiguous()
        and compressed_kv.stride(-1) == 1
        and k_pe.stride(-1) == 1
        and compressed_kv.stride(0) % 8 == 0
        and k_pe.stride(0) % 8 == 0
        and compressed_kv.data_ptr() % 16 == 0
        and k_pe.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def k3_mla_kv_append(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    latent: torch.Tensor,
    kv_lens: torch.Tensor,
    seq_len: int,
    num_heads: int,
    cos_sin: torch.Tensor,
    positions: Optional[torch.Tensor],
    helix_slots: Optional[torch.Tensor],
    helix_inactive: Optional[torch.Tensor],
    page_table: torch.Tensor,
    page_offset: int,
    kv_pages: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    bmm1_scale: torch.Tensor,
    bmm2_scale: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    host_bmm1: float,
    kv_scale: float = 1.0,
    prefetch_decode: bool = False,
) -> None:
    """One launch of the fused KV half on the current stream (see the module docstring).

    ``kv_pages``: the layer's FP8 page pool as ``[pages, page_size, row]`` (row >= 576) with unit
    stride along the row; cache row of token slot ``s`` of request ``b`` =
    ``kv_pages[page_table[b, s // page_size] + page_offset, s % page_size]``. ``page_table``:
    ``[batch, max_pages]`` int32 with unit column stride. ``helix_slots`` (per token, -1 = not
    owned) and ``helix_inactive`` (per request) are mutually exclusive. ``prefetch_decode``:
    also bring the page-table rows of the batch's KV and ``kv_lens`` into L2 for the decode
    (a hint only; the bytes written do not change).
    """
    num_tokens = compressed_kv.shape[0]
    batch = kv_lens.shape[0]
    if num_tokens == 0:
        return
    if num_tokens != batch * seq_len:
        raise ValueError(f"{num_tokens} tokens is not batch {batch} x seq_len {seq_len}")
    if helix_slots is not None and helix_inactive is not None:
        raise ValueError("helix_slots and helix_inactive are mutually exclusive")
    if kv_pages.dtype not in (torch.float8_e4m3fn, torch.uint8) or kv_pages.stride(-1) != 1:
        raise ValueError("kv_pages must be an FP8 (or uint8) pool with a unit row stride")
    if page_table.dtype != torch.int32 or page_table.stride(-1) != 1:
        raise ValueError("page_table must be int32 with a unit column stride")
    if latent.shape != (num_tokens, ROW_DIM) or not latent.is_contiguous():
        raise ValueError(f"latent must be a contiguous [{num_tokens}, {ROW_DIM}] tensor")
    for t, dtype in (
        (kv_lens, torch.int32),
        (cu_q_seqlens, torch.int32),
        (cu_kv_seqlens, torch.int32),
        (bmm1_scale, torch.float32),
        (bmm2_scale, torch.float32),
    ):
        if t.dtype != dtype:
            raise ValueError(f"expected {dtype}, got {t.dtype}")
    helix_mode = HELIX_NONE
    helix_addr = 0
    if helix_slots is not None:
        helix_mode, helix_addr = HELIX_SLOTS, helix_slots.data_ptr()
    elif helix_inactive is not None:
        helix_mode, helix_addr = HELIX_INACTIVE, helix_inactive.data_ptr()
    major, minor = torch.cuda.get_device_capability(compressed_kv.device)
    kernel = _compiled(
        major * 10 + minor,
        compressed_kv.stride(0),
        k_pe.stride(0),
        helix_mode,
        positions is not None,
        prefetch_decode,
    )

    def bf16_ptr(t):
        return make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    kernel(
        bf16_ptr(compressed_kv),
        bf16_ptr(weight),
        bf16_ptr(latent),
        Int64(latent.data_ptr()),
        Int64(k_pe.data_ptr()),
        Int64(cos_sin.data_ptr()),
        Int64(positions.data_ptr() if positions is not None else 0),
        Int64(kv_lens.data_ptr()),
        Int64(helix_addr),
        Int64(page_table.data_ptr()),
        Int32(page_table.stride(0)),
        Int64(kv_pages.data_ptr()),
        Int64(kv_pages.stride(0) * kv_pages.element_size()),
        Int32(kv_pages.stride(1) * kv_pages.element_size()),
        Int32(kv_pages.shape[1]),
        Int32(page_offset),
        Int64(fmha_scheduler_counter.data_ptr()),
        Int64(bmm1_scale.data_ptr()),
        Int64(bmm2_scale.data_ptr()),
        Int64(cu_q_seqlens.data_ptr()),
        Int64(cu_kv_seqlens.data_ptr()),
        Int32(num_tokens),
        Int32(seq_len),
        Int32(num_heads),
        Int32(batch),
        Float32(eps),
        Float32(kv_scale),
        Float32(host_bmm1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


__all__ = ["IS_AVAILABLE", "can_run", "host_bmm1_scale", "k3_mla_kv_append"]

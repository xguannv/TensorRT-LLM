# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Helix all-to-all and combine with an output gate (SM100 / SM103).

:class:`HelixA2AGatedKernel` is :class:`~.helix_a2a_fused.HelixA2AFusedKernel` (same senders, receive buffers,
tickets and combine) whose receivers multiply each combined BF16 value by ``sigmoid(gate)`` before storing it::

    out[t, h * D + d] = bf16(bf16(combine[t, h, d]) * sigmoid(gate[t, gate_col + h * D + d]))

the arithmetic of the output gate of ``k3_fp8_mla_output`` (``sigmoid`` here is ``1 / (1 + 2^(-x log2 e))`` with
``ex2.approx``). A receiver thread loads the 8 gate values of its 16-byte output column after
``griddepcontrol.wait`` and before polling the records.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .helix_a2a_fused import (
    _CHUNK_RECORDS,
    _NEG_INF_BITS,
    _STAGES,
    HelixA2AFusedKernel,
    _all_valid2,
    _all_valid4,
    _as_f32,
    _asm,
    _atom_add_u64,
    _bf16x2_fma,
    _ceil_div,
    _corrected_exp,
    _fadd,
    _fmax,
    _fmul,
    _fv,
    _globaltimer,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _iv,
    _ld_nc_v4,
    _ld_vol_v2,
    _ld_vol_v4,
    _lds_b64,
    _pack_bf16x2,
    _rcp,
    _shfl_bfly,
    _shfl_idx,
    _st_sentinel_v2,
    _st_sentinel_v4,
    _st_v4,
    _stg_u64,
    _sts_b64,
    _unpack,
    supports,
)
from .k3_prof import HEADER as _PROF_HEADER
from .k3_prof import SHAPES as _PROF_SHAPES
from .k3_prof import SLOTS as _PROF_SLOTS

_PROF_CTAS, _PROF_MARKS = _PROF_SHAPES["a2a"]


@dsl_user_op
def _bf16x2_f32(word, *, loc=None, ip=None):
    """The two BF16 values of ``word`` as FP32 (low half first)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 l, h;\nshl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 $0, l;\nmov.b32 $1, h;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _sigmoid(x, *, loc=None, ip=None):
    """``1 / (1 + exp(-x))`` in FP32."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip)],
            "{\n.reg .f32 t;\nmul.f32 t, $1, 0fBFB8AA3B;\nex2.approx.ftz.f32 t, t;\n"
            "add.f32 t, t, 0f3F800000;\nrcp.rn.f32 $0, t;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


class HelixA2AGatedKernel(HelixA2AFusedKernel):
    """Exchange + combine + output gate of one Helix MLA layer (see the module docstring)."""

    def __init__(self, *args, prof_addr: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prof_addr = prof_addr  # diagnostic marks (k3_prof.py), 0: none

    @cute.jit
    def __call__(
        self,
        o_addr: Int64,
        s_addr: Int64,
        mask_addr: Int64,
        out_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        rank_stride: Int64,
        half_bytes: Int64,
        num_tokens: Int32,
        rank: Int32,
        n_send: Int32,
        n_recv: Int32,
        gate_addr: Int64,
        gate_stride: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            half: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            stage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _STAGES * _CHUNK_RECORDS * self.RV * 4], 128
            ]

        self.shared_storage = SharedStorage
        self.gated_kernel(
            o_addr,
            s_addr,
            mask_addr,
            out_addr,
            ws_addr,
            state_addr,
            rank_stride,
            half_bytes,
            num_tokens,
            rank,
            n_send,
            n_recv,
            gate_addr,
            gate_stride,
        ).launch(
            grid=(n_send + n_recv, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def gated_kernel(
        self,
        o_addr: Int64,
        s_addr: Int64,
        mask_addr: Int64,
        out_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        rank_stride: Int64,
        half_bytes: Int64,
        num_tokens: Int32,
        rank: Int32,
        n_send: Int32,
        n_recv: Int32,
        gate_addr: Int64,
        gate_stride: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        half_slot = storage.half.data_ptr().toint()
        # diagnostic marks (k3_prof.py, prof_addr != 0): thread 0 of each CTA, %globaltimer at
        # 0 entry | 1 grid wait returned | 2 ticket taken | 3 sender: send returned (bulk reads drained) |
        # 6 receiver: first token's gate loaded | 7 first token's poll done | 8 receive loop done
        t_entry = Int64(0)
        t_wait = Int64(0)
        pbase = Int64(0)
        if cutlass.const_expr(self.prof_addr != 0):
            if tidx == 0:
                t_entry = _globaltimer()
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        if cutlass.const_expr(self.prof_addr != 0):
            if tidx == 0:
                t_wait = _globaltimer()
        # The ticket protocol of HelixA2AFusedKernel.kernel.
        if tidx == 0:
            n_ctas = Int64(n_send + n_recv)
            ticket = _atom_add_u64(state_addr, Int64(1))
            if (ticket & Int64(0xFFFFFFFF)) == n_ctas - Int64(1):
                _atom_add_u64(state_addr, Int64(1 << 32) - n_ctas)
            _sts_b64(half_slot, ((ticket >> 32) & Int64(1)) * half_bytes)
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
            if cutlass.const_expr(self.prof_addr != 0):
                slot = (ticket >> 32) % Int64(_PROF_SLOTS)
                pbase = Int64(self.prof_addr) + Int64(8) * (
                    Int64(_PROF_HEADER)
                    + (slot * Int64(_PROF_CTAS) + Int64(bidx)) * Int64(_PROF_MARKS)
                )
                _stg_u64(pbase, t_entry)
                _stg_u64(pbase + Int64(8), t_wait)
                _stg_u64(pbase + Int64(16), _globaltimer())
        if bidx < n_send:
            self.send(
                tidx, bidx, o_addr, s_addr, mask_addr, ws_addr, half_slot, rank_stride,
                num_tokens, rank, n_send, storage.stage.data_ptr().toint(),
            )  # fmt: skip
            if cutlass.const_expr(self.prof_addr != 0):
                if tidx == 0:
                    _stg_u64(pbase + Int64(24), _globaltimer())
        else:
            cute.arch.barrier()
            self.receive_gated(
                tidx,
                bidx - n_send,
                out_addr,
                ws_addr + Int64(rank) * rank_stride + _lds_b64(half_slot),
                num_tokens,
                n_recv,
                gate_addr,
                gate_stride,
                pbase,
            )

    @cute.jit
    def receive_gated(
        self,
        tidx: Int32,
        rcta: Int32,
        out_addr: Int64,
        recv_base: Int64,
        num_tokens: Int32,
        n_recv: Int32,
        gate_addr: Int64,
        gate_stride: Int32,
        pbase: Int64,
    ):
        """``HelixA2AFusedKernel.receive`` with the gate applied to each rounded output before the store.
        ``gate_addr``: the first gate value of token 0 (BF16, this rank's heads, head-major); ``gate_stride``: the
        gate rows' stride in elements. ``pbase``: thread 0's mark slot (diagnostic, prof_addr != 0)."""
        P, RV, DV, CPH = self.P, self.RV, self.DV, self.CPH
        slot_t = tidx // self.tok_threads
        c = tidx - slot_t * self.tok_threads
        lane = tidx % 32
        group = lane - lane % 16
        h = c // CPH
        col = c - h * CPH
        src_l = col % 16
        valid = Int32(c < DV)
        has_stats = valid * Int32(src_l < P)
        owns_stats = has_stats * Int32(col < 16)
        t_first = rcta * self.tpb
        for t in cutlass.range(rcta * self.tpb + slot_t, num_tokens, n_recv * self.tpb, unroll=1):
            rec0 = recv_base + Int64(t * (P * RV)) * 16
            st_addr = rec0 + Int64(src_l * RV + DV) * 16 + Int64(h * 8)
            d_addr = rec0 + Int64(c) * 16
            stats = cute.make_rmem_tensor((2,), Int32)
            stats[0] = Int32(_NEG_INF_BITS)
            stats[1] = Int32(0)
            vals = cute.make_rmem_tensor((P * 4,), Int32)
            gw = cute.make_rmem_tensor((4,), Int32)
            if valid != 0:
                # column c of the token's gate row: 8 values, 16 bytes
                g = _ld_nc_v4(gate_addr + (Int64(t) * Int64(gate_stride) + Int64(c * 8)) * 2)
                for e in cutlass.range_constexpr(4):
                    gw[e] = g[e]
                if cutlass.const_expr(self.prof_addr != 0):
                    if tidx == 0:
                        if t == t_first:
                            _stg_u64(pbase + Int64(48), _globaltimer())
                ready = Int32(0)
                while ready == 0:
                    ready = Int32(1)
                    if has_stats != 0:
                        sv = _ld_vol_v2(st_addr)
                        stats[0] = sv[0]
                        stats[1] = sv[1]
                        ready = _all_valid2(sv[0], sv[1])
                    for s in cutlass.range_constexpr(P):
                        w = _ld_vol_v4(d_addr + Int64(s * RV * 16))
                        for e in cutlass.range_constexpr(4):
                            vals[s * 4 + e] = w[e]
                        ready = ready & _all_valid4(*w)
                if cutlass.const_expr(self.prof_addr != 0):
                    if tidx == 0:
                        if t == t_first:
                            _stg_u64(pbase + Int64(56), _globaltimer())

            mx = _as_f32(stats[0])
            gmax = mx
            for off in (1, 2, 4, 8):
                gmax = _fmax(gmax, _shfl_bfly(gmax, off))
            corrected = _corrected_exp(mx, gmax, _as_f32(stats[1]))
            total = corrected
            for off in (1, 2, 4, 8):
                total = _fadd(total, _shfl_bfly(total, off))
            weight = _fmul(corrected, _rcp(total))

            acc = [Float32(0.0)] * 8
            for s in cutlass.range_constexpr(P):
                ws = _shfl_idx(weight, group + s)
                for q in cutlass.range_constexpr(4):
                    acc[2 * q], acc[2 * q + 1] = _bf16x2_fma(
                        vals[s * 4 + q], ws, acc[2 * q], acc[2 * q + 1]
                    )
            if valid != 0:
                # a list comprehension, not a for statement: the loop is unrolled in Python and indexes gw statically
                o = [self._gated_pair(acc[2 * q], acc[2 * q + 1], gw[q]) for q in range(4)]
                _st_v4(out_addr + Int64(t * DV + c) * 16, *o)
                for s in cutlass.range_constexpr(P):
                    _st_sentinel_v4(d_addr + Int64(s * RV * 16))
            if owns_stats != 0:
                _st_sentinel_v2(st_addr)
        if cutlass.const_expr(self.prof_addr != 0):
            if tidx == 0:
                _stg_u64(pbase + Int64(64), _globaltimer())

    @cute.jit
    def _gated_pair(self, lo: Float32, hi: Float32, gate_word: Int32) -> Int32:
        """The BF16 pair ``bf16(bf16(lo) * sigmoid(g_lo)), bf16(bf16(hi) * sigmoid(g_hi))``."""
        v_lo, v_hi = _bf16x2_f32(_pack_bf16x2(lo, hi))
        g_lo, g_hi = _bf16x2_f32(gate_word)
        return _pack_bf16x2(v_lo * _sigmoid(g_lo), v_hi * _sigmoid(g_hi))


@functools.cache
def _compiled(
    cp_size: int, num_heads: int, head_dim: int, has_mask: bool, use_pdl: bool, prof_addr: int = 0
):
    kernel = HelixA2AGatedKernel(
        cp_size, num_heads, head_dim, has_mask, use_pdl, prof_addr=prof_addr
    )
    addr = Int64(1 << 20)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        kernel, addr, addr, addr, addr, addr, addr, Int64(1 << 29), Int64(1 << 24), Int32(1), Int32(0),
        Int32(1), Int32(1), addr, Int32(1024), stream,
    )  # fmt: skip
    return kernel, compiled


def helix_a2a_gated(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: torch.Tensor | None,
    out: torch.Tensor,
    gate: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    num_heads: int,
    head_dim: int,
    ws_addr: int,
    rank_stride: int,
    half_bytes: int,
    state: torch.Tensor,
    max_send_ctas: int,
    max_recv_ctas: int,
    use_pdl: bool = True,
) -> None:
    """``helix_a2a_fused`` (same arguments, buffers and launch counter) with ``out = combine * sigmoid(gate)``.

    ``gate``: BF16 ``[T, num_heads * head_dim]`` view, unit column stride, row stride a multiple of 8 elements and
    16-byte aligned rows (this rank's heads, head-major, pre-sigmoid)."""
    num_tokens = partial_o.shape[0]
    if num_tokens == 0:
        return
    assert gate.dtype == torch.bfloat16 and gate.stride(1) == 1 and gate.stride(0) % 8 == 0
    assert gate.data_ptr() % 16 == 0 and gate.shape[1] == num_heads * head_dim
    from . import k3_prof

    prof_addr = k3_prof.addr("a2a")
    if prof_addr:
        k3_prof.touch()
    kernel, compiled = _compiled(
        cp_size, num_heads, head_dim, zero_kv_mask is not None, use_pdl, prof_addr
    )
    n_chunks = _ceil_div(num_tokens * cp_size, _CHUNK_RECORDS)
    n_send = (
        max_send_ctas if n_chunks >= max_send_ctas else min(n_chunks, max(1, max_send_ctas // 2))
    )
    n_recv = min(max_recv_ctas, _ceil_div(num_tokens, kernel.tpb))
    if prof_addr and n_send + n_recv > _PROF_CTAS:
        raise ValueError(f"k3_prof a2a: {n_send + n_recv} CTAs > {_PROF_CTAS}")
    mask_addr = zero_kv_mask.data_ptr() if zero_kv_mask is not None else 0
    compiled(
        Int64(partial_o.data_ptr()),
        Int64(softmax_stats.data_ptr()),
        Int64(mask_addr),
        Int64(out.data_ptr()),
        Int64(ws_addr),
        Int64(state.data_ptr()),
        Int64(rank_stride),
        Int64(half_bytes),
        Int32(num_tokens),
        Int32(cp_rank),
        Int32(n_send),
        Int32(n_recv),
        Int64(gate.data_ptr()),
        Int32(gate.stride(0)),
        cuda.CUstream(torch.cuda.current_stream(partial_o.device).cuda_stream),
    )


__all__ = ["HelixA2AGatedKernel", "helix_a2a_gated", "supports"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix all-to-all + combine + V-up (W_UV) + output gate of one MLA layer in one kernel (SM100 / SM103).

Experimental (``TRTLLM_K3LL_MK_ATTN``): a DcpValue-style tail for the Kimi-K3 Helix MLA decode. It extends
:class:`~.helix_a2a_fused.HelixA2AFusedKernel`:

* Senders are that kernel's senders. A record is ``(token, source rank)``: the BF16 512-wide latent partials of the
  destination's ``H`` heads, then their FP32 ``(max, sum)`` pairs, read straight from the split read's
  ``[T, P * H, 2]`` statistics (one pair per head, no per-half copy).
* Receivers are clusters of ``KS`` CTAs, one cluster per (head, 16-token tile). CTA ``x`` of a cluster owns latent
  columns ``[x * 512 / KS, (x + 1) * 512 / KS)`` of the head. It polls the statistics of the tile's tokens, computes the
  combine weights with the same instruction sequence as the fused kernel, polls its columns of the ``P`` partials and
  combines them in source order (FMA, one BF16 rounding: bitwise the fused kernel's output), keeps the merged latent
  in shared memory and multiplies it by its ``512 / KS`` columns of W_UV (FP8 weights, exact in BF16, BF16 MMA,
  FP32 accumulation, the 128 x 128 block scale applied to the accumulator). The ``KS`` partial products are reduced
  through distributed shared memory: CTA ``x`` sums output columns ``[x * 128 / KS, ...)`` of every CTA of its cluster,
  rounds to BF16 and applies ``sigmoid(gate)``. The only synchronization beyond the CTA is that one cluster barrier.
* W_UV fragments are loaded before ``griddepcontrol.wait``.

Receive buffers, tickets and halves are the fused kernel's (see its module docstring).
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
    MAX_CP,
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
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _iv,
    _ld_vol_v2,
    _ld_vol_v4,
    _lds_b64,
    _lv,
    _pack_bf16x2,
    _rcp,
    _shfl_bfly,
    _st_sentinel_v2,
    _st_sentinel_v4,
    _st_v4,
    _sts_b64,
    _sts_v4,
    _unpack,
)

#: Latent width and value width per head (Kimi-K3 MLA).
LATENT = 512
VDIM = 128
#: Tokens per receiver tile (the MMA's M).
TILE = 16


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _ld_nc_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _ld_nc_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _e4m3x4_to_bf16x4(word, *, loc=None, ip=None):
    """Four FP8 e4m3 values (byte 0 first) as two BF16 pairs (exact: e4m3 fits BF16)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b16 lo, hi, a, b, c, d;\n.reg .b32 h0, h1;\n.reg .f32 f0, f1, f2, f3;\n"
        "mov.b32 {lo, hi}, $2;\n"
        "cvt.rn.f16x2.e4m3x2 h0, lo;\ncvt.rn.f16x2.e4m3x2 h1, hi;\n"
        "mov.b32 {a, b}, h0;\nmov.b32 {c, d}, h1;\n"
        "cvt.f32.f16 f0, a;\ncvt.f32.f16 f1, b;\ncvt.f32.f16 f2, c;\ncvt.f32.f16 f3, d;\n"
        "cvt.rn.bf16x2.f32 $0, f1, f0;\ncvt.rn.bf16x2.f32 $1, f3, f2;\n}",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``mma.sync.m16n8k16`` BF16 x BF16 + FP32."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(v, loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
        + [_fv(v, loc=loc, ip=ip) for v in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_cluster_v2(addr, cta, x, y, *, loc=None, ip=None):
    """Store two FP32 values at shared address ``addr`` of CTA ``cta`` of this cluster."""
    _asm(
        None,
        [
            _iv(addr, loc=loc, ip=ip),
            _iv(cta, loc=loc, ip=ip),
            _fv(x, loc=loc, ip=ip),
            _fv(y, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "st.shared::cluster.v2.f32 [ra], {$2, $3};\n}",
        "r,r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


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


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class HelixA2AValueKernel(HelixA2AFusedKernel):
    """Exchange + combine + W_UV + gate of one Helix MLA layer (see the module docstring).

    Args:
        cp_size: ``P`` (at most 16).
        num_heads: ``H``, heads per rank after the exchange (even).
        has_mask: Apply a per-token ``zero_kv_mask`` (uint8) on the sender side.
        use_pdl: Launch with programmatic dependent launch.
        k_split: ``KS``, CTAs per (head, token tile) cluster (4 or 8).
        debug_latent: Also store the merged BF16 latent (``[T, H * 512]``) for tests.
    """

    def __init__(
        self,
        cp_size: int,
        num_heads: int,
        has_mask: bool,
        use_pdl: bool,
        k_split: int = 4,
        debug_latent: bool = False,
    ):
        if not supports(cp_size, num_heads, k_split):
            raise ValueError(f"cp_size={cp_size}, num_heads={num_heads}, k_split={k_split}")
        self.P = cp_size
        self.H = num_heads
        self.has_mask = has_mask
        self.use_pdl = use_pdl
        self.debug_latent = debug_latent
        self.CPH = LATENT // 8
        self.DV = num_heads * self.CPH
        self.SV = num_heads // 2
        self.RV = self.DV + self.SV
        self.KS = k_split
        self.KC = LATENT // k_split  # latent columns per CTA
        self.VK = self.KC // 8  # 16-byte vectors per token per CTA
        self.KCH = self.KC // 32  # 32-wide k chunks per CTA
        self.threads = TILE * self.VK
        self.warps = self.threads // 32
        self.NW = VDIM // self.warps  # output columns per warp
        self.NT = self.NW // 8  # MMA n-tiles per warp
        self.OC = VDIM // k_split  # output columns finalized per CTA
        self.LP = self.KC * 2 + 16  # shared latent row pitch (bytes)

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
        w_addr: Int64,
        wscale_addr: Int64,
        gate_addr: Int64,
        lat_addr: Int64,
        num_tokens: Int32,
        rank: Int32,
        n_send_groups: Int32,
        n_tiles: Int32,
        gate_stride: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            half: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            stage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _STAGES * _CHUNK_RECORDS * self.RV * 4], 128
            ]
            lat: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, TILE * self.LP // 4], 16]
            wts: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, TILE * MAX_CP], 16]
            red: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.KS * TILE * self.OC], 16
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            o_addr,
            s_addr,
            mask_addr,
            out_addr,
            ws_addr,
            state_addr,
            rank_stride,
            half_bytes,
            w_addr,
            wscale_addr,
            gate_addr,
            lat_addr,
            num_tokens,
            rank,
            n_send_groups,
            n_tiles,
            gate_stride,
        ).launch(
            grid=(self.KS, n_send_groups + self.H * n_tiles, 1),
            block=(self.threads, 1, 1),
            cluster=(self.KS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        o_addr: Int64,
        s_addr: Int64,
        mask_addr: Int64,
        out_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        rank_stride: Int64,
        half_bytes: Int64,
        w_addr: Int64,
        wscale_addr: Int64,
        gate_addr: Int64,
        lat_addr: Int64,
        num_tokens: Int32,
        rank: Int32,
        n_send_groups: Int32,
        n_tiles: Int32,
        gate_stride: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kx, by, _ = cute.arch.block_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        half_slot = storage.half.data_ptr().toint()
        recv_y = by - n_send_groups
        head = recv_y % self.H
        tile = recv_y // self.H
        # W_UV fragments of this CTA (weights: safe to read before the PDL wait). Warp w, n-tile j, lane (g, q):
        # output n = w * NW + 8 j + g, latent k = x * KC + 32 c + 8 q .. + 7 of chunk c (B and A share the order).
        lane = tidx % 32
        warp = tidx // 32
        g = lane // 4
        q = lane % 4
        wb = cute.make_rmem_tensor((self.NT * self.KCH * 4,), Int32)
        if by >= n_send_groups:
            for j in cutlass.range_constexpr(self.NT):
                n = warp * self.NW + j * 8 + g
                for c in cutlass.range_constexpr(self.KCH):
                    raw = _ld_nc_v2(
                        w_addr + Int64((head * VDIM + n) * LATENT + kx * self.KC + 32 * c + 8 * q)
                    )
                    lo = _e4m3x4_to_bf16x4(raw[0])
                    hi = _e4m3x4_to_bf16x4(raw[1])
                    base = (j * self.KCH + c) * 4
                    wb[base] = lo[0]
                    wb[base + 1] = lo[1]
                    wb[base + 2] = hi[0]
                    wb[base + 3] = hi[1]
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        if tidx == 0:
            n_ctas = Int64((n_send_groups + self.H * n_tiles) * self.KS)
            ticket = _atom_add_u64(state_addr, Int64(1))
            if (ticket & Int64(0xFFFFFFFF)) == n_ctas - Int64(1):
                _atom_add_u64(state_addr, Int64(1 << 32) - n_ctas)
            _sts_b64(half_slot, ((ticket >> 32) & Int64(1)) * half_bytes)
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
        if by < n_send_groups:
            self.send(
                tidx, by * self.KS + kx, o_addr, s_addr, mask_addr, ws_addr, half_slot, rank_stride,
                num_tokens, rank, n_send_groups * self.KS, storage.stage.data_ptr().toint(),
            )  # fmt: skip
        else:
            cute.arch.barrier()
            self.receive_value(
                tidx,
                kx,
                head,
                tile,
                wb,
                out_addr,
                ws_addr + Int64(rank) * rank_stride + _lds_b64(half_slot),
                wscale_addr,
                gate_addr,
                lat_addr,
                num_tokens,
                gate_stride,
                storage.lat.data_ptr().toint(),
                storage.wts.data_ptr().toint(),
                storage.red.data_ptr().toint(),
            )

    @cute.jit
    def receive_value(
        self,
        tidx: Int32,
        kx: Int32,
        head: Int32,
        tile: Int32,
        wb: cute.Tensor,
        out_addr: Int64,
        recv_base: Int64,
        wscale_addr: Int64,
        gate_addr: Int64,
        lat_addr: Int64,
        num_tokens: Int32,
        gate_stride: Int32,
        lat_s: Int32,
        wts_s: Int32,
        red_s: Int32,
    ):
        P, RV, DV, VK, KC, LP = self.P, self.RV, self.DV, self.VK, self.KC, self.LP
        t0 = tile * TILE
        # 1. This thread's column (token r of the tile, 16-byte vector v of the CTA's columns): issue the loads of
        #    the P sources before the statistics.
        r = tidx // VK
        v = tidx - r * VK
        t = t0 + r
        col_vec = head * self.CPH + kx * VK + v
        d_addr = recv_base + Int64((t * P) * RV + col_vec) * 16
        vals = cute.make_rmem_tensor((P * 4,), Int32)
        if t < num_tokens:
            for s in cutlass.range_constexpr(P):
                w = _ld_vol_v4(d_addr + Int64(s * RV * 16))
                for e in cutlass.range_constexpr(4):
                    vals[s * 4 + e] = w[e]
        # 2. Combine weights of (token rr, source s) = (i // 16, i % 16): a 16-lane group per token, the fused kernel's
        #    instruction sequence.
        for it in cutlass.range_constexpr(TILE * MAX_CP // self.threads):
            i = tidx + it * self.threads
            rr = i // MAX_CP
            s = i % MAX_CP
            tt = t0 + rr
            st_addr = recv_base + Int64((tt * P + s) * RV + DV) * 16 + Int64(head * 8)
            stats = cute.make_rmem_tensor((2,), Int32)
            stats[0] = Int32(_NEG_INF_BITS)
            stats[1] = Int32(0)
            if (s < P) & (tt < num_tokens):
                ready = Int32(0)
                while ready == 0:
                    sv = _ld_vol_v2(st_addr)
                    stats[0] = sv[0]
                    stats[1] = sv[1]
                    ready = _all_valid2(sv[0], sv[1])
            mx = _as_f32(stats[0])
            gmax = mx
            for off in (1, 2, 4, 8):
                gmax = _fmax(gmax, _shfl_bfly(gmax, off))
            corrected = _corrected_exp(mx, gmax, _as_f32(stats[1]))
            total = corrected
            for off in (1, 2, 4, 8):
                total = _fadd(total, _shfl_bfly(total, off))
            weight = _fmul(corrected, _rcp(total))
            _sts_f32(wts_s + (rr * MAX_CP + s) * 4, weight)
        cute.arch.barrier()
        # 3. Combine this thread's column in source order, one BF16 rounding, into shared memory; re-arm the vectors.
        lat_row = lat_s + r * LP + v * 16
        if t < num_tokens:
            ready = Int32(0)
            while ready == 0:
                ready = Int32(1)
                for s in cutlass.range_constexpr(P):
                    ready = ready & _all_valid4(
                        vals[s * 4], vals[s * 4 + 1], vals[s * 4 + 2], vals[s * 4 + 3]
                    )
                if ready == 0:
                    for s in cutlass.range_constexpr(P):
                        w = _ld_vol_v4(d_addr + Int64(s * RV * 16))
                        for e in cutlass.range_constexpr(4):
                            vals[s * 4 + e] = w[e]
            acc = [Float32(0.0)] * 8
            for s in cutlass.range_constexpr(P):
                ws = _lds_f32(wts_s + (r * MAX_CP + s) * 4)
                for e in cutlass.range_constexpr(4):
                    acc[2 * e], acc[2 * e + 1] = _bf16x2_fma(
                        vals[s * 4 + e], ws, acc[2 * e], acc[2 * e + 1]
                    )
            o = [_pack_bf16x2(acc[2 * e], acc[2 * e + 1]) for e in range(4)]
            _sts_v4(lat_row, *o)
            for s in cutlass.range_constexpr(P):
                _st_sentinel_v4(d_addr + Int64(s * RV * 16))
            if cutlass.const_expr(self.debug_latent):
                _st_v4(
                    lat_addr + Int64(t * self.H * LATENT + head * LATENT + kx * KC + v * 8) * 2, *o
                )
        else:
            _sts_v4(lat_row, Int32(0), Int32(0), Int32(0), Int32(0))
        cute.arch.barrier()
        # 4. W_UV: [16 tokens x KC] x [KC x 128] on BF16 MMAs; then the block scale of the CTA's 128-column group.
        lane = tidx % 32
        warp = tidx // 32
        g = lane // 4
        q = lane % 4
        acc4 = cute.make_rmem_tensor((self.NT * 4,), Float32)
        for k in cutlass.range_constexpr(self.NT * 4):
            acc4[k] = Float32(0.0)
        for c in cutlass.range_constexpr(self.KCH):
            x = _lds_v4(lat_s + g * LP + (32 * c + 8 * q) * 2)
            y = _lds_v4(lat_s + (g + 8) * LP + (32 * c + 8 * q) * 2)
            for j in cutlass.range_constexpr(self.NT):
                base = (j * self.KCH + c) * 4
                for s in cutlass.range_constexpr(2):
                    d = _mma_bf16(
                        x[2 * s],
                        y[2 * s],
                        x[2 * s + 1],
                        y[2 * s + 1],
                        wb[base + 2 * s],
                        wb[base + 2 * s + 1],
                        acc4[j * 4],
                        acc4[j * 4 + 1],
                        acc4[j * 4 + 2],
                        acc4[j * 4 + 3],
                    )
                    for e in cutlass.range_constexpr(4):
                        acc4[j * 4 + e] = d[e]
        scale = _as_f32(
            _ld_nc_b32(wscale_addr + Int64(head * (LATENT // 128) + (kx * KC) // 128) * 4)
        )
        # 5. Partial products to the owner CTA of each output column (distributed shared memory), one cluster barrier.
        for j in cutlass.range_constexpr(self.NT):
            col = warp * self.NW + j * 8 + 2 * q
            owner = col // self.OC
            local = col - owner * self.OC
            for h8 in cutlass.range_constexpr(2):
                addr = red_s + ((kx * TILE + g + 8 * h8) * self.OC + local) * 4
                _st_cluster_v2(
                    addr,
                    owner,
                    _fmul(acc4[j * 4 + 2 * h8], scale),
                    _fmul(acc4[j * 4 + 2 * h8 + 1], scale),
                )
        _cluster_sync()
        # 6. This CTA's output columns: sum the KS partials, round to BF16, gate, store.
        OCP = self.OC // 2
        for it in cutlass.range_constexpr(_ceil_div(TILE * OCP, self.threads)):
            i = tidx + it * self.threads
            rr = i // OCP
            cp = i - rr * OCP
            tt = t0 + rr
            if (i < TILE * OCP) & (tt < num_tokens):
                s0 = Float32(0.0)
                s1 = Float32(0.0)
                for xk in cutlass.range_constexpr(self.KS):
                    a = red_s + ((xk * TILE + rr) * self.OC + 2 * cp) * 4
                    s0 = _fadd(s0, _lds_f32(a))
                    s1 = _fadd(s1, _lds_f32(a + 4))
                p0, p1 = _bf16x2_f32(_pack_bf16x2(s0, s1))
                col = kx * self.OC + 2 * cp
                g0, g1 = _bf16x2_f32(
                    _ld_nc_b32(gate_addr + Int64(tt * gate_stride + head * VDIM + col) * 2)
                )
                o = _pack_bf16x2(_fmul(p0, _sigmoid(g0)), _fmul(p1, _sigmoid(g1)))
                _st_b32(out_addr + Int64(tt * self.H * VDIM + head * VDIM + col) * 2, o)
        # 7. Every CTA of the cluster has read the statistics (before the cluster barrier): CTA 0 re-arms them.
        if kx == 0:
            for it in cutlass.range_constexpr(TILE * MAX_CP // self.threads):
                i = tidx + it * self.threads
                rr = i // MAX_CP
                s = i % MAX_CP
                tt = t0 + rr
                if (s < P) & (tt < num_tokens):
                    _st_sentinel_v2(
                        recv_base + Int64((tt * P + s) * RV + DV) * 16 + Int64(head * 8)
                    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def supports(cp_size: int, num_heads: int, k_split: int = 4) -> bool:
    """Whether :func:`helix_a2a_value` handles this shape."""
    return 1 <= cp_size <= MAX_CP and num_heads % 2 == 0 and k_split in (4, 8)


def record_bytes(num_heads: int) -> int:
    """Bytes of one ``(token, source rank)`` record in a receive buffer."""
    return num_heads * LATENT * 2 + num_heads * 8


@functools.cache
def _compiled(
    cp_size: int, num_heads: int, has_mask: bool, use_pdl: bool, k_split: int, debug: bool
):
    kernel = HelixA2AValueKernel(cp_size, num_heads, has_mask, use_pdl, k_split, debug)
    addr = Int64(1 << 20)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        kernel,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        Int64(1 << 29),
        Int64(1 << 24),
        addr,
        addr,
        addr,
        addr,
        Int32(1),
        Int32(0),
        Int32(1),
        Int32(1),
        Int32(1),
        stream,
    )
    return kernel, compiled


def default_send_groups(num_tokens: int, cp_size: int, max_send_ctas: int, k_split: int) -> int:
    """The fused kernel's sender count, in clusters of ``k_split`` CTAs."""
    n_chunks = _ceil_div(num_tokens * cp_size, _CHUNK_RECORDS)
    n_send = (
        max_send_ctas if n_chunks >= max_send_ctas else min(n_chunks, max(1, max_send_ctas // 2))
    )
    return _ceil_div(n_send, k_split) if n_send > 0 else 0


def helix_a2a_value(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: torch.Tensor | None,
    gate: torch.Tensor,
    w_uv: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    num_heads: int,
    ws_addr: int,
    rank_stride: int,
    half_bytes: int,
    state: torch.Tensor,
    max_send_ctas: int,
    receive: bool = True,
    k_split: int = 4,
    use_pdl: bool = True,
    latent_out: torch.Tensor | None = None,
) -> None:
    """Exchange, combine, W_UV and gate of ``T = partial_o.shape[0]`` tokens into ``out``.

    ``partial_o``: BF16 ``[T, P * H * 512]`` (destination-rank major); ``softmax_stats``: FP32 ``[T, P * H, 2]``;
    ``zero_kv_mask``: bool ``[T]`` or None; ``gate``: BF16 ``[T, H * 128]`` view (row stride arbitrary, unit column
    stride), the pre-sigmoid gate of this rank's heads; ``w_uv``: FP8 ``[H, 128, 512]``, this rank's V-up rows;
    ``w_scale``: FP32 ``[H, 4]`` (or ``[H, 1, 4]``), their 128 x 128 block scales; ``out``: BF16 ``[T, H * 128]``.
    Workspace arguments as :func:`~.helix_a2a_fused.helix_a2a_fused` (records of :func:`record_bytes`).
    ``max_send_ctas = 0`` runs only the receivers, ``receive = False`` only the senders (single-GPU tests).
    ``latent_out``: optional BF16 ``[T, H * 512]``, the merged latent (compiles the debug variant).
    """
    num_tokens = partial_o.shape[0]
    if num_tokens == 0:
        return
    assert gate.stride(1) == 1 and w_uv.is_contiguous() and w_scale.is_contiguous()
    debug = latent_out is not None
    kernel, compiled = _compiled(
        cp_size, num_heads, zero_kv_mask is not None, use_pdl, k_split, debug
    )
    n_send_groups = default_send_groups(num_tokens, cp_size, max_send_ctas, k_split)
    n_tiles = _ceil_div(num_tokens, TILE) if receive else 0
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
        Int64(w_uv.data_ptr()),
        Int64(w_scale.data_ptr()),
        Int64(gate.data_ptr()),
        Int64(latent_out.data_ptr() if debug else 0),
        Int32(num_tokens),
        Int32(cp_rank),
        Int32(n_send_groups),
        Int32(n_tiles),
        Int32(gate.stride(0)),
        cuda.CUstream(torch.cuda.current_stream(partial_o.device).cuda_stream),
    )

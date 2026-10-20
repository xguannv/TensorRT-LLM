# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MNNVL all-reduce + residual add + RMSNorm in one kernel (SM100 / SM103), bit-identical to
``trtllm::mnnvl_fusion_allreduce(..., RESIDUAL_RMS_NORM)``.

The C++ path runs the two-shot all-reduce (``twoshotAllreduceKernel``) and then ``rmsNormLamport``,
or for small inputs the one-shot kernel with the norm fused (``oneshotAllreduceFusionKernel``). Both
launch with the default shared-memory carveout, so while they wait for the other ranks a large
GEMM that follows them (the DSpark drafter's [q_a | kv_a] and gate_up projections) cannot become
resident on their SMs and fetch its weight early. :class:`K3DraftArNormKernel` does the whole
all-reduce + norm in one launch with the max-shared carveout and a few hundred bytes of shared
memory, and triggers its dependents right after its own ``griddepcontrol.wait``.

Work split: cluster ``(row, 0..C-1)`` of ``T``-thread CTAs per token row, thread ``t`` of cluster
CTA ``c`` owning the eight values ``[8 (T c + t), + 8)`` (the C++ launcher's ``adjustGridConfig``
for the norm; ``T x C`` is ``112 x 8`` for a 7168-wide row and up to 19 rows, ``224 x 4`` above).

* Transport (two-shot, whatever the C++ path would use): each rank stores its sanitized values of
  row ``t`` into slot ``(t // W, rank)`` of the scatter stage of the row's owner ``t % W``; the
  owner polls the ``W`` slots, sums them in FP32 in rank order, rounds to BF16, replaces ``-0`` by
  ``+0`` and multicasts the row into every rank's broadcast stage, which every rank polls.
* Numerics of the reproduced C++ kernel (``cpp_one_shot``): the one-shot kernel sums chunks of
  eight ranks above eight ranks and multiplies ``(x * r) * gamma``; the two-shot kernel sums up to
  sixteen ranks in one chunk and multiplies ``(gamma * x) * r``. The two-shot transport carries a
  one-shot sum unchanged: it never holds ``-0`` (see ``k3_lamport_ar.K3LamportFinishKernel``).
* Residual and norm as the C++ kernels: ``s = bf16(x + residual)`` (stored as the new residual),
  BF16 squares summed in element order, butterfly warp sums (a half warp first adds 0 for its
  missing upper half), the block sum of the warp sums by warp 0's butterfly, the C CTA sums of the
  row added in CTA order from 0 through distributed shared memory, ``r = rsqrt.approx(div.rn(sum,
  H) + eps)``.
* Lamport state: the flag words of ``LamportFlags`` (current / dirty buffer, buffer bytes, dirty
  stages, bytes to clear per stage, arrival counter); each warp counts one arrival once its lanes
  read them, every CTA clears its share of the dirty buffer, and thread 0 of CTA (0, 0) rotates the
  state like the C++ two-shot call once all warps arrived. C++ and CuTe calls can alternate on one
  workspace.

PDL: ``griddepcontrol.wait`` first, then the dependents are triggered. Everything this kernel reads
is read after its wait, and its dependents may read before their own wait only data older than
this kernel's predecessor (the drafter GEMMs read their weights), which is complete by then.
"""

from __future__ import annotations

import functools
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .k3_lamport_ar import (
    LamportWorkspace,
    clean4,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    ld_acquire_u32,
    ld_nc_u64,
    ld_v4,
    ld_volatile_v4,
    multimem_st_v4,
    red_release_add,
    sanitize_bf16x2,
    st_sentinel_v4,
    st_u32,
    st_v4,
)

#: Values per thread (one 16-byte access).
ELTS = 8
#: Lamport buffers per workspace.
NUM_LAMPORT_BUFFERS = 3
#: Ranks per partial sum of the one-shot kernel (``accumulateLamportRanksChunked``).
ONE_SHOT_RANK_CHUNK = 8
#: Ranks per partial sum of the two-shot kernel.
TWO_SHOT_RANK_CHUNK = 16
#: World sizes the C++ launchers dispatch.
WORLD_SIZES = (2, 4, 8, 16, 32, 64)


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


def _unpack(res, ty, cls, n, *, loc=None, ip=None):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _bf16x2_f32(word, *, loc=None, ip=None):
    """(low, high) BF16 of a pair as FP32."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
        "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to a BF16 pair (``lo`` in the low half)."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(lo, loc=loc, ip=ip), _fv(hi, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add_bf16x2(a, b, *, loc=None, ip=None):
    """Elementwise BF16 sum of two pairs (``add.rn.bf16x2``, the C++ ``__nv_bfloat16 +``)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sq_bf16x2(word, *, loc=None, ip=None):
    """Elementwise BF16 square of a pair (``mul.rn.bf16x2``)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(word, loc=loc, ip=ip)],
            "mul.rn.bf16x2 $0, $1, $1;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fadd(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rms_scale(total, dim_bits: int, eps, *, loc=None, ip=None):
    """``rsqrtf(total / H + eps)`` as the C++ kernels compute it (IEEE division by the float ``H``
    given by its bit pattern, then the add)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(total, loc=loc, ip=ip), _fv(eps, loc=loc, ip=ip)],
            "{\n.reg .f32 q;\n"
            f"div.rn.f32 q, $1, 0f{dim_bits:08X};\n"
            "add.rn.f32 q, q, $2;\nrsqrt.approx.f32 $0, q;\n}",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly(value, offset: int, mask: int, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, {mask};",
            "=f,f",
            loc=loc,
            ip=ip,
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
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _st_cluster_f32(local_addr, peer, value, *, loc=None, ip=None):
    """Store ``value`` at ``local_addr`` of cluster CTA ``peer``'s shared memory."""
    _asm(
        None,
        [_iv(local_addr, loc=loc, ip=ip), _iv(peer, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "st.shared::cluster.f32 [ra], $2;\n}",
        "r,r,f",
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
def _bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "", loc=loc, ip=ip)


def grid_config(num_tokens: int, hidden: int, sm_count: int) -> Optional[tuple[int, int]]:
    """``(threads per CTA, CTAs per cluster)`` of the C++ norm launch (``adjustGridConfig<true>``
    for BF16 rows of ``hidden``), or None when that launch does not use one load per thread and a
    cluster (then this kernel does not reproduce it)."""
    elts = ELTS
    threads_needed = -(-hidden // elts)
    cluster = 8
    while threads_needed % cluster != 0 and cluster > 1:
        cluster //= 2
    max_div = cluster
    block = -(-threads_needed // cluster)
    while block < 128 and cluster >= 2:
        block *= 2
        cluster //= 2
    while num_tokens * cluster > sm_count and cluster > 1 and block <= 512:
        block *= 2
        cluster //= 2
    while cluster < max_div:
        cand_cluster = cluster * 2
        cand_block = -(-threads_needed // cand_cluster)
        if cand_block < 64 or num_tokens * cand_cluster > sm_count:
            break
        cluster, block = cand_cluster, cand_block
    loads = 1
    while block > 1024:
        if cluster < 8:
            cluster <<= 1
        elif loads < 8:
            loads += 1
        else:
            break
        block = -(-threads_needed // (cluster * loads))
    if loads != 1 or cluster < 2 or block * cluster * elts != hidden:
        return None
    return block, cluster


class K3DraftArNormKernel:
    """All-reduce of ``[m, H]`` BF16 partials + residual add + RMSNorm (see the module docstring)."""

    def __init__(
        self,
        world: int,
        hidden: int,
        threads: int,
        cluster: int,
        cpp_one_shot: bool,
        use_pdl: bool,
    ):
        if world not in WORLD_SIZES:
            raise ValueError(f"unsupported world size {world}")
        if threads * cluster * ELTS != hidden or threads % 16:
            raise ValueError(f"no launch shape {threads} x {cluster} for hidden={hidden}")
        self.W = world
        self.H = hidden
        self.threads = threads
        self.cluster = cluster
        self.warps = (threads + 31) // 32
        self.half_last = threads % 32 != 0
        self.cpp_one_shot = cpp_one_shot
        self.use_pdl = use_pdl
        self.rank_chunk = (
            min(world, ONE_SHOT_RANK_CHUNK) if cpp_one_shot else min(world, TWO_SHOT_RANK_CHUNK)
        )
        import struct

        self.dim_bits = struct.unpack("<I", struct.pack("<f", float(hidden)))[0]

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        res_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        res_out_addr: Int64,
        uc_addr: Int64,
        peer_ptrs: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_sum: cute.struct.MemRange[cutlass.Float32, 32]
            block_sum: cute.struct.MemRange[cutlass.Float32, 1]
            cta_sum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 8], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            x_addr,
            res_addr,
            gamma_addr,
            out_addr,
            res_out_addr,
            uc_addr,
            peer_ptrs,
            mc_addr,
            flags_addr,
            rank,
            m,
            eps,
        ).launch(
            grid=(m, self.cluster, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
            preferred_smem_carveout=100,
        )

    @cute.jit
    def _block_sum(self, v: Float32, warp: Int32, lane: Int32, warp_sum: Int32, block_sum: Int32):
        """``blockReduceSum<float, true>`` of the C++ kernels over this CTA's threads."""
        if cutlass.const_expr(self.half_last):
            if warp == self.warps - 1:
                v = _fadd(v, Float32(0.0))
                for off in (8, 4, 2, 1):
                    v = _fadd(v, _shfl_bfly(v, off, 0xFFFF))
            else:
                for off in (16, 8, 4, 2, 1):
                    v = _fadd(v, _shfl_bfly(v, off, 0xFFFFFFFF))
        else:
            for off in (16, 8, 4, 2, 1):
                v = _fadd(v, _shfl_bfly(v, off, 0xFFFFFFFF))
        if lane == 0:
            _sts_f32(warp_sum + warp * 4, v)
        _bar_sync()
        if warp == 0:
            b = Float32(0.0)
            if lane < self.warps:
                b = _lds_f32(warp_sum + lane * 4)
            for off in (16, 8, 4, 2, 1):
                b = _fadd(b, _shfl_bfly(b, off, 0xFFFFFFFF))
            if lane == 0:
                _sts_f32(block_sum, b)
        _bar_sync()
        return _lds_f32(block_sum)

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        res_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        res_out_addr: Int64,
        uc_addr: Int64,
        peer_ptrs: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
    ):
        W, H = self.W, self.H
        tidx, _, _ = cute.arch.thread_idx()
        row, cy, _ = cute.arch.block_idx()
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        warp_sum = storage.warp_sum.data_ptr().toint()
        block_sum = storage.block_sum.data_ptr().toint()
        cta_sum = storage.cta_sum.data_ptr().toint()

        col = (crank * self.threads + tidx) * ELTS
        row_bytes = Int64(H * 2)
        owner = row % W
        # Written once at setup (the peer table) or a weight (gamma): read before the wait.
        owner_base = ld_nc_u64(peer_ptrs + Int64(owner) * 8)
        gw = ld_v4(gamma_addr + Int64(col) * 2)

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()

        # LamportFlags: [current, dirty, bytes per buffer, dirty stages], bytes to clear x4.
        f0, f1, f2, f3 = ld_v4(flags_addr)
        c0, c1, c2, c3 = ld_v4(flags_addr + Int64(16))
        xw = ld_v4(x_addr + Int64(row * H + col) * 2)
        rw = ld_v4(res_addr + Int64(row * H + col) * 2)
        cur = Int64(f0)
        dirty = Int64(f1)
        buf_bytes = Int64(f2) & Int64(0xFFFFFFFF)
        stage_bytes = buf_bytes // Int64(2)

        # Scatter: this rank's values of the row into slot (row // W, rank) of the owner's scatter
        # stage (the owner's unicast mapping).
        slot = Int64((row // W) * W + rank) * row_bytes + Int64(col) * 2
        sw = [sanitize_bf16x2(w) for w in xw]
        st_v4(owner_base + (cur * Int64(2)) * stage_bytes + slot, sw[0], sw[1], sw[2], sw[3])
        # One arrival per CTA once all its threads read the flags: CTA (0, 0) rotates them only
        # after every CTA of the grid has.
        counter = flags_addr + Int64(32)
        _bar_sync()
        if tidx == 0:
            red_release_add(counter, Int32(1))

        # Owner: sum the W slots of the row in the reproduced kernel's order, round, multicast.
        if owner == rank:
            base = (
                uc_addr
                + (cur * Int64(2)) * stage_bytes
                + Int64((row // W) * W) * row_bytes
                + Int64(col) * 2
            )
            acc = [Float32(0.0)] * ELTS
            for rb in cutlass.range_constexpr(0, W, self.rank_chunk):
                chunk = [Float32(0.0)] * ELTS
                ready = Int32(0)
                while ready == 0:
                    ready = Int32(1)
                    for i in cutlass.range_constexpr(ELTS):
                        chunk[i] = Float32(0.0)
                    for rr in cutlass.range_constexpr(self.rank_chunk):
                        w = ld_volatile_v4(base + Int64(rb + rr) * row_bytes)
                        ready = ready & clean4(w[0], w[1], w[2], w[3])
                        for j in cutlass.range_constexpr(4):
                            lo, hi = _bf16x2_f32(w[j])
                            chunk[2 * j] = _fadd(chunk[2 * j], lo)
                            chunk[2 * j + 1] = _fadd(chunk[2 * j + 1], hi)
                for i in cutlass.range_constexpr(ELTS):
                    acc[i] = _fadd(acc[i], chunk[i])
            red = [sanitize_bf16x2(_pack_bf16x2(acc[2 * j], acc[2 * j + 1])) for j in range(4)]
            bcast = mc_addr + (cur * Int64(2) + Int64(1)) * stage_bytes + Int64(row) * row_bytes
            multimem_st_v4(bcast + Int64(col) * 2, red[0], red[1], red[2], red[3])

        # Re-arm this rank's dirty buffer (every stage the previous call used) while rows travel.
        n_threads = m * self.cluster * self.threads
        gtid = (row * self.cluster + cy) * self.threads + tidx
        clear = (c0, c1, c2, c3)
        for s in cutlass.range_constexpr(4):
            if s < f3:
                d_stage = buf_bytes // Int64(f3)
                cbase = uc_addr + (dirty * Int64(f3) + Int64(s)) * d_stage
                n_vec = (clear[s] + 15) >> 4
                for p in cutlass.range(gtid, n_vec, n_threads, unroll=1):
                    st_sentinel_v4(cbase + Int64(p) * Int64(16))

        # The reduced row from the broadcast stage.
        src = uc_addr + (cur * Int64(2) + Int64(1)) * stage_bytes + Int64(row) * row_bytes
        src = src + Int64(col) * 2
        xr = ld_volatile_v4(src)
        while clean4(xr[0], xr[1], xr[2], xr[3]) == 0:
            xr = ld_volatile_v4(src)

        # Residual add (the new residual) and this thread's BF16 squares in element order.
        s4 = [_add_bf16x2(xr[j], rw[j]) for j in range(4)]
        st_v4(res_out_addr + Int64(row * H + col) * 2, s4[0], s4[1], s4[2], s4[3])
        thread_sum = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            lo, hi = _bf16x2_f32(_sq_bf16x2(s4[j]))
            thread_sum = _fadd(thread_sum, lo)
            thread_sum = _fadd(thread_sum, hi)
        bsum = self._block_sum(thread_sum, warp, lane, warp_sum, block_sum)
        # Row sum: CTA c's sum into every peer's slot c, then the slots in CTA order from 0.
        if tidx < self.cluster:
            _st_cluster_f32(cta_sum + crank * 4, tidx, bsum)
        _cluster_sync()
        total = Float32(0.0)
        for c in cutlass.range_constexpr(self.cluster):
            total = _fadd(total, _lds_f32(cta_sum + c * 4))
        scale = _rms_scale(total, self.dim_bits, eps)

        ow = []
        for j in cutlass.range_constexpr(4):
            x_lo, x_hi = _bf16x2_f32(s4[j])
            g_lo, g_hi = _bf16x2_f32(gw[j])
            if cutlass.const_expr(self.cpp_one_shot):
                ow.append(
                    _pack_bf16x2(_fmul(_fmul(x_lo, scale), g_lo), _fmul(_fmul(x_hi, scale), g_hi))
                )
            else:
                ow.append(
                    _pack_bf16x2(_fmul(_fmul(g_lo, x_lo), scale), _fmul(_fmul(g_hi, x_hi), scale))
                )
        st_v4(out_addr + Int64(row * H + col) * 2, ow[0], ow[1], ow[2], ow[3])

        # CTA (0, 0): rotate the state once every warp of the grid has read it (as the C++
        # two-shot call leaves it: two stages, scatter and broadcast bytes to clear).
        if crank == 0:
            if tidx == 0:
                if row == 0:
                    arrived = ld_acquire_u32(counter)
                    while arrived < m * self.cluster:
                        arrived = ld_acquire_u32(counter)
                    nxt = f0 + 1
                    if nxt == NUM_LAMPORT_BUFFERS:
                        nxt = Int32(0)
                    st_v4(flags_addr, nxt, f0, f2, Int32(2))
                    st_v4(
                        flags_addr + Int64(16),
                        ((m + (W - 1)) // W) * (W * H * 2),
                        m * (H * 2),
                        Int32(0),
                        Int32(0),
                    )
                    st_u32(counter, Int32(0))


class K3DraftArFinishNormKernel(K3DraftArNormKernel):
    """The rest of a two-shot all-reduce whose input the producer GEMM pushed
    (``low_m_bf16_splitk.run_splitk_dense_ar_push``), then residual add + RMSNorm: the result is
    bitwise that of the producer followed by the C++ ``RESIDUAL_RMS_NORM`` fusion.

    Same CTAs, arithmetic and Lamport state as :class:`K3DraftArNormKernel`, without the scatter:

    * The CTAs do not wait for the producer. They release their dependents, read the rotation
      state, the residual and ``gamma`` and start polling as soon as they are resident. That is
      safe because the producer releases this kernel only after its own ``griddepcontrol.wait``
      (see ``low_m_bf16_splitk``): every earlier grid, the previous all-reduce on the workspace
      and the writer of the residual included, has then completed.
    * The owner polls its ``W`` scatter slots, re-loading only the slots that have not arrived,
      sums them in the reproduced kernel's order and multicasts the row; every rank polls the
      broadcast row and runs the norm.
    * Thread 0 of CTA (0, 0) alone waits for the producer (which reads the state after its own
      wait) and then for every CTA's arrival before it rotates the state; its wait also makes
      this kernel's completion imply the producer's.
    """

    @cute.jit
    def __call__(
        self,
        res_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        res_out_addr: Int64,
        uc_addr: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_sum: cute.struct.MemRange[cutlass.Float32, 32]
            block_sum: cute.struct.MemRange[cutlass.Float32, 1]
            cta_sum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 8], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            res_addr,
            gamma_addr,
            out_addr,
            res_out_addr,
            uc_addr,
            mc_addr,
            flags_addr,
            rank,
            m,
            eps,
        ).launch(
            grid=(m, self.cluster, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
            preferred_smem_carveout=100,
        )

    @cute.kernel
    def kernel(
        self,
        res_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        res_out_addr: Int64,
        uc_addr: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
    ):
        W, H = self.W, self.H
        tidx, _, _ = cute.arch.thread_idx()
        row, cy, _ = cute.arch.block_idx()
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        warp_sum = storage.warp_sum.data_ptr().toint()
        block_sum = storage.block_sum.data_ptr().toint()
        cta_sum = storage.cta_sum.data_ptr().toint()

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        col = (crank * self.threads + tidx) * ELTS
        row_bytes = Int64(H * 2)
        # LamportFlags: [current, dirty, bytes per buffer, dirty stages], bytes to clear x4.
        f0, f1, f2, f3 = ld_volatile_v4(flags_addr)
        c0, c1, c2, c3 = ld_volatile_v4(flags_addr + Int64(16))
        gw = ld_v4(gamma_addr + Int64(col) * 2)
        rw = ld_v4(res_addr + Int64(row * H + col) * 2)
        cur = Int64(f0)
        dirty = Int64(f1)
        buf_bytes = Int64(f2) & Int64(0xFFFFFFFF)
        stage_bytes = buf_bytes // Int64(2)
        # One arrival per CTA once all its threads read the flags.
        counter = flags_addr + Int64(32)
        _bar_sync()
        if tidx == 0:
            red_release_add(counter, Int32(1))

        # Owner: poll the W slots of the row (only the missing ones again), then sum them in the
        # reproduced kernel's order, round and multicast.
        if row % W == rank:
            base = (
                uc_addr
                + (cur * Int64(2)) * stage_bytes
                + Int64((row // W) * W) * row_bytes
                + Int64(col) * 2
            )
            words = cute.make_rmem_tensor((W * 4,), Int32)
            pending = Int32((1 << W) - 1) if W < 32 else Int32(-1)
            while pending != 0:
                for r in cutlass.range_constexpr(W):
                    if ((pending >> r) & 1) != 0:
                        w = ld_volatile_v4(base + Int64(r) * row_bytes)
                        for e in cutlass.range_constexpr(4):
                            words[r * 4 + e] = w[e]
                        if clean4(w[0], w[1], w[2], w[3]) != 0:
                            pending = pending & Int32(~(1 << r))
            acc = [Float32(0.0)] * ELTS
            for rb in cutlass.range_constexpr(0, W, self.rank_chunk):
                chunk = [Float32(0.0)] * ELTS
                for rr in cutlass.range_constexpr(self.rank_chunk):
                    for j in cutlass.range_constexpr(4):
                        lo, hi = _bf16x2_f32(words[(rb + rr) * 4 + j])
                        chunk[2 * j] = _fadd(chunk[2 * j], lo)
                        chunk[2 * j + 1] = _fadd(chunk[2 * j + 1], hi)
                for i in cutlass.range_constexpr(ELTS):
                    acc[i] = _fadd(acc[i], chunk[i])
            red = [sanitize_bf16x2(_pack_bf16x2(acc[2 * j], acc[2 * j + 1])) for j in range(4)]
            bcast = mc_addr + (cur * Int64(2) + Int64(1)) * stage_bytes + Int64(row) * row_bytes
            multimem_st_v4(bcast + Int64(col) * 2, red[0], red[1], red[2], red[3])

        # Re-arm this rank's dirty buffer (every stage the previous call used) while rows travel.
        n_threads = m * self.cluster * self.threads
        gtid = (row * self.cluster + cy) * self.threads + tidx
        clear = (c0, c1, c2, c3)
        for s in cutlass.range_constexpr(4):
            if s < f3:
                d_stage = buf_bytes // Int64(f3)
                cbase = uc_addr + (dirty * Int64(f3) + Int64(s)) * d_stage
                n_vec = (clear[s] + 15) >> 4
                for p in cutlass.range(gtid, n_vec, n_threads, unroll=1):
                    st_sentinel_v4(cbase + Int64(p) * Int64(16))

        # The reduced row from the broadcast stage. Rows this rank does not own cannot arrive
        # before the producer completed: their CTAs wait for it instead of spinning while it runs.
        if cutlass.const_expr(self.use_pdl):
            if row % W != rank:
                griddepcontrol_wait()
        src = uc_addr + (cur * Int64(2) + Int64(1)) * stage_bytes + Int64(row) * row_bytes
        src = src + Int64(col) * 2
        xr = ld_volatile_v4(src)
        while clean4(xr[0], xr[1], xr[2], xr[3]) == 0:
            xr = ld_volatile_v4(src)

        s4 = [_add_bf16x2(xr[j], rw[j]) for j in range(4)]
        st_v4(res_out_addr + Int64(row * H + col) * 2, s4[0], s4[1], s4[2], s4[3])
        thread_sum = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            lo, hi = _bf16x2_f32(_sq_bf16x2(s4[j]))
            thread_sum = _fadd(thread_sum, lo)
            thread_sum = _fadd(thread_sum, hi)
        bsum = self._block_sum(thread_sum, warp, lane, warp_sum, block_sum)
        if tidx < self.cluster:
            _st_cluster_f32(cta_sum + crank * 4, tidx, bsum)
        _cluster_sync()
        total = Float32(0.0)
        for c in cutlass.range_constexpr(self.cluster):
            total = _fadd(total, _lds_f32(cta_sum + c * 4))
        scale = _rms_scale(total, self.dim_bits, eps)

        ow = []
        for j in cutlass.range_constexpr(4):
            x_lo, x_hi = _bf16x2_f32(s4[j])
            g_lo, g_hi = _bf16x2_f32(gw[j])
            if cutlass.const_expr(self.cpp_one_shot):
                ow.append(
                    _pack_bf16x2(_fmul(_fmul(x_lo, scale), g_lo), _fmul(_fmul(x_hi, scale), g_hi))
                )
            else:
                ow.append(
                    _pack_bf16x2(_fmul(_fmul(g_lo, x_lo), scale), _fmul(_fmul(g_hi, x_hi), scale))
                )
        st_v4(out_addr + Int64(row * H + col) * 2, ow[0], ow[1], ow[2], ow[3])

        # CTA (0, 0): once the producer completed and every CTA read the state, rotate it as the
        # C++ two-shot call leaves it.
        if crank == 0:
            if tidx == 0:
                if row == 0:
                    if cutlass.const_expr(self.use_pdl):
                        griddepcontrol_wait()
                    arrived = ld_acquire_u32(counter)
                    while arrived < m * self.cluster:
                        arrived = ld_acquire_u32(counter)
                    nxt = f0 + 1
                    if nxt == NUM_LAMPORT_BUFFERS:
                        nxt = Int32(0)
                    st_v4(flags_addr, nxt, f0, f2, Int32(2))
                    st_v4(
                        flags_addr + Int64(16),
                        ((m + (W - 1)) // W) * (W * H * 2),
                        m * (H * 2),
                        Int32(0),
                        Int32(0),
                    )
                    st_u32(counter, Int32(0))


@functools.cache
def _compiled_finish_norm(
    world: int, hidden: int, threads: int, cluster: int, cpp_one_shot: bool, use_pdl: bool
):
    kernel = K3DraftArFinishNormKernel(world, hidden, threads, cluster, cpp_one_shot, use_pdl)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        Int32(0),
        Int32(1),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _compiled(
    world: int, hidden: int, threads: int, cluster: int, cpp_one_shot: bool, use_pdl: bool
):
    kernel = K3DraftArNormKernel(world, hidden, threads, cluster, cpp_one_shot, use_pdl)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        Int32(0),
        Int32(1),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3))


def required_bytes(num_tokens: int, hidden: int, world: int) -> int:
    """Bytes of one Lamport buffer the two-shot transport needs."""
    return 2 * (-(-num_tokens // world)) * world * hidden * 2


def can_run(x: torch.Tensor, world: int) -> bool:
    """Whether :func:`run_k3_draft_ar_norm` serves an all-reduce of ``x`` (non-raising)."""
    return (
        x.is_cuda
        and _is_supported_device(x.device.index)
        and world in WORLD_SIZES
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and grid_config(x.shape[0], x.shape[1], _sm_count(x.device.index)) is not None
    )


def run_k3_draft_ar_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    ws: LamportWorkspace,
    cpp_one_shot: bool,
    use_pdl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(RMSNorm(gamma, eps)(all_reduce(x) + residual), all_reduce(x) + residual)`` over the MNNVL
    Lamport workspace ``ws``, bitwise the C++ ``RESIDUAL_RMS_NORM`` fusion (``cpp_one_shot``: the
    one-shot kernel's result, else the two-shot kernel's). ``x`` and ``residual`` are contiguous
    BF16 ``[m, H]``; every rank of the group calls it with the same shapes."""
    m, hidden = x.shape
    cfg = grid_config(m, hidden, _sm_count(x.device.index))
    assert cfg is not None
    if required_bytes(m, hidden, ws.world) > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} x {hidden}")
    residual = residual.contiguous()
    out = torch.empty_like(x)
    res_out = torch.empty_like(x)
    _compiled(ws.world, hidden, cfg[0], cfg[1], bool(cpp_one_shot), bool(use_pdl))(
        Int64(x.data_ptr()),
        Int64(residual.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(out.data_ptr()),
        Int64(res_out.data_ptr()),
        Int64(ws.uc_local),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
        Int32(m),
        Float32(eps),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return out, res_out


def run_k3_draft_ar_finish_norm(
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    ws: LamportWorkspace,
    cpp_one_shot: bool,
    use_pdl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(RMSNorm(gamma, eps)(all_reduce(x) + residual), all_reduce(x) + residual)`` where every
    rank's producer pushed its ``x`` (``[m, H]``, ``m, H = residual.shape``) into the next two-shot
    call on ``ws`` (``low_m_bf16_splitk.run_splitk_dense_ar_push``): bitwise the C++
    ``RESIDUAL_RMS_NORM`` fusion of ``x`` (``cpp_one_shot``: the one-shot kernel's result)."""
    m, hidden = residual.shape
    cfg = grid_config(m, hidden, _sm_count(residual.device.index))
    assert cfg is not None
    if required_bytes(m, hidden, ws.world) > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} x {hidden}")
    residual = residual.contiguous()
    out = torch.empty_like(residual)
    res_out = torch.empty_like(residual)
    _compiled_finish_norm(ws.world, hidden, cfg[0], cfg[1], bool(cpp_one_shot), bool(use_pdl))(
        Int64(residual.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(out.data_ptr()),
        Int64(res_out.data_ptr()),
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
        Int32(m),
        Float32(eps),
        cuda.CUstream(torch.cuda.current_stream(residual.device).cuda_stream),
    )
    return out, res_out


__all__ = [
    "K3DraftArFinishNormKernel",
    "K3DraftArNormKernel",
    "can_run",
    "grid_config",
    "required_bytes",
    "run_k3_draft_ar_finish_norm",
    "run_k3_draft_ar_norm",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark MLA drafter block attention in one clustered launch (SM100 / SM103).

One drafter layer's absorbed-MLA block attention for ``B`` requests of ``T <= 8`` block tokens and
``H`` heads (an even number) of this rank:

    q_lat[t, h]  = bf16(q_nope[t, h] @ k_b[h])                            (absorption, 128 -> 512)
    s[t, h, p]   = [q_lat[t, h], q_pe[t, h]] . pool[p] * softmax_scale     (p < kv_len[b])
    o[t, h]      = softmax_p(s[t, h]) @ pool[p, :512]
    out[t, h]    = bf16(bf16(o[t, h]) @ v_b[h].T)                          (V up-projection, 512 -> 128)

where ``pool`` is the paged latent cache (576-wide rows: 512 latent + 64 RoPE) and ``kv_len[b]``
covers the request's context and its block (DSpark's block attention is not causal). It replaces
the absorption BMM, the cute-dsl MLA decode (split-KV kernel and reduction) and the V
up-projection BMM, with a different summation order (draft tokens only).

Work split: a cluster of ``NS`` CTAs (8 or 16) per (request, pair of heads); its 16 MMA rows are
the two heads' ``T`` tokens (8 row slots each), 256 threads per CTA. With ``C = 512 / NS`` latent
and ``V = 128 / NS`` output columns per CTA, CTA ``s`` of the cluster

* computes the absorbed query's latent columns ``[C s, C s + C)`` (``mma.sync`` BF16, FP32
  accumulate), rounds them to BF16 and stores them into every cluster CTA's shared-memory query;
* runs online-softmax attention over the 32-token KV tiles ``s, s + NS, ...``: one
  ``cp.async.bulk`` copy per KV row (page from a shared-memory copy of the request's page table,
  a zero row past ``kv_len``), four stages on mbarriers; the scores are split over the warps
  along the 576 dims (four quarters) and the tile's two halves, the quarters summed in order;
  P is rounded to BF16 for the P @ V MMA;
* owns latent columns ``[C s, C s + C)`` of the cluster's output: it merges the cluster CTAs'
  partial outputs with their row maxima and sums, rounds the normalized output to BF16 and
  multiplies it by its ``C`` latent rows of ``v_b`` (a K-split of the up-projection);
* owns output columns ``[V s, V s + V)``: it sums the cluster's K-split partials in order, rounds
  to BF16 and stores them.

PDL: before ``griddepcontrol.wait`` the kernel reads the KV lengths, the page tables, ``q_nope``,
``k_b`` and ``v_b``, computes the absorbed query and copies the context rows (positions below
``kv_len - T``) of its first three tiles. All of these were written before the kernel's
predecessor (the block RoPE kernel, which writes the block rows of the pool and ``q_pe``) started,
provided every kernel from the q_b GEMM on triggers its dependents only after its own wait (the
block RoPE kernel triggers right after its wait; without PDL it triggers at completion): when this
kernel starts, its predecessor has passed its wait, so the q_b GEMM and everything before it
completed. ``q_pe`` and the block rows are read after the wait; the dependents are triggered right
after it (they read ``out`` after their own wait).

Two switches change how, not what, the kernel computes (bitwise the same output):

* ``TLLM_DSPARK_MLA_FUSED_ASYNC=1``: the absorbed-query exchange (``st.async``), the partial
  outputs and row stats (one bulk shared-to-shared copy per owner from a staging copy in the idle
  KV ring) and the up-projection partials (``st.async``) complete on mbarriers of the receiving
  CTA, each expecting exactly the bytes it receives, instead of cluster barriers (each of which
  costs a GPU-scope fence and waits for every CTA). The wait for the partial outputs implies every
  CTA left its attention loop, so the score buffer can take the up-projection partials, and a
  CTA's last wait implies every owner received its staged partial outputs.
* ``TLLM_DSPARK_MLA_FUSED_STAGED=1``: the CTA's k_b and v_b slices arrive by one bulk copy per
  head, issued first thing, from padded copies of the weights (:func:`staged_weights`: rows padded
  by 16 bytes so that ldmatrix reads them without bank conflicts, v_b re-laid per cluster CTA),
  and the q_nope rows by one bulk copy each, into shared memory that is idle before the loop;
  the MMA fragments come from ``ldmatrix`` instead of ~80 four-byte loads per thread, and the
  slices of the next layer's weights go to L2.
"""

from __future__ import annotations

import functools
import math
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Heads per cluster.
HG = 2
#: Row slots per head (block tokens <= 8).
SLOT = 8
#: MMA rows of a cluster.
ROWS = HG * SLOT
#: KV tokens per tile, KV stages (tiles in flight).
TILE = 32
STAGES = 4
THREADS = 256
NWARPS = THREADS // 32
D_NOPE = 128
D_LAT = 512
D_ROPE = 64
D_QK = D_LAT + D_ROPE
D_V = 128
#: Bytes of a query / KV row in shared memory (16 bytes of padding keep ldmatrix conflict free).
ROW_BYTES = D_QK * 2 + 16
#: Bytes of a KV row in global memory.
KV_ROW = D_QK * 2
#: Bytes of a P row, of a normalized-output row (BF16, padded).
P_BYTES = TILE * 2 + 16
#: FP32 pitch of a partial score row.
S_PITCH = TILE + 4
#: Page-table entries of a request cached in shared memory (positions past them read the table).
PAGE_CACHE = 256
_NEG_INF = float("-inf")


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
def _ld_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ld_b32_if(addr, valid, *, loc=None, ip=None):
    """4 bytes at ``addr`` if ``valid`` else 0 (no access when not valid)."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(valid, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\nmov.b32 $0, 0;\n@p ld.global.b32 $0, [$1];\n}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_b32(addr, *, loc=None, ip=None):
    """4 bytes of a weight (read-only for the kernel's lifetime)."""
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
def _st_global_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_16(smem, addr, nbytes, *, loc=None, ip=None):
    """16 bytes global -> shared through L2, the bytes past ``nbytes`` (0 or 16) zero-filled."""
    _asm(
        None,
        [_iv(smem, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip), _iv(nbytes, loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "", loc=loc, ip=ip)


@dsl_user_op
def _cp_async_wait_group(n: int, *, loc=None, ip=None):
    _asm(None, [], f"cp.async.wait_group {n};", "", loc=loc, ip=ip)


@dsl_user_op
def _bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    """Cluster-wide barrier; shared::cluster writes before it are visible after it."""
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _gdc_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _gdc_launch(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _mapa(smem, rank, *, loc=None, ip=None):
    """Shared::cluster address of this CTA's shared address ``smem`` in cluster CTA ``rank``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(smem, loc=loc, ip=ip), _iv(rank, loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_cluster_v2_f32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(v0, loc=loc, ip=ip), _fv(v1, loc=loc, ip=ip)],
        "st.shared::cluster.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        loc=loc,
        ip=ip,
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
def _sts_v2_f32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(v0, loc=loc, ip=ip), _fv(v1, loc=loc, ip=ip)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v2_b32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
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
def _lds_v2_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ldsm_x4(addr, *, loc=None, ip=None):
    """``ldmatrix.x4``: four 8x8 BF16 matrices (lane ``8 i + r`` gives row ``r`` of matrix ``i``)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ldsm_x4_trans(addr, *, loc=None, ip=None):
    """``ldmatrix.x4.trans``: four transposed 8x8 BF16 matrices."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``: D = A * B + C."""
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
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """BF16 (round to nearest even) of ``lo`` in the low half, of ``hi`` in the high half."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(hi, loc=loc, ip=ip), _fv(lo, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_fv(x, loc=loc, ip=ip)], "ex2.approx.ftz.f32 $0, $1;", "=f,f", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly_f32(value, offset: int, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _finite_or_zero(x, *, loc=None, ip=None):
    """``x`` unless it is -inf (then 0): the subtrahend of an exponent whose row saw no key."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.eq.f32 p, $1, 0fFF800000;\nselp.f32 $0, 0f00000000, $1, p;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _neg_inf_if(flag, x, *, loc=None, ip=None):
    """-inf where ``flag`` is nonzero, else ``x``."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(flag, loc=loc, ip=ip), _fv(x, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, 0fFF800000, $2, p;\n}",
            "=f,r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _safe_rcp(x, *, loc=None, ip=None):
    """1 / x, or 0 where x is 0 (a row that saw no key)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .f32 r;\nsetp.eq.f32 p, $1, 0f00000000;\nrcp.rn.f32 r, $1;\n"
            "selp.f32 $0, 0f00000000, r, p;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pick_i(flag, a, b, *, loc=None, ip=None):
    """``a`` where ``flag`` is nonzero, else ``b``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(flag, loc=loc, ip=ip), _iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_init_fence(*, loc=None, ip=None):
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_arrive_expect_tx(addr, tx: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {tx};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_try_wait(addr, parity, *, loc=None, ip=None):
    """1 once the phase of ``parity`` of the mbarrier at ``addr`` completed."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(parity, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pick_l(flag, a, b, *, loc=None, ip=None):
    """``a`` where ``flag`` is nonzero, else ``b`` (64-bit)."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(flag, loc=loc, ip=ip), _lv(a, loc=loc, ip=ip), _lv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.b64 $0, $2, $3, p;\n}",
            "=l,r,l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly_xor1_f32(value, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            "shfl.sync.bfly.b32 $0, $1, 1, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pick_f(flag, a, b, *, loc=None, ip=None):
    """``a`` where ``flag`` is nonzero, else ``b`` (FP32)."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(flag, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}",
            "=f,r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bulk_g2s(dst, src, nbytes: int, mbar, *, loc=None, ip=None):
    """``nbytes`` from global ``src`` to this CTA's shared ``dst``, completing on ``mbar``."""
    _asm(
        None,
        [_iv(dst, loc=loc, ip=ip), _lv(src, loc=loc, ip=ip), _iv(mbar, loc=loc, ip=ip)],
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {nbytes}, [$2];",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s_n(dst, src, nbytes, mbar, *, loc=None, ip=None):
    """``nbytes`` (a multiple of 16, run-time) from global ``src`` to this CTA's shared ``dst``,
    completing on ``mbar``."""
    _asm(
        None,
        [
            _iv(dst, loc=loc, ip=ip),
            _lv(src, loc=loc, ip=ip),
            _iv(nbytes, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v4_b32(local_addr, local_mbar, peer, v0, v1, v2, v3, *, loc=None, ip=None):
    """``st.async`` of four words to ``local_addr`` of cluster CTA ``peer`` (16 bytes of the
    transaction count of its mbarrier at ``local_mbar``)."""
    _asm(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _iv(peer, loc=loc, ip=ip),
        ]
        + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $2;\n"
        "mapa.shared::cluster.u32 rb, $1, $2;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$3, $4, $5, $6}, [rb];\n}",
        "r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v2_f32(local_addr, local_mbar, peer, v0, v1, *, loc=None, ip=None):
    """``st.async`` of two FP32 values to ``local_addr`` of cluster CTA ``peer`` (8 bytes of the
    transaction count of its mbarrier at ``local_mbar``)."""
    _asm(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _iv(peer, loc=loc, ip=ip),
            _fv(v0, loc=loc, ip=ip),
            _fv(v1, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $2;\n"
        "mapa.shared::cluster.u32 rb, $1, $2;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [ra], {$3, $4}, [rb];\n}",
        "r,r,r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_s2c(local_dst, src, nbytes: int, local_mbar, peer, *, loc=None, ip=None):
    """``nbytes`` of this CTA's shared ``src`` to ``local_dst`` of cluster CTA ``peer`` (a bulk
    copy), completing on its mbarrier at ``local_mbar``."""
    _asm(
        None,
        [
            _iv(local_dst, loc=loc, ip=ip),
            _iv(src, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _iv(peer, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $3;\n"
        "mapa.shared::cluster.u32 rb, $2, $3;\n"
        f"cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [ra], [$1], {nbytes}, [rb];\n}}",
        "r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prefetch_l2_bulk(addr, nbytes: int, *, loc=None, ip=None):
    """``cp.async.bulk.prefetch.L2`` of ``nbytes`` (a multiple of 16) at ``addr``: a hint."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        f"cp.async.bulk.prefetch.L2.global [$0], {nbytes};",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def _lds_v4_b32(addr, *, loc=None, ip=None):
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
def _st_cluster_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared::cluster.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_cluster_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared::cluster.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


class DSparkMlaAttendKernel:
    """The block attention of one DSpark MLA drafter layer (see the module docstring)."""

    def __init__(
        self,
        num_heads: int,
        block_size: int,
        page_size: int,
        use_pdl: bool,
        cluster: int = 8,
        staged: bool = False,
        async_x: bool = False,
    ):
        if cluster not in (8, 16):
            raise ValueError(f"cluster size {cluster} (8 or 16)")
        if num_heads % HG:
            raise ValueError(f"{num_heads} heads: needs a multiple of {HG}")
        if not 1 <= block_size <= SLOT:
            raise ValueError(f"block size {block_size} (1..{SLOT})")
        if page_size < 1:
            raise ValueError(f"page size {page_size}")
        self.H = num_heads
        self.BLK = block_size
        self.page_size = page_size
        self.use_pdl = use_pdl
        # CTAs per cluster, latent columns per CTA (merge and up-projection K-split), output
        # columns per CTA, bytes of a normalized-output row (BF16, padded).
        self.ns = cluster
        self.lat = D_LAT // cluster
        self.vpc = D_V // cluster
        self.on_bytes = self.lat * 2 + 16
        # Staged operands: q_nope, this CTA's k_b and v_b slices arrive by bulk copies in shared
        # memory that is idle before the attention loop (the score buffer, KV stage STAGES - 1,
        # the partial-output buffer), and the MMA fragments come from ldmatrix.
        self.staged = staged
        # Row pitches of the staged operands in shared memory (and of the padded weight copies,
        # see :func:`staged_weights`): 16 bytes of padding keep the ldmatrix reads conflict free
        # (8 rows of a matrix on distinct banks); the cluster-8 v_b rows stay unpadded (their
        # padded copy would not fit the partial-output buffer).
        self.qn_pitch = D_NOPE * 2 + 16
        self.kb_pitch = KB_PITCH
        self.vb_pitch = vb_pitch(cluster)
        # Cluster exchanges (absorbed query, partial outputs, up-projection partials) by st.async
        # on per-exchange mbarriers of the receiving CTA instead of cluster barriers (each of which
        # costs a GPU-scope fence): a CTA waits for exactly the bytes it receives.
        self.async_x = async_x

    @cute.jit
    def __call__(
        self,
        q_addr: Int64,
        query_addr: Int64,
        pool_addr: Int64,
        page_stride: Int64,
        pt_addr: Int64,
        pt_stride: Int32,
        pt_len: Int32,
        kvlen_addr: Int64,
        kbt_addr: Int64,
        vb_addr: Int64,
        out_addr: Int64,
        zero_addr: Int64,
        scale_log2: Float32,
        num_reqs: Int32,
        kbs_addr: Int64,
        vbs_addr: Int64,
        pf_kb_addr: Int64,
        pf_vb_addr: Int64,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            q: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, ROWS * ROW_BYTES], 16]
            kv: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, STAGES * TILE * ROW_BYTES], 128
            ]
            # Per K quarter, the partial scores (FP32, 16 x TILE); later the up-projection
            # partials of every cluster CTA (FP32, 16 x V each).
            s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4 * ROWS * S_PITCH], 16]
            # P (BF16, 16 x TILE); later the normalized output slice (BF16, 16 x C).
            p: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, ROWS * self.on_bytes], 16]
            # Row max, row sum, rescale factor.
            stat: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 3 * ROWS], 16]
            # This CTA's latent slice of every cluster CTA's partial output, and their row stats.
            red_o: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.ns * ROWS * self.lat], 16
            ]
            red_ml: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.ns * ROWS * 2], 16]
            # One mbarrier per KV stage.
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, STAGES + 4], 16]
            # The first PAGE_CACHE entries of the request's page table.
            pages: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, PAGE_CACHE], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            q_addr,
            query_addr,
            pool_addr,
            page_stride,
            pt_addr,
            pt_stride,
            pt_len,
            kvlen_addr,
            kbt_addr,
            vb_addr,
            out_addr,
            zero_addr,
            scale_log2,
            kbs_addr,
            vbs_addr,
            pf_kb_addr,
            pf_vb_addr,
        ).launch(
            grid=(self.ns, self.H // HG, num_reqs),
            block=(THREADS, 1, 1),
            cluster=(self.ns, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def _issue_tile(
        self,
        kvs: Int32,
        mbar: Int32,
        tile: Int32,
        lane: Int32,
        mode: int,
        kv_len: Int32,
        ctx_len: Int32,
        pt_row: Int64,
        spages: Int32,
        pool_addr: Int64,
        page_stride: Int64,
        zero_addr: Int64,
    ):
        """One bulk copy of KV row ``lane`` of tile ``tile`` (four rows per warp; rows past
        ``kv_len`` copy a zero row). mode 0: row 0 arms the stage's barrier, every row but the
        block rows is copied; 1: the block rows only; 2: arm and copy every row."""
        p = tile * TILE + lane
        valid = Int32(p < kv_len)
        pc = _pick_i(valid, p, Int32(0))
        blk = valid & Int32(p >= ctx_len)
        go = Int32(1)
        if cutlass.const_expr(mode == 0):
            go = Int32(1) - blk
        elif cutlass.const_expr(mode == 1):
            go = blk
        if cutlass.const_expr(mode != 1):
            if lane == 0:
                _mbar_arrive_expect_tx(mbar, TILE * KV_ROW)
        if go != 0:
            pidx = pc // self.page_size
            page = Int32(0)
            if pidx < PAGE_CACHE:
                page = _lds_b32(spages + pidx * 4)
            else:
                page = _ld_b32(pt_row + Int64(pidx) * 4)
            src = _pick_l(
                valid,
                pool_addr + (Int64(page) * page_stride + Int64((pc % self.page_size) * D_QK)) * 2,
                zero_addr,
            )
            _bulk_g2s(kvs + lane * ROW_BYTES, src, KV_ROW, mbar)

    @cute.kernel
    def kernel(
        self,
        q_addr: Int64,
        query_addr: Int64,
        pool_addr: Int64,
        page_stride: Int64,
        pt_addr: Int64,
        pt_stride: Int32,
        pt_len: Int32,
        kvlen_addr: Int64,
        kbt_addr: Int64,
        vb_addr: Int64,
        out_addr: Int64,
        zero_addr: Int64,
        scale_log2: Float32,
        kbs_addr: Int64,
        vbs_addr: Int64,
        pf_kb_addr: Int64,
        pf_vb_addr: Int64,
    ):
        H = self.H
        BLK = self.BLK
        tidx, _, _ = cute.arch.thread_idx()
        _, grp, b = cute.arch.block_idx()
        s = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane >> 2
        t = lane & 3
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        sq = storage.q.data_ptr().toint()
        skv = storage.kv.data_ptr().toint()
        ss = storage.s.data_ptr().toint()
        sp = storage.p.data_ptr().toint()
        sstat = storage.stat.data_ptr().toint()
        sred_o = storage.red_o.data_ptr().toint()
        sred_ml = storage.red_ml.data_ptr().toint()
        smbar = storage.mbar.data_ptr().toint()
        spages = storage.pages.data_ptr().toint()
        h0 = grp * HG
        neg_inf = Float32(_NEG_INF)

        # Barriers: KV stages, then the staged operands, the absorbed-query, partial-output and
        # up-projection exchanges.
        stg = smbar + STAGES * 8
        q_bar = smbar + (STAGES + 1) * 8
        o_bar = smbar + (STAGES + 2) * 8
        v_bar = smbar + (STAGES + 3) * 8
        # Staged operands (idle buffers until the attention loop, which starts once the query
        # exchange completed): q_nope rows (zero rows for padding slots) in the score buffer, this
        # CTA's k_b rows of both heads in KV stage STAGES - 1, its v_b rows in the partial-output
        # buffer. One barrier counts all the bytes.
        qn_s = ss
        kb_s = skv + (STAGES - 1) * (TILE * ROW_BYTES)
        vb_s = sred_o
        if tidx == 0:
            for st in cutlass.range_constexpr(STAGES + (1 if self.staged else 0)):
                _mbar_init(smbar + st * 8, 1)
            if cutlass.const_expr(self.async_x):
                _mbar_init(q_bar, 1)
                _mbar_init(o_bar, 1)
                _mbar_init(v_bar, 1)
            _mbar_init_fence()
            if cutlass.const_expr(self.async_x):
                # The bytes each exchange delivers here: the other CTAs' query slices; every
                # CTA's partial-output slice and row stats; every CTA's up-projection partials.
                _mbar_arrive_expect_tx(q_bar, (self.ns - 1) * ROWS * self.lat * 2)
                _mbar_arrive_expect_tx(o_bar, self.ns * (ROWS * self.lat * 4 + ROWS * 8))
                _mbar_arrive_expect_tx(v_bar, self.ns * ROWS * self.vpc * 4)
            if cutlass.const_expr(self.staged):
                # This CTA's k_b and v_b slices, one bulk copy per head from the padded copies
                # (``kbs_addr`` [H, 512, KB_PITCH / 2], ``vbs_addr`` [H, NS, 128, vb_pitch / 2]:
                # a slice is contiguous with the shared-memory pitch), issued first thing.
                kb_bytes = self.lat * self.kb_pitch
                vb_bytes = D_V * self.vb_pitch
                _mbar_arrive_expect_tx(stg, ROWS * D_NOPE * 2 + HG * (kb_bytes + vb_bytes))
                for hh in cutlass.range_constexpr(HG):
                    kb_off = Int64((h0 + hh) * D_LAT + s * self.lat) * self.kb_pitch
                    vb_off = Int64((h0 + hh) * self.ns + s) * vb_bytes
                    _bulk_g2s(kb_s + hh * kb_bytes, kbs_addr + kb_off, kb_bytes, stg)
                    _bulk_g2s(vb_s + hh * vb_bytes, vbs_addr + vb_off, vb_bytes, stg)
                    # The next layer's slices of the same weights into L2 (hints; 0: none).
                    if pf_kb_addr != Int64(0):
                        _prefetch_l2_bulk(pf_kb_addr + kb_off, kb_bytes)
                        _prefetch_l2_bulk(pf_vb_addr + vb_off, vb_bytes)

        # Before the wait (data older than the predecessor): the KV length, the page table, the
        # context rows of the first STAGES - 1 tiles, the absorption (q_nope from the q_b GEMM,
        # k_b a weight) and the up-projection's v_b operands. All loads are issued up front.
        # Absorption: warp w computes latent column tile w of this CTA's C columns; A rows g
        # (head h0) and g + 8 (head h0 + 1). Up-projection: warp w, output column tiles 2w,
        # 2w + 1, this CTA's C latent rows of v_b.
        kv_len = _ld_b32(kvlen_addr + Int64(b) * 4)
        pt_row = pt_addr + Int64(b) * Int64(pt_stride) * 4
        pg_ok = Int32(tidx < pt_len)
        pg = _ld_b32_if(pt_row + Int64(tidx) * 4, pg_ok)
        vb_frag = []
        a_frag = []
        b_frag = []
        # With C / 8 < 8 column tiles, the other warps repeat the first ones (without storing).
        wa = warp & (self.lat // 8 - 1)
        if cutlass.const_expr(not self.staged):
            for kk in cutlass.range_constexpr(self.lat // 16):
                for j in cutlass.range_constexpr(2):
                    v = (2 * warp + j) * 8 + g
                    for hh in cutlass.range_constexpr(HG):
                        vbp = (
                            vb_addr
                            + (
                                Int64((h0 + hh) * D_V + v) * D_LAT
                                + Int64(s * self.lat + kk * 16 + 2 * t)
                            )
                            * 2
                        )
                        vb_frag.append(_ld_nc_b32(vbp))
                        vb_frag.append(_ld_nc_b32(vbp + 16))
            tok_ok = Int32(g < BLK)
            m_row = b * BLK + _pick_i(tok_ok, g, Int32(0))
            qa0 = q_addr + Int64((m_row * H + h0) * (D_NOPE + D_ROPE)) * 2
            qa1 = qa0 + (D_NOPE + D_ROPE) * 2
            n_abs = s * self.lat + wa * 8 + g
            for ks in cutlass.range_constexpr(D_NOPE // 16):
                k0 = ks * 16
                a_frag.append(_ld_b32_if(qa0 + Int64((k0 + 2 * t) * 2), tok_ok))
                a_frag.append(_ld_b32_if(qa1 + Int64((k0 + 2 * t) * 2), tok_ok))
                a_frag.append(_ld_b32_if(qa0 + Int64((k0 + 2 * t + 8) * 2), tok_ok))
                a_frag.append(_ld_b32_if(qa1 + Int64((k0 + 2 * t + 8) * 2), tok_ok))
                for hh in cutlass.range_constexpr(HG):
                    kb = (
                        kbt_addr
                        + (Int64((h0 + hh) * D_LAT + n_abs) * D_NOPE + Int64(k0 + 2 * t)) * 2
                    )
                    b_frag.append(_ld_nc_b32(kb))
                    b_frag.append(_ld_nc_b32(kb + 16))

        if tidx < ROWS:
            _sts_f32(sstat + tidx * 4, neg_inf)
            _sts_f32(sstat + (ROWS + tidx) * 4, Float32(0.0))
        if cutlass.const_expr(self.staged):
            # The q_nope rows (zero rows for padding slots), one bulk copy per row from lane 0 of
            # each warp (rows w, w + 8; warp-uniform operands), once the barrier init is visible.
            _bar_sync()
            if lane == 0:
                for r in cutlass.range_constexpr(ROWS // NWARPS):
                    qr_ = warp + r * NWARPS
                    q_ok_ = Int32((qr_ & 7) < BLK)
                    q_src_ = (
                        q_addr
                        + Int64(((b * BLK + (qr_ & 7)) * H + h0 + (qr_ >> 3)) * (D_NOPE + D_ROPE))
                        * 2
                    )
                    _bulk_g2s(
                        qn_s + qr_ * self.qn_pitch,
                        _pick_l(q_ok_, q_src_, zero_addr),
                        D_NOPE * 2,
                        stg,
                    )
        _sts_b32(spages + tidx * 4, pg)
        _bar_sync()
        # Cluster residency before the first shared::cluster store (waited on below).
        _cluster_arrive_relaxed()

        ctx_len = kv_len - BLK
        n_tiles = (kv_len + (TILE - 1)) // TILE
        my_n = _pick_i(Int32(n_tiles > s), (n_tiles - s + (self.ns - 1)) // self.ns, Int32(0))
        if lane < TILE // NWARPS:
            for k in cutlass.range_constexpr(STAGES - 1):
                if my_n > k:
                    self._issue_tile(
                        skv + k * (TILE * ROW_BYTES),
                        smbar + k * 8,
                        s + k * self.ns,
                        warp * (TILE // NWARPS) + lane,
                        0,
                        kv_len,
                        ctx_len,
                        pt_row,
                        spages,
                        pool_addr,
                        page_stride,
                        zero_addr,
                    )

        if cutlass.const_expr(self.staged):
            # The fragments the per-thread loads above would give, from the staged rows.
            while _mbar_try_wait(stg, 0) == 0:
                pass
            # (k0: the attention loop below carries this name; give it the type the per-thread
            # loads leave it with.)
            k0 = 0
            arow = (lane & 7) + ((lane >> 3) & 1) * 8
            acol = (lane >> 4) * 8
            brow = (lane >> 4) * self.lat + wa * 8 + (lane & 7)
            bcol = ((lane >> 3) & 1) * 8
            for ks in cutlass.range_constexpr(D_NOPE // 16):
                af = _ldsm_x4(qn_s + arow * self.qn_pitch + (ks * 16 + acol) * 2)
                bf = _ldsm_x4(kb_s + brow * self.kb_pitch + (ks * 16 + bcol) * 2)
                for e in cutlass.range_constexpr(4):
                    a_frag.append(af[e])
                for e in cutlass.range_constexpr(4):
                    b_frag.append(bf[e])
            for kk in cutlass.range_constexpr(self.lat // 16):
                for j in cutlass.range_constexpr(2):
                    vrow = (lane >> 4) * D_V + (2 * warp + j) * 8 + (lane & 7)
                    vf = _ldsm_x4(vb_s + vrow * self.vb_pitch + (kk * 16 + bcol) * 2)
                    for e in cutlass.range_constexpr(4):
                        vb_frag.append(vf[e])

        acc0 = [Float32(0.0)] * 4
        acc1 = [Float32(0.0)] * 4
        for ks in cutlass.range_constexpr(D_NOPE // 16):
            for hh in cutlass.range_constexpr(HG):
                acc = acc0 if hh == 0 else acc1
                d = _mma_bf16(
                    acc[0],
                    acc[1],
                    acc[2],
                    acc[3],
                    a_frag[4 * ks],
                    a_frag[4 * ks + 1],
                    a_frag[4 * ks + 2],
                    a_frag[4 * ks + 3],
                    b_frag[4 * ks + 2 * hh],
                    b_frag[4 * ks + 2 * hh + 1],
                )
                for e in cutlass.range_constexpr(4):
                    acc[e] = d[e]
        if warp < self.lat // 8:
            col = s * self.lat + warp * 8 + 2 * t
            _sts_b32(sq + g * ROW_BYTES + col * 2, _pack_bf16x2(acc0[0], acc0[1]))
            _sts_b32(sq + (g + 8) * ROW_BYTES + col * 2, _pack_bf16x2(acc1[2], acc1[3]))
        _bar_sync()
        _cluster_wait()
        # This CTA's latent slice (16 rows x C) to the other cluster CTAs, 16 bytes per store;
        # thread group tidx / chunks takes the destinations dd with dd % groups == group.
        chunks = ROWS * (self.lat // 8)
        groups = THREADS // chunks
        ch = tidx & (chunks - 1)
        crow = ch // (self.lat // 8)
        caddr = sq + crow * ROW_BYTES + (s * self.lat + (ch % (self.lat // 8)) * 8) * 2
        cv = _lds_v4_b32(caddr)
        for dd in cutlass.range_constexpr(1, self.ns):
            if (tidx // chunks) == (dd % groups):
                if cutlass.const_expr(self.async_x):
                    _st_async_v4_b32(caddr, q_bar, (s + dd) % self.ns, cv[0], cv[1], cv[2], cv[3])
                else:
                    _st_cluster_v4_b32(_mapa(caddr, (s + dd) % self.ns), cv[0], cv[1], cv[2], cv[3])

        if cutlass.const_expr(self.use_pdl):
            _gdc_wait()
            _gdc_launch()

        # After the wait: the rotated q_pe rows and the block rows of the first tiles.
        qr = (tidx >> 3) & (ROWS - 1)
        r_ok = Int32((qr & 7) < BLK)
        qm = b * BLK + _pick_i(r_ok, qr & 7, Int32(0))
        if tidx < ROWS * 8:
            qsrc = (
                query_addr + (Int64((qm * H + h0 + (qr >> 3)) * D_QK + D_LAT + (tidx & 7) * 8)) * 2
            )
            _cp_async_16(sq + qr * ROW_BYTES + (D_LAT + (tidx & 7) * 8) * 2, qsrc, r_ok * 16)
        _cp_async_commit()
        if lane < TILE // NWARPS:
            for k in cutlass.range_constexpr(STAGES - 1):
                if my_n > k:
                    self._issue_tile(
                        skv + k * (TILE * ROW_BYTES),
                        smbar + k * 8,
                        s + k * self.ns,
                        warp * (TILE // NWARPS) + lane,
                        1,
                        kv_len,
                        ctx_len,
                        pt_row,
                        spages,
                        pool_addr,
                        page_stride,
                        zero_addr,
                    )
        _cp_async_wait_group(0)
        if cutlass.const_expr(self.async_x):
            _bar_sync()
            while _mbar_try_wait(q_bar, 0) == 0:
                pass
        else:
            _cluster_sync()

        # Online-softmax attention over this CTA's tiles. Warp w: score k-steps
        # [9 (w % 4), 9 (w % 4) + 9) of the 576 dims for tokens [16 (w / 4), 16 (w / 4) + 16);
        # output columns [64 w, 64 w + 64).
        kq = warp & 3
        th = warp >> 2
        o_acc = cute.make_rmem_tensor((32,), Float32)
        for e in cutlass.range_constexpr(32):
            o_acc[e] = Float32(0.0)
        lrow = (lane & 7) + ((lane >> 3) & 1) * 8
        lcol8 = (lane >> 4) * 8
        sr = tidx >> 4
        sc = tidx & 15
        for k in cutlass.range(my_n, unroll=1):
            tile = s + k * self.ns
            stage = k % STAGES
            kvs = skv + stage * (TILE * ROW_BYTES)
            if lane < TILE // NWARPS:
                if k + (STAGES - 1) < my_n:
                    nxt = (k + (STAGES - 1)) % STAGES
                    self._issue_tile(
                        skv + nxt * (TILE * ROW_BYTES),
                        smbar + nxt * 8,
                        tile + (STAGES - 1) * self.ns,
                        warp * (TILE // NWARPS) + lane,
                        2,
                        kv_len,
                        ctx_len,
                        pt_row,
                        spages,
                        pool_addr,
                        page_stride,
                        zero_addr,
                    )
            while _mbar_try_wait(smbar + stage * 8, (k // STAGES) & 1) == 0:
                pass
            sacc = [Float32(0.0)] * 8
            for kk in cutlass.range_constexpr(9):
                k0 = (kq * 9 + kk) * 16
                a = _ldsm_x4(sq + lrow * ROW_BYTES + (k0 + lcol8) * 2)
                tok = th * 16 + (lane & 7) + (lane >> 4) * 8
                dim = k0 + ((lane >> 3) & 1) * 8
                bb = _ldsm_x4(kvs + tok * ROW_BYTES + dim * 2)
                for half in cutlass.range_constexpr(2):
                    d = _mma_bf16(
                        sacc[4 * half],
                        sacc[4 * half + 1],
                        sacc[4 * half + 2],
                        sacc[4 * half + 3],
                        a[0],
                        a[1],
                        a[2],
                        a[3],
                        bb[2 * half],
                        bb[2 * half + 1],
                    )
                    for e in cutlass.range_constexpr(4):
                        sacc[4 * half + e] = d[e]
            for nt in cutlass.range_constexpr(2):
                base = ss + ((kq * ROWS + g) * S_PITCH + th * 16 + nt * 8 + 2 * t) * 4
                _sts_v2_f32(base, sacc[4 * nt], sacc[4 * nt + 1])
                _sts_v2_f32(base + 8 * S_PITCH * 4, sacc[4 * nt + 2], sacc[4 * nt + 3])
            _bar_sync()

            # Softmax over the tile: thread -> row sr, tokens 2 sc, 2 sc + 1.
            sv = []
            for j in cutlass.range_constexpr(2):
                acc_s = _lds_f32(ss + (sr * S_PITCH + sc * 2 + j) * 4)
                for q4 in cutlass.range_constexpr(1, 4):
                    acc_s = acc_s + _lds_f32(ss + ((q4 * ROWS + sr) * S_PITCH + sc * 2 + j) * 4)
                pos = tile * TILE + sc * 2 + j
                sv.append(_neg_inf_if(Int32(pos >= kv_len), acc_s * scale_log2))
            mx = _fmax(sv[0], sv[1])
            for off in (1, 2, 4, 8):
                mx = _fmax(mx, _shfl_bfly_f32(mx, off))
            m_old = _lds_f32(sstat + sr * 4)
            m_new = _fmax(m_old, mx)
            m_use = _finite_or_zero(m_new)
            p0 = _ex2(sv[0] - m_use)
            p1 = _ex2(sv[1] - m_use)
            rs = p0 + p1
            for off in (1, 2, 4, 8):
                rs = rs + _shfl_bfly_f32(rs, off)
            alpha = _ex2(m_old - m_use)
            _sts_b32(sp + sr * P_BYTES + sc * 4, _pack_bf16x2(p0, p1))
            if sc == 0:
                l_old = _lds_f32(sstat + (ROWS + sr) * 4)
                _sts_f32(sstat + sr * 4, m_new)
                _sts_f32(sstat + (ROWS + sr) * 4, l_old * alpha + rs)
                _sts_f32(sstat + (2 * ROWS + sr) * 4, alpha)
            _bar_sync()

            # O = O * alpha + P @ V for this warp's 64 output columns.
            al0 = _lds_f32(sstat + (2 * ROWS + g) * 4)
            al1 = _lds_f32(sstat + (2 * ROWS + g + 8) * 4)
            for nt in cutlass.range_constexpr(8):
                o_acc[4 * nt] = o_acc[4 * nt] * al0
                o_acc[4 * nt + 1] = o_acc[4 * nt + 1] * al0
                o_acc[4 * nt + 2] = o_acc[4 * nt + 2] * al1
                o_acc[4 * nt + 3] = o_acc[4 * nt + 3] * al1
            for kk in cutlass.range_constexpr(TILE // 16):
                pa = _ldsm_x4(sp + lrow * P_BYTES + (kk * 16 + lcol8) * 2)
                for npair in cutlass.range_constexpr(4):
                    vcol = warp * 64 + npair * 16 + (lane >> 4) * 8
                    vtok = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8
                    vb = _ldsm_x4_trans(kvs + vtok * ROW_BYTES + vcol * 2)
                    for half in cutlass.range_constexpr(2):
                        nt = npair * 2 + half
                        d = _mma_bf16(
                            o_acc[4 * nt],
                            o_acc[4 * nt + 1],
                            o_acc[4 * nt + 2],
                            o_acc[4 * nt + 3],
                            pa[0],
                            pa[1],
                            pa[2],
                            pa[3],
                            vb[2 * half],
                            vb[2 * half + 1],
                        )
                        for e in cutlass.range_constexpr(4):
                            o_acc[4 * nt + e] = d[e]
            _bar_sync()

        # Partial outputs to the latent-slice owners (warp w's 64 columns: column tile nt goes to
        # CTA (64 w + 8 nt) / self.lat), 16 bytes per store (lanes t, t ^ 1 swap halves: even t
        # stores row g, odd t row g + 8); every CTA's row stats to every CTA.
        srow = tidx & (ROWS - 1)
        m_fin = _lds_f32(sstat + srow * 4)
        l_fin = _lds_f32(sstat + (ROWS + srow) * 4)
        if cutlass.const_expr(self.async_x):
            # Staged in KV stage 0 (idle after the loop) as [owner][row][C] FP32 and [row][2]
            # stats, then one bulk copy per owner of each (into its slot of this CTA). The
            # staging stays readable until this CTA's last wait: that wait needs every owner's
            # up-projection partials, which follow its merge of these copies.
            xo_s = skv
            xml_s = skv + self.ns * ROWS * self.lat * 4
            for nt in cutlass.range_constexpr(8):
                owner = (warp * 64 + nt * 8) // self.lat
                a0 = xo_s + ((owner * ROWS + g) * self.lat + (nt * 8) % self.lat + 2 * t) * 4
                _sts_v2_f32(a0, o_acc[4 * nt], o_acc[4 * nt + 1])
                _sts_v2_f32(a0 + 8 * self.lat * 4, o_acc[4 * nt + 2], o_acc[4 * nt + 3])
            if tidx < ROWS:
                _sts_v2_f32(xml_s + tidx * 8, m_fin, l_fin)
            _fence_proxy_async_shared()
            _bar_sync()
            # Lane 0 of warp w sends to owners w, w + 8, ...: the copies are issued in parallel
            # (a warp issues its bulk copies one after another).
            if lane == 0:
                for r in cutlass.range_constexpr((self.ns + NWARPS - 1) // NWARPS):
                    dst = warp + r * NWARPS
                    if dst < self.ns:
                        _bulk_s2c(
                            sred_o + s * (ROWS * self.lat * 4),
                            xo_s + dst * (ROWS * self.lat * 4),
                            ROWS * self.lat * 4,
                            o_bar,
                            dst,
                        )
                        _bulk_s2c(sred_ml + s * (ROWS * 8), xml_s, ROWS * 8, o_bar, dst)
            while _mbar_try_wait(o_bar, 0) == 0:
                pass
        else:
            odd = t & 1
            for nt in cutlass.range_constexpr(8):
                x = _pick_f(odd, o_acc[4 * nt], o_acc[4 * nt + 2])
                y = _pick_f(odd, o_acc[4 * nt + 1], o_acc[4 * nt + 3])
                xo = _shfl_bfly_xor1_f32(x)
                yo = _shfl_bfly_xor1_f32(y)
                w0 = _pick_f(odd, xo, o_acc[4 * nt])
                w1 = _pick_f(odd, yo, o_acc[4 * nt + 1])
                w2 = _pick_f(odd, o_acc[4 * nt + 2], xo)
                w3 = _pick_f(odd, o_acc[4 * nt + 3], yo)
                orow = g + 8 * odd
                owner = (warp * 64 + nt * 8) // self.lat
                ocol = (nt * 8) % self.lat + 2 * (t - odd)
                base = sred_o + ((s * ROWS + orow) * self.lat + ocol) * 4
                _st_cluster_v4_f32(_mapa(base, owner), w0, w1, w2, w3)
            # Thread -> row tidx % 16, destination tidx / 16.
            if tidx < ROWS * self.ns:
                _st_cluster_v2_f32(
                    _mapa(sred_ml + ((s * ROWS + srow) * 2) * 4, tidx // ROWS), m_fin, l_fin
                )
            _cluster_sync()

        # Merge of this CTA's latent slice: thread -> row sr, columns [cpt sc, cpt sc + cpt).
        mm = neg_inf
        for c in cutlass.range_constexpr(self.ns):
            mm = _fmax(mm, _lds_f32(sred_ml + ((c * ROWS + sr) * 2) * 4))
        mm_use = _finite_or_zero(mm)
        wts = []
        lsum = Float32(0.0)
        for c in cutlass.range_constexpr(self.ns):
            ml = _lds_v2_f32(sred_ml + ((c * ROWS + sr) * 2) * 4)
            wc = _ex2(ml[0] - mm_use)
            wts.append(wc)
            lsum = lsum + wc * ml[1]
        inv = _safe_rcp(lsum)
        cpt = self.lat // 16
        ov = [Float32(0.0)] * cpt
        for c in cutlass.range_constexpr(self.ns):
            src = sred_o + ((c * ROWS + sr) * self.lat + sc * cpt) * 4
            vals = _lds_v4_f32(src) if cpt == 4 else _lds_v2_f32(src)
            for e in cutlass.range_constexpr(cpt):
                ov[e] = ov[e] + wts[c] * vals[e]
        if cutlass.const_expr(cpt == 4):
            _sts_v2_b32(
                sp + sr * self.on_bytes + sc * 8,
                _pack_bf16x2(ov[0] * inv, ov[1] * inv),
                _pack_bf16x2(ov[2] * inv, ov[3] * inv),
            )
        else:
            _sts_b32(sp + sr * self.on_bytes + sc * 4, _pack_bf16x2(ov[0] * inv, ov[1] * inv))
        _bar_sync()

        # Up-projection K-split over this CTA's latent columns: warp w, output column tiles 2w and
        # 2w + 1 (each owned by the CTA of its output columns), rows g (head h0) and g + 8 (head
        # h0 + 1); the score buffer is free on every CTA by now.
        vacc0 = [Float32(0.0)] * 8
        vacc1 = [Float32(0.0)] * 8
        for kk in cutlass.range_constexpr(self.lat // 16):
            oa = _ldsm_x4(sp + lrow * self.on_bytes + (kk * 16 + lcol8) * 2)
            for j in cutlass.range_constexpr(2):
                for hh in cutlass.range_constexpr(HG):
                    vacc = vacc0 if hh == 0 else vacc1
                    f = ((kk * 2 + j) * HG + hh) * 2
                    d = _mma_bf16(
                        vacc[4 * j],
                        vacc[4 * j + 1],
                        vacc[4 * j + 2],
                        vacc[4 * j + 3],
                        oa[0],
                        oa[1],
                        oa[2],
                        oa[3],
                        vb_frag[f],
                        vb_frag[f + 1],
                    )
                    for e in cutlass.range_constexpr(4):
                        vacc[4 * j + e] = d[e]
        for j in cutlass.range_constexpr(2):
            vt = 2 * warp + j
            owner = (vt * 8) // self.vpc
            base = ss + ((s * ROWS + g) * self.vpc + (vt * 8) % self.vpc + 2 * t) * 4
            if cutlass.const_expr(self.async_x):
                _st_async_v2_f32(base, v_bar, owner, vacc0[4 * j], vacc0[4 * j + 1])
                _st_async_v2_f32(
                    base + 8 * self.vpc * 4, v_bar, owner, vacc1[4 * j + 2], vacc1[4 * j + 3]
                )
            else:
                _st_cluster_v2_f32(_mapa(base, owner), vacc0[4 * j], vacc0[4 * j + 1])
                _st_cluster_v2_f32(
                    _mapa(base + 8 * self.vpc * 4, owner), vacc1[4 * j + 2], vacc1[4 * j + 3]
                )
        if cutlass.const_expr(self.async_x):
            while _mbar_try_wait(v_bar, 0) == 0:
                pass
        else:
            _cluster_sync()

        # This CTA's self.vpc output columns: thread -> row fr, columns 2 fc + {0, 1}.
        if tidx < ROWS * (self.vpc // 2):
            fr = tidx // (self.vpc // 2)
            fc = tidx % (self.vpc // 2)
            f_ok = Int32((fr & 7) < BLK)
            fm = b * BLK + _pick_i(f_ok, fr & 7, Int32(0))
            o0 = Float32(0.0)
            o1 = Float32(0.0)
            for c in cutlass.range_constexpr(self.ns):
                pr = _lds_v2_f32(ss + ((c * ROWS + fr) * self.vpc + 2 * fc) * 4)
                o0 = o0 + pr[0]
                o1 = o1 + pr[1]
            if f_ok != 0:
                dst = out_addr + (
                    Int64((fm * H + h0 + (fr >> 3)) * D_V + s * self.vpc + 2 * fc) * 2
                )
                _st_global_b32(dst, _pack_bf16x2(o0, o1))


@functools.cache
def _compiled(
    num_heads: int,
    block_size: int,
    page_size: int,
    use_pdl: bool,
    cluster: int,
    staged: bool,
    async_x: bool,
):
    kernel = DSparkMlaAttendKernel(
        num_heads, block_size, page_size, use_pdl, cluster, staged, async_x
    )
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        Int64(1),
        a,
        Int32(1),
        Int32(1),
        a,
        a,
        a,
        a,
        a,
        Float32(1.0),
        Int32(1),
        a,
        a,
        a,
        a,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


#: ``TLLM_DSPARK_MLA_FUSED_STAGED=1``: q_nope, k_b and v_b reach the MMA fragments through a few
#: bulk copies into idle shared memory and ``ldmatrix`` instead of ~80 four-byte loads per thread
#: (k_b and v_b read from padded copies, see :func:`staged_weights`). The same fragment values.
STAGED_ENV_VAR = "TLLM_DSPARK_MLA_FUSED_STAGED"
#: ``TLLM_DSPARK_MLA_FUSED_ASYNC=1``: the three cluster exchanges complete on mbarriers of the
#: receiving CTA (``st.async``) instead of cluster barriers. The same values.
ASYNC_ENV_VAR = "TLLM_DSPARK_MLA_FUSED_ASYNC"


def staged_enabled() -> bool:
    """Whether ``TLLM_DSPARK_MLA_FUSED_STAGED=1`` stages the attention kernel's operands."""
    return os.environ.get(STAGED_ENV_VAR, "0") == "1"


def async_enabled() -> bool:
    """Whether ``TLLM_DSPARK_MLA_FUSED_ASYNC=1`` runs the cluster exchanges on mbarriers."""
    return os.environ.get(ASYNC_ENV_VAR, "0") == "1"


#: Bytes per staged k_b row (128 BF16 values + 16 bytes of padding).
KB_PITCH = D_NOPE * 2 + 16


def vb_pitch(cluster: int) -> int:
    """Bytes per staged v_b row: ``512 / cluster`` BF16 values, padded by 16 bytes for 16-CTA
    clusters."""
    return (D_LAT // cluster) * 2 + (16 if cluster == 16 else 0)


def staged_weights(k_b_t: torch.Tensor, v_b: torch.Tensor, cluster: int) -> tuple:
    """The padded weight copies ``TLLM_DSPARK_MLA_FUSED_STAGED`` copies slices of: ``k_b_t [H,
    512, 128]`` padded to ``[H, 512, KB_PITCH / 2]`` and ``v_b [H, 128, 512]`` re-laid
    ``[H, cluster, 128, vb_pitch / 2]`` (CTA ``s``'s latent rows of a head contiguous), zero
    padding (never read)."""
    heads = v_b.shape[0]
    lat = D_LAT // cluster
    kb = torch.zeros((heads, D_LAT, KB_PITCH // 2), dtype=k_b_t.dtype, device=k_b_t.device)
    kb[..., :D_NOPE] = k_b_t
    vb = torch.zeros(
        (heads, cluster, D_V, vb_pitch(cluster) // 2), dtype=v_b.dtype, device=v_b.device
    )
    vb[..., :lat] = v_b.view(heads, D_V, cluster, lat).permute(0, 2, 1, 3)
    return kb, vb


def default_cluster(batch: int) -> int:
    """CTAs per cluster: ``TLLM_DSPARK_MLA_FUSED_CLUSTER=<n>`` or ``<b>:<n>,...`` (requests: CTAs,
    the first entry with ``b >= batch``), else 16 up to two requests and 8 above (a 16-CTA cluster
    needs 16 free SMs of one GPC: at four requests, eight such clusters no longer find them)."""
    spec = os.environ.get("TLLM_DSPARK_MLA_FUSED_CLUSTER", "")
    if spec:
        if ":" not in spec:
            return int(spec)
        for item in spec.split(","):
            reqs, n = item.split(":")
            if batch <= int(reqs):
                return int(n)
    return 16 if batch <= 2 else 8


@functools.cache
def _zero_row(device: torch.device) -> torch.Tensor:
    """A zero KV row in global memory: the source of the copies of rows past the KV length."""
    return torch.zeros(D_QK, dtype=torch.bfloat16, device=device)


def can_run(
    q: torch.Tensor,
    query: torch.Tensor,
    pool: torch.Tensor,
    page_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    k_b_t: torch.Tensor,
    v_b: torch.Tensor,
) -> bool:
    """Whether :func:`run_dspark_mla_attend` serves these operands."""
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        return False
    if q.dim() != 4:
        return False
    batch, block, heads, qdim = q.shape
    return (
        q.dtype == query.dtype == pool.dtype == k_b_t.dtype == v_b.dtype == torch.bfloat16
        and q.is_contiguous()
        and query.is_contiguous()
        and qdim == D_NOPE + D_ROPE
        and tuple(query.shape) == (batch, block, heads, D_QK)
        and heads % HG == 0
        and 1 <= block <= SLOT
        and pool.dim() == 3
        and pool.shape[2] == D_QK
        and pool.stride(2) == 1
        and pool.stride(1) == D_QK
        and page_tables.dtype == torch.int32
        and page_tables.dim() == 2
        and page_tables.stride(1) == 1
        and page_tables.shape[0] >= batch
        and kv_lens.dtype == torch.int32
        and kv_lens.is_contiguous()
        and kv_lens.numel() >= batch
        and tuple(k_b_t.shape) == (heads, D_LAT, D_NOPE)
        and k_b_t.is_contiguous()
        and tuple(v_b.shape) == (heads, D_V, D_LAT)
        and v_b.is_contiguous()
        and q.data_ptr() % 16 == 0
        and query.data_ptr() % 16 == 0
        and pool.data_ptr() % 16 == 0
    )


def run_dspark_mla_attend(
    q: torch.Tensor,
    query: torch.Tensor,
    pool: torch.Tensor,
    page_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    k_b_t: torch.Tensor,
    v_b: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
    use_pdl: bool = True,
    cluster: int = 0,
    staged: tuple = None,
    prefetch: tuple = None,
) -> torch.Tensor:
    """Block attention of one drafter layer (module docstring) into ``out [B, T, H, 128]``.

    ``q [B, T, H, 192]``: q_b output (``q_nope`` in its first 128 columns); ``query [B, T, H,
    576]``: its last 64 columns hold the rotated ``q_pe``; ``pool [pages, page_size, 576]``: the
    layer's paged latent cache, the block rows already stored; ``page_tables [>= B, pages]``
    int32; ``kv_lens [>= B]`` int32 (context + block); ``k_b_t [H, 512, 128]`` (``k_b``
    transposed), ``v_b [H, 128, 512]``; ``staged``: :func:`staged_weights` of these weights for
    this cluster size (made here when None and ``TLLM_DSPARK_MLA_FUSED_STAGED=1``; callers cache
    it); ``prefetch``: :func:`staged_weights` of the next layer, whose slices the staged kernel
    pulls into L2 (a hint), or None.
    """
    batch, block, heads, _ = q.shape
    if cluster == 0:
        cluster = default_cluster(batch)
    use_staged = staged_enabled()
    if use_staged and staged is None:
        staged = staged_weights(k_b_t, v_b, cluster)
    _compiled(heads, block, pool.shape[1], bool(use_pdl), cluster, use_staged, async_enabled())(
        Int64(q.data_ptr()),
        Int64(query.data_ptr()),
        Int64(pool.data_ptr()),
        Int64(pool.stride(0)),
        Int64(page_tables.data_ptr()),
        Int32(page_tables.stride(0)),
        Int32(page_tables.shape[1]),
        Int64(kv_lens.data_ptr()),
        Int64(k_b_t.data_ptr()),
        Int64(v_b.data_ptr()),
        Int64(out.data_ptr()),
        Int64(_zero_row(q.device).data_ptr()),
        Float32(softmax_scale * math.log2(math.e)),
        Int32(batch),
        Int64(staged[0].data_ptr() if use_staged else k_b_t.data_ptr()),
        Int64(staged[1].data_ptr() if use_staged else v_b.data_ptr()),
        Int64(prefetch[0].data_ptr() if use_staged and prefetch is not None else 0),
        Int64(prefetch[1].data_ptr() if use_staged and prefetch is not None else 0),
        cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return out


__all__ = [
    "DSparkMlaAttendKernel",
    "async_enabled",
    "can_run",
    "run_dspark_mla_attend",
    "staged_enabled",
    "staged_weights",
]

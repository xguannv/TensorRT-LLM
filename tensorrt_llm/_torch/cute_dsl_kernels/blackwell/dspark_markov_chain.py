# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark vanilla-Markov greedy draft chain over a vocab-parallel LM head, one launch (SM100 / SM103).

Draft token ``j`` of request ``b`` is the global argmax over the vocab of
``base[b, j, v] + bf16(markov_w1[prev] . markov_w2[v])``, where ``prev`` is the anchor token at
``j = 0`` and draft token ``j - 1`` after it, and ``base`` is this rank's shard of the block's LM
head logits. ``vocab_parallel._markov_chain_kernel`` (Triton) runs the same chain; this kernel gives
bitwise its tokens with fewer synchronizations per step:

* Each CTA owns ``BV = 128`` shard columns and keeps their ``markov_w2`` rows in shared memory for
  the whole chain (loaded before the PDL wait). Per step it gathers the ``markov_w1`` rows of every
  request's ``prev`` into shared memory and computes the bias with the instruction the Triton
  ``tl.dot`` lowers to (``mma.sync.m16n8k16`` BF16 -> FP32, requests padded to 16 rows, the rank
  dimension in K steps of 16 in order), rounds it to BF16 like ``F.linear`` and adds the shard
  logits (loaded before the step's exchange wait).
* Argmax in ``torch.argmax`` order (NaN first, then the largest value, the lowest index among equal
  values; ``+0 == -0``) within the CTA; each CTA of a cluster of 8 sends its candidate per request
  to the cluster leader with ``st.async`` on the leader's barrier of the step; the leader takes the
  cluster's winner and multicasts the ``(value, global index)`` pair into every rank's exchange
  buffer (``-0.0`` marks an empty slot, so a published ``-0.0`` becomes ``+0.0``).
* Warp ``w`` of every CTA polls requests ``w, w + 4, ...``: all ``TP x clusters`` slots of a request
  are loaded at once and reloaded until none is empty; the winner is the next ``prev`` (CTA 0 stores
  the tokens).

Exchange sets: launch ``L`` (a device counter) uses set ``L % 3`` and clears this rank's copy of set
``(L + 2) % 3``, which this rank finished reading in launch ``L - 1`` (every rank's pairs of that
launch had arrived); another rank writes that set again only in launch ``L + 2``, which needs this
rank's pairs of launch ``L + 1``, published after this launch completed. All CTAs read ``L`` after
their ``griddepcontrol.wait`` and before the first exchange; CTA 0 advances it after the last one.
Every CTA must be resident at once (each step waits for every cluster's pair).

PDL: shared-memory staging of ``markov_w2`` (a weight) happens before ``griddepcontrol.wait``; the
counter, the clear, the anchors and the shard logits come after it, and the dependents are triggered
right after it: a next chain launch can become resident early but touches the counter and the
exchange only once this launch completed; other successors read the tokens after their own wait.
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

#: Shard columns per CTA.
BV = 128
#: CTAs per cluster (one candidate per cluster and request goes through the exchange).
CLUSTER = 8
#: Requests padded to the MMA's 16 rows.
MAX_REQS = 16
#: Draft positions per launch the exchange sets hold.
MAX_STEPS = 16
#: Threads per CTA (4 warps x 32 shard columns each).
THREADS = 128
#: Bytes of padding per shared-memory row (keeps the fragment loads conflict free).
ROW_PAD = 16
#: Exchange buffer sets (rotation period).
NUM_SETS = 3
_NEG_ZERO_BITS = -2147483648  # 0x80000000
_INT32_MAX = 2**31 - 1


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
def _ld_b64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(), [_lv(addr, loc=loc, ip=ip)], "ld.global.b64 $0, [$1];", "=l,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ld_vol_v2(addr, *, loc=None, ip=None):
    """A ``(value, index)`` pair of the exchange, read past L1 (peers write it over NVLink)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _ld_vol_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_v2(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
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
def _st_sentinel_v4(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, 0x80000000;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
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


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "", loc=loc, ip=ip)


@dsl_user_op
def _gdc_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _gdc_launch(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


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
def _round_bf16_f32(x, *, loc=None, ip=None):
    """FP32 -> BF16 (round to nearest even) -> FP32, as ``.to(bf16).to(fp32)``."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.f32.bf16 $0, h;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


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
def _shfl_bfly_b32(value, offset: int, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_fv(x, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f", loc=loc, ip=ip))


@dsl_user_op
def _bits_f32(x, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_iv(x, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _better(av, ai, bv, bi, *, loc=None, ip=None):
    """1 if ``(av, ai)`` comes before ``(bv, bi)`` in ``torch.argmax`` order: NaN beats any number
    (lower index among NaNs), else the larger value, the lower index among equal values."""
    return Int32(
        _asm(
            T.i32(),
            [
                _fv(av, loc=loc, ip=ip),
                _iv(ai, loc=loc, ip=ip),
                _fv(bv, loc=loc, ip=ip),
                _iv(bi, loc=loc, ip=ip),
            ],
            "{\n.reg .pred na, nb, lt, gt, eq, il, r, t;\n"
            "setp.nan.f32 na, $1, $1;\nsetp.nan.f32 nb, $3, $3;\n"
            "setp.lt.s32 il, $2, $4;\n"
            "setp.gt.f32 gt, $1, $3;\nsetp.eq.f32 eq, $1, $3;\n"
            # both numbers: gt | (eq & il)
            "and.pred t, eq, il;\nor.pred r, gt, t;\n"
            # a NaN: !nb | il
            "not.pred t, nb;\nor.pred t, t, il;\n"
            "selp.u32 $0, 1, 0, r;\n"
            "@na selp.u32 $0, 1, 0, t;\n"
            # b NaN, a number: never
            "not.pred t, na;\nand.pred t, t, nb;\n@t mov.u32 $0, 0;\n}",
            "=r,f,r,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pick_v(flag, av, bv, *, loc=None, ip=None):
    """``av`` if ``flag`` else ``bv`` (FP32)."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(flag, loc=loc, ip=ip), _fv(av, loc=loc, ip=ip), _fv(bv, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}",
            "=f,r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pick_i(flag, a, b, *, loc=None, ip=None):
    """``a`` if ``flag`` else ``b`` (int32)."""
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
def _ld_nc_v4_if(addr, valid, *, loc=None, ip=None):
    """16 bytes at ``addr`` if ``valid`` else zeros (no access when not valid)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip), _iv(valid, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $5, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ld_b32_if(addr, valid, *, loc=None, ip=None):
    """4 bytes at ``addr`` if ``valid`` else 0."""
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
def _pair_dirty(w0, w1, *, loc=None, ip=None):
    """1 if either word of an exchange pair is the empty sentinel."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w0, loc=loc, ip=ip), _iv(w1, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.eq.b32 p, $1, 0x80000000;\nsetp.eq.or.b32 p, $2, 0x80000000, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_to_s32(x, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_fv(x, loc=loc, ip=ip)], "cvt.rzi.s32.f32 $0, $1;", "=r,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _s32_to_f32(x, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_iv(x, loc=loc, ip=ip)], "cvt.rn.f32.s32 $0, $1;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _vote_any(flag, *, loc=None, ip=None):
    """1 if ``flag`` is nonzero on any lane of the (full) warp."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(flag, loc=loc, ip=ip)],
            "{\n.reg .pred p, q;\nsetp.ne.s32 p, $1, 0;\nvote.sync.any.pred q, p, 0xffffffff;\n"
            "selp.u32 $0, 1, 0, q;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


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
def _mbar_init_fence_cluster_sync(*, loc=None, ip=None):
    """Make the mbarrier inits visible to the cluster, then a cluster barrier."""
    _asm(
        None,
        [],
        "fence.mbarrier_init.release.cluster;\n"
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_arrive_expect_tx(addr, tx, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(tx, loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_try_wait(addr, phase: int, *, loc=None, ip=None):
    """1 once the mbarrier at ``addr`` completed phase ``phase``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n"
            f"mbarrier.try_wait.parity.shared::cta.b64 p, [$1], {phase};\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_async_cluster_v2(local_addr, local_mbar, peer, v0, v1, *, loc=None, ip=None):
    """``st.async`` of two words to ``local_addr`` of cluster CTA ``peer``, completing 8 bytes of
    the transaction count of its mbarrier at ``local_mbar``."""
    _asm(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _iv(peer, loc=loc, ip=ip),
            _iv(v0, loc=loc, ip=ip),
            _iv(v1, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $2;\n"
        "mapa.shared::cluster.u32 rb, $1, $2;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [ra], {$3, $4}, [rb];\n}",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


class DSparkMarkovChainKernel:
    """The DSpark Markov draft chain of one block (see the module docstring)."""

    def __init__(
        self,
        steps: int,
        rank: int,
        tp: int,
        num_ctas: int,
        use_pdl: bool,
        anchor_64: bool,
        num_reqs: int,
    ):
        if rank % 16 or rank > 512:
            raise ValueError(f"markov rank {rank} must be a multiple of 16 (<= 512)")
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"{steps} draft steps (1..{MAX_STEPS})")
        if num_ctas % CLUSTER:
            raise ValueError(f"{num_ctas} CTAs is not a multiple of the cluster size {CLUSTER}")
        if not 1 <= num_reqs <= MAX_REQS:
            raise ValueError(f"{num_reqs} requests (1..{MAX_REQS})")
        self.B = num_reqs
        self.steps = steps
        self.R = rank
        self.tp = tp
        self.num_ctas = num_ctas
        self.ncl = num_ctas // CLUSTER
        self.slots = tp * self.ncl
        self.use_pdl = use_pdl
        self.anchor_64 = anchor_64
        self.row_bytes = rank * 2 + ROW_PAD
        # Floats of one exchange set: [step][request][rank][cluster][value, index].
        self.set_floats = MAX_STEPS * MAX_REQS * self.slots * 2

    @cute.jit
    def __call__(
        self,
        base_addr: Int64,
        stride_req: Int32,
        stride_step: Int32,
        anchor_addr: Int64,
        w1_addr: Int64,
        w2_addr: Int64,
        tokens_addr: Int64,
        stride_tokens: Int32,
        mc_addr: Int64,
        uc_addr: Int64,
        launch_addr: Int64,
        num_reqs: Int32,
        vocab: Int32,
        n_cols: Int32,
        col_base: Int32,
        tp_rank: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            w2: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, BV * self.row_bytes], 16]
            w1: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, MAX_REQS * self.row_bytes], 16]
            # Per warp and request: best (value, index) of the warp's columns.
            cand: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4 * MAX_REQS * 2], 16]
            # Leader's mailbox: [step][cluster CTA][request][value, index].
            mbox: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, MAX_STEPS * CLUSTER * MAX_REQS * 2], 16
            ]
            prev: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MAX_REQS], 16]
            # One mbarrier per step (the leader's mailbox of that step).
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, MAX_STEPS], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            base_addr,
            stride_req,
            stride_step,
            anchor_addr,
            w1_addr,
            w2_addr,
            tokens_addr,
            stride_tokens,
            mc_addr,
            uc_addr,
            launch_addr,
            num_reqs,
            vocab,
            n_cols,
            col_base,
            tp_rank,
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=(THREADS, 1, 1),
            cluster=(CLUSTER, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def _reduce_pairs_warp(self, v: Float32, i: Int32):
        """Butterfly argmax over the 32 lanes (every lane gets the winner)."""
        for off in (16, 8, 4, 2, 1):
            ov = _bits_f32(_shfl_bfly_b32(_f32_bits(v), off))
            oi = _shfl_bfly_b32(i, off)
            f = _better(ov, oi, v, i)
            v = _pick_v(f, ov, v)
            i = _pick_i(f, oi, i)
        return v, i

    @cute.kernel
    def kernel(
        self,
        base_addr: Int64,
        stride_req: Int32,
        stride_step: Int32,
        anchor_addr: Int64,
        w1_addr: Int64,
        w2_addr: Int64,
        tokens_addr: Int64,
        stride_tokens: Int32,
        mc_addr: Int64,
        uc_addr: Int64,
        launch_addr: Int64,
        num_reqs: Int32,
        vocab: Int32,
        n_cols: Int32,
        col_base: Int32,
        tp_rank: Int32,
    ):
        R = self.R
        RB = self.row_bytes
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_id = cta // CLUSTER
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane >> 2
        t = lane & 3
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        w2s = storage.w2.data_ptr().toint()
        w1s = storage.w1.data_ptr().toint()
        cand = storage.cand.data_ptr().toint()
        mbox = storage.mbox.data_ptr().toint()
        prev_s = storage.prev.data_ptr().toint()
        mbar = storage.mbar.data_ptr().toint()
        c0 = cta * BV
        B = self.B

        # The leader's per-step mailbox barriers: initialized before any peer writes to them.
        if crank == 0:
            if tidx == 0:
                for s in cutlass.range_constexpr(self.steps):
                    _mbar_init(mbar + s * 8, 1)

        # markov_w2 rows of this CTA's columns (a weight: before the wait); zero rows past the shard.
        vec_per_row = R // 8
        for p in cutlass.range(tidx, BV * vec_per_row, THREADS, unroll=1):
            r = p // vec_per_row
            q = p - r * vec_per_row
            ok = Int32(c0 + r < n_cols)
            w = _ld_nc_v4_if(w2_addr + Int64((c0 + r) * R + q * 8) * 2, ok)
            _sts_v4(w2s + r * RB + q * 16, w[0], w[1], w[2], w[3])
        # The markov_w1 rows of padding requests stay zero.
        for p in cutlass.range(tidx, MAX_REQS * vec_per_row, THREADS, unroll=1):
            r = p // vec_per_row
            q = p - r * vec_per_row
            _sts_v4(w1s + r * RB + q * 16, Int32(0), Int32(0), Int32(0), Int32(0))

        _mbar_init_fence_cluster_sync()

        if cutlass.const_expr(self.use_pdl):
            _gdc_wait()
            _gdc_launch()

        # Exchange rotation: this launch's set, and the clear of the set two launches back. After
        # the wait: the dependents are triggered early, so the next chain launch may already be
        # resident, but it reads the counter and clears only after this launch completed.
        launch = _ld_vol_b32(launch_addr)
        cur_set = launch % NUM_SETS
        clr_set = (launch + 2) % NUM_SETS
        set_bytes = self.set_floats * 4
        clr_base = uc_addr + Int64(clr_set) * Int64(set_bytes)
        n_vec = set_bytes // 16
        for p in cutlass.range(cta * THREADS + tidx, n_vec, self.num_ctas * THREADS, unroll=1):
            _st_sentinel_v4(clr_base + Int64(p) * 16)

        if tidx < B:
            a = Int32(0)
            if cutlass.const_expr(self.anchor_64):
                a = Int32(_ld_b64(anchor_addr + Int64(tidx) * 8))
            else:
                a = _ld_b32(anchor_addr + Int64(tidx) * 4)
            _sts_b32(prev_s + tidx * 4, a)
        _bar_sync()

        set_off = Int64(cur_set) * Int64(set_bytes)
        step_floats = MAX_REQS * self.slots * 2
        nt_per_warp = BV // 4 // 8
        npl = (self.slots + 31) // 32
        neg_inf = Float32(float("-inf"))
        for s in cutlass.range_constexpr(self.steps + 1):
            # This step's shard logits of the thread's fragment (rows g and g + 8, two columns
            # per n8 tile), issued before the exchange wait: they do not depend on it.
            base_w = []
            if cutlass.const_expr(s < self.steps):
                for half in cutlass.range_constexpr(2):
                    row = g + 8 * half
                    rok = Int32(row < B)
                    rrow = _pick_i(rok, row, Int32(0))
                    brow = base_addr + Int64(rrow * stride_req + s * stride_step) * 2
                    for nt in cutlass.range_constexpr(nt_per_warp):
                        col = c0 + warp * 32 + nt * 8 + 2 * t
                        ok = rok & Int32(col < n_cols)
                        base_w.append(_ld_b32_if(brow + Int64(col) * 2, ok))
            if cutlass.const_expr(s > 0):
                # Every rank's cluster pairs of step s - 1. Warp w takes requests w, w + 4, ...:
                # all of a request's slots are loaded at once and reloaded until none is empty.
                poll = Int32(1)
                if cutlass.const_expr(s == self.steps):
                    poll = Int32(cta == 0)
                if poll != 0:
                    region = uc_addr + set_off + Int64((s - 1) * step_floats) * 4
                    for u in cutlass.range_constexpr((B + 3) // 4):
                        r = warp + 4 * u
                        if r < B:
                            addrs = []
                            oks = []
                            for k in cutlass.range_constexpr(npl):
                                j = lane + 32 * k
                                ok = Int32(j < self.slots)
                                jj = _pick_i(ok, j, Int32(0))
                                addrs.append(region + Int64((r * self.slots + jj) * 2) * 4)
                                oks.append(ok)
                            pvs = [_ld_vol_v2(addrs[k]) for k in range(npl)]
                            dirty = Int32(0)
                            for k in cutlass.range_constexpr(npl):
                                dirty = dirty | (_pair_dirty(pvs[k][0], pvs[k][1]) & oks[k])
                            while _vote_any(dirty) != 0:
                                pvs = [_ld_vol_v2(addrs[k]) for k in range(npl)]
                                dirty = Int32(0)
                                for k in cutlass.range_constexpr(npl):
                                    dirty = dirty | (_pair_dirty(pvs[k][0], pvs[k][1]) & oks[k])
                            bv = neg_inf
                            bi = Int32(_INT32_MAX)
                            for k in cutlass.range_constexpr(npl):
                                v = _bits_f32(pvs[k][0])
                                i = _f32_to_s32(_bits_f32(pvs[k][1]))
                                f = _better(v, i, bv, bi) & oks[k]
                                bv = _pick_v(f, v, bv)
                                bi = _pick_i(f, i, bi)
                            bv, bi = self._reduce_pairs_warp(bv, bi)
                            if lane == 0:
                                _sts_b32(prev_s + r * 4, bi)
                                if cta == 0:
                                    _st_b32(
                                        tokens_addr + Int64(r * stride_tokens + (s - 1)) * 4, bi
                                    )
                _bar_sync()
            if cutlass.const_expr(s < self.steps):
                # markov_w1 rows of the requests' prev tokens (zero for an out-of-vocab prev).
                for p in cutlass.range(tidx, B * vec_per_row, THREADS, unroll=1):
                    r = p // vec_per_row
                    q = p - r * vec_per_row
                    pr = _lds_b32(prev_s + r * 4)
                    ok = Int32(pr >= 0) & Int32(pr < vocab)
                    prc = _pick_i(ok, pr, Int32(0))
                    w = _ld_nc_v4_if(w1_addr + (Int64(prc) * R + Int64(q * 8)) * 2, ok)
                    _sts_v4(w1s + r * RB + q * 16, w[0], w[1], w[2], w[3])
                _bar_sync()

                # bias = A (w1 rows, 16 x R) * B (w2 rows of the warp's 32 columns, R x 32).
                acc = [Float32(0.0)] * (nt_per_warp * 4)
                for ks in cutlass.range_constexpr(R // 16):
                    k0 = ks * 16
                    a0 = _lds_b32(w1s + g * RB + (k0 + 2 * t) * 2)
                    a1 = _lds_b32(w1s + (g + 8) * RB + (k0 + 2 * t) * 2)
                    a2 = _lds_b32(w1s + g * RB + (k0 + 2 * t + 8) * 2)
                    a3 = _lds_b32(w1s + (g + 8) * RB + (k0 + 2 * t + 8) * 2)
                    for nt in cutlass.range_constexpr(nt_per_warp):
                        n = warp * 32 + nt * 8 + g
                        b0 = _lds_b32(w2s + n * RB + (k0 + 2 * t) * 2)
                        b1 = _lds_b32(w2s + n * RB + (k0 + 2 * t + 8) * 2)
                        d = _mma_bf16(
                            acc[4 * nt],
                            acc[4 * nt + 1],
                            acc[4 * nt + 2],
                            acc[4 * nt + 3],
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                        )
                        for e in cutlass.range_constexpr(4):
                            acc[4 * nt + e] = d[e]

                # Biased logits; per row (g, g + 8) the thread's best column, then the quad's.
                for half in cutlass.range_constexpr(2 if B > 8 else 1):
                    row = g + 8 * half
                    rok = Int32(row < B)
                    bv = neg_inf
                    bi = Int32(_INT32_MAX)
                    for nt in cutlass.range_constexpr(nt_per_warp):
                        col = c0 + warp * 32 + nt * 8 + 2 * t
                        lo, hi = _bf16x2_f32(base_w[half * nt_per_warp + nt])
                        v0 = _fadd(lo, _round_bf16_f32(acc[4 * nt + 2 * half]))
                        v1 = _fadd(hi, _round_bf16_f32(acc[4 * nt + 2 * half + 1]))
                        ok0 = rok & Int32(col < n_cols)
                        ok1 = rok & Int32(col + 1 < n_cols)
                        i0 = col_base + col
                        f0 = _better(v0, i0, bv, bi) & ok0
                        bv = _pick_v(f0, v0, bv)
                        bi = _pick_i(f0, i0, bi)
                        f1 = _better(v1, i0 + 1, bv, bi) & ok1
                        bv = _pick_v(f1, v1, bv)
                        bi = _pick_i(f1, i0 + 1, bi)
                    for off in (1, 2):
                        ov = _bits_f32(_shfl_bfly_b32(_f32_bits(bv), off))
                        oi = _shfl_bfly_b32(bi, off)
                        f = _better(ov, oi, bv, bi)
                        bv = _pick_v(f, ov, bv)
                        bi = _pick_i(f, oi, bi)
                    if t == 0:
                        if row < B:
                            slot = cand + ((warp * MAX_REQS + row) * 2) * 4
                            _sts_b32(slot, _f32_bits(bv))
                            _sts_b32(slot + 4, bi)
                _bar_sync()
                # CTA winner per request; peers send it to the cluster leader's mailbox (st.async on
                # the leader's barrier of the step), the leader reduces and publishes to every rank.
                if tidx < B:
                    cv = neg_inf
                    ci = Int32(_INT32_MAX)
                    for w in cutlass.range_constexpr(4):
                        slot = cand + ((w * MAX_REQS + tidx) * 2) * 4
                        v = _bits_f32(_lds_b32(slot))
                        i = _lds_b32(slot + 4)
                        f = _better(v, i, cv, ci)
                        cv = _pick_v(f, v, cv)
                        ci = _pick_i(f, i, ci)
                    if crank != 0:
                        box = mbox + (((s * CLUSTER + crank) * MAX_REQS + tidx) * 2) * 4
                        _st_async_cluster_v2(box, mbar + s * 8, Int32(0), _f32_bits(cv), ci)
                    else:
                        if tidx == 0:
                            _mbar_arrive_expect_tx(mbar + s * 8, Int32((CLUSTER - 1) * B * 8))
                        while _mbar_try_wait(mbar + s * 8, 0) == 0:
                            pass
                        for c in cutlass.range_constexpr(1, CLUSTER):
                            box = mbox + (((s * CLUSTER + c) * MAX_REQS + tidx) * 2) * 4
                            v = _bits_f32(_lds_b32(box))
                            i = _lds_b32(box + 4)
                            f = _better(v, i, cv, ci)
                            cv = _pick_v(f, v, cv)
                            ci = _pick_i(f, i, ci)
                        vb = _f32_bits(cv)
                        vb = _pick_i(Int32(vb == _NEG_ZERO_BITS), Int32(0), vb)
                        out = (
                            mc_addr
                            + set_off
                            + Int64(
                                (s * step_floats)
                                + ((tidx * self.slots + tp_rank * self.ncl + cluster_id) * 2)
                            )
                            * 4
                        )
                        _st_v2(out, vb, _f32_bits(_s32_to_f32(ci)))

        if cta == 0:
            if tidx == 0:
                _st_b32(launch_addr, launch + 1)


@functools.cache
def _compiled(
    steps: int, rank: int, tp: int, num_ctas: int, use_pdl: bool, anchor_64: bool, num_reqs: int
):
    kernel = DSparkMarkovChainKernel(steps, rank, tp, num_ctas, use_pdl, anchor_64, num_reqs)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        Int32(1),
        Int32(1),
        a,
        a,
        a,
        a,
        Int32(1),
        a,
        a,
        a,
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(0),
        Int32(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def num_ctas_for(shard: int) -> Optional[int]:
    """CTAs of the chain for a shard of ``shard`` columns, or None when it does not fit."""
    n = -(-shard // BV)
    n = -(-n // CLUSTER) * CLUSTER
    return n if n <= 148 else None


def exchange_floats(tp: int, num_ctas: int) -> int:
    """Floats of the exchange buffer (all sets) for ``tp`` ranks."""
    return NUM_SETS * MAX_STEPS * MAX_REQS * tp * (num_ctas // CLUSTER) * 2


def run_dspark_markov_chain(
    local_block: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    w2_shard: torch.Tensor,
    tokens: torch.Tensor,
    mc_addr: int,
    uc_addr: int,
    launch: torch.Tensor,
    tp: int,
    tp_rank: int,
    col_base: int,
    vocab: int,
    use_pdl: bool = True,
) -> torch.Tensor:
    """Fill ``tokens [B, K]`` (int32) with the Markov chain's draft tokens of the block logits
    ``local_block [B, K, shard]`` (BF16, this rank's vocab shard starting at ``col_base``).

    ``mc_addr`` / ``uc_addr``: multicast and local addresses of the exchange buffer
    (``exchange_floats(tp, num_ctas_for(shard))`` floats, ``-0.0`` initially; the same buffer for
    one rank at ``tp == 1``); ``launch``: its int32 launch counter. Every rank of the group launches
    it with the same shapes."""
    num_reqs, steps, shard = local_block.shape
    rank = markov_w1.shape[1]
    num_ctas = num_ctas_for(shard)
    assert num_ctas is not None and num_reqs <= MAX_REQS
    assert local_block.dtype == torch.bfloat16 and local_block.stride(-1) == 1
    assert markov_w1.dtype == torch.bfloat16 and markov_w1.is_contiguous()
    assert w2_shard.dtype == torch.bfloat16 and w2_shard.stride(-1) == 1
    assert w2_shard.stride(0) == rank and tokens.dtype == torch.int32 and tokens.stride(-1) == 1
    _compiled(steps, rank, tp, num_ctas, bool(use_pdl), anchors.dtype == torch.int64, num_reqs)(
        Int64(local_block.data_ptr()),
        Int32(local_block.stride(0)),
        Int32(local_block.stride(1)),
        Int64(anchors.data_ptr()),
        Int64(markov_w1.data_ptr()),
        Int64(w2_shard.data_ptr()),
        Int64(tokens.data_ptr()),
        Int32(tokens.stride(0)),
        Int64(mc_addr),
        Int64(uc_addr),
        Int64(launch.data_ptr()),
        Int32(num_reqs),
        Int32(vocab),
        Int32(shard),
        Int32(col_base),
        Int32(tp_rank),
        cuda.CUstream(torch.cuda.current_stream(local_block.device).cuda_stream),
    )
    return tokens


__all__ = [
    "DSparkMarkovChainKernel",
    "exchange_floats",
    "num_ctas_for",
    "run_dspark_markov_chain",
]

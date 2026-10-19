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
"""L2 prefetch of the routed experts a decode MoE layer is likely to read (SM100/SM103).

At decode batch sizes a rank of the Kimi-K3 MoE reads a few tens of its local experts per layer,
and the tokens of a request keep choosing the same experts from one step to the next. One launch
of :class:`K3ExpertL2PrefetchKernel` does two independent things:

* Record (CTA 0, optional): copy a layer's routing on this rank, as the ``moe_sort`` outputs
  describe it, into a persistent *routing record* (``int32 [RECORD_INTS]``): ``rec[0] = n``
  (tiles), then per tile ``t < n`` the local expert ``rec[1 + 2 t]`` and its row count
  ``rec[2 + 2 t] = tile_idx_to_mn_limit[t] - 128 t`` (its tokens, as long as every expert fits in
  one 128-row tile, i.e. at most 128 tokens).
* Prefetch (all CTAs, optional): from another layer's routing record (written by an earlier launch,
  e.g. the previous decode step), select the ``budget`` entries with the most rows (ties: lower
  expert, then earlier tile; entries with no rows or an expert outside ``[0, num_local)`` are never
  selected) and issue ``cp.async.bulk.prefetch.L2`` for every per-expert byte range of their weights
  (``(base, stride, bytes)`` per range, e.g. the gate_up and down weights and their scale factors),
  in pieces of ``chunk`` bytes interleaved over the warps of the grid.

The prefetches are hints: nothing waits for them and no result depends on them, so a stale record
(another step, another batch size) only changes which bytes are prefetched; selected experts are
always valid indices. On SM100/SM103 a line brought in by a bulk L2 prefetch stays in L2 until its
first read (other traffic, whatever its eviction hint, does not evict it; only newer prefetched lines
do, least recently prefetched first), and a read with an evict-first hint, as the routed GEMMs'
weight loads carry, releases it. 32 one-warp CTAs with 32 KB pieces land the data at about 5 TB/s;
much faster issue (e.g. 128 CTAs of 4-8 KB pieces) makes the L2 drop part of the requests.

Launch it on a side stream behind an event, without the PDL attribute: its incoming dependencies
stay full ones (the moe_sort outputs it records are complete) and it never holds back a programmatic
dependent. Each CTA is one warp with no shared memory, so it fits beside any resident kernel. It
triggers its programmatic dependents at once: a kernel that joins its stream reads nothing it
writes within the same launch sequence (the record is read by a later launch of this kernel).
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Largest expert budget (one selected expert per lane of a warp).
MAX_BUDGET = 32
#: Largest number of local experts and of record entries (eight slots per lane).
MAX_LOCAL = 256
#: ``int32`` elements of a routing record: the entry count, then (expert, rows) per entry.
RECORD_INTS = 1 + 2 * MAX_LOCAL
#: Largest number of per-expert byte ranges.
MAX_RANGES = 4
#: Rows of a ``moe_sort`` tile.
TILE = 128
#: Default bytes per prefetch instruction.
DEFAULT_CHUNK = 32 << 10
#: Default CTAs (one warp each).
DEFAULT_CTAS = 32
_THREADS = 32
_SLOTS = MAX_LOCAL // _THREADS


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


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _ld_s32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.relaxed.gpu.s32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_s32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.relaxed.gpu.s32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _clamp(value, hi, *, loc=None, ip=None):
    """``value`` clamped to ``[0, hi]``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip), _iv(hi, loc=loc, ip=ip)],
            "{\n.reg .b32 v;\nmax.s32 v, $1, 0;\nmin.s32 $0, v, $2;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _entry_key(rows, expert, entry, num_local, *, loc=None, ip=None):
    """Selection key of record entry ``entry`` (< 256): ``(min(rows, 255) << 16) |
    ((255 - expert) << 8) | (255 - entry)`` if ``rows > 0`` and ``0 <= expert < num_local``, else 0.
    Keys of distinct entries differ; a larger key wins (more rows, then lower expert, then earlier
    entry)."""
    return Int32(
        _asm(
            T.i32(),
            [
                _iv(rows, loc=loc, ip=ip),
                _iv(expert, loc=loc, ip=ip),
                _iv(entry, loc=loc, ip=ip),
                _iv(num_local, loc=loc, ip=ip),
            ],
            "{\n.reg .pred p, q;\n.reg .b32 r, k, x;\n"
            "min.s32 r, $1, 255;\n"
            "shl.b32 k, r, 16;\n"
            "sub.s32 x, 255, $2;\n"
            "and.b32 x, x, 255;\n"
            "shl.b32 x, x, 8;\n"
            "or.b32 k, k, x;\n"
            "sub.s32 x, 255, $3;\n"
            "and.b32 x, x, 255;\n"
            "or.b32 k, k, x;\n"
            "setp.gt.s32 p, $1, 0;\n"
            "setp.ge.s32 q, $2, 0;\n"
            "and.pred p, p, q;\n"
            "setp.lt.s32 q, $2, $4;\n"
            "and.pred p, p, q;\n"
            "selp.b32 $0, k, 0, p;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _masked_max(best, key, removed, slot, *, loc=None, ip=None):
    """``max(best, key)`` unless bit ``slot`` of ``removed`` is set (then ``best``)."""
    return Int32(
        _asm(
            T.i32(),
            [
                _iv(best, loc=loc, ip=ip),
                _iv(key, loc=loc, ip=ip),
                _iv(removed, loc=loc, ip=ip),
                _iv(slot, loc=loc, ip=ip),
            ],
            "{\n.reg .b32 b;\n.reg .pred p;\n"
            "bfe.u32 b, $3, $4, 1;\n"
            "setp.eq.u32 p, b, 0;\n"
            "max.s32 b, $1, $2;\n"
            "selp.b32 $0, b, $1, p;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _redux_max_s32(value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "redux.sync.max.s32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _count_nonneg(value, *, loc=None, ip=None):
    """Number of lanes of the warp whose ``value`` is non-negative."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b32 v;\n"
            "setp.ge.s32 p, $1, 0;\n"
            "selp.b32 v, 1, 0, p;\n"
            "redux.sync.add.u32 $0, v, 0xffffffff;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _winner_expert(top, *, loc=None, ip=None):
    """Expert of a winning key, or -1 for key 0 (nothing left to select)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(top, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b32 e;\n"
            "bfe.u32 e, $1, 8, 8;\n"
            "sub.s32 e, 255, e;\n"
            "setp.ne.s32 p, $1, 0;\n"
            "selp.b32 $0, e, -1, p;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx(value, src_lane, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip), _iv(src_lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bulk_prefetch_l2(addr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(nbytes, loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


class K3ExpertL2PrefetchKernel:
    """``ctas`` one-warp CTAs. With ``record``: CTA 0 writes the routing record of the
    ``moe_sort`` outputs (``tile_idx_to_group_idx``, ``tile_idx_to_mn_limit``,
    ``num_non_exiting_tiles``). With ``range_bytes``: every CTA selects up to ``budget`` entries of
    the source record and prefetches ``range_bytes[r]`` bytes at ``base_r + expert * stride_r`` for
    every range ``r`` of every selected expert, in pieces of ``chunk`` bytes (bases, strides and
    sizes are multiples of 16). ``sel_out`` (optional) receives the selected experts (CTA 0, one per
    slot of ``budget``, -1 past the last)."""

    def __init__(
        self,
        num_local: int,
        range_bytes: Sequence[int] = (),
        budget: int = 0,
        record: bool = False,
        ctas: int = DEFAULT_CTAS,
        chunk: int = DEFAULT_CHUNK,
    ):
        if not 1 <= num_local <= MAX_LOCAL:
            raise ValueError(f"num_local={num_local} must be in [1, {MAX_LOCAL}]")
        if len(range_bytes) > MAX_RANGES or any(n <= 0 or n % 16 for n in range_bytes):
            raise ValueError(f"range_bytes={tuple(range_bytes)}")
        if range_bytes and not 1 <= budget <= MAX_BUDGET:
            raise ValueError(f"budget={budget} must be in [1, {MAX_BUDGET}]")
        if not range_bytes and not record:
            raise ValueError("nothing to do: no ranges and no record")
        if ctas < 1 or chunk < 16 or chunk % 16:
            raise ValueError(f"ctas={ctas} chunk={chunk}")
        self.num_local = num_local
        self.range_bytes = tuple(int(n) for n in range_bytes)
        self.budget = budget
        self.record = record
        self.ctas = ctas
        self.chunk = chunk
        self.pieces = tuple((n + chunk - 1) // chunk for n in self.range_bytes)
        self.pieces_per_expert = sum(self.pieces)

    @cute.jit
    def __call__(
        self,
        rec_out: Int64,
        tile_expert: Int64,
        tile_limit: Int64,
        num_tiles: Int64,
        rec_in: Int64,
        sel_out: Int64,
        b0: Int64,
        s0: Int64,
        b1: Int64,
        s1: Int64,
        b2: Int64,
        s2: Int64,
        b3: Int64,
        s3: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            rec_out,
            tile_expert,
            tile_limit,
            num_tiles,
            rec_in,
            sel_out,
            b0,
            s0,
            b1,
            s1,
            b2,
            s2,
            b3,
            s3,
        ).launch(grid=(self.ctas, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        rec_out: Int64,
        tile_expert: Int64,
        tile_limit: Int64,
        num_tiles: Int64,
        rec_in: Int64,
        sel_out: Int64,
        b0: Int64,
        s0: Int64,
        b1: Int64,
        s1: Int64,
        b2: Int64,
        s2: Int64,
        b3: Int64,
        s3: Int64,
    ):
        lane, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        _griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.record):
            if cta == 0:
                n = _clamp(_ld_s32(num_tiles), MAX_LOCAL)
                for q in cutlass.range_constexpr(_SLOTS):
                    t = lane + q * _THREADS
                    if t < n:
                        expert = _ld_s32(tile_expert + Int64(t) * 4)
                        rows = _ld_s32(tile_limit + Int64(t) * 4) - t * TILE
                        _st_s32(rec_out + Int64(1 + 2 * t) * 4, expert)
                        _st_s32(rec_out + Int64(2 + 2 * t) * 4, rows)
                if lane == 0:
                    _st_s32(rec_out, n)
        if cutlass.const_expr(len(self.range_bytes) > 0):
            self.prefetch(lane, cta, rec_in, sel_out, b0, s0, b1, s1, b2, s2, b3, s3)

    @cute.jit
    def prefetch(
        self,
        lane: Int32,
        cta: Int32,
        rec_in: Int64,
        sel_out: Int64,
        b0: Int64,
        s0: Int64,
        b1: Int64,
        s1: Int64,
        b2: Int64,
        s2: Int64,
        b3: Int64,
        s3: Int64,
    ):
        bases = (b0, b1, b2, b3)
        strides = (s0, s1, s2, s3)
        n = _clamp(_ld_s32(rec_in), MAX_LOCAL)
        # Keys of this lane's entries (entry = slot * 32 + lane).
        keys = []
        for q in cutlass.range_constexpr(_SLOTS):
            t = lane + q * _THREADS
            rows = Int32(0)
            expert = Int32(-1)
            if t < n:
                expert = _ld_s32(rec_in + Int64(1 + 2 * t) * 4)
                rows = _ld_s32(rec_in + Int64(2 + 2 * t) * 4)
            keys.append(_entry_key(rows, expert, t, self.num_local))
        # budget rounds of a warp arg-max; lane r keeps the r-th selected expert (-1: none).
        removed = Int32(0)
        mine = Int32(-1)
        for r in cutlass.range_constexpr(self.budget):
            best = Int32(0)
            for q in cutlass.range_constexpr(_SLOTS):
                best = _masked_max(best, keys[q], removed, q)
            top = _redux_max_s32(best)
            winner = _winner_expert(top)
            entry = 255 - (top & 255)
            if winner >= 0:
                if lane == (entry & (_THREADS - 1)):
                    removed = removed | (Int32(1) << (entry >> 5))
            if lane == r:
                mine = winner
        if cta == 0:
            if lane < self.budget:
                if sel_out != Int64(0):
                    _st_s32(sel_out + Int64(lane) * 4, mine)
        num_sel = _count_nonneg(mine)
        # Pieces of the selected experts, interleaved over the warps of the grid; the loop bounds
        # are uniform over the warp, so every lane takes part in the shuffle.
        per_expert = self.pieces_per_expert
        total = num_sel * per_expert
        for first in cutlass.range(cta * _THREADS, total, self.ctas * _THREADS, unroll=1):
            piece = first + lane
            slot = piece // per_expert
            src = slot
            if src > _THREADS - 1:
                src = Int32(_THREADS - 1)
            expert = _shfl_idx(mine, src)
            if piece < total:
                within = piece - slot * per_expert
                lo = 0
                for r in cutlass.range_constexpr(len(self.range_bytes)):
                    hi = lo + self.pieces[r]
                    if within >= lo:
                        if within < hi:
                            off = Int64(within - lo) * self.chunk
                            size = Int64(self.range_bytes[r]) - off
                            if size > self.chunk:
                                size = Int64(self.chunk)
                            _bulk_prefetch_l2(
                                bases[r] + Int64(expert) * strides[r] + off, Int32(size)
                            )
                    lo = hi


@functools.cache
def _compiled(
    num_local: int,
    range_bytes: Tuple[int, ...],
    budget: int,
    record: bool,
    ctas: int,
    chunk: int,
):
    z = Int64(0)
    return cute.compile(
        K3ExpertL2PrefetchKernel(num_local, range_bytes, budget, record, ctas, chunk),
        *([z] * (6 + 2 * MAX_RANGES)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def new_record(device: torch.device) -> torch.Tensor:
    """An empty routing record (no entries)."""
    return torch.zeros(RECORD_INTS, dtype=torch.int32, device=device)


def expert_ranges(tensors: Sequence[torch.Tensor], num_local: int) -> list[tuple[int, int, int]]:
    """``(base, stride, bytes)`` of every expert's slice of each tensor whose leading dimension is
    the ``num_local`` local experts (each slice dense and 16-byte aligned, at a constant stride)."""
    out = []
    for t in tensors:
        if t.dim() < 2 or t.shape[0] != num_local or t.numel() == 0:
            raise ValueError(f"tensor {tuple(t.shape)} does not lead with {num_local} experts")
        per = t[0]
        span = (1 + sum((n - 1) * s for n, s in zip(per.shape, per.stride()))) * t.element_size()
        if span != per.numel() * t.element_size() or any(s < 0 for s in t.stride()):
            raise ValueError("expert slices must be dense")
        stride = t.stride(0) * t.element_size()
        if stride < span or t.data_ptr() % 16 or stride % 16 or span % 16:
            raise ValueError(
                f"expert slices must be 16-byte aligned (stride {stride}, bytes {span})"
            )
        out.append((t.data_ptr(), stride, span))
    return out


def run_k3_expert_l2_prefetch(
    *,
    num_local: int,
    record_out: Optional[torch.Tensor] = None,
    sort_outputs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    record_in: Optional[torch.Tensor] = None,
    ranges: Sequence[tuple[int, int, int]] = (),
    budget: int = 0,
    sel_out: Optional[torch.Tensor] = None,
    ctas: int = DEFAULT_CTAS,
    chunk: int = DEFAULT_CHUNK,
) -> None:
    """On the current stream: record ``sort_outputs`` (``(tile_idx_to_group_idx,
    tile_idx_to_mn_limit, num_non_exiting_tiles)``, ``int32``) into ``record_out``, and/or prefetch
    into L2 the weight ``ranges`` (:func:`expert_ranges`) of the ``budget`` experts with the most
    rows in ``record_in``. Records are ``int32 [RECORD_INTS]`` (:func:`new_record`)."""
    record = record_out is not None
    prefetch = record_in is not None and len(ranges) > 0 and budget > 0
    if not record and not prefetch:
        return
    for rec in (record_out, record_in):
        if rec is not None and (
            rec.dtype != torch.int32 or rec.numel() < RECORD_INTS or not rec.is_cuda
        ):
            raise ValueError(f"routing records are CUDA int32[{RECORD_INTS}]")
    if record and (sort_outputs is None or any(t.dtype != torch.int32 for t in sort_outputs)):
        raise ValueError("recording needs the int32 moe_sort outputs")
    if len(ranges) > MAX_RANGES:
        raise ValueError(f"at most {MAX_RANGES} ranges, got {len(ranges)}")
    if sel_out is not None and (sel_out.dtype != torch.int32 or sel_out.numel() < budget):
        raise ValueError("sel_out must be an int32 tensor of at least `budget` elements")
    z = Int64(0)
    args = []
    if prefetch:
        for base, stride, _ in ranges:
            args += [Int64(base), Int64(stride)]
    args += [z] * (2 * MAX_RANGES - len(args))
    sizes = tuple(int(n) for _, _, n in ranges) if prefetch else ()
    tile_expert, tile_limit, num_tiles = sort_outputs if record else (None, None, None)
    _compiled(num_local, sizes, budget if prefetch else 0, record, ctas, chunk)(
        Int64(record_out.data_ptr()) if record else z,
        Int64(tile_expert.data_ptr()) if record else z,
        Int64(tile_limit.data_ptr()) if record else z,
        Int64(num_tiles.data_ptr()) if record else z,
        Int64(record_in.data_ptr()) if prefetch else z,
        Int64(sel_out.data_ptr()) if prefetch and sel_out is not None else z,
        *args,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_CTAS",
    "MAX_BUDGET",
    "MAX_LOCAL",
    "MAX_RANGES",
    "RECORD_INTS",
    "K3ExpertL2PrefetchKernel",
    "expert_ranges",
    "new_record",
    "run_k3_expert_l2_prefetch",
]

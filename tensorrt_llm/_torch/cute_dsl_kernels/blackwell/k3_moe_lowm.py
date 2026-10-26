# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-batch NVFP4 routed-expert GEMMs of the Kimi-K3 MoE (SM100/SM103).

At decode batch sizes a rank of the routed MoE sees a few tens of touched experts with one to a
few tokens each, so both expert GEMMs only stream weights. These kernels take the place of the
regular CuteDSL gather (FC1) kernel for ``M <= 32`` tokens and produce bit-identical results.

FC1 (:class:`K3MoeFc1LowMKernel`): gate_up GEMM + SiTU + NVFP4 quantization,
``h[r, j] = SiTU(alpha[e] * gate[r, j], alpha[e] * up[r, j])`` with ``[up | gate] =
x[token(r)] @ W1[e]^T`` (block-scaled NVFP4 operands, FP32 accumulation) for every valid permuted
row ``r`` of the 128-row tile of expert ``e``, quantized to NVFP4 with the output scale
``global_sf`` exactly like ``BlockScaledContiguousGatherGroupedGemmKernel`` (same accumulation,
same epilogue instruction sequence), written in the same permuted layout (``c`` rows and the
128x4-blocked ``c_sf``) so the FC2 kernels consume it unchanged.

* Swap-AB: the expert weights are the MMA A operand (128 rows of W1 per MMA: 64 ``up`` rows and
  the matching 64 ``gate`` rows, the checkpoint's granularity-64 interleave) and *all* ``M`` tokens
  of the step are the MMA N operand (``n_tok`` = 8, 16 or 32, zero-padded by the TMA), so the
  token operand is loaded once per CTA and stays in shared memory; the accumulator columns of
  tokens not routed to the expert are ignored.
* Work unit: one 128-row block of one touched expert's W1 over the whole K (14 stages of 128x256
  NVFP4 + scale factors, 258 KB at Kimi-K3 shapes). The ``num_tiles * n / 128`` work units are
  split into contiguous ranges of equal size over the persistent CTAs, so every SM streams the
  same number of bytes (to within one work unit) through a deep TMA ring.
* Warp roles: warps 0-3 epilogue (TMEM -> registers -> transposed SMEM staging -> SiTU and
  quantization of 16 outputs of one row per thread -> 8-byte stores), warp 4 MMA, warp 5 weight
  TMA, warp 6 token operand (TMA) and token scale factors (linear -> 128x4-blocked SMEM layout).
* PDL: every warp waits on the previous grid (the tile metadata, the tokens and their scales come
  from it) and then triggers its dependents right away, so the FC2 kernel can become resident and
  stream its weights on SMs this grid leaves. Successors that read data before their own
  ``griddepcontrol.wait`` may only read data written before this grid's predecessor completed (as
  the tile metadata and the weights are for the FC2 kernels).
* Sharing SMs (``share_sm``, with a reduced ``smem_capacity``): FC1 then launches with the
  max-shared carveout, so each SM keeps the shared memory FC1 leaves free for a CTA of a kernel
  on another stream, and triggers its dependents only when a CTA starts its last work unit: a
  launched dependent grid whose CTAs wait for SM space holds back the CTAs of every other kernel.
  The carveout is passed as ``launch(preferred_smem_carveout=100)``, which CuteDSL's CUDA-dialect executor does
  not apply (the build keeps the runtime's carveout); ``k3_carveout.apply`` sets it on a build.
"""

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm, math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from .utils import fmin, griddepcontrol_launch_dependents, griddepcontrol_wait

#: L2 eviction hint for the streamed expert weights (CacheHintSm90::EVICT_FIRST).
_EVICT_FIRST = 0x12F0000000000000
#: Weight rows per MMA (UMMA M).
_MMA_M = 128
#: K extent of one pipeline stage (four 64-wide block-scaled MMAs).
_KBLK = 256
_SF_VEC = 16
#: Bytes of one 128-row x 256-K block of scale factors (four 512-byte 32x4x4 atoms).
_SF_KBLK_BYTES = 2048
#: Token tiles (UMMA N) supported by the kernels.
SUPPORTED_N_TOK = (8, 16, 32)
#: Largest number of tokens of the small-batch path.
MAX_TOKENS = 32

_EPI_WARPS = 4
_MMA_WARP = 4
_TMA_WARP = 5
_TOK_WARP = 6
_NUM_WARPS = 7
_EPI_BARRIER_ID = 1
#: FP32 padding per staged accumulator column (bank spread of the transposed reads).
_SACC_PAD = 4
_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")


def n_tok_for(num_tokens: int) -> int:
    """Smallest supported token tile holding ``num_tokens`` tokens."""
    for n_tok in SUPPORTED_N_TOK:
        if num_tokens <= n_tok:
            return n_tok
    raise ValueError(f"{num_tokens} tokens exceed the small-batch MoE path ({MAX_TOKENS})")


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
def _ldg_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


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
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _stg_v2_b32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def _stg_u64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _signal_add_release(addr, *, loc=None, ip=None):
    """GPU-scope release add of 1: every write that happens-before it (in particular the CTA's
    writes ordered by a preceding ``bar.sync``) is visible to an acquire that observes it."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], 1;",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atom_add_relaxed(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_relaxed_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prefetch_l2(addr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(nbytes, loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


#: Bytes per L2 prefetch instruction.
_PREFETCH_CHUNK = 8192


def _s2t_copy_and_partition(sSF: cute.Tensor, tSF: cute.Tensor, sf_dtype):
    """tcgen05.cp (SMEM -> TMEM) partitions of a scale-factor operand."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype)
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


@dsl_user_op
def _popc(value, *, loc=None, ip=None):
    """``popc.b32``."""
    return Int32(
        _asm(T.i32(), [_iv(value, loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _red_shared_or(addr, value, *, loc=None, ip=None):
    """``red.shared.or.b32`` at shared-memory address ``addr``."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


def _align4(v: int) -> int:
    return (v + 3) // 4 * 4


#: Most tiles one CTA's contiguous unit range may meet under self-sort (its scratch holds their metadata).
SELF_SORT_LOCAL_TILES = 8


class _SelfSortMixin:
    """Self-sort of FC1 / FC2 (``self_sort_local`` > 0): every CTA builds, from the step's top-k ids, the tile
    metadata ``moe_sort`` (or ``k3_moe_front``'s sort) would write, but only for the tiles its own unit range meets:
    the tiles are the touched local experts in increasing order, a tile's rows its tokens in increasing order (the
    front's sort, row for row). Shared-memory scratch: per local expert the mask of the tokens routed to it (at most
    32 tokens), the union of the touched experts (one bit per expert), and per local tile its expert, row count and
    ``token * topk + slot`` per row.

    `_ss_clear` zeroes the token masks before the grid dependency wait (it reads nothing); `_self_sort`, after it,
    takes three CTA barriers: each id ORs its token's bit into its expert's mask; one ballot per 32 experts gives the
    union; each id then knows its tile (the touched experts below it) and row (the tokens below it in the mask)."""

    #: `_unit_range`'s split: ``False`` contiguous ranges of equal size (to within one unit), ``True`` contiguous
    #: ranges of ``ceil(total / nctas)`` units (the CTAs past the end get none).
    ceil_chunk = False

    @cute.jit
    def _unit_range(self, total: Int32, bid, nctas):
        """This CTA's contiguous range ``[w_lo, w_hi)`` of the ``total`` work units (`ceil_chunk`)."""
        w_lo = Int32(0)
        w_hi = Int32(0)
        if cutlass.const_expr(self.ceil_chunk):
            per = (total + Int32(nctas) - Int32(1)) // Int32(nctas)
            w_lo = cutlass.min(Int32(bid) * per, total)
            w_hi = cutlass.min((Int32(bid) + Int32(1)) * per, total)
        else:
            w_lo = (Int32(bid) * total) // Int32(nctas)
            w_hi = ((Int32(bid) + 1) * total) // Int32(nctas)
        return w_lo, w_hi

    def _init_self_sort(self, n_tok: int, topk: int, self_sort_local: int, self_sort_offset: int):
        self.self_sort = self_sort_local > 0
        self.ss_local = self_sort_local
        self.ss_offset = self_sort_offset
        if self.self_sort and n_tok > 32:
            raise ValueError("self-sort: the token masks hold at most 32 tokens")
        # the scratch arrays are declared in every build (4 words each without self-sort)
        self.ss_words = max(1, -(-self_sort_local // 32)) if self.self_sort else 4
        self.ss_bits = 32 * self.ss_words if self.self_sort else 4
        self.ss_lt = SELF_SORT_LOCAL_TILES if self.self_sort else 4
        self.ss_perm = self.ss_lt * n_tok if self.self_sort else 4
        words = self.ss_bits + _align4(self.ss_words) + 2 * _align4(self.ss_lt) + self.ss_lt * n_tok
        # 5 arrays, each 16-byte aligned
        self.ss_bytes = 4 * words + 5 * 16 if self.self_sort else 0

    def _check_self_sort_grid(self, num_ctas: int):
        """The CTA ranges of ``num_ctas`` CTAs each meet at most `SELF_SORT_LOCAL_TILES` tiles."""
        if not self.self_sort:
            return
        tiles = min(self.n_tok * self.topk, self.ss_local)
        units = -(-tiles * self.nblk // num_ctas)
        need = -(-units // self.nblk) + 1
        if need > self.ss_lt:
            raise ValueError(
                f"self-sort: {num_ctas} CTAs over {tiles} x {self.nblk} units meet up to {need} tiles per CTA "
                f"(scratch holds {self.ss_lt})"
            )

    @cute.jit
    def _ss_tensors(self, storage):
        """The scratch's per-local-tile views: ``(expert, rows, expanded index of row r at lt * n_tok + r)``."""
        expert = storage.sSsExpert.get_tensor(cute.make_layout(_align4(self.ss_lt)))
        count = storage.sSsCount.get_tensor(cute.make_layout(_align4(self.ss_lt)))
        perm = storage.sSsPerm.get_tensor(cute.make_layout(self.ss_perm))
        return expert, count, perm

    @cute.jit
    def _ss_clear(self, storage, tidx, nthr: cutlass.Constexpr):
        """Zero the per-expert token masks (before a CTA barrier that precedes `_self_sort`)."""
        mask = storage.sSsBits.get_tensor(cute.make_layout(self.ss_bits))
        for i in cutlass.range_constexpr(-(-self.ss_bits // nthr)):
            j = tidx + Int32(i * nthr)
            if j < Int32(self.ss_bits):
                mask[j] = Int32(0)

    @cute.jit
    def _self_sort(
        self,
        storage,
        topk_ids_ptr: cute.Pointer,
        orig_m: Int32,
        tidx,
        bid,
        nctas,
        nthr: cutlass.Constexpr,
        trace_ptr: Optional[cute.Pointer] = None,
    ):
        """Run by all ``nthr`` threads of the CTA (full-CTA barriers), the token masks zeroed by `_ss_clear`.
        With ``trace`` and a ``trace_ptr`` (diagnostic): thread 0 stamps trace slots 12 (its ids loaded), 13 / 14
        (after barriers 1 / 2) and 15 (sort done).
        Returns the tile count; the local tiles are ``[w_lo // nblk, ...)`` for this CTA's unit range
        ``[w_lo, w_hi)`` of ``tile count * nblk`` units."""
        n_tok, K, W, NBLK = self.n_tok, self.topk, self.ss_words, self.nblk
        per = -(-(n_tok * K) // nthr)
        off = Int32(self.ss_offset)
        num_local = Int32(self.ss_local)
        topk_ids = cute.make_tensor(topk_ids_ptr, cute.make_layout(n_tok * K))
        mask = storage.sSsBits.get_tensor(cute.make_layout(self.ss_bits))
        union = storage.sSsUnion.get_tensor(cute.make_layout(_align4(W)))
        expert, count, perm = self._ss_tensors(storage)
        mask_base = storage.sSsBits.data_ptr().toint()
        n = orig_m * Int32(K)
        # 1. id j = token * K + slot: its token's bit into its local expert's mask (lx = -1: not local / no id).
        # All of this thread's ids are loaded before the first red: the red is a volatile asm, so a load after it
        # would wait for the previous one's round trip.
        es = []
        for i in cutlass.range_constexpr(per):
            j = tidx + Int32(i * nthr)
            e = Int32(-1)
            if j < n:
                e = topk_ids[j] - off
            es.append(e)
        lxs = []
        for i in cutlass.range_constexpr(per):
            j = tidx + Int32(i * nthr)
            e = es[i]
            lx = Int32(-1)
            if (e >= Int32(0)) & (e < num_local):
                lx = e
                _red_shared_or(mask_base + e * Int32(4), Int32(1) << (j // Int32(K)))
            lxs.append(lx)
        if cutlass.const_expr(self.trace and trace_ptr is not None):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 12)
        cute.arch.barrier()
        if cutlass.const_expr(self.trace and trace_ptr is not None):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 13)
        # 2. union bit lx = the expert is touched; warp w ballots experts [32 w, 32 w + 32)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        nwarps = nthr // 32
        for i in cutlass.range_constexpr(-(-W // nwarps)):
            w = warp + Int32(i * nwarps)
            if w < Int32(W):
                b = cute.arch.vote_ballot_sync(mask[w * Int32(32) + lane] != Int32(0))
                if lane == Int32(0):
                    union[w] = Int32(b)
        cute.arch.barrier()
        if cutlass.const_expr(self.trace and trace_ptr is not None):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 14)
        # 3. tiles: id j's tile = the touched experts below lx, its row = the tokens below j's in lx's mask
        u = []
        for w in cutlass.range_constexpr(W):
            u.append(union[w])
        ntiles = Int32(0)
        for w in cutlass.range_constexpr(W):
            ntiles = ntiles + _popc(u[w])
        total = ntiles * NBLK
        w_lo, w_hi = self._unit_range(total, bid, nctas)
        tile_lo = w_lo // NBLK
        tile_end = (w_hi + NBLK - 1) // NBLK
        for i in cutlass.range_constexpr(per):
            lx = lxs[i]
            if lx >= Int32(0):
                j = tidx + Int32(i * nthr)
                hi = lx >> 5
                tile = Int32(0)
                for w in cutlass.range_constexpr(W):
                    below = cutlass.select_(Int32(w) < hi, u[w], Int32(0))
                    below = cutlass.select_(
                        Int32(w) == hi, u[w] & ((Int32(1) << (lx & 31)) - Int32(1)), below
                    )
                    tile = tile + _popc(below)
                if (tile >= tile_lo) & (tile < tile_end):
                    t = j // Int32(K)
                    tok = mask[lx]
                    row = _popc(tok & ((Int32(1) << t) - Int32(1)))
                    lt = tile - tile_lo
                    perm[lt * Int32(n_tok) + row] = j
                    if row == Int32(0):
                        expert[lt] = lx
                        count[lt] = _popc(tok)
        cute.arch.barrier()
        if cutlass.const_expr(self.trace and trace_ptr is not None):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 15)
        return ntiles


class K3MoeFc1LowMKernel(_SelfSortMixin):
    """Swap-AB NVFP4 gate_up GEMM + SiTU + NVFP4 quantization for at most 32 tokens.

    :param n_tok: Token tile (MMA N): 8, 16 or 32; the step's token count must not exceed it.
    :param n: Rows of one expert's W1 (2 x intermediate, gate/up interleaved every 64 rows).
    :param k: Hidden size (W1 K).
    :param topk: Experts per token (expanded index = token * topk + slot).
    :param situ_beta: SiTU gate-side constant.
    :param situ_linear_beta: SiTU up-side constant.
    :param num_acc_stage: Accumulator stages in TMEM (MMA of the next work unit overlaps the
        epilogue of the previous one).
    :param max_ab_stage: Cap on the weight ring depth (the rest of shared memory sizes it).
    :param use_pdl: Launch with programmatic dependent launch and wait on the previous grid.
    :param smem_capacity: Shared memory one CTA may use (bytes); the weight ring gets what the
        other buffers leave. Below the device maximum, a CTA of a concurrent kernel can share the SM.
    :param share_sm: Launch with the max-shared carveout (requested only: see the module docstring) and trigger
        dependents at the start of each CTA's last work unit, for use with a reduced ``smem_capacity``.
    :param late_trigger: Trigger dependents at the start of each CTA's last work unit (``True``) or
        right after ``griddepcontrol.wait`` (``False``); ``None`` follows ``share_sm``.
    :param prefetch_experts: The weight TMA warp loads the experts of the first 32 tiles together
        with the tile count (one memory latency less before the first weight load).
    :param self_sort_local: 0: the tile metadata comes from the previous grid (``moe_sort``'s
        outputs). Else the number of local experts: every CTA sorts the step's top-k ids itself after
        its wait (`_SelfSortMixin`) and the four metadata arguments are not read.
    :param self_sort_offset: First local expert id (with ``self_sort_local``).
    :param presorted_layout: Diagnostic (moe_sort's metadata only): ``x_sf_ptr`` holds the token scales already in
        the MMA's SMEM layout (``nkb`` x 2048 bytes, `token_scales_to_smem`'s image), copied with one bulk copy;
        the token TMA and that copy are issued right after ``griddepcontrol.wait``, before the tile count is read.
    """

    def __init__(
        self,
        n_tok: int,
        n: int,
        k: int,
        topk: int,
        situ_beta: float,
        situ_linear_beta: float,
        num_acc_stage: int = 4,
        max_ab_stage: int = 16,
        use_pdl: bool = True,
        signal_tiles: bool = False,
        prefetch_tail: bool = False,
        trace: bool = False,
        smem_capacity: int = _SMEM_CAPACITY,
        share_sm: bool = False,
        late_trigger: Optional[bool] = None,
        prefetch_experts: bool = True,
        self_sort_local: int = 0,
        self_sort_offset: int = 0,
        presorted_layout: bool = False,
    ):
        assert n_tok in SUPPORTED_N_TOK, f"n_tok {n_tok} not in {SUPPORTED_N_TOK}"
        assert n % (2 * 64) == 0 and n % _MMA_M == 0, f"n {n} must be a multiple of 128"
        assert k % _KBLK == 0, f"k {k} must be a multiple of {_KBLK}"
        assert situ_beta > 0 and situ_linear_beta > 0
        self.n_tok = n_tok
        self.n = n
        self.k = k
        self.topk = topk
        self.nkb = k // _KBLK
        self.nblk = n // _MMA_M
        self.inter_cols = n // 2
        self.situ_beta = float(situ_beta)
        self.situ_linear_beta = float(situ_linear_beta)
        self.num_acc_stage = num_acc_stage
        self.use_pdl = use_pdl
        self.share_sm = share_sm
        self.late_trigger = share_sm if late_trigger is None else late_trigger
        # The weight TMA warp loads the experts of the first 32 tiles (lane i: tile i) together with
        # the tile count, so the first weight load waits for one memory latency instead of two.
        self.prefetch_experts = prefetch_experts
        # After the stores of a work unit, add 1 to tile_ready[tile] (GPU-scope release), so FC2
        # can start a tile once all n / 128 of its units are stored, without waiting for this grid.
        self.signal_tiles = signal_tiles
        # A CTA with one unit more than the others prefetches that last unit into L2 at its start,
        # so the extra unit streams from L2 instead of HBM at the end of the grid.
        self.prefetch_tail = prefetch_tail
        # development aid: per-CTA globaltimer stamps into trace_ptr[cta * 16 + event]
        self.trace = trace
        self.mma_tiler = (_MMA_M, n_tok, _KBLK)
        self.acc_cols = n_tok * num_acc_stage
        # SFA: 128 rows / 32 * 4 MMAs per stage; SFB: round_up(n_tok, 128) / 32 * 4
        self.sfa_cols = 16
        self.sfb_cols = 16
        cols = self.acc_cols + self.sfa_cols + self.sfb_cols
        self.tmem_cols = 32
        while self.tmem_cols < cols:
            self.tmem_cols *= 2
        self.threads = 32 * _NUM_WARPS
        self.a_stage_bytes = _MMA_M * _KBLK // 2 + _SF_KBLK_BYTES
        self.x_bytes = self.nkb * n_tok * _KBLK // 2
        self.sacc_row_floats = _MMA_M + _SACC_PAD
        self._init_self_sort(n_tok, topk, self_sort_local, self_sort_offset)
        assert not (self.self_sort and prefetch_tail), "self-sort: no tail prefetch"
        self.presorted_layout = presorted_layout
        assert not (self.self_sort and presorted_layout), (
            "presorted_layout: moe_sort's metadata only"
        )
        fixed = (
            self.x_bytes
            + self.nkb * _SF_KBLK_BYTES
            + n_tok * self.sacc_row_floats * 4
            + 4 * 1024  # barriers, TMEM holder and alignment slack
            + self.ss_bytes
        )
        smem_capacity = min(smem_capacity, _SMEM_CAPACITY)
        self.num_ab_stage = min(max_ab_stage, (smem_capacity - fixed) // self.a_stage_bytes)
        assert self.num_ab_stage >= 2, f"{smem_capacity} B of shared memory leave < 2 weight stages"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        w_sf: cute.Tensor,
        x_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        global_sf: cute.Tensor,
        orig_m: Int32,
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
        tile_ready_ptr: Optional[cute.Pointer] = None,
        trace_ptr: Optional[cute.Pointer] = None,
        topk_ids_ptr: Optional[cute.Pointer] = None,
    ):
        self._check_self_sort_grid(num_ctas)
        self.a_dtype = w.element_type
        w_raw = w.iterator
        w_sf_raw = w_sf.iterator
        self.b_dtype = x.element_type
        self.sf_dtype = w_sf.element_type
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        a_major = utils.LayoutEnum.from_tensor(w).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(x).mma_major_mode()

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            a_major,
            b_major,
            self.sf_dtype,
            _SF_VEC,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.nkb
        )
        sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, _SF_VEC, self.num_ab_stage
        )
        sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, _SF_VEC, self.nkb
        )

        w_sf = cute.make_tensor(
            w_sf.iterator, blockscaled_utils.tile_atom_to_shape_SF(w.shape, _SF_VEC)
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w_sf,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            x,
            cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        S = self.num_ab_stage
        NACC = self.num_acc_stage

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, S]
            a_empty: cute.struct.MemRange[cutlass.Int64, S]
            acc_full: cute.struct.MemRange[cutlass.Int64, NACC]
            acc_empty: cute.struct.MemRange[cutlass.Int64, NACC]
            tok_full: cutlass.Int64
            sfb_full: cutlass.Int64
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_layout_staged.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_layout_staged.outer)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_layout_staged)], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_layout_staged)], 1024
            ]
            sAcc: cute.struct.Align[
                cute.struct.MemRange[Float32, self.n_tok * self.sacc_row_floats], 16
            ]
            sSsBits: cute.struct.Align[cute.struct.MemRange[Int32, self.ss_bits], 16]
            sSsUnion: cute.struct.Align[cute.struct.MemRange[Int32, _align4(self.ss_words)], 16]
            sSsExpert: cute.struct.Align[cute.struct.MemRange[Int32, _align4(self.ss_lt)], 16]
            sSsCount: cute.struct.Align[cute.struct.MemRange[Int32, _align4(self.ss_lt)], 16]
            sSsPerm: cute.struct.Align[cute.struct.MemRange[Int32, self.ss_perm], 16]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_b,
            tma_tensor_b,
            x_sf_ptr,
            c_ptr,
            c_sf_ptr,
            alpha,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            global_sf,
            orig_m,
            w_raw,
            w_sf_raw,
            tile_ready_ptr,
            trace_ptr,
            topk_ids_ptr,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            cluster_layout_vmnk,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
            preferred_smem_carveout=100 if self.share_sm else None,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_nkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_mkl: cute.Tensor,
        x_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        global_sf: cute.Tensor,
        orig_m: Int32,
        w_raw: cute.Pointer,
        w_sf_raw: cute.Pointer,
        tile_ready_ptr: Optional[cute.Pointer],
        trace_ptr: Optional[cute.Pointer],
        topk_ids_ptr: Optional[cute.Pointer],
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        cluster_layout_vmnk: cute.Layout,
    ):
        S = self.num_ab_stage
        NACC = self.num_acc_stage
        NKB = self.nkb
        NBLK = self.nblk
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        a_full = storage.a_full.data_ptr()
        a_empty = storage.a_empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        tok_full = storage.tok_full
        sfb_full = storage.sfb_full
        tmem_ready = storage.tmem_ready
        tmem_free = storage.tmem_free
        tmem_holding_buf = storage.tmem_holding_buf

        if warp_idx == _TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(a_full + i, 1)
                    cute.arch.mbarrier_init(a_empty + i, 1)
                for i in cutlass.range_constexpr(NACC):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, _EPI_WARPS)
                cute.arch.mbarrier_init(tok_full, 1)
                cute.arch.mbarrier_init(sfb_full, 1 if self.presorted_layout else _EPI_WARPS)
                cute.arch.mbarrier_init(tmem_ready, 1)
                cute.arch.mbarrier_init(tmem_free, _EPI_WARPS)
        if cutlass.const_expr(self.self_sort):
            self._ss_clear(storage, tidx, self.threads)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 0)

        if cutlass.const_expr(self.use_pdl):
            # Tile metadata, tokens and token scales come from the previous grid.
            griddepcontrol_wait()
            if cutlass.const_expr(not self.late_trigger):
                griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 1)
        if cutlass.const_expr(self.presorted_layout):
            if warp_idx == _TOK_WARP:
                self._issue_tokens_and_scales(
                    tiled_mma,
                    tma_atom_b,
                    mB_mkl,
                    storage,
                    b_smem_layout_staged,
                    cluster_layout_vmnk,
                    x_sf_ptr,
                    tok_full,
                    sfb_full,
                )

        t2e_pre = Int32(0)
        total = Int32(0)
        if cutlass.const_expr(self.self_sort):
            total = (
                self._self_sort(
                    storage, topk_ids_ptr, orig_m, tidx, bid, nctas, self.threads, trace_ptr
                )
                * NBLK
            )
        else:
            if cutlass.const_expr(self.prefetch_experts):
                if warp_idx == _TMA_WARP:
                    t2e_pre = tile_idx_to_expert_idx[
                        cutlass.min(cute.arch.lane_idx(), cute.size(tile_idx_to_expert_idx) - 1)
                    ]
            total = Int32(num_non_exiting_tiles[0]) * NBLK
        # This CTA's contiguous range of work units (tile, 128-row block of W1).
        w_lo = (Int32(bid) * total) // Int32(nctas)
        w_hi = ((Int32(bid) + 1) * total) // Int32(nctas)
        num_units = w_hi - w_lo

        thr_mma = tiled_mma.get_slice(0)

        if warp_idx == _TMA_WARP:
            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            # (bM, bK, loopN, loopK, loopL)
            gA = cute.local_tile(
                mA_nkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gSFA = cute.local_tile(
                mSFA_nkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            tCgA = thr_mma.partition_A(gA)
            tCgSFA = thr_mma.partition_A(gSFA)
            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
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
            if cutlass.const_expr(self.prefetch_tail):
                if num_units > total // Int32(nctas):
                    self._prefetch_unit(w_raw, w_sf_raw, tile_idx_to_expert_idx, w_hi - 1)
            g = Int32(0)
            for j in cutlass.range(num_units, unroll=1):
                if cutlass.const_expr(self.use_pdl and self.late_trigger):
                    if j == num_units - 1:
                        griddepcontrol_launch_dependents()
                w_idx = w_lo + j
                tile = w_idx // NBLK
                blk = w_idx - tile * NBLK
                expert = Int32(0)
                if cutlass.const_expr(self.self_sort):
                    ss_expert = self._ss_tensors(storage)[0]
                    expert = ss_expert[tile - w_lo // NBLK]
                else:
                    if cutlass.const_expr(self.prefetch_experts):
                        if tile < 32:
                            expert = cute.arch.shuffle_sync(t2e_pre, tile)
                        else:
                            expert = tile_idx_to_expert_idx[tile]
                    else:
                        expert = tile_idx_to_expert_idx[tile]
                for kb in cutlass.range(NKB, unroll=1):
                    stage = g % S
                    if g >= S:
                        cute.arch.mbarrier_wait(a_empty + stage, ((g // S) & 1) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, self.a_stage_bytes)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, blk, kb, expert)],
                        tAsA[(None, stage)],
                        tma_bar_ptr=a_full + stage,
                        cache_policy=Int64(_EVICT_FIRST),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA[(None, blk, kb, expert)],
                        tAsSFA[(None, stage)],
                        tma_bar_ptr=a_full + stage,
                        cache_policy=Int64(_EVICT_FIRST),
                    )
                    if cutlass.const_expr(self.trace):
                        if g == 0:
                            with cute.arch.elect_one():
                                self._stamp(trace_ptr, bid, 2)
                    g += 1
            if cutlass.const_expr(self.trace):
                with cute.arch.elect_one():
                    self._stamp(trace_ptr, bid, 3)
        elif warp_idx == _TOK_WARP:
            if cutlass.const_expr(not self.presorted_layout):
                if num_units > 0:
                    sB = storage.sB.get_tensor(
                        b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
                    )
                    # (bN, bK, loopM, loopK, 1)
                    gB = cute.local_tile(
                        mB_mkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
                    )
                    tCgB = thr_mma.partition_B(gB)
                    b_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    )
                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_b,
                        0,
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(tok_full, self.x_bytes)
                    for kb in cutlass.range(NKB, unroll=1):
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, kb, 0)],
                            tBsB[(None, kb)],
                            tma_bar_ptr=tok_full,
                        )
        elif warp_idx == _MMA_WARP:
            cute.arch.alloc_tmem(self.tmem_cols, tmem_holding_buf)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_ready)
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_base = cute.make_tensor(
                acc_tmem_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, NACC)).layout
            )
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(acc_tmem_ptr + self.acc_cols, dtype=self.sf_dtype),
                blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    _SF_VEC,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                ),
            )
            tCtSFB = cute.make_tensor(
                cute.recast_ptr(acc_tmem_ptr + self.acc_cols + self.sfa_cols, dtype=self.sf_dtype),
                blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    _SF_VEC,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                ),
            )
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
            s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_copy_and_partition(sSFA, tCtSFA, self.sf_dtype)
            s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_copy_and_partition(sSFB, tCtSFB, self.sf_dtype)
            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kphases = cute.size(tCrA, mode=[2])

            if cutlass.const_expr(self.presorted_layout):
                # issued by every CTA: wait for the copies even without units
                cute.arch.mbarrier_wait(tok_full, 0)
                cute.arch.mbarrier_wait(sfb_full, 0)
            else:
                if num_units > 0:
                    cute.arch.mbarrier_wait(tok_full, 0)
                    cute.arch.mbarrier_wait(sfb_full, 0)
            if cutlass.const_expr(self.trace):
                with cute.arch.elect_one():
                    self._stamp(trace_ptr, bid, 5)
            g = Int32(0)
            for j in cutlass.range(num_units, unroll=1):
                acc_stage = j % NACC
                if j >= NACC:
                    cute.arch.mbarrier_wait(acc_empty + acc_stage, ((j // NACC) & 1) ^ 1)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range(NKB, unroll=1):
                    stage = g % S
                    cute.arch.mbarrier_wait(a_full + stage, (g // S) & 1)
                    if cutlass.const_expr(self.trace):
                        if g == 0:
                            with cute.arch.elect_one():
                                self._stamp(trace_ptr, bid, 6)
                    cute.copy(s2t_sfa, tCsSFA_s2t[(None, None, None, None, stage)], tCtSFA_s2t)
                    cute.copy(s2t_sfb, tCsSFB_s2t[(None, None, None, None, kb)], tCtSFB_s2t)
                    for kp in cutlass.range_constexpr(num_kphases):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc[(None, 0, 0)],
                            tCrA[(None, 0, kp, stage)],
                            tCrB[(None, 0, kp, kb)],
                            tCtAcc[(None, 0, 0)],
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(a_empty + stage)
                    g += 1
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + acc_stage)
            if cutlass.const_expr(self.trace):
                with cute.arch.elect_one():
                    self._stamp(trace_ptr, bid, 7)
            cute.arch.mbarrier_wait(tmem_free, 0)
            cute.arch.dealloc_tmem(acc_tmem_ptr, self.tmem_cols)
        elif warp_idx < _EPI_WARPS:
            if cutlass.const_expr(self.presorted_layout):
                if cutlass.const_expr(self.trace):
                    if tidx == 0:
                        cute.arch.mbarrier_wait(sfb_full, 0)
                        self._stamp(trace_ptr, bid, 4)
            else:
                if num_units > 0:
                    self.token_scales_to_smem(storage, x_sf_ptr, orig_m, tidx, sfb_full)
                    if cutlass.const_expr(self.trace):
                        if tidx == 0:
                            self._stamp(trace_ptr, bid, 4)
            self.epilogue(
                tiled_mma,
                storage,
                c_ptr,
                c_sf_ptr,
                alpha,
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                global_sf,
                tidx,
                w_lo,
                num_units,
                acc_full,
                acc_empty,
                tmem_ready,
                tmem_free,
                tmem_holding_buf,
                tile_ready_ptr,
                trace_ptr,
                bid,
            )

    @cute.jit
    def _prefetch_unit(self, w_raw, w_sf_raw, tile_idx_to_expert_idx, w_idx: Int32):
        """L2 prefetch of work unit ``w_idx``: 128 contiguous W1 rows and their contiguous
        128x4-blocked scale block, one chunk per lane."""
        tile = w_idx // self.nblk
        blk = w_idx - tile * self.nblk
        expert = tile_idx_to_expert_idx[tile]
        row_bytes = self.k // 2
        w_bytes = _MMA_M * row_bytes
        sf_bytes = _MMA_M * (self.k // _SF_VEC)
        w_chunks = (w_bytes + _PREFETCH_CHUNK - 1) // _PREFETCH_CHUNK
        sf_chunks = (sf_bytes + _PREFETCH_CHUNK - 1) // _PREFETCH_CHUNK
        w_addr = w_raw.toint() + (Int64(expert) * self.n + blk * _MMA_M) * row_bytes
        sf_addr = w_sf_raw.toint() + Int64(expert) * (self.n * (self.k // _SF_VEC)) + blk * sf_bytes
        lane = cute.arch.lane_idx()
        for c in cutlass.range(lane, w_chunks + sf_chunks, 32, unroll=1):
            if c < w_chunks:
                off = c * _PREFETCH_CHUNK
                _prefetch_l2(w_addr + off, cutlass.min(Int32(_PREFETCH_CHUNK), w_bytes - off))
            else:
                off = (c - w_chunks) * _PREFETCH_CHUNK
                _prefetch_l2(sf_addr + off, cutlass.min(Int32(_PREFETCH_CHUNK), sf_bytes - off))

    @cute.jit
    def _issue_tokens_and_scales(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_b: cute.CopyAtom,
        mB_mkl: cute.Tensor,
        storage,
        b_smem_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        x_sf_ptr: cute.Pointer,
        tok_full,
        sfb_full,
    ):
        """``presorted_layout``: the token TMA (``tok_full``) and one bulk copy of the token scales, already in
        the MMA's SMEM layout, into ``sSFB`` (``sfb_full``). Run by the token warp."""
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        gB = cute.local_tile(
            mB_mkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgB = tiled_mma.get_slice(0).partition_B(gB)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        sf_bytes = self.nkb * _SF_KBLK_BYTES
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sfb_full, sf_bytes)
            _bulk_g2s(storage.sSFB.data_ptr().toint(), x_sf_ptr.toint(), sf_bytes, sfb_full.toint())
            cute.arch.mbarrier_arrive_and_expect_tx(tok_full, self.x_bytes)
        for kb in cutlass.range(self.nkb, unroll=1):
            cute.copy(tma_atom_b, tBgB[(None, 0, kb, 0)], tBsB[(None, kb)], tma_bar_ptr=tok_full)

    @cute.jit
    def token_scales_to_smem(
        self, storage, x_sf_ptr: cute.Pointer, orig_m: Int32, tidx: Int32, sfb_full
    ):
        """Token scale factors, linear [orig_m, k / 16], into the MMA's SMEM layout: per 256-K
        block four 512-byte 32x4x4 atoms (row r, scale s at (s // 4) * 512 + (r % 32) * 16
        + (r // 32) * 4 + s % 4); rows past orig_m are zero. Every load is issued before the
        first store so the 128 epilogue threads pay one memory latency."""
        n_tok = self.n_tok
        nthreads = _EPI_WARPS * 32
        nwords = self.nkb * 4 * n_tok
        per_thread = (nwords + nthreads - 1) // nthreads
        sf_row_bytes = self.k // _SF_VEC
        x_sf_base = x_sf_ptr.toint()
        sfb_base = storage.sSFB.data_ptr().toint()
        vals = cute.make_rmem_tensor((per_thread,), Int32)
        for t in cutlass.range_constexpr(per_thread):
            i = tidx + t * nthreads
            r = i % n_tok
            rest = i // n_tok
            atom = rest % 4
            kb = rest // 4
            rr = cutlass.min(r, orig_m - 1)
            kb = cutlass.min(kb, self.nkb - 1)
            vals[t] = _ldg_u32(x_sf_base + Int64(rr * sf_row_bytes + kb * 16 + atom * 4))
        for t in cutlass.range_constexpr(per_thread):
            i = tidx + t * nthreads
            r = i % n_tok
            rest = i // n_tok
            atom = rest % 4
            kb = rest // 4
            if i < nwords:
                v = vals[t]
                if r >= orig_m:
                    v = Int32(0)
                _sts_u32(
                    sfb_base + kb * _SF_KBLK_BYTES + atom * 512 + (r % 32) * 16 + (r // 32) * 4,
                    v,
                )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(sfb_full)

    @cute.jit
    def _stamp(self, trace_ptr, bid, ev: int):
        _stg_u64(trace_ptr.toint() + Int64(bid * 16 + ev) * 8, _globaltimer())

    @cute.jit
    def _situ(self, up, gate, alpha_val, out):
        """SiTU of 16 (up, gate) pairs, the exact packed-f32x2 instruction sequence of
        ``BlockScaledContiguousGatherGroupedGemmKernel._apply_situ_epilogue``."""
        beta = self.situ_beta
        linear_beta = self.situ_linear_beta
        LOG2_E = Float32(1.4426950408889634)
        neg_log2e_pair = (-LOG2_E, -LOG2_E)
        one_pair = (Float32(1.0), Float32(1.0))
        inv_2beta = Float32(2.0 / beta)
        two_beta = Float32(2.0 * beta)
        neg_beta = Float32(-beta)
        inv_2lbeta = Float32(2.0 / linear_beta)
        two_lbeta = Float32(2.0 * linear_beta)
        neg_lbeta = Float32(-linear_beta)

        def _sigmoid(p0, p1):
            neg = cute.arch.mul_packed_f32x2((p0, p1), neg_log2e_pair)
            e = (
                cute.math.exp2(neg[0], fastmath=True),
                cute.math.exp2(neg[1], fastmath=True),
            )
            d = cute.arch.add_packed_f32x2(e, one_pair)
            return (cute.arch.rcp_approx(d[0]), cute.arch.rcp_approx(d[1]))

        alpha_pair = (Float32(alpha_val), Float32(alpha_val))
        for i in cutlass.range_constexpr(0, 16, 2):
            g = cute.arch.mul_packed_f32x2((gate[i], gate[i + 1]), alpha_pair)
            u = cute.arch.mul_packed_f32x2((up[i], up[i + 1]), alpha_pair)
            sigmoid_g = _sigmoid(g[0], g[1])
            gs = _sigmoid(*cute.arch.mul_packed_f32x2(g, (inv_2beta, inv_2beta)))
            tanh_g = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(gs, (two_beta, two_beta)), (neg_beta, neg_beta)
            )
            us = _sigmoid(*cute.arch.mul_packed_f32x2(u, (inv_2lbeta, inv_2lbeta)))
            tanh_u = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(us, (two_lbeta, two_lbeta)), (neg_lbeta, neg_lbeta)
            )
            situ_gate = cute.arch.mul_packed_f32x2(tanh_g, sigmoid_g)
            out_pair = cute.arch.mul_packed_f32x2(situ_gate, tanh_u)
            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        global_sf: cute.Tensor,
        tidx: Int32,
        w_lo: Int32,
        num_units: Int32,
        acc_full,
        acc_empty,
        tmem_ready,
        tmem_free,
        tmem_holding_buf,
        tile_ready_ptr: Optional[cute.Pointer],
        trace_ptr: Optional[cute.Pointer],
        bid: Int32,
    ):
        """Thread ``t`` drains TMEM lane ``t`` (W1 row ``t`` of the block) for all tokens into a
        token-major staging buffer, then computes 16 outputs ``[16 q, 16 q + 16)`` of the
        block's 64 for valid row ``t // 4`` (``q = t % 4``): SiTU, the 1x16 NVFP4 scale and
        the 8 quantized bytes, exactly like the regular kernel's epilogue."""
        NACC = self.num_acc_stage
        NBLK = self.nblk
        n_tok = self.n_tok
        cute.arch.mbarrier_wait(tmem_ready, 0)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_base = cute.make_tensor(
            acc_tmem_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, NACC)).layout
        )
        copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(n_tok)), Float32)
        # (128, n_tok, 1, 1, NACC)
        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], (_MMA_M, n_tok))
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((_MMA_M, n_tok)))
        rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)

        sacc_base = storage.sAcc.data_ptr().toint()
        col_bytes = self.sacc_row_floats * 4
        c_base = c_ptr.toint()
        c_sf_base = c_sf_ptr.toint()
        row_bytes = self.inter_cols // 2
        tile_sf_bytes = _MMA_M * (self.inter_cols // _SF_VEC)
        r = tidx >> 2
        q = tidx & 3
        norm_const = global_sf[0]
        rcp_limit = Float32(1.0 / 6.0)
        fp32_max = Float32(3.40282346638528859812e38)

        up = cute.make_rmem_tensor((16,), Float32)
        gate = cute.make_rmem_tensor((16,), Float32)
        tCompute = cute.make_rmem_tensor((16,), Float32)
        rSF = cute.make_rmem_tensor((2,), self.sf_dtype)
        rQ = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)

        prev_tile = Int32(-1)
        n_valid = Int32(0)
        alpha_val = Float32(0.0)
        tok = Int32(0)
        for j in cutlass.range(num_units, unroll=1):
            w_idx = w_lo + j
            tile = w_idx // NBLK
            blk = w_idx - tile * NBLK
            if tile != prev_tile:
                prev_tile = tile
                if cutlass.const_expr(self.self_sort):
                    ss_expert, ss_count, ss_perm = self._ss_tensors(storage)
                    lt = tile - w_lo // NBLK
                    n_valid = ss_count[lt]
                    alpha_val = alpha[ss_expert[lt]]
                    if r < n_valid:
                        tok = ss_perm[lt * n_tok + r] // self.topk
                else:
                    n_valid = cutlass.min(
                        cutlass.max(tile_idx_to_mn_limit[tile] - tile * _MMA_M, Int32(0)),
                        Int32(n_tok),
                    )
                    alpha_val = alpha[tile_idx_to_expert_idx[tile]]
                    if r < n_valid:
                        tok = permuted_idx_to_expanded_idx[tile * _MMA_M + r] // self.topk
            acc_stage = j % NACC
            cute.arch.mbarrier_wait(acc_full + acc_stage, (j // NACC) & 1)
            if cutlass.const_expr(self.trace):
                if tidx == 0:
                    if j == 0:
                        self._stamp(trace_ptr, bid, 8)
                    if j == num_units - 1:
                        self._stamp(trace_ptr, bid, 11)
            cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, 0, acc_stage)], rAcc)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(acc_empty + acc_stage)
            for i in cutlass.range_constexpr(cute.size(rAcc)):
                row, col = tTR_cAcc[i]
                _sts_f32(sacc_base + col * col_bytes + row * 4, rAcc[i])
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=_EPI_WARPS * 32)
            if r < n_valid:
                src = sacc_base + tok * col_bytes + q * 64
                for v in cutlass.range_constexpr(4):
                    u0, u1, u2, u3 = _lds_v4_f32(src + v * 16)
                    g0, g1, g2, g3 = _lds_v4_f32(src + 256 + v * 16)
                    up[4 * v] = u0
                    up[4 * v + 1] = u1
                    up[4 * v + 2] = u2
                    up[4 * v + 3] = u3
                    gate[4 * v] = g0
                    gate[4 * v + 1] = g1
                    gate[4 * v + 2] = g2
                    gate[4 * v + 3] = g3
                self._situ(up, gate, alpha_val, tCompute)

                # NVFP4 quantization of the 16 outputs (regular epilogue, vectorized_f32 path)
                frg = cute.logical_divide(tCompute, cute.make_layout(_SF_VEC))
                acc_frg = frg.load()
                abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
                amax = abs_acc_frg[None, 0].reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
                pv = cute.arch.mul_packed_f32x2((amax, Float32(0.0)), (rcp_limit, rcp_limit))
                pv = cute.arch.mul_packed_f32x2((pv[0], pv[1]), (norm_const, norm_const))
                rPv = cute.make_rmem_tensor((2,), Float32)
                rPv[0] = pv[0]
                rPv[1] = pv[1]
                rSF.store(rPv.load().to(self.sf_dtype))
                sf_f32 = rSF.load().to(Float32)
                acc_scale = cute.arch.mul_packed_f32x2(
                    (cute.arch.rcp_approx(sf_f32[0]), cute.arch.rcp_approx(sf_f32[1])),
                    (norm_const, norm_const),
                )
                scale0 = fmin(acc_scale[0], fp32_max, nan=True)
                for ei in cutlass.range_constexpr(0, 16, 2):
                    tCompute[ei], tCompute[ei + 1] = cute.arch.mul_packed_f32x2(
                        (tCompute[ei], tCompute[ei + 1]), (scale0, scale0)
                    )
                rQ.store(tCompute.load().to(cutlass.Float4E2M1FN))
                rQw = cute.recast_tensor(rQ, Int32)
                rSFb = cute.recast_tensor(rSF, cutlass.Uint8)
                grow = tile * _MMA_M + r
                _stg_v2_b32(c_base + Int64(grow) * row_bytes + blk * 32 + q * 8, rQw[0], rQw[1])
                _stg_u8(
                    c_sf_base
                    + Int64(tile) * tile_sf_bytes
                    + blk * 512
                    + (r % 32) * 16
                    + (r // 32) * 4
                    + q,
                    Int32(rSFb[0]),
                )
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=_EPI_WARPS * 32)
            if cutlass.const_expr(self.signal_tiles):
                # the barrier orders every thread's stores of this unit before the release
                if tidx == 0:
                    _signal_add_release(tile_ready_ptr.toint() + Int64(tile) * 4)
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 9)
                _stg_u64(trace_ptr.toint() + Int64(bid * 16 + 10) * 8, Int64(num_units))
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(tmem_free)

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        orig_m: Int32,
        m: Int32,
        l: Int32,  # noqa: E741
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
        tile_ready_ptr: Optional[cute.Pointer] = None,
        trace_ptr: Optional[cute.Pointer] = None,
        topk_ids_ptr: Optional[cute.Pointer] = None,
    ):
        """Same pointers as ``BlockScaledContiguousGatherGroupedGemmKernel.wrapper``
        (``a`` = the step's tokens [orig_m, k], ``a_sf`` their linear [orig_m, k / 16] scales,
        ``m`` = permuted rows). ``topk_ids_ptr``: int32 [orig_m, topk], read with
        ``self_sort_local`` only."""
        n, k = self.n, self.k
        num_tiles = m // _MMA_M
        x = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        w = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        w_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, k // (_SF_VEC * 4), l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
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
        global_sf = cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,)))
        self(
            x,
            w,
            w_sf,
            a_sf_ptr,
            c_ptr,
            c_sf_ptr,
            alpha,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            global_sf,
            orig_m,
            num_ctas,
            stream,
            tile_ready_ptr,
            trace_ptr,
            topk_ids_ptr,
        )


@dsl_user_op
def _red_add_v8_bf16_from_smem(dst_addr, src_smem_addr, *, loc=None, ip=None):
    """``dst[0:8] += src[0:8]``: 8 BF16 (16 bytes) read from shared memory, one vector red."""
    _asm(
        None,
        [Int64(dst_addr).ir_value(loc=loc, ip=ip), _iv(src_smem_addr, loc=loc, ip=ip)],
        "{\n.reg .b32 a, b, c, d;\nld.shared.v4.b32 {a, b, c, d}, [$1];\n"
        "red.global.add.noftz.v4.bf16x2 [$0], {a, b, c, d};\n}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_u16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.u16 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(dst_smem, src_gmem, nbytes, mbar, *, loc=None, ip=None):
    """``nbytes`` contiguous bytes global -> shared, crediting ``mbar``."""
    _asm(
        None,
        [
            _iv(dst_smem, loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            _iv(nbytes, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


class K3MoeFc2LowMKernel(_SelfSortMixin):
    """Swap-AB NVFP4 down GEMM + MoE finalize (weighted scatter-add) for at most 32 tokens.

    Computes, like ``Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel``,
    ``out[token(r), h] += bf16(alpha[e] * token_final_scale(r) * sum_k h[r, k] * W2[e][h, k])``
    for every valid permuted row ``r`` of the tile of expert ``e``. Each partial is bit-identical to
    the regular kernel's (same FP32 accumulation, same FP32 scale product, round-to-nearest BF16);
    partials are added with 16-byte ``red.global.add.noftz.v4.bf16x2`` in an unspecified order, as
    the regular kernel's bulk reduce-adds are.

    * Swap-AB: W2 rows are the MMA A operand (128 output columns per work unit), the first ``n_tok``
      permuted rows of the tile the N operand (TMA from the FC1 output, scale factors one bulk copy
      of the tile's 128x4-blocked block), loaded once per run of work units of the same tile.
    * Work unit: one 128-row block of one touched expert's W2 over the whole K (``k / 256`` stages);
      units are split into contiguous equal ranges over the persistent CTAs (expert-major, so a CTA
      sees one or two tiles).
    * PDL: the weight TMA warp and the MMA warp's weight waits do not depend on the previous grid
      (FC1): the tile metadata comes from the kernel before FC1, so with ``early_weights`` the CTA
      streams weights as soon as it is resident, and only the token-operand loads (and the MMAs
      and the epilogue behind them) wait for FC1 to complete.
    * Self-sort (``self_sort_local``): every CTA builds its tiles' metadata from the step's top-k
      ids at its start (`_SelfSortMixin`, the same tiles and rows as FC1's), before the early
      weights: the ids, like the metadata otherwise, come from the kernel before FC1.
      ``token_final_scales`` may then be the front's bf16 weights (widened exactly).
    * ``ceil_chunk``: every CTA takes ``ceil(units / CTAs)`` contiguous units (the last ones fewer or
      none) instead of ranges of equal size to within one unit, so no CTA carries one unit more than
      the others while most of the grid has finished (the same units, so the same partials).
    """

    def __init__(
        self,
        n_tok: int,
        n: int,
        k: int,
        topk: int,
        num_acc_stage: int = 4,
        max_ab_stage: int = 16,
        num_b_stage: int = 2,
        num_epi_wg: int = 2,
        use_pdl: bool = True,
        early_weights: bool = True,
        wait_tiles: bool = False,
        tile_units: int = 12,
        trace: bool = False,
        self_sort_local: int = 0,
        self_sort_offset: int = 0,
        ceil_chunk: bool = False,
    ):
        assert n_tok in SUPPORTED_N_TOK, f"n_tok {n_tok} not in {SUPPORTED_N_TOK}"
        assert n % _MMA_M == 0 and k % _KBLK == 0
        assert num_epi_wg in (1, 2, 3, 4)
        assert num_acc_stage >= num_epi_wg
        self.n_tok = n_tok
        self.n = n
        self.k = k
        self.topk = topk
        self.nkb = k // _KBLK
        self.nblk = n // _MMA_M
        self.num_acc_stage = num_acc_stage
        self.num_b_stage = num_b_stage
        # Epilogue warpgroups take turns over the work units (unit j -> warpgroup j % num_epi_wg).
        self.num_epi_wg = num_epi_wg
        self.epi_warps = 4 * num_epi_wg
        self.mma_warp = self.epi_warps
        self.tma_warp = self.epi_warps + 1
        self.tok_warp = self.epi_warps + 2
        self.use_pdl = use_pdl
        self.early_weights = early_weights and use_pdl
        # Token operand of a tile only after tile_ready[tile] >= tile_units (FC1's per-tile
        # release counters) instead of after FC1's grid completes. Every CTA that reads a tile then
        # adds 1; the reader that brings the count to tile_units + readers resets it to 0 for the
        # next FC1 / FC2 pair (the readers of a tile follow from the static unit ranges).
        self.wait_tiles = wait_tiles
        self.tile_units = tile_units
        self.trace = trace
        # _tile_readers counts a tile's readers from the equal-size ranges
        assert not (ceil_chunk and wait_tiles), "ceil_chunk: no wait_tiles"
        self.ceil_chunk = ceil_chunk
        self.mma_tiler = (_MMA_M, n_tok, _KBLK)
        self.acc_cols = n_tok * num_acc_stage
        self.sfa_cols = 16
        self.sfb_cols = 16
        cols = self.acc_cols + self.sfa_cols + self.sfb_cols
        self.tmem_cols = 32
        while self.tmem_cols < cols:
            self.tmem_cols *= 2
        self.threads = 32 * (self.epi_warps + 3)
        self.a_stage_bytes = _MMA_M * _KBLK // 2 + _SF_KBLK_BYTES
        # token operand of a tile: nkb boxes of n_tok rows x 256 K, plus the tile's scale block
        self.b_data_bytes = self.nkb * n_tok * _KBLK // 2
        self.b_sf_bytes = self.nkb * _SF_KBLK_BYTES
        self.out_row_bytes = _MMA_M * 2
        self._init_self_sort(n_tok, topk, self_sort_local, self_sort_offset)
        fixed = (
            num_b_stage * (self.b_data_bytes + self.b_sf_bytes)
            + 2 * num_epi_wg * n_tok * self.out_row_bytes
            + 2 * num_epi_wg * n_tok * 4
            + 4 * 1024
            + self.ss_bytes
        )
        self.num_ab_stage = min(max_ab_stage, (_SMEM_CAPACITY - fixed) // self.a_stage_bytes)
        assert self.num_ab_stage >= 2

    @cute.jit
    def __call__(
        self,
        h: cute.Tensor,
        w: cute.Tensor,
        w_sf: cute.Tensor,
        h_sf_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        alpha: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        token_final_scales: cute.Tensor,
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
        tile_ready_ptr: Optional[cute.Pointer] = None,
        trace_ptr: Optional[cute.Pointer] = None,
        topk_ids_ptr: Optional[cute.Pointer] = None,
    ):
        self._check_self_sort_grid(num_ctas)
        self.a_dtype = w.element_type
        self.b_dtype = h.element_type
        self.sf_dtype = w_sf.element_type
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        a_major = utils.LayoutEnum.from_tensor(w).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(h).mma_major_mode()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            a_major,
            b_major,
            self.sf_dtype,
            _SF_VEC,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        # token operand: nkb K blocks per B stage
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.nkb * self.num_b_stage
        )
        sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, _SF_VEC, self.num_ab_stage
        )
        sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, _SF_VEC, self.nkb * self.num_b_stage
        )
        w_sf = cute.make_tensor(
            w_sf.iterator, blockscaled_utils.tile_atom_to_shape_SF(w.shape, _SF_VEC)
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w_sf,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            h,
            cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        S = self.num_ab_stage
        NACC = self.num_acc_stage
        NB = self.num_b_stage

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, S]
            a_empty: cute.struct.MemRange[cutlass.Int64, S]
            b_full: cute.struct.MemRange[cutlass.Int64, NB]
            b_empty: cute.struct.MemRange[cutlass.Int64, NB]
            acc_full: cute.struct.MemRange[cutlass.Int64, NACC]
            acc_empty: cute.struct.MemRange[cutlass.Int64, NACC]
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_layout_staged.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_layout_staged.outer)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_layout_staged)], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_layout_staged)], 1024
            ]
            sOut: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, 2 * self.num_epi_wg * self.n_tok * _MMA_M],
                16,
            ]
            sTok: cute.struct.Align[cute.struct.MemRange[Int32, self.num_epi_wg * self.n_tok], 16]
            sScale: cute.struct.Align[
                cute.struct.MemRange[Float32, self.num_epi_wg * self.n_tok], 16
            ]
            sSsBits: cute.struct.Align[cute.struct.MemRange[Int32, self.ss_bits], 16]
            sSsUnion: cute.struct.Align[cute.struct.MemRange[Int32, _align4(self.ss_words)], 16]
            sSsExpert: cute.struct.Align[cute.struct.MemRange[Int32, _align4(self.ss_lt)], 16]
            sSsCount: cute.struct.Align[cute.struct.MemRange[Int32, _align4(self.ss_lt)], 16]
            sSsPerm: cute.struct.Align[cute.struct.MemRange[Int32, self.ss_perm], 16]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_b,
            tma_tensor_b,
            h_sf_ptr,
            out_ptr,
            alpha,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            token_final_scales,
            tile_ready_ptr,
            trace_ptr,
            topk_ids_ptr,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            cluster_layout_vmnk,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def _stamp(self, trace_ptr, bid, ev: int):
        _stg_u64(trace_ptr.toint() + Int64(bid * 16 + ev) * 8, _globaltimer())

    @cute.jit
    def _tile_readers(self, tile: Int32, total: Int32, nctas: Int32):
        """Number of CTAs whose unit range [b * total // nctas, (b + 1) * total // nctas) meets
        the units [tile * nblk, tile * nblk + nblk) of ``tile``."""
        u0 = tile * self.nblk
        u1 = u0 + self.nblk - 1
        # CTA owning unit u: the largest b with b * total // nctas <= u
        b0 = ((u0 + 1) * nctas + total - 1) // total - 1
        b1 = ((u1 + 1) * nctas + total - 1) // total - 1
        readers = Int32(0)
        for b in cutlass.range(b0, b1 + 1, unroll=1):
            if ((b + 1) * total) // nctas > (b * total) // nctas:
                readers += 1
        return readers

    @cute.jit
    def _wg_barrier(self, wg: Int32):
        """Named barrier of epilogue warpgroup ``wg``."""
        for i in cutlass.range_constexpr(self.num_epi_wg):
            if wg == i:
                cute.arch.barrier(barrier_id=_EPI_BARRIER_ID + i, number_of_threads=128)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_nkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_mkl: cute.Tensor,
        h_sf_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        alpha: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        token_final_scales: cute.Tensor,
        tile_ready_ptr: Optional[cute.Pointer],
        trace_ptr: Optional[cute.Pointer],
        topk_ids_ptr: Optional[cute.Pointer],
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        cluster_layout_vmnk: cute.Layout,
    ):
        S = self.num_ab_stage
        NACC = self.num_acc_stage
        NB = self.num_b_stage
        NKB = self.nkb
        NBLK = self.nblk
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        a_full = storage.a_full.data_ptr()
        a_empty = storage.a_empty.data_ptr()
        b_full = storage.b_full.data_ptr()
        b_empty = storage.b_empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        tmem_ready = storage.tmem_ready
        tmem_free = storage.tmem_free
        tmem_holding_buf = storage.tmem_holding_buf

        if warp_idx == self.tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(a_full + i, 1)
                    cute.arch.mbarrier_init(a_empty + i, 1)
                for i in cutlass.range_constexpr(NB):
                    cute.arch.mbarrier_init(b_full + i, 1)
                    cute.arch.mbarrier_init(b_empty + i, 1)
                for i in cutlass.range_constexpr(NACC):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, 4)
                cute.arch.mbarrier_init(tmem_ready, 1)
                cute.arch.mbarrier_init(tmem_free, self.epi_warps)
        if cutlass.const_expr(self.self_sort):
            self._ss_clear(storage, tidx, self.threads)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 0)

        if cutlass.const_expr(self.use_pdl and not self.early_weights and not self.wait_tiles):
            griddepcontrol_wait()

        # Tile metadata (or the ids it is sorted from) comes from the kernel before FC1 (complete once
        # FC1 is resident).
        total = Int32(0)
        if cutlass.const_expr(self.self_sort):
            orig_m = Int32(cute.size(token_final_scales, mode=[0]))
            total = (
                self._self_sort(storage, topk_ids_ptr, orig_m, tidx, bid, nctas, self.threads)
                * NBLK
            )
        else:
            total = Int32(num_non_exiting_tiles[0]) * NBLK
        w_lo, w_hi = self._unit_range(total, bid, nctas)
        num_units = w_hi - w_lo
        thr_mma = tiled_mma.get_slice(0)

        if warp_idx == self.tma_warp:
            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            gA = cute.local_tile(
                mA_nkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gSFA = cute.local_tile(
                mSFA_nkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            tCgA = thr_mma.partition_A(gA)
            tCgSFA = thr_mma.partition_A(gSFA)
            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
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
            g = Int32(0)
            for j in cutlass.range(num_units, unroll=1):
                w_idx = w_lo + j
                tile = w_idx // NBLK
                blk = w_idx - tile * NBLK
                expert = Int32(0)
                if cutlass.const_expr(self.self_sort):
                    ss_expert = self._ss_tensors(storage)[0]
                    expert = ss_expert[tile - w_lo // NBLK]
                else:
                    expert = tile_idx_to_expert_idx[tile]
                for kb in cutlass.range(NKB, unroll=1):
                    stage = g % S
                    if g >= S:
                        cute.arch.mbarrier_wait(a_empty + stage, ((g // S) & 1) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, self.a_stage_bytes)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, blk, kb, expert)],
                        tAsA[(None, stage)],
                        tma_bar_ptr=a_full + stage,
                        cache_policy=Int64(_EVICT_FIRST),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA[(None, blk, kb, expert)],
                        tAsSFA[(None, stage)],
                        tma_bar_ptr=a_full + stage,
                        cache_policy=Int64(_EVICT_FIRST),
                    )
                    if cutlass.const_expr(self.trace):
                        if g == 0:
                            with cute.arch.elect_one():
                                self._stamp(trace_ptr, bid, 2)
                    g += 1
            if cutlass.const_expr(self.trace):
                with cute.arch.elect_one():
                    self._stamp(trace_ptr, bid, 3)
        elif warp_idx == self.tok_warp:
            # Token operand of each run of units of one tile: FC1's output rows and scales.
            if cutlass.const_expr(self.early_weights and not self.wait_tiles):
                griddepcontrol_wait()
            if cutlass.const_expr(self.trace):
                with cute.arch.elect_one():
                    self._stamp(trace_ptr, bid, 1)
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            gB = cute.local_tile(
                mB_mkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            tCgB = thr_mma.partition_B(gB)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
            sfb_base = storage.sSFB.data_ptr().toint()
            h_sf_base = h_sf_ptr.toint()
            prev_tile = Int32(-1)
            run = Int32(0)
            for j in cutlass.range(num_units, unroll=1):
                tile = (w_lo + j) // NBLK
                if tile != prev_tile:
                    prev_tile = tile
                    bs = run % NB
                    if run >= NB:
                        cute.arch.mbarrier_wait(b_empty + bs, ((run // NB) & 1) ^ 1)
                    if cutlass.const_expr(self.wait_tiles):
                        # all FC1 units of the tile have stored their rows and scales
                        ready_addr = tile_ready_ptr.toint() + Int64(tile) * 4
                        ready = _ld_acquire_u32(ready_addr)
                        while ready < self.tile_units:
                            ready = _ld_acquire_u32(ready_addr)
                        cute.arch.fence_proxy("async.global")
                        with cute.arch.elect_one():
                            readers = self._tile_readers(tile, total, Int32(nctas))
                            if _atom_add_relaxed(ready_addr, 1) == self.tile_units + readers - 1:
                                _st_relaxed_u32(ready_addr, 0)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            b_full + bs, self.b_data_bytes + self.b_sf_bytes
                        )
                        # the tile's scale block: k / 64 x 512-byte atoms, contiguous
                        _bulk_g2s(
                            sfb_base + bs * self.b_sf_bytes,
                            h_sf_base + Int64(tile) * (_MMA_M * (self.k // _SF_VEC)),
                            self.b_sf_bytes,
                            (b_full + bs).toint(),
                        )
                    # rows [128 tile, 128 tile + n_tok) of the permuted FC1 output
                    for kb in cutlass.range(NKB, unroll=1):
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, tile * (_MMA_M // self.n_tok), kb, 0)],
                            tBsB[(None, bs * NKB + kb)],
                            tma_bar_ptr=b_full + bs,
                        )
                    run += 1
        elif warp_idx == self.mma_warp:
            cute.arch.alloc_tmem(self.tmem_cols, tmem_holding_buf)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_ready)
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_base = cute.make_tensor(
                acc_tmem_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, NACC)).layout
            )
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(acc_tmem_ptr + self.acc_cols, dtype=self.sf_dtype),
                blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    _SF_VEC,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                ),
            )
            tCtSFB = cute.make_tensor(
                cute.recast_ptr(acc_tmem_ptr + self.acc_cols + self.sfa_cols, dtype=self.sf_dtype),
                blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    _SF_VEC,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                ),
            )
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
            s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_copy_and_partition(sSFA, tCtSFA, self.sf_dtype)
            s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_copy_and_partition(sSFB, tCtSFB, self.sf_dtype)
            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kphases = cute.size(tCrA, mode=[2])

            g = Int32(0)
            prev_tile = Int32(-1)
            run = Int32(0)
            for j in cutlass.range(num_units, unroll=1):
                tile = (w_lo + j) // NBLK
                if tile != prev_tile:
                    if run > 0:
                        # the previous run's token operand is no longer read
                        with cute.arch.elect_one():
                            tcgen05.commit(b_empty + (run - 1) % NB)
                    prev_tile = tile
                    cute.arch.mbarrier_wait(b_full + run % NB, (run // NB) & 1)
                    run += 1
                bs = (run - 1) % NB
                acc_stage = j % NACC
                if j >= NACC:
                    cute.arch.mbarrier_wait(acc_empty + acc_stage, ((j // NACC) & 1) ^ 1)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range(NKB, unroll=1):
                    stage = g % S
                    cute.arch.mbarrier_wait(a_full + stage, (g // S) & 1)
                    if cutlass.const_expr(self.trace):
                        if g == 0:
                            with cute.arch.elect_one():
                                self._stamp(trace_ptr, bid, 6)
                    cute.copy(s2t_sfa, tCsSFA_s2t[(None, None, None, None, stage)], tCtSFA_s2t)
                    cute.copy(
                        s2t_sfb, tCsSFB_s2t[(None, None, None, None, bs * NKB + kb)], tCtSFB_s2t
                    )
                    for kp in cutlass.range_constexpr(num_kphases):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc[(None, 0, 0)],
                            tCrA[(None, 0, kp, stage)],
                            tCrB[(None, 0, kp, bs * NKB + kb)],
                            tCtAcc[(None, 0, 0)],
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(a_empty + stage)
                    g += 1
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + acc_stage)
            if cutlass.const_expr(self.trace):
                with cute.arch.elect_one():
                    self._stamp(trace_ptr, bid, 7)
            cute.arch.mbarrier_wait(tmem_free, 0)
            cute.arch.dealloc_tmem(acc_tmem_ptr, self.tmem_cols)
        elif warp_idx < self.epi_warps:
            self.epilogue(
                tiled_mma,
                storage,
                out_ptr,
                alpha,
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                token_final_scales,
                tidx,
                w_lo,
                num_units,
                acc_full,
                acc_empty,
                tmem_ready,
                tmem_free,
                tmem_holding_buf,
                trace_ptr,
                bid,
            )

        # dependents launch once every warp of every CTA is done (as the regular kernel)
        cute.arch.barrier()
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 12)
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        out_ptr: cute.Pointer,
        alpha: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        tidx: Int32,
        w_lo: Int32,
        num_units: Int32,
        acc_full,
        acc_empty,
        tmem_ready,
        tmem_free,
        tmem_holding_buf,
        trace_ptr: Optional[cute.Pointer],
        bid: Int32,
    ):
        """Warpgroup ``wg`` drains the units ``j`` with ``j % num_epi_wg == wg``. Thread ``t`` of the
        warpgroup owns output column ``128 * block + t`` (TMEM lane ``t``): it scales the
        accumulator of every valid token row to BF16 into the warpgroup's token-major staging
        buffer (two buffers, alternating), then the warpgroup red-adds 16-byte chunks of the staged
        rows to ``out``. One named barrier per unit (the alternation keeps the previous unit's reds
        ahead of the next write to a buffer)."""
        NACC = self.num_acc_stage
        NBLK = self.nblk
        NWG = self.num_epi_wg
        n_tok = self.n_tok
        cute.arch.mbarrier_wait(tmem_ready, 0)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_base = cute.make_tensor(
            acc_tmem_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, NACC)).layout
        )
        copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(n_tok)), Float32)
        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], (_MMA_M, n_tok))
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        epi_tidx = tidx % 128
        wg = tidx // 128
        thr_t2r = tiled_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((_MMA_M, n_tok)))
        rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)
        rBf = cute.make_rmem_tensor((2,), cutlass.BFloat16)

        out_base = out_ptr.toint()
        out_row_bytes = self.n * 2
        stage_bytes = n_tok * self.out_row_bytes
        sout_base = storage.sOut.data_ptr().toint() + wg * 2 * stage_bytes
        stok = storage.sTok.get_tensor(cute.make_layout((n_tok, NWG), stride=(1, n_tok)))
        sscale = storage.sScale.get_tensor(cute.make_layout((n_tok, NWG), stride=(1, n_tok)))
        chunks_per_row = _MMA_M // 8

        prev_tile = Int32(-1)
        n_valid = Int32(0)
        k = Int32(0)
        for j in cutlass.range(wg, num_units, NWG, unroll=1):
            w_idx = w_lo + j
            tile = w_idx // NBLK
            blk = w_idx - tile * NBLK
            buf = k % 2
            if tile != prev_tile:
                prev_tile = tile
                alpha_val = Float32(0.0)
                token = Int32(0)
                slot = Int32(0)
                if cutlass.const_expr(self.self_sort):
                    ss_expert, ss_count, ss_perm = self._ss_tensors(storage)
                    lt = tile - w_lo // NBLK
                    n_valid = ss_count[lt]
                    if epi_tidx < n_valid:
                        alpha_val = alpha[ss_expert[lt]]
                        expanded = ss_perm[lt * n_tok + epi_tidx]
                        token = expanded // self.topk
                        slot = expanded - token * self.topk
                        alpha_val = alpha_val * Float32(token_final_scales[(token, slot)])
                else:
                    n_valid = cutlass.min(
                        cutlass.max(tile_idx_to_mn_limit[tile] - tile * _MMA_M, Int32(0)),
                        Int32(n_tok),
                    )
                    if epi_tidx < n_valid:
                        alpha_val = alpha[tile_idx_to_expert_idx[tile]]
                        expanded = permuted_idx_to_expanded_idx[tile * _MMA_M + epi_tidx]
                        token = expanded // self.topk
                        slot = expanded - token * self.topk
                        alpha_val = alpha_val * Float32(token_final_scales[(token, slot)])
                # the previous unit's reds still read the metadata of the previous tile
                self._wg_barrier(wg)
                if epi_tidx < n_valid:
                    stok[epi_tidx, wg] = token
                    sscale[epi_tidx, wg] = alpha_val
                self._wg_barrier(wg)
            acc_stage = j % NACC
            cute.arch.mbarrier_wait(acc_full + acc_stage, (j // NACC) & 1)
            if cutlass.const_expr(self.trace):
                if tidx == 0:
                    if j == 0:
                        self._stamp(trace_ptr, bid, 8)
            cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, 0, acc_stage)], rAcc)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(acc_empty + acc_stage)
            sbuf = sout_base + buf * stage_bytes
            for i in cutlass.range_constexpr(cute.size(rAcc)):
                row, col = tTR_cAcc[i]
                if col < n_valid:
                    rBf[0] = (rAcc[i] * sscale[col, wg]).to(cutlass.BFloat16)
                    rBfw = cute.recast_tensor(rBf, cutlass.Int16)
                    _sts_u16(sbuf + col * self.out_row_bytes + row * 2, Int32(rBfw[0]))
            self._wg_barrier(wg)
            q = epi_tidx
            while q < n_valid * chunks_per_row:
                r = q // chunks_per_row
                cc = (q - r * chunks_per_row) * 8
                _red_add_v8_bf16_from_smem(
                    out_base + Int64(stok[r, wg]) * out_row_bytes + (blk * _MMA_M + cc) * 2,
                    sbuf + r * self.out_row_bytes + cc * 2,
                )
                q += 128
            k += 1
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 9)
                _stg_u64(trace_ptr.toint() + Int64(bid * 16 + 10) * 8, Int64(num_units))
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(tmem_free)

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
        m: Int32,
        l: Int32,  # noqa: E741
        num_tokens: Int32,
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
        tile_ready_ptr: Optional[cute.Pointer] = None,
        trace_ptr: Optional[cute.Pointer] = None,
        topk_ids_ptr: Optional[cute.Pointer] = None,
    ):
        """Same pointers as ``Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.wrapper``
        (``a`` = FC1's permuted output [m, k], ``a_sf`` its 128x4-blocked scales, ``c`` = the
        BF16 [num_tokens, n] output that is added to). ``topk_ids_ptr``: int32 [num_tokens, topk],
        read with ``self_sort_local`` only (``token_final_scales_ptr`` fp32 or bf16, widened exactly)."""
        n, k = self.n, self.k
        num_tiles = m // _MMA_M
        h = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        w = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        w_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, k // (_SF_VEC * 4), l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
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
            layout=cute.make_ordered_layout((num_tokens, self.topk), order=(1, 0)),
        )
        self(
            h,
            w,
            w_sf,
            a_sf_ptr,
            c_ptr,
            alpha,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            token_final_scales,
            num_ctas,
            stream,
            tile_ready_ptr,
            trace_ptr,
            topk_ids_ptr,
        )


__all__: Tuple[str, ...] = (
    "K3MoeFc1LowMKernel",
    "K3MoeFc2LowMKernel",
    "SUPPORTED_N_TOK",
    "MAX_TOKENS",
    "n_tok_for",
)

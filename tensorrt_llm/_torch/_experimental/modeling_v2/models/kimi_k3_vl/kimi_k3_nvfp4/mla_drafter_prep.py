# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA DSpark drafter's inputs of a decode step in two small launches, and the context path's side stream.

On a ``trtllm::k3_spec_accept`` step the stock worker (``DFlashWorker.prepare_1st_drafter_inputs``) and the stock
drafter forward (the block fixup, ``K3MLADrafterNorms._fixup``) build the drafter's inputs with about sixty torch
launches of one or a few elements each. `drafter_input_prep` computes all of them in one launch from the acceptance
(``accepted``, ``num_accepted``) and the worker's state, with the stock integer arithmetic:

* request ``b`` (batch row ``b``, slot ``s = slots[b]``, ``c = ctx_len[s]`` before the step, ``n = num_accepted[b]``):
  ``bonus = accepted[b, max(n - 1, 0)]``; ``qpos[b, j] = min(c + n, max_ctx) + j`` (``j < block``);
* the context store of the step's ``K + 1`` rows: ``col = max(min(c + j, counts[b] * page - 1), 0)`` (the RoPE
  position and the pool column), ``page = table[b, col // page]``, ``offset = col % page``, ``mask = j < n``;
* ``ctx_len[s] = min(c + n, max_ctx)``, every slot clamped to ``max_ctx``, ``ctx_len[dummy] = 0`` (the stock
  ``_advance_ctx_len``; all reads of ``ctx_len`` precede every write);
* ``num_ctx[b] = min(ctx_len[s], max(counts[b] * page - block, 0))`` (after the update), ``seq_lens[b] = K + 1``;
* the block's fixup (``_build_mla_block_fixup`` of ``num_ctx`` and the table rows): ``pages`` / ``offsets`` of
  ``num_ctx + j``, ``seq_lens_i32 = num_ctx + block`` and ``kv_bounds`` (it repeated ``block`` times), the pool slots
  ``pages * stride0 + offsets * stride3`` of ``k3_drafter_mla_a_post`` and the query RoPE rows ``freqs[qpos]``;
* the noise rows' embedding lookup: ``[bonus..., mask_id]`` masked to this rank's vocabulary range exactly as the stock
  ``pre_comm_embedding_ops`` (a row outside the range is +0), the input of the embedding's own all-reduce.

`noise_block` then writes the noise block ``[B, block, H]`` from the reduced rows (slot 0 the bonus row, every other
slot the mask row, or the drafter's trained mask row). Every value is an integer, a copy, or the same arithmetic on the
same operands as the stock path, so every output is bit for bit the stock one.

`CtxSide` is the side stream the context path (fc, its all-reduce + hidden_norm, the latent GEMM) runs on beside the
target LM head and the acceptance, with its fork and done events; the worker builds one at construction.
"""

from typing import Any, Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.multi_stream_utils import _record_stream

MAX_BATCH = 8
MAX_WIDTH = 16  # the most block positions or K + 1 store rows per request
FREQ_WIDTH = 64  # fp32 words of one query RoPE row (32 complex64 frequencies)


@triton.jit(
    do_not_specialize=["max_ctx", "num_slots", "dummy_slot", "vocab_start", "vocab_end", "mask_id"]
)
def _k3_drafter_input_prep(
    ACC,
    N,
    SLOT,
    CTX_LEN,
    COUNTS,
    TABLE,
    SEQ_LENS,
    FREQS,
    EMB,
    BONUS,
    QPOS,
    NUM_CTX,
    ST_POS,
    ST_PAGE,
    ST_OFF,
    ST_MASK,
    BLK_PAGES,
    BLK_OFF,
    BLK_SEQ,
    BLK_BOUNDS,
    BLK_SLOTS,
    BLK_FREQS,
    LOOKUP,
    max_ctx,
    num_slots,
    dummy_slot,
    vocab_start,
    vocab_end,
    mask_id,
    ACC_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    POOL_S0: tl.constexpr,
    POOL_S3: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    BLK: tl.constexpr,
    PAGE: tl.constexpr,
    H: tl.constexpr,
    H_TILE: tl.constexpr,
    PDL: tl.constexpr = False,
):
    # Program p < B looks up request p's bonus row, program B the mask row; program 0 also runs the bookkeeping.
    pid = tl.program_id(0)
    if PDL:
        # Every input is a previous kernel's output or state: nothing is read before the wait.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    # The embedding lookup (pre_comm_embedding_ops): in range, the row of this rank's shard; else +0.
    p = tl.minimum(pid, B - 1)
    n_p = tl.load(N + p).to(tl.int64)
    tok = tl.load(ACC + p * ACC_STRIDE + tl.maximum(n_p - 1, 0)).to(tl.int64)
    tok = tl.where(pid < B, tok, mask_id)
    in_range = (tok >= vocab_start) & (tok < vocab_end)
    row = tl.where(in_range, tok - vocab_start, 0)
    cols = tl.arange(0, H_TILE)
    for h0 in range(0, H, H_TILE):
        keep = h0 + cols < H
        w = tl.load(EMB + row * H + h0 + cols, mask=keep & in_range, other=0)
        tl.store(LOOKUP + pid * H + h0 + cols, w, mask=keep)

    if pid == 0:
        b = tl.arange(0, 8)
        lane = b < B
        # Reads: all of them before any write of ctx_len.
        n = tl.load(N + b, mask=lane, other=1).to(tl.int64)
        slot = tl.load(SLOT + b, mask=lane, other=0)
        c = tl.load(CTX_LEN + slot, mask=lane, other=0)
        cnt = tl.load(COUNTS + b, mask=lane, other=0)
        bonus = tl.load(ACC + b * ACC_STRIDE + tl.maximum(n - 1, 0), mask=lane, other=0).to(
            tl.int64
        )
        now = tl.minimum(c + n, max_ctx)
        tl.debug_barrier()
        # _advance_ctx_len: ctx_len[slots] += n, ctx_len.clamp_(max=max_ctx), ctx_len[dummy] = 0.
        tl.store(CTX_LEN + slot, now, mask=lane)
        tl.debug_barrier()
        s = tl.arange(0, 128)
        for s0 in range(0, num_slots, 128):
            sm = s0 + s < num_slots
            v = tl.load(CTX_LEN + s0 + s, mask=sm, other=0, volatile=True)
            tl.store(CTX_LEN + s0 + s, tl.minimum(v, max_ctx), mask=sm)
        tl.debug_barrier()
        one = tl.arange(0, 1)
        tl.store(CTX_LEN + dummy_slot + one, tl.zeros([1], dtype=tl.int64))
        tl.debug_barrier()
        after = tl.load(CTX_LEN + slot, mask=lane, other=0, volatile=True)
        num_ctx = tl.minimum(after, tl.maximum(cnt * PAGE - BLK, 0))
        seq = (num_ctx + BLK).to(tl.int32)
        tl.store(BONUS + b, bonus, mask=lane)
        tl.store(NUM_CTX + b, num_ctx, mask=lane)
        tl.store(SEQ_LENS + b, tl.full([8], K + 1, tl.int32), mask=lane)
        tl.store(BLK_SEQ + b, seq, mask=lane)

        j = tl.arange(0, 16)
        b2 = b[:, None]
        j2 = j[None, :]
        # The block's query positions.
        qm = lane[:, None] & (j2 < BLK)
        tl.store(QPOS + b2 * BLK + j2, now[:, None] + j2, mask=qm)
        # The context store of the K + 1 rows.
        sm2 = lane[:, None] & (j2 < K + 1)
        col = tl.maximum(tl.minimum(c[:, None] + j2, cnt[:, None] * PAGE - 1), 0)
        st_page = tl.load(TABLE + b2 * TABLE_STRIDE + col // PAGE, mask=sm2, other=0).to(tl.int64)
        st = b2 * (K + 1) + j2
        tl.store(ST_POS + st, col, mask=sm2)
        tl.store(ST_PAGE + st, st_page, mask=sm2)
        tl.store(ST_OFF + st, col % PAGE, mask=sm2)
        tl.store(ST_MASK + st, (j2 < n[:, None]).to(tl.float32), mask=sm2)
        # The block fixup and the pool slots of the block's own latents.
        pos = num_ctx[:, None] + j2
        blk_page = tl.load(TABLE + b2 * TABLE_STRIDE + pos // PAGE, mask=qm, other=0).to(tl.int64)
        blk_off = pos % PAGE
        q = b2 * BLK + j2
        tl.store(BLK_PAGES + q, blk_page, mask=qm)
        tl.store(BLK_OFF + q, blk_off, mask=qm)
        tl.store(BLK_BOUNDS + q, seq[:, None] + tl.zeros([8, 16], dtype=tl.int32), mask=qm)
        tl.store(BLK_SLOTS + q, blk_page * POOL_S0 + blk_off * POOL_S3, mask=qm)
        # The query RoPE rows freqs[qpos] (fp32 words of the complex64 table).
        f = tl.arange(0, 64)
        for jj in tl.static_range(BLK):
            src = FREQS + (now + jj)[:, None] * 64 + f[None, :]
            words = tl.load(src, mask=lane[:, None], other=0)
            tl.store(BLK_FREQS + (b2 * BLK + jj) * 64 + f[None, :], words, mask=lane[:, None])


@triton.jit
def _k3_noise_block(
    COMB,
    MASK_ROW,
    NOISE,
    B: tl.constexpr,
    BLK: tl.constexpr,
    H: tl.constexpr,
    H_TILE: tl.constexpr,
    TRAINED: tl.constexpr,
    PDL: tl.constexpr = False,
):
    # Program (b, j): noise[b, j] = j == 0 ? comb[b] : (trained mask row or comb[B]).
    b, j = tl.program_id(0), tl.program_id(1)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    cols = tl.arange(0, H_TILE)
    first = j == 0
    rest = j != 0
    src_row = tl.where(first, b, B)
    for h0 in range(0, H, H_TILE):
        keep = h0 + cols < H
        if TRAINED:
            from_comb = tl.load(COMB + src_row * H + h0 + cols, mask=keep & first, other=0)
            from_mask = tl.load(MASK_ROW + h0 + cols, mask=keep & rest, other=0)
            v = tl.where(first, from_comb, from_mask)
        else:
            v = tl.load(COMB + src_row * H + h0 + cols, mask=keep, other=0)
        tl.store(NOISE + (b * BLK + j) * H + h0 + cols, v, mask=keep)


class DrafterInputs(NamedTuple):
    """`drafter_input_prep`'s outputs (B requests, ``K + 1`` store rows each, a block of ``block`` positions)."""

    bonus: torch.Tensor  # int64 [B]: the anchor token (first_prev_tokens)
    query_positions: torch.Tensor  # int64 [B, block]
    num_ctx: torch.Tensor  # int64 [B]: the advertised context length (num_ctx_per_req)
    store_positions: torch.Tensor  # int64 [B (K + 1)]: the clamped column, also the RoPE position
    store_pages: torch.Tensor  # int64 [B (K + 1)]
    store_offsets: torch.Tensor  # int64 [B (K + 1)]
    store_mask: torch.Tensor  # fp32 [B (K + 1)]: 1 for an accepted row, else 0
    lookup: (
        torch.Tensor
    )  # [B + 1, H]: the bonus rows and the mask row before the embedding's all-reduce
    blk_pages: torch.Tensor  # int64 [B, block]
    blk_offsets: torch.Tensor  # int64 [B, block]
    blk_seq_lens: torch.Tensor  # int32 [B]
    blk_kv_bounds: torch.Tensor  # int32 [B block]
    blk_slots: torch.Tensor  # int64 [B block]: pool element offsets of the block's latents
    blk_freqs: torch.Tensor  # complex64 [B, block, 32]: freqs_cis[query_positions]


def supports(batch: int, drafts: int, block: int, hidden: int) -> bool:
    """Whether `drafter_input_prep` takes a step of these shapes."""
    return (
        0 < batch <= MAX_BATCH
        and 0 < drafts + 1 <= MAX_WIDTH
        and 0 < block <= MAX_WIDTH
        and hidden > 0
    )


def _h_tile(hidden: int) -> int:
    return min(triton.next_power_of_2(hidden), 1024)


def drafter_input_prep(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    counts: torch.Tensor,
    table: torch.Tensor,
    seq_lens: torch.Tensor,
    freqs_cis: torch.Tensor,
    embed_weight: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    mask_id: int,
    max_ctx: int,
    dummy_slot: int,
    page_size: int,
    block: int,
    pool_strides: tuple,
    pdl: bool = False,
) -> DrafterInputs:
    """One launch: the module docstring's values. Updates ``ctx_len`` (every slot) and ``seq_lens[:B]`` in place.

    ``accepted`` int32 [B, K + 1] (unit column stride) and ``num_accepted`` int32 [B]: the step's acceptance;
    ``slots`` int64 [B]: the requests' context slots; ``ctx_len`` int64 [slots]; ``counts`` int64 [>= B] and ``table``
    int32 [>= B, max_blocks]: the draft pool's block table rows of the requests (row ``b`` = batch row ``b``);
    ``seq_lens``: the requests' ``_seq_lens_cuda`` entries; ``freqs_cis``: the drafter's complex64 RoPE table
    ``[positions, 32]``; ``embed_weight``: this rank's shard ``[vocab_end - vocab_start, H]`` of the drafter's
    embedding; ``pool_strides``: the context pool's ``(stride(0), stride(3))``."""
    batch, width = accepted.shape
    drafts = width - 1
    hidden = embed_weight.shape[1]
    if not supports(batch, drafts, block, hidden):
        raise ValueError(
            f"k3_drafter_input_prep: {batch} requests of K + 1 = {width}, block {block}"
        )
    if (
        accepted.dtype != torch.int32
        or accepted.stride(1) != 1
        or num_accepted.dtype != torch.int32
        or num_accepted.numel() < batch
        or slots.dtype != torch.int64
        or slots.numel() != batch
        or not slots.is_contiguous()
        or ctx_len.dtype != torch.int64
        or not ctx_len.is_contiguous()
        or counts.dtype != torch.int64
        or table.dtype != torch.int32
        or table.stride(1) != 1
        or not seq_lens.is_contiguous()
        or seq_lens.numel() < batch
        or freqs_cis.dtype != torch.complex64
        or freqs_cis.dim() != 2
        or freqs_cis.shape[1] != FREQ_WIDTH // 2
        or not freqs_cis.is_contiguous()
        or embed_weight.element_size() != 2
        or not embed_weight.is_contiguous()
    ):
        raise ValueError("k3_drafter_input_prep: unexpected input dtypes or layouts")
    dev = accepted.device
    rows = batch * width
    i64 = dict(dtype=torch.int64, device=dev)
    bonus = torch.empty(batch, **i64)
    qpos = torch.empty(batch, block, **i64)
    num_ctx = torch.empty(batch, **i64)
    st_pos = torch.empty(rows, **i64)
    st_page = torch.empty(rows, **i64)
    st_off = torch.empty(rows, **i64)
    st_mask = torch.empty(rows, dtype=torch.float32, device=dev)
    lookup = torch.empty(batch + 1, hidden, dtype=embed_weight.dtype, device=dev)
    blk_pages = torch.empty(batch, block, **i64)
    blk_off = torch.empty(batch, block, **i64)
    blk_seq = torch.empty(batch, dtype=torch.int32, device=dev)
    blk_bounds = torch.empty(batch * block, dtype=torch.int32, device=dev)
    blk_slots = torch.empty(batch * block, **i64)
    blk_freqs = torch.empty(batch, block, 32, 2, dtype=torch.float32, device=dev)
    _k3_drafter_input_prep[(batch + 1,)](
        accepted,
        num_accepted,
        slots,
        ctx_len,
        counts,
        table,
        seq_lens,
        torch.view_as_real(freqs_cis).view(torch.int32),
        embed_weight.view(torch.int16),
        bonus,
        qpos,
        num_ctx,
        st_pos,
        st_page,
        st_off,
        st_mask,
        blk_pages,
        blk_off,
        blk_seq,
        blk_bounds,
        blk_slots,
        blk_freqs.view(torch.int32),
        lookup.view(torch.int16),
        int(max_ctx),
        ctx_len.numel(),
        int(dummy_slot),
        int(vocab_start),
        int(vocab_end),
        int(mask_id),
        accepted.stride(0),
        table.stride(0),
        int(pool_strides[0]),
        int(pool_strides[1]),
        batch,
        drafts,
        block,
        int(page_size),
        hidden,
        _h_tile(hidden),
        num_warps=4,
        **({"PDL": True, "launch_pdl": True} if pdl else {}),
    )
    return DrafterInputs(
        bonus,
        qpos,
        num_ctx,
        st_pos,
        st_page,
        st_off,
        st_mask,
        lookup,
        blk_pages,
        blk_off,
        blk_seq,
        blk_bounds,
        blk_slots,
        torch.view_as_complex(blk_freqs),
    )


def noise_block(
    reduced: torch.Tensor,
    block: int,
    trained_mask: Optional[torch.Tensor] = None,
    pdl: bool = False,
) -> torch.Tensor:
    """The drafter's noise block ``[B, block, H]`` from the reduced lookup rows ``reduced`` ``[B + 1, H]``: slot 0 of
    request ``b`` is ``reduced[b]``, every other slot ``trained_mask`` (``[H]``, the drafter's trained mask row) or,
    without one, ``reduced[B]`` (``dflash_noise_block_embedding``'s values)."""
    reduced = reduced.contiguous()
    batch, hidden = reduced.shape[0] - 1, reduced.shape[1]
    noise = torch.empty(batch, block, hidden, dtype=reduced.dtype, device=reduced.device)
    trained = trained_mask is not None
    mask_row = trained_mask.contiguous().view(-1) if trained else reduced
    if reduced.element_size() != 2 or (trained and mask_row.dtype != reduced.dtype):
        raise ValueError("k3_noise_block: 16-bit rows of one dtype")
    _k3_noise_block[(batch, block)](
        reduced.view(torch.int16),
        mask_row.view(torch.int16),
        noise.view(torch.int16),
        batch,
        block,
        hidden,
        _h_tile(hidden),
        trained,
        num_warps=4,
        **({"PDL": True, "launch_pdl": True} if pdl else {}),
    )
    return noise


class CtxSide:
    """A side stream and its fork / done events (created once, by the worker's constructor).

    `fork` records the fork event on the calling stream, enqueues ``fn`` on the side stream after it and records the
    done event; `join` makes the calling stream wait for the done event. Under CUDA-graph capture the side stream joins
    the capture through the fork event and is joined back by `join`, so the graph holds the two branches and the edge
    between them. Outside a capture `join` records the calling stream on ``fn``'s returned tensors, so their memory
    (allocated in the side stream's order) outlives the caller's reads (``maybe_execute_in_parallel``'s rule)."""

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream()
        self.fork_event = torch.cuda.Event()
        self.done_event = torch.cuda.Event()

    def fork(self, fn: Callable[[], Any]) -> Any:
        self.fork_event.record()
        with torch.cuda.stream(self.stream):
            self.fork_event.wait()
            out = fn()
            self.done_event.record()
        return out

    def join(self, out: Any) -> None:
        self.done_event.wait()
        if not torch.cuda.is_current_stream_capturing():
            _record_stream(out, torch.cuda.current_stream())

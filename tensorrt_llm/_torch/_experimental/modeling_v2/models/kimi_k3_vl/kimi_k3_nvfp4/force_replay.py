# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced replay of recorded token streams under forced acceptance (measurement only).

Switch ``TRTLLM_K3LL_FORCE_REPLAY`` (default 0, read at forward time) with the stream file in
``TRTLLM_K3LL_FORCE_REPLAY_FILE``. Port of Dongxu's ``TLLM_SPEC_DECODE_FORCE_REPLAY`` (tekit 5c9f1277b3) to the
staircase's MLA DSpark worker.

Why: under forced acceptance (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``) a request commits the target's own
tokens whatever the drafts were, so its token stream is a chaotic function of the numerics: one near-tie that flips
between two runs (BF16 atomic sums in the MoE) sends it into another stream with other routed experts. Gen-only A/B
segments of one configuration then differ by up to a few percent per step, and the streams (garbage-KV argmax
continuations) do not route like real text.

What it does, on a decode step whose generation rows follow a recorded stream:

* before the step's acceptance (`replay_begin`): every generation row is matched from its verify group (the input
  tokens of the row, its newest committed token first) and the global position ``p`` of that first token. Entry ``e``
  (a stream ``tok_e`` placed at global position ``base_e``) matches when ``j = p - base_e`` is inside the stream and
  the group equals ``tok_e[j : j + K + 1]``; the lowest such entry wins, a row without one is left alone. The forced
  acceptance's random draw is keyed by the offset of row 0 (the RNG counter is set to ``j``), so the accepted counts
  depend on where the rows are in their streams, not on how many steps the process has run.
* after the step (`replay_end`), for every matched row with ``n`` accepted tokens (bonus included): the committed
  tokens become ``tok_e[j + 1 : j + K + 2]`` (the first ``n`` are kept), the next drafts ``tok_e[j + n + 1 : j + n +
  K + 1]`` and the next step's input group ``tok_e[j + n : j + n + K + 1]``. The next step's group then matches
  ``e`` again at ``j + n``.

The target, the drafter and every kernel still run with the same shapes; only token values change (the drafter's
own outputs are replaced), so every step's routing depends on the stream alone and the segments of a one-process
ABBA replay the same streams. Two small Triton kernels per step are added to every arm alike.

The file is JSON lines ``{"prompt_len": L, "token_ids": [...]}`` (other keys ignored): stream ``token_ids`` placed at
global position ``L``, i.e. ``token_ids[0]`` is the request's first generated token (the context phase's). A
generation-only request starts on entry ``e`` when its first token is ``tok_e[0]`` and its first drafts are
``tok_e[1 : K + 1]`` (the gen-only harness sets both through DisaggregatedParams). The file is read once, on the
first replaying step.
"""

import json
import os
from typing import List, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

from . import flags as _flags

FILE_ENV = "TRTLLM_K3LL_FORCE_REPLAY_FILE"
#: Most entries (stream x placement) one file may hold: every row is compared with every entry.
MAX_ENTRIES = 64
#: Most generation rows one step may replay.
MAX_ROWS = 64


def load_entries(path: str) -> List[Tuple[int, List[int]]]:
    """``(prompt_len, token_ids)`` of the file's records, in file order, without exact repeats."""
    entries, seen = [], set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens = [int(t) for t in rec.get("token_ids") or []]
            if not tokens:
                continue
            key = (int(rec["prompt_len"]), tuple(tokens))
            if key not in seen:
                seen.add(key)
                entries.append((key[0], tokens))
    if not entries or len(entries) > MAX_ENTRIES:
        raise ValueError(
            f"{FILE_ENV}: {path} holds {len(entries)} streams with token ids, 1 to {MAX_ENTRIES} are supported"
        )
    return entries


class ReplayTables(NamedTuple):
    """Device tables: the entries (stream rows padded with copies of their last token so a group or a draft run near
    the end stays inside the row) and the per-row match the lookup leaves for the commit."""

    base: torch.Tensor  # [E] int64, global position of each entry's first token
    length: torch.Tensor  # [E] int32, recorded tokens per entry
    tok: torch.Tensor  # [E, width] int32
    state: (
        torch.Tensor
    )  # [MAX_ROWS, 2] int32: (entry or -1, offset j of the row's first input token)

    @staticmethod
    def build(entries: List[Tuple[int, List[int]]], device, max_draft_len: int) -> "ReplayTables":
        pad = 2 * (max_draft_len + 2)
        width = max(len(t) for _, t in entries) + pad
        tok = torch.empty((len(entries), width), dtype=torch.int32)
        for e, (_, tokens) in enumerate(entries):
            tok[e, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
            tok[e, len(tokens) :] = tokens[-1]
        return ReplayTables(
            base=torch.tensor([b for b, _ in entries], dtype=torch.int64).to(device),
            length=torch.tensor([len(t) for _, t in entries], dtype=torch.int32).to(device),
            tok=tok.to(device),
            state=torch.full((MAX_ROWS, 2), -1, dtype=torch.int32, device=device),
        )


@triton.jit
def _replay_lookup_kernel(
    ids_ptr,
    pos_ptr,
    tok0,
    num_rows,
    base_ptr,
    len_ptr,
    tok_ptr,
    tok_stride,
    num_entries,
    state_ptr,
    counter_ptr,
    K1: tl.constexpr,
    HAS_COUNTER: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_R)
    row_ok = rows < num_rows
    c = tl.arange(0, BLOCK_C)
    col_ok = c < K1
    tok_idx = tok0 + rows[:, None] * K1 + c[None, :]
    group = tl.load(ids_ptr + tok_idx, mask=row_ok[:, None] & col_ok[None, :], other=-1).to(
        tl.int64
    )
    pos = tl.load(pos_ptr + tok0 + rows * K1, mask=row_ok, other=0).to(tl.int64)
    e = tl.arange(0, BLOCK_E)
    e_ok = e < num_entries
    base = tl.load(base_ptr + e, mask=e_ok, other=0).to(tl.int64)
    length = tl.load(len_ptr + e, mask=e_ok, other=0).to(tl.int64)
    j = pos[:, None] - base[None, :]
    inside = row_ok[:, None] & e_ok[None, :] & (j >= 0) & (j < length[None, :])
    rec = tl.load(
        tok_ptr
        + e[None, :, None].to(tl.int64) * tok_stride
        + tl.where(inside, j, 0)[:, :, None]
        + c[None, None, :],
        mask=inside[:, :, None] & col_ok[None, None, :],
        other=-2,
    ).to(tl.int64)
    same = (rec == group[:, None, :]) | (c[None, None, :] >= K1)
    full = inside & (tl.min(same.to(tl.int32), axis=2) > 0)
    ek = tl.min(tl.where(full, e[None, :], BLOCK_E), axis=1)
    found = ek < BLOCK_E
    jk = tl.sum(tl.where(e[None, :] == ek[:, None], j, 0), axis=1)
    tl.store(state_ptr + rows * 2, tl.where(found, ek, -1).to(tl.int32), mask=row_ok)
    tl.store(state_ptr + rows * 2 + 1, tl.where(found, jk, 0).to(tl.int32), mask=row_ok)
    if HAS_COUNTER:
        first = row_ok & (rows == 0) & found
        j0 = tl.sum(tl.where(first, jk, 0), axis=0)
        n0 = tl.sum(first.to(tl.int32), axis=0)
        tl.store(counter_ptr, j0.to(tl.int64), mask=n0 > 0)


@triton.jit
def _replay_commit_kernel(
    new_ptr,
    new_stride,
    lens_ptr,
    nd_ptr,
    nd_stride,
    nn_ptr,
    nn_stride,
    row0,
    num_rows,
    tok_ptr,
    tok_stride,
    state_ptr,
    K: tl.constexpr,
    HAS_ND: tl.constexpr,
    HAS_NN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_R)
    row_ok = rows < num_rows
    ent = tl.load(state_ptr + rows * 2, mask=row_ok, other=-1)
    j = tl.load(state_ptr + rows * 2 + 1, mask=row_ok, other=0).to(tl.int64)
    ok = row_ok & (ent >= 0)
    n = tl.load(lens_ptr + row0 + rows, mask=ok, other=0).to(tl.int64)
    c = tl.arange(0, BLOCK_C)
    src = tok_ptr + tl.where(ok, ent, 0).to(tl.int64)[:, None] * tok_stride
    out_row = (row0 + rows).to(tl.int64)[:, None]
    m1 = ok[:, None] & (c[None, :] < K + 1)
    t_new = tl.load(src + j[:, None] + 1 + c[None, :], mask=m1, other=0)
    tl.store(new_ptr + out_row * new_stride + c[None, :], t_new, mask=m1)
    if HAS_NN:
        t_nn = tl.load(src + j[:, None] + n[:, None] + c[None, :], mask=m1, other=0)
        tl.store(nn_ptr + out_row * nn_stride + c[None, :], t_nn, mask=m1)
    if HAS_ND:
        m2 = ok[:, None] & (c[None, :] < K)
        t_nd = tl.load(src + j[:, None] + n[:, None] + 1 + c[None, :], mask=m2, other=0)
        tl.store(nd_ptr + out_row * nd_stride + c[None, :], t_nd, mask=m2)


def _pow2(x: int) -> int:
    return triton.next_power_of_2(max(int(x), 1))


def replay_lookup(
    tables: ReplayTables,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    tok0: int,
    num_rows: int,
    tokens_per_row: int,
    rng_counter: Optional[torch.Tensor] = None,
) -> None:
    """Match generation rows ``0 .. num_rows - 1`` (row ``g``'s group is ``input_ids[tok0 + g * tokens_per_row :][:
    tokens_per_row]`` at global position ``positions[tok0 + g * tokens_per_row]``) into ``tables.state``; set
    ``rng_counter[0]`` to row 0's offset when row 0 matches."""
    E = tables.tok.shape[0]
    _replay_lookup_kernel[(1,)](
        input_ids,
        positions,
        int(tok0),
        int(num_rows),
        tables.base,
        tables.length,
        tables.tok,
        tables.tok.stride(0),
        E,
        tables.state,
        rng_counter if rng_counter is not None else tables.state,
        K1=int(tokens_per_row),
        HAS_COUNTER=rng_counter is not None,
        BLOCK_R=_pow2(num_rows),
        BLOCK_E=_pow2(E),
        BLOCK_C=_pow2(tokens_per_row),
    )


def replay_commit(
    tables: ReplayTables,
    new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    next_draft_tokens: Optional[torch.Tensor],
    next_new_tokens: Optional[torch.Tensor],
    row0: int,
    num_rows: int,
    K: int,
) -> None:
    """Overwrite, in place, the committed tokens / next drafts / next input group of the matched rows (rows
    ``row0 + g`` of the step's outputs)."""
    nd = next_draft_tokens if next_draft_tokens is not None else new_tokens
    nn = next_new_tokens if next_new_tokens is not None else new_tokens
    _replay_commit_kernel[(1,)](
        new_tokens,
        new_tokens.stride(0),
        new_tokens_lens,
        nd,
        nd.stride(0),
        nn,
        nn.stride(0),
        int(row0),
        int(num_rows),
        tables.tok,
        tables.tok.stride(0),
        tables.state,
        K=int(K),
        HAS_ND=next_draft_tokens is not None,
        HAS_NN=next_new_tokens is not None,
        BLOCK_R=_pow2(num_rows),
        BLOCK_C=_pow2(K + 1),
    )


def _warn_once(worker, reason: str) -> None:
    seen = getattr(worker, "_k3_replay_warned", None)
    if seen is None:
        seen = worker._k3_replay_warned = set()
    if reason not in seen:
        seen.add(reason)
        logger.warning(f"TRTLLM_K3LL_FORCE_REPLAY: not replaying this step: {reason}")


def _tables(worker, device) -> Optional[ReplayTables]:
    tables = getattr(worker, "_k3_replay_tables", None)
    if tables is not None:
        return tables
    if torch.cuda.is_current_stream_capturing():
        _warn_once(worker, "the stream tables must be built outside CUDA-graph capture")
        return None
    path = os.environ.get(FILE_ENV)
    if not path:
        _warn_once(worker, f"{FILE_ENV} is not set")
        return None
    entries = load_entries(path)
    tables = worker._k3_replay_tables = ReplayTables.build(
        entries, device, int(worker.max_draft_len)
    )
    logger.info(
        f"TRTLLM_K3LL_FORCE_REPLAY: {len(entries)} entries from {path} (prompt_len "
        f"{sorted({b for b, _ in entries})}, tokens {min(len(t) for _, t in entries)}-"
        f"{max(len(t) for _, t in entries)})"
    )
    return tables


def replay_begin(worker, input_ids, position_ids, attn_metadata, spec_metadata):
    """Before the step: the rows' match and the keyed draw. Returns the commit's arguments, or None when the switch
    is off or the step does not replay (then nothing ran)."""
    if not _flags.TRTLLM_K3LL_FORCE_REPLAY:
        return None
    if float(getattr(worker, "force_num_accepted_tokens", 0.0) or 0.0) == 0.0:
        _warn_once(worker, "forced acceptance (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS) is off")
        return None
    num_contexts = int(attn_metadata.num_contexts)
    num_gens = int(attn_metadata.num_seqs) - num_contexts
    if num_gens <= 0:
        return None
    if num_gens > MAX_ROWS:
        _warn_once(worker, f"{num_gens} generation rows > {MAX_ROWS}")
        return None
    K = int(spec_metadata.runtime_draft_len)
    tok0 = int(attn_metadata.num_ctx_tokens)
    ids = input_ids.reshape(-1)
    if ids.shape[0] - tok0 != num_gens * (K + 1):
        _warn_once(
            worker, f"{ids.shape[0] - tok0} generation tokens for {num_gens} rows of {K + 1}"
        )
        return None
    pos = position_ids[0, 0] if position_ids.ndim == 3 else position_ids.reshape(-1)
    if pos.shape[0] != ids.shape[0] or pos.stride(-1) != 1 or ids.stride(-1) != 1:
        _warn_once(worker, "positions / input ids are not one contiguous row per token")
        return None
    tables = _tables(worker, ids.device)
    if tables is None:
        return None
    if not torch.cuda.is_current_stream_capturing():
        worker._ensure_force_accept_rng_state(ids.device)
    counter = getattr(worker, "_force_accept_rng_counter", None)
    replay_lookup(tables, ids, pos, tok0, num_gens, K + 1, counter)
    return tables, num_contexts, num_gens, K


def replay_end(replay, outputs: dict) -> None:
    """After the step: the matched rows' committed tokens, next drafts and next input group from their streams."""
    tables, num_contexts, num_gens, K = replay
    new = outputs.get("new_tokens")
    lens = outputs.get("new_tokens_lens")
    nd = outputs.get("next_draft_tokens")
    nn = outputs.get("next_new_tokens")
    rows = num_contexts + num_gens
    for name, t, width in (
        ("new_tokens", new, K + 1),
        ("next_draft_tokens", nd, K),
        ("next_new_tokens", nn, K + 1),
    ):
        if t is None or t.ndim != 2 or t.shape[0] < rows or t.shape[1] < width or t.stride(-1) != 1:
            raise RuntimeError(
                f"TRTLLM_K3LL_FORCE_REPLAY: unexpected {name} "
                f"{None if t is None else (tuple(t.shape), t.stride())} for {rows} rows of {width}"
            )
    if lens is None or lens.ndim != 1 or lens.shape[0] < rows:
        raise RuntimeError("TRTLLM_K3LL_FORCE_REPLAY: unexpected new_tokens_lens")
    replay_commit(tables, new, lens, nd, nn, num_contexts, num_gens, K)

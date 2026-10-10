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
"""Vocab-parallel token selection for a vocab-sharded LM head.

A column-parallel LM head leaves every TP rank with a contiguous vocab shard of
the logits. Gathering the full logits to pick one token per row moves
``rows * vocab`` values; this module instead reduces each rank's shard to one
``(value, global index)`` pair per row, exchanges only those pairs, and takes
the winner. The result equals ``torch.argmax`` over the gathered logits bit for
bit: ``NaN`` beats every number, and equal maxima resolve to the lowest global
index (rank shards are contiguous and ordered, so the lowest rank's lowest
column wins).

Three consumers:

* :func:`vocab_parallel_argmax` -- greedy target sampling.
* :func:`vocab_parallel_sample` -- temperature-only sampling through the
  Gumbel-max trick: ``argmax(logits / T + G)`` with ``G`` standard Gumbel noise
  is an exact draw from ``softmax(logits / T)``. The noise of every vocab entry
  comes from a counter-based Philox stream keyed by the row's seed, so all
  ranks agree on the winner. The distribution matches the gathered sampler; the
  individual draws do not (different random stream).
* :func:`vocab_parallel_markov_tokens` -- the DSpark vanilla-Markov greedy
  block chain on shard logits: step ``i`` adds ``markov_w2 @ markov_w1[prev_i]``
  to block position ``i`` and takes the global argmax, which becomes
  ``prev_{i+1}`` and draft token ``i``.

The pair exchange is an all-reduce (sum) of a zero-padded ``[rows, 2 * tp]``
float32 buffer in which each rank fills only its own two columns (``x + 0`` is
exact), over the MNNVL all-reduce when available and NCCL otherwise. Every
launch has a shape fixed by ``(rows, shard width)`` and nothing synchronizes
with the host, so all paths are CUDA-graph capturable.

Switches (read once per process):

* ``TLLM_SPEC_VOCAB_PARALLEL=1`` enables the greedy target argmax and the
  draft Markov chain.
* ``TLLM_SPEC_VOCAB_PARALLEL_SAMPLING=1`` additionally samples non-greedy
  target rows vocab-parallel, only when the deployment's
  ``advanced_sampling_mode`` is ``no_topk_no_topp``.
* ``TLLM_SPEC_VOCAB_PARALLEL_COMM=auto|mnnvl|nccl`` picks the pair exchange.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

VOCAB_PARALLEL_ENV_VAR = "TLLM_SPEC_VOCAB_PARALLEL"
VOCAB_PARALLEL_SAMPLING_ENV_VAR = "TLLM_SPEC_VOCAB_PARALLEL_SAMPLING"
VOCAB_PARALLEL_COMM_ENV_VAR = "TLLM_SPEC_VOCAB_PARALLEL_COMM"

_INT32_MAX = tl.constexpr(2**31 - 1)
_LOCAL_BLOCK_V = 2048
_REDUCE_BLOCK_ROWS = 32


def vocab_parallel_enabled() -> bool:
    """Whether ``TLLM_SPEC_VOCAB_PARALLEL=1`` selects the vocab-parallel paths."""
    return os.environ.get(VOCAB_PARALLEL_ENV_VAR, "0") == "1"


def vocab_parallel_sampling_enabled() -> bool:
    """Whether non-greedy target rows may be sampled vocab-parallel."""
    return vocab_parallel_enabled() and os.environ.get(VOCAB_PARALLEL_SAMPLING_ENV_VAR, "0") == "1"


@triton.jit
def _argmax_combine(val_a, idx_a, val_b, idx_b):
    # torch.argmax order: NaN beats any number, equal values go to the lower index.
    nan_a = val_a != val_a
    nan_b = val_b != val_b
    take_a = tl.where(
        nan_a,
        (~nan_b) | (idx_a < idx_b),
        (~nan_b) & ((val_a > val_b) | ((val_a == val_b) & (idx_a < idx_b))),
    )
    return tl.where(take_a, val_a, val_b), tl.where(take_a, idx_a, idx_b)


@triton.jit
def _gumbel_noise(seed, c0, c1, c2, c3):
    # One standard Gumbel sample per counter tuple: -log(-log(u)), u in (0, 1).
    r0, _, _, _ = tl.philox(seed, c0, c1, c2, c3)
    # 23 random bits mapped to the open interval (0, 1); with 23 bits the
    # midpoint (k + 0.5) / 2^23 is exact in float32, so u never rounds to 1.
    u = ((r0 >> 9).to(tl.float32) + 0.5) * (1.0 / 8388608.0)
    return -tl.log(-tl.log(u))


@triton.jit
def _store_pair(
    out_ptr,
    row,
    stride_out,
    slot,
    best_val,
    best_idx,
    col_base,
    ZERO_WIDTH: tl.constexpr,
):
    # Publishes (value, global index) at columns [2 * slot, 2 * slot + 1] of the
    # row; when ZERO_WIDTH > 0 the rest of the row is zero-filled so an
    # all-reduce sum acts as an all-gather.
    gidx = tl.where(best_idx == _INT32_MAX, col_base, col_base + best_idx)
    base = out_ptr + row.to(tl.int64) * stride_out
    if ZERO_WIDTH > 0:
        offs = tl.arange(0, ZERO_WIDTH)
        vals = tl.where(
            offs == 2 * slot,
            best_val,
            tl.where(offs == 2 * slot + 1, gidx.to(tl.float32), 0.0),
        )
        tl.store(base + offs, vals)
    else:
        tl.store(base + 2 * slot, best_val)
        tl.store(base + 2 * slot + 1, gidx.to(tl.float32))


@triton.jit
def _local_argmax_kernel(
    x_ptr,
    stride_x,
    n_cols,
    col_base,
    out_ptr,
    stride_out,
    slot,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_V)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(0, n_cols, BLOCK_V):
        cols = start + offs
        mask = cols < n_cols
        vals = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = _argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, _argmax_combine)
    _store_pair(out_ptr, row, stride_out, slot, val, idx, col_base, ZERO_WIDTH)


@triton.jit
def _local_sample_kernel(
    x_ptr,
    stride_x,
    n_cols,
    col_base,
    temp_ptr,
    seed_ptr,
    offset_ptr,
    greedy_threshold,
    out_ptr,
    stride_out,
    slot,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_V)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    temperature = tl.load(temp_ptr + row).to(tl.float32)
    is_greedy = temperature <= greedy_threshold
    inv_t = 1.0 / tl.where(is_greedy, 1.0, temperature)
    seed = tl.load(seed_ptr + row).to(tl.int64)
    offset = tl.load(offset_ptr + row).to(tl.int64)
    # Counter: (global vocab index, row, offset low, offset high) -- disjoint
    # streams per row and per step, identical on every rank.
    c1 = tl.full([BLOCK_V], 0, tl.uint32) + row.to(tl.uint32)
    c2 = tl.full([BLOCK_V], 0, tl.uint32) + (offset & 0xFFFFFFFF).to(tl.uint32)
    c3 = tl.full([BLOCK_V], 0, tl.uint32) + ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(0, n_cols, BLOCK_V):
        cols = start + offs
        mask = cols < n_cols
        vals = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if not is_greedy:
            noise = _gumbel_noise(seed, (col_base + cols).to(tl.uint32), c1, c2, c3)
            vals = vals * inv_t + noise
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = _argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, _argmax_combine)
    _store_pair(out_ptr, row, stride_out, slot, val, idx, col_base, ZERO_WIDTH)


@triton.jit
def _reduce_pairs_kernel(
    pairs_ptr,
    stride_row,
    stride_slot,
    n_rows,
    n_slots,
    out_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    slots = tl.arange(0, BLOCK_SLOTS)
    mask = (rows[:, None] < n_rows) & (slots[None, :] < n_slots)
    ptrs = pairs_ptr + rows[:, None].to(tl.int64) * stride_row + slots[None, :] * stride_slot
    vals = tl.load(ptrs, mask=mask, other=float("-inf"))
    gidx = tl.load(ptrs + 1, mask=mask, other=0.0).to(tl.int32)
    gidx = tl.where(mask, gidx, _INT32_MAX)
    _, best = tl.reduce((vals, gidx), 1, _argmax_combine)
    tl.store(out_ptr + rows, best.to(out_ptr.dtype.element_ty), mask=rows < n_rows)


@triton.jit
def _markov_reduce_embed_kernel(
    pairs_ptr,
    stride_row,
    stride_slot,
    n_slots,
    anchor_ptr,
    w1_ptr,
    emb_ptr,
    tokens_ptr,
    stride_tokens,
    vocab,
    step,
    RANK: tl.constexpr,
    FROM_ANCHOR: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Resolves prev_step for one request (the anchor at step 0, else the global
    # argmax of the previous step, which is also draft token step - 1) and
    # gathers its markov_w1 row; an out-of-vocab prev gives a zero row.
    req = tl.program_id(0)
    if FROM_ANCHOR:
        prev = tl.load(anchor_ptr + req).to(tl.int64)
    else:
        slots = tl.arange(0, BLOCK_SLOTS)
        smask = slots < n_slots
        ptrs = pairs_ptr + req.to(tl.int64) * stride_row + slots * stride_slot
        vals = tl.load(ptrs, mask=smask, other=float("-inf"))
        gidx = tl.load(ptrs + 1, mask=smask, other=0.0).to(tl.int32)
        gidx = tl.where(smask, gidx, _INT32_MAX)
        _, best = tl.reduce((vals, gidx), 0, _argmax_combine)
        prev = best.to(tl.int64)
        tl.store(tokens_ptr + req.to(tl.int64) * stride_tokens + step - 1, prev.to(tl.int32))
    valid = (prev >= 0) & (prev < vocab)
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < RANK
    w1_row = tl.load(
        w1_ptr + tl.where(valid, prev, 0) * RANK + offs_r, mask=mask_r & valid, other=0.0
    )
    tl.store(emb_ptr + req * RANK + offs_r, w1_row, mask=mask_r)


@triton.jit
def _markov_add_local_argmax_kernel(
    base_ptr,
    stride_base,
    bias_ptr,
    stride_bias,
    n_cols,
    col_base,
    out_ptr,
    stride_out,
    slot,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # base (shard logits of block position `step`) + bias, upcast to fp32 like
    # the gathered reference, then the shard argmax.
    req = tl.program_id(0)
    offs = tl.arange(0, BLOCK_V)
    base_row = base_ptr + req.to(tl.int64) * stride_base
    bias_row = bias_ptr + req.to(tl.int64) * stride_bias
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(0, n_cols, BLOCK_V):
        cols = start + offs
        mask = cols < n_cols
        base = tl.load(base_row + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
        vals = tl.where(mask, base + bias, float("-inf"))
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = _argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, _argmax_combine)
    _store_pair(out_ptr, req, stride_out, slot, val, idx, col_base, ZERO_WIDTH)


class _PairExchange:
    """Exchanges one ``(value, global index)`` float32 pair per row across TP.

    ``buffer(rows)`` returns the tensor the local kernels write into and the
    store layout (row stride, slot, zero-fill width); ``exchange`` returns the
    tensor holding every rank's pair and its read layout (row stride, slot
    stride).
    """

    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self._mnnvl = None
        backend = os.environ.get(VOCAB_PARALLEL_COMM_ENV_VAR, "auto").lower()
        if backend not in ("auto", "mnnvl", "nccl"):
            raise ValueError(
                f"{VOCAB_PARALLEL_COMM_ENV_VAR} must be auto, mnnvl or nccl; got {backend!r}"
            )
        if self.tp_size > 1 and backend in ("auto", "mnnvl"):
            from ..distributed.ops import MNNVLAllReduce

            if MNNVLAllReduce.is_mnnvl(mapping, torch.float32):
                try:
                    self._mnnvl = MNNVLAllReduce(mapping, torch.float32)
                except (RuntimeError, ValueError) as e:
                    if backend == "mnnvl":
                        raise
                    logger.info(
                        f"Vocab-parallel pair exchange: MNNVL unavailable ({e}); using NCCL."
                    )
            elif backend == "mnnvl":
                raise RuntimeError("TLLM_SPEC_VOCAB_PARALLEL_COMM=mnnvl but MNNVL is unavailable")
        self.backend = "mnnvl" if self._mnnvl is not None else "nccl"
        # Row layout of the zero-padded all-reduce buffer.
        self._width = triton.next_power_of_2(2 * self.tp_size)

    def buffer(self, rows: int, device) -> tuple[torch.Tensor, int, int, int]:
        if self.tp_size == 1 or self.backend == "nccl":
            buf = torch.empty((rows, 2), dtype=torch.float32, device=device)
            return buf, 2, 0, 0
        buf = torch.empty((rows, self._width), dtype=torch.float32, device=device)
        return buf, self._width, self.tp_rank, self._width

    def exchange(self, buf: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        rows = buf.shape[0]
        if self.tp_size == 1:
            return buf, 2, 0
        if self.backend == "mnnvl":
            out = self._mnnvl(buf, all_reduce_params=AllReduceParams())
            return out, self._width, 2
        from ..distributed.ops import allgather

        out = allgather(buf, self.mapping, dim=0)
        return out, 2, 2 * rows


class VocabParallelHead:
    """Vocab-parallel token selection over one column-parallel LM head.

    Args:
        lm_head: an ``LMHead`` (or ``Linear``) in COLUMN tensor-parallel mode
            whose weight holds this rank's contiguous vocab shard.
        vocab_size: the full vocabulary size.
    """

    def __init__(self, lm_head, vocab_size: int):
        self.lm_head = lm_head
        self.mapping = lm_head.mapping
        self.vocab_size = int(vocab_size)
        self.shard_width = int(lm_head.out_features)
        self.col_base = self.mapping.tp_rank * self.shard_width
        self.exchange = _PairExchange(self.mapping)
        self._markov_w2_shard = None
        self._markov_w2_src_ptr = None
        logger.info(
            f"Vocab-parallel LM-head token selection: tp={self.mapping.tp_size}, "
            f"shard={self.shard_width}, vocab={self.vocab_size}, exchange={self.exchange.backend}"
        )

    @staticmethod
    def supported(lm_head, vocab_size: int) -> bool:
        """Whether ``lm_head`` is a plain vocab-sharded head this class handles."""
        from ..modules.linear import TensorParallelMode

        mapping = getattr(lm_head, "mapping", None)
        if mapping is None or mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return False
        if getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN:
            return False
        if not getattr(lm_head, "gather_output", False):
            return False
        if getattr(lm_head, "padding_size", 0) != 0 or getattr(lm_head, "has_any_quant", False):
            return False
        return int(lm_head.out_features) * mapping.tp_size == int(vocab_size)

    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``[rows, shard]`` logits in the head's output dtype."""
        return self.lm_head.apply_linear(hidden_states, self.lm_head.bias)

    def shard_logits(self, hidden_states: torch.Tensor) -> "VocabShardedLogits":
        """This rank's logits wrapped for a spec worker (see ``VocabShardedLogits``)."""
        return VocabShardedLogits(self, self.local_logits(hidden_states))

    def gather_full_logits(self, local: torch.Tensor) -> torch.Tensor:
        """Full float32 logits, identical to the gathered head + ``.float()``."""
        from ..distributed import allgather

        return allgather(local, self.mapping, sizes=self.lm_head.gather_output_sizes).float()

    def _launch_local(self, kernel, local: torch.Tensor, extra_args: tuple, rows: int):
        buf, stride_out, slot, zero_width = self.exchange.buffer(rows, local.device)
        if rows > 0:
            kernel[(rows,)](
                local,
                local.stride(0),
                local.shape[-1],
                self.col_base,
                *extra_args,
                buf,
                stride_out,
                slot,
                ZERO_WIDTH=zero_width,
                BLOCK_V=_LOCAL_BLOCK_V,
                num_warps=8,
            )
        return buf

    def _reduce(self, pairs: torch.Tensor, stride_row: int, stride_slot: int, rows: int, out):
        if rows > 0:
            _reduce_pairs_kernel[(triton.cdiv(rows, _REDUCE_BLOCK_ROWS),)](
                pairs,
                stride_row,
                stride_slot,
                rows,
                self.mapping.tp_size,
                out,
                BLOCK_ROWS=_REDUCE_BLOCK_ROWS,
                BLOCK_SLOTS=triton.next_power_of_2(self.mapping.tp_size),
                num_warps=4,
            )
        return out

    def argmax(self, local: torch.Tensor, out_dtype=torch.int64) -> torch.Tensor:
        """Global argmax over the vocab of shard logits ``[rows, shard]``."""
        assert local.dim() == 2 and local.stride(-1) == 1
        rows = local.shape[0]
        buf = self._launch_local(_local_argmax_kernel, local, (), rows)
        pairs, stride_row, stride_slot = self.exchange.exchange(buf)
        out = torch.empty((rows,), dtype=out_dtype, device=local.device)
        return self._reduce(pairs, stride_row, stride_slot, rows, out)

    def sample(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
        out_dtype=torch.int64,
    ) -> torch.Tensor:
        """Temperature-only sampling (Gumbel-max); rows at or below the greedy
        temperature threshold take the plain argmax."""
        assert local.dim() == 2 and local.stride(-1) == 1
        rows = local.shape[0]
        buf = self._launch_local(
            _local_sample_kernel,
            local,
            (temperatures, seeds, offsets, float(greedy_threshold)),
            rows,
        )
        pairs, stride_row, stride_slot = self.exchange.exchange(buf)
        out = torch.empty((rows,), dtype=out_dtype, device=local.device)
        return self._reduce(pairs, stride_row, stride_slot, rows, out)

    def markov_w2_shard(self, markov_w2: torch.Tensor) -> torch.Tensor:
        """This rank's rows of ``markov_w2`` (cached; a contiguous row slice)."""
        if self._markov_w2_src_ptr != markov_w2.data_ptr():
            self._markov_w2_shard = markov_w2[self.col_base : self.col_base + self.shard_width]
            self._markov_w2_src_ptr = markov_w2.data_ptr()
        return self._markov_w2_shard

    def markov_tokens(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: Optional[torch.Tensor],
        markov_w2: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Greedy DSpark block draft tokens from shard block logits.

        Args:
            local_block: ``[B, K, shard]`` shard logits of the block positions.
            first_prev_tokens: ``[B]`` anchor token ids (step 0's prev).
            markov_w1, markov_w2: ``[vocab, rank]`` Markov head weights, or
                ``None`` for a drafter without a Markov head (plain argmax).
        Returns:
            ``[B, K]`` int32 draft tokens, bit-identical to the per-position
            argmax of ``dspark_markov_chain_logits`` on the gathered logits.
        """
        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        if markov_w1 is None:
            flat = local_block.reshape(num_reqs * block_size, shard)
            return self.argmax(flat, out_dtype=torch.int32).view(num_reqs, block_size)
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        if num_reqs == 0 or block_size == 0:
            return tokens
        rank = markov_w1.shape[1]
        w2_shard = self.markov_w2_shard(markov_w2)
        tp = self.mapping.tp_size
        emb = torch.empty((num_reqs, rank), dtype=markov_w1.dtype, device=device)
        pairs, stride_row, stride_slot = first_prev_tokens, 0, 0
        for step in range(block_size + 1):
            _markov_reduce_embed_kernel[(num_reqs,)](
                pairs,
                stride_row,
                stride_slot,
                tp,
                first_prev_tokens,
                markov_w1,
                emb,
                tokens,
                tokens.stride(0),
                self.vocab_size,
                step,
                RANK=rank,
                FROM_ANCHOR=step == 0,
                BLOCK_SLOTS=triton.next_power_of_2(tp),
                BLOCK_R=triton.next_power_of_2(rank),
                num_warps=4,
            )
            if step == block_size:
                break
            # Same F.linear as the reference chain, on this rank's w2 rows.
            bias = F.linear(emb, w2_shard)
            step_logits = local_block[:, step]
            buf, stride_out, slot, zero_width = self.exchange.buffer(num_reqs, device)
            _markov_add_local_argmax_kernel[(num_reqs,)](
                step_logits,
                step_logits.stride(0),
                bias,
                bias.stride(0),
                shard,
                self.col_base,
                buf,
                stride_out,
                slot,
                ZERO_WIDTH=zero_width,
                BLOCK_V=_LOCAL_BLOCK_V,
                num_warps=8,
            )
            pairs, stride_row, stride_slot = self.exchange.exchange(buf)
        return tokens


class VocabShardedLogits:
    """This rank's vocab shard of the target logits, handed to a spec worker in
    place of the gathered float32 logits.

    The worker either selects tokens from it vocab-parallel
    (:meth:`select_tokens`) or, for a batch the vocab-parallel samplers do not
    cover, materializes the gathered logits with :meth:`gather`.
    """

    __slots__ = ("head", "local")

    def __init__(self, head: VocabParallelHead, local: torch.Tensor):
        self.head = head
        self.local = local

    @property
    def device(self) -> torch.device:
        return self.local.device

    def gather(self) -> torch.Tensor:
        """The gathered ``[rows, vocab]`` float32 logits."""
        return self.head.gather_full_logits(self.local)

    def select_tokens(self, spec_metadata) -> torch.Tensor:
        """One target token per row: the argmax for an all-greedy batch, else a
        temperature-only draw (the caller guarantees the sampling mode)."""
        if spec_metadata.is_all_greedy_sample:
            return self.head.argmax(self.local)
        from ..pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

        rows = self.local.shape[0]
        return self.head.sample(
            self.local,
            spec_metadata.temperatures[:rows],
            spec_metadata.seeds[:rows],
            spec_metadata.offsets[:rows],
            GREEDY_TEMPERATURE_THRESHOLD,
        )

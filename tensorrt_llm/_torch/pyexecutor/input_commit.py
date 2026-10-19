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
"""One-launch upload of a decode step's input buffers (TRTLLM_BATCHED_INPUT_COMMIT=1).

Input preparation for a CUDA-graph decode step fills a few dozen small device buffers
from host data (positions, gather ids, KV lengths, helix offsets, sampling seeds, ...)
and, under the overlap scheduler, gathers the previous step's tokens and accepted
lengths on the device. Issued one by one, these copies and gathers form a chain of
~40 tiny operations that runs between two graph replays, because the buffers they
write are still read by the previous replay. ``InputCommit`` collects the host values
into a pinned arena while the step is being prepared and writes every destination
with one kernel (reading the arena directly over the host link) right before the
replay. The overlap-scheduler gathers run in the same kernel.

Contract for staged copies: between ``stage`` and ``commit`` no device operation may
read or write a staged destination. Call sites stage only buffers that the input
preparation itself does not read back on the device; the device-side gathers that do
read them (previous-step tokens) are part of the commit kernel.

The block-offset copy of ``KVCacheManagerV2`` has its own kernel here
(``copy_batch_block_offsets_from_host``): same semantics as the C++
``copyBatchBlockOffsetsToDeviceKernel`` (reads the pinned host table when it runs),
but split over many CTAs instead of one CTA per (pool, sequence).
"""

import os
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned
from tensorrt_llm.logger import logger

_ENABLED = os.environ.get("TRTLLM_BATCHED_INPUT_COMMIT", "0") == "1"
# Debug aid: after each of the first N commits (-1: all), synchronize and compare every
# destination with the staged bytes / a torch reference of the overlap gathers (raises
# on a mismatch; logs once when the first N commits have matched).
_VERIFY_COMMITS = int(os.environ.get("TRTLLM_BATCHED_INPUT_COMMIT_VERIFY", "0"))
_BLOCK_OFFSETS_TRITON = os.environ.get("TRTLLM_KV_BLOCK_OFFSETS_TRITON", "0") == "1"
_FUSED_HELIX_PREPROCESS = os.environ.get("TRTLLM_FUSED_HELIX_SPEC_PREPROCESS", "0") == "1"

_ACTIVE: Optional["InputCommit"] = None

# torch dtype -> (numpy dtype, element size) of the staged values.
_TORCH_TO_NUMPY = {
    torch.int32: (np.int32, 4),
    torch.int64: (np.int64, 8),
    torch.bool: (np.bool_, 1),
    torch.float32: (np.float32, 4),
    torch.uint8: (np.uint8, 1),
    torch.int8: (np.int8, 1),
}

# Bytes copied by one program of the commit kernel.
_COPY_CHUNK = 512
# Elements per program of the overlap fixup.
_FIX_BLOCK = 256
# Block-table entries per program of the block-offset copy.
_BLOCK_OFFSETS_BLOCK = 1024
# Above this many block-table entries per call the copy is bound by host-link
# bandwidth and the C++ kernel is as fast (GB300: 2 pools x 976 blocks or
# 1 pool x 15584 blocks at 2-32 sequences: 2.5-14.6 us vs 3.5-20.6 us; at
# 128-256 sequences of 15584 blocks both take 45-85 us).
_BLOCK_OFFSETS_TRITON_MAX_ENTRIES = 1 << 20


def _cdiv(a: int, b: int) -> int:
    # Plain integer ceil-div: triton.cdiv is a JIT function, slow to call from Python.
    return -(-a // b)


def batched_input_commit_enabled() -> bool:
    return _ENABLED


def block_offsets_triton_enabled() -> bool:
    return _BLOCK_OFFSETS_TRITON


def fused_helix_spec_preprocess_enabled() -> bool:
    return _FUSED_HELIX_PREPROCESS


def active_input_commit() -> Optional["InputCommit"]:
    """The commit collecting the current step's uploads, or None (copy directly)."""
    return _ACTIVE


@contextmanager
def input_commit_scope(commit: Optional["InputCommit"]):
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = commit
    try:
        yield commit
    finally:
        _ACTIVE = previous


@triton.jit
def _block_offsets_rows(
    src_ptr,
    dst_ptr,
    copy_idx_ptr,
    scales_ptr,
    kv_offset_ptr,
    src_max_seqs,
    dst_max_seqs,
    num_blocks,
    pool,
    seq,
    chunk,
    BLOCK: tl.constexpr,
):
    # dst[pool, seq, 0, :] = scale * src[pool, copy_idx[seq], 0, :]
    # dst[pool, seq, 1, :] = scale * src[pool, copy_idx[seq], 0, :] + kv_offset
    # with BAD_PAGE_INDEX (-1) mapped to 0 in both halves.
    row = tl.load(copy_idx_ptr + seq).to(tl.int64)
    scale = tl.load(scales_ptr + pool)
    kv_offset = tl.load(kv_offset_ptr + pool)
    pool64 = pool.to(tl.int64)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_blocks
    src_row = src_ptr + (pool64 * src_max_seqs + row) * 2 * num_blocks
    val = tl.load(src_row + offs, mask=mask, other=0)
    bad = val == -1
    k = tl.where(bad, 0, scale * val)
    v = tl.where(bad, 0, scale * val + kv_offset)
    dst_row = dst_ptr + (pool64 * dst_max_seqs + seq) * 2 * num_blocks
    tl.store(dst_row + offs, k, mask=mask)
    tl.store(dst_row + num_blocks + offs, v, mask=mask)


@triton.jit(
    do_not_specialize=["src_max_seqs", "dst_max_seqs", "num_blocks", "num_seqs", "num_chunks"]
)
def _copy_block_offsets_kernel(
    src_ptr,
    dst_ptr,
    copy_idx_ptr,
    scales_ptr,
    kv_offset_ptr,
    src_max_seqs,
    dst_max_seqs,
    num_blocks,
    num_seqs,
    num_chunks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % num_chunks
    seq = (pid // num_chunks) % num_seqs
    pool = pid // (num_chunks * num_seqs)
    _block_offsets_rows(
        src_ptr,
        dst_ptr,
        copy_idx_ptr,
        scales_ptr,
        kv_offset_ptr,
        src_max_seqs,
        dst_max_seqs,
        num_blocks,
        pool,
        seq,
        chunk,
        BLOCK,
    )


def copy_batch_block_offsets_from_host(
    host_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
) -> bool:
    """Triton version of ``copy_batch_block_offsets_to_device`` on the current stream.

    ``host_offsets`` is the pinned ``[num_pools, max_seqs, 2, max_blocks]`` host table and
    ``copy_idx`` / ``index_scales`` / ``kv_offset`` are pinned int32 host tensors; all are
    read when the kernel runs, as in the C++ kernel. Writes pools ``[0, num_pools)`` and
    sequences ``[0, len(copy_idx))`` of ``dst``. Returns False (nothing launched) when the
    tensors do not have that layout or the copy is large enough for the C++ kernel to be
    as fast, so the caller uses the C++ kernel instead.
    """
    if host_offsets.dim() != 4 or dst.dim() != 4:
        return False
    num_pools, src_max_seqs, kv_factor, num_blocks = host_offsets.shape
    num_seqs = copy_idx.shape[0]
    host_tensors = (host_offsets, copy_idx, index_scales, kv_offset)
    if not (
        kv_factor == 2
        and dst.shape[2] == 2
        and dst.shape[3] == num_blocks
        and dst.shape[0] >= num_pools
        and dst.shape[1] >= num_seqs
        and dst.is_cuda
        and dst.is_contiguous()
        and dst.dtype == torch.int32
        and all(
            t.dtype == torch.int32 and t.is_pinned() and t.is_contiguous() for t in host_tensors
        )
        and index_scales.numel() >= num_pools
        and kv_offset.numel() >= num_pools
    ):
        return False
    if num_pools * num_seqs * num_blocks > _BLOCK_OFFSETS_TRITON_MAX_ENTRIES:
        return False
    if num_seqs == 0 or num_blocks == 0:
        return True
    num_chunks = _cdiv(num_blocks, _BLOCK_OFFSETS_BLOCK)
    grid = (num_pools * num_seqs * num_chunks,)
    _copy_block_offsets_kernel[grid](
        host_offsets,
        dst,
        copy_idx,
        index_scales,
        kv_offset,
        src_max_seqs,
        dst.shape[1],
        num_blocks,
        num_seqs,
        num_chunks,
        BLOCK=_BLOCK_OFFSETS_BLOCK,
    )
    return True


@triton.jit(
    do_not_specialize=[
        "num_desc",
        "num_prev",
        "tokens_per_req",
        "draft_width",
        "new_tokens_stride_t",
        "new_tokens_stride_s",
        "next_draft_stride_s",
        "input_ids_start",
        "draft_tokens_start",
        "pos_off_len",
        "pos_off_start",
        "kv_off_len",
        "kv_off_start",
    ]
)
def _input_commit_kernel(
    desc_ptr,
    num_desc,
    slots_ptr,
    num_prev,
    tokens_per_req,
    draft_width,
    new_tokens_ptr,
    new_tokens_stride_t,
    new_tokens_stride_s,
    new_tokens_lens_ptr,
    next_draft_ptr,
    next_draft_stride_s,
    input_ids_ptr,
    input_ids_start,
    draft_tokens_ptr,
    draft_tokens_start,
    pos_off_ptr,
    pos_off_len,
    pos_off_start,
    kv_off_ptr,
    kv_off_len,
    kv_off_start,
    CHUNK: tl.constexpr,
    FIX_BLOCK: tl.constexpr,
    HAS_FIX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_desc:
        # Staged host->device copy: (dst address, src address, bytes <= CHUNK).
        desc = desc_ptr + pid * 3
        dst = tl.load(desc).to(tl.pointer_type(tl.uint8))
        src = tl.load(desc + 1).to(tl.pointer_type(tl.uint8))
        nbytes = tl.load(desc + 2)
        byte_offs = tl.arange(0, CHUNK)
        byte_mask = byte_offs < nbytes
        tl.store(dst + byte_offs, tl.load(src + byte_offs, mask=byte_mask), mask=byte_mask)
    elif HAS_FIX:
        # Overlap-scheduler inputs of the requests that ran in the previous step
        # (slot = their sequence slot in the sampler store):
        #   input_ids[input_ids_start + p*T + t] = new_tokens[t, slot_p]
        #   draft_tokens[draft_tokens_start + p*D + j] = next_draft[slot_p, j]
        #   pos_off = 0 except pos_off[pos_off_start + p*T + t] = new_tokens_lens[slot_p]
        #   kv_off = 0 except kv_off[kv_off_start + p] = new_tokens_lens[slot_p] - T
        idx = (pid - num_desc) * FIX_BLOCK + tl.arange(0, FIX_BLOCK)
        num_tok = num_prev * tokens_per_req

        m_tok = idx < num_tok
        p_tok = idx // tokens_per_req
        t_tok = idx - p_tok * tokens_per_req
        slot_tok = tl.load(slots_ptr + p_tok, mask=m_tok, other=0).to(tl.int64)
        tok = tl.load(
            new_tokens_ptr + t_tok * new_tokens_stride_t + slot_tok * new_tokens_stride_s,
            mask=m_tok,
            other=0,
        )
        tl.store(input_ids_ptr + input_ids_start + idx, tok, mask=m_tok)

        m_pos = idx < pos_off_len
        j_pos = idx - pos_off_start
        in_pos = m_pos & (j_pos >= 0) & (j_pos < num_tok)
        p_pos = tl.where(in_pos, j_pos, 0) // tokens_per_req
        slot_pos = tl.load(slots_ptr + p_pos, mask=in_pos, other=0).to(tl.int64)
        lens_pos = tl.load(new_tokens_lens_ptr + slot_pos, mask=in_pos, other=0)
        tl.store(pos_off_ptr + idx, tl.where(in_pos, lens_pos, 0), mask=m_pos)

        num_draft = num_prev * draft_width
        m_dr = idx < num_draft
        p_dr = idx // tl.maximum(draft_width, 1)
        j_dr = idx - p_dr * draft_width
        slot_dr = tl.load(slots_ptr + p_dr, mask=m_dr, other=0).to(tl.int64)
        dr = tl.load(next_draft_ptr + slot_dr * next_draft_stride_s + j_dr, mask=m_dr, other=0)
        tl.store(draft_tokens_ptr + draft_tokens_start + idx, dr, mask=m_dr)

        m_kv = idx < kv_off_len
        j_kv = idx - kv_off_start
        in_kv = m_kv & (j_kv >= 0) & (j_kv < num_prev)
        slot_kv = tl.load(slots_ptr + tl.where(in_kv, j_kv, 0), mask=in_kv, other=0).to(tl.int64)
        lens_kv = tl.load(new_tokens_lens_ptr + slot_kv, mask=in_kv, other=0)
        tl.store(kv_off_ptr + idx, tl.where(in_kv, lens_kv - tokens_per_req, 0), mask=m_kv)


@triton.jit
def _helix_local_len(g, phys, cp_size, cp_rank):
    # Rank-local prefix length of a global length g under round-robin helix pages
    # (page b on rank b % cp_size); same arithmetic as helix_local_len_vec for g >= 0.
    ledger = phys * cp_size
    full = g // ledger
    rem = g - full * ledger
    return full * phys + tl.minimum(tl.maximum(rem - cp_rank * phys, 0), phys)


@triton.jit(
    do_not_specialize=[
        "num_pos",
        "num_helix",
        "kv_start",
        "kv_count",
        "bound_start",
        "num_bound_seqs",
        "tokens_per_seq",
        "phys",
        "cp_size",
        "cp_rank",
    ]
)
def _helix_spec_preprocess_kernel(
    pos_ptr,
    pos_off_ptr,
    num_pos,
    helix_pos_ptr,
    helix_slots_ptr,
    helix_bounds_ptr,
    num_helix,
    kv_lens_ptr,
    kv_off_ptr,
    kv_start,
    kv_count,
    bound_start,
    num_bound_seqs,
    tokens_per_seq,
    phys,
    cp_size,
    cp_rank,
    BLOCK: tl.constexpr,
):
    # Every element is read and written by the lane that owns its index, so the
    # programs need no ordering between them.
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # position_ids[i] += pos_off[i]
    m_pos = idx < num_pos
    pos = tl.load(pos_ptr + idx, mask=m_pos, other=0)
    off = tl.load(pos_off_ptr + idx, mask=m_pos, other=0)
    tl.store(pos_ptr + idx, pos + off, mask=m_pos)

    # helix positions += pos_off; per-token write slot (-1 when another rank owns the
    # page) and attention bound; the last token of group s also sets the rank-local
    # kv length kv_lens[bound_start + s] to its bound.
    m_hx = idx < num_helix
    hp = tl.load(helix_pos_ptr + idx, mask=m_hx, other=0) + tl.load(
        pos_off_ptr + idx, mask=m_hx, other=0
    )
    tl.store(helix_pos_ptr + idx, hp, mask=m_hx)
    active = ((hp // phys) % cp_size) == cp_rank
    slot = tl.where(active, _helix_local_len(hp, phys, cp_size, cp_rank), -1)
    bound = _helix_local_len(hp + 1, phys, cp_size, cp_rank)
    tl.store(helix_slots_ptr + idx, slot, mask=m_hx)
    tl.store(helix_bounds_ptr + idx, bound, mask=m_hx)
    seq = idx // tokens_per_seq
    m_last = m_hx & (idx - seq * tokens_per_seq == tokens_per_seq - 1) & (seq < num_bound_seqs)
    tl.store(kv_lens_ptr + bound_start + seq, bound, mask=m_last)

    # kv_lens[kv_start + j] += kv_off[j] where no group bound replaces it.
    kv_i = kv_start + idx
    m_kv = (idx < kv_count) & ((kv_i < bound_start) | (kv_i >= bound_start + num_bound_seqs))
    kv = tl.load(kv_lens_ptr + kv_i, mask=m_kv, other=0)
    tl.store(kv_lens_ptr + kv_i, kv + tl.load(kv_off_ptr + idx, mask=m_kv, other=0), mask=m_kv)


def fused_helix_spec_preprocess(
    position_ids: torch.Tensor,
    pos_offsets: torch.Tensor,
    helix_positions: torch.Tensor,
    helix_slots: torch.Tensor,
    helix_bounds: torch.Tensor,
    kv_lens: torch.Tensor,
    kv_offsets: torch.Tensor,
    kv_start: int,
    bound_start: int,
    tokens_per_seq: int,
    phys: int,
    cp_size: int,
    cp_rank: int,
) -> None:
    """One kernel for the overlap correction + helix verify-group buffers of a decode step.

    Same result as, in order: ``position_ids += pos_offsets[:n]``,
    ``kv_lens[kv_start:kv_start + len(kv_offsets)] += kv_offsets``,
    ``helix_positions += pos_offsets[:len(helix_positions)]`` and
    ``TrtllmAttentionMetadata.recompute_helix_spec_buffers`` (write slots, attention
    bounds, and ``kv_lens[bound_start + s]`` = bound of the last token of group s).
    All buffers are contiguous int32 views; positions are non-negative.
    """
    num_pos = position_ids.numel()
    num_helix = helix_positions.numel()
    kv_count = kv_offsets.numel()
    assert num_helix % tokens_per_seq == 0
    num_bound_seqs = num_helix // tokens_per_seq
    assert pos_offsets.numel() >= max(num_pos, num_helix)
    assert helix_slots.numel() >= num_helix and helix_bounds.numel() >= num_helix
    total = max(num_pos, num_helix, kv_count, num_bound_seqs, 1)
    block = 256
    _helix_spec_preprocess_kernel[(_cdiv(total, block),)](
        position_ids,
        pos_offsets,
        num_pos,
        helix_positions,
        helix_slots,
        helix_bounds,
        num_helix,
        kv_lens,
        kv_offsets,
        kv_start,
        kv_count,
        bound_start,
        num_bound_seqs,
        tokens_per_seq,
        phys,
        cp_size,
        cp_rank,
        BLOCK=block,
    )


class _PinnedArena:
    """A pinned host byte buffer with typed numpy views for staging values."""

    def __init__(self, capacity: int):
        # The commit kernel reads the arena over the host link: it must be pinned
        # (InputCommit is only created where prefer_pinned()).
        self.tensor = torch.empty(capacity, dtype=torch.uint8, pin_memory=prefer_pinned())
        assert self.tensor.is_pinned(), "InputCommit needs pinned host memory"
        self.bytes = self.tensor.numpy()
        self.base = self.tensor.data_ptr()
        self.cursor = 0
        # Whole-arena typed views; staged values start at 16-byte aligned offsets.
        self.views = {
            np_dtype: self.bytes[: capacity - capacity % size].view(np_dtype)
            for np_dtype, size in _TORCH_TO_NUMPY.values()
        }

    @property
    def capacity(self) -> int:
        return self.bytes.shape[0]

    def alloc(self, nbytes: int) -> int:
        offset = self.cursor
        self.cursor = (offset + nbytes + 15) & ~15
        return offset


class InputCommit:
    """Collects one step's host->device input copies and writes them with one kernel.

    ``begin()`` opens a step, ``stage()`` records a copy (the host values are copied
    into a pinned arena immediately, so the caller may reuse its source right away),
    ``set_overlap_fixup()`` records the previous-step gathers, and ``commit()``
    launches the kernel on the current stream. Two arenas alternate between steps;
    an event recorded after each commit kernel guards an arena's reuse.
    """

    def __init__(self, capacity: int = 1 << 20):
        self._arenas = [_PinnedArena(capacity), _PinnedArena(capacity)]
        self._events: List[Optional[torch.cuda.Event]] = [None, None]
        self._index = 0
        self._arena = self._arenas[0]
        self._entries: List[Tuple[int, int, int]] = []
        self._fixup: Optional[dict] = None
        self._open = False
        self._verify_dsts: List[Tuple[torch.Tensor, int, int]] = []
        self._num_commits = 0

    @property
    def is_open(self) -> bool:
        return self._open

    def _verifying(self) -> bool:
        return _VERIFY_COMMITS < 0 or self._num_commits < _VERIFY_COMMITS

    def begin(self) -> None:
        assert not self._open, "InputCommit.begin() called twice without commit()"
        self._index ^= 1
        event = self._events[self._index]
        if event is not None:
            event.synchronize()
        self._arena = self._arenas[self._index]
        self._arena.cursor = 0
        self._entries.clear()
        self._fixup = None
        self._verify_dsts.clear()
        self._open = True

    def abort(self) -> None:
        self._entries.clear()
        self._fixup = None
        self._open = False

    def _reserve(self, nbytes: int) -> int:
        if self._arena.cursor + nbytes > self._arena.capacity:
            self._grow(self._arena.cursor + nbytes)
        return self._arena.alloc(nbytes)

    def _grow(self, needed: int) -> None:
        # Rare. Replaces the arena of the open step (nothing reads it yet) by a larger
        # one and rebases the host addresses recorded so far.
        old = self._arena
        new = _PinnedArena(max(2 * old.capacity, 2 * needed))
        new.bytes[: old.cursor] = old.bytes[: old.cursor]
        new.cursor = old.cursor
        delta = new.base - old.base
        self._entries = [(dst, src + delta, n) for dst, src, n in self._entries]
        if self._fixup is not None:
            self._fixup["slots_addr"] += delta
        self._arenas[self._index] = new
        self._arena = new

    def stage(self, dst: torch.Tensor, values: Union[Sequence, np.ndarray, torch.Tensor]) -> None:
        """Record ``dst.copy_(values)``; ``values`` has ``dst.numel()`` elements."""
        assert self._open
        assert dst.is_cuda and dst.is_contiguous(), "staged destinations must be contiguous"
        n = dst.numel()
        if n == 0:
            return
        np_dtype, itemsize = _TORCH_TO_NUMPY[dst.dtype]
        nbytes = n * itemsize
        offset = self._reserve(nbytes)
        arena = self._arena
        first = offset // itemsize
        view = arena.views[np_dtype][first : first + n]
        if isinstance(values, torch.Tensor):
            src = values.reshape(-1)
            if src.dtype != dst.dtype:
                src = src.to(dst.dtype)
            view[:] = src.numpy()
        else:
            view[:] = values
        self._entries.append((dst.data_ptr(), arena.base + offset, nbytes))
        if self._verifying():
            self._verify_dsts.append((dst, offset, nbytes))

    def stage_host_values(self, values: Sequence[int], dtype=np.int32) -> Tuple[int, int]:
        """Copy ``values`` into the arena without a destination; returns (address, offset)."""
        assert self._open
        arr = np.asarray(values, dtype=dtype)
        offset = self._reserve(max(arr.nbytes, 1))
        self._arena.bytes[offset : offset + arr.nbytes].view(dtype)[:] = arr
        return self._arena.base + offset, offset

    def set_overlap_fixup(
        self,
        *,
        previous_slots: Sequence[int],
        tokens_per_req: int,
        draft_width: int,
        new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_start: int,
        draft_tokens: torch.Tensor,
        draft_tokens_start: int,
        pos_offsets: torch.Tensor,
        pos_offsets_start: int,
        kv_offsets: torch.Tensor,
        kv_offsets_start: int,
    ) -> None:
        """Record the overlap-scheduler gathers of ``_prepare_tp_inputs``.

        ``new_tokens`` is ``[T_store, num_slots, 1]``, ``next_draft_tokens`` is
        ``[num_slots, D_store]``; ``pos_offsets`` / ``kv_offsets`` are zeroed except the
        ranges the gathers write.
        """
        assert self._open and self._fixup is None
        for t in (
            new_tokens,
            new_tokens_lens,
            next_draft_tokens,
            input_ids,
            draft_tokens,
            pos_offsets,
            kv_offsets,
        ):
            assert t.is_cuda and t.dtype == torch.int32
        assert input_ids.is_contiguous() and draft_tokens.is_contiguous()
        assert pos_offsets.is_contiguous() and kv_offsets.is_contiguous()
        assert new_tokens.dim() == 3 and next_draft_tokens.dim() == 2
        assert new_tokens.stride(2) == 1 or new_tokens.shape[2] == 1
        assert next_draft_tokens.stride(1) == 1
        num_prev = len(previous_slots)
        assert num_prev * tokens_per_req <= input_ids.numel() - input_ids_start
        assert num_prev * draft_width <= draft_tokens.numel() - draft_tokens_start
        slots_addr, _ = self.stage_host_values(previous_slots if num_prev else [0])
        self._fixup = dict(
            slots_addr=slots_addr,
            num_prev=num_prev,
            tokens_per_req=tokens_per_req,
            draft_width=draft_width,
            new_tokens=new_tokens,
            new_tokens_lens=new_tokens_lens,
            next_draft_tokens=next_draft_tokens,
            input_ids=input_ids,
            input_ids_start=input_ids_start,
            draft_tokens=draft_tokens,
            draft_tokens_start=draft_tokens_start,
            pos_offsets=pos_offsets,
            pos_offsets_start=pos_offsets_start,
            kv_offsets=kv_offsets,
            kv_offsets_start=kv_offsets_start,
        )

    def commit(self) -> None:
        """Launch the commit kernel on the current stream and close the step."""
        assert self._open
        entries = self._entries
        fixup = self._fixup
        if not entries and fixup is None:
            self._open = False
            return
        num_desc = sum(_cdiv(nbytes, _COPY_CHUNK) for _, _, nbytes in entries)
        # Reserving may grow the arena and rebase the recorded host addresses, so the
        # entries and the fixup are read again afterwards.
        desc_offset = self._reserve(max(24 * num_desc, 8))
        self._open = False
        entries = self._entries
        fixup = self._fixup
        arena = self._arena
        descs: List[int] = []
        for dst, src, nbytes in entries:
            for start in range(0, nbytes, _COPY_CHUNK):
                descs.extend((dst + start, src + start, min(_COPY_CHUNK, nbytes - start)))
        if descs:
            arena.bytes[desc_offset : desc_offset + 8 * len(descs)].view(np.int64)[:] = descs
        desc_tensor = arena.tensor[desc_offset : desc_offset + max(8 * len(descs), 8)].view(
            torch.int64
        )
        if fixup is not None:
            slots = _host_view(arena, fixup["slots_addr"], max(fixup["num_prev"], 1))
            num_fix = max(
                fixup["num_prev"] * fixup["tokens_per_req"],
                fixup["pos_offsets"].numel(),
                fixup["num_prev"] * fixup["draft_width"],
                fixup["kv_offsets"].numel(),
            )
            num_fix_programs = _cdiv(max(num_fix, 1), _FIX_BLOCK)
            new_tokens = fixup["new_tokens"]
            args = (
                slots,
                fixup["num_prev"],
                fixup["tokens_per_req"],
                fixup["draft_width"],
                new_tokens,
                new_tokens.stride(0),
                new_tokens.stride(1),
                fixup["new_tokens_lens"],
                fixup["next_draft_tokens"],
                fixup["next_draft_tokens"].stride(0),
                fixup["input_ids"],
                fixup["input_ids_start"],
                fixup["draft_tokens"],
                fixup["draft_tokens_start"],
                fixup["pos_offsets"],
                fixup["pos_offsets"].numel(),
                fixup["pos_offsets_start"],
                fixup["kv_offsets"],
                fixup["kv_offsets"].numel(),
                fixup["kv_offsets_start"],
            )
        else:
            num_fix_programs = 0
            dummy = desc_tensor
            args = (
                dummy,
                0,
                1,
                0,
                dummy,
                0,
                0,
                dummy,
                dummy,
                0,
                dummy,
                0,
                dummy,
                0,
                dummy,
                0,
                0,
                dummy,
                0,
                0,
            )
        _input_commit_kernel[(num_desc + num_fix_programs,)](
            desc_tensor,
            num_desc,
            *args,
            CHUNK=_COPY_CHUNK,
            FIX_BLOCK=_FIX_BLOCK,
            HAS_FIX=fixup is not None,
        )
        event = self._events[self._index]
        if event is None:
            event = torch.cuda.Event()
            self._events[self._index] = event
        event.record()
        if self._verifying():
            self._verify(fixup)
            if self._num_commits + 1 == _VERIFY_COMMITS:
                logger.info(f"InputCommit verify: the first {_VERIFY_COMMITS} commits matched")
        self._num_commits += 1
        self._entries = []
        self._fixup = None

    def _verify(self, fixup: Optional[dict]) -> None:
        torch.cuda.current_stream().synchronize()
        arena = self._arena
        for dst, offset, nbytes in self._verify_dsts:
            expected = torch.from_numpy(arena.bytes[offset : offset + nbytes].copy())
            got = dst.reshape(-1).view(torch.uint8).cpu()
            if not torch.equal(got, expected):
                raise RuntimeError(
                    f"InputCommit verify: staged copy mismatch for a {dst.dtype} "
                    f"destination of {dst.numel()} elements"
                )
        if fixup is None:
            return
        num_prev = fixup["num_prev"]
        tokens = fixup["tokens_per_req"]
        width = fixup["draft_width"]
        slots_view = _host_view(arena, fixup["slots_addr"], max(num_prev, 1))[:num_prev]
        slots = slots_view.to(device="cuda", dtype=torch.long)
        lens = fixup["new_tokens_lens"]
        ref_ids = fixup["new_tokens"].transpose(0, 1)[slots, :tokens].flatten()
        ref_draft = fixup["next_draft_tokens"][slots, :width].flatten()
        ref_pos = torch.zeros_like(fixup["pos_offsets"])
        start = fixup["pos_offsets_start"]
        ref_pos[start : start + num_prev * tokens] = lens[slots.repeat_interleave(tokens)]
        ref_kv = torch.zeros_like(fixup["kv_offsets"])
        start = fixup["kv_offsets_start"]
        ref_kv[start : start + num_prev] = lens[slots] - tokens
        ids_start = fixup["input_ids_start"]
        draft_start = fixup["draft_tokens_start"]
        checks = (
            ("input_ids", fixup["input_ids"][ids_start : ids_start + num_prev * tokens], ref_ids),
            (
                "draft_tokens",
                fixup["draft_tokens"][draft_start : draft_start + num_prev * width],
                ref_draft,
            ),
            ("pos_offsets", fixup["pos_offsets"], ref_pos),
            ("kv_offsets", fixup["kv_offsets"], ref_kv),
        )
        for name, got, ref in checks:
            if not torch.equal(got, ref.to(got.dtype)):
                raise RuntimeError(f"InputCommit verify: overlap fixup mismatch in {name}")


def warmup_input_commit_kernels(commit: Optional[InputCommit]) -> None:
    """Compile the kernels of this module ahead of serving (the commit is not used
    during engine warmup, so its first use would otherwise pay the JIT compile)."""
    device = torch.device("cuda", torch.cuda.current_device())
    if commit is not None:
        buf = torch.zeros(64, dtype=torch.int32, device=device)
        new_tokens = torch.zeros((2, 2, 1), dtype=torch.int32, device=device)
        lens = torch.ones(2, dtype=torch.int32, device=device)
        draft = torch.zeros((2, 1), dtype=torch.int32, device=device)
        pos_off = torch.zeros(4, dtype=torch.int32, device=device)
        kv_off = torch.zeros(2, dtype=torch.int32, device=device)
        for with_fixup in (False, True):
            commit.begin()
            commit.stage(buf[:4], [1, 2, 3, 4])
            if with_fixup:
                commit.set_overlap_fixup(
                    previous_slots=[0, 1],
                    tokens_per_req=2,
                    draft_width=1,
                    new_tokens=new_tokens,
                    new_tokens_lens=lens,
                    next_draft_tokens=draft,
                    input_ids=buf[16:32],
                    input_ids_start=0,
                    draft_tokens=buf[32:48],
                    draft_tokens_start=0,
                    pos_offsets=pos_off,
                    pos_offsets_start=0,
                    kv_offsets=kv_off,
                    kv_offsets_start=0,
                )
            commit.commit()
    if _BLOCK_OFFSETS_TRITON:
        host = torch.zeros((1, 2, 2, 4), dtype=torch.int32, pin_memory=prefer_pinned())
        dst = torch.zeros((1, 2, 2, 4), dtype=torch.int32, device=device)
        copy_idx = maybe_pin_memory(torch.tensor([1, 0], dtype=torch.int32))
        ones = maybe_pin_memory(torch.ones(1, dtype=torch.int32))
        copy_batch_block_offsets_from_host(host, dst, copy_idx, ones, ones)
    torch.cuda.synchronize()


def _host_view(arena: _PinnedArena, address: int, count: int) -> torch.Tensor:
    offset = address - arena.base
    return arena.tensor[offset : offset + 4 * count].view(torch.int32)

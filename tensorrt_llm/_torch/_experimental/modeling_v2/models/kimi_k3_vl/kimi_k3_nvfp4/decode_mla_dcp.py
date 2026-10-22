# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-read decode context parallelism (DCP) for Kimi-K3 MLA decode under TP.

Under plain TP every rank holds, and reads, the full latent KV of every request (MLA has a single
latent KV head), so the per-step KV read is replicated ``tp_size`` times. Split-read DCP keeps the
cache exactly as it is (every rank still writes every new token), but each rank attends with *all*
heads over only the pages it is responsible for, and the per-head partial results are exchanged and
combined:

    rank r: q for all heads  x  pages {r, r + R, r + 2R, ...} of each request  ->  partial O, (max, sum)
    all-to-all: rank s receives the partials of its own head group from every rank
    combine:    softmax-weighted sum over ranks (the Helix combine, ``trtllm::helix_post_process``)

Page ownership is the Helix round-robin ledger (page ``b`` belongs to rank ``b % R``), so the
per-token visible bounds are ``helix_local_len_tensor`` of the global causal bound. Only attention
reads are split; the KV cache manager, the cache write and disaggregated KV transfer are unchanged.

Under Helix context parallelism (``tp 1 x cp 16``, ``CpType.HELIX``) the same exchange runs over the CP
group, but the cache itself is sharded: the KV cache manager gives each rank only its round-robin pages
(``kv_cache_block_offsets`` lists them in order, ``kv_lens_cuda`` is the rank-local length) and the new
tokens' KV lands on their owner rank only (inside ``mla_rope_generation``). The page table is then read as
is, and the per-token bounds come from the global positions of the step's tokens
(``helix_position_offsets``); see ``make_helix_plan``.

This module only holds the per-rank pieces (page-table slicing, bounds, the kernel call and the
combine). The caller owns the collectives so the pieces can be tested on one GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.utils import helix_local_len_tensor

from . import flags as _flags
from . import mla_decode_pdl  # noqa: F401  (registers trtllm::k3_mla_decode_fp8_pdl)
from .flags import TRTLLM_K3LL_MLA_DCP

#: Latent and rope widths of Kimi-K3's absorbed MLA.
KV_LORA_RANK = 512
QK_ROPE_DIM = 64


@dataclass
class SplitReadPlan:
    """Per-step, per-rank inputs of the split-read MLA kernel. All tensors are device tensors.

    page_table: ``[local_blocks, batch]`` int32, this rank's pages of each request, a transposed view of a
        contiguous ``[batch, local_blocks]`` buffer (the kernel is compiled for a dim-0-major page table).
    cache_seqs: ``[batch]`` int32, number of tokens of each request held in this rank's pages.
    kv_bounds: ``[batch * q_len]`` int32, per query token, the number of this rank's tokens it may
        attend to (its causal bound restricted to this rank's pages).
    empty: ``[batch * q_len]`` bool, tokens that see no key on this rank (their partials are
        undefined and must be replaced by the identity of the combine).
    """

    page_table: torch.Tensor
    cache_seqs: torch.Tensor
    kv_bounds: Optional[torch.Tensor]
    empty: Optional[torch.Tensor]


def local_block_index(max_blocks: int, cp_size: int, cp_rank: int, device) -> torch.Tensor:
    """Column indices ``cp_rank, cp_rank + cp_size, ...`` of a ``[batch, max_blocks]`` block table."""
    return torch.arange(cp_rank, max_blocks, cp_size, dtype=torch.long, device=device)


def make_split_read_plan(
    block_rows: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    page_size: int,
    cp_size: int,
    cp_rank: int,
    block_index: Optional[torch.Tensor] = None,
) -> SplitReadPlan:
    """Build this rank's page table and bounds from the full (replicated) block table.

    Args:
        block_rows: ``[batch, max_blocks]`` page ids of each request's global pages, in order.
        seq_lens: ``[batch]`` global KV length of each request *including* the ``q_len`` new tokens.
        q_len: query tokens per request (1 + draft length); they are the last ``q_len`` positions.
    """
    if block_index is None:
        block_index = local_block_index(block_rows.shape[1], cp_size, cp_rank, block_rows.device)
    # [local_blocks, batch] view of a contiguous [batch, local_blocks] buffer: the kernel's page table is dim-0-major
    page_table = block_rows.index_select(1, block_index).to(torch.int32).contiguous().t()
    seq_lens = seq_lens.to(torch.int32)
    cache_seqs = helix_local_len_tensor(seq_lens, page_size, cp_size, cp_rank).to(torch.int32)
    offsets = torch.arange(q_len, dtype=torch.int32, device=seq_lens.device)
    token_ends = (seq_lens.view(-1, 1) - (q_len - 1) + offsets.view(1, -1)).reshape(-1)
    kv_bounds = helix_local_len_tensor(token_ends, page_size, cp_size, cp_rank).to(torch.int32)
    return SplitReadPlan(page_table, cache_seqs, kv_bounds, kv_bounds == 0)


def make_helix_plan(
    local_block_rows: torch.Tensor,
    local_lens: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    cp_size: int,
    cp_rank: int,
) -> SplitReadPlan:
    """This rank's page table and bounds when the cache is Helix-sharded (only this rank's pages stored).

    Args:
        local_block_rows: ``[batch, local_blocks]`` page ids of this rank's pages of each request, in order (page
            ``j`` of the row holds the request's global page ``j * cp_size + cp_rank``).
        local_lens: ``[batch]`` number of tokens of each request held on this rank, the step's new tokens this
            rank owns included.
        positions: ``[batch * q_len]`` global position of every query token of the step.
    """
    page_table = local_block_rows.to(torch.int32).contiguous().t()
    kv_bounds = helix_local_len_tensor(
        positions.to(torch.int32) + 1, page_size, cp_size, cp_rank
    ).to(torch.int32)
    return SplitReadPlan(page_table, local_lens.to(torch.int32), kv_bounds, kv_bounds == 0)


def workspace_bytes(num_heads: int, q_len: int, max_batch: int) -> int:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        num_heads, q_len, KV_LORA_RANK, max_batch, cutlass.Float32
    )


def split_read_attention(
    q: torch.Tensor,
    kv_pages: torch.Tensor,
    plan: SplitReadPlan,
    out: torch.Tensor,
    softmax_stats: Optional[torch.Tensor],
    workspace: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
    max_batch: int,
) -> None:
    """All-head FP8 MLA decode over this rank's pages.

    Args:
        q: ``[batch, q_len, heads, 576]`` FP8 (e4m3) absorbed queries of *all* heads.
        kv_pages: ``[num_pages, page_size, >=576]`` FP8 view of the layer's page pool.
        out: ``[batch, q_len, heads, 512]`` bf16, normalized partial output (written).
        softmax_stats: ``[batch * q_len, heads, 2]`` fp32 ``(max, sum)`` per row (written).
    """
    batch, q_len, heads, _ = q.shape
    c_latent = kv_pages[..., :KV_LORA_RANK].permute(1, 2, 0)
    c_rope = kv_pages[..., KV_LORA_RANK : KV_LORA_RANK + QK_ROPE_DIM].permute(1, 2, 0)
    # TRTLLM_K3LL_MLA_DECODE_PDL: the same op with programmatic dependent launch (mla_decode_pdl.py)
    op = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
    if _flags.TRTLLM_K3LL_MLA_DECODE_PDL:
        from . import census as _census

        _census.count(f"mla_decode_pdl:{_census.mode()}")
        op = torch.ops.trtllm.k3_mla_decode_fp8_pdl
    op(
        q[..., :KV_LORA_RANK].permute(2, 3, 1, 0),
        q[..., KV_LORA_RANK:].permute(2, 3, 1, 0),
        c_latent,
        c_rope,
        plan.page_table,
        plan.cache_seqs,
        out.permute(2, 3, 1, 0),
        workspace,
        heads,
        q_len,
        kv_pages.shape[1],
        softmax_scale,
        output_scale,
        max_batch,
        softmax_stats,
        plan.kv_bounds,
    )


def sanitize_empty(out: torch.Tensor, softmax_stats: torch.Tensor, empty: torch.Tensor) -> None:
    """Make rows without local keys the identity of the combine, in place: O = 0, (max, sum) = (-inf, 0)."""
    rows = out.view(empty.shape[0], -1)
    rows.masked_fill_(empty.view(-1, 1), 0.0)
    softmax_stats[..., 0].masked_fill_(empty.view(-1, 1), float("-inf"))
    softmax_stats[..., 1].masked_fill_(empty.view(-1, 1), 0.0)


def combine(gathered_o: torch.Tensor, gathered_stats: torch.Tensor) -> torch.Tensor:
    """Softmax-weighted combine of per-rank partials.

    Args:
        gathered_o: ``[cp, tokens, heads * 512]`` bf16 normalized partials.
        gathered_stats: ``[cp, tokens, heads, 2]`` fp32 ``(max, sum)``.
    Returns:
        ``[tokens, heads * 512]`` bf16.
    """
    return torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)


# ----------------------------------------------------------------------------------------------------------------------
# Helix: the exchange on MNNVL Lamport buffers (TRTLLM_K3LL_HELIX_Q_AG_MNNVL, TRTLLM_K3LL_HELIX_A2A_FUSED).
# ----------------------------------------------------------------------------------------------------------------------

#: Bytes of one rank's FP8 absorbed queries per token and head.
Q_BYTES = KV_LORA_RANK + QK_ROPE_DIM
#: Values per head of the fused exchange: a 512-wide latent head travels as two 256-wide halves that carry the same
#: (max, sum), which the fused kernel combines exactly as one head (its weights depend on the statistics only).
FUSED_HEAD_DIM = 256
_HALVES = KV_LORA_RANK // FUSED_HEAD_DIM


def gather_q_mnnvl(q_local: torch.Tensor, workspace) -> torch.Tensor:
    """All-gather of every rank's FP8 absorbed queries over ``workspace`` (the TP group's ``MnnvlWorkspace``, rank i
    = CP rank i): one ``comm/mnnvl_allgather_split`` call that carries the bytes as fp32 words.

    Args:
        q_local: ``[tokens, heads_per_rank, 576]`` FP8 (or uint8) bytes of this rank's heads, contiguous.
    Returns:
        ``[tokens, world * heads_per_rank, 576]`` uint8, the ranks' heads in rank order (``allgather`` along dim 1),
        except that a 4-byte group whose word is the fp32 ``-0.0`` (FP8 bytes ``0, 0, 0, -0.0``) arrives as zeros:
        the Lamport buffers' empty word never travels. That is one FP8 ``-0.0`` made ``+0.0``, the same value.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
        mnnvl_allgather_split,
    )

    tokens, heads = q_local.shape[:2]
    words = q_local.reshape(tokens, heads * Q_BYTES).view(torch.uint8).view(torch.float32)
    _, gathered = mnnvl_allgather_split(words, 0, workspace)
    return gathered.view(torch.uint8).view(tokens, workspace.world_size * heads, Q_BYTES)


def fused_supports(cp_size: int, heads_per_rank: int) -> bool:
    """Whether ``exchange_combine_fused`` takes this split (the fused kernel's support at 256-wide halves)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.helix_a2a_fused import supports

    return supports(cp_size, heads_per_rank * _HALVES, FUSED_HEAD_DIM)


def fused_buffer_bytes(cp_size: int, heads_per_rank: int, max_tokens: int) -> int:
    """Receive buffer per rank (two halves) that holds a step of ``max_tokens`` tokens in one launch."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.helix_a2a_fused import record_bytes

    half = max_tokens * cp_size * record_bytes(heads_per_rank * _HALVES, FUSED_HEAD_DIM)
    return 2 * (-(-half // (1 << 20)) << 20)


def exchange_combine_fused(
    out: torch.Tensor, softmax_stats: torch.Tensor, empty: torch.Tensor, fused, heads_per_rank: int
) -> torch.Tensor:
    """The exchange of every head group's partials to its owner and their combine, in one ``helix_a2a_fused`` launch.

    Args:
        out: ``[tokens, cp * heads_per_rank * 512]`` bf16 normalized partials of every head (head group s, the heads
            of rank s, at columns ``[s * heads_per_rank * 512, ...)``), contiguous.
        softmax_stats: ``[tokens, cp * heads_per_rank, 2]`` fp32 ``(max, sum)``.
        empty: ``[tokens]`` bool, tokens that see no key on this rank; their partials need not be sanitized (the
            sender sends the combine's identity for them).
        fused: the CP group's ``HelixAllToAllFused``.
    Returns:
        ``[tokens, heads_per_rank * 512]`` bf16, the combined output of this rank's heads (what ``combine`` returns
        after the all-to-all).
    """
    tokens = out.shape[0]
    heads = softmax_stats.shape[1]
    halves = (
        softmax_stats.view(tokens, heads, 1, 2)
        .expand(tokens, heads, _HALVES, 2)
        .reshape(tokens, -1)
    )
    return fused(
        out.view(tokens, -1),
        halves,
        heads_per_rank * _HALVES,
        FUSED_HEAD_DIM,
        zero_kv_mask=empty,
    )


def default_softmax_scale(qk_nope_head_dim: int = 128, q_scaling: float = 1.0) -> float:
    return 1.0 / (math.sqrt(qk_nope_head_dim + QK_ROPE_DIM) * q_scaling)


# ----------------------------------------------------------------------------------------------------------------------
# The target-side runtime: the TP group doubles as the CP group of the split read.
# ----------------------------------------------------------------------------------------------------------------------

_LOG2_E = 1.4426950408889634
PAGE_SIZE = 64
#: Workspace / autotuner sizing: the decode step contract's ceilings (requests x tokens per request).
MAX_REQUESTS = 8
MAX_Q_LEN = 8


class MlaSplitRead:
    """The FP8-cache MLA decode attention of a target's MLA layers (one instance shared by all of them).

    Replaces the stock generation attention of ``K3DecodeMLA._quantized_decode`` after ``mla_rope_generation`` (which
    appended the new tokens to this rank's full copy of the cache and wrote the FP8 absorbed queries):

    * mode 1: ``trtllm::cute_dsl_mla_decode_fp8_blackwell`` on this rank's heads over the full KV, with the kernel's
      own causal mask. No collective, no per-step preparation beyond the stock page-table view.
    * mode 2 (split-read DCP over the TP group): all-gather the queries, attend with every head over this rank's
      round-robin pages, all-to-all each head group's partials to its owner, combine. The page table and bounds of
      this rank are built once per step, by the first MLA layer, and reused by the others (every MLA layer of a step
      sees the same requests and lengths). Every rank of the group must make the same calls in the same order (the
      collectives are NCCL, graph-capturable).
    * Helix (``mapping`` has Helix CP; any nonzero mode): mode 2's exchange over the CP group on the Helix-sharded
      cache (``make_helix_plan``). ``gather_mapping`` is a TP mapping over the same ranks (the MLA's row-parallel
      output mapping), whose TP rank equals the CP rank, for the query all-gather.
    """

    _shared: dict = {}

    @classmethod
    def shared(cls, mapping, heads_per_rank: int, gather_mapping=None) -> "MlaSplitRead":
        key = (id(mapping), heads_per_rank)
        if key not in cls._shared:
            cls._shared[key] = cls(mapping, heads_per_rank, gather_mapping=gather_mapping)
        return cls._shared[key]

    def __init__(
        self, mapping, heads_per_rank: int, mode: int = TRTLLM_K3LL_MLA_DCP, gather_mapping=None
    ):
        if mode not in (1, 2):
            raise ValueError(f"TRTLLM_K3LL_MLA_DCP must be 0, 1 or 2, got {mode}")
        self.mode = mode
        self.helix = bool(mapping.has_cp_helix())
        if self.helix:
            if gather_mapping is None or (gather_mapping.tp_size, gather_mapping.tp_rank) != (
                mapping.cp_size,
                mapping.cp_rank,
            ):
                raise ValueError(
                    "Helix MLA decode needs a TP mapping over the CP group whose TP rank is the CP rank; got "
                    f"cp {mapping.cp_rank}/{mapping.cp_size}, gather mapping "
                    f"{None if gather_mapping is None else (gather_mapping.tp_rank, gather_mapping.tp_size)}"
                )
            self.cp = int(mapping.cp_size)
            self.rank = int(mapping.cp_rank)
            self.group = list(mapping.cp_group)
            self.gather_mapping = gather_mapping
        else:
            self.cp = int(mapping.tp_size) if mode == 2 else 1
            self.rank = int(mapping.tp_rank) if mode == 2 else 0
            self.group = list(mapping.tp_group)
            self.gather_mapping = mapping
        self.mapping = mapping
        self.heads_per_rank = heads_per_rank
        self.heads = heads_per_rank * self.cp
        self._scales: dict = {}
        self._workspace: Optional[torch.Tensor] = None
        self._block_index: dict = {}
        self._layers: set = set()
        self._first_layer = -1
        self._plan: Optional[SplitReadPlan] = None
        self._logged = False
        # Helix exchange state on MNNVL (build_helix_comm); None: the NCCL collectives.
        self._q_gather_ws = None
        self._a2a_fused = None

    @staticmethod
    def applies(mqa, mapping) -> bool:
        return bool(
            TRTLLM_K3LL_MLA_DCP
            and (TRTLLM_K3LL_MLA_DCP == 1 or mapping.tp_size > 1 or mapping.has_cp_helix())
            and getattr(mqa, "has_fp8_kv_cache", False)
            and not getattr(mqa, "has_fp4_kv_cache", False)
        )

    def build_helix_comm(self, mnnvl_workspace, tp_mapping) -> str:
        """Build the Helix exchange's MNNVL state the load-time switches ask for. Collective over the CP group, eager,
        before any capture; every rank of the group calls it at the same point. Returns what it built (for the log).

        * ``TRTLLM_K3LL_HELIX_Q_AG_MNNVL``: the query all-gather runs on ``mnnvl_workspace``, the decode collectives'
          workspace over ``tp_mapping`` (the repurposed TP16 mapping), when its group and ranks are the CP group's.
        * ``TRTLLM_K3LL_HELIX_A2A_FUSED``: a ``HelixAllToAllFused`` of the CP mapping sized for the decode step
          contract's ceiling.

        The forward reads the switches at run time and falls back to NCCL where one is 0 (or its state was not
        built)."""
        if not self.helix:
            return "not Helix"
        built = []
        if _flags.TRTLLM_K3LL_HELIX_Q_AG_MNNVL and mnnvl_workspace is not None:
            from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
                required_buffer_bytes,
            )

            need = required_buffer_bytes(
                MAX_REQUESTS * MAX_Q_LEN, 0, self.heads_per_rank * Q_BYTES // 4, self.cp
            )
            if (
                mnnvl_workspace.world_size == self.cp
                and mnnvl_workspace.rank == self.rank
                and list(tp_mapping.tp_group) == self.group
                and need <= mnnvl_workspace.buffer_bytes
            ):
                self._q_gather_ws = mnnvl_workspace
                built.append("q all-gather on MNNVL")
        if _flags.TRTLLM_K3LL_HELIX_A2A_FUSED and fused_supports(self.cp, self.heads_per_rank):
            from tensorrt_llm._torch.distributed.helix_a2a_fused import HelixAllToAllFused

            self._a2a_fused = HelixAllToAllFused.get(
                self.mapping,
                fused_buffer_bytes(self.cp, self.heads_per_rank, MAX_REQUESTS * MAX_Q_LEN),
            )
            built.append(
                f"fused exchange + combine ({self._a2a_fused.half_bytes >> 20} MiB halves)"
            )
        return ", ".join(built) or "NCCL only"

    def register(self, layer_idx: int) -> None:
        self._layers.add(layer_idx)
        self._first_layer = min(self._layers)

    @staticmethod
    def _layer_inputs(mqa, meta, num_requests: int):
        """FP8 page pool ``[pages, 64, >=576]``, global block rows ``[R, max_blocks]``, this layer's page offset in
        an interleaved pool, and lengths (the stock CuTe DSL MLA backend's normalization)."""
        kv_pool = meta.kv_cache_manager.get_buffers(mqa.layer_idx)
        packed_block = 1
        for size in kv_pool.shape[1:]:
            packed_block *= size
        block_stride = kv_pool.stride(0)
        layers_in_pool = block_stride // packed_block if packed_block else 1
        layer_in_pool = 0
        if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
            layer_in_pool = kv_pool.storage_offset() // packed_block
            kv_pool = kv_pool.as_strided(
                (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
                (packed_block, *kv_pool.stride()[1:]),
                0,
            )
        pool = kv_pool.view(torch.float8_e4m3fn)
        if (
            pool.dim() != 5
            or pool.shape[1] != 1
            or pool.shape[3] != 1
            or pool.shape[2] != PAGE_SIZE
        ):
            raise RuntimeError(
                f"MLA decode expects [pages, 1, 64, 1, D] FP8 pages, got {tuple(pool.shape)}"
            )
        pool_idx = int(meta.host_kv_cache_pool_mapping[mqa.get_local_layer_idx(meta), 0])
        gen = slice(meta.num_contexts, meta.num_contexts + num_requests)
        block_rows = meta.kv_cache_block_offsets[pool_idx, gen, 0, :]
        return (
            pool[:, 0, :, 0, : KV_LORA_RANK + QK_ROPE_DIM],
            block_rows,
            int(layer_in_pool),
            meta.kv_lens_cuda_runtime[gen],
        )

    def _layer_scales(self, layer_idx: int, bmm1: torch.Tensor, bmm2: torch.Tensor):
        scales = self._scales.get(layer_idx)
        if scales is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"MLA decode: FP8 scales of layer {layer_idx} not cached before capture"
                )
            scales = (float(bmm1[1].item()) / _LOG2_E, float(bmm2[0].item()))
            self._scales[layer_idx] = scales
        return scales

    def _step_plan(self, layer_idx, block_rows, lens, q_len, meta=None) -> SplitReadPlan:
        if self._plan is None or layer_idx == self._first_layer:
            if self.helix:
                tokens = block_rows.shape[0] * q_len
                self._plan = make_helix_plan(
                    block_rows,
                    lens,
                    meta.helix_position_offsets[:tokens],
                    PAGE_SIZE,
                    self.cp,
                    self.rank,
                )
                return self._plan
            index = self._block_index.get(block_rows.shape[1])
            if index is None:
                index = local_block_index(
                    block_rows.shape[1], self.cp, self.rank, block_rows.device
                )
                self._block_index[block_rows.shape[1]] = index
            self._plan = make_split_read_plan(
                block_rows, lens, q_len, PAGE_SIZE, self.cp, self.rank, index
            )
        return self._plan

    def forward(
        self,
        mqa,
        quant_q: torch.Tensor,
        meta,
        q_len: int,
        bmm1: torch.Tensor,
        bmm2: torch.Tensor,
        all_heads: bool = False,
    ) -> torch.Tensor:
        """``quant_q``: ``[tokens, heads_per_rank, 576]`` FP8 bytes of this rank's heads, or with ``all_heads``
        (Helix only) ``[tokens, heads, 576]`` of every head, computed on this rank, which then needs no query
        all-gather. Returns the attention latent of this rank's heads, ``[tokens, heads_per_rank * 512]`` bf16 (what
        the stock attention returns)."""
        num_requests = meta.num_generations
        tokens = num_requests * q_len
        hp, cp = self.heads_per_rank, self.cp
        if all_heads and cp == 1:
            raise ValueError("all_heads queries are a Helix split read's")
        q_local = (
            quant_q[:tokens]
            .view(torch.float8_e4m3fn)
            .view(tokens, self.heads if all_heads else hp, KV_LORA_RANK + QK_ROPE_DIM)
        )
        pages, block_rows, layer_in_pool, lens = self._layer_inputs(mqa, meta, num_requests)
        if self._workspace is None:
            self._workspace = torch.empty(
                workspace_bytes(self.heads, MAX_Q_LEN, MAX_REQUESTS),
                dtype=torch.uint8,
                device=quant_q.device,
            )
        softmax_scale, output_scale = self._layer_scales(mqa.layer_idx, bmm1, bmm2)
        out = torch.empty(
            num_requests,
            q_len,
            self.heads,
            KV_LORA_RANK,
            dtype=torch.bfloat16,
            device=quant_q.device,
        )
        if not self._logged and self.rank == 0:
            from tensorrt_llm.logger import logger

            logger.info(
                f"K3 MLA decode TRTLLM_K3LL_MLA_DCP={self.mode} helix={self.helix}: cp={cp} heads={self.heads} "
                f"({hp}/rank) "
                f"layers={sorted(self._layers)} requests={num_requests} q_len={q_len}"
            )
            self._logged = True

        if cp == 1:
            # the stock backend's operands: a [blocks, R] view of the block rows, global lengths, built-in causal mask
            page_table = block_rows.t()
            if layer_in_pool:
                page_table = page_table + layer_in_pool
            plan = SplitReadPlan(page_table, lens, None, None)
            split_read_attention(
                q_local.view(num_requests, q_len, hp, KV_LORA_RANK + QK_ROPE_DIM),
                pages,
                plan,
                out,
                None,
                self._workspace,
                softmax_scale,
                output_scale,
                MAX_REQUESTS,
            )
            return out.view(tokens, hp * KV_LORA_RANK)

        from tensorrt_llm._torch.distributed.ops import allgather, alltoall_helix

        if all_heads:
            q_all = q_local
        elif self._q_gather_ws is not None and _flags.TRTLLM_K3LL_HELIX_Q_AG_MNNVL:
            q_all = gather_q_mnnvl(q_local.view(torch.uint8), self._q_gather_ws)
        else:
            q_all = allgather(q_local.view(torch.uint8), self.gather_mapping, dim=1)
        q_all = q_all.view(torch.float8_e4m3fn).view(
            num_requests, q_len, self.heads, KV_LORA_RANK + QK_ROPE_DIM
        )
        plan = self._step_plan(mqa.layer_idx, block_rows, lens, q_len, meta)
        if layer_in_pool:
            plan = SplitReadPlan(
                plan.page_table + layer_in_pool, plan.cache_seqs, plan.kv_bounds, plan.empty
            )
        stats = torch.empty(tokens, self.heads, 2, dtype=torch.float32, device=quant_q.device)
        split_read_attention(
            q_all,
            pages,
            plan,
            out,
            stats,
            self._workspace,
            softmax_scale,
            output_scale,
            MAX_REQUESTS,
        )
        if self._a2a_fused is not None and _flags.TRTLLM_K3LL_HELIX_A2A_FUSED:
            return exchange_combine_fused(
                out.view(tokens, -1), stats, plan.empty, self._a2a_fused, hp
            )
        sanitize_empty(out, stats, plan.empty)
        # head group s -> rank s: [cp, tokens, hp * 512] partials and [cp, tokens, hp, 2] stats, one chunk per rank
        o_chunks = out.view(tokens, cp, hp * KV_LORA_RANK).transpose(0, 1).contiguous()
        s_chunks = stats.view(tokens, cp, hp, 2).transpose(0, 1).contiguous()
        o_recv, s_recv = alltoall_helix([*o_chunks.unbind(0), *s_chunks.unbind(0)], self.group)
        return combine(o_recv, s_recv)

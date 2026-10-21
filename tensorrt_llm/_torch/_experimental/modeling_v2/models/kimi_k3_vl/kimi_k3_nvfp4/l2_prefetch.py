# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch of the next decoder layer's attention weights during a MoE layer's latency-bound tail
(``TRTLLM_K3LL_L2_PREFETCH``).

At decode batch sizes a MoE layer streams its routed experts from HBM, then runs kernels that leave HBM mostly idle
(the latent all-reduce, the sandwich tail's all-reduce and residual update), and the next layer starts with its
weight-heavy attention projections (KDA's FP8 QKVG, 22 MB per rank at TP16; MLA's q_a / kv_a and q_b). Right after
the routed experts are issued, a side stream waits for the main stream and issues ``k3_l2_prefetch``
(``cp.async.bulk.prefetch.L2``) for every weight of the next layer's attention module, so its projections read them
from L2. The FP8 projection kernels load their weights with an evict-first hint, so the prefetched lines survive the
expert streams in between (see ``k3_l2_prefetch``).

Hints only: nothing waits for the prefetches and no result depends on them (bit-exact). The main stream joins the
side stream once per forward (``join``), after the last layer, so a captured graph has no dangling branch.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn

#: Smallest tensor worth a prefetch range (norm weights and biases are not).
MIN_BYTES = 64 << 10
#: Most bytes prefetched for one layer: well inside L2, and inside the MoE tail's idle HBM time at ~4 TB/s.
MAX_TOTAL_BYTES = 64 << 20


def _weights(sub: nn.Module):
    """The weights a module's forward reads: its parameters and its weight / scale tensor attributes (not buffers such
    as a rotary cos / sin cache, which can exceed L2)."""
    yield from sub._parameters.values()
    for name, v in vars(sub).items():
        if isinstance(v, torch.Tensor) and ("weight" in name or "scale" in name):
            yield v


def module_ranges(
    module: nn.Module, min_bytes: int = MIN_BYTES, max_total: int = MAX_TOTAL_BYTES
) -> List[Tuple[int, int]]:
    """``(address, bytes)`` of every CUDA weight of at least ``min_bytes`` that ``module`` or a submodule holds
    (``_weights``), each once, largest first, 16-byte aligned, at most ``max_total`` bytes in all."""
    seen = set()
    ranges = []
    for sub in module.modules():
        for t in _weights(sub):
            if t is None or not t.is_cuda or not t.is_contiguous():
                continue
            addr = t.data_ptr()
            nbytes = t.numel() * t.element_size()
            if addr in seen or nbytes < min_bytes:
                continue
            seen.add(addr)
            start = (addr + 15) & ~15
            end = (addr + nbytes) & ~15
            if end > start:
                ranges.append((start, end - start))
    ranges.sort(key=lambda r: -r[1])
    kept, total = [], 0
    for r in ranges:
        if total + r[1] <= max_total:
            kept.append(r)
            total += r[1]
    return kept


class K3L2Prefetcher:
    """The side stream of the prefetch launches, and whether it holds work the main stream has not joined."""

    def __init__(self, device: torch.device):
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
            k3_expert_l2_prefetch,
            k3_l2_prefetch,
        )

        self._dense = k3_l2_prefetch
        self._experts = k3_expert_l2_prefetch
        self.stream = torch.cuda.Stream(device=device)
        self.pending = False

    def new_record(self, device: torch.device) -> torch.Tensor:
        """A routing record (``k3_expert_l2_prefetch``) for one MoE layer."""
        return self._experts.new_record(device)

    def expert_ranges(self, tensors: Sequence[torch.Tensor], num_local: int) -> list:
        """``(base, stride, bytes)`` of every local expert's slice of each of ``tensors``."""
        return self._experts.expert_ranges(tensors, num_local)

    def _fork(self) -> None:
        self.stream.wait_stream(torch.cuda.current_stream())
        self.pending = True

    def issue(self, ranges: Sequence[Tuple[int, int]], record=None) -> None:
        """Once the main stream's work so far is done, on the side stream: prefetch ``ranges``, and with ``record`` =
        ``(routing record, moe_sort outputs, local experts)`` copy that routing into the record."""
        if not ranges and record is None:
            return
        self._fork()
        per_launch = self._dense.MAX_RANGES
        with torch.cuda.stream(self.stream):
            for i in range(0, len(ranges), per_launch):
                self._dense.run_k3_l2_prefetch(list(ranges[i : i + per_launch]))
            if record is not None:
                rec, sort_outputs, num_local = record
                for t in sort_outputs:
                    t.record_stream(self.stream)
                self._experts.run_k3_expert_l2_prefetch(
                    num_local=num_local, record_out=rec, sort_outputs=sort_outputs
                )

    def prefetch_experts(
        self, record: torch.Tensor, ranges: Sequence, num_local: int, budget: int
    ) -> None:
        """Once the main stream's work so far is done, on the side stream: prefetch the weights (``ranges``) of the
        ``budget`` experts with the most tokens in ``record``."""
        self._fork()
        with torch.cuda.stream(self.stream):
            self._experts.run_k3_expert_l2_prefetch(
                num_local=num_local, record_in=record, ranges=ranges, budget=budget
            )

    def join(self) -> None:
        """Make the main stream wait for the side stream's launches (once per forward)."""
        if self.pending:
            torch.cuda.current_stream().wait_stream(self.stream)
            self.pending = False

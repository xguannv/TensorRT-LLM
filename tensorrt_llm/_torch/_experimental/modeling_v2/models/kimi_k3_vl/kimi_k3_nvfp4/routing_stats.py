# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode routing statistics (``TRTLLM_K3LL_ROUTING_STATS_DIR``, flags.py), for offline expert placement.

Rank 0 counts, for every forward step the lowm MoE decode path runs (``decode_moe_lowm.py``: the front's routing at
up to 32 tokens, ``k3_route_quant`` on a wide step the front does not take), the tokens routed to each checkpoint
expert of each MoE layer: all tokens, and with ``TRTLLM_K3LL_ROUTING_STATS_STRIDE`` (tokens per request, 1 + draft
length) also the first token of each request. Every rank of the TP group routes the same tokens the same way, so one
rank's counts are the model's. Routed ids are slot ids; under ``TRTLLM_K3LL_EXPERT_PLACEMENT`` a per-layer table maps
them back to checkpoint experts, so the counts never depend on the placement that ran.

The counts accumulate on the GPU into a ring of steps indexed by a device step counter (the first MoE layer the path
takes starts a step), so they are captured into the CUDA graphs. Once per executor iteration
(``ExpertStatistic.register_iteration_callback``) the host snapshots the counter on the compute stream, copies the
ring on a side stream into pinned memory and collects the steps of earlier copies that have finished, never
synchronizing; a writer thread stores ``routing_<host>_<pid>_<n>.npz`` chunks every
``TRTLLM_K3LL_ROUTING_STATS_CHUNK`` steps, every 30 s and at exit. A chunk holds ``counts`` uint16
``[steps, planes, num_layers, num_experts]`` (plane 0 all tokens, plane 1 the first token of each request), ``step``,
``num_tokens`` and the executor ``iteration`` of each step, and the running ``totals`` int64
``[planes, num_layers, num_experts]`` over every step collected so far. Steps before the first executor iteration
(warm-up, graph capture) are not collected.

Cost when on: a gather and an ``index_add_`` per MoE layer on rank 0 (so rank 0 runs slower: never time with it on),
a ~21 MB ring per plane. Off (the default) nothing is built and nothing runs.
"""

from __future__ import annotations

import atexit
import os
import queue
import socket
import threading
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger


class RoutingStats:
    """Per-step, per-layer checkpoint-expert token counts of the routed MoE (see the module docstring)."""

    RING_STEPS = 64
    # Executor iterations between ring copies; well below RING_STEPS so no finished step is overwritten first.
    COPY_INTERVAL = 16
    CHUNK_SECONDS = 30.0
    _HOST_BUFFERS = 2

    def __init__(
        self,
        out_dir: str,
        num_layers: int,
        num_experts: int,
        first_layer: int,
        expert_of_slot: Dict[int, Sequence[int]],
        stride: int = 0,
        chunk_steps: int = 1024,
    ) -> None:
        """``first_layer``: the MoE layer whose ``record`` starts a step. ``expert_of_slot``: the placement's tables
        by layer (layers not in it are the identity)."""
        self.out_dir = out_dir
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.first_layer = first_layer
        self.stride = max(0, stride)
        self.chunk_steps = max(1, chunk_steps)
        self.planes = 2 if self.stride > 0 else 1
        self._plane_numel = num_layers * num_experts
        self._step_numel = self.planes * self._plane_numel
        # (layer, slot) -> offset of (layer, checkpoint expert) in a step's plane.
        lut = np.arange(self._plane_numel, dtype=np.int64).reshape(num_layers, num_experts)
        for layer, table in expert_of_slot.items():
            if 0 <= layer < num_layers:
                lut[layer] = layer * num_experts + np.asarray(table, dtype=np.int64)
        self._lut_host = torch.from_numpy(lut.reshape(-1))
        # Device buffers are allocated at the first ``record`` (eager, before any capture).
        self._allocated = False
        self._pending: list = []  # (host buffer, iteration) in flight
        self._next_buffer = 0
        self._calls = 0
        self._call_iterations: list = []
        self._warmup_steps = 0
        self._last_step: Optional[int] = None
        self._dropped_steps = 0
        self._totals = np.zeros((self.planes, num_layers, num_experts), dtype=np.int64)
        self._num_steps = 0
        self._chunk_counts: list = []
        self._chunk_meta: list = []  # (step, num_tokens, executor iteration)
        self._chunk_index = 0
        self._last_flush = time.monotonic()
        self._tag = f"{socket.gethostname()}_{os.getpid()}"
        os.makedirs(out_dir, exist_ok=True)
        self._writes: "queue.Queue[Optional[Tuple[str, dict]]]" = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        atexit.register(self.close)

    # ---- device side (captured into the CUDA graphs) ----

    def _allocate(self, device: torch.device) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the first RoutingStats.record() must run eagerly, not under a capture"
            )
        r = self.RING_STEPS
        with torch.inference_mode(False):
            self._ring = torch.zeros(r * self._step_numel, dtype=torch.int32, device=device)
            self._ntok = torch.zeros(r, dtype=torch.int32, device=device)
            self._counter = torch.zeros(1, dtype=torch.int64, device=device)
            self._slot = torch.zeros(1, dtype=torch.int64, device=device)
            self._base = torch.zeros(1, dtype=torch.int64, device=device)
            self._step_offsets = torch.arange(self._step_numel, dtype=torch.int64, device=device)
            self._lut = self._lut_host.to(device)
            self._snapshot = torch.zeros(1, dtype=torch.int64, device=device)
            self._copy_stream = torch.cuda.Stream(device=device)
            self._host = [
                (
                    maybe_pin_memory(torch.empty(r, self._step_numel, dtype=torch.int32)),
                    maybe_pin_memory(torch.empty(r, dtype=torch.int32)),
                    maybe_pin_memory(torch.empty(1, dtype=torch.int64)),
                )
                for _ in range(self._HOST_BUFFERS)
            ]
            self._snapshot_events = [torch.cuda.Event() for _ in range(self._HOST_BUFFERS)]
            self._copy_events = [torch.cuda.Event() for _ in range(self._HOST_BUFFERS)]
        self._allocated = True

    def record(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Count ``topk_ids`` (``[num_tokens, top_k]`` routed slot ids) of MoE layer ``layer_idx``."""
        num_tokens = topk_ids.shape[0]
        if not self._allocated:
            self._allocate(topk_ids.device)
        if layer_idx == self.first_layer:
            # A new forward step: claim the next ring slot and clear it.
            self._counter.add_(1)
            torch.remainder(self._counter, self.RING_STEPS, out=self._slot)
            torch.mul(self._slot, self._step_numel, out=self._base)
            self._ring.index_fill_(0, self._base + self._step_offsets, 0)
            self._ntok.index_fill_(0, self._slot, num_tokens)
        if num_tokens == 0:
            return
        where = self._lut.index_select(
            0, topk_ids.reshape(-1).to(torch.int64) + layer_idx * self.num_experts
        )
        self._ring.index_add_(0, self._base + where, torch.ones_like(where, dtype=torch.int32))
        if self.planes == 2 and num_tokens % self.stride == 0:
            first = where.view(num_tokens, -1)[:: self.stride].reshape(-1)
            self._ring.index_add_(
                0, self._base + self._plane_numel + first, torch.ones_like(first, dtype=torch.int32)
            )

    # ---- host side (executor thread, once per iteration, never synchronizes) ----

    def on_iteration(self, iteration: int) -> None:
        """Collect the copies that have finished and start the next one."""
        if not self._allocated:
            return
        while self._pending and self._copy_events[self._pending[0][0]].query():
            self._collect(*self._pending.pop(0))
        self._calls += 1
        self._call_iterations.append(iteration)
        if len(self._pending) == self._HOST_BUFFERS or (
            self._last_step is not None and self._calls % self.COPY_INTERVAL
        ):
            return
        buffer = self._next_buffer
        self._next_buffer = (buffer + 1) % self._HOST_BUFFERS
        # Every step up to the snapshot has finished at this point of the compute stream.
        self._snapshot.copy_(self._counter)
        self._snapshot_events[buffer].record()
        ring_host, ntok_host, counter_host = self._host[buffer]
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._snapshot_events[buffer])
            counter_host.copy_(self._snapshot, non_blocking=True)
            ntok_host.copy_(self._ntok, non_blocking=True)
            ring_host.copy_(self._ring.view(self.RING_STEPS, -1), non_blocking=True)
            self._copy_events[buffer].record(self._copy_stream)
        self._pending.append((buffer, iteration))

    def _collect(self, buffer: int, iteration: int) -> None:
        del iteration
        ring_host, ntok_host, counter_host = self._host[buffer]
        last = int(counter_host[0])
        if self._last_step is None:
            # Everything before the first executor iteration is warm-up and graph capture.
            self._last_step = self._warmup_steps = last
            logger.info(f"Kimi K3 routing statistics: collecting from forward step {last + 1}")
            return
        # A slot can be rewritten by later steps while it is copied; keep a two-step margin.
        first = max(self._last_step + 1, last - self.RING_STEPS + 3)
        self._dropped_steps += first - (self._last_step + 1)
        ring, ntok = ring_host.numpy(), ntok_host.numpy()
        shape = (self.planes, self.num_layers, self.num_experts)
        for step in range(first, last + 1):
            slot = step % self.RING_STEPS
            counts = ring[slot].reshape(shape)
            self._totals += counts
            self._num_steps += 1
            self._chunk_counts.append(counts.astype(np.uint16))
            call = step - self._warmup_steps - 1
            it = self._call_iterations[call] if 0 <= call < len(self._call_iterations) else -1
            self._chunk_meta.append((step, int(ntok[slot]), it))
        self._last_step = max(self._last_step, last)
        if len(self._chunk_counts) >= self.chunk_steps or (
            last >= first and time.monotonic() - self._last_flush > self.CHUNK_SECONDS
        ):
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        self._last_flush = time.monotonic()
        payload = {
            "totals": self._totals.copy(),
            "num_steps": np.int64(self._num_steps),
            "dropped_steps": np.int64(self._dropped_steps),
            "stride": np.int64(self.stride),
            "first_layer": np.int64(self.first_layer),
            "time": np.float64(time.time()),
        }
        if self._chunk_counts:
            meta = np.asarray(self._chunk_meta, dtype=np.int64)
            payload.update(
                counts=np.stack(self._chunk_counts),
                step=meta[:, 0],
                num_tokens=meta[:, 1],
                iteration=meta[:, 2],
            )
        name = os.path.join(self.out_dir, f"routing_{self._tag}_{self._chunk_index:05d}.npz")
        self._chunk_index += 1
        self._chunk_counts, self._chunk_meta = [], []
        self._writes.put((name, payload))

    def _write_loop(self) -> None:
        while True:
            item = self._writes.get()
            if item is None:
                return
            name, payload = item
            # Written under a name the routing_*.npz readers do not match, then renamed.
            tmp = os.path.join(os.path.dirname(name), "." + os.path.basename(name))
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **payload)
            os.replace(tmp, name)

    def close(self) -> None:
        """Write what was collected (a final chunk) and stop the writer."""
        if self._writer.is_alive():
            self._flush_chunk()
            self._writes.put(None)
            self._writer.join(timeout=60)
        logger.info(
            f"Kimi K3 routing statistics: {self._num_steps} steps in {self.out_dir} "
            f"({self._dropped_steps} dropped)"
        )


_STATS: Optional[RoutingStats] = None


def create(
    out_dir: str,
    num_layers: int,
    num_experts: int,
    first_layer: int,
    expert_of_slot: Dict[int, Sequence[int]],
    stride: int,
    chunk_steps: int,
) -> RoutingStats:
    """The process's recorder (one per process; a second call returns the first), hooked to the executor's
    iteration callback."""
    global _STATS
    if _STATS is None:
        from tensorrt_llm._torch.moe.expert_statistic import ExpertStatistic

        _STATS = RoutingStats(
            out_dir, num_layers, num_experts, first_layer, expert_of_slot, stride, chunk_steps
        )
        ExpertStatistic.register_iteration_callback(_STATS.on_iteration)
        logger.info(
            f"Kimi K3 routing statistics -> {out_dir} (first MoE layer {first_layer}, stride {_STATS.stride}, "
            f"{len(expert_of_slot)} placed layers mapped back to checkpoint experts)"
        )
    return _STATS

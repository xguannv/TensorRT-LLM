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
"""Spill seat for a disaggregated generation worker (K3 low latency).

``TRTLLM_K3LL_GEN_SPILL_WAIT_S=<T>`` (seconds; unset = off) lets a generation worker whose engine is built with
``max_batch_size = B + 1`` decode at most ``B`` requests per step, and a ``B + 1``-th only when a request has been
waiting for a seat for at least ``T`` seconds since it reached the worker. While that request (the spill request)
decodes, a seat freed by any other request may be refilled at once, so the batch stays at ``B + 1`` until the spill
request finishes. A gen worker that is saturated at ``B`` seats otherwise queues requests for many seconds (TTFT p90 at
AgentX c84 is mostly this wait), while a ``B + 1`` batch for every step would move the decode step of every token.

The decision must be identical on every rank. The time it is based on is rank 0's monotonic clock, broadcast with the
per-iteration request count (``RequestBroadcaster`` carries it when the spill seat is on), and a request's arrival is
the broadcast time of the iteration that first fetched it, so every rank sees the same numbers.

Engaged evidence: rank 0 logs ``[k3ll] gen spill`` every ``LOG_EVERY`` steps that ran generation requests (steps at
``B + 1``, spill admissions).
"""

import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

from tensorrt_llm.logger import logger

SPILL_ENV = "TRTLLM_K3LL_GEN_SPILL_WAIT_S"
LOG_EVERY = 2000
STAMP_TTL_S = 3600.0


def spill_wait_s() -> Optional[float]:
    """The spill threshold in seconds, or None when the switch is unset or invalid (off)."""
    value = os.environ.get(SPILL_ENV)
    if value is None or value.strip() == "":
        return None
    try:
        wait = float(value)
    except ValueError:
        logger.warning(f"{SPILL_ENV}={value!r} is not a number; the spill seat stays off")
        return None
    if wait < 0:
        logger.warning(f"{SPILL_ENV}={value!r} is negative; the spill seat stays off")
        return None
    return wait


def plan_admission(
    base: int,
    wait_s: float,
    now: float,
    running_ids: Iterable[int],
    spill_ids: Set[int],
    new_ids: List[int],
    fetch_time: Dict[int, float],
) -> Tuple[Set[int], Optional[int], int]:
    """Which new generation requests may start decoding this step.

    Args:
        base: ``B``, the batch the worker decodes when nobody waits long (engine max_batch_size - 1).
        wait_s: the spill threshold ``T``.
        now: the broadcast time of this iteration.
        running_ids: requests that are already decoding (scheduled for generation before).
        spill_ids: requests that started decoding through the spill seat.
        new_ids: requests whose KV is in and that have not decoded yet, in scheduling order.
        fetch_time: broadcast time at which each request was first fetched by this worker.

    Returns:
        ``(admitted, spill_id, cap)``: the new requests that may start, the one among them that takes the spill seat
        (None if none does), and the batch cap of this step (``B`` or ``B + 1``).
    """
    running = list(running_ids)
    spill_active = any(r in spill_ids for r in running)
    head_wait = max((now - fetch_time.get(r, now) for r in new_ids), default=0.0)
    cap = base + 1 if (spill_active or (new_ids and head_wait >= wait_s)) else base
    free = max(0, cap - len(running))
    admitted = new_ids[:free]
    spill_id = None
    if admitted and not spill_active and len(running) + len(admitted) > base:
        spill_id = admitted[-1]
    return set(admitted), spill_id, cap


class GenSpillState:
    """Per-worker spill-seat state, shared by the executor (time, fetch stamps) and the scheduler (admission)."""

    def __init__(self, wait_s: float, max_batch_size: int, rank: int):
        self.wait_s = wait_s
        self.base = max_batch_size - 1
        self.rank = rank
        self.now = 0.0
        self.fetch_time: Dict[int, float] = {}
        self.decoding: Set[int] = set()
        self.spill: Set[int] = set()
        self.steps = 0
        self.full_steps = 0
        self.admissions = 0
        if rank == 0:
            logger.info(
                f"[k3ll] gen spill ON: decode {self.base} requests per step, a {self.base + 1}-th only after a "
                f"request waited >= {wait_s:g} s ({SPILL_ENV})"
            )

    def stamp(self, request_ids: Iterable[int]) -> None:
        """Record the fetch time (this iteration's broadcast time) of newly fetched requests."""
        for rid in request_ids:
            self.fetch_time.setdefault(rid, self.now)

    def forget(self, live_ids: Set[int]) -> None:
        """Drop the state of requests no longer in the active list. Fetch stamps of requests not active (finished, or
        still in the waiting queue) are kept for ``STAMP_TTL_S``."""
        self.decoding &= live_ids
        self.spill &= live_ids
        if len(self.fetch_time) > 256:
            for rid in [
                r
                for r, t in self.fetch_time.items()
                if r not in live_ids and self.now - t > STAMP_TTL_S
            ]:
                del self.fetch_time[rid]

    def plan(
        self, running_ids: List[int], new_ids: List[int]
    ) -> Tuple[Set[int], Optional[int], int]:
        return plan_admission(
            self.base, self.wait_s, self.now, running_ids, self.spill, new_ids, self.fetch_time
        )

    def record(self, scheduled_gen_ids: List[int], spill_id: Optional[int]) -> None:
        """After a step's generation requests are chosen: bookkeeping and the periodic engaged-evidence log."""
        self.decoding.update(scheduled_gen_ids)
        if spill_id is not None and spill_id in scheduled_gen_ids:
            self.spill.add(spill_id)
            self.admissions += 1
        if scheduled_gen_ids:
            self.steps += 1
            if len(scheduled_gen_ids) > self.base:
                self.full_steps += 1
            if self.rank == 0 and self.steps % LOG_EVERY == 0:
                logger.info(
                    f"[k3ll] gen spill: wait_s={self.wait_s:g} steps={self.steps} "
                    f"steps_at_{self.base + 1}={self.full_steps} ({100.0 * self.full_steps / self.steps:.1f}%) "
                    f"spill_admissions={self.admissions}"
                )

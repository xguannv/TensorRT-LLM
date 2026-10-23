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
"""KV prefetch admission of a disaggregated generation worker
(TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS)."""

import collections

import pytest

from tensorrt_llm._torch.pyexecutor.disagg_gen_prefetch import (
    PREFETCH_REQUESTS_ENV,
    num_prefetch_admissions,
    pop_prefetch_requests,
    prefetch_request_limit,
)


class _FakeQueue:
    """FCFS queue with the waiting-queue calls the helper uses."""

    def __init__(self, items):
        self._items = collections.deque(items)

    def __bool__(self):
        return bool(self._items)

    def peek_request(self):
        return self._items[0]

    def pop_request(self):
        return self._items.popleft()

    def remaining(self):
        return list(self._items)


@pytest.mark.parametrize(
    "value, batch, expected",
    [
        (None, 2, 0),
        ("0", 2, 0),
        ("1", 2, 1),
        ("2", 2, 2),
        ("5", 2, 2),
        ("-3", 2, 0),
        ("abc", 2, 0),
        ("1", 1, 1),
        ("3", 4, 3),
    ],
)
def test_prefetch_request_limit(monkeypatch, value, batch, expected):
    if value is None:
        monkeypatch.delenv(PREFETCH_REQUESTS_ENV, raising=False)
    else:
        monkeypatch.setenv(PREFETCH_REQUESTS_ENV, value)
    assert prefetch_request_limit(batch) == expected


@pytest.mark.parametrize(
    "limit, batch, active, pending_init, expected",
    [
        # Disabled: never admits past the cap.
        (0, 2, 2, 0, 0),
        (0, 2, 0, 0, 0),
        # Batch full, nothing pending: one extra request may start its transfer.
        (1, 2, 2, 0, 1),
        # The extra request is already admitted.
        (1, 2, 3, 0, 0),
        (1, 2, 3, 1, 0),
        # A slot freed while the prefetched request waits: refill the prefetch.
        (1, 2, 2, 0, 1),
        # Cold start: the base admission took a whole batch of INIT requests
        # this iteration, the next scheduling pass cannot prepare more.
        (1, 2, 2, 2, 0),
        (2, 2, 2, 1, 1),
        (2, 2, 2, 0, 2),
        (2, 2, 3, 0, 1),
        (2, 2, 4, 0, 0),
        # Over-admitted state (e.g. requests re-added) never yields negatives.
        (1, 2, 5, 0, 0),
        (1, 2, 2, 3, 0),
    ],
)
def test_num_prefetch_admissions(limit, batch, active, pending_init, expected):
    assert num_prefetch_admissions(limit, batch, active, pending_init) == expected


def test_num_prefetch_admissions_bounds_total_admission():
    """Active requests never exceed batch + limit, and INIT requests never
    exceed one batch, whatever the starting state."""
    batch = 2
    for limit in range(0, batch + 1):
        for active in range(0, batch + limit + 2):
            for pending in range(0, active + 1):
                extra = num_prefetch_admissions(limit, batch, active, pending)
                assert extra >= 0
                if extra:
                    assert active + extra <= batch + limit
                    assert pending + extra <= batch


def test_pop_prefetch_requests_keeps_order_and_stops_at_other_request():
    queue = _FakeQueue(["g1", "g2", "ctx", "g3"])
    popped = pop_prefetch_requests(queue, 5, lambda item: item.startswith("g"))
    assert popped == ["g1", "g2"]
    assert queue.remaining() == ["ctx", "g3"]


def test_pop_prefetch_requests_respects_max_count():
    queue = _FakeQueue(["g1", "g2", "g3"])
    assert pop_prefetch_requests(queue, 1, lambda item: True) == ["g1"]
    assert queue.remaining() == ["g2", "g3"]
    assert pop_prefetch_requests(queue, 0, lambda item: True) == []
    assert queue.remaining() == ["g2", "g3"]


def test_pop_prefetch_requests_empty_queue():
    queue = _FakeQueue([])
    assert pop_prefetch_requests(queue, 3, lambda item: True) == []


def test_pop_prefetch_requests_head_not_prefetchable():
    queue = _FakeQueue(["ctx", "g1"])
    assert pop_prefetch_requests(queue, 2, lambda item: item.startswith("g")) == []
    assert queue.remaining() == ["ctx", "g1"]


class _FakeItem:
    """Waiting-queue item: ``kind`` "gen" (generation-only) or "ctx"."""

    def __init__(self, name, kind="gen"):
        self.name = name
        self.kind = kind
        self.child_req_ids = None
        self.request = None


class _FakeActive:
    def __init__(self, init_state=False):
        self.is_disagg_generation_init_state = init_state


class _ExecutorStub:
    """The attributes PyExecutor._pop_from_waiting_queue reads."""

    def __init__(self, batch, limit, active):
        self.max_num_active_requests = batch
        self._disagg_gen_prefetch_limit = limit
        self.active_requests = active
        self.enable_attention_dp = False
        self.is_benchmark_disagg = False
        self._benchmark_fill_phase_active = False
        self.is_warmup = False
        self._mm_encoder_item_scheduling_enabled = False

    def _get_request_admission_capacity(self):
        return self.max_num_active_requests


@pytest.mark.parametrize(
    "limit, active, queue, expected",
    [
        # Batch full: only the prefetch admission, one generation-only request.
        (1, [_FakeActive(), _FakeActive()], ["g1", "g2"], ["g1"]),
        # Switch off: nothing past the batch size.
        (0, [_FakeActive(), _FakeActive()], ["g1", "g2"], []),
        # The prefetched request already waits: nothing more.
        (1, [_FakeActive(), _FakeActive(), _FakeActive()], ["g1"], []),
        # One slot free: the normal admission plus one prefetch.
        (1, [_FakeActive()], ["g1", "g2", "g3"], ["g1", "g2"]),
        # Cold start: a whole batch of INIT requests, no prefetch this pass.
        (1, [], ["g1", "g2", "g3"], ["g1", "g2"]),
        # A context request at the head is never admitted past the cap.
        (1, [_FakeActive(), _FakeActive()], ["c1", "g1"], []),
        # Still-INIT request counts against the per-pass INIT bound.
        (2, [_FakeActive(), _FakeActive(), _FakeActive(init_state=True)], ["g1", "g2"], ["g1"]),
    ],
)
def test_executor_pop_from_waiting_queue_prefetch(monkeypatch, limit, active, queue, expected):
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    from tensorrt_llm._torch.pyexecutor.scheduler import create_waiting_queue

    monkeypatch.setattr(py_executor, "_is_prefetchable_queue_item", lambda item: item.kind == "gen")
    waiting_queue = create_waiting_queue()
    waiting_queue.extend(
        _FakeItem(name, "ctx" if name.startswith("c") else "gen") for name in queue
    )
    stub = _ExecutorStub(batch=2, limit=limit, active=list(active))
    popped = py_executor.PyExecutor._pop_from_waiting_queue(stub, waiting_queue, len(active))
    assert [item.name for item in popped] == expected
    assert [item.name for item in waiting_queue] == [n for n in queue if n not in expected]


class _GenRequest:
    """Scheduled generation request: transfer-complete (needs a new sequence slot) or decoding."""

    def __init__(self, request_id, transfer_complete):
        self.request_id = request_id
        self.py_request_id = request_id
        self.is_disagg_generation_transmission_complete = transfer_complete


class _RevertRecorder:
    def __init__(self):
        self.reverted = []

    def revert_allocate_generation(self, req):
        self.reverted.append(req.py_request_id)


class _SeatStub:
    """The attributes PyExecutor._defer_transfer_complete_without_seat and _revert_gen_alloc read."""

    def __init__(self, py_executor, seq_slot_manager):
        self._py_executor = py_executor
        self.resource_manager = type("RM", (), {})()
        self.resource_manager.resource_managers = {
            py_executor.ResourceManagerType.SEQ_SLOT_MANAGER: seq_slot_manager
        }
        self._is_kv_manager_v2 = True
        self.kv_cache_manager = _RevertRecorder()
        self.enable_joint_kv_cache_reuse = False
        self.draft_kv_cache_manager = None

    def _revert_gen_alloc(self, scheduled_batch):
        return self._py_executor.PyExecutor._revert_gen_alloc(self, scheduled_batch)


@pytest.mark.parametrize(
    "num_seats, held, scheduled, expected_kept, expected_deferred",
    [
        # b1 under the overlap executor: the finished request (id 1) still holds the only
        # seat when the prefetched request (id 2), already transfer-complete, is scheduled.
        (1, [1], [(2, True)], [], [2]),
        # The seat is back: the transfer-complete request goes ahead.
        (1, [], [(2, True)], [2], []),
        # Decoding requests already own their seats and are never deferred.
        (2, [1, 3], [(1, False), (4, True)], [1], [4]),
        # One free seat, two transfer-complete requests: the first keeps it, in batch order.
        (2, [1], [(5, True), (6, True)], [5], [6]),
    ],
)
def test_defer_transfer_complete_without_seat(
    num_seats, held, scheduled, expected_kept, expected_deferred
):
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

    seq_slot_manager = SeqSlotManager(num_seats)
    for request_id in held:
        seq_slot_manager.slot_manager.add_slot(request_id)
    stub = _SeatStub(py_executor, seq_slot_manager)
    requests = [_GenRequest(request_id, tc) for request_id, tc in scheduled]

    kept, num_deferred = py_executor.PyExecutor._defer_transfer_complete_without_seat(
        stub, requests
    )

    assert [req.request_id for req in kept] == expected_kept
    assert num_deferred == len(expected_deferred)
    assert stub.kv_cache_manager.reverted == expected_deferred
    # Deferring takes no seat: every kept transfer-complete request still finds one.
    assert len(seq_slot_manager.slot_manager.free_slots) >= sum(
        1 for req in kept if req.is_disagg_generation_transmission_complete
    )

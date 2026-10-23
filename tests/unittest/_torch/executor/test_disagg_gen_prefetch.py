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
    """Active generation request: transfer-complete (needs a new sequence slot) or decoding."""

    def __init__(self, request_id, transfer_complete):
        self.request_id = request_id
        self.py_request_id = request_id
        self.is_disagg_generation_transmission_complete = transfer_complete


class _SeatStub:
    """The attributes PyExecutor._hold_transfer_complete_without_seat and _schedulable_requests read."""

    def __init__(self, py_executor, seq_slot_manager, prefetch_limit=1, active=()):
        self._py_executor = py_executor
        self.resource_manager = type("RM", (), {})()
        self.resource_manager.resource_managers = {
            py_executor.ResourceManagerType.SEQ_SLOT_MANAGER: seq_slot_manager
        }
        self.kv_cache_manager = type("KVCM", (), {})()
        self._disagg_gen_prefetch_limit = prefetch_limit
        self.active_requests = list(active)

    def _hold_transfer_complete_without_seat(self, active_requests):
        return self._py_executor.PyExecutor._hold_transfer_complete_without_seat(
            self, active_requests
        )


def _seat_manager(num_seats, held):
    from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

    seq_slot_manager = SeqSlotManager(num_seats)
    for request_id in held:
        seq_slot_manager.slot_manager.add_slot(request_id)
    return seq_slot_manager


@pytest.mark.parametrize(
    "num_seats, held, active, expected_kept",
    [
        # b1: the earlier request (2) finished its transfer after the later one (1) took the
        # only seat and started decoding. 2 must not reach the scheduler, or it takes the one
        # request-budget entry, 1 never decodes and the seat never frees.
        (1, [1], [(2, True), (1, False)], [1]),
        # b1 under the overlap executor: the finished request still holds the only seat.
        (1, [1], [(2, True)], []),
        # b1, the seat is back.
        (1, [], [(2, True)], [2]),
        # b2, both seats held by decoders, the transfer-complete request first in order.
        (2, [1, 3], [(4, True), (1, False), (3, False)], [1, 3]),
        # b2, one free seat, two transfer-complete requests: the earlier one keeps it.
        (2, [1], [(5, True), (1, False), (6, True)], [5, 1]),
        # b4, two free seats, three transfer-complete requests interleaved with decoders.
        (4, [1, 2], [(7, True), (1, False), (8, True), (2, False), (9, True)], [7, 1, 8, 2]),
        # b4, no free seat.
        (
            4,
            [1, 2, 3, 4],
            [(5, True), (1, False), (6, True), (2, False), (3, False), (4, False)],
            [1, 2, 3, 4],
        ),
    ],
)
def test_hold_transfer_complete_without_seat(num_seats, held, active, expected_kept):
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    seq_slot_manager = _seat_manager(num_seats, held)
    stub = _SeatStub(py_executor, seq_slot_manager)
    requests = [_GenRequest(request_id, tc) for request_id, tc in active]

    kept, num_held = py_executor.PyExecutor._hold_transfer_complete_without_seat(stub, requests)

    assert [req.request_id for req in kept] == expected_kept
    assert num_held == len(requests) - len(kept)
    # Every kept transfer-complete request finds a free seat.
    assert len(seq_slot_manager.slot_manager.free_slots) >= sum(
        1 for req in kept if req.is_disagg_generation_transmission_complete
    )


def test_hold_transfer_complete_without_seat_nothing_held_returns_same_list():
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    stub = _SeatStub(py_executor, _seat_manager(2, [1]))
    requests = [_GenRequest(1, False), _GenRequest(2, True)]

    kept, num_held = py_executor.PyExecutor._hold_transfer_complete_without_seat(stub, requests)

    assert kept is requests and num_held == 0


@pytest.mark.parametrize("prefetch_limit", [0, 1])
def test_schedulable_requests_gate_only_with_prefetch(prefetch_limit):
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    active = [_GenRequest(2, True), _GenRequest(1, False)]
    stub = _SeatStub(py_executor, _seat_manager(1, [1]), prefetch_limit, active)

    schedulable, num_held = py_executor.PyExecutor._schedulable_requests(stub)

    if prefetch_limit == 0:
        # Prefetch off: the scheduler sees the active requests unchanged.
        assert schedulable is stub.active_requests and num_held == 0
    else:
        assert [req.request_id for req in schedulable] == [1] and num_held == 1


class _SimRequest(_GenRequest):
    def __init__(self, request_id, decode_steps):
        super().__init__(request_id, transfer_complete=True)
        self.decode_steps = decode_steps
        self.decoded = 0


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("long_first", [True, False])
def test_seat_gate_no_starvation_with_arriving_decoders(batch, long_first):
    """Executor-level simulation of the seat gate with a FIFO request budget.

    Each iteration: the gate filters the active requests, a stand-in scheduler takes the first
    ``batch`` of them, a scheduled transfer-complete request takes a seat and decodes, a decoder
    that has run its steps finishes and frees its seat, and a new transfer-complete request
    arrives at the back. Every request finishes, and a held request gets the first seat that
    frees ahead of any later request.
    """
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    seq_slot_manager = _seat_manager(batch, [])
    slots = seq_slot_manager.slot_manager
    stub = _SeatStub(py_executor, seq_slot_manager)
    # Decoders already holding every seat, behind one early transfer-complete request: the
    # ordering that livelocked the post-schedule gate.
    early = _SimRequest(100, decode_steps=3)
    decoders = []
    for i in range(batch):
        req = _SimRequest(i, decode_steps=5 if long_first else 2 + i)
        req.is_disagg_generation_transmission_complete = False
        slots.add_slot(req.request_id)
        decoders.append(req)
    active = [early] + decoders
    next_id, finished, seat_order = 200, [], []
    for _ in range(200):
        schedulable, _ = py_executor.PyExecutor._hold_transfer_complete_without_seat(stub, active)
        scheduled = schedulable[:batch]
        assert scheduled, "the batch must never be empty while requests are active"
        for req in scheduled:
            if req.is_disagg_generation_transmission_complete:
                slots.add_slot(req.request_id)  # raises if no seat is free
                seat_order.append(req.request_id)
                req.is_disagg_generation_transmission_complete = False
            req.decoded += 1
        for req in [r for r in scheduled if r.decoded >= r.decode_steps]:
            slots.remove_slot(req.request_id)
            active.remove(req)
            finished.append(req.request_id)
        if next_id < 230:
            active.append(_SimRequest(next_id, decode_steps=2))
            next_id += 1
        if not active:
            break
    assert not active and 100 in finished
    # Seats go to transfer-complete requests in arrival order: the early one first.
    assert seat_order[0] == 100 and seat_order == sorted(seat_order)


class _DebugRequest:
    def __init__(self, request_id, state_name):
        self.py_request_id = request_id
        self.state = type("State", (), {"name": state_name})()
        self.py_seq_slot = None
        self.py_decoding_iter = 0


class _DebugStub(_SeatStub):
    """_SeatStub plus what PyExecutor._prefetch_debug_note reads."""

    def __init__(self, py_executor, seq_slot_manager, rank, idle_limit, active):
        super().__init__(py_executor, seq_slot_manager, active=active)
        self.dist = type("Dist", (), {"rank": rank})()
        self._prefetch_debug_idle_limit = idle_limit
        self._prefetch_debug_deferred = 0
        self._prefetch_debug_idle = 0
        self._prefetch_debug_dumps = 0
        self.active_requests = active
        self.waiting_queue = []
        self.inflight_req_ids = set()
        self.kv_cache_manager.index_mapper = type("IM", (), {"num_free_slots": lambda self: 2})()
        self.kv_cache_manager.kv_cache_map = {}


@pytest.mark.parametrize("rank", [0, 1])
def test_prefetch_debug_note_dumps_idle_worker(monkeypatch, rank):
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

    lines = []
    monkeypatch.setattr(py_executor.logger, "info", lines.append)
    stub = _DebugStub(
        py_executor,
        SeqSlotManager(1),
        rank=rank,
        idle_limit=2,
        active=[_DebugRequest(7, "DISAGG_GENERATION_TRANS_COMPLETE")],
    )
    note = py_executor.PyExecutor._prefetch_debug_note

    note(stub, 1, [], [])  # first deferral: a new power of two
    note(stub, 1, [], [])  # second deferral: 2, a new power of two; idle 2 -> dump 1
    note(stub, 0, [], [])  # idle 3
    note(stub, 0, [], [])  # idle 4 -> dump 2
    note(stub, 0, [], [object()])  # a scheduled request resets the idle count

    if rank != 0:
        assert lines == []
        return
    assert [line for line in lines if "deferred_without_seat=" in line and "dump" not in line] == [
        "disagg_gen_prefetch deferred_without_seat=1",
        "disagg_gen_prefetch deferred_without_seat=2",
    ]
    dumps = [line for line in lines if "idle dump" in line]
    assert len(dumps) == 4  # two dumps, each a header and one request line
    assert "free_seats=1" in dumps[0] and "free_index_slots=2" in dumps[0]
    assert "request 7 state=DISAGG_GENERATION_TRANS_COMPLETE" in dumps[1]
    assert stub._prefetch_debug_idle == 0

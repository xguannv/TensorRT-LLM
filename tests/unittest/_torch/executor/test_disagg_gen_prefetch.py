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
import types

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


class _FakeRequest:
    """Request with the fields the sequence-slot path reads; ``state`` is "context",
    "generating", "to_complete" (GENERATION_TO_COMPLETE), "trans_complete"
    (DISAGG_GENERATION_TRANS_COMPLETE) or "complete"."""

    def __init__(self, request_id, state="generating"):
        self.request_id = request_id
        self.py_request_id = request_id
        self.state = state
        self.seq_slot = None
        self.py_seq_slot = None
        self.return_perf_metrics = False

    @property
    def is_disagg_generation_transmission_complete(self):
        return self.state == "trans_complete"

    @property
    def is_disagg_generation_init_state(self):
        return False


class _FakeKvCacheManager:
    def __init__(self):
        self.reverted = []

    def revert_allocate_generation(self, req):
        self.reverted.append(req.request_id)


class _JoinStub:
    """The attributes PyExecutor._defer_joins_without_seq_slot reads."""

    def __init__(self, seq_slot_manager, resource_manager_type):
        self.resource_manager = types.SimpleNamespace(
            resource_managers={resource_manager_type: seq_slot_manager}
        )
        self._is_kv_manager_v2 = True
        self.kv_cache_manager = _FakeKvCacheManager()
        self.enable_joint_kv_cache_reuse = False
        self.draft_kv_cache_manager = None


def _schedule(requests, max_batch_size):
    """V2-scheduler seat rule: a request whose step in flight is its last
    (GENERATION_TO_COMPLETE) is not scheduled and leaves its seat to the next request."""
    seats = [r for r in requests if r.state in ("context", "generating", "trans_complete")]
    return seats[:max_batch_size]


def _batch(scheduled_batch_type, requests):
    batch = scheduled_batch_type()
    batch.context_requests_last_chunk = [r for r in requests if r.state == "context"]
    batch.generation_requests = [r for r in requests if r.state != "context"]
    return batch


def _take_seq_slots(seq_slots, batch, scheduled_batch_type):
    """The slot allocations of one iteration: joining requests first
    (_prepare_disagg_gen_transmission_complete), then the rest of the batch
    (ResourceManager.prepare_resources)."""
    joining = scheduled_batch_type()
    joining.context_requests_last_chunk = [
        r for r in batch.generation_requests if r.is_disagg_generation_transmission_complete
    ]
    seq_slots.prepare_resources(joining)
    for r in joining.context_requests_last_chunk:
        r.state = "generating"
    seq_slots.prepare_resources(batch)


def test_prefetched_request_waits_for_the_finishing_requests_slot():
    """Full batch (A, B) plus one prefetched request C whose KV transfer completed before a
    slot freed. A's step in flight is its last (GENERATION_TO_COMPLETE): the scheduler gives
    its seat to C while A still holds its sequence slot. Without the deferral the slot
    allocation of C's first step fails (the e2e crash, NoFreeSlotsError); with it, C stays
    out of this batch, its KV growth is reverted, and it joins once A is terminated."""
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    from tensorrt_llm._torch.pyexecutor.resource_manager import (
        NoFreeSlotsError,
        ResourceManagerType,
    )
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

    def full_batch_and_prefetched_join():
        a, b = _FakeRequest(1), _FakeRequest(2)
        c = _FakeRequest(3, "trans_complete")
        seq_slots = SeqSlotManager(2)
        _take_seq_slots(seq_slots, _batch(ScheduledRequests, [a, b]), ScheduledRequests)
        a.state = "to_complete"
        batch = _batch(ScheduledRequests, _schedule([a, b, c], 2))
        assert [r.request_id for r in batch.generation_requests] == [2, 3]
        return a, b, c, seq_slots, batch

    # Reproduction: A's seat goes to C while A keeps its slot.
    _, _, _, seq_slots, batch = full_batch_and_prefetched_join()
    with pytest.raises(NoFreeSlotsError):
        _take_seq_slots(seq_slots, batch, ScheduledRequests)

    # Fix: C is deferred in this iteration and joins after A's termination.
    a, b, c, seq_slots, batch = full_batch_and_prefetched_join()
    a_slot = a.seq_slot
    stub = _JoinStub(seq_slots, ResourceManagerType.SEQ_SLOT_MANAGER)
    assert py_executor.PyExecutor._defer_joins_without_seq_slot(stub, batch) == 1
    assert [r.request_id for r in batch.generation_requests] == [2]
    assert batch.batch_size == 1
    assert stub.kv_cache_manager.reverted == [3]
    _take_seq_slots(seq_slots, batch, ScheduledRequests)
    assert c.seq_slot is None and c.is_disagg_generation_transmission_complete
    # A's last step is processed later in the iteration and A is terminated.
    a.state = "complete"
    seq_slots.free_resources(a)
    # Next iteration: C joins on A's former slot.
    batch = _batch(ScheduledRequests, _schedule([b, c], 2))
    assert py_executor.PyExecutor._defer_joins_without_seq_slot(stub, batch) == 0
    assert [r.request_id for r in batch.generation_requests] == [2, 3]
    _take_seq_slots(seq_slots, batch, ScheduledRequests)
    assert c.seq_slot == a_slot and c.py_seq_slot == a_slot
    assert stub.kv_cache_manager.reverted == [3]
    assert seq_slots.num_free_slots() == 0


def test_defer_joins_keeps_joins_that_fit_and_drops_the_rest_in_order():
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

    seq_slots = SeqSlotManager(4)
    held = [_FakeRequest(i) for i in (1, 2, 3)]
    _take_seq_slots(seq_slots, _batch(ScheduledRequests, held), ScheduledRequests)
    joins = [_FakeRequest(i, "trans_complete") for i in (7, 8)]
    batch = _batch(ScheduledRequests, [held[2], *joins])
    stub = _JoinStub(seq_slots, ResourceManagerType.SEQ_SLOT_MANAGER)
    # One free slot: the first joining request stays, the second waits.
    assert py_executor.PyExecutor._defer_joins_without_seq_slot(stub, batch) == 1
    assert [r.request_id for r in batch.generation_requests] == [3, 7]
    assert stub.kv_cache_manager.reverted == [8]
    _take_seq_slots(seq_slots, batch, ScheduledRequests)
    assert joins[0].seq_slot is not None and joins[1].seq_slot is None
    # No joining request: nothing to do, whatever the slot count.
    batch = _batch(ScheduledRequests, held[:2])
    assert py_executor.PyExecutor._defer_joins_without_seq_slot(stub, batch) == 0
    assert [r.request_id for r in batch.generation_requests] == [1, 2]


def test_defer_joins_keeps_the_slots_new_context_requests_take():
    """A new context request of the batch takes its slot after the joining requests took
    theirs; the join waits instead of the context request finding no slot."""
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

    seq_slots = SeqSlotManager(2)
    finishing = _FakeRequest(1)
    _take_seq_slots(seq_slots, _batch(ScheduledRequests, [finishing]), ScheduledRequests)
    finishing.state = "to_complete"
    new_context = _FakeRequest(4, "context")
    joining = _FakeRequest(3, "trans_complete")
    batch = _batch(ScheduledRequests, _schedule([finishing, new_context, joining], 2))
    assert batch.batch_size == 2
    stub = _JoinStub(seq_slots, ResourceManagerType.SEQ_SLOT_MANAGER)
    assert py_executor.PyExecutor._defer_joins_without_seq_slot(stub, batch) == 1
    assert batch.generation_requests == []
    assert stub.kv_cache_manager.reverted == [3]
    _take_seq_slots(seq_slots, batch, ScheduledRequests)
    assert new_context.seq_slot is not None and joining.seq_slot is None
    assert seq_slots.num_free_slots() == 0

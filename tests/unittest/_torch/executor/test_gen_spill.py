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
"""Spill seat of a disaggregated generation worker (TRTLLM_K3LL_GEN_SPILL_WAIT_S)."""

import pytest

from tensorrt_llm._torch.pyexecutor.gen_spill import (
    SPILL_ENV,
    GenSpillState,
    plan_admission,
    spill_wait_s,
)
from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster

B, T = 4, 10.0


def _plan(running, new, waited, spill=(), base=B, wait=T):
    """``waited``: seconds each new request has waited (same order as ``new``); now = 100."""
    fetch = {r: 100.0 - w for r, w in zip(new, waited)}
    return plan_admission(base, wait, 100.0, running, set(spill), list(new), fetch)


def test_switch_parsing(monkeypatch):
    monkeypatch.delenv(SPILL_ENV, raising=False)
    assert spill_wait_s() is None
    for bad in ("", "x", "-1"):
        monkeypatch.setenv(SPILL_ENV, bad)
        assert spill_wait_s() is None
    monkeypatch.setenv(SPILL_ENV, "12.5")
    assert spill_wait_s() == 12.5


def test_wait_not_met_caps_at_base():
    admitted, spill, cap = _plan(running=[1, 2, 3], new=[10, 11], waited=[3.0, 1.0])
    assert cap == B and admitted == {10} and spill is None
    admitted, spill, cap = _plan(running=[1, 2, 3, 4], new=[10], waited=[T - 0.1])
    assert cap == B and admitted == set() and spill is None


def test_wait_met_admits_one_spill():
    admitted, spill, cap = _plan(running=[1, 2, 3, 4], new=[10, 11], waited=[T, T + 5])
    assert cap == B + 1 and admitted == {10} and spill == 10


def test_never_more_than_base_plus_one():
    admitted, spill, cap = _plan(
        running=[1, 2], new=[10, 11, 12, 13], waited=[50.0, 40.0, 30.0, 20.0]
    )
    assert cap == B + 1 and admitted == {10, 11, 12} and spill == 12
    admitted, _, cap = _plan(running=[1, 2, 3, 4, 5], new=[10], waited=[50.0], spill=[5])
    assert cap == B + 1 and admitted == set()


def test_spill_seat_busy_refills_a_freed_seat():
    # request 5 holds the spill seat; one of the others finished: the freed seat is refilled without waiting T
    admitted, spill, cap = _plan(running=[1, 2, 3, 5], new=[10], waited=[0.5], spill=[5])
    assert cap == B + 1 and admitted == {10} and spill is None


def test_no_new_requests():
    admitted, spill, cap = _plan(running=[1, 2], new=[], waited=[])
    assert cap == B and admitted == set() and spill is None


def test_state_bookkeeping():
    st = GenSpillState(T, B + 1, rank=1)
    st.now = 5.0
    st.stamp([10, 11])
    st.now = 20.0
    st.stamp([10, 12])  # a second stamp of 10 keeps its first fetch time
    assert st.fetch_time == {10: 5.0, 11: 5.0, 12: 20.0}
    st.record([1, 2, 3, 4], None)
    admitted, spill, cap = st.plan([1, 2, 3, 4], [10, 11, 12])
    assert cap == B + 1 and admitted == {10} and spill == 10
    st.record([1, 2, 3, 4, 10], spill)
    assert st.spill == {10} and st.admissions == 1 and st.full_steps == 1 and st.steps == 2
    st.forget({2, 3, 4, 10, 11, 12})
    assert st.decoding == {2, 3, 4, 10} and st.spill == {10}


class _Dist:
    rank = 0
    world_size = 1
    has_pp = False


class _Hang:
    def pause(self):
        import contextlib

        return contextlib.nullcontext()


def test_broadcaster_off_path_unchanged():
    rb = RequestBroadcaster(_Dist(), _Hang())
    assert rb.carry_time is False
    assert rb.broadcast([]) == ([], None)
    assert rb.last_time == 0.0


def test_broadcaster_carries_time():
    rb = RequestBroadcaster(_Dist(), _Hang())
    rb.carry_time = True
    rb.broadcast([])
    assert rb.last_time > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

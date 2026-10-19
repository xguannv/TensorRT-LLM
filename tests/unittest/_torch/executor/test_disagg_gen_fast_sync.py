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
"""TRTLLM_DISAGG_GEN_FAST_SYNC: cheap per-iteration disaggregated status collectives.

The generation transfer-status poll and the idle-progress votes run collectives on
every executor iteration. With the switch on, an idle status poll is decided by one
fixed-size int64 allgather of the per-rank receive-session counts, and the two
idle-progress votes share one int64 allgather. These tests simulate the ranks of one
model-parallel group and check that (1) every rank takes the same branch and (2) the
decisions are those of the original collectives.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


class _GroupDist:
    """One rank's view of a simulated TP x CP group whose collectives see every rank's input."""

    def __init__(self, rank: int, group: "_Group"):
        self.rank = rank
        self.group = group
        self.world_size = group.size
        self.tp_size = 1
        self.cp_size = group.size
        self.pp_size = 1

    def tp_cp_allgather_int64(self, values):
        self.group.int64_calls += 1
        return np.asarray(self.group.int64_inputs(), dtype=np.int64).reshape(self.group.size, -1)

    def allreduce(self, value, op=ReduceOp.SUM):
        self.group.pickle_calls += 1
        assert op == ReduceOp.MAX
        return max(self.group.gen_votes)

    def tp_cp_allgather(self, value):
        self.group.pickle_calls += 1
        return list(self.group.ctx_votes)


class _Group:
    def __init__(self, size: int):
        self.size = size
        self.int64_calls = 0
        self.pickle_calls = 0
        self.gen_votes = [0] * size
        self.ctx_votes = [0] * size
        self.session_counts = [0] * size
        self.int64_inputs = lambda: [[c] for c in self.session_counts]


def _make_transceiver(rank: int, group: _Group, *, probe: bool) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._dist = _GroupDist(rank, group)
    transceiver._mapping = SimpleNamespace(
        enable_attention_dp=False, world_size=group.size, pp_size=1
    )
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = True
    transceiver._gen_idle_probe = probe
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._gen_allgather = Mock(side_effect=lambda local: [local] * group.size)
    transceiver._transfer_worker = SimpleNamespace(sweep_stale_req_infos=lambda: None)
    return transceiver


@pytest.mark.parametrize("at_least", [0, 1])
def test_idle_status_poll_matches_full_consensus(at_least: int) -> None:
    group = _Group(4)
    fast = _make_transceiver(0, group, probe=True)
    full = _make_transceiver(0, group, probe=False)
    fast._poll_gen_sessions_for_poll_interval = Mock()
    full._poll_gen_sessions_for_poll_interval = Mock()

    assert fast.check_gen_transfer_status(at_least) == full.check_gen_transfer_status(at_least)
    assert full._gen_allgather.call_count == 2
    if at_least == 0:
        # Idle everywhere: one int64 allgather, no pickled consensus.
        assert group.int64_calls == 1
        fast._gen_allgather.assert_not_called()
    else:
        # At-least-N polls keep their original pacing and collectives.
        assert group.int64_calls == 0
        assert fast._gen_allgather.call_count == 2


def test_status_poll_runs_full_consensus_when_any_rank_receives() -> None:
    group = _Group(4)
    group.session_counts = [0, 0, 1, 0]
    ranks = [_make_transceiver(r, group, probe=True) for r in range(group.size)]
    # Only rank 2 holds a session; ranks 0, 1 and 3 must still enter the consensus.
    for transceiver in ranks:
        assert transceiver.check_gen_transfer_status(0) == ([], [], [])
        assert transceiver._gen_allgather.call_count == 2
    assert group.int64_calls == group.size


def test_probe_reads_the_live_session_count() -> None:
    group = _Group(2)
    transceiver = _make_transceiver(0, group, probe=True)
    transceiver._recv_sessions = {7: object(), 8: object()}
    seen = []

    def _gather(values):
        seen.append(list(values))
        return np.zeros((group.size, 1), dtype=np.int64)

    transceiver._dist.tp_cp_allgather_int64 = _gather
    assert transceiver._any_rank_has_recv_sessions() is False
    assert seen == [[2]]


def _executor(group: _Group, rank: int, fused: bool) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.dist = _GroupDist(rank, group)
    executor._fused_disagg_status_entry = fused
    executor._check_disagg_gen_cache_transfer_status = Mock()
    executor._check_disagg_ctx_cache_transfer_status = Mock()
    return executor


@pytest.mark.parametrize("num_ranks", [2, 4])
def test_fused_idle_progress_votes_match_separate_votes(num_ranks: int) -> None:
    # Every combination of per-rank inputs (num_fitting_reqs == 0, admission wait, idle).
    per_rank_inputs = list(itertools.product([False, True], repeat=3))
    for combo in itertools.product(per_rank_inputs, repeat=num_ranks):
        group = _Group(num_ranks)
        # Votes as each rank computes them in _check_disagg_transfer_progress_when_idle.
        group.gen_votes = [int(no_fit and wait) for no_fit, wait, _ in combo]
        group.ctx_votes = [int(idle or no_fit) for no_fit, _, idle in combo]
        group.int64_inputs = lambda g=group: [
            [gen, ctx] for gen, ctx in zip(g.gen_votes, g.ctx_votes)
        ]
        for rank, (no_fit, wait, idle) in enumerate(combo):
            calls = []
            for fused in (False, True):
                executor = _executor(group, rank, fused)
                PyExecutor._check_disagg_transfer_progress_when_idle(
                    executor,
                    num_fitting_reqs=0 if no_fit else 1,
                    fitting_disagg_gen_init_requests=[],
                    wait_for_disagg_gen_transfer_progress=wait,
                    all_gen_first=False,
                    is_idle=idle,
                )
                calls.append(
                    (
                        executor._check_disagg_gen_cache_transfer_status.call_args_list,
                        executor._check_disagg_ctx_cache_transfer_status.call_args_list,
                    )
                )
            assert calls[0] == calls[1], (combo, rank)
        # Fused: one int64 allgather per rank; separate: one or two pickled collectives.
        assert group.int64_calls == num_ranks


def test_fused_votes_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_DISAGG_GEN_FAST_SYNC", raising=False)
    group = _Group(4)
    executor = object.__new__(PyExecutor)
    executor.dist = _GroupDist(0, group)
    assert PyExecutor._use_fused_disagg_status_entry(executor) is False

    monkeypatch.setenv("TRTLLM_DISAGG_GEN_FAST_SYNC", "1")
    executor = object.__new__(PyExecutor)
    executor.dist = _GroupDist(0, group)
    assert PyExecutor._use_fused_disagg_status_entry(executor) is True

    # PP > 1: the world vote spans more ranks than TP x CP, keep separate collectives.
    executor = object.__new__(PyExecutor)
    executor.dist = _GroupDist(0, group)
    executor.dist.pp_size = 2
    executor.dist.world_size = 8
    assert PyExecutor._use_fused_disagg_status_entry(executor) is False

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
"""``TRTLLM_K3LL_DISAGG_GEN_FAST_SYNC`` (``disaggregation/transceiver.py``, CPU only), on a transceiver whose
collectives are stubbed as 16 ranks of one CP group (Helix16 generation: world = TP x CP, PP 1, no attention DP):

* switch off: ``check_gen_transfer_status`` runs exactly the pre-change collectives (no int64 probe);
* switch on, no receive session on any rank: one int64 all-gather, no pickled all-gather, the same empty status;
* switch on, a session on some other rank, or a blocking / at-least-N poll, or attention DP / PP > 1: the full
  poll runs as with the switch off (same collectives after the probe, same status);
* the switch is read per call.
"""

import numpy as np
import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags
from tensorrt_llm._torch.disaggregation import transceiver as tx

pytestmark = pytest.mark.cpu_only

WORLD = 16


class _Dist:
    def __init__(self, other_rank_sessions=0):
        self.rank = 0
        self.calls = []
        self.other = other_rank_sessions

    def allgather(self, obj):
        self.calls.append("allgather")
        return [obj] * WORLD

    def tp_cp_allgather_int64(self, values):
        self.calls.append("int64")
        rows = [np.asarray(values, dtype=np.int64)] + [np.asarray([self.other], dtype=np.int64)] * (
            WORLD - 1
        )
        return np.stack(rows)


class _Mapping:
    def __init__(self, adp=False, pp=1):
        self.enable_attention_dp = adp
        self.pp_size = pp
        self.world_size = WORLD


def _transceiver(dist, adp=False, pp=1):
    t = object.__new__(tx.KvCacheTransceiverV2)
    m = _Mapping(adp, pp)
    t._mapping = m
    t._dist = dist
    t._ever_had_recv_session = False
    t._gen_need_sync = True
    t._gen_allgather = dist.allgather
    t._gen_idle_probe_ok = t._gen_need_sync and not m.enable_attention_dp and m.pp_size == 1
    t._gen_idle_skips = 0
    t._recv_sessions = {}
    t._recv_reqs = {}
    return t


def _status(st):
    return (
        (list(st[0]), list(st[1]), list(st[2]))
        if isinstance(st, tuple)
        else (list(st.completed), list(st.failed), list(st.cancelled))
    )


@pytest.mark.parametrize("other", (0, 1))
@pytest.mark.parametrize("at_least", (0, None, 2))
@pytest.mark.parametrize("adp,pp", ((False, 1), (True, 1), (False, 2)))
def test_fast_sync_matches_off(monkeypatch, other, at_least, adp, pp):
    if at_least is not None and at_least > 0:
        # an at-least-N poll waits for sessions first; keep the stub from polling
        monkeypatch.setattr(
            tx.KvCacheTransceiverV2, "_poll_gen_sessions_for_poll_interval", lambda *a: None
        )
    out = {}
    for v in (0, 1):
        monkeypatch.setattr(flags, "TRTLLM_K3LL_DISAGG_GEN_FAST_SYNC", v)
        dist = _Dist(other)
        t = _transceiver(dist, adp, pp)
        out[v] = (
            _status(t.check_gen_transfer_status(at_least)),
            list(dist.calls),
            t._gen_idle_skips,
        )
    off_status, off_calls, _ = out[0]
    on_status, on_calls, on_skips = out[1]
    assert "int64" not in off_calls
    assert on_status == off_status == ([], [], [])
    skip = at_least == 0 and other == 0 and not adp and pp == 1
    if skip:
        assert on_calls == ["int64"] and on_skips == 1
    else:
        probed = at_least == 0 and not adp and pp == 1
        assert on_calls == (["int64"] if probed else []) + off_calls and on_skips == 0

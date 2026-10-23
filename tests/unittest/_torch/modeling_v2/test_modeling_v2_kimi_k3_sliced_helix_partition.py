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
"""``TRTLLM_K3LL_SLICED_HELIX_PARTITION`` (``pyexecutor/request_utils.py``, CPU only):

* switch on = switch off, exactly: ``partition_context_for_helix`` (prompt as a list and as an int32 array) and
  ``merge_helix_requests`` (tokens, position ids, ``total_input_len_cp`` / ``seqlen_this_rank_cp`` of every rank's
  LlmRequest) over prompt lengths around the block and rank boundaries, empty ranks included, 16 / 4 / 1 CP ranks;
* the switch is read per call from the flags module (a runtime toggle reaches it);
* the serialized prompt bytes are the prompt (the int32 path is taken, not the fallback);
* timing at 153600 tokens x 16 ranks, printed (no threshold: the host is not fixed).
"""

import time

import numpy as np
import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags
from tensorrt_llm._torch.pyexecutor import request_utils as ru
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm.bindings import executor as trtllm

pytestmark = pytest.mark.cpu_only

LENS = (
    1,
    2,
    63,
    64,
    65,
    127,
    128,
    129,
    64 * 16 - 1,
    64 * 16,
    64 * 16 + 1,
    64 * 17 + 5,
    9999,
    153600,
)


@pytest.fixture
def switch(monkeypatch):
    def set_(v):
        monkeypatch.setattr(flags, "TRTLLM_K3LL_SLICED_HELIX_PARTITION", v)

    return set_


def _request(tokens):
    return trtllm.Request(
        input_token_ids=tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )


@pytest.mark.parametrize("cp_size,tpb", [(16, 64), (4, 2), (1, 64), (16, 32)])
def test_partition_on_equals_off(switch, cp_size, tpb):
    rng = np.random.default_rng(cp_size * 1000 + tpb)
    for n in LENS:
        tokens = rng.integers(0, 160000, n).astype(np.int32)
        as_list = tokens.tolist()
        for rank in range(cp_size):
            switch(0)
            off = ru.partition_context_for_helix(as_list, rank, cp_size, tpb)
            switch(1)
            on_list = ru.partition_context_for_helix(as_list, rank, cp_size, tpb)
            on_arr = ru.partition_context_for_helix(tokens, rank, cp_size, tpb)
            assert on_list == off and on_arr == off, (n, rank)
            for got in (on_list, on_arr):
                assert all(type(x) is int for x in got[0][:3] + got[1][:3])


@pytest.mark.parametrize("n", (13, 64 * 16 + 7, 153600))
def test_merge_on_equals_off(switch, monkeypatch, n):
    cp_size, tpb = 16, 64
    tokens = np.random.default_rng(n).integers(0, 160000, n).astype(np.int32).tolist()
    item = RequestQueueItem(id=7, request=_request(tokens))
    seen = []
    real = ru.executor_request_to_llm_request

    def spy(**kw):
        seen.append((list(kw["input_token_ids"]), list(kw["position_ids"])))
        return real(**kw)

    monkeypatch.setattr(ru, "executor_request_to_llm_request", spy)
    for rank in range(cp_size):
        got = {}
        for v in (0, 1):
            switch(v)
            seen.clear()
            (req,) = ru.merge_helix_requests(
                [item], rank, cp_size, tpb, exclude_last_generation_logits=False
            )
            got[v] = (
                seen[0],
                req.get_tokens(0),
                req.total_input_len_cp,
                req.seqlen_this_rank_cp,
                req.request_id,
            )
        assert got[0] == got[1], (n, rank)
        assert all(type(x) is int for x in got[1][0][0][:3] + got[1][0][1][:3])
    assert (
        ru._PROMPT_BYTES_VERIFIED is True
    )  # the int32-bytes path, not the input_token_ids fallback


def test_prompt_bytes_are_the_prompt():
    tokens = [5, 0, 159999, 2**31 - 1, 7] * 100
    arr = ru._prompt_token_array(_request(tokens))
    assert isinstance(arr, np.ndarray) and arr.tolist() == tokens


def test_timing_153600(switch):
    cp_size, tpb, n = 16, 64, 153600
    tokens = np.random.default_rng(3).integers(0, 160000, n).astype(np.int32).tolist()
    item = RequestQueueItem(id=9, request=_request(tokens))
    ms = {}
    for v in (0, 1, 0, 1):
        switch(v)
        t0 = time.perf_counter()
        for rank in (0, 7, 15):
            ru.merge_helix_requests(
                [item], rank, cp_size, tpb, exclude_last_generation_logits=False
            )
        ms.setdefault(v, []).append((time.perf_counter() - t0) * 1e3 / 3)
    print(
        f"merge_helix_requests per rank at {n} tokens x {cp_size} ranks: off {ms[0]} ms, on {ms[1]} ms"
    )

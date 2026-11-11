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
"""Transfer thread counts from cache_transceiver_config, else the environment."""

import pytest
from pydantic import ValidationError

import tensorrt_llm._torch.disaggregation.native.transfer as transfer
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

pytestmark = pytest.mark.cpu_only


def _agent_threads(monkeypatch, num_threads, env_value):
    created = []
    monkeypatch.setattr(
        transfer, "NixlTransferAgent", lambda *args, **kwargs: created.append(kwargs)
    )
    if env_value is None:
        monkeypatch.delenv("TRTLLM_NIXL_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_NIXL_NUM_THREADS", env_value)
    transfer._create_nixl_agent("agent", rank=0, world_size=1, num_threads=num_threads)
    return created[0]["num_threads"]


@pytest.mark.parametrize(
    ("num_threads", "env_value", "expected"),
    [(None, None, 8), (None, "1", 1), (2, "1", 2), (2, None, 2)],
)
def test_nixl_agent_threads(monkeypatch, num_threads, env_value, expected):
    assert _agent_threads(monkeypatch, num_threads, env_value) == expected


def test_config_fields():
    config = CacheTransceiverConfig()
    assert (config.kv_transfer_num_threads, config.nixl_num_threads) == (None, None)
    config = CacheTransceiverConfig(kv_transfer_num_threads=8, nixl_num_threads=1)
    assert (config.kv_transfer_num_threads, config.nixl_num_threads) == (8, 1)
    with pytest.raises(ValidationError):
        CacheTransceiverConfig(nixl_num_threads=0)

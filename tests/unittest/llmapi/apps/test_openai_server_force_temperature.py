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
"""TRTLLM_SERVE_FORCE_TEMPERATURE parsing (benchmarking aid of the OpenAI server)."""

import pytest

from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve import openai_server


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("1.0", 1.0), ("0", 0.0), ("0.7", 0.7)],
)
def test_forced_sampling_temperature(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TRTLLM_SERVE_FORCE_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_SERVE_FORCE_TEMPERATURE", value)
    assert openai_server._forced_sampling_temperature() == expected


@pytest.mark.parametrize("value", ["-1", "abc"])
def test_forced_sampling_temperature_rejects_invalid(monkeypatch, value):
    monkeypatch.setenv("TRTLLM_SERVE_FORCE_TEMPERATURE", value)
    with pytest.raises(ValueError):
        openai_server._forced_sampling_temperature()


def test_forced_temperature_turns_greedy_request_into_sampling():
    """A client's temperature 0 is greedy; the forced 1.0 is not."""
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4)
    assert params._greedy_decoding
    params.temperature = 1.0
    assert not params._greedy_decoding

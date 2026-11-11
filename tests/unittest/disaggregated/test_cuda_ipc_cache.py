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
"""Bounded UCX CUDA IPC remote-handle cache for KV cache transfers."""

import os

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.disaggregation.cuda_ipc_cache import (
    CUDA_IPC_CACHE_MAX_REGIONS_ENV,
    CUDA_IPC_CACHE_MAX_SIZE_ENV,
    bound_cuda_ipc_cache,
)
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

pytestmark = pytest.mark.cpu_only


def test_bound_applied_when_unset():
    env = {"UCX_TLS": "cuda_ipc,cuda_copy"}
    assert bound_cuda_ipc_cache(512, env) == "512"
    assert env[CUDA_IPC_CACHE_MAX_REGIONS_ENV] == "512"
    assert CUDA_IPC_CACHE_MAX_SIZE_ENV not in env


def test_bound_is_idempotent():
    env = {}
    bound_cuda_ipc_cache(4096, env)
    # A second transceiver in the same process sees the limit as already configured.
    assert bound_cuda_ipc_cache(512, env) is None
    assert env[CUDA_IPC_CACHE_MAX_REGIONS_ENV] == "4096"


@pytest.mark.parametrize(
    "user_env",
    [
        {CUDA_IPC_CACHE_MAX_REGIONS_ENV: "inf"},
        {CUDA_IPC_CACHE_MAX_REGIONS_ENV: "64"},
        {CUDA_IPC_CACHE_MAX_SIZE_ENV: "8G"},
    ],
    ids=["regions_inf", "regions_64", "size_only"],
)
def test_user_limit_respected(user_env):
    env = dict(user_env)
    assert bound_cuda_ipc_cache(512, env) is None
    assert env == user_env


def test_process_environment_used_by_default(monkeypatch):
    # setenv first so monkeypatch restores the original state, including the
    # variable bound_cuda_ipc_cache() sets below.
    for name in (CUDA_IPC_CACHE_MAX_REGIONS_ENV, CUDA_IPC_CACHE_MAX_SIZE_ENV):
        monkeypatch.setenv(name, "0")
        monkeypatch.delenv(name)
    assert bound_cuda_ipc_cache(512) == "512"
    assert os.environ[CUDA_IPC_CACHE_MAX_REGIONS_ENV] == "512"


def test_config_field():
    assert CacheTransceiverConfig().ucx_cuda_ipc_cache_max_regions is None
    config = CacheTransceiverConfig(backend="NIXL", ucx_cuda_ipc_cache_max_regions=512)
    assert config.ucx_cuda_ipc_cache_max_regions == 512
    with pytest.raises(ValidationError):
        CacheTransceiverConfig(ucx_cuda_ipc_cache_max_regions=0)

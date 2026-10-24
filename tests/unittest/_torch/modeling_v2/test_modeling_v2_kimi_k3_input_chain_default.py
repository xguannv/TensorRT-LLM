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
"""``TRTLLM_K3LL_INPUT_CHAIN`` (``kimi_k3_nvfp4/flags.py``): the executor's four input-chain switches
(``pyexecutor/input_commit.py``) default to on for the K3 NVFP4 target. 1 forces the defaults on and 0 off; the default
(-1) turns them on only on ``input_chain.INPUT_CHAIN_CAPABILITIES`` (SM103). A switch set in the environment wins
either way; without the K3 target (no defaults installed) the switches follow the environment only. No GPU needed:
the device capability is patched and no kernel runs.
"""

import pathlib

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import input_commit

ON = {
    "TRTLLM_BATCHED_INPUT_COMMIT": "1",
    "TRTLLM_KV_BLOCK_OFFSETS_TRITON": "2",
    "TRTLLM_STATE_INDEX_COPY_KERNEL": "1",
    "TRTLLM_SPEC_SAMPLER_FUSED_SCATTER": "1",
}


def _input_chain():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        input_chain,
    )

    return input_chain


def _enabled():
    return {
        "commit": input_commit.batched_input_commit_enabled(),
        "triton": input_commit.block_offsets_triton_enabled(),
        "pair": input_commit.block_offsets_pair_enabled(),
        "state": input_commit.state_index_copy_kernel_enabled(),
        "scatter": input_commit.spec_sampler_fused_scatter_enabled(),
    }


ALL_ON = dict(commit=True, triton=True, pair=True, state=True, scatter=True)
ALL_OFF = dict(commit=False, triton=False, pair=False, state=False, scatter=False)


@pytest.fixture
def switches(monkeypatch):
    """Unset environment switches and no model defaults; restored afterwards."""
    saved = dict(input_commit._MODEL_DEFAULTS)
    for name in input_commit.INPUT_CHAIN_SWITCHES:
        monkeypatch.setitem(input_commit._ENV_SWITCHES, name, None)
    input_commit.set_model_defaults({})
    yield monkeypatch
    input_commit._MODEL_DEFAULTS.clear()
    input_commit._MODEL_DEFAULTS.update(saved)
    monkeypatch.undo()
    input_commit._resolve_switches()


def _install(monkeypatch, value, capability):
    chain = _input_chain()
    monkeypatch.setattr(chain._flags, "TRTLLM_K3LL_INPUT_CHAIN", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    chain._on_by_default.cache_clear()
    try:
        return chain.enabled(), chain.install()
    finally:
        chain._on_by_default.cache_clear()


@pytest.mark.parametrize(
    "value,capability,on",
    [
        (1, (10, 3), True),
        (1, (10, 7), True),
        (0, (10, 3), False),
        (0, (10, 7), False),
        (-1, (10, 3), True),
        (-1, (10, 7), False),
        (-1, (10, 0), False),
    ],
)
def test_switch_value_and_capability(switches, value, capability, on):
    enabled, resolved = _install(switches, value, capability)
    assert enabled is on
    assert _enabled() == (ALL_ON if on else ALL_OFF)
    expected = (lambda name: (ON[name], "model default")) if on else (lambda name: ("0", "default"))
    assert resolved == {name: expected(name) for name in input_commit.INPUT_CHAIN_SWITCHES}


def test_env_zero_wins_over_k3_default(switches):
    switches.setitem(input_commit._ENV_SWITCHES, "TRTLLM_BATCHED_INPUT_COMMIT", "0")
    switches.setitem(input_commit._ENV_SWITCHES, "TRTLLM_KV_BLOCK_OFFSETS_TRITON", "1")
    _, resolved = _install(switches, 1, (10, 3))
    assert _enabled() == dict(commit=False, triton=True, pair=False, state=True, scatter=True)
    assert resolved["TRTLLM_BATCHED_INPUT_COMMIT"] == ("0", "env")
    assert resolved["TRTLLM_KV_BLOCK_OFFSETS_TRITON"] == ("1", "env")
    assert resolved["TRTLLM_STATE_INDEX_COPY_KERNEL"] == ("1", "model default")


def test_env_one_wins_over_k3_off(switches):
    switches.setitem(input_commit._ENV_SWITCHES, "TRTLLM_SPEC_SAMPLER_FUSED_SCATTER", "1")
    switches.setitem(input_commit._ENV_SWITCHES, "TRTLLM_KV_BLOCK_OFFSETS_TRITON", "2")
    _, resolved = _install(switches, 0, (10, 3))
    assert _enabled() == dict(commit=False, triton=True, pair=True, state=False, scatter=True)
    assert resolved["TRTLLM_SPEC_SAMPLER_FUSED_SCATTER"] == ("1", "env")
    assert resolved["TRTLLM_BATCHED_INPUT_COMMIT"] == ("0", "default")


def test_without_k3_target_environment_only(switches):
    # No model installed defaults: everything off unless set in the environment.
    assert _enabled() == ALL_OFF
    switches.setitem(input_commit._ENV_SWITCHES, "TRTLLM_STATE_INDEX_COPY_KERNEL", "1")
    input_commit._resolve_switches()
    assert _enabled() == dict(ALL_OFF, state=True)
    # A K3 target built with its switch off clears the defaults a previous one installed.
    _install(switches, 1, (10, 3))
    assert input_commit.batched_input_commit_enabled()
    _install(switches, 0, (10, 3))
    assert _enabled() == dict(ALL_OFF, state=True)


def test_generic_code_does_not_import_k3():
    # The executor modules consult input_commit only; the K3 target pushes its defaults in.
    root = pathlib.Path(input_commit.__file__).resolve().parents[1]
    for rel in (
        "pyexecutor/input_commit.py",
        "pyexecutor/engine/runners/decoder/runner.py",
        "pyexecutor/kv_cache/kv_cache_manager_v2.py",
        "pyexecutor/kv_cache/mamba_cache_manager.py",
        "speculative/spec_sampler_base.py",
        "attention/backends/trtllm.py",
    ):
        assert "kimi_k3_nvfp4" not in (root / rel).read_text(), rel


def test_default_is_auto():
    chain = _input_chain()
    assert chain._flags._FLAGS["TRTLLM_K3LL_INPUT_CHAIN"][1] == -1
    assert chain.INPUT_CHAIN_CAPABILITIES == ((10, 3),)
    assert chain.INPUT_CHAIN_DEFAULTS == ON
    assert "TRTLLM_FUSED_HELIX_SPEC_PREPROCESS" not in chain.INPUT_CHAIN_DEFAULTS

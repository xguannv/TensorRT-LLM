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
"""``TRTLLM_K3LL_SPEC_STEP_COPIES_SM103`` and ``TRTLLM_K3LL_SPEC_HEAD_FUSED`` (``kimi_k3_nvfp4/flags.py``) reach the
engine's generic modules through ``kimi_k3_nvfp4/engine_switches.py``, no GPU needed:

* both are registered with default 0;
* after ``install`` the generic modules read the flags module's attribute on every call (what a one-process A/B
  sets between steps and recaptures), whatever the environment says; ``uninstall`` gives the environment back;
* no K3 module other than ``flags`` binds either name (a switch bound at import would not follow the A/B);
* with the census on, each launch the generic modules report is counted as ``dispatch:spec_step_copies_sm103:<mode>``
  / ``dispatch:spec_head_fused:<mode>``; with it off nothing is installed to count.

The values those paths compute are tested in ``test_spec_step_copies_sm103.py`` and ``test_spec_head_fused.py``.
"""

import sys

import pytest

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
_SWITCHES = ("TRTLLM_K3LL_SPEC_STEP_COPIES_SM103", "TRTLLM_K3LL_SPEC_HEAD_FUSED")


def _modules():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        census,
        engine_switches,
        flags,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.spec_step_copies import op
    from tensorrt_llm._torch.speculative import spec_head_kernels

    return flags, census, engine_switches, op, spec_head_kernels


@pytest.fixture(autouse=True)
def _restore():
    yield
    _modules()[2].uninstall()


def test_registered_default_off():
    flags = _modules()[0]
    table = flags.flags()
    for name in _SWITCHES:
        assert name in table, name
        assert table[name][1] == 0, f"{name} must default to 0"


@pytest.mark.parametrize("env", [None, "0", "1"], ids=["unset", "0", "1"])
def test_generic_modules_read_the_flags_per_call(monkeypatch, env):
    flags, _, engine_switches, op, spec_head_kernels = _modules()
    for name in _SWITCHES:
        if env is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, env)
    engine_switches.install()
    for step, (copies, head) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1), (0, 0))):
        monkeypatch.setattr(flags, "TRTLLM_K3LL_SPEC_STEP_COPIES_SM103", copies)
        monkeypatch.setattr(flags, "TRTLLM_K3LL_SPEC_HEAD_FUSED", head)
        assert op.sm103_enabled() is bool(copies), step
        assert spec_head_kernels.spec_head_fusion_enabled() is bool(head), step
    engine_switches.uninstall()
    assert op.sm103_enabled() == (env == "1")
    assert spec_head_kernels.spec_head_fusion_enabled() == (env == "1")


def test_no_k3_module_binds_the_switches():
    """What the one-process A/B tool refuses: a K3 module other than flags holding the switch by value."""
    _modules()
    bound = [
        (module_name, name)
        for module_name, module in list(sys.modules.items())
        if module is not None and module_name.startswith(_PKG) and module_name != f"{_PKG}.flags"
        for name in _SWITCHES
        if name in getattr(module, "__dict__", {})
    ]
    assert not bound, bound


@pytest.mark.parametrize("census_on", [True, False], ids=["census", "no_census"])
def test_dispatch_census_keys(monkeypatch, census_on):
    _, census, engine_switches, op, spec_head_kernels = _modules()
    monkeypatch.setattr(census, "ENABLED", census_on)
    monkeypatch.setattr(census, "_counts", type(census._counts)())
    engine_switches.install()
    spec_head_kernels.note_dispatch()
    spec_head_kernels.note_dispatch()
    if census_on:
        op._sm103_dispatched()  # what a launch on SM 103 calls
        assert census.counts() == {
            "dispatch:spec_head_fused:eager": 2,
            "dispatch:spec_step_copies_sm103:eager": 1,
        }
    else:
        assert op._sm103_dispatched is None
        assert census.counts() == {}

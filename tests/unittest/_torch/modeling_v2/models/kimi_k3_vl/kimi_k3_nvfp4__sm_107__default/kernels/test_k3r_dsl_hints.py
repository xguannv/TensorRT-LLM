# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 Rubin CuTe DSL scheduling hints (k3_rubin/kernels/dsl_hints.py).

Each wrapper emits its ``cute.nvgpu`` hint only when the DSL provides all four hints, and nothing
otherwise. Both cases execute the module against a fake ``cutlass.cute``, so they need neither a
GPU nor the CuTe DSL; where the DSL is installed, the real module is checked against it."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_KERNELS = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default.kernels"
_HINTS = ("cfence", "warp_switch", "sched_res_busy_xu64", "reset_sched_res_busy_xu64")


def _dsl_hints_source() -> Path:
    # Locates the package without importing it.
    package_dir = importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]
    return Path(package_dir, *_KERNELS.split(".")[1:], "dsl_hints.py")


def _load_dsl_hints(monkeypatch, nvgpu_hints):
    """Execute a private copy of dsl_hints against a fake ``cutlass.cute`` whose ``nvgpu`` has
    exactly ``nvgpu_hints``. Returns the module and the names of the hints it emitted."""
    emitted = []
    nvgpu = types.SimpleNamespace(
        **{name: (lambda name=name: emitted.append(name)) for name in nvgpu_hints}
    )
    cute = types.ModuleType("cutlass.cute")
    cute.nvgpu = nvgpu
    cutlass = types.ModuleType("cutlass")
    cutlass.cute = cute
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    spec = importlib.util.spec_from_file_location("_k3r_dsl_hints_under_test", _dsl_hints_source())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, emitted


@pytest.mark.parametrize(
    "nvgpu_hints", [(), ("cfence",), _HINTS[:3]], ids=["public", "cfence-only", "three-of-four"]
)
def test_wrappers_emit_nothing_without_every_hint(monkeypatch, nvgpu_hints):
    module, emitted = _load_dsl_hints(monkeypatch, nvgpu_hints)

    assert module.HAS_SCHEDULING_HINTS is False
    for name in _HINTS:
        getattr(module, name)()
    assert emitted == []


def test_wrappers_emit_their_hint_with_every_hint(monkeypatch):
    module, emitted = _load_dsl_hints(monkeypatch, _HINTS)

    assert module.HAS_SCHEDULING_HINTS is True
    for name in _HINTS:
        getattr(module, name)()
    assert emitted == list(_HINTS)
    module.cfence()
    module.cfence()
    assert emitted[len(_HINTS) :] == ["cfence", "cfence"]


def test_installed_dsl_hints_are_shared_with_the_context_fmha_helpers():
    cute = pytest.importorskip("cutlass.cute")
    dsl_hints = importlib.import_module(f"{_KERNELS}.dsl_hints")
    helpers = importlib.import_module(f"{_KERNELS}.k3_context_fmha_helpers")

    assert dsl_hints.HAS_SCHEDULING_HINTS == all(hasattr(cute.nvgpu, name) for name in _HINTS)
    assert helpers.HAS_SCHEDULING_HINTS is dsl_hints.HAS_SCHEDULING_HINTS
    for name in _HINTS:
        assert getattr(helpers, name) is getattr(dsl_hints, name)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 NVFP4 switch table (``flags.py``) and path census (``census.py``), host-side, no GPU."""

import importlib
import types

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"


def _reload(monkeypatch, census_value):
    if census_value is None:
        monkeypatch.delenv("TRTLLM_K3LL_CENSUS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_K3LL_CENSUS", census_value)
    flags = importlib.reload(importlib.import_module(f"{_PKG}.flags"))
    census = importlib.reload(importlib.import_module(f"{_PKG}.census"))
    return flags, census


def _decode_step(attn_metadata, num_tokens):
    """Stand-in classifier: a step of generation requests only is a decode step, anything else generic."""
    if attn_metadata.num_contexts == 0:
        return types.SimpleNamespace(decode=True)
    return None


def _classes():
    class Mixer:
        def forward(self, x, take_k3):
            return x + 1 if take_k3 else self.forward_generic(x)

        def forward_generic(self, x):
            return x + 2

    class Model:
        def __init__(self):
            self.mixer = Mixer()

        def forward(
            self, attn_metadata, input_ids=None, position_ids=None, inputs_embeds=None, take_k3=True
        ):
            return self.mixer.forward(input_ids, take_k3)

    return Model, Mixer


def _meta(num_contexts):
    return types.SimpleNamespace(num_contexts=num_contexts)


def test_flags_default_and_override(monkeypatch):
    flags, _ = _reload(monkeypatch, None)
    assert flags.TRTLLM_K3LL_CENSUS == 0
    assert flags.flags()["TRTLLM_K3LL_CENSUS"][:2] == (0, 0)
    flags, _ = _reload(monkeypatch, "1")
    assert flags.TRTLLM_K3LL_CENSUS == 1
    flags.log_flags()


def test_flags_reject_non_integer(monkeypatch):
    monkeypatch.setenv("TRTLLM_K3LL_CENSUS", "yes")
    with pytest.raises(ValueError, match="TRTLLM_K3LL_CENSUS"):
        importlib.reload(importlib.import_module(f"{_PKG}.flags"))
    monkeypatch.delenv("TRTLLM_K3LL_CENSUS")
    importlib.reload(importlib.import_module(f"{_PKG}.flags"))


def test_census_off_leaves_classes_untouched(monkeypatch):
    _, census = _reload(monkeypatch, None)
    Model, Mixer = _classes()
    forward, forward_generic = Model.forward, Mixer.forward_generic
    census.install_model(Model, _decode_step)
    census.install(Mixer)
    assert Model.forward is forward and Mixer.forward_generic is forward_generic
    assert Model().forward(_meta(0), torch.zeros(1), take_k3=False).item() == 2
    assert census.counts() == {}


def test_census_counts_fallbacks_only_inside_decode_steps(monkeypatch):
    _, census = _reload(monkeypatch, "1")
    Model, Mixer = _classes()
    census.install_model(Model, _decode_step)
    census.install(Mixer)
    model = Model()
    x = torch.zeros(1)
    # Results are those of the unwrapped code.
    assert model.forward(_meta(0), x).item() == 1  # decode step, decode kernels
    assert model.forward(_meta(0), x, take_k3=False).item() == 2  # decode step, fallback
    assert model.forward(_meta(1), x, take_k3=False).item() == 2  # generic step: not a fallback
    assert model.forward(_meta(0), input_ids=x, take_k3=False).item() == 2  # keyword call
    assert census.counts() == {
        "step:decode:eager": 3,
        "step:generic:eager": 1,
        "fallback:Mixer:eager": 2,
    }
    # Installing twice wraps once.
    census.install(Mixer)
    census.install_model(Model, _decode_step)
    model.forward(_meta(0), x, take_k3=False)
    assert census.counts()["fallback:Mixer:eager"] == 3
    _reload(monkeypatch, None)

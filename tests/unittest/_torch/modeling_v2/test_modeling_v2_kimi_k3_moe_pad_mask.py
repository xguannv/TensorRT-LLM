# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_MOE_PAD_MASK`` (``kimi_k3_nvfp4/decode_moe_lowm.py``) host-side, no GPU: the switch's default and doc,
``pad_rows``' decision, the NVFP4 MoE layer handing the mask to the front's sort only (and counting
``dispatch:moe_pad_mask``), and the MoE module passing a step's ``pad_rows`` to the NVFP4 layer only. The kernel side
(the sort leaving the padding requests' tokens out, the real rows' expert outputs bit for bit) is
``cute_dsl_kernels/kimi_k3/test_k3_moe_front_pad_mask.py`` on 4 SM100 / SM103 GPUs.
"""

import types
from dataclasses import replace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{_PKG}.{name}")


def _step(requests, tokens):
    return _mod("implementation").DecodeStep(requests * tokens, requests, tokens)


def _manager(mask):
    return types.SimpleNamespace(_dummy_request_mask=mask)


def test_switch_default_off_and_documented():
    value, default, doc = _mod("flags").flags()["TRTLLM_K3LL_MOE_PAD_MASK"]
    assert default == 0
    assert "padding" in doc and "forward time" in doc and "FRONT_SORT" in doc


@pytest.mark.parametrize("requests,tokens", [(4, 8), (2, 8), (4, 1), (1, 8), (8, 1)])
def test_pad_rows_takes_a_decode_step(monkeypatch, requests, tokens):
    lowm = _mod("decode_moe_lowm")
    monkeypatch.setattr(lowm._flags, "TRTLLM_K3LL_MOE_PAD_MASK", 1)
    mask = torch.zeros(8, dtype=torch.bool)
    pad = lowm.pad_rows(_step(requests, tokens), _manager(mask), torch.device("cpu"))
    assert pad is not None and pad.mask is mask and pad.tokens_per_request == tokens


def test_pad_rows_off_by_default(monkeypatch):
    lowm = _mod("decode_moe_lowm")
    monkeypatch.setattr(lowm._flags, "TRTLLM_K3LL_MOE_PAD_MASK", 0)
    mask = torch.zeros(8, dtype=torch.bool)
    assert lowm.pad_rows(_step(4, 8), _manager(mask), torch.device("cpu")) is None


@pytest.mark.parametrize(
    "case",
    ["no_step", "not_decode", "no_manager", "no_mask", "int_mask", "short_mask", "2d_mask", "other_device",
     "strided_mask"],
)  # fmt: skip
def test_pad_rows_declines(monkeypatch, case):
    """Every step or manager the mask does not cover keeps the sort as it was (None)."""
    impl, lowm = _mod("implementation"), _mod("decode_moe_lowm")
    monkeypatch.setattr(lowm._flags, "TRTLLM_K3LL_MOE_PAD_MASK", 1)
    step, mask, device = _step(4, 8), torch.zeros(8, dtype=torch.bool), torch.device("cpu")
    manager = _manager(mask)
    if case == "no_step":
        step = None
    elif case == "not_decode":
        step = impl.DecodeStep(6)  # a small step with context requests: no per-request rows
    elif case == "no_manager":
        manager = None
    elif case == "no_mask":
        manager = _manager(None)
    elif case == "int_mask":
        manager = _manager(torch.zeros(8, dtype=torch.int32))
    elif case == "short_mask":
        manager = _manager(torch.zeros(3, dtype=torch.bool))
    elif case == "2d_mask":
        manager = _manager(torch.zeros(2, 4, dtype=torch.bool))
    elif case == "other_device":
        device = torch.device("cuda", 0)
    elif case == "strided_mask":
        manager = _manager(torch.zeros(16, dtype=torch.bool)[::2])
    assert lowm.pad_rows(step, manager, device) is None


def test_decode_step_compares_without_pad_rows():
    impl, lowm = _mod("implementation"), _mod("decode_moe_lowm")
    step = _step(4, 8)
    assert step.pad_rows is None
    padded = replace(step, pad_rows=lowm.PadRows(torch.zeros(4, dtype=torch.bool), 8))
    assert padded == step and hash(padded) == hash(step)
    assert "pad_rows" not in repr(padded)
    assert impl.DecodeStep(32, 4, 8).wide and padded.wide


def _layer(lowm, front_sort: bool):
    """A K3DecodeMoeNvfp4Layer without weights: only what `_front_only` reads."""
    layer = object.__new__(lowm.K3DecodeMoeNvfp4Layer)
    layer.state = types.SimpleNamespace(
        sort_ctr=torch.zeros(1, dtype=torch.int32) if front_sort else None,
        front_head=lambda rows, front32=None: "head",
    )
    layer.front_weight = layer.bias = layer.input_scale = None
    layer.shared_cols, layer.offset = 64, 224
    layer.experts = (torch.empty(224, 1),)
    return layer


def _moe():
    return types.SimpleNamespace(
        gate=types.SimpleNamespace(routed_scaling_factor=2.5), _situ_betas=(4.0, 25.0)
    )


@pytest.mark.parametrize("front_sort", [True, False])
@pytest.mark.parametrize("with_pad", [True, False])
def test_front_only_hands_the_mask_to_the_sort(monkeypatch, front_sort, with_pad):
    lowm, census = _mod("decode_moe_lowm"), _mod("census")
    calls = []

    def fake_front(*args, **kwargs):
        calls.append(kwargs)
        rows = args[0].shape[0]
        return (None, None, torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(rows, 1))

    monkeypatch.setattr(lowm, "k3_moe_front", fake_front)
    # The sort's layout without the kernel module (its words only size sort_out here).
    monkeypatch.setattr(lowm, "sort_layout", lambda rows, num_local: (0, 0, 0, rows * 16))
    monkeypatch.setattr(census, "ENABLED", True)
    monkeypatch.setattr(census, "_counts", type(census._counts)())
    mask = torch.zeros(4, dtype=torch.bool)
    pad = lowm.PadRows(mask, 8) if with_pad else None
    routing, _ = _layer(lowm, front_sort)._front_only(_moe(), torch.zeros(32, 16), True, pad)
    (kwargs,) = calls
    assert (kwargs["sort_out"] is not None) == front_sort and (routing[0] is not None) == front_sort
    engaged = with_pad and front_sort
    if engaged:
        assert kwargs["pad_mask"] is mask and kwargs["pad_tokens"] == 8
    else:
        assert kwargs.get("pad_mask") is None and kwargs.get("pad_tokens", 1) == 1
    counted = sum(v for k, v in census.counts().items() if k.startswith("dispatch:moe_pad_mask:"))
    assert counted == int(engaged)


@pytest.mark.parametrize("nvfp4_layer", [True, False])
@pytest.mark.parametrize("with_pad", [True, False])
def test_moe_passes_pad_rows_to_the_nvfp4_layer(nvfp4_layer, with_pad):
    impl, lowm = _mod("implementation"), _mod("decode_moe_lowm")
    seen = []
    decode = object.__new__(lowm.K3DecodeMoeNvfp4Layer) if nvfp4_layer else types.SimpleNamespace()
    decode.takes = lambda hidden_states, step, partial_tail: True
    decode.forward = lambda *args, **kwargs: seen.append(kwargs) or "out"
    moe = types.SimpleNamespace(decode_moe=decode, decode_gemvs=None)
    step = _step(4, 8)
    pad = lowm.PadRows(torch.zeros(4, dtype=torch.bool), 8)
    if with_pad:
        step = replace(step, pad_rows=pad)
    out = impl.KimiK3MoERuntime.forward(moe, torch.zeros(32, 16), partial_tail=True, step=step)
    assert out == "out"
    (kwargs,) = seen
    if with_pad and nvfp4_layer:
        assert kwargs["pad"] is pad
    else:
        assert "pad" not in kwargs

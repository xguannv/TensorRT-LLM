# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_MOE_LOWM_FC12`` (``kimi_k3_nvfp4/decode_moe_lowm.py``) on one process, no GPU needed: the switch's
default and doc, the engagement decision by rows, the FC1 output buffer's size, and the wide MoE step's dispatch with
stand-ins. The kernels against ``K3MoeLowMDynKernel`` (SM100 / SM103) are
``cute_dsl_kernels/kimi_k3/test_k3_moe_lowm.py::test_moe_nvfp4_lowm_fc12``.
"""

from types import SimpleNamespace

import pytest
import torch

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{_PKG}.{name}")


def _lowm_op():
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    return nvfp4_lowm_op


def test_switch_default_sm103_and_documented():
    _, default, doc = _mod("flags").flags()["TRTLLM_K3LL_MOE_LOWM_FC12"]
    assert default == -1 and "SM103" in doc
    assert "Not bitwise" in doc and "17-32" in doc and "at most 16" in doc
    op = _lowm_op()
    assert (op.FC12_MIN_TOKENS, op.FC12_MAX_TOKENS) == (17, 32)


@pytest.mark.parametrize(
    "value,rows,built,fc12",
    [
        (0, 17, True, False),
        (0, 32, True, False),
        (1, 8, True, False),
        (1, 9, True, False),
        (1, 16, True, False),
        (1, 17, True, True),
        (1, 24, True, True),
        (1, 32, True, True),
        (1, 33, True, False),
        (1, 17, False, False),
        (1, 32, False, False),
    ],
)
def test_takes_by_rows(monkeypatch, value, rows, built, fc12):
    """Off (the default) nothing changes; on, 17-32 rows run FC1 + FC2 where the load built the state, at most 16
    rows keep the single kernel."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_LOWM_FC12", value)
    state = SimpleNamespace(fc12=object() if built else None)
    assert dm.fc12_takes(state, rows) is fc12


def test_takes_off_reads_no_state(monkeypatch):
    """Off, the decision does not touch the state (the switch-off path is the one before the switch)."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_LOWM_FC12", 0)
    assert dm.fc12_takes(SimpleNamespace(), 32) is False


@pytest.mark.parametrize(
    "tokens,top_k,local,tiles",
    [
        (32, 16, 224, 226),
        (17, 16, 224, 224 + (272 - 224) // 128),
        (1, 16, 224, 16),
        (8, 8, 16, 16 + 48 // 128),
    ],
)
def test_max_permuted_rows(tokens, top_k, local, tiles):
    """``moe_sort``'s allocation (getMaxPermutedPaddedCount), which bounds the front's sort (one tile per touched
    expert)."""
    op = _lowm_op()
    rows = op.max_permuted_rows(tokens, top_k, local)
    assert rows == tiles * 128
    assert rows >= min(tokens * top_k, local) * 128


def test_new_fc12_sizes():
    op = _lowm_op()
    buffers = op.new_fc12(torch.device("cpu"), 1536, 224, 16)
    rows = op.max_permuted_rows(32, 16, 224)
    assert tuple(buffers.h.shape) == (rows, 768 // 2) and buffers.h.dtype == torch.uint8
    assert buffers.h_sf.numel() == rows * 768 // 16 and not buffers.zeroed


class _Stop(Exception):
    pass


def _wide_layer(dm, monkeypatch, calls):
    """A ``K3DecodeMoeNvfp4Layer`` whose front, experts and FC12 state are stand-ins; records what ``_wide`` calls."""
    layer = object.__new__(dm.K3DecodeMoeNvfp4Layer)

    def begin(rows, cols, device):
        step = ("step", rows, cols)
        calls.append(("begin", rows, cols))
        return step

    layer.state = SimpleNamespace(
        front_head=lambda rows, front32=None: object(),
        head32=None,
        lu=None,
        fc12=SimpleNamespace(begin=begin),
    )
    layer.lu = None
    layer.shared_cols = 384
    layer.experts = (None, None, None, torch.zeros(1, 3584, 1))

    def front_routed(moe, x, front32=None, fc12=None):
        calls.append(("front_routed", fc12))
        raise _Stop

    layer._front_routed = front_routed
    monkeypatch.setattr(dm._tail_lu, "takes", lambda *args: False)
    return layer


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("rows", [9, 16, 17, 32])
def test_wide_dispatch(monkeypatch, value, rows):
    """On, a step of 17-32 rows starts zeroing its routed output before the front (``K3MoeFc12.begin``) and hands
    the step to the routed experts; otherwise the experts get no step (the single kernel, as before)."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_LOWM_FC12", value)
    calls = []
    layer = _wide_layer(dm, monkeypatch, calls)
    x = torch.zeros(rows, 7168, dtype=torch.bfloat16)
    with pytest.raises(_Stop):
        dm.K3DecodeMoeNvfp4Layer._wide(layer, None, x, gemvs=None)
    if value and rows >= 17:
        step = ("step", rows, 3584)
        assert calls == [("begin", rows, 3584), ("front_routed", step)]
    else:
        assert calls == [("front_routed", None)]


@pytest.mark.parametrize(
    "cap,on", [((10, 3), True), ((10, 0), False), ((10, 7), False), ((9, 0), False)]
)
def test_default_follows_capability(monkeypatch, cap, on):
    """-1 (the default): on only on SM103; 1 / 0 override on every device."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    assert dm.MOE_LOWM_FC12_CAPABILITIES == ((10, 3),)
    monkeypatch.setattr(dm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dm.torch.cuda, "get_device_capability", lambda *a: cap)
    dm._fc12_on_by_default.cache_clear()
    try:
        monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_LOWM_FC12", -1)
        assert dm.fc12_enabled() is on
        assert dm.fc12_takes(SimpleNamespace(fc12=object()), 32) is on
        monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_LOWM_FC12", 1)
        assert dm.fc12_enabled() is True
        monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_LOWM_FC12", 0)
        assert dm.fc12_enabled() is False
    finally:
        dm._fc12_on_by_default.cache_clear()

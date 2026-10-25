# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_MOE_SELF_SORT`` (``kimi_k3_nvfp4/decode_moe_lowm.py``, ``nvfp4_lowm_op.run(self_sort=True)``) on
one process, no GPU needed: the switch's default and doc, the engagement by tokens and local experts, and the decode
layer's front / experts calls with stand-ins (the front skips its sort exactly where the expert kernel sorts). The
kernel's own sort against ``moe_sort`` (SM100 / SM103) is ``cute_dsl_kernels/kimi_k3/test_k3_moe_lowm.py``.
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
    _, default, doc = _mod("flags").flags()["TRTLLM_K3LL_MOE_SELF_SORT"]
    assert default == -1
    assert (
        "at most 16 tokens" in doc
        and "k3_moe_front" in doc
        and "moe_sort" in doc
        and "SM103" in doc
    )


@pytest.mark.parametrize("value,capability,on", [
    (-1, (10, 3), True), (-1, (10, 0), False), (-1, (10, 7), False), (1, (10, 0), True), (0, (10, 3), False),
])  # fmt: skip
def test_default_follows_the_capability(monkeypatch, value, capability, on):
    """-1 resolves by ``MOE_SELF_SORT_CAPABILITIES`` (SM103 only); 1 / 0 force it either way."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_SELF_SORT", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    dm._self_sort_on_by_default.cache_clear()
    try:
        assert dm.self_sort_enabled() is on
    finally:
        dm._self_sort_on_by_default.cache_clear()


@pytest.mark.parametrize(
    "tokens,local,takes",
    [(0, 224, False), (1, 224, True), (8, 224, True), (9, 224, True), (16, 224, True), (17, 224, False),
     (32, 224, False), (16, 256, True), (16, 257, False)],
)  # fmt: skip
def test_takes_by_tokens_and_local_experts(tokens, local, takes):
    op = _lowm_op()
    assert op.self_sort_takes(tokens, local) is takes


def _layer(dm, sort_ctr=True):
    layer = object.__new__(dm.K3DecodeMoeNvfp4Layer)
    layer.state = SimpleNamespace(sort_ctr=torch.zeros(1) if sort_ctr else None, counters=None)
    layer.experts = (torch.zeros(224, 1),)
    layer.stats = None
    layer.layer_idx = 3
    layer.offset, layer.fc2_input_scale, layer.situ = 0, None, (1.0, 1.0)
    return layer


@pytest.mark.parametrize("switch", [0, 1])
@pytest.mark.parametrize("rows", [8, 16, 24])
def test_front_sorts_only_where_the_kernel_does_not(monkeypatch, switch, rows):
    """With the front's sort built, a step the expert kernel sorts (switch on, at most 16 rows) runs the front
    without ``sort_out`` and hands the ids to the experts with ``self_sort``; every other step keeps the front's
    sort."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_SELF_SORT", switch)
    layer = _layer(dm)
    fronts, runs = [], []

    def front(moe, x, sort_out=None, front32=None):
        fronts.append(sort_out is not None)
        return "ids", "weights", "x_q", "x_sf", "shared"

    monkeypatch.setattr(layer, "_front", front, raising=False)
    monkeypatch.setattr(dm, "sort_layout", lambda rows, local: (0, 0, 0, 16))
    monkeypatch.setattr(
        layer, "_routed", lambda *a, fc12=None, self_sort=None: runs.append(("routed", self_sort)) or "y",
        raising=False,
    )  # fmt: skip
    monkeypatch.setattr(
        dm.nvfp4_lowm_op, "run_sorted", lambda *a, **k: runs.append(("sorted",)) or "y"
    )
    monkeypatch.setattr(dm, "sort_views", lambda *a: ())
    x = torch.zeros(rows, 4)
    routing, shared = layer._front_only(None, x)
    layer._routed_after(routing)
    kernel_sorts = bool(switch) and rows <= 16
    assert fronts == [not kernel_sorts]
    assert runs == ([("routed", None)] if kernel_sorts else [("sorted",)])
    assert layer.self_sort_takes(rows) is kernel_sorts


@pytest.mark.parametrize("switch", [0, 1])
def test_routed_passes_the_decision_to_run(monkeypatch, switch):
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_SELF_SORT", switch)
    layer = _layer(dm, sort_ctr=False)
    seen = []
    monkeypatch.setattr(
        dm.nvfp4_lowm_op, "run", lambda *a, self_sort=False, **k: seen.append(self_sort) or "y"
    )
    for rows in (8, 16, 24):
        layer._routed(torch.zeros(rows, 16, dtype=torch.int32), None, None, None)
    assert seen == [bool(switch), bool(switch), False]
    seen.clear()
    layer._routed(torch.zeros(8, 16, dtype=torch.int32), None, None, None, self_sort=not switch)
    assert seen == [not switch]


# ``TRTLLM_K3LL_MOE_SELF_SORT_FC12``: the same for the 17-32-token FC1 + FC2 launch.
def test_fc12_switch_default_sm103_and_documented():
    _, default, doc = _mod("flags").flags()["TRTLLM_K3LL_MOE_SELF_SORT_FC12"]
    assert default == -1
    assert (
        "17-32 tokens" in doc
        and "K3MoeFc1LowMKernel" in doc
        and "k3_moe_front" in doc
        and "SM103" in doc
    )


@pytest.mark.parametrize("value,capability,on", [
    (-1, (10, 3), True), (-1, (10, 0), False), (-1, (10, 7), False), (1, (10, 0), True), (0, (10, 3), False),
])  # fmt: skip
def test_fc12_default_follows_the_capability(monkeypatch, value, capability, on):
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_SELF_SORT_FC12", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    dm._self_sort_on_by_default.cache_clear()
    try:
        assert dm.self_sort_fc12_enabled() is on
    finally:
        dm._self_sort_on_by_default.cache_clear()


@pytest.mark.parametrize(
    "tokens,local,takes",
    [(0, 224, False), (1, 224, True), (17, 224, True), (32, 224, True), (33, 224, False), (32, 256, True),
     (32, 257, False)],
)  # fmt: skip
def test_fc12_takes_by_tokens_and_local_experts(tokens, local, takes):
    assert _lowm_op().self_sort_takes(tokens, local, fc12=True) is takes


@pytest.mark.parametrize("small,fc12_switch", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("rows", [16, 24, 32])
def test_fc12_front_sorts_only_where_the_kernels_do_not(monkeypatch, small, fc12_switch, rows):
    """On an FC1 + FC2 step the front skips its sort exactly where the FC12 switch is on (the 16-token switch does
    not reach it), and `_routed` passes the same decision to ``run``."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_SELF_SORT", small)
    monkeypatch.setattr(dm._flags, "TRTLLM_K3LL_MOE_SELF_SORT_FC12", fc12_switch)
    layer = _layer(dm)
    fronts, runs = [], []

    def front(moe, x, sort_out=None, front32=None):
        fronts.append(sort_out is not None)
        return torch.zeros(x.shape[0], 16, dtype=torch.int32), "weights", "x_q", "x_sf", "shared"

    monkeypatch.setattr(layer, "_front", front, raising=False)
    monkeypatch.setattr(dm, "sort_layout", lambda rows, local: (0, 0, 0, 16))
    monkeypatch.setattr(
        dm.nvfp4_lowm_op,
        "run",
        lambda *a, self_sort=False, fc12=None, **k: runs.append(("run", self_sort)) or "y",
    )
    monkeypatch.setattr(
        dm.nvfp4_lowm_op, "run_sorted", lambda *a, **k: runs.append(("sorted",)) or "y"
    )
    monkeypatch.setattr(dm, "sort_views", lambda *a: ())
    step = SimpleNamespace(out=None, launch=lambda: "fc12")
    layer._front_routed(None, torch.zeros(rows, 4), fc12=step)
    kernels_sort = bool(fc12_switch)
    assert fronts == [not kernels_sort]
    assert runs == ([("run", True)] if kernels_sort else [("sorted",)])
    assert layer.self_sort_takes(rows, fc12=True) is kernels_sort


def test_mask_switch_documented_and_read_at_forward_time(monkeypatch):
    """TRTLLM_K3LL_MOE_SELF_SORT_MASK: off by default, documented, read from the flags module at call time."""
    flags = _mod("flags")
    _, default, doc = flags.flags()["TRTLLM_K3LL_MOE_SELF_SORT_MASK"]
    assert default == 0
    assert (
        "TRTLLM_K3LL_MOE_SELF_SORT_FC12" in doc
        and "K3MoeLowMDynKernel" in doc
        and "same metadata" in doc
    )
    lowm = _mod("decode_moe_lowm")
    for value, on in ((0, False), (1, True)):
        monkeypatch.setattr(flags, "TRTLLM_K3LL_MOE_SELF_SORT_MASK", value)
        assert lowm.self_sort_mask_enabled() is on

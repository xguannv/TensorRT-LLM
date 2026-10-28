# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_SANDWICH_ACT_EARLY`` (``kimi_k3_nvfp4/decode_comm.py``) on one process: the switch's default and doc,
its resolution, that the narrow tail passes it to ``comm/k3_sandwich_tail`` (counted once per engaged call) and that
the load compiles both builds, with stand-ins, no GPU needed. The kernel's bits (on = off in every form, graph
replays) are checked by ``cute_dsl_kernels/kimi_k3/test_k3_sandwich.py``'s ``act_early`` check.
"""

from types import SimpleNamespace

import pytest
import torch

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{_PKG}.{name}")


def test_switch_default_off_and_documented():
    flags = _mod("flags")
    _, default, doc = flags.flags()["TRTLLM_K3LL_SANDWICH_ACT_EARLY"]
    assert default == 0 and "Bitwise" in doc and "SANDWICH_ACT_EARLY_CAPABILITIES" in doc
    assert flags.accepted_values()["TRTLLM_K3LL_SANDWICH_ACT_EARLY"] == (0, 1)


def test_resolution(monkeypatch):
    """-1 resolves by ``SANDWICH_ACT_EARLY_CAPABILITIES`` (SM103 only); 1 / 0 force it either way."""
    dc = _mod("decode_comm")
    flags = _mod("flags")
    dev = torch.device("cuda", 0)
    for cap, want in (((10, 3), True), ((10, 0), False), ((10, 7), False)):
        dc._sandwich_act_early_by_default.cache_clear()
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None, c=cap: c)
        monkeypatch.setattr(flags, "TRTLLM_K3LL_SANDWICH_ACT_EARLY", -1)
        assert dc.sandwich_act_early_on(dev) is want, cap
        for forced in (0, 1):
            monkeypatch.setattr(flags, "TRTLLM_K3LL_SANDWICH_ACT_EARLY", forced)
            assert dc.sandwich_act_early_on(dev) is bool(forced)
    dc._sandwich_act_early_by_default.cache_clear()


@pytest.mark.parametrize("value", [0, 1])
def test_tail_passes_the_switch(monkeypatch, value):
    """``K3DecodeComm.sandwich_tail`` reads the switch at every call, passes it as ``act_early`` and counts the engaged
    calls in the census."""
    dc = _mod("decode_comm")
    flags = _mod("flags")
    census = _mod("census")
    monkeypatch.setattr(flags, "TRTLLM_K3LL_SANDWICH_ACT_EARLY", value)
    calls = []
    monkeypatch.setattr(
        dc, "k3_sandwich_tail", lambda *a, **kw: calls.append(kw["act_early"]) or (None, None)
    )
    counted = []
    monkeypatch.setattr(census, "count", lambda key: counted.append(key))
    comm = object.__new__(dc.K3DecodeComm)
    comm.sandwich = None
    pending = SimpleNamespace(latent=torch.zeros(8, 3584, dtype=torch.bfloat16), act=None, weight=None, lo=0,
                              lat_eps=1e-6)  # fmt: skip
    w = SimpleNamespace(weight=torch.ones(7168, dtype=torch.bfloat16), eps=1e-6)
    comm.sandwich_tail(pending, None, torch.zeros(0, 8, 7168, dtype=torch.bfloat16), w, w, w)
    assert calls == [bool(value)]
    assert counted == (["dispatch:sandwich_act_early:eager"] if value else [])


def test_compile_tail_compiles_both_builds(monkeypatch):
    """``compile_tail`` (before any capture) runs the kernel with and without ``act_early``, whatever the switch says
    (it is read at forward time)."""
    dc = _mod("decode_comm")
    flags = _mod("flags")
    calls = []
    monkeypatch.setattr(
        dc, "k3_sandwich_tail", lambda *a, **kw: calls.append(kw.get("act_early", False))
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **kw: None)
    comm = object.__new__(dc.K3DecodeComm)
    comm.sandwich = None
    for value in (0, 1):
        calls.clear()
        monkeypatch.setattr(flags, "TRTLLM_K3LL_SANDWICH_ACT_EARLY", value)
        comm.compile_tail(3584, 384, torch.zeros(7168, 640, dtype=torch.bfloat16))
        assert calls == [False, True], (value, calls)

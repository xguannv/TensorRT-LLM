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
"""``TRTLLM_K3_DEEPGEMM_SM_RESERVE`` on Kimi K3's FP8 weight-read linear (SM100/SM103).

The reserve shrinks the persistent DeepGEMM grid of the swap-AB path so it fits beside a
kernel that holds SMs (the DWDP fabric gather on CTX workers). It must not change the result,
must launch the smaller grid, and must leave DeepGEMM's process-wide SM count as it found it.
"""

import json

import pytest
import torch
from _torch.helpers import calc_diff

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch.models import modeling_kimi_linear as k3
from tensorrt_llm._torch.models.modeling_kimi_linear import (
    _Fp8BlockScaleWeightReadLinear as K3Fp8Linear,
)
from tensorrt_llm._utils import is_sm_100f

RESERVE = 30
MIN_K = 1024

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f() or k3._rubin_mxfp8_is_available(),
    reason="DeepGEMM swap-AB weight read runs on SM100/SM103",
)


@pytest.fixture
def sm_reserve(monkeypatch):
    """Set the module's reserve for one test; the per-device SM count cache follows it."""

    def set_reserve(value):
        monkeypatch.setattr(k3, "_K3_DEEPGEMM_SM_RESERVE", value)
        k3._k3_deep_gemm_num_sms.cache_clear()

    monkeypatch.setattr(k3, "_K3_DEEPGEMM_SM_RESERVE_MIN_K", MIN_K)

    # Every M below must reach DeepGEMM, not the opt-in low-M CuTe GEMM.
    monkeypatch.setattr(k3, "_K3_FP8W_LOWM_GEMM", False)

    yield set_reserve
    k3._k3_deep_gemm_num_sms.cache_clear()


def _device_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _make(out_features, in_features, seed=0):
    torch.random.manual_seed(seed)
    w = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / in_features**0.5
    )
    weight, weight_scale = K3Fp8Linear.quantize_weight(w)
    return w, K3Fp8Linear(weight, weight_scale, out_features)


def _kernel_grids(tmp_path, fn):
    """Grid x of every DeepGEMM kernel ``fn`` launches, from the profiler's trace."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    trace = tmp_path / "trace.json"
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    grids = [
        e["args"]["grid"][0]
        for e in events
        if e.get("cat") == "kernel" and "deep_gemm" in e.get("name", "")
    ]
    assert grids, "no DeepGEMM kernel was launched"
    return grids


@pytest.mark.parametrize("m", [256, 4096])
@pytest.mark.parametrize("out_features, in_features", [(3584, 7168), (7168, 1536)])
def test_reserve_keeps_output(sm_reserve, m, out_features, in_features):
    """Only the grid changes: each output tile accumulates the same K blocks in the same order."""
    w, mod = _make(out_features, in_features)
    x = torch.randn((m, in_features), device="cuda", dtype=torch.bfloat16)

    sm_reserve(0)
    full = mod(x)
    sm_reserve(RESERVE)
    reserved = mod(x)

    assert torch.equal(reserved, full)
    expected = (x.float() @ w.float().t()).to(x.dtype)
    assert calc_diff(reserved, expected) < 5e-3


def test_reserve_launches_smaller_grid(sm_reserve, tmp_path):
    """The persistent grid is the device's SMs minus the reserve, and only with the reserve set."""
    _, mod = _make(12288, 7168, seed=1)
    x = torch.randn((8192, 7168), device="cuda", dtype=torch.bfloat16)
    sms = _device_sms()

    sm_reserve(0)
    mod(x)
    assert set(_kernel_grids(tmp_path, lambda: mod(x))) == {sms}

    sm_reserve(RESERVE)
    mod(x)
    assert set(_kernel_grids(tmp_path, lambda: mod(x))) == {sms - RESERVE}


def _num_sms_seen_by_gemm(mod, x, set_reserve, reserves):
    """DeepGEMM's SM count inside the GEMM, one ``mod(x)`` call per reserve in ``reserves``."""
    seen = []
    real_gemm = torch.ops.trtllm.fp8_swap_ab_gemm

    class _GemmSpy:
        def __call__(self, *args, **kwargs):
            seen.append(deep_gemm.get_num_sms())
            return real_gemm(*args, **kwargs)

    torch.ops.trtllm.fp8_swap_ab_gemm = _GemmSpy()
    try:
        for reserve in reserves:
            set_reserve(reserve)
            mod(x)
    finally:
        torch.ops.trtllm.fp8_swap_ab_gemm = real_gemm
    return seen


def test_reserve_is_scoped_to_the_call(sm_reserve):
    """DeepGEMM sees the reduced SM count inside the GEMM and its own count again after it."""
    _, mod = _make(2048, 1024, seed=2)
    x = torch.randn((512, 1024), device="cuda", dtype=torch.bfloat16)
    before = deep_gemm.get_num_sms()

    seen = _num_sms_seen_by_gemm(mod, x, sm_reserve, [0, RESERVE])

    assert seen == [before, _device_sms() - RESERVE]
    assert deep_gemm.get_num_sms() == before


def test_reserve_skips_low_k(sm_reserve):
    """Projections below the K threshold (MLA kv_b, K = 512) keep DeepGEMM's full grid."""
    _, mod = _make(2048, 512, seed=4)
    x = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    before = deep_gemm.get_num_sms()

    seen = _num_sms_seen_by_gemm(mod, x, sm_reserve, [RESERVE])

    assert seen == [before]


def test_reserve_restored_when_gemm_raises(sm_reserve):
    """A failing GEMM must not leave the reduced SM count behind."""
    _, mod = _make(2048, 1024, seed=3)
    x = torch.randn((64, 1024), device="cuda", dtype=torch.bfloat16)
    real_gemm = torch.ops.trtllm.fp8_swap_ab_gemm

    def _fail(*args, **kwargs):
        raise RuntimeError("injected")

    before = deep_gemm.get_num_sms()
    sm_reserve(RESERVE)
    torch.ops.trtllm.fp8_swap_ab_gemm = _fail
    try:
        with pytest.raises(RuntimeError, match="injected"):
            mod(x)
    finally:
        torch.ops.trtllm.fp8_swap_ab_gemm = real_gemm
    assert deep_gemm.get_num_sms() == before


def test_reserve_must_leave_an_sm(sm_reserve):
    sm_reserve(_device_sms())
    with pytest.raises(ValueError, match="TRTLLM_K3_DEEPGEMM_SM_RESERVE"):
        k3._k3_deep_gemm_num_sms(torch.cuda.current_device())

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
"""TRTLLM_K3LL_DEEPGEMM_SM_RESERVE: the DeepGEMM swap-AB FP8 GEMM runs on fewer SMs only while the
variable is set and K is large enough, and DeepGEMM's SM count is restored after every call."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco


class _FakeDeepGemm:
    def __init__(self, sms: int, fail: bool = False):
        self.sms = sms
        self.fail = fail
        self.calls = []

    def get_num_sms(self):
        return self.sms

    def set_num_sms(self, n):
        self.sms = n

    def fp8_gemm_nt(self, a, b, out, disable_ue8m0_cast):
        self.calls.append(self.sms)
        if self.fail:
            raise RuntimeError("boom")


@pytest.fixture
def fake(monkeypatch):
    dg = _FakeDeepGemm(148)
    monkeypatch.setattr(tco, "deep_gemm", dg)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _dev: SimpleNamespace(multi_processor_count=148)
    )
    return dg


def _run(k):
    tco._deep_gemm_fp8_gemm_nt(None, None, torch.empty(1), k=k, disable_ue8m0_cast=True)


def test_off_by_default(monkeypatch, fake):
    monkeypatch.delenv("TRTLLM_K3LL_DEEPGEMM_SM_RESERVE", raising=False)
    monkeypatch.setenv("TRTLLM_K3_DEEPGEMM_SM_RESERVE", "16")  # Dongxu's name does not turn it on
    _run(7168)
    assert fake.calls == [148] and fake.sms == 148


def test_reserve_applies_and_restores(monkeypatch, fake):
    monkeypatch.setenv("TRTLLM_K3LL_DEEPGEMM_SM_RESERVE", "16")
    _run(7168)
    _run(512)  # below the default min K of 1024: full grid
    assert fake.calls == [132, 148] and fake.sms == 148


def test_restores_after_an_error(monkeypatch, fake):
    monkeypatch.setenv("TRTLLM_K3LL_DEEPGEMM_SM_RESERVE", "16")
    fake.fail = True
    with pytest.raises(RuntimeError):
        _run(7168)
    assert fake.sms == 148


def test_reserve_must_leave_sms(monkeypatch, fake):
    monkeypatch.setenv("TRTLLM_K3LL_DEEPGEMM_SM_RESERVE", "148")
    with pytest.raises(ValueError):
        _run(7168)

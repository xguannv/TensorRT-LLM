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
"""Tests for the PyTorch memory cap set after the KV cache pool: cap = free + reserved - margin."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import _util

GIB = 1024**3
TOTAL = 256 * GIB
FREE = 30 * GIB
RESERVED = 70 * GIB


@pytest.fixture
def fake_cuda(monkeypatch):
    fractions = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda f, *a, **k: fractions.append(f)
    )
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: RESERVED)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (FREE, TOTAL))
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return fractions


def _cap_bytes(fractions):
    # The first call resets the fraction to 1.0; the last one is the cap.
    assert fractions[0] == 1.0
    return fractions[-1] * TOTAL


def test_margin_unset_keeps_32_mib(fake_cuda, monkeypatch):
    monkeypatch.delenv(_util.TORCH_MEM_SAFETY_MARGIN_GIB_ENV, raising=False)
    _util._adjust_torch_mem_fraction()
    assert _cap_bytes(fake_cuda) == pytest.approx(FREE + RESERVED - 32 * 1024**2, abs=1)


def test_margin_empty_keeps_32_mib(fake_cuda, monkeypatch):
    monkeypatch.setenv(_util.TORCH_MEM_SAFETY_MARGIN_GIB_ENV, "")
    _util._adjust_torch_mem_fraction()
    assert _cap_bytes(fake_cuda) == pytest.approx(FREE + RESERVED - 32 * 1024**2, abs=1)


@pytest.mark.parametrize("gib", ["2", "0.5", "0"])
def test_margin_set(fake_cuda, monkeypatch, gib):
    monkeypatch.setenv(_util.TORCH_MEM_SAFETY_MARGIN_GIB_ENV, gib)
    _util._adjust_torch_mem_fraction()
    assert _cap_bytes(fake_cuda) == pytest.approx(FREE + RESERVED - float(gib) * GIB, abs=1)


@pytest.mark.parametrize("bad", ["-1", "two", "nan"])
def test_margin_invalid(fake_cuda, monkeypatch, bad):
    monkeypatch.setenv(_util.TORCH_MEM_SAFETY_MARGIN_GIB_ENV, bad)
    with pytest.raises(ValueError):
        _util._adjust_torch_mem_fraction()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CuTe DSL FP8 MLA decode, balanced split, small-batch kernel variant.

Up to 32 requests the runner compiles the balanced-split kernel with a
register schedule (``balanced_max_requests=32``) and a reduction that runs one
CTA group per M tile, optionally split by columns. The result must be BITWISE
identical to the general variant (``balanced_max_requests=1024``, one reduction
CTA group per partial slot) and must not depend on the column split. The
opt-in request-aligned split (TRTLLM_MLA_ALIGNED_SPLIT) and BF16 partials
(TRTLLM_MLA_PARTIAL_BF16) move the split-KV rounding only: they are checked
against the fp32 reference and, where nothing changes, bitwise.

Requires 1 SM100/SM103 GPU and nvidia-cutlass-dsl.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cute_dsl_mla_flat_fold import (  # noqa: E402
    _HAVE_DEPS,
    _is_sm100_or_sm103,
    _padded_inputs,
    _reference_bounded,
    _run_guarded,
)

if _HAVE_DEPS:
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (  # noqa: E402
        CuteDSLNVMlaDecodeBlackwellRunner,
    )

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_sm100_or_sm103(), reason="needs sm100/sm103"),
    pytest.mark.skipif(not _HAVE_DEPS, reason="needs nvidia-cutlass-dsl"),
]

_BALANCED = 0  # CuteDSLNVMlaDecodeBlackwellRunner.BALANCED_SPLIT_TACTIC

_SMALL_LENS = {
    # K3 Helix CP16 gen-only shapes (uniform per-rank KV)
    "b1": [7300],
    "b4": [7300] * 4,
    "b8": [7300] * 8,
    "b16": [7300] * 16,
    # mixed lengths with tile boundaries, very short requests and no KV here
    "mixed": [3000, 70, 1500, 8191, 64, 513, 0, 2048, 4500, 9],
    # skewed per-rank KV: one long request among short ones
    "skew": [600, 0, 61000, 2048, 9000, 130, 15000, 7300, 64],
    # the largest batch of the small variant
    "b32": [700 + 263 * i for i in range(32)],
}


def _run(x, H, S, monkeypatch, env=None, small=True):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if not small:
        monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SMALL_BATCH", 0)
    try:
        return _run_guarded(x, H, S, _BALANCED, True, monkeypatch, persistent=True)
    finally:
        for k in env or {}:
            monkeypatch.delenv(k)
        monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SMALL_BATCH", 32)


def _assert_same(a, b):
    for t, u in zip(a, b):
        if t is None:
            assert u is None
        else:
            assert torch.equal(t, u)


def _assert_close_to_reference(x, o, S, H):
    gold, valid = _reference_bounded(x, S, H)
    v = valid.to("cuda")
    err = (o.float() - gold).abs()[v].max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


@pytest.mark.parametrize("lens_name", list(_SMALL_LENS))
@pytest.mark.parametrize("H,S,helix", [(96, 7, True), (96, 8, True), (96, 8, False)])
def test_small_variant_is_bitwise_identical(H, S, helix, lens_name, monkeypatch):
    x = _padded_inputs(_SMALL_LENS[lens_name], S, H, helix=helix, seed=21)
    general = _run(x, H, S, monkeypatch, small=False)
    small = _run(x, H, S, monkeypatch)
    _assert_same(small, general)
    _assert_close_to_reference(x, small[0], S, H)


@pytest.mark.parametrize("lens_name", ["b1", "b4", "mixed", "skew"])
@pytest.mark.parametrize("col_split", ["1", "2", "4"])
def test_reduction_col_split_is_bitwise_identical(col_split, lens_name, monkeypatch):
    H, S = 96, 7
    x = _padded_inputs(_SMALL_LENS[lens_name], S, H, helix=True, seed=22)
    ref = _run(x, H, S, monkeypatch, small=False)
    new = _run(x, H, S, monkeypatch, env={"TRTLLM_MLA_REDUCTION_COL_SPLIT": col_split})
    _assert_same(new, ref)


@pytest.mark.parametrize("lens_name", list(_SMALL_LENS))
@pytest.mark.parametrize("H,S", [(96, 7), (96, 8)])
def test_aligned_split(H, S, lens_name, monkeypatch):
    lens = _SMALL_LENS[lens_name]
    x = _padded_inputs(lens, S, H, helix=True, seed=23)
    even = _run(x, H, S, monkeypatch)
    aligned = _run(x, H, S, monkeypatch, env={"TRTLLM_MLA_ALIGNED_SPLIT": "4"})
    _assert_close_to_reference(x, aligned[0], S, H)
    # only the split-KV rounding may change
    diff = (aligned[0].float() - even[0].float()).abs().max().item()
    assert diff <= 0.05 * even[0].float().abs().max().item(), diff
    for b, L in enumerate(lens):
        if L <= 0:
            assert (aligned[0][b] == 0).all()
    if len(lens) > 12:
        # 12 lanes per chunk cannot be aligned to more requests
        _assert_same(aligned, even)


@pytest.mark.parametrize("lens_name", ["b1", "b4", "b16", "mixed", "skew", "b32"])
@pytest.mark.parametrize("small", [True, False])
def test_bf16_partials(small, lens_name, monkeypatch):
    H, S = 96, 8
    lens = _SMALL_LENS[lens_name]
    x = _padded_inputs(lens, S, H, helix=True, seed=24)
    fp32 = _run(x, H, S, monkeypatch, small=small)
    bf16 = _run(x, H, S, monkeypatch, small=small, env={"TRTLLM_MLA_PARTIAL_BF16": "1"})
    _assert_close_to_reference(x, bf16[0], S, H)
    # a BF16 rounding of each partial: within a few BF16 steps of the output
    diff = (bf16[0].float() - fp32[0].float()).abs()
    scale = fp32[0].float().abs().clamp_min(2.0**-10)
    assert (diff <= 2.0**-6 * scale + 2.0**-12).all(), (diff / scale).max().item()
    # LSE and softmax stats come from the FP32 partial LSEs
    assert torch.equal(bf16[1], fp32[1])
    assert torch.equal(bf16[2], fp32[2])

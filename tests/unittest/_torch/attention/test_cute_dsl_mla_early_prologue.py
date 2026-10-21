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
"""CuTe DSL FP8 MLA decode, balanced split: the early prologue (TRTLLM_MLA_EARLY_PROLOGUE).

With PDL the kernel may build its schedule and load the first K tile of each cluster's first
segment before ``griddepcontrol.wait`` (only a tile that holds no KV this step can append), then
loads the query first. That only reorders loads: O, LSE, the Helix softmax statistics and the
fused V up-projection must be BITWISE identical to the default kernel, for batches 1 to 33,
short KV (where a cluster's first tile is among the last ones of its request), a request
without KV, eagerly and in a CUDA graph; without PDL the option changes nothing.

Requires 1 SM100/SM103 GPU and nvidia-cutlass-dsl.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_cute_dsl_mla_flat_fold as flat_fold  # noqa: E402
import test_cute_dsl_mla_fused_v_up as fused_v_up  # noqa: E402
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

# (H, S, per-rank KV lengths)
_CASES = {
    "b1_long": (96, 7, [10247]),
    "b2_long": (96, 7, [10247, 10247]),
    "b2_skew": (96, 7, [10247, 3000]),
    "b4_mtp7": (96, 8, [4096, 8192, 300, 5000]),
    # every cluster's first tile is one of the last tiles of its request
    "b2_short": (96, 7, [200, 130]),
    "b3_no_kv": (96, 7, [5000, 0, 900]),
    "b33_ragged": (96, 7, [3000, 70, 1500, 8191, 64, 513, 0, 2048, 4500, 9, 600] * 3),
    "b8_s1": (128, 1, [2000, 64, 1, 900, 5000, 128, 129, 3000]),
}


def _run(x, H, S, monkeypatch, early, pdl="1"):
    monkeypatch.setenv("TRTLLM_MLA_PDL", pdl)
    monkeypatch.setenv("TRTLLM_MLA_EARLY_PROLOGUE", "1" if early else "0")
    return _run_guarded(x, H, S, _BALANCED, True, monkeypatch, persistent=True)


def _assert_same(a, b):
    for t, u in zip(a, b):
        if t is None:
            assert u is None
        else:
            assert torch.equal(t, u)


def test_option(monkeypatch):
    monkeypatch.delenv("TRTLLM_MLA_EARLY_PROLOGUE", raising=False)
    opts = dict(CuteDSLNVMlaDecodeBlackwellRunner._fp8_kernel_options(2, True))
    assert opts["early_prologue"] is False
    monkeypatch.setenv("TRTLLM_MLA_EARLY_PROLOGUE", "1")
    opts = dict(CuteDSLNVMlaDecodeBlackwellRunner._fp8_kernel_options(2, True))
    assert opts["early_prologue"] is True
    # the uniform split has no early prologue
    assert "early_prologue" not in dict(
        CuteDSLNVMlaDecodeBlackwellRunner._fp8_kernel_options(2, False)
    )


@pytest.mark.parametrize("helix", [True, False], ids=["helix", "plain"])
@pytest.mark.parametrize("case", list(_CASES))
def test_early_prologue_is_bitwise_identical(case, helix, monkeypatch):
    H, S, lens = _CASES[case]
    if helix and S == 1:
        pytest.skip("the Helix bounds need S > 1")
    monkeypatch.setattr(flat_fold, "_MAX_BATCH", 64)
    x = _padded_inputs(lens, S, H, helix=helix, seed=61)
    ref = _run(x, H, S, monkeypatch, early=False)
    gold, valid = _reference_bounded(x, S, H) if helix else (None, None)
    if gold is not None:
        v = valid.to("cuda")
        err = (ref[0].float() - gold).abs()[v].max().item()
        assert err <= 0.05 * gold.abs().max().item(), err
    _assert_same(_run(x, H, S, monkeypatch, early=True), ref)


@pytest.mark.parametrize("case", ["b2_long", "b2_short"])
def test_without_pdl_the_option_is_inert(case, monkeypatch):
    H, S, lens = _CASES[case]
    monkeypatch.setattr(flat_fold, "_MAX_BATCH", 64)
    x = _padded_inputs(lens, S, H, helix=True, seed=62)
    ref = _run(x, H, S, monkeypatch, early=False, pdl="0")
    _assert_same(_run(x, H, S, monkeypatch, early=True, pdl="0"), ref)
    _assert_same(_run(x, H, S, monkeypatch, early=True, pdl="1"), ref)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("case", ["b1_long", "b2_long", "b4_mtp7", "b2_short", "b3_no_kv"])
def test_fused_v_up_is_bitwise_identical(case, graph, monkeypatch):
    """The Helix decode with the V up-projection fused into its reduction (the K3 GEN path)."""
    monkeypatch.setattr(flat_fold, "_MAX_BATCH", fused_v_up.MAX_BATCH)
    monkeypatch.setenv("TRTLLM_MLA_PARTIAL_BF16", "1")
    monkeypatch.setenv("TRTLLM_MLA_RESCALE_THRESHOLD", "8")
    monkeypatch.setenv("TRTLLM_MLA_PDL", "1")
    H, S, lens = _CASES[case]
    x = _padded_inputs(lens, S, H, True, 63)
    w = (torch.randn(H, 128, 512, device="cuda") * 0.05).to(torch.bfloat16)
    monkeypatch.setenv("TRTLLM_MLA_EARLY_PROLOGUE", "0")
    _, stats_ref, v_ref = fused_v_up._run(x, S, w, fused=True, graph=graph)
    monkeypatch.setenv("TRTLLM_MLA_EARLY_PROLOGUE", "1")
    _, stats, v = fused_v_up._run(x, S, w, fused=True, graph=graph)
    assert not torch.isnan(v_ref).any()
    assert torch.equal(v, v_ref)
    assert torch.equal(torch.isnan(stats), torch.isnan(stats_ref))
    assert torch.equal(torch.nan_to_num(stats), torch.nan_to_num(stats_ref))

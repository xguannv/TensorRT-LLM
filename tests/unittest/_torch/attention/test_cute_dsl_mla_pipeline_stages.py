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
"""CuTe DSL FP8 MLA decode, balanced split: pipeline depths, schedule size and
reduction variant.

The balanced-split kernel runs 3 K-tile stages and 2 P stages while its
shared-memory schedule holds at most 256 requests (2 and 3 past that), sizes
the schedule to the batch rounded up to a power of two, and above 24 requests
reduces per partial slot instead of per M tile. None of this may change the
result: every stage combination that fits in shared memory, every schedule
size and both reduction variants must give BITWISE identical outputs. A stage
combination that does not fit is rejected when the kernel is compiled.

Requires 1 SM100/SM103 GPU and nvidia-cutlass-dsl.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_cute_dsl_mla_flat_fold as flat_fold  # noqa: E402
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
_MAX_BATCH = 256

# (H, S, per-rank KV lengths); short KV keeps the tests fast while every
# cluster still runs several segments.
_CASES = {
    "b16_s8": (96, 8, [1800] * 16),
    "b28_s8": (96, 8, [2000] * 28),
    "b32_s8_mixed": (96, 8, [300 + 97 * i for i in range(32)]),
    "b33_s7": (96, 7, [3000, 70, 1500, 8191, 64, 513, 0, 2048, 4500, 9, 600] * 3),
    "b64_s6": (96, 6, [1200] * 64),
    "b130_s7_ragged": (96, 7, [64 + (i * 977) % 2900 for i in range(130)]),
}


def _run(x, H, S, monkeypatch, env=None):
    monkeypatch.setattr(flat_fold, "_MAX_BATCH", _MAX_BATCH)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    try:
        return _run_guarded(x, H, S, _BALANCED, True, monkeypatch, persistent=True)
    finally:
        for k in env or {}:
            monkeypatch.delenv(k)


def _assert_same(a, b):
    for t, u in zip(a, b):
        if t is None:
            assert u is None
        else:
            assert torch.equal(t, u)


def _options(batch_size):
    return dict(CuteDSLNVMlaDecodeBlackwellRunner._fp8_kernel_options(batch_size, True))


@pytest.mark.parametrize(
    "batch_size,max_requests,stages",
    [
        (1, 32, (3, 2)),
        (24, 32, (3, 2)),
        (25, 32, (3, 2)),
        (32, 32, (3, 2)),
        (33, 64, (3, 2)),
        (64, 64, (3, 2)),
        (65, 128, (3, 2)),
        (200, 256, (3, 2)),
        (256, 256, (3, 2)),
        (257, 512, (2, 3)),
        (1024, 1024, (2, 3)),
    ],
)
def test_default_pipeline_options(batch_size, max_requests, stages, monkeypatch):
    monkeypatch.delenv("TRTLLM_MLA_K_STAGES", raising=False)
    monkeypatch.delenv("TRTLLM_MLA_P_STAGES", raising=False)
    monkeypatch.delenv("TRTLLM_MLA_TILE_REDUCTION", raising=False)
    opts = _options(batch_size)
    assert opts["balanced_max_requests"] == max_requests
    assert (opts["k_stages"], opts["p_stages"]) == stages
    assert opts["tile_reduction"] == (batch_size <= 24)
    monkeypatch.setenv("TRTLLM_MLA_K_STAGES", "2")
    monkeypatch.setenv("TRTLLM_MLA_P_STAGES", "3")
    opts = _options(batch_size)
    assert (opts["k_stages"], opts["p_stages"]) == (2, 3)


@pytest.mark.parametrize("case", list(_CASES))
def test_pipeline_stages_are_bitwise_identical(case, monkeypatch):
    H, S, lens = _CASES[case]
    x = _padded_inputs(lens, S, H, helix=True, seed=41)
    ref = _run(x, H, S, monkeypatch, env={"TRTLLM_MLA_K_STAGES": "2", "TRTLLM_MLA_P_STAGES": "3"})
    gold, valid = _reference_bounded(x, S, H)
    v = valid.to("cuda")
    err = (ref[0].float() - gold).abs()[v].max().item()
    assert err <= 0.05 * gold.abs().max().item(), err
    for k, p in ((3, 2), (2, 2)):
        new = _run(
            x, H, S, monkeypatch, env={"TRTLLM_MLA_K_STAGES": str(k), "TRTLLM_MLA_P_STAGES": str(p)}
        )
        _assert_same(new, ref)
    # the defaults
    _assert_same(_run(x, H, S, monkeypatch), ref)


@pytest.mark.parametrize("case", ["b33_s7", "b130_s7_ragged"])
def test_schedule_size_is_bitwise_identical(case, monkeypatch):
    """The batch-sized schedule matches the 1024-request schedule."""
    H, S, lens = _CASES[case]
    x = _padded_inputs(lens, S, H, helix=True, seed=42)
    sized = _run(x, H, S, monkeypatch)
    orig = CuteDSLNVMlaDecodeBlackwellRunner._fp8_kernel_options.__func__

    def full_table(cls, batch_size, balanced_split):
        opts = dict(orig(cls, batch_size, balanced_split))
        opts.update(balanced_max_requests=1024, k_stages=2, p_stages=3)
        return tuple(opts.items())

    monkeypatch.setattr(
        CuteDSLNVMlaDecodeBlackwellRunner, "_fp8_kernel_options", classmethod(full_table)
    )
    _assert_same(_run(x, H, S, monkeypatch), sized)


@pytest.mark.parametrize("case", ["b16_s8", "b28_s8", "b32_s8_mixed"])
def test_reduction_variants_are_bitwise_identical(case, monkeypatch):
    """Tile- and slot-indexed reductions of the register-schedule batches."""
    H, S, lens = _CASES[case]
    x = _padded_inputs(lens, S, H, helix=True, seed=44)
    tile = _run(x, H, S, monkeypatch, env={"TRTLLM_MLA_TILE_REDUCTION": "1"})
    slot = _run(x, H, S, monkeypatch, env={"TRTLLM_MLA_TILE_REDUCTION": "0"})
    _assert_same(slot, tile)


def test_stages_that_do_not_fit_are_rejected(monkeypatch):
    H, S, lens = _CASES["b64_s6"]
    x = _padded_inputs(lens, S, H, helix=True, seed=43)
    with pytest.raises(ValueError, match="shared storage"):
        _run(x, H, S, monkeypatch, env={"TRTLLM_MLA_K_STAGES": "3", "TRTLLM_MLA_P_STAGES": "3"})

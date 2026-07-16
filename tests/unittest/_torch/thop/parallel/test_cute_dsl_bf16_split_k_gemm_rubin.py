# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-K correctness tests for the Rubin (SM107) BF16 dense GEMM runner.

Exercises the ``CuteDSLBf16RubinGemmRunner`` split-K path end to end: the GEMM
writes FP32 partials into a workspace and a reduction kernel sums them into the
output.  The reference is ``act @ weight.T``.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = [
    pytest.mark.skipif(
        get_sm_version() != 107,
        reason="This test is only supported on Rubin (SM 107) GPUs",
    ),
    pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available"),
    pytest.mark.skipif(
        not IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
        reason="Rubin CuTe DSL internal package is not available",
    ),
]


def _select_split_k_tactic(tactics, split_k_slices):
    """Pick a base tactic with the requested split count and 1-CTA MMA."""
    candidates = [
        t for t in tactics if t[0] == "base" and t[1] is False and t[-1] == split_k_slices
    ]
    assert candidates, f"no base tactic with split_k_slices={split_k_slices} found"
    return candidates[0]


@pytest.mark.parametrize("split_k_slices", [2, 4, 8])
@pytest.mark.parametrize("c_dtype", [torch.bfloat16, torch.float32])
def test_cute_dsl_bf16_split_k_gemm_rubin(split_k_slices, c_dtype):
    """Split-K GEMM matches the dense reference for large-K, small-N shapes."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    torch.manual_seed(2026)
    runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(use_tvm_ffi=True)
    runner.__class__.kernel_cache.clear()
    runner.__class__.split_k_gemm_cache.clear()
    runner.__class__.split_k_reduction_cache.clear()

    # Large K and small N so get_valid_tactics offers split>1 candidates.
    m, n, k = 256, 256, 8192
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(m, n, dtype=c_dtype, device="cuda")

    tactics = runner.get_valid_tactics([act, weight, output], None)
    tactic = _select_split_k_tactic(tactics, split_k_slices)

    expected = act.float() @ weight.t().float()
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)


@pytest.mark.parametrize("split_k_slices", [2, 4])
def test_cute_dsl_bf16_split_k_matches_split1_rubin(split_k_slices):
    """Split-K output matches the non-split (split=1) output bit-for-bit-ish."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    torch.manual_seed(7)
    runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(use_tvm_ffi=True)
    runner.__class__.kernel_cache.clear()
    runner.__class__.split_k_gemm_cache.clear()
    runner.__class__.split_k_reduction_cache.clear()

    m, n, k = 256, 256, 8192
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    tactics = runner.get_valid_tactics(
        [act, weight, torch.empty(m, n, dtype=torch.bfloat16, device="cuda")], None
    )

    out_split1 = torch.empty(m, n, dtype=torch.float32, device="cuda")
    runner([act, weight, out_split1], tactic=_select_split_k_tactic(tactics, 1))

    out_splitk = torch.empty(m, n, dtype=torch.float32, device="cuda")
    runner([act, weight, out_splitk], tactic=_select_split_k_tactic(tactics, split_k_slices))

    torch.cuda.synchronize()
    # Both accumulate in FP32; only the K reduction order differs.
    torch.testing.assert_close(out_splitk, out_split1, rtol=1e-3, atol=1e-2)


def test_cute_dsl_bf16_split_k_ugpu_rubin(monkeypatch):
    """Split-K respects the uGPU half-GEMM sliced output."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    torch.manual_seed(99)
    runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(use_tvm_ffi=True)
    runner.__class__.kernel_cache.clear()
    runner.__class__.split_k_gemm_cache.clear()
    runner.__class__.split_k_reduction_cache.clear()

    split_k_slices = 4
    m, n, k = 256, 256, 8192
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    expected = act.float() @ weight.t().float()

    tactics = runner.get_valid_tactics(
        [act, weight, torch.empty(m, n, dtype=torch.bfloat16, device="cuda")], None
    )
    tactic = _select_split_k_tactic(tactics, split_k_slices)

    wide_output = torch.empty(m, n * 2, dtype=torch.bfloat16, device="cuda")
    for ugpu_id in range(2):
        wide_output.fill_(float("nan"))
        monkeypatch.setattr(cute_dsl_custom_ops, "get_current_ugpu", lambda: ugpu_id)
        runner([act, weight, wide_output], tactic=tactic)
        torch.cuda.synchronize()

        mine = wide_output[:, ugpu_id * n : (ugpu_id + 1) * n]
        other = wide_output[:, (1 - ugpu_id) * n : (2 - ugpu_id) * n]
        torch.testing.assert_close(mine.float(), expected, rtol=1e-2, atol=1.0)
        assert torch.isnan(other).all()

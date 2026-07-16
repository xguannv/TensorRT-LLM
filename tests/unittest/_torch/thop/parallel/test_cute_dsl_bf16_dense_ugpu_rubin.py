# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
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


def _select_tactic(tactics, kernel_variant):
    tactic_name = "base"
    if kernel_variant == "preferred_cluster":
        tactic_name = "preferred_cluster"
    candidates = [tactic for tactic in tactics if tactic[0] == tactic_name and tactic[1] is False]
    assert candidates, f"no {kernel_variant} tactic found"
    return candidates[0]


def _run_ugpu_partition(
    monkeypatch,
    cute_dsl_custom_ops,
    run_partition,
    output,
    expected,
    partition_dim,
    partition_size,
    ugpu_id,
):
    output.fill_(float("nan"))
    monkeypatch.setattr(cute_dsl_custom_ops, "get_current_ugpu", lambda: ugpu_id)
    run_partition()
    torch.cuda.synchronize()

    slices = [slice(None)] * output.dim()
    slices[partition_dim] = slice(ugpu_id * partition_size, (ugpu_id + 1) * partition_size)
    output_slice = output[tuple(slices)]

    other_slices = [slice(None)] * output.dim()
    other_ugpu_id = 1 - ugpu_id
    other_slices[partition_dim] = slice(
        other_ugpu_id * partition_size, (other_ugpu_id + 1) * partition_size
    )
    other_slice = output[tuple(other_slices)]

    torch.testing.assert_close(output_slice.float(), expected.float(), rtol=1e-2, atol=1.0)
    assert torch.isnan(other_slice).all()


def _clear_bf16_bmm_state(cute_dsl_custom_ops):
    AutoTuner.get().clear_cache()
    cute_dsl_custom_ops.CuteDSLBf16RubinBmmRunner.kernel_cache.clear()


@pytest.mark.parametrize(
    ("kernel_variant", "mnk"),
    [("base", (128, 128, 128)), ("preferred_cluster", (256, 256, 128))],
)
def test_cute_dsl_bf16_gemm_ugpu_rubin(monkeypatch, kernel_variant, mnk):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    torch.manual_seed(123)
    runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(use_tvm_ffi=True)
    runner.__class__.kernel_cache.clear()

    m, n, k = mnk
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    tactics = runner.get_valid_tactics([act, weight, output], None)
    tactic = _select_tactic(tactics, kernel_variant)

    expected = act.float() @ weight.t().float()
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)

    wide_output = torch.empty(m, n * 2, dtype=torch.bfloat16, device="cuda")
    for ugpu_id in range(2):
        _run_ugpu_partition(
            monkeypatch,
            cute_dsl_custom_ops,
            lambda: runner([act, weight, wide_output], tactic=tactic),
            wide_output,
            expected,
            partition_dim=1,
            partition_size=n,
            ugpu_id=ugpu_id,
        )


@pytest.mark.parametrize(
    ("kernel_variant", "bm_nk"),
    [("base", (2, 64, 128, 128)), ("preferred_cluster", (2, 256, 256, 128))],
)
def test_cute_dsl_bf16_bmm_ugpu_rubin(monkeypatch, kernel_variant, bm_nk):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    torch.manual_seed(456)
    runner = cute_dsl_custom_ops.CuteDSLBf16RubinBmmRunner(use_tvm_ffi=True)
    runner.__class__.kernel_cache.clear()

    batch_size, m, n, k = bm_nk
    act = torch.randn(batch_size, m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(batch_size, n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(batch_size, m, n, dtype=torch.bfloat16, device="cuda")
    tactics = runner.get_valid_tactics([act, weight, output], None)
    tactic = _select_tactic(tactics, kernel_variant)

    expected = torch.bmm(act.float(), weight.transpose(1, 2).float())
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)

    wide_output = torch.empty(batch_size, m, n * 2, dtype=torch.bfloat16, device="cuda")
    for ugpu_id in range(2):
        _run_ugpu_partition(
            monkeypatch,
            cute_dsl_custom_ops,
            lambda: runner([act, weight, wide_output], tactic=tactic),
            wide_output,
            expected,
            partition_dim=2,
            partition_size=n,
            ugpu_id=ugpu_id,
        )


def test_cute_dsl_bf16_bmm_autotune_all_tactics_rubin():
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    torch.manual_seed(789)
    _clear_bf16_bmm_state(cute_dsl_custom_ops)

    batch_size, m, n, k = 1, 256, 256, 128
    act = torch.randn(batch_size, m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(batch_size, n, k, dtype=torch.bfloat16, device="cuda")
    expected = torch.bmm(act.float(), weight.transpose(1, 2).float())

    output = torch.empty(batch_size, m, n, dtype=torch.bfloat16, device="cuda")
    with autotune(skip_dynamic_tuning_buckets=True):
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)

    runner = cute_dsl_custom_ops.CuteDSLBf16RubinBmmRunner(use_tvm_ffi=True)
    expected_tactics = runner.get_valid_tactics([act, weight, output], None)
    assert any(t[0] == "base" and t[1] is True for t in expected_tactics)
    assert any(t[0] == "preferred_cluster" and t[1] is True for t in expected_tactics)

    tuner = AutoTuner.get()
    with tuner.capture() as all_tactics:
        captured_output = torch.empty_like(output)
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, captured_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output.float(), expected, rtol=1e-2, atol=1.0)

    tested_tactics = []
    for ((captured_runner, tactic),) in all_tactics:
        replay_output = torch.empty_like(output)
        tested_tactics.append(tactic)
        with tuner.replay(((captured_runner, tactic),)):
            torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, replay_output)
        torch.cuda.synchronize()
        torch.testing.assert_close(replay_output.float(), expected, rtol=1e-2, atol=1.0)

    assert tested_tactics == expected_tactics


@pytest.mark.parametrize("mnk", [(16462, 2112, 7168), (16384, 24576, 1536)])
def test_cute_dsl_bf16_gemm_preferred_cluster_full_coverage_rubin(mnk):
    """Multi-wave full-output-coverage regression for the preferred-cluster GEMM.

    The Rubin preferred-cluster mega-kernel runs a flexible cluster launch:
    the hardware downgrades some preferred (4,2)=8-CTA clusters to the
    fallback (2,1) shape (always happens on 216-SM parts due to GPC
    fragmentation). Both cluster populations MUST schedule over the single
    preferred tile partition; if the fallback-shaped clusters use a different
    partition, whole cluster work-items are never visited and the
    corresponding output tiles keep stale memory -- a silent accuracy bug.

    Small single-wave shapes (one persistent wave covers every tile) never
    expose this. Here M is large enough to force many waves; we NaN-poison the
    output so any unvisited tile shows up as NaN rows, independent of error
    tolerance, and require every preferred_cluster tactic to write the full
    output and match the reference.
    """
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    m, n, k = mnk
    torch.manual_seed(41)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(use_tvm_ffi=True)
    tactics = runner.get_valid_tactics([a, w, c], None)
    pc = [t for t in tactics if isinstance(t, tuple) and t[0] == "preferred_cluster"]
    assert pc, "no preferred_cluster tactic offered at this multi-wave shape"

    for tactic in pc:
        c.fill_(float("nan"))
        runner([a, w, c], tactic=tactic)
        torch.cuda.synchronize()
        nan_rows = int(torch.isnan(c).any(dim=1).sum().item())
        assert nan_rows == 0, f"tactic {tactic} left {nan_rows} output rows unwritten"
        worst = 0.0
        for lo in range(0, m, 4096):
            hi = min(m, lo + 4096)
            ref = a[lo:hi].float() @ w.float().t()
            worst = max(worst, (c[lo:hi].float() - ref).abs().max().item())
        assert worst < 0.5, f"tactic {tactic} wrong output: max_abs={worst:.3f}"

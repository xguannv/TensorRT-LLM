# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Kimi K3 attention-residual + RMSNorm op."""

import pytest
import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

HIDDEN_SIZE = 7168
ATTN_RES_RMS_EPS = 1e-6
OUTPUT_RMS_EPS = 1e-6


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {
        (10, 0),
        (10, 3),
    }


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 attention-residual kernels require SM100/SM103",
)


def _production_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply the unfused production RMSNorm that the new op replaces."""
    if IS_FLASHINFER_AVAILABLE:
        from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm

        return flashinfer_rmsnorm(hidden_states.contiguous(), weight, eps)

    hidden_float = hidden_states.float()
    variance = hidden_float.square().mean(dim=-1, keepdim=True)
    normalized = hidden_float * torch.rsqrt(variance + eps)
    return weight * normalized.to(hidden_states.dtype)


def _make_inputs(
    num_tokens: int,
    num_snapshots: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    layer_residual = (
        torch.randn(
            num_tokens,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    block_residual = (
        torch.randn(
            num_snapshots,
            num_tokens,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    res_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    score_rms_weight = 1 + torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    output_rms_weight = 1 + torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    return (
        layer_residual,
        block_residual,
        res_weight.contiguous(),
        score_rms_weight.contiguous(),
        output_rms_weight.contiguous(),
    )


def _unfused_reference(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
) -> torch.Tensor:
    mixed, _rsigma, _probs, _logits = torch.ops.trtllm.attn_res_fwd(
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        ATTN_RES_RMS_EPS,
    )
    return _production_rms_norm(mixed, output_rms_weight, OUTPUT_RMS_EPS)


def _fused(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.trtllm.attn_res_rmsnorm_fwd(
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
        ATTN_RES_RMS_EPS,
        OUTPUT_RMS_EPS,
    )


def _similarity(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(),
        expected_float.flatten(),
        dim=0,
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    return cosine, relative_l2


@pytest.mark.parametrize(
    ("num_tokens", "num_snapshots"),
    [
        (1, 0),  # N=1 single-CTA decode
        (1, 1),  # N=2 single-CTA decode
        (1, 2),  # N=3 general decode
        (1, 3),  # N=4 single-CTA decode
        (1, 7),  # N=8 CTA-cluster decode
        (1, 11),  # N=12 CTA-cluster decode
        (16, 1),  # multi-token general path
        (64, 5),  # multi-token, partially filled candidate chunk
        (1024, 11),  # K3 N=12/T=1024 specialization
    ],
)
@torch.no_grad()
def test_attn_res_rmsnorm_matches_unfused(
    num_tokens: int,
    num_snapshots: int,
) -> None:
    inputs = _make_inputs(num_tokens, num_snapshots)
    expected = _unfused_reference(*inputs)
    actual = _fused(*inputs)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_attn_res_rmsnorm_cuda_graph_replay() -> None:
    inputs = _make_inputs(num_tokens=1, num_snapshots=3)
    expected = _unfused_reference(*inputs)

    _fused(*inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _fused(*inputs)
    graph.replay()
    torch.cuda.synchronize()

    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill integration test for direct recurrent-state pool writes."""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch  # noqa: E402

NUM_HEADS = 96
HEAD_DIM = 128
LOWER_BOUND = -5.0


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {
        (10, 0),
        (10, 3),
    }


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(),
        expected_float.flatten(),
        dim=0,
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999
    assert relative_l2 < 3e-2


@torch.no_grad()
def test_kda_prefill_transpose_scatter_writes_selected_pool_slots() -> None:
    optimized = KDAKernelDispatch(
        use_optimized_prefill=True,
        use_optimized_decode=False,
    )
    reference = KDAKernelDispatch(
        use_optimized_prefill=False,
        use_optimized_decode=False,
    )
    assert optimized.prefill_kernel_path == "optimized"
    assert reference.prefill_kernel_path == "fla"

    generator = torch.Generator(device="cuda").manual_seed(616)

    def random_tensor(
        *shape: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        return torch.randn(
            *shape,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        ).to(dtype)

    batch_size, sequence_length = 2, 256
    q = random_tensor(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
    k = random_tensor(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
    v = random_tensor(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
    g = random_tensor(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
    beta = random_tensor(
        batch_size,
        sequence_length,
        NUM_HEADS,
        dtype=torch.float32,
    )
    a_log = random_tensor(NUM_HEADS, dtype=torch.float32) * 0.5
    dt_bias = random_tensor(NUM_HEADS * HEAD_DIM, dtype=torch.float32) * 0.1

    def call_args() -> dict:
        return {
            "q": q.clone(),
            "k": k.clone(),
            "v": v.clone(),
            "g": g.clone(),
            "beta": beta.clone(),
            "A_log": a_log,
            "dt_bias": dt_bias,
            "scale": HEAD_DIM**-0.5,
            "initial_state": None,
            "safe_gate": True,
            "lower_bound": LOWER_BOUND,
            "cu_seqlens": None,
        }

    out_ref, state_ref = reference.prefill_chunk_kda(**call_args())

    sentinel = -12345.0
    state_pool = torch.full(
        (5, NUM_HEADS, HEAD_DIM, HEAD_DIM),
        sentinel,
        dtype=torch.float32,
        device="cuda",
    )
    slot_indices = torch.tensor([4, 1], dtype=torch.int64, device="cuda")
    out_opt, state_opt = optimized.prefill_chunk_kda(
        **call_args(),
        final_state_pool=state_pool,
        final_state_indices=slot_indices,
    )

    assert state_opt is None
    _assert_close(out_opt, out_ref)
    _assert_close(state_pool.index_select(0, slot_indices), state_ref)

    untouched_slots = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")
    untouched = state_pool.index_select(0, untouched_slots)
    torch.testing.assert_close(
        untouched,
        torch.full_like(untouched, sentinel),
        rtol=0,
        atol=0,
    )

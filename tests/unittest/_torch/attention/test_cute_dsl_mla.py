# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE


def test_fp8_mla_scales_remain_tensors() -> None:
    q = torch.empty(1)
    bmm1_scale = torch.tensor([0.25, 0.25 * 1.442695], dtype=torch.float32)
    bmm2_scale = torch.tensor([4.0], dtype=torch.float32)

    actual_bmm1_scale, actual_bmm2_scale = CuteDslMlaFmha._get_fp8_scales(
        q,
        SimpleNamespace(
            mla_bmm1_scale=bmm1_scale,
            mla_bmm2_scale=bmm2_scale,
        ),
    )

    assert actual_bmm1_scale is bmm1_scale
    assert actual_bmm2_scale is bmm2_scale


def test_explicit_fp8_cute_dsl_bypasses_perf_gate() -> None:
    metadata = SimpleNamespace(num_contexts=0, num_generations=1)
    attn = SimpleNamespace(
        flashinfer_mla_backend="cute-dsl",
        mla_backend_policy=None,
    )

    assert CuteDslMlaFmha._is_fp8_cute_dsl_selected(
        attn,
        metadata,
        num_tokens=1,
        kernel_dtype=torch.float8_e4m3fn,
    )


def test_fp8_cute_dsl_respects_per_batch_backend_policy() -> None:
    metadata = SimpleNamespace(num_contexts=0, num_generations=1)
    attn = SimpleNamespace(
        flashinfer_mla_backend="cute-dsl",
        mla_backend_policy=lambda _backend, _metadata, _num_tokens: "trtllm-gen",
    )

    assert not CuteDslMlaFmha._is_fp8_cute_dsl_selected(
        attn,
        metadata,
        num_tokens=2,
        kernel_dtype=torch.float8_e4m3fn,
    )


def test_fp8_h96_decode_does_not_bypass_backend_selection() -> None:
    favorable, reason = CuteDslMlaFmha._is_perf_favorable(
        num_heads=96,
        batch_size=1,
        seq_len_q=1,
        kernel_dtype=torch.float8_e4m3fn,
    )

    assert not favorable
    assert "not a perf win" in reason


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
def test_fp8_mla_custom_op_accepts_tensor_scales() -> None:
    # Import registers the CuTe DSL custom ops when the package is available.
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401

    schema = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell.default._schema
    argument_types = {argument.name: str(argument.type) for argument in schema.arguments}

    assert argument_types["softmax_scale"] == "Tensor"
    assert argument_types["output_scale"] == "Tensor"

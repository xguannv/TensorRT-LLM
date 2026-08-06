# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_multi_ctas_kv_counter_size,
)


class _AttentionStub:
    def __init__(
        self,
        *,
        is_mla_enable: bool,
        has_fp8_kv_cache: bool,
        flashinfer_mla_backend: str = "trtllm-gen",
    ) -> None:
        self.is_mla_enable = is_mla_enable
        self.has_fp8_kv_cache = has_fp8_kv_cache
        self.flashinfer_mla_backend = flashinfer_mla_backend
        self.kv_lora_rank = 512 if is_mla_enable else None
        self.head_dim = 576
        self.v_head_dim = 512 if is_mla_enable else None


def _get_total_num_blocks(manager: SimpleNamespace, kv_factor: int = 2) -> int:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = kv_factor
    return fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=manager))


def test_flashinfer_uses_v2_page_index_upper_bound_directly() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=50_000_000,
        impl=SimpleNamespace(get_page_index_upper_bound=lambda *_: 50_000_000),
        num_local_layers=36,
    )
    assert _get_total_num_blocks(manager) == 50_000_000


def test_flashinfer_preserves_legacy_pool_scaling() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=1024,
        impl=SimpleNamespace(),
        num_local_layers=36,
    )
    assert _get_total_num_blocks(manager, kv_factor=2) == 1024 * 36 * 2


def test_multi_ctas_kv_counter_size_covers_beam_expanded_batch() -> None:
    # The kernel keeps one counter per head per decoder sequence. Sizing off the
    # request count alone under-allocates under beam search, but only once the
    # product clears the multi-processor floor, so pick a case that does.
    num_heads, batch, beam, sm_count = 6, 16, 2, 148
    needed = num_heads * batch * beam * torch.int32.itemsize
    assert _get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) < needed
    assert _get_multi_ctas_kv_counter_size(num_heads, batch * beam, sm_count) >= needed


def test_multi_ctas_kv_counter_size_keeps_multi_processor_floor() -> None:
    num_heads, batch, sm_count = 6, 1, 148
    assert _get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) >= (
        sm_count * torch.int32.itemsize
    )


def test_prepare_workspace_sizes_counter_for_max_num_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_heads, max_num_requests, beam_width, sm_count = 6, 16, 2, 148
    max_num_sequences = max_num_requests * beam_width

    def check_counter_size_args(
        actual_num_heads: int,
        actual_max_num_sequences: int,
        actual_sm_count: int,
    ) -> int:
        assert (actual_num_heads, actual_max_num_sequences, actual_sm_count) == (
            num_heads,
            max_num_sequences,
            sm_count,
        )
        raise RuntimeError("counter size arguments observed")

    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen."
        "_get_multi_ctas_kv_counter_size",
        check_counter_size_args,
    )

    fmha = SimpleNamespace(
        attn=SimpleNamespace(num_heads=num_heads),
        _multi_processor_count=sm_count,
    )
    metadata = SimpleNamespace(
        max_num_requests=max_num_requests,
        beam_width=beam_width,
        max_num_sequences=max_num_sequences,
    )
    with pytest.raises(RuntimeError, match="counter size arguments observed"):
        FlashInferTrtllmGenFmha.prepare_workspace(
            fmha,
            q=SimpleNamespace(),
            k=None,
            v=None,
            metadata=metadata,
            forward_args=SimpleNamespace(),
            workspace=SimpleNamespace(),
        )


def test_flashinfer_cute_dsl_mla_backend_rejects_fp8_kv_cache() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=True,
        flashinfer_mla_backend="cute-dsl",
    )

    with pytest.raises(ValueError, match="does not support FP8 KV cache"):
        FlashInferTrtllmGenFmha(attn)


@pytest.mark.parametrize(
    ("requested_backend", "num_contexts", "num_generations", "num_tokens", "expected_backend"),
    [
        ("cute-dsl", 0, 4, 4, "cute-dsl"),
        ("cute-dsl", 1, 3, 4, "trtllm-gen"),
        ("cute-dsl", 0, 4, 8, "trtllm-gen"),
        ("trtllm-gen", 1, 3, 4, "trtllm-gen"),
    ],
)
def test_flashinfer_mla_backend_falls_back_for_mixed_batches(
    requested_backend: str,
    num_contexts: int,
    num_generations: int,
    num_tokens: int,
    expected_backend: str,
) -> None:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._mla_backend = requested_backend

    assert (
        fmha._get_effective_mla_backend(
            SimpleNamespace(num_contexts=num_contexts, num_generations=num_generations),
            num_tokens,
        )
        == expected_backend
    )

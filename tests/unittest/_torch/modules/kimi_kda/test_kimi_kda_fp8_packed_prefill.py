# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity for the Kimi K3 FP8 packed q/k/v prefill projection."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("fla")

from tensorrt_llm._torch.models.modeling_kimi_linear import (
    _Fp8BlockScaleWeightReadLinear,
    _fill_kda_fp8_resident_projections,
    _k3_kda_fp8_resident_jobs,
)
from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention


class _Cfg:
    hidden_size = 768
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": 6,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


class _Layer(nn.Module):
    def __init__(self, attention: KimiKDALinearAttention) -> None:
        super().__init__()
        self.is_kda = True
        self.linear_attn = attention


class _Model(nn.Module):
    def __init__(self, attention: KimiKDALinearAttention) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(attention)])


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 FP8 projection is supported only on SM100/SM103",
)


def _make_attention() -> KimiKDALinearAttention:
    """A KDA attention module whose q/k/v/g/o are FP8-resident and filled.

    Mirrors production: the projections are built as empty FP8 placeholders at
    construction time and filled from checkpoint FP8 + block-scale pairs, which
    here are cast from random BF16 rather than read off disk.
    """
    from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

    def proj_factory(in_features: int, out_features: int) -> nn.Module:
        return _Fp8BlockScaleWeightReadLinear.empty_placeholder(out_features, in_features)

    attention = KimiKDALinearAttention(
        _Cfg(),
        layer_idx=0,
        fp8_resident_projections=True,
        proj_factory=proj_factory,
    ).to("cuda")
    jobs = list(_k3_kda_fp8_resident_jobs(_Model(attention), prefix=""))
    assert len(jobs) == 5, f"expected 5 FP8-resident projections, got {[j[2] for j in jobs]}"

    key_to_attr = {key: attr for _layer_idx, _mixer, attr, key in jobs}

    def _pair_of(key: str):
        mod = getattr(attention, key_to_attr[key])
        weight = (
            torch.randn(mod.out_features, mod.weight.shape[1], device="cuda", dtype=torch.bfloat16)
            * 0.05
        )
        fp8, scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        return fp8, scale.float()

    assert _fill_kda_fp8_resident_projections(jobs, _pair_of) == 5
    return attention


def _assert_numerically_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999
    assert relative_l2 < 3e-2


@torch.no_grad()
def test_fp8_packed_qkv_projection_matches_separate_views() -> None:
    torch.manual_seed(0)
    attention = _make_attention()
    hidden = torch.randn(1, 193, _Cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.05

    qkvg_proj = attention.qkvg_proj
    assert qkvg_proj is not None
    packed = qkvg_proj(hidden)[..., : 3 * attention.proj_size]
    actual = packed.split(attention.proj_size, dim=-1)
    expected = (attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden))

    for packed_part, separate_part in zip(actual, expected):
        _assert_numerically_close(packed_part, separate_part)


@pytest.mark.parametrize(
    "sequence_lengths,use_initial_states,has_initial_states",
    [
        ([17, 31, 64], False, [False, False, False]),
        ([1, 129], True, [True, False]),
    ],
)
@torch.no_grad()
def test_fp8_packed_qkv_prefill_matches_separate_path_and_updates_state(
    sequence_lengths: list[int],
    use_initial_states: bool,
    has_initial_states: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fla.modules.conv.triton.ops as fla_conv_ops

    torch.manual_seed(1)
    attention = _make_attention()
    num_prefills = len(sequence_lengths)
    num_tokens = sum(sequence_lengths)
    slots = num_prefills + 3
    d = attention.proj_size
    h = _Cfg.linear_attn_config["num_heads"]
    head_dim = _Cfg.linear_attn_config["head_dim"]
    conv_size = _Cfg.linear_attn_config["short_conv_kernel_size"]
    slot_indices = torch.arange(2, 2 + num_prefills, device="cuda", dtype=torch.long)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()], device="cuda", dtype=torch.long
    )
    chunk_indices = torch.tensor(
        [
            [sequence_index, chunk_index]
            for sequence_index, length in enumerate(sequence_lengths)
            for chunk_index in range((length + 63) // 64)
        ],
        device="cuda",
        dtype=torch.long,
    )
    metadata = SimpleNamespace(
        use_initial_states=use_initial_states,
        has_initial_states=torch.tensor(has_initial_states, device="cuda", dtype=torch.bool),
        kda_chunk_indices=chunk_indices,
        kda_varlen_is_aligned=all(length % 64 == 0 for length in sequence_lengths),
        kda_single_sequence_length=(sequence_lengths[0] if num_prefills == 1 else None),
    )
    hidden = torch.randn(num_tokens, _Cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.05
    hidden_pristine = hidden.clone()
    conv_seed = torch.randn(slots, 3 * d, conv_size, device="cuda", dtype=torch.bfloat16) * 0.02
    state_seed = (
        torch.randn(slots, h, head_dim, head_dim, device="cuda", dtype=torch.float32) * 0.01
    )

    calls = {"qkvg": 0, "q": 0, "k": 0, "v": 0}

    def _count(name):
        def _hook(_module, _inputs, _output):
            calls[name] += 1

        return _hook

    fused_qkvg = attention.qkvg_proj
    assert fused_qkvg is not None
    handles = [
        fused_qkvg.register_forward_hook(_count("qkvg")),
        attention.q_proj.register_forward_hook(_count("q")),
        attention.k_proj.register_forward_hook(_count("k")),
        attention.v_proj.register_forward_hook(_count("v")),
    ]
    monkeypatch.setattr(
        fla_conv_ops,
        "prepare_chunk_indices",
        lambda *_args, **_kwargs: pytest.fail(
            "KDA metadata should bypass short-convolution chunk-index preparation"
        ),
    )
    try:
        attention.qkvg_proj = None
        ref_conv = conv_seed.clone()
        ref_state = state_seed.clone()
        expected = attention.forward_prefill(
            hidden,
            cu_seqlens,
            metadata,
            num_prefills,
            ref_conv,
            ref_state,
            slot_indices,
        )
        assert calls == {"qkvg": 0, "q": 1, "k": 1, "v": 1}

        calls.update(qkvg=0, q=0, k=0, v=0)
        attention.qkvg_proj = fused_qkvg
        actual_conv = conv_seed.clone()
        actual_state = state_seed.clone()
        actual = attention.forward_prefill(
            hidden,
            cu_seqlens,
            metadata,
            num_prefills,
            actual_conv,
            actual_state,
            slot_indices,
        )
        assert calls == {"qkvg": 1, "q": 0, "k": 0, "v": 0}
    finally:
        attention.qkvg_proj = fused_qkvg
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(hidden, hidden_pristine, rtol=0, atol=0)
    _assert_numerically_close(actual, expected)
    _assert_numerically_close(
        actual_conv.index_select(0, slot_indices), ref_conv.index_select(0, slot_indices)
    )
    _assert_numerically_close(
        actual_state.index_select(0, slot_indices), ref_state.index_select(0, slot_indices)
    )

    untouched = torch.tensor([0, 1, slots - 1], device="cuda", dtype=torch.long)
    torch.testing.assert_close(
        actual_conv.index_select(0, untouched), conv_seed.index_select(0, untouched), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_state.index_select(0, untouched),
        state_seed.index_select(0, untouched),
        rtol=0,
        atol=0,
    )

    repeat_conv = conv_seed.clone()
    repeat_state = state_seed.clone()
    repeated = attention.forward_prefill(
        hidden,
        cu_seqlens,
        metadata,
        num_prefills,
        repeat_conv,
        repeat_state,
        slot_indices,
    )
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(repeat_conv, actual_conv, rtol=0, atol=0)
    torch.testing.assert_close(repeat_state, actual_state, rtol=0, atol=0)

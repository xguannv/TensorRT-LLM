# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete K3 MLA outputs with the pinned Hugging Face implementation."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _torch.attention.multi_gpu.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.interface import KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _checkpoint_weights(reference: KimiMLAAttention, fp8: bool) -> dict:
    projections = {}
    for name, parameter in reference.named_parameters():
        if parameter.ndim != 2:
            continue
        weight = parameter.detach().clone()
        pair = {"weight": weight}
        if fp8:
            rows, cols = weight.shape
            padded = F.pad(weight.float(), (0, -cols % 128, 0, -rows % 128))
            blocks = padded.view(math.ceil(rows / 128), 128, math.ceil(cols / 128), 128)
            scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
            expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
            pair = {
                "weight": (weight.float() / expanded).to(torch.float8_e4m3fn),
                "weight_scale": scale,
            }
            # Both implementations see the same stored weights. Only TRT-LLM
            # quantizes activations and uses FP8 GEMM/BMM kernels.
            with torch.no_grad():
                parameter.copy_((pair["weight"].float() * expanded).to(parameter.dtype))
        projections[name.removesuffix(".weight")] = pair
    return projections


def _assert_matches_reference(actual: torch.Tensor, expected: torch.Tensor, fp8: bool) -> None:
    # DeepGEMM weight resmoothing and activation rounding compound across the
    # A/B projections and absorption (about 7% relative L2 on this fixture).
    # Bound both elementwise outliers and the overall error relative to the signal.
    atol, rtol = (0.1, 0.1) if fp8 else (0.005, 0.03)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    relative_l2 = (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)
    ).item()
    assert relative_l2 < (0.08 if fp8 else 0.01), f"relative L2 error: {relative_l2}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "fp8", [False, pytest.param(True, marks=skip_pre_blackwell)], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
def test_k3_mla_matches_huggingface(fp8: bool, gate: bool) -> None:
    torch.manual_seed(42)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=gate,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for name in weights
        }
        if fp8
        else None,
        use_cute_dsl_blockscaling_mm=True,
        use_cute_dsl_blockscaling_bmm=True,
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    reference.cuda()

    batch, context_length = 2, 128
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        batch,
        context_length + 1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    causal_mask = torch.full(
        (context_length + 1, context_length + 1), float("-inf"), device="cuda"
    ).triu(1)[None, None]
    expected = reference(hidden, attention_mask=causal_mask)

    context = hidden[:, :context_length].reshape(-1, config.hidden_size).contiguous()
    scenario = SimpleNamespace(
        ctx_len=context_length,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.bfloat16,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {runtime.mixer.layer_idx_str: weakref.ref(runtime.mixer)},
        }
        with model_extra_attrs(extra_attrs):
            actual = runtime(context, metadata).view(batch, context_length, -1)
            _assert_matches_reference(actual, expected[:, :context_length], fp8)

        for request_id in range(batch):
            assert manager.kv_cache_map[request_id].capacity >= context_length + 1
        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.ones(batch, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=[context_length] * batch,
            max_num_tokens=batch,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[context_length] * batch
            ),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        with model_extra_attrs(extra_attrs):
            actual = runtime(hidden[:, -1].contiguous(), metadata)
        _assert_matches_reference(actual, expected[:, -1], fp8)
    finally:
        manager.shutdown()


def build_k3_mla_fp8_kv(num_heads: int = 96, max_positions: int = 1024, seed: int = 11):
    """One K3 MLA layer with FP8 block-scaled projections (DeepGEMM layout) for an FP8 KV cache.

    Returns ``(config, mapping, model, runtime)``; ``runtime.mixer`` is the MLA module.
    """
    torch.manual_seed(seed)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=max_positions,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8=True)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for name in weights
        },
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    return config, mapping, model, runtime


def context_metadata(manager, mapping, seq_lens, cached_lens, runtime_features=None):
    """Context-phase metadata of ``seq_lens`` new tokens over ``cached_lens`` cached tokens per request.

    ``prompt_lens`` is the chunk's own length, as the executor sets it for a context chunk (the chunked-prefill path
    reads it as the query lengths of the last, causal partition)."""
    kwargs = {} if runtime_features is None else {"runtime_features": runtime_features}
    metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(range(len(seq_lens))),
        max_num_requests=len(seq_lens),
        num_contexts=len(seq_lens),
        prompt_lens=list(seq_lens),
        max_num_tokens=sum(seq_lens),
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached_lens)),
        mapping=mapping,
        enable_context_mla_with_cached_kv=True,
        **kwargs,
    )
    metadata.prepare()
    return metadata


@pytest.mark.parametrize(
    "value,capability,expected",
    [
        ("", (10, 3), True),
        ("-1", (10, 3), True),
        ("", (10, 0), False),
        ("", (10, 7), False),
        ("1", (10, 0), True),
        ("0", (10, 3), False),
    ],
)
def test_k3ll_mla_ctx_switch_default(value, capability, expected, monkeypatch) -> None:
    """TRTLLM_K3LL_MLA_CTX_FP8 / _CUTE_FMHA: 1 on, 0 off, -1 or unset on SM103 only."""
    from tensorrt_llm._torch.attention import mla as mla_module

    mla_module._mla_ctx_fp8_by_default.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    for name in (mla_module.MLA_CTX_FP8_SWITCH, mla_module.MLA_CTX_CUTE_FMHA_SWITCH):
        monkeypatch.setenv(name, value)
        assert mla_module._k3ll_ctx_switch_on(name, torch.device("cuda", 0)) is expected
    mla_module._mla_ctx_fp8_by_default.cache_clear()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)
@pytest.mark.parametrize("fmha", ["trtllm-gen", "cute"])
def test_k3ll_mla_ctx_fp8_matches_default(fmha: str, monkeypatch) -> None:
    """Chunked prefill over a cached FP8 KV cache with TRTLLM_K3LL_MLA_CTX_FP8=1 (k3_mla_kv_expand writes FP8 K / V,
    a ragged FP8 FMHA attends them) against the attention op's path (=0): the same FP8 K / V / Q values, a different
    FMHA kernel, so the output matches to FMHA rounding noise. Several KV partitions plus the final causal one; two
    requests with different cached lengths."""
    from tensorrt_llm._torch.attention import mla as mla_module
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures

    if fmha == "cute" and not mla_module._mla_ctx_cute_fmha_available():
        pytest.skip("FlashInfer CuTe DSL FMHA not importable")
    if fmha == "trtllm-gen" and mla_module._flashinfer_fp8_ragged_mla() is None:
        pytest.skip("FlashInfer trtllm-gen ragged FMHA not importable")
    config, mapping, model, runtime = build_k3_mla_fp8_kv()
    mla = runtime.mixer
    assert mla._fused_fp8_kv_expand_eligible()

    total, cached = 384, [256, 200]
    new = [total - c for c in cached]
    batch = len(cached)
    scenario = SimpleNamespace(
        ctx_len=total,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.float8_e4m3fn,
    )
    manager, _ = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    generator = torch.Generator(device="cuda").manual_seed(5)
    hidden = torch.randn(
        batch, total, config.hidden_size, dtype=torch.bfloat16, device="cuda", generator=generator
    )

    fused_calls = []
    fused_expand = mla._fused_fp8_kv_expand

    def counting_expand(*args, **kwargs):
        fused_calls.append(args[0].shape[0])
        return fused_expand(*args, **kwargs)

    monkeypatch.setattr(mla, "_fused_fp8_kv_expand", counting_expand)
    monkeypatch.setenv(mla_module.MLA_CTX_CUTE_FMHA_SWITCH, "1" if fmha == "cute" else "0")
    outputs = {}
    calls_before = {}
    try:
        monkeypatch.setenv(mla_module.MLA_CTX_FP8_SWITCH, "0")
        # The first step fills the FP8 cache with the first cached[i] tokens of each request.
        first = context_metadata(manager, mapping, cached, [0] * batch)
        extra_attrs = {
            "attention_metadata": weakref.ref(first),
            "mla_layers": {mla.layer_idx_str: weakref.ref(mla)},
        }
        with model_extra_attrs(extra_attrs):
            runtime(torch.cat([hidden[i, : cached[i]] for i in range(batch)]), first)
        second_input = torch.cat([hidden[i, cached[i] :] for i in range(batch)])
        # 64-token KV partitions: the loop runs ceil(456 / 64) = 8 of them.
        features = AttentionRuntimeFeatures(
            chunked_prefill=True, chunk_size=64, chunked_prefill_buffer_batch_size=1
        )
        for flag in ("0", "1"):
            monkeypatch.setenv(mla_module.MLA_CTX_FP8_SWITCH, flag)
            calls_before[flag] = mla_module._MLA_CTX_FP8_CALLS[0]
            second = context_metadata(manager, mapping, new, cached, features)
            extra_attrs["attention_metadata"] = weakref.ref(second)
            with model_extra_attrs(extra_attrs):
                outputs[flag] = runtime(second_input, second)
            torch.cuda.synchronize()
            if flag == "0":
                # Switch off: the attention op's path, no fused expansion, no census count.
                assert fused_calls == []
                assert mla_module._MLA_CTX_FP8_CALLS[0] == calls_before["0"]
        assert mla_module._MLA_CTX_FP8_CALLS[0] == calls_before["1"] + 1
        assert mla._mla_ctx_fp8_fmha(second_input.device) == fmha
    finally:
        manager.shutdown()

    # The first fused step of a process compiles the kernels on 1-token inputs (warmup).
    while fused_calls and fused_calls[0] == 1:
        fused_calls.pop(0)
    assert fused_calls == [64] * 7 + [456 - 7 * 64] + [sum(new)]
    expected = outputs["0"].float()
    assert expected.abs().sum() > 0
    diff = (outputs["1"].float() - expected).abs()
    relative_l2 = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected)).item()
    print(
        f"mla_ctx_fp8 fmha={fmha}: relative L2 {relative_l2:.3e}, max abs {diff.max().item():.3e}, "
        f"mean abs {diff.mean().item():.3e}, mean |ref| {expected.abs().mean().item():.3e}"
    )
    assert torch.isfinite(outputs["1"]).all()
    if fmha == "trtllm-gen":
        assert relative_l2 < 2e-3, f"relative L2 error: {relative_l2}"
        torch.testing.assert_close(outputs["1"], outputs["0"], atol=2e-2, rtol=2e-2)
    else:
        # A different FMHA: its FP8 rounding of P is independent of the trtllm-gen one, and each is ~2.5e-2
        # (relative L2) from an FP32 reference on these diffuse random rows (test_k3_mla_ctx_fmha.py checks the
        # kernel against that reference).
        assert relative_l2 < 8e-2, f"relative L2 error: {relative_l2}"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete K3 MLA outputs with the pinned Hugging Face implementation."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from _torch.modules.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures, KVCacheParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
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


def _k3_mla_config(gate: bool) -> KimiLinearConfig:
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
    return config


def _build_k3_mla(
    config: KimiLinearConfig,
    reference: KimiMLAAttention,
    mapping: Mapping,
    fp8: bool,
):
    """K3 model with one MLA layer holding ``reference``'s (FP8-quantized) weights."""
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
    return model, runtime


def _decode_metadata(manager, mapping: Mapping, batch: int, context_length: int):
    """Metadata of one generation step after a ``context_length``-token prefill."""
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
    return metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "fp8", [False, pytest.param(True, marks=skip_pre_blackwell)], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
def test_k3_mla_matches_huggingface(fp8: bool, gate: bool) -> None:
    torch.manual_seed(42)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = _k3_mla_config(gate)
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    model, runtime = _build_k3_mla(config, reference, mapping, fp8)
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
            manager.impl.add_token(request_id)
        metadata = _decode_metadata(manager, mapping, batch, context_length)
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        with model_extra_attrs(extra_attrs):
            actual = runtime(hidden[:, -1].contiguous(), metadata)
        _assert_matches_reference(actual, expected[:, -1], fp8)
    finally:
        manager.shutdown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)
@pytest.mark.parametrize("batch", [2, 7])
def test_k3_mla_fp8_lowm_fused_a(batch: int, monkeypatch) -> None:
    """``TRTLLM_K3_FP8W_LOWM_FUSED_A``: the fused 2112-row Q-A/KV-A projection runs the low-M
    FP8 GEMM at decode, and the MLA output still matches Hugging Face and the DeepGEMM run."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    torch.manual_seed(42)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = _k3_mla_config(gate=True)
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    model, runtime = _build_k3_mla(config, reference, mapping, fp8=True)
    reference.cuda()
    mla = runtime.mixer
    # The loader swaps every FP8 attention Linear for the K3 weight-read module; the swap
    # keeps the partial-tile mark of the fused projection.
    fused_a = mla.kv_a_proj_with_mqa
    assert isinstance(fused_a, kimi._Fp8BlockScaleWeightReadLinear)
    assert mla.fuse_qkv_a_proj and fused_a.fp8w_lowm_partial_n_tile
    assert not mla.o_proj.fp8w_lowm_partial_n_tile
    n = config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim
    assert fused_a.weight.shape == (n, config.hidden_size)

    calls = []
    can_run = kimi._k3_lowm_gemm_can_run

    def spy(x, weight, weight_scale, allow_partial_n_tile=False):
        result = can_run(x, weight, weight_scale, allow_partial_n_tile=allow_partial_n_tile)
        calls.append((weight.shape[0], x.shape[0], result))
        return result

    monkeypatch.setattr(kimi, "_k3_lowm_gemm_can_run", spy)
    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)

    context_length = 128
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
    expected = reference(hidden, attention_mask=causal_mask)[:, -1]

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
    outputs = {}
    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {mla.layer_idx_str: weakref.ref(mla)},
        }
        with model_extra_attrs(extra_attrs):
            runtime(context, metadata)
        for request_id in range(batch):
            manager.impl.add_token(request_id)
        decode_in = hidden[:, -1].contiguous()
        for flag in (False, True):
            monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_FUSED_A", flag)
            calls.clear()
            metadata = _decode_metadata(manager, mapping, batch, context_length)
            extra_attrs["attention_metadata"] = weakref.ref(metadata)
            with model_extra_attrs(extra_attrs):
                # The runtime may hand back a reused buffer: keep a copy per run.
                outputs[flag] = runtime(decode_in, metadata).clone()
            torch.cuda.synchronize()
            assert (n, batch, flag) in calls, calls
    finally:
        manager.shutdown()

    for flag in (False, True):
        _assert_matches_reference(outputs[flag], expected, fp8=True)
    relative_l2 = (
        torch.linalg.vector_norm(outputs[True].float() - outputs[False].float())
        / torch.linalg.vector_norm(outputs[False].float())
    ).item()
    assert relative_l2 < 2e-2, relative_l2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
def test_k3_mla_fp8_kv_decode_q_absorb() -> None:
    """FP8 KV cache decode: the FP8 query built in one kernel
    (``fp8_nope_q_absorb``) matches the BF16 absorption BMM + RoPE-kernel
    quantization path (q_pe bitwise, the absorbed q_nope to one FP8 step), and
    the KV append and BMM scales of the KV-only RoPE kernel are unchanged."""
    torch.manual_seed(7)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=96,
        num_key_value_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8=False)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
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
    mla = runtime.mixer

    batch, context_length = 3, 128
    scenario = SimpleNamespace(
        ctx_len=context_length,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.float8_e4m3fn,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    generator = torch.Generator(device="cuda").manual_seed(99)
    hidden = torch.randn(
        batch * (context_length + 1),
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    captured = {}

    def capture_attention(attn_backend, q, k, v, position_ids, attn_metadata, **kwargs):
        # Stop at the decode kernel: record what the Q/KV preparation produced.
        captured["q"] = kwargs["quant_q_buffer"].clone()
        captured["bmm1"] = kwargs["mla_bmm1_scale"].clone()
        captured["bmm2"] = kwargs["mla_bmm2_scale"].clone()
        return q.new_zeros(q.shape[0], mla.num_heads_tp * mla.kv_lora_rank)

    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {mla.layer_idx_str: weakref.ref(mla)},
        }
        context = hidden.view(batch, context_length + 1, -1)[:, :context_length]
        with model_extra_attrs(extra_attrs):
            runtime(context.reshape(-1, config.hidden_size).contiguous(), metadata)

        for request_id in range(batch):
            manager.impl.add_token(request_id)
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
        decode_in = hidden.view(batch, context_length + 1, -1)[:, -1].contiguous()
        kv_pool = manager.get_buffers(0)
        results = {}
        mla._attn_forward_gen = capture_attention
        for fused in (False, True):
            mla.fp8_nope_q_absorb = fused
            kv_pool.view(torch.uint8).zero_()
            with model_extra_attrs(extra_attrs):
                runtime(decode_in, metadata)
            torch.cuda.synchronize()
            results[fused] = dict(captured, kv=kv_pool.clone())
    finally:
        manager.shutdown()

    old, new = results[False], results[True]
    head_dim = config.kv_lora_rank + config.qk_rope_head_dim
    q_old = old["q"].view(torch.float8_e4m3fn).view(batch, 96, head_dim).float()
    q_new = new["q"].view(torch.float8_e4m3fn).view(batch, 96, head_dim).float()
    assert torch.equal(q_new[..., 512:], q_old[..., 512:])
    nope_old, nope_new = q_old[..., :512], q_new[..., :512]
    step = torch.exp2(torch.floor(torch.log2(nope_old.abs().clamp_min(2.0**-6))) - 3)
    assert ((nope_new - nope_old).abs() <= step * 1.0001).all()
    # double rounding (FP32 -> BF16 -> FP8) moves ~3% of values by one step
    assert (nope_new != nope_old).float().mean().item() < 0.1
    assert torch.equal(new["bmm1"], old["bmm1"]) and torch.equal(new["bmm2"], old["bmm2"])
    assert torch.equal(new["kv"].view(torch.uint8), old["kv"].view(torch.uint8))
    assert new["kv"].view(torch.uint8).any()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)
@pytest.mark.parametrize(
    "chunked,mode",
    [(True, "1"), (False, "1"), (True, "2"), (True, "2cute")],
    ids=[
        "chunked_prefill",
        "cached_kv",
        "chunked_prefill_fp8_fmha",
        "chunked_prefill_fp8_cute_fmha",
    ],
)
def test_k3_mla_ctx_fp8_expand_matches_default(chunked: bool, mode: str, monkeypatch) -> None:
    """Context attention over cached FP8 KV with ``TRTLLM_K3_MLA_CTX_FP8_EXPAND=1`` (one FP8
    GEMM writes K and V) is bit-identical to the DeepGEMM kv_b_proj + cat path, through the
    chunked-prefill loop (several KV chunks plus the final uncached partition) and through the
    plain cached-KV path. With ``=2`` the GEMM writes FP8 K / V and FlashInfer's FP8 FMHA runs
    the partitions: same FP8 inputs as the attention op's FP8 context path, a different FMHA
    kernel, so the output matches to accumulation-order noise."""
    torch.manual_seed(11)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=96,
        num_key_value_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=1024,
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
    mla = runtime.mixer
    del reference

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

    def context_metadata(seq_lens, cached_lens, runtime_features=None):
        kwargs = {} if runtime_features is None else {"runtime_features": runtime_features}
        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=batch,
            prompt_lens=list(seq_lens),
            max_num_tokens=sum(seq_lens),
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
            **kwargs,
        )
        metadata.prepare()
        return metadata

    fused_calls = []
    fused_expand = mla._fused_fp8_kv_expand

    def counting_expand(*args, **kwargs):
        fused_calls.append(args[0].shape[0])
        return fused_expand(*args, **kwargs)

    monkeypatch.setattr(mla, "_fused_fp8_kv_expand", counting_expand)
    outputs = {}
    try:
        # The first chunk fills the FP8 cache with the first cached[i] tokens of each request.
        first = context_metadata(cached, [0] * batch)
        extra_attrs = {
            "attention_metadata": weakref.ref(first),
            "mla_layers": {mla.layer_idx_str: weakref.ref(mla)},
        }
        with model_extra_attrs(extra_attrs):
            runtime(torch.cat([hidden[i, : cached[i]] for i in range(batch)]), first)
        second_input = torch.cat([hidden[i, cached[i] :] for i in range(batch)])
        # 64-token KV chunks: the chunked-prefill loop runs ceil(456 / 64) = 8 partitions.
        features = (
            AttentionRuntimeFeatures(
                chunked_prefill=True, chunk_size=64, chunked_prefill_buffer_batch_size=1
            )
            if chunked
            else None
        )
        if mode == "2cute":
            # The CuTe DSL FMHA (TRTLLM_K3_MLA_CTX_FMHA=cute) on the same FP8 partitions.
            monkeypatch.setenv("TRTLLM_K3_MLA_CTX_FMHA", "cute")
        for flag in ("0", mode):
            monkeypatch.setenv("TRTLLM_K3_MLA_CTX_FP8_EXPAND", flag[:1])
            second = context_metadata(new, cached, features)
            extra_attrs["attention_metadata"] = weakref.ref(second)
            with model_extra_attrs(extra_attrs):
                outputs[flag] = runtime(second_input, second)
            torch.cuda.synchronize()
        assert mla._use_fused_fp8_kv_expand()
    finally:
        manager.shutdown()

    # The first fused step of a process compiles the kernels on 1-token inputs (warmup).
    while fused_calls and fused_calls[0] == 1:
        fused_calls.pop(0)
    if chunked:
        assert fused_calls == [64] * 7 + [456 - 7 * 64] + [sum(new)]
    else:
        assert fused_calls == [total * batch]
    assert outputs["0"].abs().sum() > 0
    if mode == "1":
        assert torch.equal(outputs[mode], outputs["0"])
    else:
        assert mla._use_fp8_direct_chunked_prefill(
            torch.empty(1, dtype=torch.bfloat16), torch.empty(1, dtype=torch.bfloat16)
        )
        expected = outputs["0"].float()
        relative_l2 = (
            torch.linalg.vector_norm(outputs[mode].float() - expected)
            / torch.linalg.vector_norm(expected)
        ).item()
        if mode == "2":
            assert relative_l2 < 2e-3, f"relative L2 error: {relative_l2}"
            torch.testing.assert_close(outputs[mode], outputs["0"], atol=2e-2, rtol=2e-2)
        else:
            # A different FMHA: its FP8 rounding of P is independent of the trtllm-gen one, and
            # each is ~2.5e-2 (relative L2) from an FP32 reference on these diffuse random rows
            # (test_k3_mla_ctx_cute_fmha.py checks the kernel against that reference).
            assert relative_l2 < 8e-2, f"relative L2 error: {relative_l2}"

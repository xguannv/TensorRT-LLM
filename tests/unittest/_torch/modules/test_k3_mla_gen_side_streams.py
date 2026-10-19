# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi-K3 MLA Helix generation with the side-stream switches is bitwise the default path.

One Helix CP rank (16-way group) on one GPU: the real ``KimiK3MLAAttention`` (FP8 projections,
FP8 KV cache, CuTe FP8 MLA decode, V up-projection before the all-to-all) with the all-to-all
replaced by a local stand-in, run eagerly and in CUDA graphs with multi-stream enabled. Every
combination of ``TRTLLM_K3_MLA_GATE_SIDE``, ``TRTLLM_K3_MLA_Q_NORM_PDL``,
``TRTLLM_K3_MLA_KV_SIDE`` (``_PRIO``), ``TRTLLM_K3_MLA_KV_FUSED``, ``TRTLLM_K3_MLA_GATE_EARLY`` and
``TRTLLM_K3_MLA_KV_JOIN_EARLY`` must give the default path's output bit for bit; the fused KV
kernel must also write the same cache bytes and decode buffers as the kernels it replaces.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

H, HEADS, CP, CP_RANK, PAGE = 7168, 96, 16, 1, 64
_SHAPES = {
    "q_a_proj": (1536, H),
    "kv_a_proj_with_mqa": (576, H),
    "q_b_proj": (HEADS * 192, 1536),
    "kv_b_proj": (HEADS * 256, 512),
    "g_proj": (HEADS * 128, H),
    "o_proj": (H, HEADS * 128),
}


class _LocalHelix:
    """``HelixAllToAllNative`` stand-in: every peer 'sends' this rank's partials (the zero-KV
    mask applied as the real sender does)."""

    @classmethod
    def get(cls, mapping):
        return cls()

    def alltoall_native(self, partial_o, softmax_stats, zero_kv_mask=None):
        if zero_kv_mask is None:
            return partial_o.clone(), softmax_stats.clone()
        mask = zero_kv_mask.view(-1, *([1] * (partial_o.dim() - 1)))
        out_o = partial_o.masked_fill(mask, 0.0)
        stats = softmax_stats.view(*softmax_stats.shape[:-1], -1, 2)
        sm = stats[..., 0].masked_fill(mask, float("-inf"))
        ss = stats[..., 1].masked_fill(mask, 0.0)
        return out_o, torch.stack([sm, ss], dim=-1).view_as(softmax_stats)


def _load(model, projections):
    name_map, _, _ = model.checkpoint_name_plan("")
    config = model.model_config.pretrained_config
    params = dict(model.named_parameters())
    weights = {}
    for name, key in name_map.items():
        param = params[name]
        if name.endswith("gate_up_proj.weight"):
            for source in ("gate_proj", "up_proj"):
                weights[key.replace("gate_up_proj", source)] = torch.zeros(
                    config.intermediate_size, config.hidden_size, dtype=param.dtype
                )
        else:
            shape = param.shape
            if name == "lm_head.weight":
                shape = (config.vocab_size, config.hidden_size)
            elif name.endswith(".mlp.down_proj.weight"):
                shape = (config.hidden_size, config.intermediate_size)
            weights[key] = torch.zeros(shape, dtype=param.dtype)
    weights.update({f"model.layers.0.self_attn.{n}": v for n, v in projections.items()})
    model.load_weights(weights)
    # What the model loader runs next (``ModelLoader._walk_full_post_load``): among others it
    # converts the FP8 projections' scales to the layout their GEMMs read.
    for module in model.modules():
        post_load_weights = getattr(module, "post_load_weights", None)
        if post_load_weights is not None and not getattr(module, "_weights_removed", False):
            post_load_weights()


def _build(batch, q_len, kv_len, monkeypatch):
    from tensorrt_llm._torch.attention_backend.interface import KVCacheParams
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear as mk
    from tensorrt_llm._torch.modules import attention as attn_mod
    from tensorrt_llm._torch.pyexecutor.llm_request import (
        LlmRequest,
        LlmRequestState,
        SamplingConfig,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
    from tensorrt_llm.bindings.executor import KvCacheConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import CpType, Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
    from tensorrt_llm.sampling_params import SamplingParams

    monkeypatch.setattr(attn_mod, "HelixAllToAllNative", _LocalHelix)
    # The formal GEN switches this path is built for (read at import / construction).
    monkeypatch.setattr(mk, "_K3_HELIX_V_UP_BEFORE_A2A", True)
    monkeypatch.setattr(mk, "_K3_FP8W_LOWM_GEMM", True)
    monkeypatch.setattr(mk, "_K3_FP8W_LOWM_FUSED_A", True)
    cfg = KimiLinearConfig(
        vocab_size=128,
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        attn_res_block_size=1,
        mla_use_output_gate=True,
        mla_use_nope=True,
        max_position_embeddings=kv_len + 256,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": HEADS},
    )
    mapping = Mapping(
        world_size=CP,
        tp_size=1,
        cp_size=CP,
        rank=CP_RANK,
        cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False, "fifo_version": 2},
    )
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict={
            f"model.layers.0.self_attn.{n}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for n in _SHAPES
        },
        allreduce_strategy=AllReduceStrategy.NCCL,
    )
    with torch.device("cuda"):
        model = mk.KimiLinearForCausalLM(model_config)
    gen = torch.Generator().manual_seed(11)
    projections = {}
    for name, shape in _SHAPES.items():
        projections[f"{name}.weight"] = torch.randn(shape, generator=gen).to(torch.float8_e4m3fn)
        grid = tuple(math.ceil(d / 128) for d in shape)
        scale = torch.exp2(torch.randint(-10, -7, grid, generator=gen).float())
        projections[f"{name}.weight_scale"] = scale[:, None, :, None]
    projections["q_a_layernorm.weight"] = (1 + 0.1 * torch.randn(1536, generator=gen)).bfloat16()
    projections["kv_a_layernorm.weight"] = (1 + 0.1 * torch.randn(512, generator=gen)).bfloat16()
    _load(model, projections)
    mixer = model.model.layers[0].self_attn.mixer
    mixer.fp8_nope_q_absorb = True
    assert mixer._helix_v_up_enabled()

    local = Mapping(world_size=1, tp_size=1, rank=0)
    max_seq = kv_len + q_len + PAGE
    manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=math.ceil(max_seq / PAGE) * PAGE * batch, enable_block_reuse=False
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=PAGE,
        max_seq_len=max_seq,
        max_batch_size=batch,
        mapping=local,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
    )
    requests = []
    for rid in range(batch):
        req = LlmRequest(
            request_id=rid,
            max_new_tokens=q_len + 1,
            input_tokens=[1] * kv_len,
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.is_dummy_request = True
        req.paged_kv_block_ids = []
        requests.append(req)
    manager.impl.add_sequence_batch([(rid, kv_len, 1) for rid in range(batch)], requests)
    for req in requests:
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        req.prompt_len = kv_len
        req.py_prompt_len = kv_len
    for rid in range(batch):
        for _ in range(q_len):
            manager.impl.add_token(rid)
    pool = manager.get_buffers(0)
    kv_gen = torch.Generator(device="cuda").manual_seed(3)
    vals = (torch.randn(pool.shape, device="cuda", generator=kv_gen) * 0.5).to(torch.float8_e4m3fn)
    pool.view(torch.uint8).copy_(vals.view(torch.uint8))
    md = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([q_len] * batch, dtype=torch.int),
        request_ids=list(range(batch)),
        max_num_requests=batch,
        num_contexts=0,
        prompt_lens=[kv_len] * batch,
        max_num_tokens=batch * q_len,
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[kv_len] * batch),
        mapping=local,
        enable_context_mla_with_cached_kv=True,
    )
    md.prepare()
    x_gen = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn((batch * q_len, H), device="cuda", generator=x_gen).to(torch.bfloat16)
    return model, mixer, manager, md, x


def _set(
    monkeypatch, k3mla, gate, qnorm, kv, prio, kv_fused=False, gate_early=False, kv_join=False
):
    monkeypatch.setattr(k3mla, "_K3_MLA_GATE_SIDE", gate)
    monkeypatch.setattr(k3mla, "_K3_MLA_Q_NORM_PDL", qnorm)
    monkeypatch.setattr(k3mla, "_K3_MLA_KV_SIDE", kv)
    monkeypatch.setattr(k3mla, "_K3_MLA_KV_SIDE_PRIO", prio)
    monkeypatch.setattr(k3mla, "_K3_MLA_KV_FUSED", kv_fused)
    monkeypatch.setattr(k3mla, "_K3_MLA_GATE_EARLY", gate_early)
    monkeypatch.setattr(k3mla, "_K3_MLA_KV_JOIN_EARLY", kv_join)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_side_stream_paths_match_default(batch, monkeypatch):
    import weakref

    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla
    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.utils import model_extra_attrs

    q_len = 8 if batch == 4 else 7
    model, mixer, manager, md, x = _build(batch, q_len, 448, monkeypatch)
    extra = {
        "attention_metadata": weakref.ref(md),
        "mla_layers": {mixer.layer_idx_str: weakref.ref(mixer)},
    }
    pool = manager.get_buffers(0)
    pool_init = pool.clone()
    # Count the entry points of each switch's path, to check every path is taken.
    calls = dict.fromkeys(("gated", "sigmoid", "q_norm", "kv_side", "kv_fused"), 0)

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(mixer.o_proj, "forward_gated", counted("gated", mixer.o_proj.forward_gated))
    monkeypatch.setattr(
        mixer.g_proj, "forward_sigmoid", counted("sigmoid", mixer.g_proj.forward_sigmoid)
    )
    monkeypatch.setattr(
        q_norm_pdl,
        "rmsnorm_release_at_launch",
        counted("q_norm", q_norm_pdl.rmsnorm_release_at_launch),
    )
    monkeypatch.setattr(
        mixer, "_forward_gen_kv_side", counted("kv_side", mixer._forward_gen_kv_side)
    )
    monkeypatch.setattr(mixer, "_kv_append_fused", counted("kv_fused", mixer._kv_append_fused))
    results = {}
    taken = {}
    try:
        with torch.inference_mode():
            for cfg in [
                (False,) * 7,
                (True, False, False, False, False, False, False),
                (True, True, True, False, False, False, False),
                (True, True, True, True, False, False, False),
                (True, False, True, True, False, False, False),
                (True, True, True, True, True, False, False),
                (True, False, True, False, True, False, False),
                (True, True, True, True, True, False, True),
                (True, True, True, True, True, True, False),
                (True,) * 7,
            ]:
                _set(monkeypatch, k3mla, *cfg)
                calls.update(dict.fromkeys(calls, 0))
                pool.copy_(pool_init)
                with model_extra_attrs(extra):
                    eager = mixer(x, md).clone()
                taken[cfg, "eager"] = dict(calls)
                calls.update(dict.fromkeys(calls, 0))
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with with_multi_stream(True), torch.cuda.stream(s), model_extra_attrs(extra):
                    mixer(x, md)
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with (
                    with_multi_stream(True),
                    torch.cuda.graph(graph, stream=s),
                    model_extra_attrs(extra),
                ):
                    out = mixer(x, md)
                taken[cfg, "graph"] = dict(calls)
                replays = []
                for _ in range(3):
                    pool.copy_(pool_init)
                    graph.replay()
                    torch.cuda.synchronize()
                    replays.append(out.clone())
                results[cfg] = (eager, replays)
    finally:
        manager.shutdown()
    # Eager runs and graph replays are compared separately: the decode op may pick another
    # tactic outside a capture. The default path's replays must agree among themselves too.
    ref_eager, ref_graph = results[(False,) * 7][0], results[(False,) * 7][1][0]
    assert torch.isfinite(ref_graph).all() and ref_graph.abs().max() > 0
    assert torch.isfinite(ref_eager).all() and ref_eager.abs().max() > 0
    for (cfg, mode), counts in taken.items():
        gate, qnorm, kv, _, kv_fused, _, _ = cfg
        # The eager runs are single-stream: the KV half stays on the main stream there.
        side = kv and mode == "graph"
        expect = {
            "gated": gate,
            "sigmoid": gate,
            "q_norm": qnorm,
            "kv_side": side,
            "kv_fused": side and kv_fused,
        }
        assert {k: v > 0 for k, v in counts.items()} == expect, (cfg, mode, counts)
    for cfg, (eager, replays) in results.items():
        assert torch.equal(eager, ref_eager), f"eager {cfg}"
        for r in replays:
            assert torch.equal(r, ref_graph), f"graph {cfg}"


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_mxfp8_q_a_input_matches_bf16_input(batch, monkeypatch):
    """The fused q_a / kv_a projection on the MXFP8 form of the layer input (``next_input_mxfp8``,
    as the decoder layer hands it over) gives the BF16 input's output bit for bit, eagerly and in
    CUDA graphs, on the default path and with every side-stream switch."""
    mx = pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_fp8w_lowm_mxfp8")
    import weakref

    from tensorrt_llm._torch.models import modeling_kimi_linear as mk

    if not hasattr(mk._Fp8BlockScaleWeightReadLinear, "forward_mxfp8"):
        pytest.skip("needs _Fp8BlockScaleWeightReadLinear.forward_mxfp8")

    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.utils import model_extra_attrs

    q_len = 8 if batch == 4 else 7
    model, mixer, manager, md, x = _build(batch, q_len, 448, monkeypatch)
    extra = {
        "attention_metadata": weakref.ref(md),
        "mla_layers": {mixer.layer_idx_str: weakref.ref(mixer)},
    }
    pool = manager.get_buffers(0)
    pool_init = pool.clone()
    assert mixer.q_a_takes_mxfp8(x.shape[0])
    xq, xs = mx.mxfp8_quantize_reference(x)
    calls = [0]
    forward_mxfp8 = mixer.kv_a_proj_with_mqa.forward_mxfp8

    def counted(*args, **kwargs):
        calls[0] += 1
        return forward_mxfp8(*args, **kwargs)

    monkeypatch.setattr(mixer.kv_a_proj_with_mqa, "forward_mxfp8", counted)
    results = {}
    try:
        with torch.inference_mode():
            for cfg in [(False,) * 4, (True,) * 4]:
                _set(monkeypatch, k3mla, *cfg)
                for use_mx in (False, True):

                    def run():
                        if use_mx:
                            mixer.next_input_mxfp8 = (xq, xs)
                        return mixer(x, md)

                    pool.copy_(pool_init)
                    with model_extra_attrs(extra):
                        eager = run().clone()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with with_multi_stream(True), torch.cuda.stream(s), model_extra_attrs(extra):
                        run()
                    torch.cuda.current_stream().wait_stream(s)
                    graph = torch.cuda.CUDAGraph()
                    with (
                        with_multi_stream(True),
                        torch.cuda.graph(graph, stream=s),
                        model_extra_attrs(extra),
                    ):
                        out = run()
                    pool.copy_(pool_init)
                    graph.replay()
                    torch.cuda.synchronize()
                    results[cfg, use_mx] = (eager, out.clone())
                    assert mixer.next_input_mxfp8 is None
    finally:
        manager.shutdown()
    assert calls[0] == 6  # eager, warm-up and capture of the two MXFP8 runs
    for cfg in [(False,) * 4, (True,) * 4]:
        (eager, graph), (eager_mx, graph_mx) = results[cfg, False], results[cfg, True]
        assert torch.isfinite(graph).all() and graph.abs().max() > 0
        assert torch.equal(eager_mx, eager), f"eager {cfg}"
        assert torch.equal(graph_mx, graph), f"graph {cfg}"


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_fused_v_up_matches_bmm(batch, monkeypatch):
    """``helix_v_up_in_attention`` (TRTLLM_K3_MLA_FUSED_V_UP): the CuTe DSL decode applies the V
    up-projection in its reduction; the layer output is bitwise the BMM path's, eagerly and in
    CUDA graphs, on the default path and with every side-stream switch (with the fused V
    up-projection the output gate is enqueued late by default, early with
    TRTLLM_K3_MLA_GATE_EARLY)."""
    import weakref

    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.utils import model_extra_attrs

    monkeypatch.setenv("TRTLLM_MLA_PARTIAL_BF16", "1")
    q_len = 8 if batch == 4 else 7
    model, mixer, manager, md, x = _build(batch, q_len, 448, monkeypatch)
    extra = {
        "attention_metadata": weakref.ref(md),
        "mla_layers": {mixer.layer_idx_str: weakref.ref(mixer)},
    }
    pool = manager.get_buffers(0)
    pool_init = pool.clone()
    bmm_calls = [0]
    project = mixer._helix_project_v_partials

    def counted(*args, **kwargs):
        bmm_calls[0] += 1
        return project(*args, **kwargs)

    monkeypatch.setattr(mixer, "_helix_project_v_partials", counted)
    results = {}
    try:
        with torch.inference_mode():
            for cfg in [(False,) * 7, (True,) * 4 + (False,) * 3, (True,) * 7]:
                _set(monkeypatch, k3mla, *cfg)
                for fused in (False, True):
                    mixer.helix_v_up_in_attention = fused
                    bmm_calls[0] = 0
                    pool.copy_(pool_init)
                    with model_extra_attrs(extra):
                        eager = mixer(x, md).clone()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with with_multi_stream(True), torch.cuda.stream(s), model_extra_attrs(extra):
                        mixer(x, md)
                    torch.cuda.current_stream().wait_stream(s)
                    graph = torch.cuda.CUDAGraph()
                    with (
                        with_multi_stream(True),
                        torch.cuda.graph(graph, stream=s),
                        model_extra_attrs(extra),
                    ):
                        out = mixer(x, md)
                    pool.copy_(pool_init)
                    graph.replay()
                    torch.cuda.synchronize()
                    results[cfg, fused] = (eager, out.clone())
                    # eager, warm-up and capture each project V once on the BMM path, never fused
                    assert bmm_calls[0] == (0 if fused else 3), (cfg, fused, bmm_calls[0])
    finally:
        manager.shutdown()
    ref_graph = results[(False,) * 7, False][1]
    for cfg in [(False,) * 7, (True,) * 4 + (False,) * 3, (True,) * 7]:
        (eager, graph), (eager_f, graph_f) = results[cfg, False], results[cfg, True]
        assert torch.equal(graph, ref_graph), f"graph {cfg} vs default"
        assert torch.isfinite(graph).all() and graph.abs().max() > 0
        assert torch.equal(eager_f, eager), f"eager {cfg}"
        assert torch.equal(graph_f, graph), f"graph {cfg}"


def _kv_half_inputs(num_tokens, seed):
    """Raw kv_a projection rows ``[compressed_kv | k_pe]`` with signed zeros, values the E4M3
    conversion saturates and E4M3 subnormals mixed in."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn((num_tokens, 2112), device="cuda", generator=gen) * 2.0
    special = torch.tensor([0.0, -0.0, 600.0, -900.0, 1e-3, -2e-3, 448.0, -448.0], device="cuda")
    pick = torch.randint(0, 9, raw.shape, device="cuda", generator=gen)
    raw = torch.where(pick < 8, special[pick.clamp(max=7)], raw)
    # Some rows tiny or huge as a whole (the RMS scale is large / small there).
    raw[0] *= 1e-4
    raw[-1] *= 30.0
    raw = raw.to(torch.bfloat16)
    return raw[:, 1536:2048], raw[:, 2048:2112]


@pytest.mark.parametrize("helix", ["none", "inactive", "slots"])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_kv_append_matches_rope_generation(batch, helix, monkeypatch):
    """``k3_mla_kv_append`` (TRTLLM_K3_MLA_KV_FUSED) writes the latent row, the paged FP8 cache
    rows and the decode's scale / scheduler buffers bit for bit like the FlashInfer RMSNorm, the
    k_pe copy and ``mla_rope_generation(q_rope_applied=True)`` of the default KV half, for every
    Helix write mode (none, per-request inactive flags, per-token slots with positions)."""
    q_len = 8 if batch == 4 else 7
    kv_len = 448
    model, mixer, manager, md, _ = _build(batch, q_len, kv_len, monkeypatch)
    num_tokens = batch * q_len
    dev = torch.device("cuda")
    if helix == "inactive":
        md.helix_position_offsets = torch.randint(
            0, kv_len, (num_tokens,), dtype=torch.int32, device=dev
        )
        md.helix_is_inactive_rank = torch.tensor(
            [b % 2 == 1 for b in range(batch)], dtype=torch.bool, device=dev
        )
    elif helix == "slots":
        md.helix_position_offsets = torch.randint(
            0, kv_len, (num_tokens,), dtype=torch.int32, device=dev
        )
        md.helix_is_inactive_rank = torch.ones(batch, dtype=torch.bool, device=dev)
        slots = torch.arange(num_tokens, dtype=torch.int32, device=dev) % q_len + kv_len
        slots[1::3] = -1
        # Rows inside the cached context, distinct per request (no two tokens share a row).
        slots[2::5] = (torch.arange(2, num_tokens, 5, dtype=torch.int32, device=dev) * 37) % kv_len
        md.helix_local_slots = slots
        md._helix_spec_tokens_valid = True
    pool = manager.get_buffers(0)
    pool_init = pool.clone()
    heads = mixer.num_heads_tp

    def buffers():
        return (
            torch.full((batch + 1,), -7, dtype=torch.int32, device=dev),
            torch.full((batch + 1,), -7, dtype=torch.int32, device=dev),
            torch.full((1,), 7, dtype=torch.uint32, device=dev),
            torch.full((2,), -7.0, dtype=torch.float32, device=dev),
            torch.full((1,), -7.0, dtype=torch.float32, device=dev),
        )

    try:
        with torch.inference_mode():
            for seed in range(3):
                compressed_kv, k_pe = _kv_half_inputs(num_tokens, seed)
                assert mixer._kv_fused_applies(compressed_kv, k_pe, md) is False
                monkeypatch.setattr(
                    "tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention._K3_MLA_KV_FUSED",
                    True,
                )
                assert mixer._kv_fused_applies(compressed_kv, k_pe, md)
                # Default KV half.
                pool.copy_(pool_init)
                ref = buffers()
                latent_ref = mixer._latent_row(compressed_kv, k_pe)
                fused_q = torch.empty((num_tokens, heads, 576), dtype=torch.bfloat16, device=dev)
                quant_q = torch.zeros((num_tokens, heads, 576), dtype=torch.uint8, device=dev)
                q_pe_shape = torch.empty(
                    (num_tokens, heads, 192), dtype=torch.bfloat16, device=dev
                )[..., 128:]
                mixer.mqa.mla_rope_generation(
                    fused_q,
                    q_pe_shape,
                    latent_ref,
                    md,
                    *ref,
                    quant_q,
                    quant_scale_qkv=mixer._fp8_q_unit_scale,
                    q_rope_applied=True,
                )
                torch.cuda.synchronize()
                pool_ref = pool.clone()
                # Fused kernel.
                pool.copy_(pool_init)
                got = buffers()
                latent = mixer._kv_append_fused(compressed_kv, k_pe, md, *got)
                torch.cuda.synchronize()
                assert torch.equal(latent.view(torch.int16), latent_ref.view(torch.int16)), seed
                assert torch.equal(pool.view(torch.uint8), pool_ref.view(torch.uint8)), seed
                # The default path only writes the scheduler slots of the batch.
                for a, b in zip(got, ref):
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (seed, a, b)
                assert not torch.equal(pool.view(torch.uint8), pool_init.view(torch.uint8))
                monkeypatch.setattr(
                    "tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention._K3_MLA_KV_FUSED",
                    False,
                )
    finally:
        manager.shutdown()

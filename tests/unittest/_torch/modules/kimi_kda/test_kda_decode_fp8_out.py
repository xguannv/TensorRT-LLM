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
"""Kimi-K3 KDA verify emitting the DeepGEMM FP8 o_proj operand (TRTLLM_K3_KDA_DECODE_FP8_OUT).

``kda_mtp_decode_impl(..., two_cta=True, fp8_out=True)`` quantizes the gated-RMSNorm output in the
two-CTA kernel's epilogue. Its E4M3 values and MN-major packed UE8M0 scales must equal, bit for
bit, ``fp8_quantize_1x128_packed_ue8m0`` applied to the BF16 output of the same call without
``fp8_out``, and the recurrent state, conv windows and replay caches must be untouched by the
switch, over several verify rounds with mixed acceptance, eagerly and under CUDA graph replay. The
mixer must route multi-wave verify calls whose o_proj runs DeepGEMM to it only when the switch is
on, with a bitwise identical o_proj output.
"""

import pytest
import torch

from tests.unittest._torch.modules.kimi_kda.test_kda_decode_fused_fab import (
    MUTATED,
    _assert_same,
    _clone,
    _decode_kwargs,
    _layer,
    _next_acceptance,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)


def _impl():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import kda_mtp_decode_impl

    return kda_mtp_decode_impl


def _layer_bf16(batch, num_spec, seed):
    lay = _layer(batch, num_spec, seed=seed)
    lay["state"] = lay["state"].to(torch.bfloat16)
    return lay


def _gate(num_rows):
    g = torch.Generator(device="cpu").manual_seed(num_rows)
    gate = (torch.randn(num_rows, 6 * 128, generator=g) * 0.5).bfloat16().cuda()
    beta = torch.randn(num_rows, 6, generator=g).bfloat16().cuda()
    return gate, beta


def _quantized_reference(out_bf16):
    rows = out_bf16.shape[1]
    return torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(out_bf16.view(rows, -1), False)


def _assert_fp8_pair(name, actual, expected):
    values, scales = actual
    ref_values, ref_scales = expected
    _assert_same(f"{name} values", values.view(torch.uint8), ref_values.view(torch.uint8))
    assert scales.shape == ref_scales.shape and scales.stride() == ref_scales.stride(), name
    _assert_same(f"{name} scales", scales, ref_scales)
    # the TMA padding rows past the last token are zero, as the quantize kernel leaves them
    rows, lead = scales.shape[0], scales.stride(1)
    buf = scales.as_strided((scales.shape[1], lead), (lead, 1))
    assert not buf[:, rows:].any(), name


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("batch,num_spec", [(26, 2), (32, 6), (64, 7), (128, 6), (256, 6)])
@torch.no_grad()
def test_fp8_out_bitwise(batch, num_spec, use_pdl):
    lay = _layer_bf16(batch, num_spec, 800 + batch + num_spec)
    gate, beta = _gate(batch * (num_spec + 1))
    ref, fp8 = _clone(lay), _clone(lay)
    for rnd in range(3):
        kw = _decode_kwargs(ref, batch, num_spec, g=gate, beta=beta, use_pdl=use_pdl)
        out_ref = _impl()(**kw, two_cta=True)
        kw = _decode_kwargs(fp8, batch, num_spec, g=gate, beta=beta, use_pdl=use_pdl)
        out_fp8 = _impl()(**kw, two_cta=True, fp8_out=True)
        torch.cuda.synchronize()
        assert torch.isfinite(out_ref.float()).all()
        _assert_fp8_pair(f"round {rnd}", out_fp8, _quantized_reference(out_ref))
        for key in MUTATED:
            _assert_same(f"round {rnd} {key}", fp8[key], ref[key])
        for state in (ref, fp8):
            _next_acceptance(state, batch, num_spec, seed=900 + rnd)


@torch.no_grad()
def test_fp8_out_cuda_graph():
    batch, num_spec = 128, 6
    lay = _layer_bf16(batch, num_spec, 31)
    gate, beta = _gate(batch * (num_spec + 1))
    ref, graphed = _clone(lay), _clone(lay)
    stream = torch.cuda.Stream()

    def step(state):
        kw = _decode_kwargs(state, batch, num_spec, g=gate, beta=beta, use_pdl=True)
        return _impl()(**kw, two_cta=True, fp8_out=True)

    scratch = _clone(lay)
    with torch.cuda.stream(stream):
        step(scratch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out_graph = step(graphed)
    torch.cuda.synchronize()
    for rnd in range(3):
        out_ref = _impl()(**_decode_kwargs(ref, batch, num_spec, g=gate, beta=beta), two_cta=True)
        graph.replay()
        torch.cuda.synchronize()
        _assert_fp8_pair(f"replay {rnd}", out_graph, _quantized_reference(out_ref))
        for key in MUTATED:
            _assert_same(f"replay {rnd} {key}", graphed[key], ref[key])
        for state in (ref, graphed):
            _next_acceptance(state, batch, num_spec, seed=950 + rnd)


@torch.no_grad()
def test_mixer_routes_fp8_out_to_o_proj(monkeypatch):
    """forward_verify_fused with the switch on: the FP8-output verify for a multi-wave grid whose
    FP8 block-scale o_proj runs DeepGEMM, bitwise the same o_proj output and pools as with the
    switch off."""
    from tensorrt_llm._torch.modules import linear as linear_module
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer
    from tensorrt_llm._torch.modules.linear import Linear
    from tests.unittest._torch.modules.kimi_kda.test_kimi_kda_fused_verify_parity import (
        HIDDEN,
        H,
        M,
        _make_fused_layer_cache,
        _make_pools,
        _make_runtime,
    )

    monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_2CTA", True)
    monkeypatch.setattr(linear_module, "_FP8W_LOWM_GEMM", False)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    batch = num_sms // H + 4  # more (head, request) tiles than SMs
    steps = M + 1
    runtimes, pools, caches = [], [], []
    for _ in range(2):
        rt = _make_runtime(seed=1, checkpoint_fp8=True)
        rt.finalize_decode_weights()
        for module in rt.modules():
            if isinstance(module, Linear):
                module.post_load_weights()
        conv_pool, ssm_pool = _make_pools(batch, seed=2)
        runtimes.append(rt)
        pools.append(ssm_pool.to(torch.bfloat16))
        caches.append(_make_fused_layer_cache(batch, conv_pool))
    assert kimi_kda_mixer._o_proj_takes_prequantized_fp8(runtimes[1].o_proj, batch * steps)
    calls = []
    verify_fp8 = runtimes[1]._dispatch.mtp_verify_fp8_packed

    def _capturing(**kwargs):
        calls.append(kwargs)
        return verify_fp8(**kwargs)

    runtimes[1]._dispatch.mtp_verify_fp8_packed = _capturing
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(3)
    for rnd in range(2):
        x = (torch.randn(batch * steps, HIDDEN, generator=gen, device="cuda") * 0.5).bfloat16()
        outs = []
        for i, (rt, ssm, cache) in enumerate(zip(runtimes, pools, caches)):
            monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_FP8_OUT", i == 1)
            outs.append(rt.forward_verify_fused(x, steps, cache, ssm, slots))
        assert len(calls) == rnd + 1 and calls[-1]["two_cta"]
        _assert_same(f"round {rnd} out", outs[1], outs[0])
        _assert_same(f"round {rnd} state", pools[1], pools[0])
        accept = torch.randint(0, M + 1, (batch,), generator=gen, device="cuda", dtype=torch.int32)
        for cache in caches:
            cache.prev_num_accepted_tokens.copy_(accept)


@pytest.mark.parametrize("mid_gemm", [True, False])
@torch.no_grad()
def test_mixer_routes_fp8_out_to_k3_weight_read_o_proj(monkeypatch, mid_gemm):
    """With the Kimi-K3 FP8 weight-read o_proj (as the model swaps it in): the FP8-output verify
    feeds its packed operand to the mid-M FP8 GEMM where the ``o`` family applies, else to the
    prequantized DeepGEMM GEMM; bitwise the same o_proj output as the BF16 verify + o_proj."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mk
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer
    from tensorrt_llm._torch.modules.linear import Linear
    from tests.unittest._torch.modules.kimi_kda.test_kimi_kda_fused_verify_parity import (
        HIDDEN,
        H,
        M,
        _make_fused_layer_cache,
        _make_pools,
        _make_runtime,
    )

    monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_2CTA", True)
    monkeypatch.setattr(mk, "_K3_FP8W_LOWM_GEMM", False)
    monkeypatch.setattr(mk, "_K3_FP8_MID_FAMILIES", frozenset({"o"} if mid_gemm else ()))
    monkeypatch.setattr(mk, "_K3_FP8_MID_MIN_M", 1)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    batch = num_sms // H + 4  # more (head, request) tiles than SMs
    steps = M + 1
    runtimes, pools, caches = [], [], []
    for _ in range(2):
        rt = _make_runtime(seed=1, checkpoint_fp8=True)
        rt.finalize_decode_weights()
        # As the model loader does: swap the checkpoint FP8 o_proj, then post-load everything.
        assert mk._swap_linear_to_fp8_weight_read(rt, "o_proj", (Linear,)) == 1
        assert rt.o_proj.k3_mid_family == "o"
        for module in rt.modules():
            if isinstance(module, Linear):
                module.post_load_weights()
        rt.o_proj.post_load_weights()
        assert rt.o_proj.prepare_mid_gemm() == mid_gemm
        conv_pool, ssm_pool = _make_pools(batch, seed=2)
        runtimes.append(rt)
        pools.append(ssm_pool.to(torch.bfloat16))
        caches.append(_make_fused_layer_cache(batch, conv_pool))
    rows = batch * steps
    assert kimi_kda_mixer._o_proj_takes_prequantized_fp8(runtimes[1].o_proj, rows)
    assert runtimes[1].o_proj._mid_gemm_applies_rows(rows, packed=True) == mid_gemm
    packed_calls = []
    forward_fp8_packed = runtimes[1].o_proj.forward_fp8_packed

    def _capturing(act, act_scale):
        packed_calls.append((act.shape, act_scale.stride()))
        return forward_fp8_packed(act, act_scale)

    runtimes[1].o_proj.forward_fp8_packed = _capturing
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(5)
    for rnd in range(2):
        x = (torch.randn(rows, HIDDEN, generator=gen, device="cuda") * 0.5).bfloat16()
        outs = []
        for i, (rt, ssm, cache) in enumerate(zip(runtimes, pools, caches)):
            monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_FP8_OUT", i == 1)
            outs.append(rt.forward_verify_fused(x, steps, cache, ssm, slots))
        assert len(packed_calls) == rnd + 1
        _assert_same(f"round {rnd} out", outs[1], outs[0])
        _assert_same(f"round {rnd} state", pools[1], pools[0])
        accept = torch.randint(0, M + 1, (batch,), generator=gen, device="cuda", dtype=torch.int32)
        for cache in caches:
            cache.prev_num_accepted_tokens.copy_(accept)

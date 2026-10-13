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
"""Kimi-K3 KDA verify on the two-CTAs-per-SM kernel (TRTLLM_K3_KDA_DECODE_2CTA).

``kda_mtp_decode_impl(..., two_cta=True)`` must reproduce the one-CTA v2 kernel bit for bit
(outputs, committed recurrent state, conv windows, replay caches) over several verify rounds with
mixed acceptance, for fp32 and bf16 recurrent states, with and without PDL, eagerly and under CUDA
graph replay; the mixer must route multi-wave verify calls to it only when the switch is on and
the result must not change.
"""

import pytest
import torch

from tests.unittest._torch.modules.kimi_kda.test_kda_decode_fused_fab import (
    MUTATED,
    N1,
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


def _layer_with_state(batch, num_spec, seed, state_dtype):
    lay = _layer(batch, num_spec, seed=seed)
    lay["state"] = lay["state"].to(state_dtype)
    return lay


def _gate(lay, num_rows):
    g = torch.Generator(device="cpu").manual_seed(num_rows)
    gate = (torch.randn(num_rows, 6 * 128, generator=g) * 0.5).bfloat16().cuda()
    beta = torch.randn(num_rows, 6, generator=g).bfloat16().cuda()
    return gate, beta


STATE_DTYPES = [
    pytest.param(torch.float32, id="fp32_state"),
    pytest.param(torch.bfloat16, id="bf16_state"),
]


@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("batch,num_spec", [(32, 6), (32, 7), (64, 5), (128, 6), (26, 2)])
@torch.no_grad()
def test_two_cta_verify_bitwise(batch, num_spec, use_pdl, state_dtype):
    lay = _layer_with_state(batch, num_spec, 200 + batch + num_spec, state_dtype)
    gate, beta = _gate(lay, batch * (num_spec + 1))
    ref, two = _clone(lay), _clone(lay)
    for rnd in range(3):
        out_ref = _impl()(**_decode_kwargs(ref, batch, num_spec, g=gate, beta=beta))
        kw = _decode_kwargs(two, batch, num_spec, g=gate, beta=beta, use_pdl=use_pdl)
        out_two = _impl()(**kw, two_cta=True)
        torch.cuda.synchronize()
        assert torch.isfinite(out_ref.float()).all()
        _assert_same(f"round {rnd} out", out_two, out_ref)
        for key in MUTATED:
            _assert_same(f"round {rnd} {key}", two[key], ref[key])
        for state in (ref, two):
            _next_acceptance(state, batch, num_spec, seed=300 + rnd)


@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@torch.no_grad()
def test_two_cta_cuda_graph(state_dtype):
    batch, num_spec = 64, 7
    lay = _layer_with_state(batch, num_spec, 17, state_dtype)
    gate, beta = _gate(lay, batch * (num_spec + 1))
    ref, graphed = _clone(lay), _clone(lay)
    stream = torch.cuda.Stream()

    def step(state):
        kw = _decode_kwargs(state, batch, num_spec, g=gate, beta=beta, use_pdl=True)
        return _impl()(**kw, two_cta=True)

    scratch = _clone(lay)
    with torch.cuda.stream(stream):
        step(scratch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out_graph = step(graphed)
    torch.cuda.synchronize()
    for rnd in range(3):
        out_ref = _impl()(**_decode_kwargs(ref, batch, num_spec, g=gate, beta=beta))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(f"replay {rnd} out", out_graph, out_ref)
        for key in MUTATED:
            _assert_same(f"replay {rnd} {key}", graphed[key], ref[key])
        for state in (ref, graphed):
            _next_acceptance(state, batch, num_spec, seed=400 + rnd)


@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@torch.no_grad()
def test_two_cta_falls_back_on_fused_forget_gate(state_dtype):
    """The two-CTA kernel takes the forget gate from the projection only; a call with the fused
    forget gate runs the v2 kernel unchanged."""
    batch, num_spec = 32, 6
    lay = _layer_with_state(batch, num_spec, 5, state_dtype)
    m = batch * (num_spec + 1)
    g = torch.Generator(device="cpu").manual_seed(m)
    bfa = (torch.randn(m, N1, generator=g) * 0.5).bfloat16().cuda()
    ref, two = _clone(lay), _clone(lay)
    out_ref = _impl()(**_decode_kwargs(ref, batch, num_spec, bfa=bfa, use_pdl=True))
    kw = _decode_kwargs(two, batch, num_spec, bfa=bfa, use_pdl=True)
    out_two = _impl()(**kw, two_cta=True)
    _assert_same("out", out_two, out_ref)
    for key in MUTATED:
        _assert_same(key, two[key], ref[key])


@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@torch.no_grad()
def test_mixer_routes_multi_wave_verify(monkeypatch, state_dtype):
    """forward_verify_fused with the switch on: the two-CTA kernel for a multi-wave grid, bitwise
    the same outputs and pools as with the switch off."""
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer
    from tests.unittest._torch.modules.kimi_kda.test_kimi_kda_fused_verify_parity import (
        HIDDEN,
        H,
        M,
        _make_fused_layer_cache,
        _make_pools,
        _make_runtime,
    )

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    batch = num_sms // H + 4  # more (head, request) tiles than SMs
    steps = M + 1
    runtimes, pools, caches = [], [], []
    for _ in range(2):
        rt = _make_runtime(seed=1)
        rt.finalize_decode_weights()
        conv_pool, ssm_pool = _make_pools(batch, seed=2)
        runtimes.append(rt)
        pools.append(ssm_pool.to(state_dtype))
        caches.append(_make_fused_layer_cache(batch, conv_pool))
    calls = []
    verify = runtimes[1]._dispatch.mtp_verify

    def _capturing(**kwargs):
        calls.append(kwargs)
        return verify(**kwargs)

    runtimes[1]._dispatch.mtp_verify = _capturing
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(3)
    for rnd in range(2):
        x = (torch.randn(batch * steps, HIDDEN, generator=gen, device="cuda") * 0.5).bfloat16()
        outs = []
        for i, (rt, ssm, cache) in enumerate(zip(runtimes, pools, caches)):
            monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_2CTA", i == 1)
            outs.append(rt.forward_verify_fused(x, steps, cache, ssm, slots))
        assert calls[-1]["two_cta"]
        _assert_same(f"round {rnd} out", outs[1], outs[0])
        _assert_same(f"round {rnd} state", pools[1], pools[0])
        accept = torch.randint(0, M + 1, (batch,), generator=gen, device="cuda", dtype=torch.int32)
        for cache in caches:
            cache.prev_num_accepted_tokens.copy_(accept)

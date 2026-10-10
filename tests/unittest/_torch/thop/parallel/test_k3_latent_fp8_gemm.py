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
"""Bit-exactness of the fused Kimi-K3 latent projections (``k3_latent_fp8_gemm``).

``trtllm::k3_latent_down_fp4`` is checked against the unfused path it replaces: 1x128 UE8M0 FP8
activation quantization, the DeepGEMM FP8 GEMM with the resmoothed weight and its packed UE8M0
scale (BF16 output), then ``fp4_quantize`` with the experts' global scale and linear scale
layout; codes and scales must match byte for byte. ``trtllm::k3_latent_up_residual`` is checked
against the same GEMM followed by the two BF16 adds of the MoE residual update.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

N, K = 3584, 7168


def _module():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_latent_fp8_gemm

    return k3_latent_fp8_gemm


def _weight(n, k, seed):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    return _Fp8BlockScaleWeightReadLinear.quantize_weight(w)


def _activation(m, k, seed):
    """Hidden-state-like rows: per-channel log-normal scales, outlier channels, a zero block."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    ch = torch.exp(torch.randn(k, device="cuda", generator=gen) * 0.7)
    ch[torch.randperm(k, device="cuda", generator=gen)[:16]] *= 30
    x = (torch.randn((m, k), device="cuda", generator=gen) * ch).to(torch.bfloat16)
    x[0, :128] = 0
    return x


def _reference(x, w, w_scale, global_sf):
    from tensorrt_llm import deep_gemm

    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    deep_gemm.fp8_gemm_nt((a, a_sf), (w, w_scale), y, disable_ue8m0_cast=True)
    q, sf = torch.ops.trtllm.fp4_quantize(y, global_sf, 16, False, False)
    m, n = y.shape
    return y, q.view(torch.uint8).view(m, n // 2), sf.view(torch.uint8).reshape(-1)[: m * n // 16]


@pytest.mark.parametrize("m", [1, 7, 100, 224, 300, 448, 672, 896, 1000, 1792])
def test_bit_exact_vs_unfused(m):
    ld = _module()
    w, w_scale = _weight(N, K, seed=m)
    x = _activation(m, K, seed=m)
    y, _, _ = _reference(x, w, w_scale, torch.ones(1, device="cuda"))
    global_sf = torch.tensor([448.0 * 6.0 / y.abs().amax().float().item()], device="cuda")
    _, q_ref, sf_ref = _reference(x, w, w_scale, global_sf)
    assert ld.can_run(x, w, w_scale, global_sf)
    q, sf = torch.ops.trtllm.k3_latent_down_fp4(x, w, w_scale, global_sf)
    assert q.dtype == torch.uint8 and q.shape == (m, N // 2)
    assert sf.dtype == torch.uint8 and sf.shape == (m, N // 16)
    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(sf.reshape(-1), sf_ref, rtol=0, atol=0)


@pytest.mark.parametrize("bn", [128, 192, 256])
@pytest.mark.parametrize("m", [96, 896])
def test_column_tiles(m, bn):
    """Every column tile (including the partial last 192-wide tile) is bit-exact."""
    ld = _module()
    w, w_scale = _weight(N, K, seed=7)
    x = _activation(m, K, seed=3)
    global_sf = torch.tensor([150.0], device="cuda")
    _, q_ref, sf_ref = _reference(x, w, w_scale, global_sf)
    q = torch.empty((m, N // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty((m, N // 16), dtype=torch.uint8, device="cuda")
    ld.run_k3_latent_down_fp4(x, w, w_scale, global_sf, q, sf, bn=bn)
    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(sf.reshape(-1), sf_ref, rtol=0, atol=0)


def test_activation_quant_matches_triton():
    """The 1x128 activation quantization equals the Triton kernel bit for bit."""
    from tensorrt_llm.quantization.utils import fp8_quantize

    ld = _module()
    m = 300
    x = _activation(m, K, seed=11)
    # Amax mantissas right at the 1.75 * 2^E rounding boundary of the UE8M0 scale.
    x[1, 128:256] = 1.75 * 2.0**-3
    x[2, 256:384] = torch.tensor(1.7578125 * 2.0**5, dtype=torch.bfloat16)
    m_tiles = (m + 127) // 128
    xq = torch.empty((m, K), dtype=torch.float8_e4m3fn, device="cuda")
    xs = torch.empty((K // 128) * m_tiles * 512, dtype=torch.uint8, device="cuda")
    ld.quantize_act_1x128(x, xq, xs)
    a_ref, s_ref = fp8_quantize.triton_fp8_quantize_1x128(x, use_ue8m0=True)
    torch.testing.assert_close(xq.view(torch.uint8), a_ref.view(torch.uint8), rtol=0, atol=0)
    # Atom bytes [K/128][m_tiles][(r % 32) * 16 + (r // 32) * 4 + k32] -> [m, K/32].
    atoms = xs.view(K // 128, m_tiles, 32, 4, 4).permute(1, 3, 2, 0, 4)
    sf = atoms.reshape(m_tiles * 128, K // 32)[:m]
    exp_ref = ((s_ref.t().contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    torch.testing.assert_close(sf, exp_ref.repeat_interleave(4, dim=1), rtol=0, atol=0)


def test_cuda_graph_capture():
    ld = _module()
    m = 448
    w, w_scale = _weight(N, K, seed=5)
    x = _activation(m, K, seed=5)
    global_sf = torch.tensor([150.0], device="cuda")
    q0, sf0 = torch.ops.trtllm.k3_latent_down_fp4(x, w, w_scale, global_sf)
    q = torch.empty_like(q0)
    sf = torch.empty_like(sf0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ld.run_k3_latent_down_fp4(x, w, w_scale, global_sf, q, sf)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(q, q0, rtol=0, atol=0)
    torch.testing.assert_close(sf, sf0, rtol=0, atol=0)


def test_can_run_rejects():
    ld = _module()
    w, w_scale = _weight(N, K, seed=1)
    x = _activation(4, K, seed=1)
    gsf = torch.tensor([1.0], device="cuda")
    assert not ld.can_run(x.float(), w, w_scale, gsf)
    assert not ld.can_run(x, w, w_scale.contiguous().t().contiguous(), gsf)
    assert not ld.can_run(x, w, w_scale, None)
    assert not ld.can_run(x[:, :1024], w, w_scale, gsf)


def _deepgemm_bf16(x, w, w_scale):
    from tensorrt_llm import deep_gemm

    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    deep_gemm.fp8_gemm_nt((a, a_sf), (w, w_scale), y, disable_ue8m0_cast=True)
    return y


@pytest.mark.parametrize("m", [1, 100, 224, 448, 896, 1000])
def test_latent_up_residual_bit_exact(m):
    """``residual + (latent_up(x) + shared)`` equals the unfused GEMM + two BF16 adds."""
    ld = _module()
    n, k = 7168, 3584
    w, w_scale = _weight(n, k, seed=m + 100)
    x = _activation(m, k, seed=m + 100)
    gen = torch.Generator(device="cuda").manual_seed(m)
    shared = (torch.randn((m, n), device="cuda", generator=gen) * 4).to(torch.bfloat16)
    residual = (torch.randn((m, n), device="cuda", generator=gen) * 30).to(torch.bfloat16)
    ref = residual + (_deepgemm_bf16(x, w, w_scale) + shared)
    assert ld.can_run_residual(x, w, w_scale, shared, residual)
    out = torch.ops.trtllm.k3_latent_up_residual(x, w, w_scale, shared, residual)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("bn", [128, 192, 256])
def test_latent_up_residual_tiles(bn):
    """Persistent schedules (several tiles per CTA) of every column tile are bit-exact."""
    ld = _module()
    n, k, m = 7168, 3584, 1100
    w, w_scale = _weight(n, k, seed=bn)
    x = _activation(m, k, seed=bn)
    shared = torch.randn((m, n), device="cuda").to(torch.bfloat16)
    residual = torch.randn((m, n), device="cuda").to(torch.bfloat16)
    ref = residual + (_deepgemm_bf16(x, w, w_scale) + shared)
    out = torch.empty_like(residual)
    ld.run_k3_latent_up_residual(x, w, w_scale, shared, residual, out, bn=bn)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

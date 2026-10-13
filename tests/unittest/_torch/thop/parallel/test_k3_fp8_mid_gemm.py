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
"""Bit-exactness of the Kimi-K3 mid-M FP8 GEMM (``k3_fp8_mid_gemm``).

The BF16 output is checked byte for byte against the path it replaces (1x128 UE8M0 activation
quantization + DeepGEMM with the resmoothed weight and its packed UE8M0 scale), and the residual
epilogue against that GEMM followed by the two BF16 adds of the MoE residual update.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

# Per-rank K3 decode projections (N, K): latent_up, o_proj, MLA q_b, MLA fused q_a/kv_a.
SHAPES = [(7168, 3584), (7168, 768), (18432, 1536), (2112, 7168)]


def _mid():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8_mid_gemm

    return k3_fp8_mid_gemm


def _weight(n, k, seed):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    if n % 128:
        # Checkpoint-style weight whose last 128-row block is partial (DeepGEMM-packed scale).
        from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

        wf, ws = per_block_cast_to_fp8(w, use_ue8m0=False)
        return _Fp8BlockScaleWeightReadLinear.prepare_checkpoint_scale(wf, ws)
    return _Fp8BlockScaleWeightReadLinear.quantize_weight(w)


def _activation(m, k, seed):
    """Hidden-state-like rows: per-channel log-normal scales, outlier channels, a zero block."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    ch = torch.exp(torch.randn(k, device="cuda", generator=gen) * 0.7)
    ch[torch.randperm(k, device="cuda", generator=gen)[:16]] *= 30
    x = (torch.randn((m, k), device="cuda", generator=gen) * ch).to(torch.bfloat16)
    x[0, :128] = 0
    return x


def _deepgemm_bf16(x, w, w_scale):
    from tensorrt_llm import deep_gemm

    a = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    deep_gemm.fp8_gemm_nt(a, (w, w_scale), y, disable_ue8m0_cast=True)
    return y


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 100, 224, 448, 896, 1000, 1792])
def test_bit_exact_vs_deepgemm(n, k, m):
    mid = _mid()
    w, w_scale = _weight(n, k, seed=m + n)
    x = _activation(m, k, seed=m)
    w_sf = mid.expand_weight_scale(w_scale, n, k)
    assert mid.can_run(x, w, w_sf)
    out = torch.ops.trtllm.k3_fp8_mid_gemm(x, w, w_sf)
    torch.testing.assert_close(out, _deepgemm_bf16(x, w, w_scale), rtol=0, atol=0)


@pytest.mark.parametrize("tn", [64, 128, 192, 256])
@pytest.mark.parametrize("cluster", [(2, 1), (1, 1)])
def test_tiles(tn, cluster):
    """Every token tile, 2-CTA and 1-CTA, with several tiles per CTA, is bit-exact."""
    mid = _mid()
    n, k, m = 7168, 3584, 1100
    w, w_scale = _weight(n, k, seed=tn)
    x = _activation(m, k, seed=tn)
    w_sf = mid.expand_weight_scale(w_scale, n, k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    xq, xs = mid.quantize_act(x)
    tm = 256 if cluster[0] == 2 else 128
    mid.run_k3_fp8_mid_gemm(xq, xs, w, w_sf, out, tiler=(tm, tn), cluster=cluster)
    torch.testing.assert_close(out, _deepgemm_bf16(x, w, w_scale), rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 224, 448, 896, 1000, 1792])
@pytest.mark.parametrize("tn", [None, 128, 256])
def test_residual_bit_exact(m, tn):
    """``residual + (latent_up(x) + shared)`` equals the unfused GEMM + two BF16 adds."""
    mid = _mid()
    n, k = 7168, 3584
    w, w_scale = _weight(n, k, seed=m + 7)
    x = _activation(m, k, seed=m + 7)
    gen = torch.Generator(device="cuda").manual_seed(m)
    shared = (torch.randn((m, n), device="cuda", generator=gen) * 4).to(torch.bfloat16)
    residual = (torch.randn((m, n), device="cuda", generator=gen) * 30).to(torch.bfloat16)
    ref = residual + (_deepgemm_bf16(x, w, w_scale) + shared)
    w_sf = mid.expand_weight_scale(w_scale, n, k)
    assert mid.can_run(x, w, w_sf, shared, residual)
    if tn is None:
        out = torch.ops.trtllm.k3_fp8_mid_residual(x, w, w_sf, shared, residual)
    else:
        out = torch.empty_like(residual)
        mid.k3_fp8_mid_gemm(x, w, w_sf, out, shared, residual, tiler=(256, tn), cluster=(2, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_cuda_graph_capture():
    mid = _mid()
    n, k, m = 7168, 768, 896
    w, w_scale = _weight(n, k, seed=3)
    w_sf = mid.expand_weight_scale(w_scale, n, k)
    x = _activation(m, k, seed=3)
    out = torch.ops.trtllm.k3_fp8_mid_gemm(x, w, w_sf)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = torch.ops.trtllm.k3_fp8_mid_gemm(x, w, w_sf)
    x.copy_(_activation(m, k, seed=4))
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_g, _deepgemm_bf16(x, w, w_scale), rtol=0, atol=0)
    assert not torch.equal(out_g, out)


def test_can_run_rejects():
    mid = _mid()
    n, k = 7168, 768
    w, w_scale = _weight(n, k, seed=5)
    w_sf = mid.expand_weight_scale(w_scale, n, k)
    x = torch.randn((64, k), device="cuda", dtype=torch.bfloat16)
    assert mid.can_run(x, w, w_sf)
    assert not mid.can_run(x, w, None)
    assert not mid.can_run(x.float(), w, w_sf)
    assert not mid.can_run(x[:, :640], w, w_sf)
    assert not mid.can_run(x, w, w_sf[:-512])
    assert not mid.can_run(torch.empty((0, k), device="cuda", dtype=torch.bfloat16), w, w_sf)
    shared = torch.empty((64, n), device="cuda", dtype=torch.float32)
    assert not mid.can_run(x, w, w_sf, shared, shared)


@pytest.mark.parametrize("m", [300, 896])
def test_weight_read_linear_forward(monkeypatch, m):
    """``_Fp8BlockScaleWeightReadLinear`` with its family enabled runs the mid-M GEMM bit-exactly."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mk

    monkeypatch.setattr(mk, "_K3_FP8_MID_FAMILIES", frozenset({"o"}))
    monkeypatch.setattr(mk, "_K3_FP8_MID_MIN_M", 1)
    n, k = 7168, 768
    w, w_scale = _weight(n, k, seed=m)
    lin = mk._Fp8BlockScaleWeightReadLinear(w, w_scale, n)
    lin.k3_mid_family = "o"
    assert lin.prepare_mid_gemm()
    x = _activation(m, k, seed=m)
    assert lin.mid_gemm_applies(x)
    torch.testing.assert_close(lin(x), _deepgemm_bf16(x, w, w_scale), rtol=0, atol=0)
    lin.k3_mid_family = "qb"
    lin.mid_weight_sf = None
    assert not lin.prepare_mid_gemm()


@pytest.mark.parametrize("m", [1, 128, 224, 448, 896, 1000, 1792])
@pytest.mark.parametrize("tn", [None, 64, 128, 192, 256])
def test_packed_scale_residual_bit_exact(m, tn):
    """Activation quantized upstream with DeepGEMM-packed scales (the layout of the fused MNNVL
    all-reduce + RMSNorm + FP8 block quantization) gives the unfused result bit for bit."""
    mid = _mid()
    n, k = 7168, 3584
    w, w_scale = _weight(n, k, seed=m + 11)
    x = _activation(m, k, seed=m + 11)
    gen = torch.Generator(device="cuda").manual_seed(m + 1)
    shared = (torch.randn((m, n), device="cuda", generator=gen) * 4).to(torch.bfloat16)
    residual = (torch.randn((m, n), device="cuda", generator=gen) * 30).to(torch.bfloat16)
    ref = residual + (_deepgemm_bf16(x, w, w_scale) + shared)
    w_sf = mid.expand_weight_scale(w_scale, n, k)
    xq, xs = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    assert mid.can_run_packed(xq, xs, w, w_sf, shared, residual)
    if tn is None:
        out = torch.ops.trtllm.k3_fp8_mid_residual_packed(xq, xs, w, w_sf, shared, residual)
    else:
        out = torch.empty_like(residual)
        mid.run_k3_fp8_mid_gemm(
            xq,
            xs,
            w,
            w_sf,
            out,
            shared,
            residual,
            tiler=(256, tn),
            cluster=(2, 1),
            xs_packed_ld=xs.stride(1),
        )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_can_run_packed_rejects():
    mid = _mid()
    n, k, m = 7168, 3584, 96
    w, w_scale = _weight(n, k, seed=9)
    w_sf = mid.expand_weight_scale(w_scale, n, k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    xq, xs = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    shared = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)
    assert mid.can_run_packed(xq, xs, w, w_sf, shared, shared)
    assert not mid.can_run_packed(xq, xs, w, None, shared, shared)
    assert not mid.can_run_packed(xq, xs.contiguous(), w, w_sf, shared, shared)
    assert not mid.can_run_packed(xq[:, : k - 128], xs, w, w_sf, shared, shared)
    assert not mid.can_run_packed(xq, xs, w, w_sf, shared.float(), shared)

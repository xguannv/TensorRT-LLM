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
"""Bit-exactness of the fused MLA context KV expansion (``k3_mla_kv_expand``).

The kernel replaces the context path's 1x128 UE8M0 activation quantization, the DeepGEMM
``kv_b_proj`` GEMM (BF16 output ``[T, H * (128 + 128)]``) and the ``[T, H, 192]`` concatenation
of ``k_nope`` with the broadcast ``k_pe``. K and the V half of the output buffer must match that
path byte for byte.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

NOPE, ROPE, V, RANK = 128, 64, 128, 512


def _module():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_kv_expand

    return k3_mla_kv_expand


def _weight(num_heads, seed):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((num_heads * (NOPE + V), RANK), device="cuda", generator=gen) * 0.05
    return _Fp8BlockScaleWeightReadLinear.quantize_weight(w.to(torch.bfloat16))


def _latent(t, seed, cache_values=True):
    """``[T, RANK + ROPE]`` latent rows; by default FP8 values, as read back from the KV cache."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lat = torch.randn((t, RANK + ROPE), device="cuda", generator=gen) * 2
    if cache_values:
        lat = lat.to(torch.float8_e4m3fn)
    lat = lat.to(torch.bfloat16)
    if t > 1:
        lat[1, :128] = 0  # an all-zero activation block
    return lat


def _reference(compressed_kv, k_pe, w, w_scale, num_heads):
    from tensorrt_llm import deep_gemm

    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(compressed_kv, False)
    kv = torch.empty(
        (compressed_kv.shape[0], w.shape[0]), dtype=torch.bfloat16, device=compressed_kv.device
    )
    deep_gemm.fp8_gemm_nt((a, a_sf), (w, w_scale), kv, disable_ue8m0_cast=True)
    k_nope = kv[:, : num_heads * NOPE].view(-1, num_heads, NOPE)
    k = torch.cat((k_nope, k_pe.reshape(-1, 1, ROPE).expand(-1, num_heads, -1)), dim=-1)
    return k.reshape(compressed_kv.shape[0], -1), kv[:, num_heads * NOPE :]


def _run(compressed_kv, k_pe, w, w_scale, num_heads, **kwargs):
    t = compressed_kv.shape[0]
    k = torch.full(
        (t, num_heads * (NOPE + ROPE)), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    kv = torch.full((t, num_heads * (NOPE + V)), float("nan"), dtype=torch.bfloat16, device="cuda")
    _module().run_k3_mla_kv_expand(compressed_kv, k_pe, w, w_scale, k, kv, num_heads, **kwargs)
    return k, kv


def _assert_bitwise(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("kpe_layout", ["contiguous", "latent_view"])
@pytest.mark.parametrize("epi_warps", [4, 8])
@pytest.mark.parametrize("bn", [128, 192, 256])
@pytest.mark.parametrize("t", [1, 7, 100, 128, 300, 1000, 4097])
def test_bit_exact_vs_deepgemm_and_cat(t, bn, epi_warps, kpe_layout):
    num_heads = 96
    w, w_scale = _weight(num_heads, seed=t)
    lat = _latent(t, seed=t + 1)
    compressed_kv = lat[:, :RANK].contiguous()
    # The chunked-prefill loop passes k_pe gathered from the cache ([T, 64] contiguous); its
    # last partition passes the RoPE slice of latent_cache (row stride 576).
    k_pe = lat[:, RANK:].contiguous() if kpe_layout == "contiguous" else lat[:, RANK:]
    k_ref, v_ref = _reference(compressed_kv, k_pe, w, w_scale, num_heads)
    k, kv = _run(compressed_kv, k_pe, w, w_scale, num_heads, bn=bn, epi_warps=epi_warps)
    _assert_bitwise(k, k_ref)
    _assert_bitwise(kv[:, num_heads * NOPE :], v_ref)


@pytest.mark.parametrize("num_heads", [16, 32, 128])
def test_head_counts(num_heads):
    t = 520
    w, w_scale = _weight(num_heads, seed=3)
    lat = _latent(t, seed=4, cache_values=False)
    compressed_kv = lat[:, :RANK].contiguous()
    k_pe = lat[:, RANK:]
    k_ref, v_ref = _reference(compressed_kv, k_pe, w, w_scale, num_heads)
    k, kv = _run(compressed_kv, k_pe, w, w_scale, num_heads)
    _assert_bitwise(k, k_ref)
    _assert_bitwise(kv[:, num_heads * NOPE :], v_ref)


@pytest.mark.parametrize("num_ctas", [1, 5, 130])
def test_grid_sizes(num_ctas):
    """Persistent CTAs that run many tiles (reserved SMs) produce the same K / V."""
    num_heads, t = 96, 900
    w, w_scale = _weight(num_heads, seed=9)
    lat = _latent(t, seed=10)
    compressed_kv = lat[:, :RANK].contiguous()
    k_pe = lat[:, RANK:].contiguous()
    k_ref, v_ref = _reference(compressed_kv, k_pe, w, w_scale, num_heads)
    k, kv = _run(compressed_kv, k_pe, w, w_scale, num_heads, num_ctas=num_ctas)
    _assert_bitwise(k, k_ref)
    _assert_bitwise(kv[:, num_heads * NOPE :], v_ref)


def test_custom_op_and_operand_checks():
    kx = _module()
    num_heads, t = 96, 200
    w, w_scale = _weight(num_heads, seed=12)
    lat = _latent(t, seed=13)
    compressed_kv = lat[:, :RANK].contiguous()
    k_pe = lat[:, RANK:]
    k_ref, v_ref = _reference(compressed_kv, k_pe, w, w_scale, num_heads)
    k = torch.empty((t, num_heads * (NOPE + ROPE)), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((t, num_heads * (NOPE + V)), dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.k3_mla_kv_expand(compressed_kv, k_pe, w, w_scale, k, kv, num_heads)
    _assert_bitwise(k, k_ref)
    _assert_bitwise(kv[:, num_heads * NOPE :], v_ref)
    with pytest.raises(ValueError):
        kx.check_operands(lat[:, :RANK], k_pe, w, w_scale, k, kv, num_heads)  # strided latent
    with pytest.raises(ValueError):
        kx.check_operands(compressed_kv, k_pe, w, w_scale, k, kv, num_heads * 2)


@pytest.mark.parametrize("kpe_layout", ["contiguous", "latent_view"])
@pytest.mark.parametrize("bn", [128, 192])
@pytest.mark.parametrize("t", [1, 100, 1000, 4097])
def test_fp8_outputs(t, bn, kpe_layout):
    """FP8 K / V are the unit-scale E4M3 conversions of the BF16 K / V (what the attention op's
    FP8 context path computes from them), with V contiguous."""
    num_heads = 96
    w, w_scale = _weight(num_heads, seed=t + 20)
    lat = _latent(t, seed=t + 21, cache_values=False)
    compressed_kv = lat[:, :RANK].contiguous()
    k_pe = lat[:, RANK:].contiguous() if kpe_layout == "contiguous" else lat[:, RANK:]
    k_ref, v_ref = _reference(compressed_kv, k_pe, w, w_scale, num_heads)
    k = torch.empty((t, num_heads * (NOPE + ROPE)), dtype=torch.float8_e4m3fn, device="cuda")
    v = torch.empty((t, num_heads * V), dtype=torch.float8_e4m3fn, device="cuda")
    _module().run_k3_mla_kv_expand(compressed_kv, k_pe, w, w_scale, k, v, num_heads, bn=bn)
    assert torch.equal(k.view(torch.uint8), k_ref.float().to(torch.float8_e4m3fn).view(torch.uint8))
    assert torch.equal(v.view(torch.uint8), v_ref.float().to(torch.float8_e4m3fn).view(torch.uint8))

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
"""Correctness of the low-M FP8-weight CuTe DSL GEMM (``trtllm::k3_fp8w_lowm_gemm``).

The kernel quantizes activations in-kernel to MXFP8 (one power-of-two scale per 1x32
group) and consumes the same resmoothed FP8 weight + DeepGEMM-packed UE8M0 scale as
``fp8_swap_ab_gemm``. Its error against an FP32 reference must not exceed the error of
the current SM100/SM103 path, which quantizes activations with one scale per 1x128.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)

# (N, K): Kimi-K3 Helix decode projections plus small shapes with K % 512 != 0.
SHAPES = [(3072, 7168), (7168, 768), (384, 1280), (256, 640)]
MS = [1, 7, 8, 14, 28, 32, 56, 64]


def _kernel_module():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm


def _make_weight(n, k, seed):
    """N(0, 0.02) weight quantized per 128x128 block with power-of-two scales."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((n, k), device="cuda", generator=gen) * 0.02
    blocks = w.view(n // 128, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    return w_fp8, scale.float()


def _make_activation(m, k, seed):
    """N(0, 1) activations with a few large outliers (up to ~1e3)."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    x = torch.randn((m, k), device="cuda", generator=gen)
    idx = torch.randint(0, m * k, (max(1, m * k // 2048),), device="cuda", generator=gen)
    mag = torch.rand(idx.shape, device="cuda", generator=gen) * 900.0 + 100.0
    x.view(-1)[idx] = mag * torch.sign(torch.randn(idx.shape, device="cuda", generator=gen))
    return x.to(torch.bfloat16)


def _reference(x, w_fp8, scale):
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return x.float() @ (w_fp8.float() * expanded).t()


def _mxfp8_emulated_reference(x, w_fp8, scale):
    """FP32 GEMM on the exact MXFP8 activation the kernel produces (1x32 UE8M0 + E4M3)."""
    m, k = x.shape
    groups = x.float().view(m, k // 32, 32)
    amax = groups.abs().amax(-1, keepdim=True).clamp(min=1.0014e-4)
    exp = torch.ceil(torch.log2(amax / 448.0))
    xq = (groups / torch.exp2(exp)).to(torch.float8_e4m3fn).float() * torch.exp2(exp)
    return _reference(xq.view(m, k), w_fp8, scale)


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _current_path(w_fp8, scale, n):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    lin = _Fp8BlockScaleWeightReadLinear(w_fp8.clone(), scale.clone(), n)
    lin._weights_transformed = False
    lin.transform_weights()
    return lin


def test_pack_matches_deepgemm_layout():
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    mod = _kernel_module()
    for n, k in [(384, 1280), (256, 640), (7168, 768)]:
        w_fp8, scale = _make_weight(n, k, seed=3)
        _, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
        expected = transform_sf_into_required_layout(
            scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
        )
        assert packed.dtype == expected.dtype == torch.int32
        assert packed.shape == expected.shape and packed.stride() == expected.stride()
        assert torch.equal(packed, expected)


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_matches_reference_and_beats_current_path(m, n, k):
    _kernel_module()  # registers trtllm::k3_fp8w_lowm_gemm
    w_fp8, scale = _make_weight(n, k, seed=n + k)
    x = _make_activation(m, k, seed=m)
    ref = _reference(x, w_fp8, scale)

    current = _current_path(w_fp8, scale, n)
    out_current = current(x)
    # Drop-in: consume exactly the tensors the current module holds.
    out = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, current.weight, current.weight_scale)
    torch.cuda.synchronize()

    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    err_new = _rel_err(out, ref)
    err_current = _rel_err(out_current, ref)
    assert err_new < 5e-2, f"m={m} n={n} k={k}: rel err {err_new:.3e}"
    # Bit-level check of the MMA / reduction against the emulated MXFP8 activation.
    err_emu = _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale))
    assert err_emu < 5e-3, f"m={m} n={n} k={k}: rel err vs MXFP8 emulation {err_emu:.3e}"
    # MXFP8 (1x32) scales are never coarser than the current 1x128 ones, so every quantized
    # activation is at least as accurate; only FP32 summation order / BF16 output rounding
    # may differ, hence the tiny relative slack.
    assert err_new <= err_current * (1.0 + 2e-3), (
        f"m={m} n={n} k={k}: new {err_new:.3e} > current {err_current:.3e}"
    )


@pytest.mark.parametrize("split_k", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [5, 16, 40])
def test_split_k_tactics(m, split_k):
    mod = _kernel_module()
    n, k = 384, 1280
    w_fp8, scale = _make_weight(n, k, seed=11)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    x = _make_activation(m, k, seed=5)
    ref = _mxfp8_emulated_reference(x, w_fp8, scale)
    tactic = mod.K3LowMTactic(mma_n=mod.mma_n_for(m), split_k=split_k, a_stages=2, x_stages=4)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    mod.run_k3_fp8w_lowm_gemm(x, w, packed, out, tactic=tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 5e-3


def test_cuda_graph_capture():
    mod = _kernel_module()
    n, k, m = 7168, 768, 14
    w_fp8, scale = _make_weight(n, k, seed=7)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    x = _make_activation(m, k, seed=1)
    eager = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, w, packed)
    out = torch.empty_like(eager)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.k3_fp8w_lowm_gemm_out(x, w, packed, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.k3_fp8w_lowm_gemm_out(x, w, packed, out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_rejects_unsupported_inputs():
    mod = _kernel_module()
    n, k = 256, 640
    w_fp8, scale = _make_weight(n, k, seed=2)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_fp8w_lowm_gemm(
            torch.zeros((65, k), device="cuda", dtype=torch.bfloat16), w, packed
        )
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_fp8w_lowm_gemm(
            torch.zeros((4, k), device="cuda", dtype=torch.bfloat16), w, packed.contiguous()
        )


@pytest.mark.parametrize("m", [1, 28, 64, 65])
def test_weight_read_linear_dispatch(m, monkeypatch):
    """``_Fp8BlockScaleWeightReadLinear`` runs the low-M kernel for M <= 64 when enabled."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    torch.manual_seed(0)
    n, k = 3072, 7168
    weight = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
    linear = kimi._Fp8BlockScaleWeightReadLinear(
        *kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(weight), n
    )
    x = _make_activation(m, k, seed=m)

    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", False)
    current = linear(x)
    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)
    out = linear(x)

    if m <= _kernel_module().MAX_M:
        direct = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, linear.weight, linear.weight_scale)
        torch.testing.assert_close(out, direct, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, current, rtol=0, atol=0)
    assert _rel_err(out, current.float()) < 5e-2

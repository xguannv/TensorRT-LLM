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
"""Correctness of the low-M FP8-weight CuTe DSL GEMM (``trtllm::k3_fp8w_lowm_gemm``) and of the Kimi K3 NVFP4
target's dispatch to it (``kimi_k3_nvfp4/fp8w_lowm.py``).

The kernel quantizes activations in-kernel to MXFP8 (one power-of-two scale per 1x32
group) and consumes the same resmoothed FP8 weight + DeepGEMM-packed UE8M0 scale as
``fp8_swap_ab_gemm``. Its error against an FP32 reference must not exceed the error of
``fp8_swap_ab_gemm`` on the same tensors, which quantizes activations with one scale per 1x128.
Taken from Dongxu Yang's user/dongxuy/k3_lowlat (tests/unittest/_torch/thop/parallel/test_k3_fp8w_lowm_gemm.py), with
the reference path switched to the trunk's ``fp8_swap_ab_gemm`` and a dispatch test added.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)

# (N, K): Kimi-K3 Helix decode projections plus small shapes with K % 512 != 0.
SHAPES = [(3072, 7168), (7168, 768), (384, 1280), (256, 640)]
MS = [1, 7, 8, 14, 28, 32, 56, 64, 65, 112, 128, 168, 224, 256]


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
    n, k = w_fp8.shape
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
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
    """The tensors DeepGEMM's FP8 block-scale Linear holds (resmoothed weight, packed UE8M0 scale) and its GEMM."""
    w, packed = _kernel_module().prepare_k3_fp8w_lowm_weight(w_fp8.clone(), scale.clone())

    def gemm(x):
        return torch.ops.trtllm.fp8_swap_ab_gemm(x, w, packed, disable_ue8m0_cast=True)

    return types.SimpleNamespace(weight=w, weight_scale=packed, __call__=gemm, gemm=gemm)


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
    out_current = current.gemm(x)
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


@pytest.mark.parametrize("split_k", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "m, mma_n", [(40, 64), (65, 96), (128, 128), (150, 160), (170, 192), (200, 224), (250, 256)]
)
def test_mid_m_tactics(m, mma_n, split_k):
    """Quantization kernel + TMA-fed GEMM, including split-K reduce-scatter."""
    mod = _kernel_module()
    n, k = 384, 1280
    w_fp8, scale = _make_weight(n, k, seed=13)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    x = _make_activation(m, k, seed=9)
    ref = _mxfp8_emulated_reference(x, w_fp8, scale)
    base = mod.K3MidMTactic(mma_n=mma_n, split_k=split_k, a_stages=1, b_stages=2)
    tactic = mod.dataclasses.replace(base, a_stages=mod.max_mid_a_stages(n, k, base))
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    mod.run_k3_fp8w_lowm_gemm(x, w, packed, out, tactic=tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 5e-3


def test_low_m_cutover_dispatch(monkeypatch):
    """Shapes with a lower low-M cut-over run M above it on the mid-M path."""
    mod = _kernel_module()
    n, k, m = 384, 1280, 40
    monkeypatch.setitem(mod._LOW_M_CUTOVER, (n, k), 32)
    assert mod.low_m_max_for(n, k) == 32
    w_fp8, scale = _make_weight(n, k, seed=17)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    x = _make_activation(m, k, seed=5)
    out = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, w, packed)
    mid = torch.empty_like(out)
    mod.run_k3_fp8w_lowm_gemm(x, w, packed, mid, tactic=mod.default_mid_tactic(m, n, k))
    torch.cuda.synchronize()
    assert torch.equal(out, mid)
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale)) < 5e-3


@pytest.mark.parametrize("m", [14, 200])
def test_cuda_graph_capture(m):
    mod = _kernel_module()
    n, k = 7168, 768
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
            torch.zeros((mod.MAX_M + 1, k), device="cuda", dtype=torch.bfloat16), w, packed
        )
    assert not mod.can_run(
        torch.zeros((mod.MAX_M + 1, k), device="cuda", dtype=torch.bfloat16), w, packed
    )
    gate = mod.max_m_for(n, k)
    assert mod.LOW_M_MAX <= gate <= mod.MAX_M
    assert mod.can_run(torch.zeros((gate, k), device="cuda", dtype=torch.bfloat16), w, packed)
    assert not mod.can_run(
        torch.zeros((gate + 1, k), device="cuda", dtype=torch.bfloat16), w, packed
    )
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_fp8w_lowm_gemm(
            torch.zeros((4, k), device="cuda", dtype=torch.bfloat16), w, packed.contiguous()
        )


class _FakeFp8Linear(torch.nn.Module):
    """The attributes fp8w_lowm.takes reads, and an apply_linear that records its calls."""

    def __init__(self, w, packed):
        super().__init__()
        self.weight, self.weight_scale = w, packed
        self.use_cute_dsl_blockscaling_mm = False
        self.disable_deep_gemm = False
        self.stock_calls = 0

    def apply_linear(self, input, bias, lora_params=None, layer_idx=None):
        self.stock_calls += 1
        return torch.ops.trtllm.fp8_swap_ab_gemm(
            input, self.weight, self.weight_scale, disable_ue8m0_cast=True
        )


@pytest.mark.parametrize("m", [1, 8, 32, 300])
def test_target_dispatch(m):
    """fp8w_lowm.install routes decode-size inputs to the low-M kernel and everything else to the module's GEMM."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        fp8w_lowm,
    )

    mod = _kernel_module()
    n, k = 3072, 7168
    w, packed = mod.prepare_k3_fp8w_lowm_weight(*_make_weight(n, k, seed=21))
    lin = _FakeFp8Linear(w, packed)
    raw = _FakeFp8Linear(w, packed)
    raw.use_cute_dsl_blockscaling_mm = True
    assert fp8w_lowm.install([lin, raw]) == 1 and fp8w_lowm.install([lin]) == 0
    x = _make_activation(m, k, seed=m)
    out = lin.apply_linear(x, None)
    lowm = mod.can_run(x, w, packed)
    expected = (
        torch.ops.trtllm.k3_fp8w_lowm_gemm(x, w, packed) if lowm else raw.apply_linear(x, None)
    )
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert lin.stock_calls == (0 if lowm else 1)
    assert lowm == (m <= mod.max_m_for(n, k))
    lin.apply_linear(
        x[:1], torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    )  # a bias keeps the stock GEMM
    assert lin.stock_calls == (1 if lowm else 2)

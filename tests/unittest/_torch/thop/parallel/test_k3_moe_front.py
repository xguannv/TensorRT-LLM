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
"""Kimi-K3 small-batch MoE front end: bit-exact replacements and their runtime plumbing.

* ``trtllm::k3_router_logits`` (``k3_router_exact``) vs ``trtllm::dsv3_router_gemm_op``: identical
  FP32 bits for ``M <= 16``, eager and in a replayed CUDA graph.
* ``trtllm::k3_fp8w_lowm_gemm_nvfp4`` vs ``trtllm::fp4_quantize`` of ``trtllm::k3_fp8w_lowm_gemm``:
  identical codes and scales.
* ``KimiK3MoERuntime``: ``TRTLLM_K3_ROUTER_EXACT`` / ``TRTLLM_K3_LD_FP4`` produce the same tensors
  as the paths they replace; ``TRTLLM_K3_MOE_L2_PREFETCH`` issues its prefetches on the latent-down
  stream only when a later MoE layer joins that stream, and captures into a CUDA graph.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)

E, H, LAT = 896, 7168, 3584


def _hidden(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, H), device="cuda", generator=gen)
    x[:, ::7] *= 30.0
    x[:, 1::11] = 0.0
    return x.to(torch.bfloat16)


def _gate(seed):
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    return (torch.randn((E, H), device="cuda", generator=gen) * 0.04).to(torch.bfloat16)


def _bits(t):
    return t.contiguous().view(torch.int32)


@pytest.mark.parametrize("m", [1, 2, 7, 8, 9, 14, 16])
@pytest.mark.parametrize("seed", [0, 1])
def test_router_exact_bitwise(m, seed):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_router_exact  # noqa: F401

    w = _gate(seed)
    x = _hidden(m, seed)
    ref = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
    out = torch.ops.trtllm.k3_router_logits(x, w)
    assert out.shape == (m, E) and out.dtype == torch.float32
    assert torch.equal(_bits(out), _bits(ref))


def test_router_exact_cuda_graph():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_router_exact  # noqa: F401

    w = _gate(3)
    x = _hidden(14, 3)
    torch.ops.trtllm.k3_router_logits(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.trtllm.k3_router_logits(x, w)
    for seed in range(3):
        x.copy_(_hidden(14, 10 + seed))
        graph.replay()
        torch.cuda.synchronize()
        ref = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
        assert torch.equal(_bits(out), _bits(ref))


def test_router_exact_can_run():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_router_exact

    w = _gate(0)
    assert k3_router_exact.can_run(_hidden(16, 0), w)
    assert not k3_router_exact.can_run(_hidden(17, 0), w)
    assert not k3_router_exact.can_run(_hidden(4, 0).float(), w)
    assert not k3_router_exact.can_run(_hidden(4, 0), w[:448])
    assert not k3_router_exact.can_run(_hidden(4, 0), w.float())


def _fp8_linear(n, k, seed):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear as F8

    gen = torch.Generator(device="cuda").manual_seed(2000 + seed)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w8, s = F8.quantize_weight(w)
    return F8(w8, s, n)


@pytest.mark.parametrize("shape", [(LAT, H), (768, H), (H, LAT)])
@pytest.mark.parametrize("m", [1, 7, 14, 16, 32, 64])
def test_lowm_nvfp4_matches_fp4_quantize(shape, m):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lm

    n, k = shape
    lin = _fp8_linear(n, k, m)
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn((m, k), device="cuda", generator=gen)
    if m > 2:
        x[1, : k // 2] = 0.0
        x[2] *= 1e-6
    x = x.to(torch.bfloat16)
    out = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, lin.weight, lin.weight_scale)
    for gsf_value in (1.0, 448.0 * 6.0 / max(out.float().abs().max().item(), 1e-6)):
        gsf = torch.tensor([gsf_value], device="cuda", dtype=torch.float32)
        ref_q, ref_sf = torch.ops.trtllm.fp4_quantize(out, gsf, 16, False, False)
        assert lm.can_run_nvfp4(x, lin.weight, lin.weight_scale, gsf)
        q, sf = torch.ops.trtllm.k3_fp8w_lowm_gemm_nvfp4(x, lin.weight, lin.weight_scale, gsf)
        assert torch.equal(q, ref_q.view(torch.uint8).view(m, n // 2))
        assert torch.equal(sf, ref_sf.view(torch.uint8).reshape(-1)[: m * (n // 16)].view(m, -1))


def test_lowm_nvfp4_can_run():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lm

    lin = _fp8_linear(LAT, H, 0)
    gsf = torch.ones(1, device="cuda")
    x = torch.randn((14, H), device="cuda").to(torch.bfloat16)
    assert lm.can_run_nvfp4(x, lin.weight, lin.weight_scale, gsf)
    assert not lm.can_run_nvfp4(x, lin.weight, lin.weight_scale, gsf.double())
    assert not lm.can_run_nvfp4(x, lin.weight, lin.weight_scale, torch.ones(2, device="cuda"))
    big = torch.randn((65, H), device="cuda").to(torch.bfloat16)
    assert not lm.can_run_nvfp4(big, lin.weight, lin.weight_scale, gsf)


class _Backend:
    """What ``_lowm_fp4_latent_input`` reads from the CuteDSL NVFP4 backend."""

    has_nvfp4 = True
    scaling_vector_size = 16

    def __init__(self):
        self.fc31_input_scale = torch.tensor([123.0], device="cuda")

    def quantize_input_async(self, *args, **kwargs):
        raise AssertionError("not called by the runtime helpers under test")


def _runtime(layer_idx=1, group=None):
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    rt = mkl.KimiK3MoERuntime.__new__(mkl.KimiK3MoERuntime)
    nn.Module.__init__(rt)
    weight = _gate(layer_idx)

    def compute_logits(hidden):
        return torch.ops.trtllm.dsv3_router_gemm_op(
            hidden.reshape(-1, H), weight.t(), bias=None, out_dtype=torch.float32
        )

    rt.gate = SimpleNamespace(weight=weight, gating_dim=H, compute_logits=compute_logits)
    rt.routed_expert_down_proj = _fp8_linear(LAT, H, layer_idx)
    rt.routed_expert_up_proj = _fp8_linear(H, LAT, 10 + layer_idx)
    rt.shared_experts = SimpleNamespace(
        gate_up_proj=_fp8_linear(768, H, 20 + layer_idx),
        down_proj=_fp8_linear(H, 384, 30 + layer_idx),
    )
    rt.routed_experts = SimpleNamespace(
        comm=None, apply_router_weight_on_input=False, backend=_Backend()
    )
    rt.layer_idx = layer_idx
    rt._router_exact = False
    rt.latent_down_stream = torch.cuda.Stream()
    rt.moe_fc2_event = torch.cuda.Event()
    rt._moe_group = group if group is not None else weakref.WeakValueDictionary()
    rt._moe_group[layer_idx] = rt
    rt._l2_ranges_cache = {}
    return rt


def test_runtime_router_exact():
    rt = _runtime()
    for m in (7, 14, 16, 32):
        x = _hidden(m, m)
        ref = rt.gate.compute_logits(x)
        rt._router_exact = True
        out = rt._router_logits(x)
        rt._router_exact = False
        assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("mode", ["0", "1", "kda"])
def test_router_exact_for_layer(monkeypatch, mode):
    """TRTLLM_K3_ROUTER_EXACT=1 selects every MoE layer, =kda only those after a KDA block."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    cfg = SimpleNamespace(
        linear_attn_config={"kda_layers": [1, 2, 3, 5, 6, 7], "full_attn_layers": [4, 8]}
    )
    monkeypatch.setattr(mkl, "_K3_ROUTER_EXACT_MODE", mode)
    got = [mkl._router_exact_for_layer(cfg, i) for i in range(8)]
    kda = [mkl._is_kda_layer(cfg, i) for i in range(8)]
    assert got == {"0": [False] * 8, "1": [True] * 8, "kda": kda}[mode]
    assert kda == [True, True, True, False, True, True, True, False]


def test_runtime_ld_fp4(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    rt = _runtime()
    monkeypatch.setattr(mkl, "_K3_FP8W_LOWM_GEMM", True)
    x = _hidden(14, 5)
    monkeypatch.setattr(mkl, "_K3_LD_FP4", False)
    assert rt._lowm_fp4_latent_input(x) is None
    monkeypatch.setattr(mkl, "_K3_LD_FP4", True)
    fused = rt._lowm_fp4_latent_input(x)
    assert isinstance(fused, mkl.Fp4QuantizedTensor) and not fused.is_sf_swizzled
    gsf = rt.routed_experts.backend.fc31_input_scale
    ref_q, ref_sf = torch.ops.trtllm.fp4_quantize(
        rt.routed_expert_down_proj(x), gsf, 16, False, False
    )
    assert torch.equal(fused.fp4_tensor, ref_q.view(torch.uint8).view(14, LAT // 2))
    assert torch.equal(
        fused.scaling_factor, ref_sf.view(torch.uint8).reshape(-1)[: 14 * LAT // 16].view(14, -1)
    )
    # Not served: too many tokens for the low-M kernel, or an expert backend that is not NVFP4.
    assert rt._lowm_fp4_latent_input(_hidden(65, 0)) is None
    rt.routed_experts.backend.has_nvfp4 = False
    assert rt._lowm_fp4_latent_input(x) is None


def test_runtime_l2_prefetch(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    group = weakref.WeakValueDictionary()
    first = _runtime(1, group)
    last = _runtime(2, group)
    last.latent_down_stream = first.latent_down_stream
    assert first._next_moe_layer() is last and last._next_moe_layer() is None
    assert len(first.l2_prefetch_ranges("lu")) == 2
    assert len(last.l2_prefetch_ranges("front")) == 7
    calls = []
    import tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_l2_prefetch as pf

    real = pf.run_k3_l2_prefetch

    def spy(ranges, **kwargs):
        calls.append((torch.cuda.current_stream(), list(ranges)))
        real(ranges, **kwargs)

    monkeypatch.setattr(pf, "run_k3_l2_prefetch", spy)
    monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH", frozenset({"lu", "front"}))
    static = torch.zeros(4, device="cuda")
    with with_multi_stream(True):
        # The last MoE layer issues nothing (no later latent-down work joins the stream).
        last._issue_l2_prefetch(14)
        assert not calls
        # Too many tokens: nothing.
        first._issue_l2_prefetch(65)
        assert not calls
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static.add_(1.0)
            first._issue_l2_prefetch(14)
            # The next MoE layer's latent-down work follows on the same stream and joins main.
            fork = torch.cuda.Event()
            fork.record()
            ready = torch.cuda.Event()
            with torch.cuda.stream(last.latent_down_stream):
                fork.wait()
                static.add_(1.0)
                ready.record()
            ready.wait()
    assert [r for _, r in calls] == [
        first.l2_prefetch_ranges("lu"),
        last.l2_prefetch_ranges("front"),
    ]
    assert all(s == first.latent_down_stream for s, _ in calls)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static, torch.full_like(static, 4.0))

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
"""Kimi-K3 MLA output gate fused with the o_proj 1x128 FP8 quantization.

``gate_mul_fp8_packed(a, g)`` must give, byte for byte, the E4M3 values and the whole packed
UE8M0 scales (the tail bytes of the last word included) of ``fp8_quantize_1x128_packed_ue8m0(a * g.sigmoid())``,
and the gated o_proj output through the K3 FP8 weight-read projection must be unchanged.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)


def _inputs(m, n, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ch = torch.exp(torch.randn(n, device="cuda", generator=gen))
    a = (torch.randn((m, n), device="cuda", generator=gen) * ch).to(torch.bfloat16)
    g = (torch.randn((m, n), device="cuda", generator=gen) * 4).to(torch.bfloat16)
    if m > 2:
        a[0, :128] = 0  # an all-zero block
        g[1, :64] = 90.0  # saturated gates
        g[2, :64] = -90.0
    return a, g


def _assert_scales(s, rs):
    """Same words for every row (the bytes past the last block included); the padding rows of
    the fused op's buffer (up to the leading dimension) are zero, as TMA may read them."""
    assert s.shape == rs.shape and s.stride() == rs.stride()
    assert torch.equal(s, rs)
    rows, ld = s.shape[0], s.stride(1)
    assert not s.as_strided((s.shape[1], ld), (ld, 1))[:, rows:].any()


@pytest.mark.parametrize("m", [1, 3, 7, 128, 203, 448, 896, 1792])
@pytest.mark.parametrize("n", [768, 1536])
def test_bitwise_vs_unfused(m, n):
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_packed

    a, g = _inputs(m, n, seed=m * 7 + n)
    q, s = gate_mul_fp8_packed(a, g)
    rq, rs = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(a * g.sigmoid(), False)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    _assert_scales(s, rs)


def test_strided_rows():
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_packed

    m, n = 300, 768
    a_full, g_full = _inputs(m, 2 * n, seed=5)
    a, g = a_full[:, :n], g_full[:, n:]
    q, s = gate_mul_fp8_packed(a, g)
    rq, rs = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0((a * g.sigmoid()).contiguous(), False)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    _assert_scales(s, rs)


@pytest.mark.parametrize("m", [203, 896])
@pytest.mark.parametrize("mid_gemm", [True, False])
def test_gated_o_proj_unchanged(monkeypatch, m, mid_gemm):
    from tensorrt_llm._torch.models import modeling_kimi_linear as mk
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_packed

    monkeypatch.setattr(mk, "_K3_FP8_MID_FAMILIES", frozenset({"o"} if mid_gemm else ()))
    monkeypatch.setattr(mk, "_K3_FP8_MID_MIN_M", 1)
    monkeypatch.setattr(mk, "_K3_FP8W_LOWM_GEMM", False)
    n, k = 7168, 768
    gen = torch.Generator(device="cuda").manual_seed(m)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    lin = mk._Fp8BlockScaleWeightReadLinear(
        *mk._Fp8BlockScaleWeightReadLinear.quantize_weight(w), n
    )
    lin.k3_mid_family = "o"
    assert lin.prepare_mid_gemm() == mid_gemm
    a, g = _inputs(m, k, seed=m + 1)
    assert lin.takes_fp8_packed_input(m)
    ref = lin(a * g.sigmoid())
    out = lin.forward_fp8_packed(*gate_mul_fp8_packed(a, g))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

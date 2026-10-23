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
"""``run_k3_fp8w_lowm_gemm_mxfp8``: the low-M FP8-weight GEMM on an upstream MXFP8 activation.

Fed with the MXFP8 form of a BF16 activation (``mxfp8_quantize_reference``, the arithmetic of the
low-M kernel's in-kernel quantization), every output bit equals ``run_k3_fp8w_lowm_gemm`` on the
BF16 activation, with and without ``TRTLLM_K3_LOWM_SF_SLOTS``, with and without ``sf_early_release``
(``TRTLLM_K3LL_MX_SF_EARLY_RELEASE``: each K tile released once its scales are folded), eagerly and
in a CUDA graph behind the attention-residual kernel that produces the MXFP8 activation.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)

# Kimi-K3 projections fed by the attention-residual mix (K = 7168: KDA qkvg, latent down, MLA
# q_a + kv_a, shared gate-up) and the KDA / MLA o_proj (K = 768).
SHAPES = [(3072, 7168), (3584, 7168), (2112, 7168), (12288, 7168), (7168, 768)]
MS = [1, 7, 8, 14, 28, 32, 33, 64]


def _modules():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

    return lowm, mx


def _weight(n, k, seed):
    """N(0, 0.02) weight of whole 128-row tiles, power-of-two 128x128 block scales, first N rows."""
    lowm, _ = _modules()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n_pad = -(-n // 128) * 128
    w = torch.randn((n_pad, k), device="cuda", generator=gen) * 0.02
    blocks = w.view(n_pad // 128, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n_pad, k)
    return lowm.prepare_k3_fp8w_lowm_weight(w_fp8[:n].contiguous(), scale.float())


def _activation(m, k, seed):
    """N(0, 1) with outliers and one all-zero 1x32 group (the amax floor)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), device="cuda", generator=gen)
    idx = torch.randint(0, m * k, (max(1, m * k // 2048),), device="cuda", generator=gen)
    x.view(-1)[idx] = 300.0
    x[0, :32] = 0
    return x.to(torch.bfloat16)


def _bitwise(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_matches_low_m_kernel(m, n, k):
    lowm, mx = _modules()
    if m > lowm.low_m_max_for(n, k):
        pytest.skip("M is served by the mid-M path")
    w, ws = _weight(n, k, seed=n + k)
    x = _activation(m, k, seed=m)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    lowm.run_k3_fp8w_lowm_gemm(x, w, ws, ref)
    xq, xs = mx.mxfp8_quantize_reference(x)
    assert mx.can_run_mxfp8(xq, xs, w, ws, allow_partial_n_tile=True)
    out = torch.empty_like(ref)
    mx.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, w, ws, out)
    torch.cuda.synchronize()
    assert _bitwise(out, ref)


@pytest.mark.parametrize("m", [7, 32, 64])
def test_sf_slots_bitwise(m, monkeypatch):
    """``TRTLLM_K3_LOWM_SF_SLOTS`` does not change the result."""
    lowm, mx = _modules()
    n, k = 3584, 7168
    w, ws = _weight(n, k, seed=3)
    xq, xs = mx.mxfp8_quantize_reference(_activation(m, k, seed=m))
    outs = []
    for on in (False, True):
        monkeypatch.setattr(lowm, "_SF_SLOTS", on)
        monkeypatch.setattr(mx, "_COMPILED", {})
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        mx.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, w, ws, out)
        outs.append(out)
    torch.cuda.synchronize()
    assert _bitwise(outs[0], outs[1])


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", [1, 7, 14, 32, 64])
def test_sf_early_release_bitwise(m, n, k):
    """``sf_early_release`` (each K tile released once its scales are folded) gives the low-M
    kernel's bits, for the default tactic and one with a shorter activation ring (which keeps the
    release after every fold), and is compiled as its own kernel next to the default one."""
    lowm, mx = _modules()
    if m > lowm.low_m_max_for(n, k):
        pytest.skip("M is served by the mid-M path")
    w, ws = _weight(n, k, seed=n + 2 * k)
    x = _activation(m, k, seed=5 * m + 1)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    lowm.run_k3_fp8w_lowm_gemm(x, w, ws, ref)
    xq, xs = mx.mxfp8_quantize_reference(x)
    base = mx.default_mx_tactic(m, n, k)
    tactics = [None]
    if base.b_stages > 2:
        short = mx.K3LowMMxTactic(
            mma_n=base.mma_n, split_k=base.split_k, a_stages=base.a_stages, b_stages=2
        )
        tactics.append(short)
    for tactic in tactics:
        for early in (True, False):
            out = torch.empty_like(ref)
            mx.run_k3_fp8w_lowm_gemm_mxfp8(
                xq, xs, w, ws, out, tactic=tactic, sf_early_release=early
            )
            torch.cuda.synchronize()
            assert _bitwise(out, ref), (tactic, early)
    assert mx.is_compiled(m, n, k, True, True) and mx.is_compiled(m, n, k, True, False)


def test_deep_k_tactic_fits_shared_memory():
    """Long K slices at mma_n >= 32 shorten the activation ring instead of failing."""
    lowm, mx = _modules()
    torch.zeros(1, device="cuda")  # the split-K heuristic queries the device's cluster occupancy
    for m in (8, 16, 32):
        tactic = mx.default_mx_tactic(m, 12288, 7168)
        mx.validate_tactic(12288, 7168, tactic)
        assert tactic.a_stages >= 6
    assert mx.default_mx_tactic(14, 3072, 7168).b_stages == -(
        -56 // lowm.default_tactic(14, 3072, 7168).split_k
    )


def test_can_run_rejects_unsupported_inputs():
    lowm, mx = _modules()
    n, k = 3072, 7168
    w, ws = _weight(n, k, seed=1)
    xq, xs = mx.mxfp8_quantize_reference(_activation(14, k, seed=1))
    assert mx.can_run_mxfp8(xq, xs, w, ws)
    assert not mx.can_run_mxfp8(xq.view(torch.uint8), xs, w, ws)
    assert not mx.can_run_mxfp8(xq, xs[:, :-1].contiguous(), w, ws)
    assert not mx.can_run_mxfp8(xq[:, :-128].contiguous(), xs[:, :-4].contiguous(), w, ws)
    big_q, big_s = mx.mxfp8_quantize_reference(_activation(65, k, seed=2))
    assert not mx.can_run_mxfp8(big_q, big_s, w, ws)
    w_part, ws_part = _weight(2112, k, seed=2)
    assert not mx.can_run_mxfp8(xq, xs, w_part, ws_part)
    assert mx.can_run_mxfp8(xq, xs, w_part, ws_part, allow_partial_n_tile=True)


@pytest.mark.parametrize("m", [7, 14, 32])
def test_cuda_graph_behind_attn_res(m):
    """attn_res with the MXFP8 side output -> MXFP8 GEMM (both PDL) in a CUDA graph equals the
    BF16 low-M GEMM on the attn_res output."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res  # noqa: F401

    lowm, mx = _modules()
    hidden, n, snaps = 7168, 3072, 5
    w, ws = _weight(n, hidden, seed=7)
    gen = torch.Generator(device="cuda").manual_seed(m)

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(torch.bfloat16)

    layer, block = rn(m, hidden), rn(snaps, m, hidden)
    weights = (
        rn(hidden, scale=0.05),
        (1 + rn(hidden, scale=0.1).float()).to(torch.bfloat16),
        (1 + rn(hidden, scale=0.1).float()).to(torch.bfloat16),
    )
    outs = {}

    def body():
        normed, xq, xs = torch.ops.trtllm.k3_attn_res_rmsnorm_mx(layer, block, *weights, 1e-6, 1e-5)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        mx.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, w, ws, out)
        outs["normed"], outs["out"] = normed, out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for seed in range(3):
        layer.copy_(torch.randn(m, hidden, device="cuda", generator=gen).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        lowm.run_k3_fp8w_lowm_gemm(outs["normed"], w, ws, ref)
        torch.cuda.synchronize()
        assert _bitwise(outs["out"], ref), f"replay {seed}"

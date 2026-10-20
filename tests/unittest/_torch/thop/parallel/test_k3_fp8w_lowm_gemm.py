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

import functools

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


@pytest.mark.parametrize("m", [1, 28, 64, 65, 224, 225, 256, 257])
def test_weight_read_linear_dispatch(m, monkeypatch):
    """``_Fp8BlockScaleWeightReadLinear`` runs this kernel for M <= max_m_for(N, K)."""
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

    if m <= _kernel_module().max_m_for(n, k):
        direct = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, linear.weight, linear.weight_scale)
        torch.testing.assert_close(out, direct, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, current, rtol=0, atol=0)
    assert _rel_err(out, current.float()) < 5e-2


# ---------------------------------------------------------------------------
# Partial last N tile (N % 128 != 0), e.g. the MLA fused [q_a; kv_a; k_pe] projection.
# ---------------------------------------------------------------------------

#: (N, K): the Kimi-K3 / DeepSeek-V3 MLA fused-A projection (1536 + 512 + 64 rows) and
#: small shapes whose last tile keeps 80 / 16 rows.
PARTIAL_SHAPES = [(2112, 7168), (336, 1280), (144, 640)]
PARTIAL_MS = [1, 7, 8, 13, 32, 63, 64, 65, 100, 128, 129, 192, 224]


def _make_padded_weight(n, k, seed):
    """FP8 weight of ceil(N/128) * 128 rows (rows >= N random too) plus its block scales."""
    n_pad = -(-n // 128) * 128
    w_fp8, scale = _make_weight(n_pad, k, seed)
    return w_fp8, scale


def _run(mod, x, w, packed, tactic=None):
    out = torch.empty((x.shape[0], w.shape[0]), device="cuda", dtype=torch.bfloat16)
    mod.run_k3_fp8w_lowm_gemm(x, w, packed, out, tactic=tactic)
    return out


def test_pack_matches_deepgemm_layout_partial_n():
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    mod = _kernel_module()
    for n, k in PARTIAL_SHAPES:
        w_pad, scale = _make_padded_weight(n, k, seed=3)
        w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
        expected = transform_sf_into_required_layout(
            scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
        )
        assert w.shape == (n, k)
        assert packed.shape == expected.shape == (n, -(-k // 512))
        assert packed.stride() == expected.stride() == (1, n)
        assert torch.equal(packed, expected)


@pytest.mark.parametrize("n, k", PARTIAL_SHAPES)
@pytest.mark.parametrize("m", PARTIAL_MS)
def test_partial_n_tile_matches_padded_weight(m, n, k):
    """The N real columns equal the zero-padded weight's run and ignore the padding rows."""
    mod = _kernel_module()
    w_pad, scale = _make_padded_weight(n, k, seed=n + k)
    x = _make_activation(m, k, seed=m)

    w_n, packed_n = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    zero_pad = w_pad.clone()
    zero_pad[n:] = 0
    w_z, packed_z = mod.prepare_k3_fp8w_lowm_weight(zero_pad, scale)
    w_g, packed_g = mod.prepare_k3_fp8w_lowm_weight(w_pad, scale)
    # The partial and padded weights share the tactic (same tiles, same tuned entry).
    if m <= mod.low_m_max_for(n, k):
        assert mod.default_tactic(m, n, k) == mod.default_tactic(m, w_z.shape[0], k)
    else:
        assert mod.default_mid_tactic(m, n, k) == mod.default_mid_tactic(m, w_z.shape[0], k)

    out = _run(mod, x, w_n, packed_n)
    out_zero = _run(mod, x, w_z, packed_z)
    out_garbage = _run(mod, x, w_g, packed_g)
    torch.cuda.synchronize()

    assert out.shape == (m, n)
    assert torch.isfinite(out).all()
    # Rows of a tile are independent: zero or random padding rows change nothing.
    assert torch.equal(out, out_zero[:, :n])
    assert torch.equal(out, out_garbage[:, :n])
    assert not out_zero[:, n:].any()
    ref = _mxfp8_emulated_reference(x, w_pad[:n], scale)
    assert _rel_err(out, ref) < 5e-3


@pytest.mark.parametrize("m", [1, 8, 32, 64, 65, 128, 224, 256])
def test_partial_n_tile_stays_in_bounds(m):
    """No store lands past the [M, N] output (guard band after it stays untouched)."""
    mod = _kernel_module()
    n, k = 2112, 7168
    w_pad, scale = _make_padded_weight(n, k, seed=21)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = _make_activation(m, k, seed=m + 3)
    guard = 128 * 64
    buf = torch.full((m * n + guard,), -7.0, device="cuda", dtype=torch.bfloat16)
    out = buf[: m * n].view(m, n)
    mod.k3_fp8w_lowm_gemm_out(x, w, packed, out)
    torch.cuda.synchronize()
    assert (buf[m * n :] == -7.0).all()
    zero_pad = w_pad.clone()
    zero_pad[n:] = 0
    w_z, packed_z = mod.prepare_k3_fp8w_lowm_weight(zero_pad, scale)
    assert torch.equal(out, _run(mod, x, w_z, packed_z)[:, :n])


@pytest.mark.parametrize("split_k", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [5, 16, 40, 64])
def test_partial_n_tile_split_k_tactics(m, split_k):
    mod = _kernel_module()
    n, k = 336, 1280
    w_pad, scale = _make_padded_weight(n, k, seed=11)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = _make_activation(m, k, seed=5)
    tactic = mod.K3LowMTactic(mma_n=mod.mma_n_for(m), split_k=split_k, a_stages=2, x_stages=4)
    out = _run(mod, x, w, packed, tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_pad[:n], scale)) < 5e-3


@pytest.mark.parametrize("split_k", [1, 2, 3, 5])
@pytest.mark.parametrize("m, mma_n", [(40, 64), (65, 96), (128, 128), (200, 224), (250, 256)])
def test_partial_n_tile_mid_m_tactics(m, mma_n, split_k):
    mod = _kernel_module()
    n, k = 336, 1280
    w_pad, scale = _make_padded_weight(n, k, seed=13)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = _make_activation(m, k, seed=9)
    base = mod.K3MidMTactic(mma_n=mma_n, split_k=split_k, a_stages=1, b_stages=2)
    tactic = mod.dataclasses.replace(base, a_stages=mod.max_mid_a_stages(n, k, base))
    out = _run(mod, x, w, packed, tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_pad[:n], scale)) < 5e-3


def test_partial_n_tile_gate():
    """``can_run`` takes N % 128 != 0 only on request; the gate follows the padded shape."""
    mod = _kernel_module()
    n, k = 2112, 7168
    assert mod.max_m_for(n, k) == mod.max_m_for(2176, k) == 224
    assert mod.low_m_max_for(n, k) == mod.low_m_max_for(2176, k)
    w_pad, scale = _make_padded_weight(n, k, seed=2)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = torch.zeros((8, k), device="cuda", dtype=torch.bfloat16)
    assert not mod.can_run(x, w, packed)
    assert mod.can_run(x, w, packed, allow_partial_n_tile=True)
    x_big = torch.zeros((225, k), device="cuda", dtype=torch.bfloat16)
    assert not mod.can_run(x_big, w, packed, allow_partial_n_tile=True)
    # Whole tiles need no opt-in; N must stay a multiple of N_ALIGN.
    w_full, packed_full = mod.prepare_k3_fp8w_lowm_weight(w_pad[:2048].contiguous(), scale[:16])
    assert mod.can_run(x, w_full, packed_full)
    with pytest.raises(ValueError):
        mod.prepare_k3_fp8w_lowm_weight(w_pad[:2104].contiguous(), scale)


@pytest.mark.parametrize("m", [1, 8, 32, 64, 128, 224])
def test_partial_n_tile_cuda_graph(m):
    mod = _kernel_module()
    n, k = 2112, 7168
    w_pad, scale = _make_padded_weight(n, k, seed=7)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
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


def _fp8_block_scale_linear(w_fp8, scale, partial_n_tile):
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    n, k = w_fp8.shape
    linear = Linear(
        k,
        n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128),
    ).cuda()
    linear.fp8w_lowm_partial_n_tile = partial_n_tile
    linear.load_weights([{"weight": w_fp8, "weight_scale": scale}])
    linear.post_load_weights()
    return linear


@pytest.mark.parametrize("m", [1, 8, 32, 64, 128, 224, 225])
@pytest.mark.parametrize("fused_a", [False, True])
def test_fp8_block_scale_linear_partial_n_dispatch(m, fused_a, monkeypatch):
    """``FP8BlockScalesLinearMethod`` serves a marked 2112-row linear with this kernel."""
    from tensorrt_llm._torch.modules import linear as linear_mod

    mod = _kernel_module()
    n, k = 2112, 7168
    w_pad, scale = _make_padded_weight(n, k, seed=31)
    linear = _fp8_block_scale_linear(w_pad[:n].contiguous(), scale, partial_n_tile=True)
    assert linear.weight.shape == (n, k)
    assert linear.weight_scale.shape == (n, 14) and linear.weight_scale.stride() == (1, n)
    x = _make_activation(m, k, seed=m)

    monkeypatch.setattr(linear_mod, "_FP8W_LOWM_GEMM", True)
    monkeypatch.setattr(linear_mod, "_FP8W_LOWM_FUSED_A", fused_a)
    out = linear(x)
    current = torch.ops.trtllm.fp8_swap_ab_gemm(
        x, linear.weight, linear.weight_scale, disable_ue8m0_cast=True
    )
    torch.cuda.synchronize()
    assert out.shape == (m, n) and out.is_contiguous()
    if fused_a and m <= mod.max_m_for(n, k):
        direct = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, linear.weight, linear.weight_scale)
        assert torch.equal(out, direct)
    else:
        assert torch.equal(out, current)
    ref = _reference(x, w_pad[:n], scale)
    assert _rel_err(out, ref) < 5e-2
    assert _rel_err(out, ref) <= _rel_err(current, ref) * (1.0 + 2e-3)

    # An unmarked linear never takes the partial-tile path.
    plain = _fp8_block_scale_linear(w_pad[:n].contiguous(), scale, partial_n_tile=False)
    assert torch.equal(plain(x), current)


@pytest.mark.parametrize("m", [8, 32, 64, 128, 224])
def test_fp8_block_scale_linear_partial_n_cuda_graph(m, monkeypatch):
    from tensorrt_llm._torch.modules import linear as linear_mod

    monkeypatch.setattr(linear_mod, "_FP8W_LOWM_GEMM", True)
    monkeypatch.setattr(linear_mod, "_FP8W_LOWM_FUSED_A", True)
    n, k = 2112, 7168
    w_pad, scale = _make_padded_weight(n, k, seed=37)
    linear = _fp8_block_scale_linear(w_pad[:n].contiguous(), scale, partial_n_tile=True)
    x = _make_activation(m, k, seed=m)
    eager = linear(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        linear(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = linear(x)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def _weight_read_linear(w_fp8, scale, partial_n_tile):
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    linear = kimi._Fp8BlockScaleWeightReadLinear(w_fp8.clone(), scale.clone(), w_fp8.shape[0])
    linear._weights_transformed = False
    linear.transform_weights()
    linear.fp8w_lowm_partial_n_tile = partial_n_tile
    return linear


@pytest.mark.parametrize("m", [1, 8, 32, 64, 128, 224, 225])
@pytest.mark.parametrize("fused_a", [False, True])
def test_weight_read_linear_partial_n_dispatch(m, fused_a, monkeypatch):
    """Kimi-K3 ``_Fp8BlockScaleWeightReadLinear`` (the runtime MLA fused-A module) serves the
    marked 2112-row weight with this kernel only under ``TRTLLM_K3_FP8W_LOWM_FUSED_A``."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    mod = _kernel_module()
    n, k = 2112, 7168
    w_pad, scale = _make_padded_weight(n, k, seed=41)
    linear = _weight_read_linear(w_pad[:n].contiguous(), scale, partial_n_tile=True)
    x = _make_activation(m, k, seed=m)

    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)
    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_FUSED_A", fused_a)
    out = linear(x)
    current = torch.ops.trtllm.fp8_swap_ab_gemm(
        x, linear.weight, linear.weight_scale, disable_ue8m0_cast=True
    )
    torch.cuda.synchronize()
    assert out.shape == (m, n) and out.is_contiguous()
    if fused_a and m <= mod.max_m_for(n, k):
        direct = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, linear.weight, linear.weight_scale)
        assert torch.equal(out, direct)
    else:
        assert torch.equal(out, current)
    ref = _reference(x, w_pad[:n], scale)
    assert _rel_err(out, ref) <= _rel_err(current, ref) * (1.0 + 2e-3)

    unmarked = _weight_read_linear(w_pad[:n].contiguous(), scale, partial_n_tile=False)
    assert torch.equal(unmarked(x), current)


@pytest.mark.parametrize("m", [8, 32, 64, 128, 224])
def test_weight_read_linear_partial_n_cuda_graph(m, monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)
    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_FUSED_A", True)
    n, k = 2112, 7168
    w_pad, scale = _make_padded_weight(n, k, seed=43)
    linear = _weight_read_linear(w_pad[:n].contiguous(), scale, partial_n_tile=True)
    x = _make_activation(m, k, seed=m)
    eager = linear(x)
    assert torch.equal(
        eager, torch.ops.trtllm.k3_fp8w_lowm_gemm(x, linear.weight, linear.weight_scale)
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        linear(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = linear(x)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


# ---------------------------------------------------------------------------
# Gated low-M GEMM: out = bf16(x * gate) @ W^T (the MLA output gate fused into o_proj).
# ---------------------------------------------------------------------------

#: (N, K): the Kimi-K3 Helix o_proj, a split-K shape, and a partial last N tile.
GATED_SHAPES = [(7168, 768), (384, 1280), (2112, 7168)]
GATED_MS = [1, 7, 8, 14, 28, 32, 56, 64]


def _make_gate(m, k, seed, row_stride=None):
    """Sigmoid gate in BF16 (with exact 0 / 1 entries), optionally a column slice of a wider
    tensor (row stride ``row_stride``)."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 2)
    width = k if row_stride is None else row_stride
    g = (torch.randn((m, width), device="cuda", generator=gen) * 4).to(torch.bfloat16).sigmoid()
    g.view(-1)[::97] = 0
    g.view(-1)[1::89] = 1
    return g[:, width - k :] if row_stride is not None else g


@pytest.mark.parametrize("n, k", GATED_SHAPES)
@pytest.mark.parametrize("m", GATED_MS)
@pytest.mark.parametrize("strided", [False, True], ids=["contig", "strided"])
def test_gated_matches_multiply_then_gemm(m, n, k, strided):
    """Bitwise equal to PyTorch's ``x * gate`` followed by the plain low-M kernel."""
    mod = _kernel_module()
    w_pad, scale = _make_padded_weight(n, k, seed=n + k)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = _make_activation(m, k, seed=m)
    gate = _make_gate(m, k, seed=m, row_stride=k + 1024 if strided else None)
    plain = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    mod.run_k3_fp8w_lowm_gemm(x * gate, w, packed, plain, tactic=mod.default_tactic(m, n, k))
    out = mod.k3_fp8w_lowm_gemm_gated(x, gate, w, packed)
    torch.cuda.synchronize()
    assert torch.equal(out, plain)
    ref = _mxfp8_emulated_reference((x * gate).contiguous(), w_pad[:n], scale)
    assert _rel_err(out, ref) < 5e-3


@pytest.mark.parametrize("m", [7, 14, 32])
def test_gated_cuda_graph_after_side_stream(m):
    """Graph replay with the gate written on another stream right before the launch."""
    mod = _kernel_module()
    n, k = 7168, 768
    w_fp8, scale = _make_weight(n, k, seed=7)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    x = _make_activation(m, k, seed=1)
    logits = _make_activation(m, k, seed=2)
    eager = mod.k3_fp8w_lowm_gemm_gated(x, logits.sigmoid(), w, packed)
    out = torch.empty_like(eager)
    gate = torch.empty_like(x)
    side = torch.cuda.Stream()
    fork, join = torch.cuda.Event(), torch.cuda.Event()

    def step():
        fork.record()
        with torch.cuda.stream(side):
            fork.wait()
            torch.sigmoid(logits, out=gate)
            join.record()
        join.wait()
        mod.k3_fp8w_lowm_gemm_gated(x, gate, w, packed, out=out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(3):
        out.zero_()
        gate.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_gated_rejects_unsupported_inputs():
    mod = _kernel_module()
    n, k = 7168, 768
    w_fp8, scale = _make_weight(n, k, seed=2)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    x = torch.zeros((8, k), device="cuda", dtype=torch.bfloat16)
    assert mod.can_run_gated(x, torch.zeros_like(x), w, packed)
    # M above the low-M kernel, wrong dtype / shape, unaligned row stride.
    big = torch.zeros((mod.low_m_max_for(n, k) + 1, k), device="cuda", dtype=torch.bfloat16)
    assert not mod.can_run_gated(big, torch.zeros_like(big), w, packed)
    assert not mod.can_run_gated(x, torch.zeros_like(x, dtype=torch.float32), w, packed)
    assert not mod.can_run_gated(x, torch.zeros((8, k - 128), device="cuda").bfloat16(), w, packed)
    wide = torch.zeros((8, k + 4), device="cuda", dtype=torch.bfloat16)
    assert not mod.can_run_gated(x, wide[:, 4:], w, packed)
    with pytest.raises(ValueError):
        mod.k3_fp8w_lowm_gemm_gated(big, torch.zeros_like(big), w, packed)


@pytest.mark.parametrize("m", [7, 14, 64, 65])
def test_weight_read_linear_forward_gated(m, monkeypatch):
    """``forward_gated`` serves M <= LOW_M_MAX and equals ``forward(x * gate)`` bitwise."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)
    torch.manual_seed(0)
    n, k = 7168, 768
    weight = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
    linear = kimi._Fp8BlockScaleWeightReadLinear(
        *kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(weight), n
    )
    x = _make_activation(m, k, seed=m)
    gate = _make_gate(m, k, seed=m)
    applies = linear.gated_applies(x, gate)
    assert applies == (m <= _kernel_module().low_m_max_for(n, k))
    if applies:
        out = linear.forward_gated(x, gate)
        torch.cuda.synchronize()
        assert torch.equal(out, linear(x * gate))


# ---------------------------------------------------------------------------
# Sigmoid epilogue: torch.sigmoid of the BF16 output (the MLA output gate projection).
# ---------------------------------------------------------------------------


def _sigmoid_all_bf16_kernel():
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    mod = _kernel_module()

    @cute.kernel
    def kernel(src: cute.Tensor, dst: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 256 + tidx
        dst[i] = mod.sigmoid_bf16(src[i])

    @cute.jit
    def launch(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
        kernel(src, dst).launch(grid=(256, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def test_sigmoid_bf16_matches_torch_on_every_bf16():
    """``sigmoid_bf16`` equals ``torch.sigmoid`` on all 65536 BF16 bit patterns."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32, device="cuda").to(torch.int16)
    src = bits.view(torch.bfloat16)
    dst = torch.empty_like(src)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _sigmoid_all_bf16_kernel()(from_dlpack(src), from_dlpack(dst), stream)
    torch.cuda.synchronize()
    ref = torch.sigmoid(src)
    same = dst.view(torch.int16) == ref.view(torch.int16)
    both_nan = torch.isnan(dst) & torch.isnan(ref)
    assert bool((same | both_nan).all()), f"{int((~(same | both_nan)).sum())} BF16 inputs differ"


@pytest.mark.parametrize("n, k", [(768, 7168), (384, 1280), (2112, 7168)])
@pytest.mark.parametrize("m", [1, 7, 14, 32, 64])
def test_sigmoid_epilogue_matches_torch(m, n, k):
    """``k3_fp8w_lowm_gemm_sigmoid`` is ``torch.sigmoid`` of the plain kernel, bitwise."""
    mod = _kernel_module()
    w_pad, scale = _make_padded_weight(n, k, seed=n + k)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = _make_activation(m, k, seed=m) * 0.05
    plain = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    mod.run_k3_fp8w_lowm_gemm(x, w, packed, plain, tactic=mod.default_tactic(m, n, k))
    out = mod.k3_fp8w_lowm_gemm_sigmoid(x, w, packed)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.sigmoid(plain))


@pytest.mark.parametrize("n, k", [(768, 7168), (2112, 7168)])
@pytest.mark.parametrize("m", [1, 7, 14, 16, 28, 32])
def test_sigmoid_smem_request_matches_default(m, n, k):
    """``smem_request`` only changes the launch footprint: bitwise the default launch's output,
    in eager launches and CUDA graph replays."""
    mod = _kernel_module()
    w_pad, scale = _make_padded_weight(n, k, seed=n + k + 1)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = _make_activation(m, k, seed=m + 1) * 0.05
    ref = mod.k3_fp8w_lowm_gemm_sigmoid(x, w, packed)
    out = mod.k3_fp8w_lowm_gemm_sigmoid(x, w, packed, smem_request=mod._SMEM_CAPACITY)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    graph_out = torch.empty_like(ref)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod.k3_fp8w_lowm_gemm_sigmoid(x, w, packed, graph_out, smem_request=mod._SMEM_CAPACITY)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        mod.k3_fp8w_lowm_gemm_sigmoid(x, w, packed, graph_out, smem_request=mod._SMEM_CAPACITY)
    graph_out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, ref)


def test_sigmoid_smem_request_capacity():
    """An ``smem_request`` above the per-block shared memory capacity is refused."""
    mod = _kernel_module()
    n, k, m = 768, 7168, 32
    with pytest.raises(ValueError, match="smem_request"):
        mod.K3Fp8wLowMGemmKernel(
            n,
            k,
            mod.default_tactic(m, n, k),
            True,
            sigmoid=True,
            smem_request=mod._SMEM_CAPACITY + 1,
        )


@pytest.mark.parametrize("m", [7, 14, 64, 65])
def test_weight_read_linear_forward_sigmoid(m, monkeypatch):
    """``forward_sigmoid`` equals ``forward(x).sigmoid()`` bitwise on every path."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)
    torch.manual_seed(0)
    n, k = 768, 7168
    weight = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
    linear = kimi._Fp8BlockScaleWeightReadLinear(
        *kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(weight), n
    )
    x = _make_activation(m, k, seed=m)
    out = linear.forward_sigmoid(x)
    exclusive = linear.forward_sigmoid(x, smem_request=_kernel_module()._SMEM_CAPACITY)
    torch.cuda.synchronize()
    assert torch.equal(out, linear(x).sigmoid())
    assert torch.equal(exclusive, out)


# Kimi-K3 low-M projections (N, K): latent down / up, KDA qkvg, KDA / MLA o_proj, MLA q_a+kv_a,
# MLA q_b, MLA gate, shared expert gate-up / down.
K3_LOW_M_SHAPES = [
    (3584, 7168),
    (7168, 3584),
    (3072, 7168),
    (7168, 768),
    (2112, 7168),
    (18432, 1536),
    (768, 7168),
    (12288, 7168),
    (7168, 6144),
]


def _use_sf_slots(monkeypatch, mod, on):
    """Compile kernels from here on with ``TRTLLM_K3_LOWM_SF_SLOTS`` = ``on`` (fresh caches)."""
    monkeypatch.setattr(mod, "_SF_SLOTS", on)
    for name in ("_compiled_kernel", "_compiled_gated_kernel"):
        monkeypatch.setattr(mod, name, functools.cache(getattr(mod, name).__wrapped__))


def test_sf_slots_fit_tmem_allocation(monkeypatch):
    """Default tactics whose weight ring wraps get several SFB slots inside the unchanged TMEM
    allocation; the others keep one."""
    mod = _kernel_module()
    monkeypatch.setattr(mod, "_SF_SLOTS", True)
    torch.zeros(1, device="cuda")  # the split-K heuristic queries the device's cluster occupancy
    wrapping = 0
    for n, k in K3_LOW_M_SHAPES:
        for m in (8, 16, 32, 64):
            if m > mod.low_m_max_for(n, k):
                continue
            tactics = [mod.default_tactic(m, n, k)]
            try:
                tactics.append(mod.gated_tactic(m, n, k))
            except ValueError:
                pass  # no gated tactic fits this shape
            for tactic in tactics:
                kt_max = -(-(k // mod.CTA_K) // tactic.split_k)
                b_stages = min(tactic.b_stages, kt_max)
                a_stages = min(tactic.a_stages, kt_max)
                slots = mod._sf_slots(tactic.mma_n, b_stages, kt_max, a_stages)
                assert b_stages % slots == 0
                assert tactic.mma_n + mod._SF_TMEM_COLS * (1 + slots) <= mod._TMEM_COLS
                assert (slots > 1) == (kt_max > a_stages)
                wrapping += slots > 1
    assert wrapping > 0


@pytest.mark.parametrize("n, k", K3_LOW_M_SHAPES)
@pytest.mark.parametrize("m", [7, 14, 32, 64])
def test_sf_slots_bitwise(m, n, k, monkeypatch):
    """Per-K-tile TMEM scale slots do not change a single output bit."""
    mod = _kernel_module()
    w_pad, scale = _make_padded_weight(n, k, seed=n + k)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
    x = _make_activation(m, k, seed=m)
    outs = []
    for on in (False, True):
        _use_sf_slots(monkeypatch, mod, on)
        outs.append(_run(mod, x, w, packed))
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("split_k", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "m, b_stages, quant_warps, quant_groups",
    [(5, 4, 4, 4), (5, 8, 4, 4), (40, 2, 4, 2), (40, 4, 8, 4), (40, 6, 8, 2)],
)
def test_sf_slots_split_k_tactics(m, b_stages, quant_warps, quant_groups, split_k, monkeypatch):
    """Two to four slots (B ring depths 2-8, the weight ring wrapping), ranks with unequal K-tile
    counts."""
    mod = _kernel_module()
    n, k = 384, 1280
    w_fp8, scale = _make_weight(n, k, seed=11)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    x = _make_activation(m, k, seed=5)
    tactic = mod.K3LowMTactic(
        mma_n=mod.mma_n_for(m),
        split_k=split_k,
        a_stages=2,
        x_stages=quant_groups * 2,
        b_stages=b_stages,
        quant_warps=quant_warps,
        quant_groups=quant_groups,
    )
    outs = []
    for on in (False, True):
        _use_sf_slots(monkeypatch, mod, on)
        outs.append(_run(mod, x, w, packed, tactic=tactic))
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("m", [7, 32, 64])
def test_sf_slots_epilogue_variants(m, monkeypatch):
    """NVFP4 output, gated and sigmoid kernels with and without scale slots."""
    mod = _kernel_module()
    results = []
    for on in (False, True):
        _use_sf_slots(monkeypatch, mod, on)
        res = []
        n, k = 7168, 768
        w_pad, scale = _make_padded_weight(n, k, seed=1)
        w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad, scale)
        x = _make_activation(m, k, seed=m)
        gsf = torch.full((1,), 448.0 * 6.0 / 64.0, device="cuda", dtype=torch.float32)
        codes = torch.empty((m, n // 2), device="cuda", dtype=torch.uint8)
        sf = torch.empty((m, n // 16), device="cuda", dtype=torch.uint8)
        mod.run_k3_fp8w_lowm_gemm_nvfp4(x, w, packed, gsf, codes, sf)
        res += [codes, sf]
        n, k = 2112, 7168
        w_pad, scale = _make_padded_weight(n, k, seed=2)
        w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad[:n].contiguous(), scale)
        x = _make_activation(m, k, seed=m)
        res.append(mod.k3_fp8w_lowm_gemm_gated(x, _make_gate(m, k, seed=m), w, packed))
        n, k = 768, 7168
        w_pad, scale = _make_padded_weight(n, k, seed=3)
        w, packed = mod.prepare_k3_fp8w_lowm_weight(w_pad, scale)
        res.append(mod.k3_fp8w_lowm_gemm_sigmoid(_make_activation(m, k, seed=m) * 0.05, w, packed))
        results.append(res)
    torch.cuda.synchronize()
    for a, b in zip(*results):
        assert torch.equal(a, b)


@pytest.mark.parametrize("m", [1, 7, 14, 32, 64, 65])
def test_weight_read_linear_forward_mxfp8(m, monkeypatch):
    """``forward_mxfp8`` on the MXFP8 form of ``x`` equals ``forward(x)`` bitwise on every path
    (the MXFP8 GEMM where ``forward`` runs the low-M kernel, ``forward`` itself elsewhere)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)
    torch.manual_seed(0)
    n, k = 3072, 7168
    weight = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
    linear = kimi._Fp8BlockScaleWeightReadLinear(
        *kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(weight), n
    )
    assert linear.takes_mxfp8_input(m) == (m <= _kernel_module().low_m_max_for(n, k))
    x = _make_activation(m, k, seed=m)
    xq, xs = mx.mxfp8_quantize_reference(x)
    out = linear.forward_mxfp8(x, xq, xs)
    torch.cuda.synchronize()
    assert torch.equal(out, linear(x))

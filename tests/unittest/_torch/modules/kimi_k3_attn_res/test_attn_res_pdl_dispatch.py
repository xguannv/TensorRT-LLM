# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-side dispatch of ``TRTLLM_K3_ATTN_RES_PDL`` (``modeling_kimi_linear``).

With the switch on, the decode-sized fused attention-residual calls go to the CuTe DSL kernel
(``trtllm::k3_attn_res_*``) and must return exactly what the C++ path returns with it off; the
deferred MoE residual update (mode 2) must give the same prefix and normalized mix as
materializing it first.
"""

from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN = 7168
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)


def _modules(seed: int):
    torch.manual_seed(seed)
    proj = nn.Linear(HIDDEN, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    nn.init.normal_(proj.weight, std=0.02)
    norm = mkl.KimiK3RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    out_norm = RMSNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN, device="cuda"))
        out_norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN, device="cuda"))
    return proj, norm, out_norm


def _rows(*shape):
    return torch.randn(*shape, device="cuda").to(torch.bfloat16)


def _patched(mode: str):
    return mock.patch.multiple(
        mkl,
        _K3_ATTN_RES_PDL=mode != "0",
        _K3_ATTN_RES_PDL_MODE=mode,
        _ATTN_RES_TOPOLOGY="split",
        _FUSED_ATTN_RES_MAX_TOKENS=32,
    )


def _close_to_cpp(actual: torch.Tensor, expected: torch.Tensor, exact: bool):
    if exact:
        assert torch.equal(actual, expected)
    else:
        # Statistics summed in another order: rare differences of one or two BF16 rounding steps.
        a, e = actual.float(), expected.float()
        bound = (
            torch.maximum(a.abs(), e.abs()) * 2.0**-6
            + 2.0**-8 * e.pow(2).mean(-1, keepdim=True).sqrt()
        )
        assert bool(((a - e).abs() <= bound).all())
        assert float(((a - e) != 0).float().mean()) <= 2e-2


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("snapshots", [1, 3, 4, 8])
@pytest.mark.parametrize("m", [7, 32, 56])
def test_dispatch_matches_cpp(m: int, snapshots: int, exact: bool, monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", exact)
    proj, norm, out_norm = _modules(m + snapshots)
    prefix, addend, block = _rows(m, HIDDEN), _rows(m, HIDDEN), _rows(snapshots, m, HIDDEN)
    with _patched("0"):
        ref_out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
        ref_upd, ref_add_out = mkl._apply_attn_res_add_and_rmsnorm(
            prefix, addend, block, proj, norm, out_norm
        )
    with _patched("1"):
        assert mkl._k3_attn_res_pdl(prefix, None, block, proj, norm, out_norm) is not None
        out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
        upd, add_out = mkl._apply_attn_res_add_and_rmsnorm(
            prefix, addend, block, proj, norm, out_norm
        )
    if m <= 32:
        # Below the split threshold the C++ path is the per-token kernel.
        _close_to_cpp(out, ref_out, exact)
        _close_to_cpp(add_out, ref_add_out, exact)
        assert torch.equal(upd, ref_upd)
    else:
        # Above it the C++ path is the persistent kernel (different summation order).
        for actual, expected in ((out, ref_out), (add_out, ref_add_out), (upd, ref_upd)):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("snapshots", [1, 8])
@pytest.mark.parametrize("m", [7, 32])
@pytest.mark.parametrize("exact", [True, False])
def test_deferred_moe_residual(m: int, snapshots: int, exact: bool, monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", exact)
    proj, norm, out_norm = _modules(m * snapshots)
    deferred = mkl._DeferredMoEResidual(_rows(m, HIDDEN), _rows(m, HIDDEN), _rows(m, HIDDEN))
    block = _rows(snapshots, m, HIDDEN)
    with _patched("2"):
        assert mkl._defer_moe_residual(m)
        assert not mkl._defer_moe_residual(mkl._K3_ATTN_RES_PDL_MAX_M + 1)
        prefix, out = mkl._k3_attn_res_pdl_moe_add(deferred, block, proj, norm, out_norm)
    with _patched("0"):
        ref_prefix = mkl._materialize_prefix_sum(deferred)
        ref_out = mkl._apply_attn_res_and_rmsnorm(ref_prefix, block, proj, norm, out_norm)
    assert torch.equal(prefix, ref_prefix)
    _close_to_cpp(out, ref_out, exact)


class _FakeMoE(nn.Module):
    """Stands in for ``KimiK3MoERuntime``: routed/shared outputs derived from the input."""

    def forward(
        self, hidden_states, all_rank_num_tokens=None, *, residual=None, defer_residual=False
    ):
        routed = (hidden_states.float() * 0.5).to(torch.bfloat16)
        shared = (hidden_states.float() * -0.25).to(torch.bfloat16)
        if defer_residual:
            return mkl._DeferredMoEResidual(residual, routed, shared)
        from tensorrt_llm._torch.modules.kimi_k3_moe_residual import _fused_moe_residual_add

        return _fused_moe_residual_add(routed, shared, residual)


def _fake_layer(layer_idx: int, seed: int):
    layer = mkl.KimiLinearDecoderLayer.__new__(mkl.KimiLinearDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.attn_res_block_size = 12
    layer.is_kda = True
    layer.is_moe = True
    layer.linear_attn = lambda h, metadata: (h.float() * 0.75).to(torch.bfloat16)
    layer.block_sparse_moe = _FakeMoE()
    proj_a, norm_a, layer.input_layernorm = _modules(seed)
    proj_b, norm_b, layer.post_attention_layernorm = _modules(seed + 1)
    layer.self_attention_res_proj, layer.self_attention_res_norm = proj_a, norm_a
    layer.mlp_res_proj, layer.mlp_res_norm = proj_b, norm_b
    return layer


@pytest.mark.parametrize("m", [7, 32])
def test_decoder_layers_with_deferred_moe_residual(m: int, monkeypatch):
    """Two decoder layers: mode 2 (MoE add folded into the next mix) equals the C++ path."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", True)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run(mode):
        h, snaps = hidden, 3
        with _patched(mode):
            for layer in layers:
                h, snaps = layer(h, bank, snaps, None)
                if mode == "2":
                    assert isinstance(h, mkl._DeferredMoEResidual)
            return mkl._materialize_prefix_sum(h)

    assert torch.equal(run("2"), run("0"))


class _FakeKdaMixer:
    """Stands in for the KDA mixer: records the MXFP8 hand-off it is given for each call."""

    def __init__(self):
        self.next_input_mxfp8 = None
        self.seen = []

    def qkvg_takes_mxfp8(self, num_rows):
        return True

    def __call__(self, h, metadata):
        self.seen.append((h, self.next_input_mxfp8))
        self.next_input_mxfp8 = None
        return (h.float() * 0.75).to(torch.bfloat16)


@pytest.mark.parametrize("m", [7, 32])
def test_decoder_layers_mxfp8_handoff(m: int, monkeypatch):
    """``TRTLLM_K3_ATTN_RES_MXFP8``: a KDA layer whose pre-attention mix folds in the previous MoE
    residual hands the mixer the MXFP8 form of its output, bitwise the low-M GEMM's quantization of
    the BF16 rows; the layer outputs do not change, and context batches get no hand-off."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

    monkeypatch.setattr(k3_attn_res, "_EXACT", False)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    for layer in layers:
        layer.linear_attn = _FakeKdaMixer()
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run(mxfp8, num_contexts=0):
        monkeypatch.setattr(mkl, "_K3_ATTN_RES_MXFP8", mxfp8)
        metadata = SimpleNamespace(num_contexts=num_contexts)
        for layer in layers:
            layer.linear_attn.seen.clear()
        h, snaps = hidden, 3
        with _patched("2"):
            for layer in layers:
                h, snaps = layer(h, bank, snaps, metadata)
            return mkl._materialize_prefix_sum(h)

    ref = run(False)
    assert all(x is None for layer in layers for _, x in layer.linear_attn.seen)
    assert torch.equal(run(True, num_contexts=1), ref)
    assert all(x is None for layer in layers for _, x in layer.linear_attn.seen)
    assert torch.equal(run(True), ref)
    # The first layer's input is a plain row (no deferred MoE update), the second one's is not.
    (_, first), (rows, handoff) = layers[0].linear_attn.seen[0], layers[1].linear_attn.seen[0]
    assert first is None and handoff is not None
    ref_q, ref_s = mx.mxfp8_quantize_reference(rows)
    assert torch.equal(handoff[0].view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(handoff[1], ref_s)

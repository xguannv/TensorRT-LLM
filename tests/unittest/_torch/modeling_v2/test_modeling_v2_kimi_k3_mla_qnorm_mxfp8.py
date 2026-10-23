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
"""``TRTLLM_K3LL_MLA_QNORM_MXFP8`` (``kimi_k3_nvfp4/flags.py``) on one GPU: the released q_a norm also writes the
MXFP8 form of its output and Q-B runs ``k3_fp8w_lowm_mxfp8`` on it (``decode_mla_q.q_norm_and_q_b``).

* ``q_norm_pdl.rmsnorm_mxfp8_release_at_launch``: its BF16 output bit for bit ``rmsnorm_release_at_launch``'s, its
  ``(xq, xs)`` bit for bit ``k3_fp8w_lowm_mxfp8.mxfp8_quantize_reference`` of that output, at 1-64 rows, on a
  contiguous q_a and on the q_a columns of the fused q_a / kv_a output (row stride 2112);
* ``q_norm_and_q_b`` with the switch on: the MXFP8 GEMM ran (``MX_STATS``), Q-B's output and the norm bit for bit the
  switch-off path (the released norm and Q-B's low-M GEMM on the BF16 norm); without ``release`` the switch does
  nothing (the module's norm, no MXFP8 kernel);
* captured in a CUDA graph (kernels compiled eagerly first), replayed with new q_a rows: bit for bit the eager
  switch-off path.
"""

import functools

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the low-M FP8 GEMMs run on SM100 / SM103",
)

Q_LORA = 1536
FUSED_A = 2112  # q_a | kv_a | k_pe columns of the fused q_a / kv_a output
Q_B_N = 96 * (128 + 64)  # every head's NoPE + RoPE query columns
ROWS = (1, 8, 16, 32, 64)


def _modules():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
        fp8w_lowm,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx
    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl

    return decode_mla_q, fp8w_lowm, lowm, mx, q_norm_pdl


@functools.cache
def _weight():
    _, _, lowm, _, _ = _modules()
    g = torch.Generator(device="cuda").manual_seed(5)
    w = torch.randn((Q_B_N, Q_LORA), device="cuda", generator=g) * 0.02
    blocks = w.view(Q_B_N // 128, 128, Q_LORA // 128, 128)
    scale = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0))
    )
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(Q_B_N, Q_LORA)
    return lowm.prepare_k3_fp8w_lowm_weight(w_fp8.contiguous(), scale.float())


class _QB(nn.Module):
    """Q-B as ``fp8w_lowm.install`` routes it (its stock forward must not run)."""

    def __init__(self):
        super().__init__()
        _, fl, _, _, _ = _modules()
        self.weight, self.weight_scale = _weight()
        self._locality_domain_weight_shards = None
        self._k3ll_fp8w_lowm = True
        self._k3ll_fp8w_partial_n_tile = False

        def stock(x, **kwargs):
            raise AssertionError("the stock GEMM ran")

        self.forward = functools.partial(fl._forward, self, stock)


@functools.cache
def _norm():
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    norm = RMSNorm(hidden_size=Q_LORA, eps=1e-6, dtype=torch.bfloat16).cuda()
    g = torch.Generator(device="cuda").manual_seed(3)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(Q_LORA, device="cuda", generator=g))
    return norm


def _q_a(rows, seed, strided):
    """q_a rows: the first ``Q_LORA`` columns of a fused q_a / kv_a output (``strided``) or a contiguous tensor."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = FUSED_A if strided else Q_LORA
    ag = (torch.randn((rows, width), device="cuda", generator=g) * 3).bfloat16()
    ag.view(-1)[
        torch.randint(0, ag.numel(), (max(1, ag.numel() // 4096),), device="cuda", generator=g)
    ] = 200.0
    return ag[:, :Q_LORA]


def _same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rows", ROWS)
def test_norm_mxfp8(rows, strided):
    _, fl, _, mx, qn = _modules()
    norm = _norm()
    q_a = _q_a(rows, rows, strided)
    assert qn.can_use_mxfp8(q_a, norm.weight)
    eps = float(norm.variance_epsilon)
    want = qn.rmsnorm_release_at_launch(q_a, norm.weight, eps)
    xq, xs = fl.new_mxfp8_input(rows, Q_LORA, q_a.device)
    got = qn.rmsnorm_mxfp8_release_at_launch(q_a, norm.weight, eps, xq, xs)
    rq, rs = mx.mxfp8_quantize_reference(want)
    torch.cuda.synchronize()
    assert _same(got, want)
    assert torch.equal(xq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(xs, rs)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rows", ROWS)
def test_q_b_on_equals_off(rows, strided):
    dq, fl, _, _, _ = _modules()
    norm, q_b = _norm(), _QB()
    q_a = _q_a(rows, 100 + rows, strided)
    assert fl.takes_mxfp8_input(q_b, rows)
    n_off, q_off, s_off = dq.q_norm_and_q_b(q_a, norm, q_b, True, False)
    before = fl.MX_STATS["mxfp8"]
    n_on, q_on, s_on = dq.q_norm_and_q_b(q_a, norm, q_b, True, True)
    torch.cuda.synchronize()
    assert fl.MX_STATS["mxfp8"] - before == 1, "the MXFP8 GEMM did not run"
    assert (s_off, s_on) == ("+qnorm_pdl", "+qnorm_mxfp8")
    assert _same(n_on, n_off) and _same(q_on, q_off)
    # no release: the switch does nothing
    before = fl.MX_STATS["mxfp8"]
    n_plain, q_plain, s_plain = dq.q_norm_and_q_b(q_a, norm, q_b, False, True)
    torch.cuda.synchronize()
    assert fl.MX_STATS["mxfp8"] == before and s_plain == ""
    assert _same(n_plain, n_off) and _same(q_plain, q_off)


@pytest.mark.parametrize("rows", (8, 32))
def test_graph_replay(rows):
    dq, fl, _, _, _ = _modules()
    norm, q_b = _norm(), _QB()
    q_a = _q_a(rows, 7, True).clone()  # a persistent buffer, written in place below
    dq.q_norm_and_q_b(q_a, norm, q_b, True, True)  # compiles eagerly
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, q_cap, s_cap = dq.q_norm_and_q_b(q_a, norm, q_b, True, True)
    assert s_cap == "+qnorm_mxfp8"
    for seed in (11, 12):
        q_a.copy_(_q_a(rows, seed, False))
        graph.replay()
        torch.cuda.synchronize()
        _, q_ref, _ = dq.q_norm_and_q_b(q_a, norm, q_b, True, False)
        torch.cuda.synchronize()
        assert _same(q_cap, q_ref)

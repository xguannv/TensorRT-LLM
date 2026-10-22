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
"""``TRTLLM_K3LL_DRAFTER_MLA_FUSED``: one MLA DSpark drafter layer's attention (`K3MLADrafterNorms._attention_qkv`) at
the K3 drafter's per-rank shapes, 8 / 16 / 32 rows (1 / 2 / 4 requests of an 8-token block).

* Off (now, or at launch): bit for bit the chain it replaced, rebuilt here from the stock ops (the ``[q_a | kv_a]``
  linear, ``k3_drafter_mla_a_post``, q_b, ``_mla_rope_into`` + the absorption ``bmm_out``, the drafter's cute-dsl
  block decode, the V up-projection einsum).
* On: one ``dspark_mla_attend`` launch per layer; against an fp32 attention over the same pool (non-causal block, as
  the cute-dsl decode's ``kv_bounds``) within ``FP32_TOL`` and no worse than ``WORSE`` times the off arm's error; and
  against the off arm within ``ON_OFF_TOL``.
* Both arms from a captured CUDA graph replayed with rewritten inputs: every replay bit for bit the eager call on the
  same inputs.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    flags,
    mla_drafter,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    mla_drafter_qkv as qkv,
)
from tensorrt_llm._torch.models.modeling_dspark import (
    MLADSparkForCausalLM,
    _build_mla_block_fixup,
    _mla_rope_into,
)

HIDDEN = 7168
NOPE, ROPE, LATENT, V_DIM = 128, 64, 512, 128
BLOCK = 8
PAGE = 64
CTX_LENS = (3000, 517, 4100, 1700)
FP32_TOL = 2e-2  # max |out - fp32| / max |fp32|
WORSE = 2.0
ON_OFF_TOL = 2e-2

needs_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="dspark_mla_attend needs SM100 / SM103",
)


class _Drafter:
    """What `K3MLADrafterNorms`' attention reads of a ``MLADSparkForCausalLM``: its geometry and its cute-dsl block
    decode (the class's own methods)."""

    _cute_dsl_block_decode = MLADSparkForCausalLM._cute_dsl_block_decode
    _assert_cute_dsl_can_implement = MLADSparkForCausalLM._assert_cute_dsl_can_implement
    _mla_decode_workspace = MLADSparkForCausalLM._mla_decode_workspace

    def __init__(self, heads):
        self._num_heads = heads
        self.qk_nope_head_dim = NOPE
        self.qk_rope_head_dim = ROPE
        self.kv_lora_rank = LATENT
        self.softmax_scale = (NOPE + ROPE) ** -0.5
        self._cute_dsl_checked_shape = None
        self._mla_workspace = None

    def _mla_block_decode_variant(self, paged):
        return "cute_dsl"


def _layer(heads, g):
    def w(*shape, scale):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    q_b = torch.nn.Linear(qkv.Q_LORA, heads * (NOPE + ROPE), bias=False, device="cuda").to(
        torch.bfloat16
    )
    q_b.weight.data = w(heads * (NOPE + ROPE), qkv.Q_LORA, scale=qkv.Q_LORA**-0.5)
    attn = SimpleNamespace(
        q_b_proj=q_b,
        q_a_layernorm=SimpleNamespace(
            weight=(1 + 0.1 * torch.rand(qkv.Q_LORA, generator=g, device="cuda")).bfloat16(),
            variance_epsilon=1e-6,
        ),
        kv_a_layernorm=SimpleNamespace(
            weight=(1 + 0.1 * torch.rand(LATENT, generator=g, device="cuda")).bfloat16(),
            variance_epsilon=1e-6,
        ),
        _k_b_proj=w(heads, NOPE, LATENT, scale=0.1),
        _v_b_proj=w(heads, V_DIM, LATENT, scale=0.05),
    )
    a_weight = w(qkv.A_WIDTH, HIDDEN, scale=HIDDEN**-0.5)
    return attn, a_weight


def _case(batch, heads, seed=0):
    """A layer, its block inputs and a paged latent pool holding every request's context (random rows) and room for
    the block."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    attn, a_weight = _layer(heads, g)
    ctx = torch.tensor(CTX_LENS[:batch], dtype=torch.int64, device="cuda")
    per_req = [math.ceil((int(c) + BLOCK) / PAGE) for c in ctx]
    pages = sum(per_req) + 2
    perm = torch.randperm(pages, generator=torch.Generator().manual_seed(seed)).cuda()
    width = max(per_req)
    page_tables = torch.zeros(batch, width, dtype=torch.int64, device="cuda")
    start = 0
    for b, n in enumerate(per_req):
        page_tables[b, :n] = perm[start : start + n]
        start += n
    pool = (
        torch.randn(pages, 1, 1, PAGE, LATENT + ROPE, generator=g, device="cuda") * 0.5
    ).bfloat16()
    normed = (torch.randn(batch * BLOCK, HIDDEN, generator=g, device="cuda")).bfloat16()
    angles = torch.rand(batch, BLOCK, ROPE // 2, generator=g, device="cuda") * 6.28
    blk_freqs = torch.polar(torch.ones_like(angles), angles)
    fixup = _build_mla_block_fixup(ctx, page_tables, BLOCK, PAGE)
    slots = (fixup.pages * pool.stride(0) + fixup.offsets * pool.stride(3)).reshape(-1)
    qkv_state = (
        slots,
        torch.view_as_real(blk_freqs).reshape(batch * BLOCK, -1).contiguous(),
        blk_freqs,
    )
    return SimpleNamespace(
        attn=attn,
        a_weight=a_weight,
        normed=normed,
        pool=pool,
        fixup=fixup,
        qkv=qkv_state,
        batch=batch,
        heads=heads,
        ctx=ctx,
        page_tables=page_tables,
    )


def _owner(heads, fused):
    return mla_drafter.K3MLADrafterNorms(
        _Drafter(heads), None, None, None, frozenset(), a_weights=None, fused=fused
    )


def _run(owner, c):
    return owner._attention_qkv(
        c.attn, c.a_weight, c.normed, c.qkv, c.pool, c.fixup, c.batch, BLOCK
    )


def _reference_chain(drafter, c):
    """The chain before the switch, from the stock ops; returns the output and the absorbed query (the fp32
    reference's input)."""
    attn, batch, heads = c.attn, c.batch, c.heads
    rows = batch * BLOCK
    a = torch.nn.functional.linear(c.normed, c.a_weight)
    slots, freqs, blk_freqs = c.qkv
    q_a = torch.ops.trtllm.k3_drafter_mla_a_post(
        a,
        attn.q_a_layernorm.weight,
        attn.kv_a_layernorm.weight,
        freqs,
        c.pool,
        slots,
        attn.q_a_layernorm.variance_epsilon,
        attn.kv_a_layernorm.variance_epsilon,
    )
    q = attn.q_b_proj(q_a).view(batch, BLOCK, heads, NOPE + ROPE)
    query = torch.empty(batch, BLOCK, heads, LATENT + ROPE, dtype=q.dtype, device=q.device)
    assert _mla_rope_into(q, blk_freqs, query, heads, ROPE, LATENT)
    torch.ops.trtllm.bmm_out(
        q[..., :NOPE].view(rows, heads, NOPE).transpose(0, 1),
        attn._k_b_proj.to(q.dtype),
        query[..., :LATENT].view(rows, heads, LATENT).transpose(0, 1),
    )
    ctx_out = drafter._cute_dsl_block_decode(query, c.pool, c.fixup, BLOCK)
    out = torch.einsum("bshc,hvc->bshv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))
    return out.reshape(rows, -1), query


def _fp32_attention(c, query):
    """fp32 attention of the absorbed query over each request's context and block (non-causal), then V up."""
    pool = c.pool[:, 0, 0].float()
    outs = []
    for b in range(c.batch):
        n = int(c.ctx[b]) + BLOCK
        pages = c.page_tables[b, : math.ceil(n / PAGE)]
        kv = pool[pages].reshape(-1, LATENT + ROPE)[:n]
        qb = query[b].float()  # [BLOCK, H, 576]
        s = torch.einsum("qhd,nd->qhn", qb, kv) * _Drafter(c.heads).softmax_scale
        o = torch.einsum("qhn,nc->qhc", torch.softmax(s, -1), kv[:, :LATENT])
        outs.append(torch.einsum("qhc,hvc->qhv", o, c.attn._v_b_proj.float()))
    return torch.stack(outs).reshape(c.batch * BLOCK, -1)


def _rel(x, ref):
    return ((x.float() - ref.float()).abs().max() / ref.float().abs().max()).item()


def _rewrite(c, seed):
    """New block inputs and context rows in place (same shapes, same addresses), for graph replays."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    c.normed.copy_(torch.randn(c.normed.shape, generator=g, device="cuda").bfloat16())
    c.pool.copy_((torch.randn(c.pool.shape, generator=g, device="cuda") * 0.5).bfloat16())


@needs_sm100
@pytest.mark.parametrize("heads", [4, 6])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_off_is_the_replaced_chain_bit_for_bit(batch, heads, monkeypatch):
    for launch, now in ((False, 1), (True, 0), (False, 0)):
        monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_MLA_FUSED", now)
        c = _case(batch, heads)
        owner = _owner(heads, fused=launch)
        got = _run(owner, c)
        c2 = _case(batch, heads)
        want, _ = _reference_chain(owner.drafter, c2)
        assert torch.equal(got, want), f"launch {launch} now {now}: not bit for bit"
        assert torch.equal(c.pool, c2.pool)
        assert not owner._fused_ran


@needs_sm100
@pytest.mark.parametrize("heads", [4, 6])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_on_matches_fp32_and_the_off_arm(batch, heads, monkeypatch):
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_MLA_FUSED", 1)
    c = _case(batch, heads)
    owner = _owner(heads, fused=True)
    on = _run(owner, c)
    assert owner._fused_ran, "the fused kernel did not run"
    c2 = _case(batch, heads)
    off, query = _reference_chain(owner.drafter, c2)
    ref = _fp32_attention(c2, query)
    e_on, e_off, e_on_off = _rel(on, ref), _rel(off, ref), _rel(on, off)
    print(
        f"[mla_fused] batch {batch} heads {heads}: on {e_on:.2e} off {e_off:.2e} on-off {e_on_off:.2e}"
    )
    assert e_on <= FP32_TOL, e_on
    assert e_on <= max(WORSE * e_off, 4e-3), (e_on, e_off)
    assert e_on_off <= ON_OFF_TOL, e_on_off
    assert torch.isfinite(on).all()


@needs_sm100
@pytest.mark.parametrize("fused_on", [0, 1])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_graph_replay_matches_eager(batch, fused_on, monkeypatch):
    heads = 4
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_MLA_FUSED", fused_on)
    c = _case(batch, heads)
    owner = _owner(heads, fused=True)
    _run(owner, c)  # eager first: builds the operands and the kernel
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out = _run(owner, c)
    torch.cuda.current_stream().wait_stream(stream)
    for r in range(2):
        _rewrite(c, 100 + r)
        pool0 = c.pool.clone()
        graph.replay()
        torch.cuda.synchronize()
        got = out.clone()
        pool_after = c.pool.clone()
        c.pool.copy_(pool0)
        want = _run(owner, c)
        assert torch.equal(got, want), f"replay {r}: not bit for bit the eager call"
        assert torch.equal(pool_after, c.pool)
    assert bool(owner._fused_ran) == bool(fused_on)


@needs_sm100
def test_capture_without_an_eager_build_keeps_the_separate_kernels(monkeypatch):
    """Under capture, a build that has not run eagerly is refused: the graph holds the separate kernels."""
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_MLA_FUSED", 1)
    c = _case(1, 4)
    owner = _owner(4, fused=True)
    owner._fused_operands(c.attn, capturing=False)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out = _run(owner, c)
    torch.cuda.current_stream().wait_stream(stream)
    assert not owner._fused_ran
    _rewrite(c, 7)
    pool0 = c.pool.clone()
    graph.replay()
    torch.cuda.synchronize()
    got = out.clone()
    c.pool.copy_(pool0)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_MLA_FUSED", 0)
    assert torch.equal(got, _run(owner, c))


@needs_sm100
@pytest.mark.parametrize("fused_on", [0, 1])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_drafter_gemm_in_the_attention(batch, fused_on, monkeypatch):
    """``TRTLLM_K3LL_DRAFTER_GEMM`` in the same layer step: ``[q_a | kv_a]`` (no decode GEMV site here) and q_b on the
    pinned low-M GEMM, with and without the fused attention, against the off arms within ``ON_OFF_TOL``; off, bit
    for bit."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_drafter_gemm,
    )

    heads = 4
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_MLA_FUSED", fused_on)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_GEMM", 0)
    c = _case(batch, heads)
    owner = mla_drafter.K3MLADrafterNorms(
        _Drafter(heads),
        None,
        None,
        None,
        frozenset(),
        fused=True,
        gemm=mla_drafter_gemm.DrafterGemm(),
    )
    off = _run(owner, c)
    assert not owner.gemm._ran
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_GEMM", 1)
    c2 = _case(batch, heads)
    on = _run(owner, c2)
    assert {key[0] for key in owner.gemm._ran} == {"q_kv_a", "q_b"}
    e = _rel(on, off)
    print(f"[drafter_gemm] batch {batch} fused {fused_on}: on-off {e:.2e}")
    assert e <= ON_OFF_TOL, e

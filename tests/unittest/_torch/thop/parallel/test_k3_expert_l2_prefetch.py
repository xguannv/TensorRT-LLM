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
"""``k3_expert_l2_prefetch``: the routing record matches ``moe_sort``, the selection is the
top-``budget`` entries by rows, and the prefetch leaves memory untouched eagerly, on a side stream
and under CUDA graph replay, also from a garbage record."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

E, TOPK, NLOC = 896, 16, 224


def _mod():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_expert_l2_prefetch

    return k3_expert_l2_prefetch


def _routing(m, seed, groups=4):
    gen = torch.Generator().manual_seed(seed)
    # Correlated tokens (as the draft tokens of one request): each token keeps most of a core set.
    core = torch.randperm(E, generator=gen)[:TOPK]
    ids = []
    for _ in range(m):
        keep = core[torch.rand(TOPK, generator=gen) < 0.7]
        rest = torch.tensor([e for e in torch.randperm(E, generator=gen).tolist() if e not in keep])
        ids.append(torch.cat([keep, rest[: TOPK - keep.numel()]]))
    ids = torch.stack(ids).to(torch.int32).cuda()
    scales = torch.rand(m, TOPK, generator=gen).softmax(-1).float().cuda()
    return ids, scales


def _ref_select(entries, budget, num_local):
    """entries: [(expert, rows)] in record order."""
    keys = [
        (-min(rows, 255), expert, t)
        for t, (expert, rows) in enumerate(entries)
        if rows > 0 and 0 <= expert < num_local
    ]
    sel = [expert for _, expert, _ in sorted(keys)[:budget]]
    return sel + [-1] * (budget - len(sel))


@pytest.mark.parametrize("m", [1, 7, 14, 32])
@pytest.mark.parametrize("group", [0, 3])
def test_record_matches_moe_sort(m, group):
    mod = _mod()
    ids, scales = _routing(m, 10 * m + group)
    srt = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=E,
        top_k=TOPK,
        local_expert_offset=group * NLOC,
        local_num_experts=NLOC,
        tile_tokens_dim=128,
    )
    t2g, t2lim, nne = srt[0], srt[1], srt[5]
    rec = torch.full((mod.RECORD_INTS,), -7, dtype=torch.int32, device="cuda")
    mod.run_k3_expert_l2_prefetch(num_local=NLOC, record_out=rec, sort_outputs=(t2g, t2lim, nne))
    torch.cuda.synchronize()
    n = int(nne[0])
    assert int(rec[0]) == n
    entries = rec[1 : 1 + 2 * n].view(n, 2).tolist()
    assert [e for e, _ in entries] == t2g[:n].tolist()
    # rows of each entry = tokens of this rank routed to the expert (one tile per expert)
    local = ids - group * NLOC
    counts = {}
    for e in local[(local >= 0) & (local < NLOC)].tolist():
        counts[e] = counts.get(e, 0) + 1
    assert {e: r for e, r in entries} == counts
    # untouched past the entries
    assert torch.all(rec[1 + 2 * n :] == -7)


@pytest.mark.parametrize("budget", [1, 8, 16, 32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection(budget, seed):
    mod = _mod()
    gen = torch.Generator().manual_seed(seed)
    n = int(torch.randint(0, 200, (1,), generator=gen))
    experts = torch.randperm(NLOC + 20, generator=gen)[:n] - 10  # some invalid (< 0 or >= NLOC)
    rows = torch.randint(-2, 20, (n,), generator=gen)  # some empty
    rows[: n // 4] = 7  # ties
    rec = mod.new_record("cuda")
    rec[0] = n
    if n:
        rec[1 : 1 + 2 * n] = torch.stack([experts, rows], 1).reshape(-1).to(torch.int32).cuda()
    sel = torch.full((budget,), -9, dtype=torch.int32, device="cuda")
    w = torch.zeros(NLOC, 64, dtype=torch.uint8, device="cuda")
    mod.run_k3_expert_l2_prefetch(
        num_local=NLOC,
        record_in=rec,
        ranges=mod.expert_ranges([w], NLOC),
        budget=budget,
        sel_out=sel,
        ctas=3,
    )
    torch.cuda.synchronize()
    assert sel.tolist() == _ref_select(list(zip(experts.tolist(), rows.tolist())), budget, NLOC)


def _weights(num_local, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shapes = ((num_local, 96, 448), (num_local, 96, 56), (num_local, 448, 48), (num_local, 448, 8))
    return [
        torch.randint(0, 255, s, generator=gen, dtype=torch.uint8, device="cuda") for s in shapes
    ]


def test_prefetch_untouched_eager_side_stream_graph():
    mod = _mod()
    ws = _weights(NLOC, 0)
    refs = [w.clone() for w in ws]
    ranges = mod.expert_ranges(ws, NLOC)
    assert ranges[0] == (ws[0].data_ptr(), 96 * 448, 96 * 448)
    ids, scales = _routing(14, 3)
    srt = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=E,
        top_k=TOPK,
        local_expert_offset=NLOC,
        local_num_experts=NLOC,
        tile_tokens_dim=128,
    )
    sort_outputs = (srt[0], srt[1], srt[5])
    rec_a, rec_b = mod.new_record("cuda"), mod.new_record("cuda")
    sel = torch.empty(16, dtype=torch.int32, device="cuda")
    kw = dict(num_local=NLOC, ranges=ranges, budget=16, sel_out=sel, chunk=4096)
    # eager: record a, prefetch from the (empty) record b, then the other way round
    mod.run_k3_expert_l2_prefetch(
        record_out=rec_a, sort_outputs=sort_outputs, record_in=rec_b, **kw
    )
    torch.cuda.synchronize()
    assert sel.tolist() == [-1] * 16
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mod.run_k3_expert_l2_prefetch(
            record_out=rec_b, sort_outputs=sort_outputs, record_in=rec_a, **kw
        )
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    n = int(rec_a[0])
    expect = _ref_select(rec_a[1 : 1 + 2 * n].view(n, 2).tolist(), 16, NLOC)
    assert sel.tolist() == expect and expect[0] >= 0
    assert torch.equal(rec_a, rec_b)
    # CUDA graph: the record is written at replay, the prefetch reads the record of the last replay
    graph = torch.cuda.CUDAGraph()
    rec_c = mod.new_record("cuda")
    sel.fill_(-5)
    with torch.cuda.graph(graph):
        mod.run_k3_expert_l2_prefetch(
            record_out=rec_c, sort_outputs=sort_outputs, record_in=rec_c, **kw
        )
    torch.cuda.synchronize()
    assert int(rec_c[0]) == 0 and sel.tolist() == [-5] * 16  # captured, not run
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(rec_c, rec_a)
    graph.replay()
    torch.cuda.synchronize()
    assert sel.tolist() == expect
    # a garbage record selects valid experts only and touches nothing
    junk = torch.randint(-(2**31), 2**31 - 1, (mod.RECORD_INTS,), dtype=torch.int32, device="cuda")
    junk[0] = 300
    mod.run_k3_expert_l2_prefetch(record_in=junk, **kw)
    torch.cuda.synchronize()
    assert all(-1 <= e < NLOC for e in sel.tolist())
    for w, r in zip(ws, refs):
        assert torch.equal(w, r)


def test_expert_ranges_and_arguments():
    mod = _mod()
    w = torch.empty(8, 32, 64, dtype=torch.uint8, device="cuda")
    assert mod.expert_ranges([w], 8) == [(w.data_ptr(), 32 * 64, 32 * 64)]
    # padded per-expert stride: the slice stays dense
    big = torch.empty(8, 40, 64, dtype=torch.uint8, device="cuda")
    assert mod.expert_ranges([big[:, :32]], 8) == [(big.data_ptr(), 40 * 64, 32 * 64)]
    with pytest.raises(ValueError):
        mod.expert_ranges([w[:, :, :32]], 8)  # not dense
    with pytest.raises(ValueError):
        mod.expert_ranges([w], 7)  # wrong expert count
    with pytest.raises(ValueError):
        mod.K3ExpertL2PrefetchKernel(NLOC)  # nothing to do
    with pytest.raises(ValueError):
        mod.K3ExpertL2PrefetchKernel(NLOC, (64,), budget=mod.MAX_BUDGET + 1)
    with pytest.raises(ValueError):
        mod.K3ExpertL2PrefetchKernel(mod.MAX_LOCAL + 1, (64,), budget=8)
    # nothing requested: no launch, no error
    mod.run_k3_expert_l2_prefetch(num_local=NLOC)


class _Backend:
    """What ``_expert_l2_weight_ranges`` reads from the CuteDSL NVFP4 backend."""

    def __init__(self, num_local, seed):
        w1, w1s, w2, w2s = _weights(num_local, seed)
        self.w3_w1_weight, self.w2_weight = w1, w2
        self.quant_scales = type("Scales", (), {"fc1_weight_block": w1s, "fc2_weight_block": w2s})()


def _runtimes(n_layers, num_local=NLOC):
    import weakref

    from torch import nn

    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    group = weakref.WeakValueDictionary()
    state = mkl._ExpertL2PrefetchState()
    rts = []
    for i in range(n_layers):
        rt = mkl.KimiK3MoERuntime.__new__(mkl.KimiK3MoERuntime)
        nn.Module.__init__(rt)
        rt.layer_idx = 1 + i
        rt.experts_per_rank = num_local
        rt.routed_experts = type("Moe", (), {"backend": _Backend(num_local, i)})()
        rt._moe_group = group
        group[rt.layer_idx] = rt
        rt._expl2_state = state
        rt._expl2_event = torch.cuda.Event()
        rt._expl2_attn_event = torch.cuda.Event()
        rts.append(rt)
    return rts, state


def _route_outs(m, seed, group=1):
    ids, scales = _routing(m, seed)
    srt = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=E,
        top_k=TOPK,
        local_expert_offset=group * NLOC,
        local_num_experts=NLOC,
        tile_tokens_dim=128,
    )
    return (ids, scales, *srt), (group * NLOC, NLOC)


def test_runtime_hook_records_prefetches_and_joins(monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_expert_l2_prefetch as xpf
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    monkeypatch.setattr(mkl, "_K3_EXPERT_L2_PREFETCH", 8)
    rts, state = _runtimes(3)
    calls = []
    real = xpf.run_k3_expert_l2_prefetch

    def spy(**kw):
        calls.append((torch.cuda.current_stream(), kw))
        real(**kw)

    monkeypatch.setattr(xpf, "run_k3_expert_l2_prefetch", spy)

    def forward(m, routes):
        """Per layer: the attention block's prefetch point, the routing, the record after the
        routed FC2, and the join site (a no-op but in the last layer)."""
        for rt, (outs, target) in zip(rts, routes):
            rt.issue_expert_l2_prefetch(m)
            rt._attach_route(outs, target)
            assert rt._expl2_route is outs
            rt._record_expert_l2_routing(m)
            assert rt._expl2_route is None
            rt._join_expert_l2_prefetch()

    with with_multi_stream(True):
        # too many tokens / no routing-tail outputs: nothing issued, nothing allocated
        outs, target = _route_outs(40, 0)
        rts[0]._attach_route(outs, target)
        rts[0]._record_expert_l2_routing(40)
        rts[0]._record_expert_l2_routing(14)
        rts[0].issue_expert_l2_prefetch(14)
        assert not calls and rts[0]._expl2_record is None
        # eager (the warmup): records allocated for every layer and written; nothing recorded
        # before, so nothing prefetched; the stream is joined by the last layer
        forward(14, [_route_outs(14, seed) for seed in (1, 2, 3)])
        assert state.stream is not None and not state.pending
        assert all(rt._expl2_record is not None and rt._expl2_recorded for rt in rts)
        assert all(s == state.stream for s, _ in calls)
        assert [kw.get("record_out") is rt._expl2_record for (_, kw), rt in zip(calls, rts)] == [
            True
        ] * 3
        assert all(kw.get("ctas") == 1 and "record_in" not in kw for _, kw in calls)
        torch.cuda.synchronize()
        # captured: each layer prefetches its own experts from its record, then records again
        calls.clear()
        routes = [_route_outs(14, seed) for seed in (4, 5, 6)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(14, routes)
        assert len(calls) == 6 and not state.pending
        assert all(s == state.stream for s, _ in calls)
        for i, rt in enumerate(rts):
            pf = calls[2 * i][1]
            assert pf["record_in"] is rt._expl2_record and pf["budget"] == 8
            assert pf["ranges"] == rt._expert_l2_weight_ranges()
            assert calls[2 * i + 1][1]["record_out"] is rt._expl2_record
    graph.replay()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    # the records hold the routing captured in the graph
    for rt, (outs, _) in zip(rts, routes):
        n = int(outs[7][0])
        assert int(rt._expl2_record[0]) == n
        assert rt._expl2_record[1 : 1 + 2 * n : 2].tolist() == outs[2][:n].tolist()
    # backend weights unchanged by the prefetches
    for i, rt in enumerate(rts):
        ref = _Backend(NLOC, i)
        assert torch.equal(rt.routed_experts.backend.w3_w1_weight, ref.w3_w1_weight)
        assert torch.equal(rt.routed_experts.backend.w2_weight, ref.w2_weight)
    ranges = rts[0]._expert_l2_weight_ranges()
    assert ranges and ranges[0][1] == rts[0].routed_experts.backend.w3_w1_weight[0].numel()


def test_runtime_hook_off():
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    rts, state = _runtimes(2)
    for rt in rts:
        rt._expl2_state = None
    outs, target = _route_outs(14, 0)
    with with_multi_stream(True):
        rts[0].issue_expert_l2_prefetch(14)
        rts[0]._attach_route(outs, target)
        assert rts[0]._expl2_route is None
        rts[0]._record_expert_l2_routing(14)
        rts[1]._join_expert_l2_prefetch()
    assert rts[0]._expl2_record is None and state.stream is None
    assert mkl.KimiK3MoERuntime._expl2_state is None
    # the attention modules carry no hook by default
    from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import KimiK3MLAAttention
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

    assert KimiKDALinearAttention.expert_l2_prefetch_hook is None
    assert KimiK3MLAAttention.expert_l2_prefetch_hook is None


@pytest.mark.parametrize("budget", [0, 8])
def test_decoder_layer_wires_the_hook(monkeypatch, budget):
    """KimiLinearDecoderLayer hands its attention block the MoE block's prefetch entry point
    (KDA: the mixer itself; MLA: KimiMLARuntime.mixer) only when the switch is on."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    class _Attn(nn.Module):
        expert_l2_prefetch_hook = None

        def __init__(self, *args, **kwargs):
            super().__init__()

    class _MLA(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.mixer = _Attn()

    class _MoE(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def issue_expert_l2_prefetch(self, num_tokens):
            pass

    monkeypatch.setattr(mkl, "KimiKDALinearAttention", _Attn)
    monkeypatch.setattr(mkl, "KimiMLARuntime", _MLA)
    monkeypatch.setattr(mkl, "KimiK3MoERuntime", _MoE)
    monkeypatch.setattr(mkl, "_K3_EXPERT_L2_PREFETCH", budget)
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1), quant_config=QuantConfig()
    )
    cfg = SimpleNamespace(
        hidden_size=256,
        intermediate_size=512,
        num_experts=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        linear_attn_config={"kda_layers": [1, 2], "full_attn_layers": [3]},
        rms_norm_eps=1e-5,
        attn_res_block_size=1,
    )
    aux = {t: object() for t in AuxStreamType}
    kda = mkl.KimiLinearDecoderLayer(model_config, cfg, layer_idx=1, aux_stream_dict=aux)
    mla = mkl.KimiLinearDecoderLayer(model_config, cfg, layer_idx=2, aux_stream_dict=aux)
    assert kda.is_moe and kda.is_kda and mla.is_moe and not mla.is_kda
    if budget:
        assert (
            kda.linear_attn.expert_l2_prefetch_hook == kda.block_sparse_moe.issue_expert_l2_prefetch
        )
        assert (
            mla.self_attn.mixer.expert_l2_prefetch_hook
            == mla.block_sparse_moe.issue_expert_l2_prefetch
        )
    else:
        assert kda.linear_attn.expert_l2_prefetch_hook is None
        assert mla.self_attn.mixer.expert_l2_prefetch_hook is None

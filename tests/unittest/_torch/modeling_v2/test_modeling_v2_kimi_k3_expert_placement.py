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
"""``TRTLLM_K3LL_EXPERT_PLACEMENT`` (kimi_k3_nvfp4/expert_placement.py) and ``TRTLLM_K3LL_ROUTING_STATS_DIR``
(kimi_k3_nvfp4/routing_stats.py) of the Kimi K3 NVFP4 target.

  host        : the placement file's checks; the per-expert loaders read checkpoint expert ``e`` into a slot whose
                routed id is the backend's; the router rows and bias permuted in place.
  routing     (one GPU, SM100 / SM103) : the router GEMM on permuted rows gives the same logits in permuted columns,
                bit for bit; the generic routing (noaux_tc) and ``k3_route_quant`` on them return slot ids that map
                back to the same checkpoint experts with bit-identical weights (and latent).
  experts     (one GPU, SM100 / SM103) : four simulated EP ranks of ``k3_moe_nvfp4_lowm`` holding the experts by a
                random placement against the contiguous ones: every (token, expert) contribution bit for bit, the
                full sum (the ranks' partials hold other experts, so another bf16 summation order) within the bf16
                accumulation bound of the exact sum.
  stats       (one GPU) : the recorder's counts, eagerly and from CUDA-graph replays, mapped back to checkpoint
                experts through the placement.
  front       (4 GPUs, one process each) : ``k3_moe_front`` with the router rows permuted (each TP rank's 224-row
                slice of the permuted router): ids map back to the same checkpoint experts, weights, NVFP4 latent and
                shared activation bit for bit.
"""

import json
import pickle
import sys
import traceback
import types

import numpy as np
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    expert_placement as placement,
)

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the multi-GPU check is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

NUM_EXPERTS, EP, TOP_K, HIDDEN, LATENT = 896, 4, 16, 7168, 3584
RSF = 2.827
GATE_CAP, LINEAR_CAP = 4.0, 25.0
SF16 = 16


def _trtllm():
    """``torch.ops.trtllm`` (named only in a nested function, see test_k3_moe_front.py)."""

    def namespace():
        return torch.ops.trtllm

    return namespace()


def _sm_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


_needs_sm100 = pytest.mark.skipif(not _sm_ok(), reason="needs sm_100 or sm_103")


def _random_placement(num_experts: int, seed: int) -> list:
    """A random permutation that moves most experts to another EP group."""
    rng = np.random.default_rng(seed)
    return [int(e) for e in rng.permutation(num_experts)]


def _write(tmp_path, name: str, **fields) -> str:
    data = {"format": placement.FORMAT_V1, "num_experts": NUM_EXPERTS, "ep_size": EP}
    data.update(fields)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


# --------------------------------------------------------------------------------------------------------------------
# host
# --------------------------------------------------------------------------------------------------------------------
def test_load_placement(tmp_path):
    perm = _random_placement(NUM_EXPERTS, 1)
    identity = list(range(NUM_EXPERTS))
    path = _write(tmp_path, "v1.json", expert_of_slot={"1": perm, "2": identity})
    tables = placement.load_placement(path, NUM_EXPERTS, EP)
    assert tables == {1: tuple(perm)}, "identity layers are dropped, permuted ones kept as given"
    v2 = _write(
        tmp_path,
        "v2.json",
        format=placement.FORMAT_V2,
        replica_slots=8,
        expert_of_slot={"5": perm},
        replicas={"5": [[0, 1]]},
    )
    assert placement.load_placement(v2, NUM_EXPERTS, EP) == {5: tuple(perm)}
    assert placement.local_checkpoint_experts(range(224, 448), None) == tuple(range(224, 448))
    assert placement.local_checkpoint_experts(range(224, 448), perm) == tuple(perm[224:448])


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"format": "kimi_k3_expert_placement/v9"}, "expected format"),
        ({"num_experts": 384}, "placement is for 384 experts"),
        ({"ep_size": 16}, "balanced for ep_size=16"),
        ({"expert_of_slot": {"1": [0] * NUM_EXPERTS}}, "not a permutation"),
        ({"expert_of_slot": {"1": list(range(NUM_EXPERTS - 1))}}, "not a permutation"),
        ({"expert_of_slot": {"one": list(range(NUM_EXPERTS))}}, "not an integer"),
        ({"expert_of_slot": [list(range(NUM_EXPERTS))]}, "must map layer indices"),
    ],
    ids=["format", "num_experts", "ep_size", "duplicate", "short", "layer_key", "not_a_dict"],
)
def test_load_placement_rejects(tmp_path, fields, match):
    fields.setdefault("expert_of_slot", {"1": list(range(NUM_EXPERTS))})
    path = _write(tmp_path, "bad.json", **fields)
    with pytest.raises(ValueError, match=match):
        placement.load_placement(path, NUM_EXPERTS, EP)


def test_check_layers():
    tables = {1: (0,), 92: (0,), 200: (0,)}
    # Layer 200 is past the model (a layer-truncated debug model): ignored.
    placement.check_layers(tables, range(1, 93), 93, "p.json")
    with pytest.raises(ValueError, match=r"permutes layers \[0\]"):
        placement.check_layers({0: (0,), 1: (0,)}, range(1, 93), 93, "p.json")


def test_permute_router_rows():
    gen = torch.Generator().manual_seed(3)
    weight = torch.randn(NUM_EXPERTS, 64, generator=gen).bfloat16()
    bias = torch.randn(NUM_EXPERTS, generator=gen)
    w0, b0 = weight.clone(), bias.clone()
    perm = _random_placement(NUM_EXPERTS, 4)
    ptr = weight.data_ptr()
    placement.permute_router_rows(weight, bias, perm)
    assert weight.data_ptr() == ptr, "permuted in place"
    for slot in (0, 1, 223, 224, 895):
        assert torch.equal(weight[slot], w0[perm[slot]]) and bias[slot] == b0[perm[slot]]
    assert torch.equal(weight, w0[perm]) and torch.equal(bias, b0[perm])


class _RecordingQuantMethod:
    def __init__(self):
        self.calls = []

    def load_streaming_nvfp4_expert(self, backend, *, global_expert_id, local_slot_id, **tensors):
        self.calls.append((global_expert_id, local_slot_id, sorted(set(tensors.values()))))

    load_packed_mxfp4_expert = load_streaming_nvfp4_expert


@pytest.mark.parametrize("loader_name", ["_load_nvfp4_expert", "_load_packed_mxfp4_expert"])
def test_expert_loader_slot_and_key(loader_name):
    """The per-expert loader reads the checkpoint expert it is given into the slot, whose routed id is the
    backend's: what the inherited loader does with ``KimiK3MoERuntime.local_expert_ids`` under a placement."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        implementation as impl,
    )

    perm = _random_placement(NUM_EXPERTS, 5)
    backend = types.SimpleNamespace(
        initial_local_expert_ids=list(range(224, 448)), quant_method=_RecordingQuantMethod()
    )
    ckpt = placement.local_checkpoint_experts(backend.initial_local_expert_ids, perm)
    loader = getattr(impl, loader_name)
    for local_slot_id in (0, 7, 223):
        loader(backend, "L.experts", ckpt[local_slot_id], local_slot_id, lambda key: key)
    for (global_id, slot, keys), local_slot_id in zip(backend.quant_method.calls, (0, 7, 223)):
        assert (global_id, slot) == (224 + local_slot_id, local_slot_id)
        prefixes = {k.rsplit(".", 2)[0] for k in keys}
        assert prefixes == {f"L.experts.{perm[224 + local_slot_id]}"}, prefixes


# --------------------------------------------------------------------------------------------------------------------
# routing (one GPU)
# --------------------------------------------------------------------------------------------------------------------
def _gate_config():
    return types.SimpleNamespace(
        num_experts_per_token=TOP_K,
        num_experts=NUM_EXPERTS,
        routed_scaling_factor=RSF,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        hidden_size=HIDDEN,
    )


def _gates(perm):
    """The target's gate with random bf16 rows and fp32 bias, and a copy permuted to ``perm``'s slot order."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        implementation as impl,
    )

    g = torch.Generator(device="cuda").manual_seed(71)
    gates = []
    for _ in range(2):
        gate = impl.KimiK3MoEGate(_gate_config(), logits_gemm_dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            gate.weight.copy_(0.02 * torch.randn(gate.weight.shape, generator=g, device="cuda"))
            gate.e_score_correction_bias.copy_(
                0.05 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
            )
        gates.append(gate)
        g.manual_seed(71)
    placement.permute_router_rows(gates[1].weight.data, gates[1].e_score_correction_bias.data, perm)
    return gates


@_needs_sm100
@pytest.mark.parametrize("m", [1, 8, 16, 32, 64])
def test_routing_maps_back(m):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import (
        op,  # noqa: F401  (registers the op)
    )

    perm = _random_placement(NUM_EXPERTS, 10 + m)
    index = torch.tensor(perm, device="cuda")
    gate, placed = _gates(perm)
    gen = torch.Generator(device="cuda").manual_seed(100 + m)
    x = torch.randn(m, HIDDEN, device="cuda", generator=gen).bfloat16()
    logits, logits_p = gate.compute_logits(x), placed.compute_logits(x)
    assert torch.equal(logits_p, logits[:, index]), (
        "router logits are not the permuted columns bit for bit"
    )

    ids, w = gate.routing_method.apply(logits)
    ids_p, w_p = placed.routing_method.apply(logits_p)
    _same_selection(index[ids_p.long()], w_p, ids, w, "generic routing (noaux_tc)")

    latent = (torch.randn(m, LATENT, device="cuda", generator=gen) * 0.3).bfloat16()
    gsf = torch.tensor([448.0 * 6.0 / float(latent.abs().max())], device="cuda")
    bias, bias_p = gate.e_score_correction_bias, placed.e_score_correction_bias
    ref = _trtllm().k3_route_quant(logits, bias, latent, RSF, False, gsf)
    new = _trtllm().k3_route_quant(logits_p, bias_p, latent, RSF, False, gsf)
    torch.cuda.synchronize()
    _same_selection(index[new[0].long()], new[1], ref[0], ref[1], "k3_route_quant")
    for a, b, what in zip(new[2:], ref[2:], ("latent codes", "latent scales")):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
            f"k3_route_quant: {what} differ"
        )


def _same_selection(ckpt_ids, weights, ref_ids, ref_weights, what):
    """Each token selects the same checkpoint experts with bit-identical weights (compared in checkpoint-id order,
    so an op that orders its top-k by id, not by score, compares too)."""
    a_ids, a_order = ckpt_ids.long().sort(dim=-1)
    b_ids, b_order = ref_ids.long().sort(dim=-1)
    assert torch.equal(a_ids, b_ids), f"{what}: slot ids map to other checkpoint experts"
    a_w, b_w = weights.gather(-1, a_order), ref_weights.gather(-1, b_order)
    assert torch.equal(a_w.contiguous().view(torch.uint8), b_w.contiguous().view(torch.uint8)), (
        f"{what}: weights differ"
    )


# --------------------------------------------------------------------------------------------------------------------
# experts (one GPU, four simulated EP ranks)
# --------------------------------------------------------------------------------------------------------------------
# The small lowm shape of test_k3_moe_lowm.py, as EP4: 64 experts, 16 per rank.
S_HIDDEN, S_INTER, S_EXPERTS, S_TOP_K = 1024, 256, 64, 8
S_LOCAL = S_EXPERTS // EP


def _quantize(t):
    gsf = t.abs().max().float() / (448 * 6)
    q, sf = _trtllm().fp4_quantize(t, 1 / gsf, SF16, False)
    return q, sf, gsf


def _checkpoint_experts(seed):
    """All experts of one layer as NVFP4 in the CuteDSL backend's buffer shapes (each expert's 128x4-blocked scales
    a whole block of rows, so selecting experts keeps them intact), with per-expert alphas."""
    torch.manual_seed(seed)
    w1 = torch.randn(S_EXPERTS, 2 * S_INTER, S_HIDDEN, device="cuda", dtype=torch.bfloat16)
    w1, w1_sf, w1_gsf = _quantize(w1)
    w2 = torch.randn(S_EXPERTS, S_HIDDEN, S_INTER, device="cuda", dtype=torch.bfloat16)
    w2, w2_sf, w2_gsf = _quantize(w2)
    fc2_input_scale = torch.tensor([448 * 6 / 4.0], dtype=torch.float32, device="cuda")
    spread = torch.linspace(0.5, 1.5, S_EXPERTS, device="cuda")
    return types.SimpleNamespace(
        w1=w1.view(torch.uint8).view(S_EXPERTS, 2 * S_INTER, -1),
        w1_sf=w1_sf.view(torch.uint8).view(S_EXPERTS, 2 * S_INTER, -1),
        w2=w2.view(torch.uint8).view(S_EXPERTS, S_HIDDEN, -1),
        w2_sf=w2_sf.view(torch.uint8).view(S_EXPERTS, S_HIDDEN, -1),
        w1_gsf=w1_gsf,
        alpha2=(spread * (w2_gsf / fc2_input_scale[0])).float(),
        spread=spread,
        fc2_input_scale=fc2_input_scale,
    )


def _rank_weights(ex, experts, x_gsf):
    """One EP rank's expert buffers holding checkpoint experts ``experts`` (in slot order)."""
    sel = torch.tensor(experts, device="cuda")
    alpha1 = (ex.spread * (x_gsf * ex.w1_gsf)).float()
    return tuple(
        t.index_select(0, sel).contiguous()
        for t in (ex.w1, ex.w1_sf, alpha1, ex.w2, ex.w2_sf, ex.alpha2)
    )


@_needs_sm100
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 32])
def test_ep4_contributions(num_tokens):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op
    from tensorrt_llm._torch.utils import unswizzle_sf

    ok, why = nvfp4_lowm_op.supported(torch.device("cuda"), 2 * S_INTER, S_HIDDEN, num_tokens)
    assert ok, why
    perm = _random_placement(S_EXPERTS, 20 + num_tokens)
    slot_of = torch.empty(S_EXPERTS, dtype=torch.int32, device="cuda")
    slot_of[torch.tensor(perm, device="cuda")] = torch.arange(
        S_EXPERTS, dtype=torch.int32, device="cuda"
    )
    torch.manual_seed(2000 + num_tokens)
    logits = torch.randn(num_tokens, S_EXPERTS, device="cuda")
    scales, ids = logits.topk(S_TOP_K, dim=-1)
    ids = ids.to(torch.int32)  # checkpoint expert ids
    scales = scales.softmax(dim=-1).float()
    ids_p = slot_of[ids.long()]  # the slot ids the permuted router returns
    x = torch.randn(num_tokens, S_HIDDEN, device="cuda", dtype=torch.bfloat16)
    x_q, x_sf, x_gsf = _quantize(x)
    x_sf = unswizzle_sf(x_sf, (num_tokens + 127) // 128 * 128, S_HIDDEN)[:num_tokens].contiguous()
    x_q, x_sf = x_q.view(torch.uint8), x_sf.view(torch.uint8)
    ex = _checkpoint_experts(3000 + num_tokens)
    counters = nvfp4_lowm_op.new_counters(torch.device("cuda"))
    contiguous = [
        _rank_weights(ex, range(r * S_LOCAL, (r + 1) * S_LOCAL), x_gsf) for r in range(EP)
    ]
    placed = [_rank_weights(ex, perm[r * S_LOCAL : (r + 1) * S_LOCAL], x_gsf) for r in range(EP)]

    def partials(token_ids, weights, token_scales):
        out = []
        for r in range(EP):
            y = nvfp4_lowm_op.run(x_q, x_sf, token_ids, token_scales, weights[r], r * S_LOCAL, S_EXPERTS,
                                  ex.fc2_input_scale, counters, GATE_CAP, LINEAR_CAP)  # fmt: skip
            out.append(y.float())
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0, "counters not reset"
        return out

    # One top-k slot at a time: every token's expert sits on one rank, the others add zeros, so the ranks' sum is that
    # (token, expert) contribution exactly, whichever rank computed it.
    per_slot = []
    for slot in range(S_TOP_K):
        one_hot = torch.zeros_like(scales)
        one_hot[:, slot] = scales[:, slot]
        ref = torch.stack(partials(ids, contiguous, one_hot)).sum(0)
        new = torch.stack(partials(ids_p, placed, one_hot)).sum(0)
        assert torch.equal(new, ref), f"top-k slot {slot}: a (token, expert) contribution differs"
        assert bool((ref != 0).any()), f"top-k slot {slot}: all zero"
        per_slot.append(ref.double())
    exact = torch.stack(per_slot).sum(0)
    bound = (S_TOP_K + EP) * 2.0**-8 * torch.stack(per_slot).abs().sum(0) + 1e-30
    for name, token_ids, weights in (("contiguous", ids, contiguous), ("placed", ids_p, placed)):
        total = torch.stack(partials(token_ids, weights, scales)).sum(0).double()
        err = (total - exact).abs()
        assert bool((err <= bound).all()), f"{name}: max err/bound {(err / bound).max()}"


# --------------------------------------------------------------------------------------------------------------------
# stats (one GPU)
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("stride", [0, 4])
def test_routing_stats(tmp_path, stride):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        routing_stats,
    )

    layers, experts, top_k, rows = 4, 64, 8, 8
    perm = _random_placement(experts, 30)
    stats = routing_stats.RoutingStats(
        str(tmp_path),
        layers,
        experts,
        first_layer=1,
        expert_of_slot={2: perm},
        stride=stride,
        chunk_steps=1000,
    )
    stats.COPY_INTERVAL = 1
    gen = torch.Generator(device="cuda").manual_seed(31)

    def step_ids():
        return [
            torch.rand(rows, experts, device="cuda", generator=gen)
            .topk(top_k, dim=-1)[1]
            .to(torch.int32)
            for _ in range(1, layers)
        ]

    def record(step):
        for layer, ids in zip(range(1, layers), step):
            stats.record(layer, ids)

    record(step_ids())  # warm-up, before the first executor iteration
    stats.on_iteration(0)
    torch.cuda.synchronize()
    expected = []
    for _ in range(3):  # eager steps
        step = step_ids()
        record(step)
        expected.append(step)
    stats.on_iteration(1)  # collects the warm-up snapshot, starts a copy
    graph_step = [
        torch.zeros(rows, top_k, dtype=torch.int32, device="cuda") for _ in range(1, layers)
    ]
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        record(graph_step)  # warm the capture's allocations (one more step)
    torch.cuda.current_stream().wait_stream(side)
    expected.append([t.clone() for t in graph_step])
    with torch.cuda.graph(graph):
        record(graph_step)
    for _ in range(2):  # replayed steps
        for t, new in zip(graph_step, step_ids()):
            t.copy_(new)
        graph.replay()
        expected.append([t.clone() for t in graph_step])
    torch.cuda.synchronize()
    stats.on_iteration(2)
    torch.cuda.synchronize()
    stats.on_iteration(3)
    stats.close()
    chunks = sorted(tmp_path.glob("routing_*.npz"))
    assert chunks, "no chunk written"
    counts = np.concatenate([np.load(c)["counts"] for c in chunks if "counts" in np.load(c).files])
    assert counts.shape == (len(expected), 2 if stride else 1, layers, experts), counts.shape
    ckpt = np.asarray(perm)
    for got, step in zip(counts, expected):
        assert not got[:, 0].any(), "layer 0 counted"
        for layer, ids in zip(range(1, layers), step):
            ids = ids.cpu().numpy()
            if layer == 2:
                ids = ckpt[ids]  # slot ids back to checkpoint experts
            assert np.array_equal(got[0, layer], np.bincount(ids.reshape(-1), minlength=experts))
            if stride:
                first = np.bincount(ids[::stride].reshape(-1), minlength=experts)
                assert np.array_equal(got[1, layer], first)
    last = np.load(chunks[-1])
    assert np.array_equal(last["totals"], counts.sum(0).astype(np.int64))
    assert int(last["num_steps"]) == len(expected) and int(last["dropped_steps"]) == 0
    assert list(
        np.concatenate([np.load(c)["num_tokens"] for c in chunks if "counts" in np.load(c).files])
    ) == [rows] * len(expected)


# --------------------------------------------------------------------------------------------------------------------
# front (4 GPUs)
# --------------------------------------------------------------------------------------------------------------------
WORLD = 4


def _run_front(_=None):
    try:
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        import tensorrt_llm  # noqa: F401
        from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe_front import (
            K3MoeHeadWorkspace,
            front_weight,
            k3_moe_front,
        )
        from tensorrt_llm.mapping import Mapping

        mapping = Mapping(
            world_size=world, rank=rank, gpus_per_node=torch.cuda.device_count(), tp_size=world
        )
        wl, we, inter = LATENT // world, NUM_EXPERTS // world, 6144 // world
        perm = _random_placement(NUM_EXPERTS, 40)
        index = torch.tensor(perm, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(11)  # the same on every rank
        router = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.16).bfloat16()
        bias = (torch.randn(NUM_EXPERTS, device="cuda", generator=gen) * 0.05).float()
        x32 = torch.randn(32, HIDDEN, device="cuda", generator=gen).bfloat16()
        wgen = torch.Generator(device="cuda").manual_seed(111 + rank)
        down = (torch.randn(wl, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
        gate_up = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
        rows = slice(rank * we, (rank + 1) * we)
        fronts = (
            front_weight(torch.cat([down, router[rows]]), gate_up),
            front_weight(torch.cat([down, router[index][rows]]), gate_up),
        )
        biases = (bias, bias[index].contiguous())
        gsf = torch.tensor([448.0 * 6.0 / 0.5], device="cuda")
        # k3_moe_front's 32-token build is specialized for the TP16 per-rank shapes: at world 4, up to 16 tokens.
        workspaces = {m_max: K3MoeHeadWorkspace.create(mapping, fabric_handle=True, m_max=m_max)
                      for m_max in (8, 16)}  # fmt: skip
        results = []
        with torch.inference_mode():
            for m in (1, 2, 8, 16):
                ws = workspaces[8 if m <= 8 else 16]
                x = x32[:m].contiguous()
                ref, new = (
                    k3_moe_front(x, front, b, RSF, inter, GATE_CAP, LINEAR_CAP, ws, nvfp4_gsf=gsf)
                    for front, b in zip(fronts, biases)
                )
                torch.cuda.synchronize()
                row = dict(
                    M=m,
                    ids_map_back=torch.equal(index[new[0].long()], ref[0].long()),
                    **{name: torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                       for name, a, b in zip(("weights", "latent", "scales", "shared"), new[1:], ref[1:])},
                )  # fmt: skip
                row["rank"], row["good"] = (
                    rank,
                    all(v for k, v in row.items() if k not in ("M", "rank")),
                )
                row["ok"] = all(comm.allgather(bool(row["good"])))
                results.append(row)
        return results
    except Exception:
        traceback.print_exc()
        raise


@pytest.mark.skipif(
    MPI is None or not _sm_ok() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} SM100 / SM103 GPUs with MNNVL and mpi4py",
)
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_front_permuted_router(mpi_pool_executor):
    per_rank = list(mpi_pool_executor.map(_run_front, [None] * WORLD))
    for rows in per_rank:
        for row in rows:
            if not row["good"] or row["rank"] == 0:
                print(
                    "OPCHECK op=k3_moe_front_placed "
                    + " ".join(f"{k}={v}" for k, v in row.items()),
                    flush=True,
                )
    assert all(row["ok"] for rows in per_rank for row in rows)


if __name__ == "__main__":
    # `srun -n 4 --mpi=pmix python3 <this file> front` (or as a gates.sh MATRIX item, which passes --world-size).
    if "front" in sys.argv[1:] or "--world-size" in sys.argv[1:]:
        rows = _run_front()
        if rows[0]["rank"] == 0:
            for row in rows:
                print(
                    "OPCHECK op=k3_moe_front_placed " + " ".join(f"{k}={v}" for k, v in row.items())
                )
            print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
        sys.exit(0 if all(r["ok"] for r in rows) else 1)
    sys.exit(pytest.main(["-q", __file__] + sys.argv[1:]))

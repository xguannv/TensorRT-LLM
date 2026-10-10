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
"""Correctness of the Kimi-K3 routing tail (``trtllm::k3_route_tail``, ``trtllm::k3_route``).

``k3_route_tail`` replaces ``noaux_tc_op`` -> ``moe_sort`` on given FP32 logits:

* top-k ids and weights must be bit-identical to ``noaux_tc_op`` on the same logits;
* the ``moe_sort`` outputs must be bit-identical to a token-ordered sort of the ids, and match
  ``moe_sort``'s tile metadata exactly (``moe_sort``'s cluster kernel orders the rows of a tile
  by atomics above 12 tokens, so there only the set of rows per tile is compared).

``k3_route`` computes the logits itself with a split-K GEMV; they must match the FP32 product
to a tight tolerance and the routing must be exact on them.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)

NUM_EXPERTS = 896
HIDDEN = 7168
TOP_K = 16
TILE = 128
MS = [1, 4, 5, 7, 13, 16, 17, 28, 32, 33, 56, 64, 65, 112, 128, 224, 256]
# (local_expert_offset, local_num_experts): EP16 ranks, TP4xEP4 ranks, TP8xEP2, no EP.
LAYOUTS = [(0, 56), (7 * 56, 56), (15 * 56, 56), (0, 224), (3 * 224, 224), (448, 448), (0, 896)]


def _module():
    import tensorrt_llm  # noqa: F401  (registers the reference ops)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_route_tail

    return k3_route_tail


def _logits(m, seed, mode="rand"):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "rand":
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen) * 2
        bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    elif mode == "skew":
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen)
        logits[:, 224:232] += 6
        bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    else:  # ties: all-equal rows and rows with few distinct values
        logits = torch.zeros((m, NUM_EXPERTS), device="cuda")
        bias = torch.zeros((NUM_EXPERTS,), device="cuda")
        half = m // 2
        logits[:half] = torch.randint(-2, 3, (half, NUM_EXPERTS), device="cuda", generator=gen)
    return logits, bias


def _noaux(logits, bias, scale):
    vals, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, scale)
    return ids.to(torch.int32), vals.to(torch.float32)


def _assert_bit_equal(actual, expected, what):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, what
    a = actual.view(torch.int32) if actual.dtype == torch.float32 else actual
    e = expected.view(torch.int32) if expected.dtype == torch.float32 else expected
    mismatch = (a != e).nonzero()
    assert mismatch.numel() == 0, f"{what}: {mismatch.shape[0]} mismatches, first {mismatch[:4]}"


def _token_ordered_sort(ids, offset, num_local):
    """Python ``moe_sort`` with the rows of each expert in ascending token order."""
    m = ids.shape[0]
    rows = {}
    for t, row in enumerate(ids.cpu().tolist()):
        for k, e in enumerate(row):
            j = e - offset
            if 0 <= j < num_local:
                rows.setdefault(j, []).append(t * TOP_K + k)
    tile_expert, tile_limit, e2p, p2e = [], [], torch.full((m, TOP_K), -1, dtype=torch.int32), {}
    for j in sorted(rows):
        base = len(tile_expert) * TILE
        count = len(rows[j])
        for c in range((count + TILE - 1) // TILE):
            tile_expert.append(j)
            tile_limit.append(min(base + (c + 1) * TILE, base + count))
        for r, expanded in enumerate(rows[j]):
            e2p.view(-1)[expanded] = base + r
            p2e[base + r] = expanded
    return tile_expert, tile_limit, e2p, p2e


def _check_outputs(outs, logits, bias, scale, offset, num_local):
    ids, vals, tile_expert, tile_limit, e2p, p2e, total, num_tiles = outs[:8]
    ref_ids, ref_vals = _noaux(logits, bias, scale)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")

    m = ids.shape[0]
    ref = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=vals,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=num_local,
        tile_tokens_dim=TILE,
    )
    for got, exp in zip((tile_expert, tile_limit, e2p, p2e, total, num_tiles), ref):
        assert got.dtype == exp.dtype == torch.int32 and got.shape == exp.shape
    n = int(num_tiles.item())
    _assert_bit_equal(num_tiles, ref[5], "num_non_exiting_tiles")
    _assert_bit_equal(total, ref[4], "total_num_padded_tokens")
    _assert_bit_equal(tile_expert[:n], ref[0][:n], "tile_idx_to_expert_idx")
    _assert_bit_equal(tile_limit[:n], ref[1][:n], "tile_idx_to_mn_limit")

    c_expert, c_limit, c_e2p, c_p2e = _token_ordered_sort(ids, offset, num_local)
    assert tile_expert[:n].tolist() == c_expert and tile_limit[:n].tolist() == c_limit
    _assert_bit_equal(e2p.cpu(), c_e2p, "expanded_idx_to_permuted_idx (token order)")
    p2e_cpu = p2e.cpu()
    for perm, expanded in c_p2e.items():
        assert int(p2e_cpu[perm]) == expanded, f"permuted_idx_to_expanded_idx[{perm}]"

    ref_e2p, ref_p2e = ref[2].cpu(), ref[3].cpu()
    if m <= 12:
        _assert_bit_equal(e2p.cpu(), ref_e2p, "expanded_idx_to_permuted_idx vs moe_sort")
    else:
        _assert_bit_equal(e2p.cpu() < 0, ref_e2p < 0, "non-local entries vs moe_sort")
    # moe_sort's cluster kernel also splits a multi-tile expert's rows over its tiles in atomic
    # order, so the rows are compared per expert.
    rows, ref_rows = {}, {}
    for t in range(n):
        lo, hi = t * TILE, int(tile_limit[t])
        expert = int(tile_expert[t])
        rows.setdefault(expert, []).extend(p2e_cpu[lo:hi].tolist())
        ref_rows.setdefault(expert, []).extend(ref_p2e[lo:hi].tolist())
    for expert in rows:
        assert sorted(rows[expert]) == sorted(ref_rows[expert]), f"local expert {expert}"


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("mode", ["rand", "skew", "ties"])
def test_tail_matches_noaux_and_sort(m, mode):
    _module()
    logits, bias = _logits(m, seed=m, mode=mode)
    for offset, num_local in LAYOUTS:
        outs = torch.ops.trtllm.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE)
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [7, 32, 224])
@pytest.mark.parametrize("scale", [2.5, 0.3])
def test_tail_routed_scaling_factor(m, scale):
    _module()
    logits, bias = _logits(m, seed=100 + m)
    outs = torch.ops.trtllm.k3_route_tail(logits, bias, scale, 224, 224, TILE)
    _check_outputs(outs, logits, bias, scale, 224, 224)


@pytest.mark.parametrize("m", [1, 7, 16, 32, 64, 128])
@pytest.mark.parametrize(
    "num_ctas,warps_per_token,num_warps",
    [(1, 1, 16), (2, 2, 16), (4, 4, 16), (8, 1, 16), (8, 1, 8), (8, 2, 8), (4, 4, 8)],
)
def test_tail_configs(m, num_ctas, warps_per_token, num_warps):
    rt = _module()
    if (m + num_ctas - 1) // num_ctas > 32:
        pytest.skip("more than 32 tokens per CTA")
    logits, bias = _logits(m, seed=7 * m + num_ctas)
    outs = rt._route_tail_outputs(logits, m, bias, 56, TILE)
    rt.run_k3_route_tail(
        logits,
        m,
        bias,
        1.0,
        outs[0],
        outs[1],
        outs[2:],
        7 * 56,
        56,
        num_ctas=num_ctas,
        warps_per_token=warps_per_token,
        num_warps=num_warps,
    )
    _check_outputs(outs, logits, bias, 1.0, 7 * 56, 56)


@pytest.mark.parametrize("m", [7, 32, 100])
def test_tail_split_partials(m):
    """``[S, R, 896]`` partials (R > M) are summed in split order."""
    _module()
    gen = torch.Generator(device="cuda").manual_seed(m)
    partials = torch.randn((5, m + 3, NUM_EXPERTS), device="cuda", generator=gen)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    outs = torch.ops.trtllm.k3_route_tail(partials, bias, 1.0, 0, 224, TILE, m)
    logits = partials[0, :m].clone()
    for s in range(1, 5):
        logits = logits + partials[s, :m]
    _check_outputs(outs, logits, bias, 1.0, 0, 224)


def test_tail_cuda_graph_replay():
    _module()
    m = 32
    logits, bias = _logits(m, seed=3)
    static = logits.clone()
    torch.ops.trtllm.k3_route_tail(static, bias, 1.0, 224, 224, TILE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = torch.ops.trtllm.k3_route_tail(static, bias, 1.0, 224, 224, TILE)
    for seed in range(4):
        fresh, _ = _logits(m, seed=10 + seed, mode="skew" if seed % 2 else "rand")
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        _check_outputs(outs, static, bias, 1.0, 224, 224)


@pytest.mark.parametrize("m", [1, 7, 13, 16, 28, 32, 33, 56, 64])
@pytest.mark.parametrize("offset,num_local", [(7 * 56, 56), (224, 224), (0, 896)])
def test_route_gemv(m, offset, num_local):
    _module()
    gen = torch.Generator(device="cuda").manual_seed(m + offset)
    h = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn((NUM_EXPERTS, HIDDEN), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    outs = torch.ops.trtllm.k3_route(h, w, bias, 1.0, offset, num_local, TILE)
    logits = outs[8]
    exact = (h.double() @ w.double().t()).float()
    torch.testing.assert_close(logits, exact, rtol=0, atol=5e-5)
    ref_gemm = torch.ops.trtllm.dsv3_router_gemm_op(h, w.t(), bias=None, out_dtype=torch.float32)
    torch.testing.assert_close(logits, ref_gemm, rtol=0, atol=1e-4)
    _check_outputs(outs, logits, bias, 1.0, offset, num_local)
    # The selection also agrees with the production GEMM's logits except at near-ties.
    chain_ids, _ = _noaux(ref_gemm, bias, 1.0)
    ids = outs[0]
    same = sum(len(set(a) & set(b)) for a, b in zip(ids.cpu().tolist(), chain_ids.cpu().tolist()))
    assert same >= 0.99 * m * TOP_K


def test_routing_method_deferred():
    """``_KimiK3RoutingMethod`` runs a deferred router once on the logits it is given."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod

    bias = torch.zeros(NUM_EXPERTS, device="cuda")
    method = _KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    logits, _ = _logits(4, seed=1)
    seen = []

    def deferred(x):
        seen.append(x)
        return "ids", "weights"

    method.deferred = deferred
    assert method.apply(logits) == ("ids", "weights")
    assert seen == [logits] and method.deferred is None
    ids, _ = method.apply(logits)
    _assert_bit_equal(ids, _noaux(logits, bias, 1.0)[0], "regular routing after the deferred one")


def test_moe_runtime_routes_with_tail(monkeypatch):
    """``TRTLLM_K3_ROUTE_TAIL=1``: ``KimiK3MoERuntime`` hands the routed experts the tail's
    top-k (with the ``moe_sort`` outputs attached) when they ask for the routing."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    _module()
    m, offset, num_local = 28, 224, 224
    logits, bias = _logits(m, seed=5)
    routing = _KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    seen = {}

    class _Routed:
        routing_method = routing
        use_dp = False
        backend = SimpleNamespace(
            slot_start=offset, expert_size_per_partition=num_local, num_slots=NUM_EXPERTS
        )

        def __call__(self, value, router_logits, **kwargs):
            seen["routing"] = routing.apply(router_logits)
            return value

    class _Identity(nn.Module):
        def forward(self, value, **kwargs):
            return value

    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(
        compute_logits=lambda value: logits,
        e_score_correction_bias=bias,
        routed_scaling_factor=1.0,
        num_experts=NUM_EXPERTS,
    )
    runtime.routed_experts = _Routed()
    runtime.routed_expert_down_proj = _Identity()
    runtime.routed_expert_norm = _Identity()
    runtime.routed_expert_up_proj = _Identity()
    runtime.shared_experts = _Identity()
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime.shared_expert_stream = None
    runtime.latent_down_stream = None
    runtime.moe_main_event = runtime.moe_shared_event = runtime.moe_routed_event = None
    runtime.moe_latent_fork_event = runtime.moe_latent_ready_event = None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "1")
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value)

    hidden = torch.zeros((m, 8), device="cuda")
    with with_multi_stream(False):
        runtime(hidden)
    ids, weights = seen["routing"]
    ref = torch.ops.trtllm.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE)
    _assert_bit_equal(ids, ref[0], "topk_ids")
    _assert_bit_equal(weights, ref[1], "topk_weights")
    key, sort_outputs = ids._k3_presorted
    assert key == (offset, num_local, TILE) and len(sort_outputs) == 6
    _check_outputs((ids, weights, *sort_outputs), logits, bias, 1.0, offset, num_local)
    assert routing.deferred is None

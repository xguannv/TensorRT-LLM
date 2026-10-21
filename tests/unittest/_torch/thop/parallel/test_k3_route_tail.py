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

``k3_route_tail`` replaces ``noaux_tc_op`` -> ``moe_sort`` on given FP32 logits (one cluster up to
256 tokens, a grid of CTAs with a last-CTA sort above):

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
    if mode == "realistic":
        # Router-like logits: hidden states (unit norm) against unit expert directions, plus
        # per-expert popularity (some hot experts); the tokens of one request (an MTP verify
        # window of 7) share most of their hidden state; bias spread ~0.19 as in Kimi-K3.
        dim = 512
        directions = torch.randn((NUM_EXPERTS, dim), device="cuda", generator=gen)
        directions = directions / directions.norm(dim=1, keepdim=True)
        popularity = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.7
        requests = torch.randn(((m + 6) // 7, dim), device="cuda", generator=gen)
        hidden = requests.repeat_interleave(7, dim=0)[:m] * 0.85
        hidden = hidden + torch.randn((m, dim), device="cuda", generator=gen) * 0.5
        hidden = hidden / hidden.norm(dim=1, keepdim=True)
        logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
        bias = (torch.rand((NUM_EXPERTS,), device="cuda", generator=gen) - 0.5) * 0.19
        return logits.contiguous(), bias
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


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("mode", ["rand", "skew", "ties"])
def test_tail_fast_matches_noaux_and_sort(m, mode):
    """``fast`` mode (rolled tie fallback, top-k stored after the cluster barrier): same outputs."""
    rt = _module()
    logits, bias = _logits(m, seed=m, mode=mode)
    for offset, num_local in LAYOUTS:
        outs = rt._route_tail_outputs(logits, m, bias, num_local, TILE)
        rt.run_k3_route_tail(
            logits, m, bias, 1.0, outs[0], outs[1], outs[2:], offset, num_local, fast=True
        )
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [1, 7, 14, 16, 32, 64, 128])
@pytest.mark.parametrize(
    "num_ctas,warps_per_token,num_warps",
    [(1, 1, 16), (2, 2, 16), (8, 1, 8), (8, 2, 8), (8, 4, 8), (4, 2, 4), (8, 1, 1)],
)
@pytest.mark.parametrize("mode", ["realistic", "ties"])
def test_tail_fast_configs(m, num_ctas, warps_per_token, num_warps, mode):
    rt = _module()
    if (m + num_ctas - 1) // num_ctas > 32:
        pytest.skip("more than 32 tokens per CTA")
    logits, bias = _logits(m, seed=11 * m + num_ctas, mode=mode)
    outs = rt._route_tail_outputs(logits, m, bias, 224, TILE)
    rt.run_k3_route_tail(
        logits,
        m,
        bias,
        1.0,
        outs[0],
        outs[1],
        outs[2:],
        224,
        224,
        num_ctas=num_ctas,
        warps_per_token=warps_per_token,
        num_warps=num_warps,
        fast=True,
    )
    _check_outputs(outs, logits, bias, 1.0, 224, 224)


@pytest.mark.parametrize("fast", [False, True])
def test_tail_prefetch_next_bias(fast):
    """The next layer's bias prefetch is a hint: same outputs, also replayed in a CUDA graph;
    tensors that are not a 16-byte aligned fp32 [896] are not prefetched."""
    rt = _module()
    m = 14
    logits, bias = _logits(m, seed=21, mode="realistic")
    next_bias = torch.randn((NUM_EXPERTS,), device="cuda") * 0.01
    outs = torch.ops.trtllm.k3_route_tail(logits, bias, 1.0, 224, 224, TILE, -1, next_bias)
    _check_outputs(outs, logits, bias, 1.0, 224, 224)
    assert rt._prefetch_ptr(next_bias, NUM_EXPERTS * 4) == next_bias.data_ptr()
    assert rt._prefetch_ptr(next_bias[1:], NUM_EXPERTS * 4) == 0
    assert rt._prefetch_ptr(None, NUM_EXPERTS * 4) == 0
    static = logits.clone()
    out_static = rt._route_tail_outputs(static, m, bias, 224, TILE)

    def launch():
        rt.run_k3_route_tail(
            static,
            m,
            bias,
            1.0,
            out_static[0],
            out_static[1],
            out_static[2:],
            224,
            224,
            fast=fast,
            prefetch_bias=next_bias,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for seed in range(3):
        fresh, _ = _logits(m, seed=60 + seed, mode="realistic")
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        _check_outputs(out_static, static, bias, 1.0, 224, 224)


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


# ---------------------------------------------------------------------------
# Multi-CTA path (M > 256)
# ---------------------------------------------------------------------------

WIDE_MS = [257, 384, 512, 896, 1344]
# EP16 ranks (56 local experts), EP8 (112) and TP4xEP4 (224).
WIDE_LAYOUTS = [(0, 56), (7 * 56, 56), (15 * 56, 56), (448, 112), (3 * 224, 224)]


@pytest.mark.parametrize("m", WIDE_MS)
@pytest.mark.parametrize("mode", ["realistic", "rand", "skew", "ties"])
def test_wide_matches_noaux_and_sort(m, mode):
    rt = _module()
    logits, bias = _logits(m, seed=m, mode=mode)
    for offset, num_local in WIDE_LAYOUTS:
        if not rt.wide_sort_fits(m, num_local):
            assert not rt.wide_table_fits(m, num_local, 8)
            assert not rt.wide_table_fits(m, num_local, 16)
            with pytest.raises(ValueError):
                torch.ops.trtllm.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE)
            continue
        outs = torch.ops.trtllm.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE)
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [257, 896, 2048])
def test_wide_routed_scaling_factor_and_max_m(m):
    _module()
    logits, bias = _logits(m, seed=200 + m, mode="realistic")
    outs = torch.ops.trtllm.k3_route_tail(logits, bias, 2.5, 7 * 56, 56, TILE)
    _check_outputs(outs, logits, bias, 2.5, 7 * 56, 56)


@pytest.mark.parametrize("m", [300, 896, 1344])
@pytest.mark.parametrize(
    "tokens_per_cta,num_warps,warps_per_token",
    [(8, 8, 1), (8, 16, 1), (16, 8, 1), (16, 16, 1), (16, 8, 2), (16, 16, 4)],
)
def test_wide_configs(m, tokens_per_cta, num_warps, warps_per_token):
    rt = _module()
    logits, bias = _logits(m, seed=11 * m + tokens_per_cta, mode="realistic")
    for offset, num_local in [(7 * 56, 56), (224, 224)]:
        outs = rt._route_tail_outputs(logits, m, bias, num_local, TILE)
        if not rt.wide_table_fits(m, num_local, tokens_per_cta):
            with pytest.raises(ValueError):
                rt.run_k3_route_wide(
                    logits,
                    m,
                    bias,
                    1.0,
                    outs[0],
                    outs[1],
                    outs[2:],
                    offset,
                    num_local,
                    tokens_per_cta=tokens_per_cta,
                )
            continue
        # Poison the outputs: every entry the contract covers must be written.
        for t in outs:
            t.view(torch.int32).fill_(-7)
        rt.run_k3_route_wide(
            logits,
            m,
            bias,
            1.0,
            outs[0],
            outs[1],
            outs[2:],
            offset,
            num_local,
            tokens_per_cta=tokens_per_cta,
            num_warps=num_warps,
            warps_per_token=warps_per_token,
        )
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [257, 1344, 2048])
@pytest.mark.parametrize("tokens_per_cta,num_warps", [(8, 4), (16, 8), (32, 8)])
def test_wide_topk_only(m, tokens_per_cta, num_warps):
    rt = _module()
    logits, bias = _logits(m, seed=m + 1, mode="realistic")
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.float32, device="cuda")
    rt.run_k3_route_wide(
        logits,
        m,
        bias,
        1.0,
        ids,
        vals,
        None,
        0,
        NUM_EXPERTS,
        tokens_per_cta=tokens_per_cta,
        num_warps=num_warps,
    )
    ref_ids, ref_vals = _noaux(logits, bias, 1.0)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")


def test_wide_split_partials():
    """``[S, R, 896]`` partials (R > M) are summed in split order above 256 tokens too."""
    _module()
    m = 300
    gen = torch.Generator(device="cuda").manual_seed(m)
    partials = torch.randn((2, m + 5, NUM_EXPERTS), device="cuda", generator=gen)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    outs = torch.ops.trtllm.k3_route_tail(partials, bias, 1.0, 7 * 56, 56, TILE, m)
    _check_outputs(outs, partials[0, :m] + partials[1, :m], bias, 1.0, 7 * 56, 56)


def test_wide_default_config():
    """8-token CTAs (compact sort tables) up to 1024 tokens where the tables fit, else 16."""
    rt = _module()
    assert rt.default_wide_config(384, 56) == (8, 8, 1)
    assert rt.default_wide_config(896, 56) == (8, 8, 1)
    assert rt.default_wide_config(1344, 56) == (16, 16, 1)
    assert rt.wide_table_fits(1344, 56, 8)
    assert rt.default_wide_config(2048, 56) == (16, 16, 1)
    assert rt.wide_sort_fits(2048, 56)
    assert rt.default_wide_config(512, 224) == (16, 16, 1)


def test_wide_rejects_large_local_range():
    rt = _module()
    logits, bias = _logits(300, seed=1)
    assert not rt.can_use_k3_route_tail(300, logits, 448)
    assert rt.can_use_k3_route_tail(300, logits, 224)
    assert not rt.can_use_k3_route_tail(1344, logits, 224)
    assert rt.can_use_k3_route_tail(rt.WIDE_MAX_M, logits, 56)
    assert not rt.can_use_k3_route_tail(rt.WIDE_MAX_M + 1, logits, 56)
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_route_tail(logits, bias, 1.0, 0, 448, TILE)


def test_wide_cuda_graph_replay():
    """Replays reuse the arrival counter: several launches (different grids) per graph."""
    _module()
    shapes = [(896, 7 * 56, 56), (512, 224, 224), (1344, 15 * 56, 56)]
    statics, outs = [], []
    for m, offset, num_local in shapes:
        logits, bias = _logits(m, seed=3 + m, mode="realistic")
        statics.append((logits.clone(), bias))
        torch.ops.trtllm.k3_route_tail(statics[-1][0], bias, 1.0, offset, num_local, TILE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (static, bias), (m, offset, num_local) in zip(statics, shapes):
            outs.append(torch.ops.trtllm.k3_route_tail(static, bias, 1.0, offset, num_local, TILE))
    for seed in range(4):
        for (static, bias), (m, _, _) in zip(statics, shapes):
            mode = ("realistic", "rand", "skew", "ties")[seed]
            static.copy_(_logits(m, seed=20 + seed, mode=mode)[0])
        graph.replay()
        torch.cuda.synchronize()
        for (static, bias), out, (m, offset, num_local) in zip(statics, outs, shapes):
            _check_outputs(out, static, bias, 1.0, offset, num_local)


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


@pytest.mark.parametrize("m,offset,num_local", [(28, 224, 224), (896, 7 * 56, 56)])
def test_moe_runtime_routes_with_tail(monkeypatch, m, offset, num_local):
    """``TRTLLM_K3_ROUTE_TAIL=1``: ``KimiK3MoERuntime`` hands the routed experts the tail's
    top-k (with the ``moe_sort`` outputs attached) when they ask for the routing."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    _module()
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
        # The runtime reads the shared expert's down_proj (MoE residual / shared-tail paths).
        down_proj = None

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
    runtime._merge_moe_allreduce = False
    runtime.shared_down_all_reduce = None
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


def test_route_tail_target_limits(monkeypatch):
    """The routing tail serves up to ``TRTLLM_K3_ROUTE_TAIL_MAX_M`` tokens, above 256 only with
    at most 224 local experts."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod

    rt = _module()
    bias = torch.zeros(NUM_EXPERTS, device="cuda")
    routing = _KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(e_score_correction_bias=bias, num_experts=NUM_EXPERTS)

    def target(num_tokens, num_local):
        runtime.routed_experts = SimpleNamespace(
            routing_method=routing,
            use_dp=False,
            backend=SimpleNamespace(
                slot_start=0, expert_size_per_partition=num_local, num_slots=NUM_EXPERTS
            ),
        )
        return runtime._route_tail_target(num_tokens)

    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "1")
    assert target(256, 896) == (0, 896)
    assert target(257, 896) is None
    assert target(1344, 56) == (0, 56)
    assert rt.wide_workspace_ready(bias.device)
    assert target(1344, 224) is None  # sort tables do not fit
    assert target(rt.WIDE_MAX_M + 1, 56) is None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL_MAX_M", 256)
    assert target(256, 56) == (0, 56)
    assert target(257, 56) is None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "0")
    assert target(28, 56) is None


# ---------------------------------------------------------------------------
# UMMA router GEMM (k3_router_umma, TRTLLM_K3_ROUTER_GEMV=umma)
# ---------------------------------------------------------------------------

UMMA_MS = [1, 7, 13, 16, 17, 28, 32, 33, 56, 64, 65, 112, 128, 224, 256]


def _umma_module():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_router_umma

    return k3_router_umma


def _router_inputs(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn((NUM_EXPERTS, HIDDEN), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    return h, w, bias


@pytest.mark.parametrize("m", UMMA_MS)
@pytest.mark.parametrize("splits", [5, 8])
def test_umma_partials(m, splits):
    """Every partial is the FP32 product over its K slice; the slices tile K in order, the
    result is deterministic and no slot is left unwritten."""
    ru = _umma_module()
    h, w, _ = _router_inputs(m, seed=m + 100 * splits)
    parts = torch.full((splits, m, NUM_EXPERTS), float("nan"), device="cuda")
    ru.run_k3_router_umma(h, w, parts)
    torch.cuda.synchronize()
    assert not torch.isnan(parts).any()
    chunks = HIDDEN // 64
    start = 0
    for s in range(splits):
        count = chunks // splits + (1 if s < chunks % splits else 0)
        lo, hi = start * 64, (start + count) * 64
        exact = (h[:, lo:hi].double() @ w[:, lo:hi].double().t()).float()
        torch.testing.assert_close(parts[s], exact, rtol=0, atol=2e-5)
        start += count
    assert start == chunks
    exact = (h.double() @ w.double().t()).float()
    torch.testing.assert_close(parts.sum(0), exact, rtol=0, atol=5e-5)
    again = torch.empty_like(parts)
    ru.run_k3_router_umma(h, w, again)
    torch.cuda.synchronize()
    _assert_bit_equal(again, parts, "second launch")


@pytest.mark.parametrize("m", [1, 7, 16, 32, 33, 56, 64, 112, 128, 224, 256])
@pytest.mark.parametrize("offset,num_local", [(7 * 56, 56), (224, 224), (0, 896)])
def test_route_umma(monkeypatch, m, offset, num_local):
    """``k3_route`` on the UMMA GEMM: the logits match the FP32 product, the routing is exact on
    them and agrees with the production GEMM's selection except at near-ties."""
    rt = _module()
    _umma_module()
    monkeypatch.setattr(rt, "ROUTER_GEMV", "umma")
    h, w, bias = _router_inputs(m, seed=m + offset)
    assert rt.can_use_k3_route(h, w)
    outs = torch.ops.trtllm.k3_route(h, w, bias, 1.0, offset, num_local, TILE)
    logits = outs[8]
    exact = (h.double() @ w.double().t()).float()
    torch.testing.assert_close(logits, exact, rtol=0, atol=5e-5)
    ref_gemm = torch.ops.trtllm.dsv3_router_gemm_op(h, w.t(), bias=None, out_dtype=torch.float32)
    torch.testing.assert_close(logits, ref_gemm, rtol=0, atol=1e-4)
    _check_outputs(outs, logits, bias, 1.0, offset, num_local)
    chain_ids, _ = _noaux(ref_gemm, bias, 1.0)
    same = sum(
        len(set(a) & set(b)) for a, b in zip(outs[0].cpu().tolist(), chain_ids.cpu().tolist())
    )
    assert same >= 0.99 * m * TOP_K


def test_route_umma_limits(monkeypatch):
    rt = _module()
    _umma_module()
    h, w, _ = _router_inputs(257, seed=1)
    monkeypatch.setattr(rt, "ROUTER_GEMV", "mma")
    assert rt.can_use_k3_route(h[:64], w) and not rt.can_use_k3_route(h[:65], w)
    monkeypatch.setattr(rt, "ROUTER_GEMV", "umma")
    assert rt.can_use_k3_route(h[:256], w) and not rt.can_use_k3_route(h, w)


def test_route_umma_cuda_graph_replay(monkeypatch):
    rt = _module()
    _umma_module()
    monkeypatch.setattr(rt, "ROUTER_GEMV", "umma")
    m, offset, num_local = 32, 224, 224
    h, w, bias = _router_inputs(m, seed=7)
    static = h.clone()
    torch.ops.trtllm.k3_route(static, w, bias, 1.0, offset, num_local, TILE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = torch.ops.trtllm.k3_route(static, w, bias, 1.0, offset, num_local, TILE)
    for seed in range(3):
        fresh, _, _ = _router_inputs(m, seed=20 + seed)
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        eager = torch.ops.trtllm.k3_route(static, w, bias, 1.0, offset, num_local, TILE)
        for got, exp, what in zip(outs, eager, ("ids", "weights")):
            _assert_bit_equal(got, exp, what)
        _assert_bit_equal(outs[8], eager[8], "logits")
        _check_outputs(outs, outs[8], bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m,offset,num_local", [(7, 224, 224), (56, 0, 896), (128, 0, 896)])
def test_moe_runtime_routes_with_umma_gemm(monkeypatch, m, offset, num_local):
    """``TRTLLM_K3_ROUTE_TAIL=1`` + ``TRTLLM_K3_ROUTER_GEMM=umma``: the runtime computes the
    UMMA GEMM's split-K partials, hands the routed experts their first split as the logits and
    routes with the tail on all splits when the experts ask for the routing."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    _module()
    ru = _umma_module()
    h, w, bias = _router_inputs(m, seed=11 + m)
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
            seen["logits"] = router_logits
            seen["routing"] = routing.apply(router_logits)
            return value

    class _Identity(nn.Module):
        # The runtime reads the shared expert's down_proj (MoE residual / shared-tail paths).
        down_proj = None

        def forward(self, value, **kwargs):
            return value

    def _no_cublas(value):
        raise AssertionError("the cuBLAS router GEMM must not run")

    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(
        compute_logits=_no_cublas,
        weight=w,
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
    runtime._merge_moe_allreduce = False
    runtime.shared_down_all_reduce = None
    runtime.shared_expert_stream = None
    runtime.latent_down_stream = None
    runtime.moe_main_event = runtime.moe_shared_event = runtime.moe_routed_event = None
    runtime.moe_latent_fork_event = runtime.moe_latent_ready_event = None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "1")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTER_GEMM", "umma")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTER_GEMM_MAX_M", 256)
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value)

    with with_multi_stream(False):
        runtime(h)
    parts = torch.empty((ru.DEFAULT_SPLITS, m, NUM_EXPERTS), device="cuda")
    ru.run_k3_router_umma(h, w, parts)
    torch.cuda.synchronize()
    _assert_bit_equal(seen["logits"], parts[0], "placeholder logits = first split")
    ids, weights = seen["routing"]
    ref = torch.ops.trtllm.k3_route_tail(parts, bias, 1.0, offset, num_local, TILE)
    _assert_bit_equal(ids, ref[0], "topk_ids")
    _assert_bit_equal(weights, ref[1], "topk_weights")
    key, sort_outputs = ids._k3_presorted
    assert key == (offset, num_local, TILE)
    logits = parts[0].clone()
    for s in range(1, parts.shape[0]):
        logits = logits + parts[s]
    _check_outputs((ids, weights, *sort_outputs), logits, bias, 1.0, offset, num_local)
    assert routing.deferred is None
    # Above TRTLLM_K3_ROUTER_GEMM_MAX_M the cuBLAS GEMM serves the forward again.
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTER_GEMM_MAX_M", m - 1)
    assert not runtime._can_umma_route(h)


# ---------------------------------------------------------------------------
# Redundant expert copies (trtllm::k3_route_tail_rep, TRTLLM_K3_EXPERT_REPLICAS)
# ---------------------------------------------------------------------------

REP_EP = 4
REP_HOME = NUM_EXPERTS // REP_EP


def _copy_entries(slots, seed, hot=()):
    """``slots`` copies per EP rank of experts homed on other ranks, in a seeded priority order
    (``hot`` first)."""
    gen = torch.Generator().manual_seed(seed)
    rest = torch.randperm(NUM_EXPERTS, generator=gen).tolist()
    order = list(hot) + [x for x in rest if x not in hot]
    entries, per_rank = [], [0] * REP_EP
    for x in order:
        options = [r for r in range(REP_EP) if r != x // REP_HOME and per_rank[r] < slots]
        if options:
            r = options[int(torch.randint(len(options), (1,), generator=gen))]
            entries.append((x, r))
            per_rank[r] += 1
        if len(entries) == REP_EP * slots:
            break
    return entries


def _copy_slots(ids, entries, slots):
    """Routed-MoE slot (``[M, 16]`` int32) of every selection under the reference copy choice."""
    rt = _module()
    num_local = REP_HOME + slots
    choice = rt.replica_choice(ids.flatten().tolist(), entries)
    copy_of = {}
    for (x, r), on_copy in zip(entries, choice):
        if on_copy:  # copy slots of a rank in ascending expert order
            mine = sorted(e for e, q in entries if q == r)
            copy_of[x] = r * num_local + REP_HOME + mine.index(x)
    return torch.tensor(
        [
            [copy_of.get(x, (x // REP_HOME) * num_local + x % REP_HOME) for x in row]
            for row in ids.cpu().tolist()
        ],
        dtype=torch.int32,
    )


def _check_sort(sort_outs, sel, vals, offset, num_local, num_slots):
    """The sort outputs of one rank against ``moe_sort`` / the token-ordered sort of ``sel``."""
    tile_expert, tile_limit, e2p, p2e, total, num_tiles = sort_outs
    ref = torch.ops.trtllm.moe_sort(
        token_selected_experts=sel.cuda(),
        token_final_scales=vals,
        num_experts=num_slots,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=num_local,
        tile_tokens_dim=TILE,
    )
    n = int(num_tiles.item())
    _assert_bit_equal(num_tiles, ref[5], "num_non_exiting_tiles")
    _assert_bit_equal(total, ref[4], "total_num_padded_tokens")
    _assert_bit_equal(tile_expert[:n], ref[0][:n], "tile_idx_to_expert_idx")
    _assert_bit_equal(tile_limit[:n], ref[1][:n], "tile_idx_to_mn_limit")
    c_expert, c_limit, c_e2p, c_p2e = _token_ordered_sort(sel, offset, num_local)
    assert tile_expert[:n].tolist() == c_expert and tile_limit[:n].tolist() == c_limit
    _assert_bit_equal(e2p.cpu(), c_e2p, "expanded_idx_to_permuted_idx (token order)")
    p2e_cpu = p2e.cpu()
    for perm, expanded in c_p2e.items():
        assert int(p2e_cpu[perm]) == expanded, f"permuted_idx_to_expanded_idx[{perm}]"


def _run_replicas(
    logits, bias, entries, slots, scale=1.0, num_tokens=-1, fast=None, shape=None, prefetch=False
):
    """``k3_route_tail_rep`` of every EP rank; checks top-k, the sort and that every selection is
    computed on exactly one rank. Returns the reference slots. ``fast`` / ``shape`` (``(ctas,
    warps per token, warps)``) / ``prefetch`` (a next-layer bias and table to prefetch) launch
    through ``run_k3_route_tail_rep``."""
    rt = _module()
    m = logits.shape[0] if num_tokens < 0 else num_tokens
    num_local = REP_HOME + slots
    owner = torch.zeros((m, TOP_K), dtype=torch.int32)
    sel = None
    for rank in range(REP_EP):
        table = torch.tensor(rt.replica_table(entries, rank), dtype=torch.int32, device="cuda")
        if fast is None and shape is None and not prefetch:
            outs = torch.ops.trtllm.k3_route_tail_rep(
                logits,
                bias,
                scale,
                rank * num_local,
                num_local,
                rank * REP_HOME,
                table,
                len(entries),
                TILE,
                num_tokens,
            )
        else:
            outs = rt._route_tail_outputs(logits, m, bias, num_local, TILE)
            kwargs = {}
            if shape is not None:
                kwargs = dict(num_ctas=shape[0], warps_per_token=shape[1], num_warps=shape[2])
            if prefetch:
                kwargs["prefetch_bias"] = torch.randn((NUM_EXPERTS,), device="cuda")
                kwargs["prefetch_table"] = torch.tensor(
                    rt.replica_table(entries[::-1], (rank + 1) % REP_EP),
                    dtype=torch.int32,
                    device="cuda",
                )
            rt.run_k3_route_tail_rep(
                logits,
                m,
                bias,
                scale,
                outs[0],
                outs[1],
                outs[2:],
                num_local,
                rank * REP_HOME,
                table,
                len(entries),
                fast=fast,
                **kwargs,
            )
        if sel is None:
            if logits.dim() == 2:
                flat = logits[:m]
            else:
                flat = logits[0, :m].clone()
                for s in range(1, logits.shape[0]):
                    flat = flat + logits[s, :m]
            ref_ids, ref_vals = _noaux(flat, bias, scale)
            sel = _copy_slots(ref_ids, entries, slots)
        _assert_bit_equal(outs[0], ref_ids, "topk_ids")
        _assert_bit_equal(outs[1], ref_vals, "topk_weights")
        _check_sort(outs[2:], sel, ref_vals, rank * num_local, num_local, REP_EP * num_local)
        owner += (outs[4].cpu() >= 0).to(torch.int32)
    assert (owner == 1).all(), "every selection is computed on exactly one rank"
    return sel


@pytest.mark.parametrize("m", [1, 7, 14, 16, 28, 32, 64, 112, 256])
@pytest.mark.parametrize("slots", [4, 8, 16])
@pytest.mark.parametrize("mode", ["realistic", "rand", "skew"])
def test_tail_replicas(m, slots, mode):
    logits, bias = _logits(m, seed=31 * m + slots, mode=mode)
    hot = range(224, 232) if mode == "skew" else ()
    _run_replicas(logits, bias, _copy_entries(slots, seed=m + slots, hot=hot), slots)


def test_tail_replicas_move_load_off_the_busiest_rank():
    """Many hot experts on rank 1 with copies elsewhere: the step moves some of them, and the
    busiest rank touches fewer experts."""
    m = 14
    logits, bias = _logits(m, seed=5)
    logits[:, 224:264] += 6
    slots = 8
    entries = _copy_entries(slots, seed=3, hot=range(224, 264))
    sel = _run_replicas(logits, bias, entries, slots)
    num_local = REP_HOME + slots
    per_rank = [len({s for s in sel.flatten().tolist() if s // num_local == r}) for r in range(4)]
    ids, _ = _noaux(logits, bias, 1.0)
    home = [len({x for x in ids.flatten().tolist() if x // REP_HOME == r}) for r in range(4)]
    assert home.index(max(home)) == 1 and max(per_rank) < max(home), (home, per_rank)


@pytest.mark.parametrize("slots", [1, 8])
def test_tail_replicas_without_entries(slots):
    """No two-copy expert: the home slots serve, as the plain tail does."""
    m = 32
    logits, bias = _logits(m, seed=9, mode="realistic")
    _run_replicas(logits, bias, [], slots)


@pytest.mark.parametrize("m", [7, 100])
def test_tail_replicas_split_partials(m):
    gen = torch.Generator(device="cuda").manual_seed(m)
    partials = torch.randn((5, m + 3, NUM_EXPERTS), device="cuda", generator=gen)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    _run_replicas(partials, bias, _copy_entries(8, seed=m), 8, num_tokens=m)


def test_tail_replicas_routed_scaling_factor():
    logits, bias = _logits(28, seed=77, mode="realistic")
    _run_replicas(logits, bias, _copy_entries(4, seed=77), 4, scale=2.5)


def test_tail_replicas_cuda_graph_replay():
    rt = _module()
    m, slots = 14, 8
    entries = _copy_entries(slots, seed=21)
    num_local = REP_HOME + slots
    logits, bias = _logits(m, seed=4, mode="realistic")
    static = logits.clone()
    rank = 2
    table = torch.tensor(rt.replica_table(entries, rank), dtype=torch.int32, device="cuda")
    args = (bias, 1.0, rank * num_local, num_local, rank * REP_HOME, table, len(entries), TILE)
    torch.ops.trtllm.k3_route_tail_rep(static, *args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = torch.ops.trtllm.k3_route_tail_rep(static, *args)
    for seed in range(4):
        fresh, _ = _logits(m, seed=40 + seed, mode="realistic" if seed % 2 else "skew")
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        ref_ids, ref_vals = _noaux(static, bias, 1.0)
        _assert_bit_equal(outs[0], ref_ids, "topk_ids")
        sel = _copy_slots(ref_ids, entries, slots)
        _check_sort(outs[2:], sel, ref_vals, rank * num_local, num_local, REP_EP * num_local)


def test_tail_replicas_rejects_bad_arguments():
    rt = _module()
    logits, bias = _logits(7, seed=1)
    table = torch.tensor(rt.replica_table([], 0), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):  # local slots do not start at rank * local_num_experts
        torch.ops.trtllm.k3_route_tail_rep(logits, bias, 1.0, 224, 232, 224 * 2, table, 0, TILE)
    with pytest.raises(ValueError):  # more than REP_LOCAL_CAP local slots
        torch.ops.trtllm.k3_route_tail_rep(logits, bias, 1.0, 0, 300, 0, table, 0, TILE)
    with pytest.raises(ValueError):  # wrong table size
        torch.ops.trtllm.k3_route_tail_rep(logits, bias, 1.0, 0, 232, 0, table[:-4], 0, TILE)
    big, big_bias = _logits(257, seed=2)
    with pytest.raises(ValueError):  # above the cluster tail
        torch.ops.trtllm.k3_route_tail_rep(big, big_bias, 1.0, 0, 232, 0, table, 0, TILE)


@pytest.mark.parametrize("m", [1, 7, 14, 16, 28, 32, 64, 112, 256])
@pytest.mark.parametrize("slots", [4, 8, 16])
@pytest.mark.parametrize("mode", ["realistic", "rand", "skew"])
def test_tail_replicas_fast(m, slots, mode):
    """``fast`` copy choice (class masks of the table, top-k stored after the barrier): same
    choice and outputs as the reference."""
    logits, bias = _logits(m, seed=31 * m + slots, mode=mode)
    hot = range(224, 232) if mode == "skew" else ()
    _run_replicas(logits, bias, _copy_entries(slots, seed=m + slots, hot=hot), slots, fast=True)


@pytest.mark.parametrize("m", [7, 14, 32])
@pytest.mark.parametrize(
    "shape", [(8, 2, 8), (8, 4, 8), (8, 1, 4), (4, 2, 8), (2, 2, 16), (1, 1, 16)]
)
def test_tail_replicas_fast_configs(m, shape):
    if (m + shape[0] - 1) // shape[0] > 32:
        pytest.skip("more than 32 tokens per CTA")
    logits, bias = _logits(m, seed=5 * m + shape[0], mode="realistic")
    _run_replicas(logits, bias, _copy_entries(8, seed=m), 8, fast=True, shape=shape)


def test_tail_replicas_fast_many_entries_and_ties():
    """Two chunks of entries (64 two-copy experts) and heavy ties (the exact fallback) in fast
    mode, and the hot-expert case of :func:`test_tail_replicas_move_load_off_the_busiest_rank`."""
    m = 14
    logits, bias = _logits(m, seed=5)
    logits[:, 224:264] += 6
    entries = _copy_entries(16, seed=3, hot=range(224, 264))
    assert len(entries) == 64
    _run_replicas(logits, bias, entries, 16, fast=True)
    ties, tie_bias = _logits(m, seed=8, mode="ties")
    _run_replicas(ties, tie_bias, _copy_entries(8, seed=8), 8, fast=True)


@pytest.mark.parametrize("fast", [False, True])
def test_tail_replicas_prefetch_next_layer(fast):
    """Prefetching the next layer's bias and table changes nothing."""
    logits, bias = _logits(14, seed=17, mode="realistic")
    _run_replicas(logits, bias, _copy_entries(8, seed=17), 8, fast=fast, prefetch=True)


def test_tail_replicas_fast_cuda_graph_replay():
    rt = _module()
    m, slots = 14, 8
    entries = _copy_entries(slots, seed=21)
    num_local = REP_HOME + slots
    logits, bias = _logits(m, seed=4, mode="realistic")
    static = logits.clone()
    rank = 1
    table = torch.tensor(rt.replica_table(entries, rank), dtype=torch.int32, device="cuda")
    next_table = torch.tensor(rt.replica_table(entries, 2), dtype=torch.int32, device="cuda")
    outs = rt._route_tail_outputs(static, m, bias, num_local, TILE)

    def launch():
        rt.run_k3_route_tail_rep(
            static,
            m,
            bias,
            1.0,
            outs[0],
            outs[1],
            outs[2:],
            num_local,
            rank * REP_HOME,
            table,
            len(entries),
            fast=True,
            prefetch_bias=bias,
            prefetch_table=next_table,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for seed in range(4):
        fresh, _ = _logits(m, seed=40 + seed, mode="realistic" if seed % 2 else "skew")
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        ref_ids, ref_vals = _noaux(static, bias, 1.0)
        _assert_bit_equal(outs[0], ref_ids, "topk_ids")
        _assert_bit_equal(outs[1], ref_vals, "topk_weights")
        sel = _copy_slots(ref_ids, entries, slots)
        _check_sort(outs[2:], sel, ref_vals, rank * num_local, num_local, REP_EP * num_local)


def test_replica_table_class_masks():
    """Class masks: entry ``i`` sets bit ``i % 32`` of word ``2 (4 h + r) + i // 32``."""
    rt = _module()
    entries = _copy_entries(16, seed=2)
    t = rt.replica_table(entries, 0)
    assert len(t) == rt.REP_TABLE_WORDS and rt.REP_TABLE_WORDS % 4 == 0
    want = [0] * 32
    for i, (x, r) in enumerate(entries):
        want[2 * (4 * (x // REP_HOME) + r) + i // 32] |= 1 << (i % 32)
    got = [w & 0xFFFFFFFF for w in t[rt.REP_CLASS_OFF : rt.REP_CLASS_OFF + 32]]
    assert got == want
    assert all(got[2 * (5 * h)] == 0 and got[2 * (5 * h) + 1] == 0 for h in range(4))


def test_replica_table_and_choice_reference():
    rt = _module()
    entries = [(5, 1), (300, 0), (700, 1), (230, 3)]
    t = rt.replica_table(entries, 1)
    assert len(t) == rt.REP_TABLE_WORDS
    # Copies on rank 1: experts 5 and 700, copy slots 224 and 225 (ascending).
    assert t[rt.REP_COPYX_OFF] == 5 and t[rt.REP_COPYX_OFF + 1] == 700
    assert t[rt.REP_COPYX_OFF + 2] == -1
    assert t[rt.REP_MINE_OFF + 0] == 1 << 5 and t[rt.REP_MINE_OFF + 700 // 32] == 1 << (700 % 32)
    assert t[rt.REP_PREFIX_OFF + 0] == 0 and t[rt.REP_PREFIX_OFF + 1] == 1
    assert t[rt.REP_PREFIX_OFF + 700 // 32] == 1 and t[rt.REP_PREFIX_OFF + 27] == 2
    assert t[rt.REP_FLEX_OFF + 300 // 32] == 1 << (300 % 32)
    assert t[rt.REP_COUNT_OFF] == 4
    assert t[rt.REP_LIST_OFF + 1] == 300 | (1 << 10) | (0 << 12) | (1 << 14)
    # Choice: rank 0 has touched 3 single-copy experts, rank 1 one, rank 3 none.
    touched = [1, 2, 3, 250, 5, 300, 230]
    # entry (5, 1): home 0 (3) vs copy 1 (1) -> copy; entry (300, 0): home 1 (2) vs 0 (3) -> home;
    # entry (700, 1): untouched; entry (230, 3): home 1 (3) vs 3 (0) -> copy.
    assert rt.replica_choice(touched, entries) == [True, False, False, True]
    # Ties stay home.
    assert rt.replica_choice([5], [(5, 1)]) == [False]
    assert rt.replica_choice([5, 1], [(5, 1)]) == [True]
    with pytest.raises(ValueError):
        rt.replica_table([(5, 0)], 0)  # copy on the home rank


def test_moe_runtime_routes_with_expert_copies(monkeypatch):
    """``TRTLLM_K3_EXPERT_REPLICAS`` in ``KimiK3MoERuntime``: the routing tail target covers the
    rank's ``224 + S`` slots up to 256 tokens, ``_tail_route`` runs ``k3_route_tail_rep`` with the
    rank's table, and the routing method maps the home ids of results without ``moe_sort`` outputs
    to home-copy slots."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod
    from tensorrt_llm._torch.modules.kimi_k3_expert_placement import KimiK3ExpertReplicas

    rt = _module()
    slots, rank = 8, 2
    entries = tuple(_copy_entries(slots, seed=12))
    replicas = KimiK3ExpertReplicas(NUM_EXPERTS, REP_EP, slots, entries)
    num_local = replicas.num_local
    m = 14
    logits, bias = _logits(m, seed=13, mode="realistic")
    routing = _KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(
        e_score_correction_bias=bias, num_experts=NUM_EXPERTS, routed_scaling_factor=1.0
    )
    runtime.routed_experts = SimpleNamespace(
        routing_method=routing,
        use_dp=False,
        backend=SimpleNamespace(
            slot_start=rank * num_local,
            expert_size_per_partition=num_local,
            num_slots=replicas.num_slots,
        ),
    )
    runtime.expert_replicas = replicas
    runtime.expert_lo = rank * num_local
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "1")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL_MAX_M", 2048)
    assert runtime.replica_rank == rank
    target = runtime._route_tail_target(m)
    assert target == (rank * num_local, num_local)
    assert runtime._route_tail_target(257) is None  # only the cluster tail chooses copies
    ids, weights = runtime._tail_route(logits, target)
    key, sort_outputs = ids._k3_presorted
    assert key == (rank * num_local, num_local, TILE)
    ref_ids, ref_vals = _noaux(logits, bias, 1.0)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(weights, ref_vals, "topk_weights")
    sel = _copy_slots(ref_ids, list(entries), slots)
    _check_sort(sort_outputs, sel, ref_vals, rank * num_local, num_local, replicas.num_slots)
    # Fallback paths (no moe_sort outputs attached): home ids -> home-copy slots.
    mapped, _ = routing.apply(logits)
    _assert_bit_equal(
        mapped.cpu(), ref_ids.cpu() + (ref_ids.cpu() // REP_HOME) * slots, "slot_map of home ids"
    )
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "gemv")
    assert not runtime._can_gemv_route(torch.zeros(m, HIDDEN, device="cuda"))


def test_moe_runtime_tail_prefetch_targets(monkeypatch):
    """``TRTLLM_K3_ROUTE_TAIL_PREFETCH``: a layer's tail prefetches the next MoE layer's bias and
    replica table, the last layer's the first layer's; off, nothing; the routing is unchanged."""
    import weakref
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod

    _module()
    group = weakref.WeakValueDictionary()
    layers = []
    for idx in (1, 2, 3):
        runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(
            modeling_kimi_linear.KimiK3MoERuntime
        )
        nn.Module.__init__(runtime)
        runtime.layer_idx = idx
        runtime._moe_group = group
        bias = torch.randn((NUM_EXPERTS,), device="cuda") * 0.01
        runtime.gate = SimpleNamespace(
            e_score_correction_bias=bias, num_experts=NUM_EXPERTS, routed_scaling_factor=1.0
        )
        runtime.routed_experts = SimpleNamespace(
            routing_method=_KimiK3RoutingMethod(
                top_k=TOP_K,
                n_group=1,
                topk_group=1,
                routed_scaling_factor=1.0,
                callable_e_score_correction_bias=lambda b=bias: b,
                is_fused=True,
            ),
            use_dp=False,
            backend=SimpleNamespace(slot_start=224, expert_size_per_partition=224, num_slots=896),
        )
        group[idx] = runtime
        layers.append(runtime)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL_PREFETCH", False)
    assert layers[0]._tail_prefetch_targets() == (None, None)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL_PREFETCH", True)
    assert layers[0]._tail_prefetch_targets()[0] is layers[1].gate.e_score_correction_bias
    assert layers[1]._tail_prefetch_targets()[0] is layers[2].gate.e_score_correction_bias
    assert layers[2]._tail_prefetch_targets()[0] is layers[0].gate.e_score_correction_bias
    assert layers[0]._tail_prefetch_targets()[1] is None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "1")
    m = 14
    logits, _ = _logits(m, seed=23, mode="realistic")
    bias = layers[0].gate.e_score_correction_bias
    target = layers[0]._route_tail_target(m)
    assert target == (224, 224)
    ids, weights = layers[0]._tail_route(logits, target)
    ref_ids, ref_vals = _noaux(logits, bias, 1.0)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(weights, ref_vals, "topk_weights")
    key, sort_outputs = ids._k3_presorted
    _check_outputs((ids, weights, *sort_outputs), logits, bias, 1.0, 224, 224)

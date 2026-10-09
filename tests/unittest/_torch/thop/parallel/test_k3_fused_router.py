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
"""Correctness of the fused Kimi-K3 router (``trtllm::k3_fused_router_{topk,sort}``).

The fused kernel replaces ``dsv3_router_gemm_op`` -> ``noaux_tc_op`` -> ``moe_sort``. Its
FP32 accumulation order differs from the reference GEMM, so:

* the logits must match the FP32 product (and ``dsv3_router_gemm_op``) to a tight tolerance;
* top-k ids and weights must be bit-identical to ``noaux_tc_op`` applied to the kernel's own
  logits;
* the ``moe_sort`` outputs must be bit-identical to a token-ordered sort of the kernel's own
  ids (experts with more than 128 tokens span several tiles). ``moe_sort`` itself produces
  exactly that order up to 12 tokens (single-CTA kernels); above that its cluster kernel orders
  the rows of a tile by shared-memory atomics (it is not deterministic from run to run), so
  there only the tile metadata must match exactly and the rows of each tile must be the same
  set.
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
# M <= 64 runs the mma.sync GEMM path, larger M the tcgen05 one.
MS = [1, 7, 14, 17, 28, 33, 56, 64, 65, 112, 128, 168, 224, 256]
# (local_expert_offset, local_num_experts): EP16 ranks, TP4xEP4 ranks, TP8xEP2 ranks, no EP.
LAYOUTS = [
    (0, 56),
    (7 * 56, 56),
    (15 * 56, 56),
    (0, 224),
    (3 * 224, 224),
    (0, 448),
    (448, 448),
    (0, 896),
]


def _kernel_module():
    import tensorrt_llm  # noqa: F401  (registers the reference ops)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fused_router

    return k3_fused_router


def _inputs(m, seed, *, w_std=0.02, bias_std=0.01):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn((NUM_EXPERTS, HIDDEN), device="cuda", generator=gen) * w_std).to(
        torch.bfloat16
    )
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * bias_std
    return h, w, bias


def _noaux(logits, bias, scale):
    vals, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, scale)
    return ids.to(torch.int32), vals.to(torch.float32)


def _assert_bit_equal(actual, expected, what):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, what
    a = actual.view(torch.int32) if actual.dtype == torch.float32 else actual
    e = expected.view(torch.int32) if expected.dtype == torch.float32 else expected
    mismatch = (a != e).nonzero()
    assert mismatch.numel() == 0, f"{what}: {mismatch.shape[0]} mismatches, first {mismatch[:4]}"


def _check_topk(k3, h, w, bias, scale=1.0):
    ids, vals, logits = torch.ops.trtllm.k3_fused_router_topk(h, w, bias, scale, True)
    exact = (h.double() @ w.double().t()).float()
    torch.testing.assert_close(logits, exact, rtol=0, atol=5e-5)
    ref_gemm = torch.ops.trtllm.dsv3_router_gemm_op(h, w.t(), bias=None, out_dtype=torch.float32)
    torch.testing.assert_close(logits, ref_gemm, rtol=0, atol=1e-4)
    ref_ids, ref_vals = _noaux(logits, bias, scale)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")
    return ids, vals, logits


def _token_ordered_sort(ids, offset, num_local):
    """Python ``moe_sort`` with the rows of each expert in ascending token order."""
    m = ids.shape[0]
    ids = ids.cpu()
    rows = {}
    for t in range(m):
        for k in range(TOP_K):
            j = int(ids[t, k]) - offset
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


def _check_sort(k3, h, w, bias, offset, num_local):
    outs = torch.ops.trtllm.k3_fused_router_sort(h, w, bias, 1.0, offset, num_local, TILE, True)
    ids, vals, tile_expert, tile_limit, e2p, p2e, total, num_tiles, logits = outs
    ref_ids, ref_vals = _noaux(logits, bias, 1.0)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")

    m = h.shape[0]
    ref = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=vals,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=num_local,
        tile_tokens_dim=TILE,
    )
    for got, exp, name in zip(
        (tile_expert, tile_limit, e2p, p2e, total, num_tiles),
        ref,
        (
            "tile_idx_to_expert_idx",
            "tile_idx_to_mn_limit",
            "expanded_idx_to_permuted_idx",
            "permuted_idx_to_expanded_idx",
            "total_num_padded_tokens",
            "num_non_exiting_tiles",
        ),
    ):
        assert got.dtype == exp.dtype == torch.int32 and got.shape == exp.shape, name
    n = int(num_tiles.item())
    _assert_bit_equal(num_tiles, ref[5], "num_non_exiting_tiles")
    _assert_bit_equal(total, ref[4], "total_num_padded_tokens")
    _assert_bit_equal(tile_expert[:n], ref[0][:n], "tile_idx_to_expert_idx")
    _assert_bit_equal(tile_limit[:n], ref[1][:n], "tile_idx_to_mn_limit")

    # Exact against the token-ordered sort of the same ids.
    c_expert, c_limit, c_e2p, c_p2e = _token_ordered_sort(ids, offset, num_local)
    assert tile_expert[:n].tolist() == c_expert and tile_limit[:n].tolist() == c_limit
    _assert_bit_equal(e2p.cpu(), c_e2p, "expanded_idx_to_permuted_idx (token order)")
    p2e_cpu = p2e.cpu()
    for perm, expanded in c_p2e.items():
        assert int(p2e_cpu[perm]) == expanded, f"permuted_idx_to_expanded_idx[{perm}]"

    # Against moe_sort: identical when its single-CTA kernels run, same rows per tile otherwise.
    ref_e2p, ref_p2e = ref[2].cpu(), ref[3].cpu()
    if m <= 12:
        _assert_bit_equal(e2p.cpu(), ref_e2p, "expanded_idx_to_permuted_idx vs moe_sort")
    else:
        _assert_bit_equal(e2p.cpu() < 0, ref_e2p < 0, "non-local entries vs moe_sort")
    for t in range(n):
        lo, hi = t * TILE, int(tile_limit[t])
        assert sorted(p2e_cpu[lo:hi].tolist()) == sorted(ref_p2e[lo:hi].tolist()), f"tile {t}"
    return outs


@pytest.mark.parametrize("m", MS)
def test_topk_matches_noaux_on_own_logits(m):
    k3 = _kernel_module()
    h, w, bias = _inputs(m, seed=m)
    ids, vals, logits = _check_topk(k3, h, w, bias)
    # The selection also agrees with the original chain except at near-ties.
    ref_logits = torch.ops.trtllm.dsv3_router_gemm_op(h, w.t(), bias=None, out_dtype=torch.float32)
    chain_ids, _ = _noaux(ref_logits, bias, 1.0)
    same = sum(len(set(a) & set(b)) for a, b in zip(ids.cpu().tolist(), chain_ids.cpu().tolist()))
    assert same >= 0.99 * m * TOP_K


@pytest.mark.parametrize("m", [7, 28])
@pytest.mark.parametrize("scale", [2.5, 0.3])
def test_topk_routed_scaling_factor(m, scale):
    k3 = _kernel_module()
    h, w, bias = _inputs(m, seed=100 + m)
    _check_topk(k3, h, w, bias, scale)


@pytest.mark.parametrize("m", [1, 7, 28, 64])
def test_topk_ties(m):
    """Exact score ties must break toward the smaller expert index like ``noaux_tc_op``."""
    k3 = _kernel_module()
    h, w, bias = _inputs(m, seed=200 + m)
    # Experts scattered over 112 classes of identical weight rows and biases: their logits and
    # scores tie exactly.
    gen = torch.Generator(device="cuda").manual_seed(m)
    cls = torch.randperm(NUM_EXPERTS, device="cuda", generator=gen) % (NUM_EXPERTS // 8)
    w = w[cls].contiguous()
    bias = bias[cls].contiguous()
    _check_topk(k3, h, w, bias)
    # All scores equal: experts 0..15 with weight 1/16.
    zeros = torch.zeros_like(h)
    ids, vals, _ = _check_topk(k3, zeros, w, torch.zeros_like(bias))
    assert ids.cpu().tolist() == [list(range(TOP_K))] * m
    # Large negative biases (negative selection scores).
    _check_topk(k3, h, w, bias - 2.0)


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_sort_matches_moe_sort(m, layout):
    k3 = _kernel_module()
    h, w, bias = _inputs(m, seed=300 + m)
    _check_sort(k3, h, w, bias, *layout)


@pytest.mark.parametrize("m", [4, 12, 129, 200, 256])
def test_sort_concentrated_routing(m):
    """Every token routed to the same 16 experts (bias dominates the score); above 128 tokens
    each of them spans two tiles, the second one partial."""
    k3 = _kernel_module()
    h, w, bias = _inputs(m, seed=400 + m)
    bias = torch.zeros_like(bias)
    bias[100:116] = 10.0
    outs = _check_sort(k3, h, w, bias, 56, 56)
    # Experts 100..111 of the 16 favored ones are local to [56, 112).
    assert int(outs[7]) == 12 * ((m + TILE - 1) // TILE)
    _check_sort(k3, h, w, bias, 0, 224)
    _check_sort(k3, h, w, bias, 0, 448)


def test_cuda_graph_replay():
    """Replays must reproduce eager results (the kernel re-arms its persistent workspace)."""
    k3 = _kernel_module()
    m, m_big = 28, 200
    h, w, bias = _inputs(m_big, seed=500)
    w2 = (w.float() * -0.5).to(torch.bfloat16)
    static_big = h.clone()
    static_h = static_big[:m]

    def run():
        a = torch.ops.trtllm.k3_fused_router_sort(static_h, w, bias, 1.0, 0, 56, TILE, True)
        b = torch.ops.trtllm.k3_fused_router_topk(static_big, w2, bias, 1.0, False)
        c = torch.ops.trtllm.k3_fused_router_sort(static_big, w2, bias, 1.0, 448, 448, TILE, False)
        return a, b, c

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for step in range(4):
        new_h, _, _ = _inputs(m_big, seed=600 + step)
        static_big.copy_(new_h)
        graph.replay()
        torch.cuda.synchronize()
        eager = run()
        for got, exp in zip(captured, eager):
            for g, e in zip(got[:2], exp[:2]):
                _assert_bit_equal(g, e, f"replay {step}")
        n = int(eager[0][7].item())
        _assert_bit_equal(captured[0][2][:n], eager[0][2][:n], "replay tiles")
        _assert_bit_equal(captured[0][4], eager[0][4], "replay expanded_idx_to_permuted_idx")
        _assert_bit_equal(captured[2][4], eager[2][4], "replay expanded_idx_to_permuted_idx")

    # Between launches the arrival counter is zero and every hand-off slot (expert ids and both
    # paths' partial logits) holds the sentinel again.
    for counter, slots in k3._WORKSPACES.values():
        assert int(counter[0]) == 0
        assert bool((slots == -1).all())


def test_rejects_unsupported_inputs():
    k3 = _kernel_module()
    h, w, bias = _inputs(257, seed=1)
    assert not k3.can_use_k3_fused_router(h, w, bias)
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_fused_router_topk(h, w, bias, 1.0, False)
    assert not k3.can_use_k3_fused_router(h[:8], w, bias.half())
    assert k3.can_use_k3_fused_router(h[:8], w, bias)


# ---------------------------------------------------------------------------
# Standalone low-M GEMV (``trtllm::k3_bf16_lowm_gemv``)
# ---------------------------------------------------------------------------

# (N, K): the KDA decode projections, plus edge shapes (single 8-row tile, N not a multiple
# of 16, split-K at large N).
GEMV_SHAPES = [(136, 7168), (768, 128), (8, 128), (1000, 1024), (4096, 7168)]


@pytest.mark.parametrize("shape", GEMV_SHAPES)
@pytest.mark.parametrize("m", [1, 7, 28, 56, 64])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_gemv_matches_fp32(shape, m, out_dtype):
    _kernel_module()
    n, k = shape
    gen = torch.Generator(device="cuda").manual_seed(n + k + m)
    x = torch.randn((m, k), device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    out = torch.ops.trtllm.k3_bf16_lowm_gemv(x, w, out_dtype)
    assert out.dtype == out_dtype and out.shape == (m, n)
    ref = x.float() @ w.float().t()
    if out_dtype == torch.float32:
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5 * k**0.5)
    else:
        torch.testing.assert_close(out.float(), ref, rtol=2**-8, atol=1e-5 * k**0.5)
    # Split-K partials are reduced in a fixed order: repeated calls are bit-identical.
    assert torch.equal(out, torch.ops.trtllm.k3_bf16_lowm_gemv(x, w, out_dtype))


def test_gemv_cuda_graph_replay():
    k3 = _kernel_module()
    n, k, m = 136, 7168, 28
    assert (
        k3.gemv_num_splits(n, k, m, torch.cuda.get_device_properties(0).multi_processor_count) > 1
    )
    x = torch.randn((m, k), device="cuda").to(torch.bfloat16)
    w = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops.trtllm.k3_bf16_lowm_gemv(x, w, torch.float32)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.trtllm.k3_bf16_lowm_gemv(x, w, torch.float32)
    for _ in range(3):
        x.copy_(torch.randn((m, k), device="cuda").to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, torch.ops.trtllm.k3_bf16_lowm_gemv(x, w, torch.float32))
    for counters in k3._GEMV_COUNTERS.values():
        assert int(counters.abs().sum()) == 0


def test_gemv_rejects_unsupported_inputs():
    k3 = _kernel_module()
    x = torch.randn((8, 7168), device="cuda").to(torch.bfloat16)
    assert not k3.can_use_k3_bf16_lowm_gemv(x, torch.zeros((12, 7168), device="cuda").bfloat16())
    assert not k3.can_use_k3_bf16_lowm_gemv(
        x[:, :7000], torch.zeros((8, 7000), device="cuda").bfloat16()
    )
    assert k3.can_use_k3_bf16_lowm_gemv(x, torch.zeros((8, 7168), device="cuda").bfloat16())

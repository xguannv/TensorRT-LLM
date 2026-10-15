# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tactic pins of the NVFP4 MoE gather (FC1) / finalize (FC2) kernels (TRTLLM_MOE_GATHER_GEMM_TACTIC,
TRTLLM_MOE_FINALIZE_TACTIC): the autotuner candidates they leave, and that the tactics they choose between
compute the same bits."""

import pytest
import torch

from tensorrt_llm._torch.utils import ActivationType

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)

# One expert-parallel rank of the Kimi-K3 routed MoE (EP16).
NUM_EXPERTS, TOP_K, NUM_LOCAL, HIDDEN, INTER = 896, 16, 56, 3584, 3072
LOCAL_OFFSET = NUM_LOCAL


def _cdo():
    import tensorrt_llm  # noqa: F401  (registers the ops)
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    return cute_dsl_custom_ops


def _inputs(num_tokens, one_local_expert_per_token, seed=0):
    """Random NVFP4 operands and the moe_sort metadata of a random routing of ``num_tokens``."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)

    def rb(shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=gen, dtype=torch.uint8, device=dev)

    cpu = torch.Generator().manual_seed(seed)
    if one_local_expert_per_token:
        local = torch.randint(0, NUM_LOCAL, (num_tokens, 1), generator=cpu) + LOCAL_OFFSET
        remote = torch.stack(
            [
                torch.randperm(NUM_EXPERTS - NUM_LOCAL, generator=cpu)[: TOP_K - 1]
                for _ in range(num_tokens)
            ]
        )
        remote = torch.where(remote >= LOCAL_OFFSET, remote + NUM_LOCAL, remote)
        ids = torch.cat([local, remote], 1)
    else:
        ids = torch.stack(
            [torch.randperm(NUM_EXPERTS, generator=cpu)[:TOP_K] for _ in range(num_tokens)]
        )
    ids = ids.to(torch.int32).to(dev)
    scales = torch.rand(num_tokens, TOP_K, generator=gen, device=dev).softmax(-1)
    t2g, t2lim, _, p2e, _, nne = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=LOCAL_OFFSET,
        local_num_experts=NUM_LOCAL,
        tile_tokens_dim=128,
    )
    fp4 = torch.float4_e2m1fn_x2
    g1 = [
        rb((num_tokens, HIDDEN // 2)).view(fp4),
        rb((NUM_LOCAL, 2 * INTER, HIDDEN // 2)).view(fp4),
        rb((num_tokens, HIDDEN // 16), 0x30, 0x40),
        rb((NUM_LOCAL, 2 * INTER, HIDDEN // 16), 0x30, 0x40),
        torch.full((NUM_LOCAL,), 1e-2, dtype=torch.float32, device=dev),
        t2g,
        t2lim,
        p2e,
        nne,
        torch.ones(1, dtype=torch.float32, device=dev),
    ]
    w2 = rb((NUM_LOCAL, HIDDEN, INTER // 2)).view(fp4)
    w2sf = rb((NUM_LOCAL, HIDDEN, INTER // 16), 0x30, 0x40)
    alpha2 = torch.full((NUM_LOCAL,), 1e-2, dtype=torch.float32, device=dev)

    def g2(c, c_sf, out):
        return [c, w2, c_sf, w2sf, alpha2, out, t2g, t2lim, p2e, nne, scales]

    return g1, g2


def _runners(cdo):
    r1 = cdo.Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner(
        NUM_EXPERTS,
        TOP_K,
        NUM_LOCAL,
        LOCAL_OFFSET,
        128,
        16,
        ActivationType.SiTu,
        float("inf"),
        1.0,
        1.0,
    )
    r2 = cdo.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
        NUM_EXPERTS, TOP_K, NUM_LOCAL, LOCAL_OFFSET, 128, torch.bfloat16
    )
    return r1, r2


@pytest.mark.parametrize(
    "g1_pin, g2_pin, g1_expect, g2_expect",
    [
        ("", "", {128, 256}, {(128, 1), (128, 2), (256, 1), (256, 2)}),
        ("128", "128x2", {128}, {(128, 2)}),
        ("256", "128", {256}, {(128, 1), (128, 2)}),
        ("64", "512x1", {128, 256}, {(128, 1), (128, 2), (256, 1), (256, 2)}),
    ],
)
def test_tactic_pin_filters_candidates(monkeypatch, g1_pin, g2_pin, g1_expect, g2_expect):
    cdo = _cdo()
    monkeypatch.setenv("TRTLLM_MOE_GATHER_GEMM_TACTIC", g1_pin)
    monkeypatch.setenv("TRTLLM_MOE_FINALIZE_TACTIC", g2_pin)
    r1, r2 = _runners(cdo)
    g1, g2 = _inputs(256, one_local_expert_per_token=False)
    t1 = r1.get_valid_tactics(g1, None)
    assert {t[0][1] for t in t1} == g1_expect
    c = torch.empty(g1[7].size(0), INTER // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    c_sf = torch.empty(g1[7].size(0) * INTER // 16, dtype=torch.uint8, device="cuda")
    out = torch.empty(256, HIDDEN, dtype=torch.bfloat16, device="cuda")
    t2 = r2.get_valid_tactics(g2(c, c_sf, out), None)
    assert {(t[0][1], t[1][1]) for t in t2} == g2_expect
    # The pin is part of the runner identity, so tuning caches of different pins never mix.
    assert r1.unique_id()[-1] == (tuple(int(v) for v in g1_pin.split("x")) if g1_pin else None)


def test_invalid_tactic_pin_raises(monkeypatch):
    cdo = _cdo()
    monkeypatch.setenv("TRTLLM_MOE_GATHER_GEMM_TACTIC", "wide")
    with pytest.raises(ValueError, match="TRTLLM_MOE_GATHER_GEMM_TACTIC"):
        _runners(cdo)


@pytest.mark.parametrize("num_tokens", [256, 896])
def test_pinned_tactics_compute_the_same_bits(num_tokens):
    """FC1 with MMA tile N 128 or 256, and FC2 with cluster N 1 or 2, give bitwise-equal outputs
    (FC2 with one local expert per token, so the finalize scatter-add order cannot differ)."""
    cdo = _cdo()
    r1, r2 = _runners(cdo)
    g1, g2 = _inputs(num_tokens, one_local_expert_per_token=True, seed=num_tokens)
    c_a, sf_a = r1(g1, tactic=((128, 128), (1, 1), False))
    c_b, sf_b = r1(g1, tactic=((128, 256), (1, 1), False))
    rows = int(g1[8].item()) * 128  # rows of the valid (non-exiting) tiles
    assert torch.equal(c_a.view(torch.uint8)[:rows], c_b.view(torch.uint8)[:rows])
    sf_rows = sf_a.view(-1, INTER // 16)
    assert torch.equal(sf_rows[:rows], sf_b.view(-1, INTER // 16)[:rows])

    outs = []
    for cluster_n in (1, 2):
        out = torch.zeros(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        r2(g2(c_a, sf_a, out), tactic=((128, 128), (1, cluster_n), False))
        outs.append(out)
    torch.cuda.synchronize()
    assert outs[0].abs().sum() > 0
    assert torch.equal(outs[0], outs[1])

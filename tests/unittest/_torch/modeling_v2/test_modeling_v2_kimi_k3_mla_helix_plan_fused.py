# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED``: `make_helix_plan` in one launch gives the torch path's tensors bit for bit,
also inside a CUDA graph."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PAGE = 64


def _mods():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
        flags,
    )

    return decode_mla_dcp, flags


def _inputs(batch, local_blocks, q_len, rows_dtype, positions_dtype, seed, transposed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if transposed:
        rows = torch.randint(0, 1 << 20, (local_blocks, batch), generator=g).t()
    else:
        rows = torch.randint(0, 1 << 20, (batch, local_blocks), generator=g)
    lens = torch.randint(0, 9000, (batch,), generator=g)
    start = torch.randint(-3, 600_000, (batch, 1), generator=g)
    positions = (start + torch.arange(q_len)).reshape(-1)
    # edge positions: page and ledger boundaries, the first token, a padded slot
    edges = torch.tensor([0, PAGE - 1, PAGE, 16 * PAGE - 1, 16 * PAGE, 16 * PAGE + 5 * PAGE, -1])
    positions[: min(len(edges), positions.numel())] = edges[: positions.numel()]
    return (
        rows.to(rows_dtype).cuda(),
        lens.to(torch.int64 if rows_dtype == torch.int64 else torch.int32).cuda(),
        positions.to(positions_dtype).cuda(),
    )


def _assert_same(a, b):
    assert (
        a.page_table.shape == b.page_table.shape and a.page_table.stride() == b.page_table.stride()
    )
    for x, y in (
        (a.page_table, b.page_table),
        (a.cache_seqs, b.cache_seqs),
        (a.kv_bounds, b.kv_bounds),
    ):
        assert x.dtype == y.dtype == torch.int32 and torch.equal(x, y)
    assert a.empty.dtype == b.empty.dtype == torch.bool and torch.equal(a.empty, b.empty)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "rows_dtype,positions_dtype", [(torch.int32, torch.int32), (torch.int64, torch.int64)]
)
@pytest.mark.parametrize("cp_rank", [0, 5, 15])
@pytest.mark.parametrize(
    "batch,local_blocks,q_len", [(1, 1, 1), (2, 40, 8), (3, 300, 8), (4, 513, 8), (8, 7, 6)]
)
def test_fused_plan_bitwise(
    batch, local_blocks, q_len, cp_rank, rows_dtype, positions_dtype, transposed, monkeypatch
):
    dcp, flags = _mods()
    rows, lens, positions = _inputs(
        batch, local_blocks, q_len, rows_dtype, positions_dtype, batch + cp_rank, transposed
    )
    monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED", 0)
    ref = dcp.make_helix_plan(rows, lens, positions, PAGE, 16, cp_rank)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED", 1)
    got = dcp.make_helix_plan(rows, lens, positions, PAGE, 16, cp_rank)
    _assert_same(got, ref)


def test_fused_plan_in_graph(monkeypatch):
    dcp, flags = _mods()
    monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED", 1)
    rows, lens, positions = _inputs(4, 100, 8, torch.int32, torch.int32, 1, False)
    out = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dcp.make_helix_plan(rows, lens, positions, PAGE, 16, 3)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.append(dcp.make_helix_plan(rows, lens, positions, PAGE, 16, 3))
    for seed in range(3):
        new_rows, new_lens, new_positions = _inputs(
            4, 100, 8, torch.int32, torch.int32, 10 + seed, False
        )
        rows.copy_(new_rows)
        lens.copy_(new_lens)
        positions.copy_(new_positions)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED", 0)
        _assert_same(out[0], dcp.make_helix_plan(rows, lens, positions, PAGE, 16, 3))
        monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED", 1)


def test_switch_default_off_and_documented():
    _, flags = _mods()
    value, default, doc = flags.flags()["TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED"]
    assert default == 0 and "bitwise" in doc

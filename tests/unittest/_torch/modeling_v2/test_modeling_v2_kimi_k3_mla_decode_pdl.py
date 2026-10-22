# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_MLA_DECODE_PDL``: the split read's CuTe DSL FP8 MLA decode with programmatic dependent launch
(``kimi_k3_nvfp4/mla_decode_pdl.py``), one GPU.

The switch is read by ``decode_mla_dcp.split_read_attention`` per call. Both arms against the op the off arm calls,
``trtllm::cute_dsl_mla_decode_fp8_blackwell`` called directly (the chain before the switch), bit for bit: the latent
output and, in the Helix form, the softmax statistics. PDL changes only when a grid may start, not the arithmetic,
so the on arm is bit for bit too. The forms the staircase calls: Helix (every head, this rank's pages, statistics and
per-token bounds) at 8 / 16 / 32 rows per head and TP16's mode 1 (6 heads, full KV, no statistics). Eagerly and from a
CUDA graph replayed with new inputs, in which a kernel right before each decode rewrites its queries (a grid that read
them before the wait would see the old ones).
"""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="CuTe DSL MLA decode runs on SM100/SM103/SM107 only",
)

LATENT, ROPE, PAGE, CP = 512, 64, 64, 16
LAYERS = 2


def _mods():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
        flags,
    )

    return decode_mla_dcp, flags


def _cache(seq_lens, q_len, heads, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = [max(1, math.ceil(n / PAGE)) for n in seq_lens]
    num_pages = sum(blocks) + 5
    perm = torch.randperm(num_pages, generator=g)
    block_rows = torch.zeros(len(seq_lens), max(blocks), dtype=torch.int32)
    k = 0
    for b, nb in enumerate(blocks):
        block_rows[b, :nb] = perm[k : k + nb].to(torch.int32)
        k += nb
    pool = (torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 0.5).to(device)
    q = (torch.randn(len(seq_lens), q_len, heads, LATENT + ROPE, generator=g) * 0.5).to(device)
    return pool.to(torch.float8_e4m3fn), block_rows.to(device), q.to(torch.float8_e4m3fn)


class _Arm:
    """One decode call on persistent buffers: ``via`` "op" calls the shared op directly, "off" / "on" call
    ``split_read_attention`` with the switch at 0 / 1."""

    def __init__(self, dcp, via, q, pool, plan, helix, max_batch):
        batch, q_len, heads, _ = q.shape
        self.dcp, self.via, self.pool, self.plan, self.max_batch = dcp, via, pool, plan, max_batch
        self.q_src = q.clone()
        self.q = torch.empty_like(q)
        self.out = torch.full((batch, q_len, heads, LATENT), float("nan"), dtype=torch.bfloat16,
                              device=q.device)  # fmt: skip
        self.stats = (
            torch.full((batch * q_len, heads, 2), float("nan"), device=q.device) if helix else None
        )
        self.ws = torch.empty(
            dcp.workspace_bytes(heads, q_len, max_batch), dtype=torch.uint8, device=q.device
        )
        self.scale = dcp.default_softmax_scale()

    def __call__(self):
        _, flags = _mods()
        # the kernel right before the decode writes its queries
        self.q.view(torch.uint8).copy_(self.q_src.view(torch.uint8))
        if self.via == "op":
            batch, q_len, heads, _ = self.q.shape
            torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell(
                self.q[..., :LATENT].permute(2, 3, 1, 0),
                self.q[..., LATENT:].permute(2, 3, 1, 0),
                self.pool[..., :LATENT].permute(1, 2, 0),
                self.pool[..., LATENT : LATENT + ROPE].permute(1, 2, 0),
                self.plan.page_table,
                self.plan.cache_seqs,
                self.out.permute(2, 3, 1, 0),
                self.ws,
                heads,
                q_len,
                PAGE,
                self.scale,
                1.0,
                self.max_batch,
                self.stats,
                self.plan.kv_bounds,
            )
        else:
            saved = flags.TRTLLM_K3LL_MLA_DECODE_PDL
            flags.TRTLLM_K3LL_MLA_DECODE_PDL = int(self.via == "on")
            try:
                self.dcp.split_read_attention(
                    self.q, self.pool, self.plan, self.out, self.stats, self.ws, self.scale, 1.0,
                    self.max_batch,
                )  # fmt: skip
            finally:
                flags.TRTLLM_K3LL_MLA_DECODE_PDL = saved
        return self.out, self.stats


def _same(a, b, what):
    for x, y, name in ((a[0], b[0], "latent"), (a[1], b[1], "stats")):
        if x is None:
            continue
        assert torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32),
                           y.view(torch.int16) if y.dtype == torch.bfloat16 else y.view(torch.int32)), (
            f"{what}: {name} differs bitwise"
        )  # fmt: skip


FORMS = {
    # name: (global lengths, q_len, heads, rank or None for mode 1)
    "helix_8_rows": ([153600 + 8], 8, 96, 5),
    "helix_16_rows": ([153600 + 8, 70000 + 8], 8, 96, 3),
    "helix_32_rows": ([153600 + 8, 20000 + 8, 300, 5000], 8, 96, 7),
    "tp16_mode1_8_rows": ([20000 + 8], 8, 6, None),
    "tp16_mode1_32_rows": ([20000, 9000, 300, 5000], 8, 6, None),
}


@pytest.mark.parametrize("form", list(FORMS))
def test_switch_arms_bit_identical_to_the_op(form):
    dcp, _ = _mods()
    device = torch.device("cuda")
    seq_lens, q_len, heads, rank = FORMS[form]
    pool, block_rows, q = _cache(seq_lens, q_len, heads, 11, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    if rank is None:
        plan = dcp.SplitReadPlan(block_rows.t(), lens, None, None)
    else:
        plan = dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, CP, rank)
    max_batch = dcp.MAX_REQUESTS
    arms = {
        via: _Arm(dcp, via, q, pool, plan, rank is not None, max_batch)
        for via in ("op", "off", "on")
    }
    ref = [t.clone() if t is not None else None for t in arms["op"]()]
    for via in ("off", "on"):
        _same(arms[via](), ref, f"{form} {via} eager")

    for via in ("op", "off", "on"):
        arm = arms[via]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(LAYERS):
                arm()
        for seed in (21, 22):
            _, _, q_new = _cache(seq_lens, q_len, heads, seed, device)
            arm.q_src.copy_(q_new)
            arm.out.fill_(float("nan"))
            if arm.stats is not None:
                arm.stats.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            got = [t.clone() if t is not None else None for t in (arm.out, arm.stats)]
            eager = _Arm(dcp, "op", q_new, pool, plan, rank is not None, max_batch)()
            _same(got, eager, f"{form} {via} replay {seed}")

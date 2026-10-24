# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``trtllm::k3_fp8_kda_verify`` (the NVFP4 staircase's KDA verify) at every verify width the staircase
takes, odd ones included (MTP6: 7 tokens per request), on a real cache object built for that width
(``_kda_cells.build_manager``). Every round's outputs and the state committed after each golden token against the
float64 verify over each request's committed history (``_kda_cells.F64Verify``), the next round starting from the
drafts the sampler accepted; the other slots untouched. Also a captured round replayed against eager, at 7 tokens.
"""

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.verify import (
    k3_fp8_kda_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="requires an SM100-family GPU",
)

ROUNDS = 4
POOL_NAMES = ("cs_q", "cs_k", "cs_v", "ssm", "state_tok")
# (requests, 1 + num_spec): MTP6 (7 tokens) at bs1 / bs2 / bs4 / bs8, then the other widths 2..8.
CELLS = [(1, 7), (2, 7), (4, 7), (8, 7), (2, 8), (2, 6), (2, 5), (2, 3), (2, 2)]


def _verify(wt, pools, proj_src, slots, num_spec) -> torch.Tensor:
    proj = proj_src.clone()  # a plain copy kernel before the launch, as in the model
    return k3_fp8_kda_verify(
        proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slots, pools["pending"],
        num_spec, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
    )  # fmt: skip


def _proj(num: int, steps: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(num * steps, kc.PROJ, generator=gen, device="cuda").bfloat16()


@pytest.mark.parametrize("num,steps", CELLS, ids=[f"{r}x{t}" for r, t in CELLS])
def test_verify_widths_against_f64(num, steps) -> None:
    kc.load_ops()
    num_spec = steps - 1
    mgr = kc.build_manager(1, num_spec=num_spec)
    try:
        slots = kc.request_slots(mgr, num, first_id=800 + 10 * steps)
        p = kc.layer_pools(mgr, 0)
        with torch.inference_mode():
            kc.fill_pools(p, 3000 + 10 * num + steps)
            wt = kc.make_weights(700 + steps)
            ref = kc.F64Verify(wt, p, slots, num_spec)
            gen = torch.Generator(device="cuda").manual_seed(70 + 10 * num + steps)
            for rnd in range(ROUNDS):
                before = kc.snapshot(p)
                proj = _proj(num, steps, gen)
                out = _verify(wt, p, proj, slots, num_spec)
                want = ref(proj)
                torch.cuda.synchronize()
                assert kc.rel(out, want) <= kc.TOL_OUT, (rnd, kc.rel(out, want))
                for n, s in enumerate(slots.tolist()):
                    assert kc.rel(p["ssm"][s], ref.last[n][0][0]) <= kc.TOL_STATE, (rnd, n)
                others = kc.other_slots(p["ssm"].shape[0], slots.tolist())
                assert kc.same_rows(p, before, others, POOL_NAMES), rnd
                accepted = kc.pending_schedule(num, num_spec, rnd)
                p["pending"][slots.long()] = torch.tensor(
                    accepted, dtype=torch.int32, device="cuda"
                )
                ref.commit(accepted)
    finally:
        mgr.shutdown()


def test_seven_tokens_graph_replay() -> None:
    """2 requests x 7 tokens: one captured launch replayed ROUNDS times equals the same rounds run eagerly."""
    kc.load_ops()
    num, steps = 2, 7
    num_spec = steps - 1
    mgr = kc.build_manager(1, num_spec=num_spec)
    try:
        slots = kc.request_slots(mgr, num, first_id=900)
        p = kc.layer_pools(mgr, 0)
        with torch.inference_mode():
            kc.fill_pools(p, 3100)
            eager = kc.clone_pools(p)
            wt = kc.make_weights(710)
            gen = torch.Generator(device="cuda").manual_seed(71)
            projs = [_proj(num, steps, gen) for _ in range(ROUNDS)]
            want = []
            for rnd in range(ROUNDS):
                want.append(_verify(wt, eager, projs[rnd], slots, num_spec).clone())
                eager["pending"][slots.long()] = torch.tensor(
                    kc.pending_schedule(num, num_spec, rnd), dtype=torch.int32, device="cuda"
                )
            static = projs[0].clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                out = _verify(wt, p, static, slots, num_spec)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            # the capture did not run the kernel; start from the same pools as the eager rounds did
            for rnd in range(ROUNDS):
                static.copy_(projs[rnd])
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, want[rnd]), rnd
                p["pending"][slots.long()] = torch.tensor(
                    kc.pending_schedule(num, num_spec, rnd), dtype=torch.int32, device="cuda"
                )
        assert all(torch.equal(p[n], eager[n]) for n in POOL_NAMES)
    finally:
        mgr.shutdown()

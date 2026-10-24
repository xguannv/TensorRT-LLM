# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for TRTLLM_K3LL_KDA_VERIFY_HOIST (``trtllm::k3_fp8_kda_verify``'s prologue loads issued while the cluster
forms): the on arm bit for bit the off arm, outputs and every pool (state, records, conv caches), at every verify width
the staircase takes and every count of accepted drafts, on a real cache object with a BF16 and an FP32 pool state;
eager and a captured launch replayed; the census line counted only on the on arm.
"""

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import census
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags as F
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.verify import (
    k3_fp8_kda_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="requires an SM100-family GPU",
)

POOL_NAMES = ("cs_q", "cs_k", "cs_v", "ssm", "state_tok")
# (requests, 1 + num_spec)
CELLS = [(1, 8), (2, 8), (4, 8), (2, 7), (4, 7), (2, 5), (2, 2)]


def _verify(wt, pools, proj, slots, num_spec) -> torch.Tensor:
    return k3_fp8_kda_verify(
        proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slots, pools["pending"],
        num_spec, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
    )  # fmt: skip


def _arm(value: int, wt, pools, proj, slots, num_spec, graph: bool):
    saved = F.TRTLLM_K3LL_KDA_VERIFY_HOIST
    F.TRTLLM_K3LL_KDA_VERIFY_HOIST = value
    try:
        if not graph:
            return _verify(wt, pools, proj, slots, num_spec).clone()
        before = kc.clone_pools(pools)
        _verify(wt, pools, proj, slots, num_spec)  # compiles outside the capture
        for name in POOL_NAMES:
            pools[name].copy_(before[name])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
            out = _verify(wt, pools, proj, slots, num_spec)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for name in POOL_NAMES:
            pools[name].copy_(before[name])
        g.replay()
        torch.cuda.synchronize()
        return out.clone()
    finally:
        F.TRTLLM_K3LL_KDA_VERIFY_HOIST = saved


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("num,steps", CELLS, ids=[f"{r}x{t}" for r, t in CELLS])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_hoist_bitwise(num, steps, state_dtype, graph) -> None:
    kc.load_ops()
    num_spec = steps - 1
    mgr = kc.build_manager(1, num_spec=num_spec, state_dtype=state_dtype)
    try:
        slots = kc.request_slots(mgr, num, first_id=500 + 10 * steps)
        p = kc.layer_pools(mgr, 0)
        with torch.inference_mode():
            wt = kc.make_weights(900 + steps)
            gen = torch.Generator(device="cuda").manual_seed(90 + 10 * num + steps)
            proj = torch.randn(num * steps, kc.PROJ, generator=gen, device="cuda").bfloat16()
            for pend in range(num_spec + 1):
                kc.fill_pools(p, 4000 + 10 * num + steps)
                p["state_tok"].copy_(
                    torch.randn(p["state_tok"].shape, generator=gen, device="cuda") * 0.1
                )
                # requests at different counts: pend, pend - 1, ... (wrapping), so one launch mixes them
                accepted = [(pend - i) % (num_spec + 1) for i in range(num)]
                p["pending"][slots.long()] = torch.tensor(
                    accepted, dtype=torch.int32, device="cuda"
                )
                start = kc.clone_pools(p)
                census._counts.clear()
                census.ENABLED = True
                try:
                    out_off = _arm(0, wt, p, proj, slots, num_spec, graph)
                    off_counts = dict(census.counts())
                    after_off = kc.snapshot(p)
                    for name in POOL_NAMES:
                        p[name].copy_(start[name])
                    census._counts.clear()
                    out_on = _arm(1, wt, p, proj, slots, num_spec, graph)
                    on_counts = dict(census.counts())
                finally:
                    census.ENABLED = False
                assert torch.equal(out_on, out_off), (pend, accepted)
                for name in POOL_NAMES:
                    assert torch.equal(p[name], after_off[name]), (name, pend, accepted)
                assert not any(k.startswith("dispatch:kda_verify_hoist") for k in off_counts), (
                    off_counts
                )
                mode = "capture" if graph else "eager"
                assert on_counts.get(f"dispatch:kda_verify_hoist:{mode}", 0) >= 1, on_counts
    finally:
        mgr.shutdown()

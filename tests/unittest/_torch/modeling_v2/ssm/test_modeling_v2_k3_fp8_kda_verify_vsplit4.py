# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for TRTLLM_K3LL_KDA_VERIFY_VSPLIT4 (``trtllm::k3_fp8_kda_verify`` with 4 CTAs of 512 threads per (head,
request) cluster instead of 8 of 256): the on arm bit for bit the off arm, outputs and every pool (state, records,
conv caches), at verify widths 3..8 with the requests of one launch at different counts of accepted drafts, the
[f_a | b] rows in the fused projection or separate, the prologue hoisted or not, on a real cache object with a BF16 and
an FP32 pool state; eager and a captured launch replayed. The value is the smallest request count that uses the 4-CTA
split: the census line is counted exactly at those launches.
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
CELLS = [(1, 8), (2, 8), (3, 8), (4, 8), (2, 7), (4, 7), (3, 6), (2, 5), (4, 4), (2, 3)]
BFA_COLS = kc.K + kc.H + 2  # a separate [f_a | b | pad] row
CENSUS = "dispatch:kda_verify_vsplit4"


def _verify(wt, pools, proj, slots, num_spec, bfa) -> torch.Tensor:
    return k3_fp8_kda_verify(
        proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slots, pools["pending"],
        num_spec, kc.LOWER_BOUND, kc.SCALE, kc.EPS, bfa=bfa,
    )  # fmt: skip


def _arm(value: int, wt, pools, proj, slots, num_spec, graph: bool, bfa):
    saved = F.TRTLLM_K3LL_KDA_VERIFY_VSPLIT4
    F.TRTLLM_K3LL_KDA_VERIFY_VSPLIT4 = value
    try:
        if not graph:
            return _verify(wt, pools, proj, slots, num_spec, bfa).clone()
        before = kc.clone_pools(pools)
        _verify(wt, pools, proj, slots, num_spec, bfa)  # compiles outside the capture
        for name in POOL_NAMES:
            pools[name].copy_(before[name])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
            out = _verify(wt, pools, proj, slots, num_spec, bfa)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for name in POOL_NAMES:
            pools[name].copy_(before[name])
        g.replay()
        torch.cuda.synchronize()
        return out.clone()
    finally:
        F.TRTLLM_K3LL_KDA_VERIFY_VSPLIT4 = saved


def test_default() -> None:
    assert F._FLAGS["TRTLLM_K3LL_KDA_VERIFY_VSPLIT4"][1] == 0
    assert F.accepted_values()["TRTLLM_K3LL_KDA_VERIFY_VSPLIT4"] == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("hoist", [1, 0], ids=["hoist", "no_hoist"])
@pytest.mark.parametrize("sep_bfa", [False, True], ids=["fused_bfa", "sep_bfa"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("num,steps", CELLS, ids=[f"{r}x{t}" for r, t in CELLS])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_vsplit4_bitwise(num, steps, state_dtype, graph, sep_bfa, hoist, monkeypatch) -> None:
    monkeypatch.setattr(F, "TRTLLM_K3LL_KDA_VERIFY_HOIST", hoist)
    kc.load_ops()
    num_spec = steps - 1
    mgr = kc.build_manager(1, num_spec=num_spec, state_dtype=state_dtype)
    try:
        slots = kc.request_slots(mgr, num, first_id=700 + 10 * steps)
        p = kc.layer_pools(mgr, 0)
        with torch.inference_mode():
            wt = kc.make_weights(1300 + steps)
            gen = torch.Generator(device="cuda").manual_seed(230 + 10 * num + steps)
            proj = torch.randn(num * steps, kc.PROJ, generator=gen, device="cuda").bfloat16()
            bfa = None
            if sep_bfa:
                bfa = torch.randn(num * steps, BFA_COLS, generator=gen, device="cuda").bfloat16()
            for pend in range(num_spec + 1):
                kc.fill_pools(p, 5300 + 10 * num + steps)
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
                    out_off = _arm(0, wt, p, proj, slots, num_spec, graph, bfa)
                    off_counts = dict(census.counts())
                    after_off = kc.snapshot(p)
                    for name in POOL_NAMES:
                        p[name].copy_(start[name])
                    census._counts.clear()
                    out_on = _arm(1, wt, p, proj, slots, num_spec, graph, bfa)
                    on_counts = dict(census.counts())
                finally:
                    census.ENABLED = False
                assert torch.equal(out_on, out_off), (pend, accepted)
                for name in POOL_NAMES:
                    assert torch.equal(p[name], after_off[name]), (name, pend, accepted)
                assert not any(k.startswith(CENSUS) for k in off_counts), off_counts
                mode = "capture" if graph else "eager"
                assert on_counts.get(f"{CENSUS}:{mode}", 0) >= 1, on_counts
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("least", [2, 3, 4])
def test_vsplit4_least_requests(least) -> None:
    """The value is the smallest request count of a launch that uses the 4-CTA split."""
    kc.load_ops()
    mgr = kc.build_manager(1, num_spec=7, state_dtype=torch.bfloat16)
    saved = F.TRTLLM_K3LL_KDA_VERIFY_VSPLIT4
    try:
        with torch.inference_mode():
            wt = kc.make_weights(1400)
            gen = torch.Generator(device="cuda").manual_seed(1401)
            F.TRTLLM_K3LL_KDA_VERIFY_VSPLIT4 = least
            all_slots = kc.request_slots(mgr, 4, first_id=1500)
            p = kc.layer_pools(mgr, 0)
            for num in (1, 2, 3, 4):
                slots = all_slots[:num]
                kc.fill_pools(p, 1402 + num)
                proj = torch.randn(num * 8, kc.PROJ, generator=gen, device="cuda").bfloat16()
                census._counts.clear()
                census.ENABLED = True
                try:
                    _verify(wt, p, proj, slots, 7, None)
                    counts = dict(census.counts())
                finally:
                    census.ENABLED = False
                used = counts.get(f"{CENSUS}:eager", 0) >= 1
                assert used == (num >= least), (least, num, counts)
    finally:
        F.TRTLLM_K3LL_KDA_VERIFY_VSPLIT4 = saved
        mgr.shutdown()

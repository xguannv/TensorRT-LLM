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
"""Kimi-K3 ULL4 MLA attention-side scheduling switch against today's path, one GPU.

``features.ull4_mla_plan_early``: the step's FP4 MLA metadata (generation lengths fused with K3's
packed split plan, the device page table) written ahead on a side stream
(``_fp4_mla_decode.prelaunch_generation_metadata``): the layer's decode then launches no metadata
kernel and gives the same cache bytes, outputs and statistics.

Helpers: ``test_k3_ull2_mla.py`` (FP4 chain, same directory).
"""

import math

import pytest
import test_k3_ull2_mla as u2
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)


def _plan(md):
    """K3's packed split plan workspace of the step."""
    return md.fp4_mla_state.workspaces[u2.qf.k3_decode()._PLAN_WS]


@u2.qf.needs_k3_decode
@pytest.mark.parametrize("n", [1, 2])
def test_step_metadata_prelaunch(n, k3_features):
    """The step metadata written ahead on a side stream: the chain's decode launches no metadata
    kernel, and the cache bytes, decode output and statistics are bitwise those of the chain whose
    decode writes the metadata itself; also captured in a CUDA graph."""
    dec = u2.qf.k3_decode()
    k3_features(ull2_mla_decode_pdl=True)
    chain = u2._Chain(n, 5000 + 1000 * n, seed=80 + n)
    md = chain.md
    state = md.fp4_mla_state
    try:
        # What the metadata resets at every step (Fp4MlaState.invalidate_generation_lengths).
        state.invalidate_generation_lengths()
        chain.run(early_trigger=True, plan_early=False)
        # Eager plans hold spare task slots the plan kernel leaves untouched: compare against a
        # reference written over the same fill.
        _plan(md).fill_(-1)
        state.invalidate_generation_lengths()
        chain.run(early_trigger=True, plan_early=False)
        ref = chain.result()
        ref_plan = _plan(md).clone()
        ref_lens = state.generation_kv_lens.clone()
        pools = [p.clone() for p in u2.qf._pools(chain.mgr)]
        assert dec._step(md).plan_key is not None

        def prelaunched_run():
            state.invalidate_generation_lengths()
            dec.begin_step(md)
            side = torch.cuda.Stream()
            fork, done = torch.cuda.Event(), torch.cuda.Event()
            fork.record()
            with torch.cuda.stream(side):
                fork.wait()
                launched = dec.prelaunch_generation_metadata(
                    md, num_heads=u2.H, use_helix=True, device=torch.device("cuda")
                )
                done.record()
            torch.cuda.current_stream().wait_event(done)
            before = dec._step(md).metadata_launches
            chain.run(early_trigger=True, new_step=False)
            return launched, dec._step(md).metadata_launches - before

        chain.out.fill_(torch.nan)
        _plan(md).fill_(-1)
        plan_ptr = _plan(md).data_ptr()
        launched, extra_launches = prelaunched_run()
        assert launched and extra_launches == 0
        # An index-less device must not reallocate the plan workspace.
        assert _plan(md).data_ptr() == plan_ptr
        u2._assert_bitwise(chain.result(), ref)
        assert torch.equal(_plan(md), ref_plan)
        assert torch.equal(state.generation_kv_lens, ref_lens)
        for got, want in zip(u2.qf._pools(chain.mgr), pools):
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        # Captured: the prelaunch, the join and the chain in one graph.
        prelaunched_run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prelaunched_run()
        for _ in range(2):
            chain.out.fill_(torch.nan)
            _plan(md).fill_(-1)
            graph.replay()
            u2._assert_bitwise(chain.result(), ref)
            assert torch.equal(_plan(md), ref_plan)
        assert math.isfinite(float(ref[0].float().abs().max()))
    finally:
        chain.mgr.shutdown()

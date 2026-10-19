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
"""CuTe DSL FP8 MLA decode with the Helix V up-projection fused into the balanced-split reduction.

``CuteDSLNVMlaDecodeBlackwellRunner.forward(..., v_weight=, v_out=)`` must give the same V rows
and softmax statistics, bit for bit, as the plain decode followed by the BMM the MLA module runs
(``trtllm::bmm_out`` of the BF16 O rows with the per-head V weight): batches 1, 2 and 4 at MTP
depths 6 and 7, M tiles cut into many partials (long KV) or covered by one cluster (short KV),
a request without KV, Helix per-token KV bounds, eagerly and in a CUDA graph. Past the fused
kernel's row limit the runner falls back to the BMM.

Requires 1 SM100/SM103 GPU and nvidia-cutlass-dsl.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_cute_dsl_mla_flat_fold as flat_fold  # noqa: E402
from test_cute_dsl_mla_flat_fold import _HAVE_DEPS, _is_sm100_or_sm103  # noqa: E402

if _HAVE_DEPS:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla import mla_decode_fp8_v_up

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_sm100_or_sm103(), reason="needs sm100/sm103"),
    pytest.mark.skipif(not _HAVE_DEPS, reason="needs nvidia-cutlass-dsl"),
]

H, D, PAGE, MAX_BATCH = 96, 512, 64, 64
_BALANCED = ((128, 128), (128, 256), 0, True)
_CASES = {
    "b1_long": ([10240], 7),
    "b2_long": ([10240, 6000], 7),
    "b4_mtp7": ([4096, 8192, 300, 5000], 8),
    "b2_short": ([640, 130], 7),
    "b3_no_kv": ([5000, 0, 900], 7),
}


@pytest.fixture(autouse=True)
def _partial_env(monkeypatch):
    # the formal GEN settings of the balanced split (BF16 partials, rescale threshold)
    monkeypatch.setenv("TRTLLM_MLA_PARTIAL_BF16", "1")
    monkeypatch.setenv("TRTLLM_MLA_RESCALE_THRESHOLD", "8")


def _run(x, S, w, fused, graph=False):
    """(o, stats, v) of one decode; ``fused`` passes the V weight to the runner, else the BMM of
    the MLA module follows the plain decode."""
    B = len(x["lens"])
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=H,
        seq_len_q=S,
        page_size=PAGE,
        max_batch_size=MAX_BATCH,
        emit_softmax_stats=True,
    )
    ws_size = runner.get_workspace_layout(H, S, D, MAX_BATCH, cutlass.Float32)[4]
    inputs = list(x["inputs"])
    o = torch.full((B, S, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    inputs[6] = o.permute(2, 3, 1, 0)
    inputs[7] = torch.full((ws_size,), 0xFF, dtype=torch.uint8, device="cuda")
    stats = torch.full((B * S, H, 2), float("nan"), dtype=torch.float32, device="cuda")
    inputs[8] = stats
    inputs[9] = x["kv_bounds"]
    v = torch.full((B * S, H, 128), float("nan"), dtype=torch.bfloat16, device="cuda")

    def step():
        if fused:
            runner.forward(inputs, tactic=_BALANCED, v_weight=w, v_out=v)
        else:
            runner.forward(inputs, tactic=_BALANCED)
            torch.ops.trtllm.bmm_out(
                o.view(B * S, H, D).transpose(0, 1), w.transpose(1, 2), v.transpose(0, 1)
            )

    step()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        v.fill_(float("nan"))
        stats.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
    return o, stats, v


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("case", list(_CASES))
def test_fused_v_up_matches_decode_and_bmm(case, graph, monkeypatch):
    monkeypatch.setattr(flat_fold, "_MAX_BATCH", MAX_BATCH)
    lens, S = _CASES[case]
    x = flat_fold._padded_inputs(lens, S, H, True, 11)
    w = (torch.randn(H, 128, D, device="cuda") * 0.05).to(torch.bfloat16)
    _, stats_ref, v_ref = _run(x, S, w, fused=False, graph=graph)
    _, stats, v = _run(x, S, w, fused=True, graph=graph)
    assert not torch.isnan(v_ref).any()
    assert torch.equal(v, v_ref)
    assert torch.equal(torch.isnan(stats), torch.isnan(stats_ref))
    assert torch.equal(torch.nan_to_num(stats), torch.nan_to_num(stats_ref))


def test_rows_past_the_fused_limit_take_the_bmm(monkeypatch):
    """35 rows per head (more than MAX_ROWS) run the plain decode and the BMM."""
    monkeypatch.setattr(flat_fold, "_MAX_BATCH", MAX_BATCH)
    assert mla_decode_fp8_v_up.MAX_ROWS < 35
    lens, S = [500] * 5, 7
    x = flat_fold._padded_inputs(lens, S, H, True, 3)
    w = (torch.randn(H, 128, D, device="cuda") * 0.05).to(torch.bfloat16)
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN, num_heads=H, seq_len_q=S, page_size=PAGE
    )
    v_out = torch.empty((len(lens) * S, H, 128), dtype=torch.bfloat16, device="cuda")
    assert runner._v_up_rows(len(lens), True, w, v_out) == 0
    _, stats_ref, v_ref = _run(x, S, w, fused=False)
    _, stats, v = _run(x, S, w, fused=True)
    assert torch.equal(v, v_ref)
    assert torch.equal(torch.nan_to_num(stats), torch.nan_to_num(stats_ref))

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
"""Kimi-K3 ULL3 MLA attention-side scheduling against the late-released chain, one GPU.

The epilogue release: K3's FP4 packed decode whose split reduction the fused Helix tail folds
(``DeferredPackedReduce(release_early=False)``) releases the tail when its producer warps finish
the last KV tile instead of at its completion. The tail, launched as its programmatic dependent,
stages ``W_UV`` and the plan during the decode's end and folds after ``griddepcontrol.wait``.
Checks: the tail output (fold, V up-projection, combine, MXFP8 o_proj operand) is bitwise that of
the chain whose decode releases nothing (the tail waits for its completion), eagerly and over CUDA
graph replays, at n1 / n2 with FP16 and FP32 split partials.

``features.ull3_mla_gate_at``: the per-batch placement policy of the side-stream output gate
parses as documented.

The FP4 cache and chain helpers are those of ``test_k3_ull2_mla.py`` / ``test_k3_mla_q_fused.py``
(same directory).
"""

import pytest
import test_k3_ull2_mla as u2
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

H, LORA, Q_LEN = u2.H, u2.LORA, u2.Q_LEN


class _Tail:
    """The fused tail on a local receive buffer (``P = 1``): fold, V up-projection, push into its
    own buffer, combine, MXFP8 operand of ``bf16(out * gate)``."""

    def __init__(self, num_tokens, seed):
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default.kernels import (  # noqa: E501
            k3_helix_tail as tail_mod,
        )

        self.mod = tail_mod
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        self.w_uv = (torch.randn((H, 128, LORA), device="cuda", generator=gen) * 0.05).to(
            torch.bfloat16
        )
        k = H * 128
        self.gate = torch.sigmoid(torch.randn((num_tokens, k), device="cuda", generator=gen)).to(
            torch.bfloat16
        )
        self.out = torch.empty((num_tokens, k), dtype=torch.bfloat16, device="cuda")
        self.xq = torch.empty((num_tokens, k), dtype=torch.float8_e4m3fn, device="cuda")
        self.xs = torch.empty((num_tokens, k // 32), dtype=torch.uint8, device="cuda")
        self.half_bytes = num_tokens * tail_mod.record_bytes(H, 128) + 4096
        self.buffer = torch.full((2 * self.half_bytes // 4,), -1, dtype=torch.int32, device="cuda")
        self.state = torch.zeros(8, dtype=torch.int64, device="cuda")

    def launch(self, rows, stats, fold):
        self.mod.k3_helix_tail(
            rows,
            stats,
            self.w_uv,
            self.out,
            fold=fold,
            zero_kv_mask=None,
            cp_rank=0,
            cp_size=1,
            ws_addr=self.buffer.data_ptr(),
            rank_stride=self.buffer.numel() * 4,
            half_bytes=self.half_bytes,
            state=self.state,
            max_ctas=torch.cuda.get_device_properties(0).multi_processor_count,
            use_pdl=True,
            mx_out=(self.gate, self.xq, self.xs),
            ull2=True,
        )

    def result(self):
        torch.cuda.synchronize()
        return self.out.clone(), self.xq.view(torch.uint8).clone(), self.xs.clone()


def _assert_same(got, ref):
    """Tail output (BF16), MXFP8 codes and exponents bit for bit."""
    assert torch.equal(got[0].view(torch.int16), ref[0].view(torch.int16))
    assert torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2])


def _layer(chain, tail):
    """Decode with the split reduction deferred (late release), then the fused tail on it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default.kernels.fp4_mla_packed_reduce import (  # noqa: E501
        DeferredPackedReduce,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default.mla import (  # noqa: E501
        fold_inputs_of,
    )

    with DeferredPackedReduce(release_early=False) as deferred:
        chain.run(early_trigger=True)
    fold = fold_inputs_of(deferred, chain.out.view(chain.T, -1), chain.stats)
    assert fold is not None, "the split reduction was not deferred"
    tail.launch(chain.out.view(chain.T * H, LORA), chain.stats.view(chain.T * H, 2), fold)


@u2.qf.needs_k3_decode
@pytest.mark.parametrize("f16_partials", [True, False], ids=["fp16", "fp32"])
@pytest.mark.parametrize("n", [1, 2])
def test_tail_trigger_matches_late_release(n, f16_partials, monkeypatch, k3_features):
    k3_features(
        ull2_mla_tail=True,
        ull2_mla_decode_pdl=True,
        fp4_mla_reduce_pdl=True,
        fp4_mla_f16_partials=f16_partials,
    )
    recorder = u2._record_decode(monkeypatch)
    module = recorder.module
    chain = u2._Chain(n, 6000 + 1500 * n, seed=70 + n)
    tail = _Tail(chain.T, seed=5 + n)
    try:
        # The reference: the decode releases nothing, the tail waits for its completion.
        recorder.override = {"launch_dependents": False}
        _layer(chain, tail)
        ref = tail.result()
        assert recorder.releases[-1] is False
        assert torch.isfinite(ref[0].float()).all() and ref[0].abs().max() > 0
        recorder.override = {}
        for _ in range(2):
            tail.out.fill_(torch.nan)
            _layer(chain, tail)
            _assert_same(tail.result(), ref)
            assert recorder.releases[-1] == module.LAUNCH_DEPENDENTS_EPILOGUE
        # CUDA graph: the programmatic edges decode -> tail as the model captures them.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _layer(chain, tail)
        torch.cuda.synchronize()
        for _ in range(3):
            tail.out.fill_(torch.nan)
            graph.replay()
            _assert_same(tail.result(), ref)
        assert bool((tail.buffer == -1).all()), "the receive buffer was not re-armed"
    finally:
        chain.mgr.shutdown()


def test_gate_place_policy():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default import (
        mla as k3mla,
    )

    assert k3mla._parse_ull3_gate_at("decode") == ("decode", 16, "decode")
    assert k3mla._parse_ull3_gate_at("exchange:16,start") == ("exchange", 16, "start")
    assert k3mla._parse_ull3_gate_at("exchange") == ("exchange", 16, "exchange")
    assert k3mla._parse_ull3_gate_at("exchange:8,start") == ("exchange", 8, "start")
    for bad in ("early", "exchange,late", "start:x", "query", "tail"):
        with pytest.raises(ValueError):
            k3mla._parse_ull3_gate_at(bad)

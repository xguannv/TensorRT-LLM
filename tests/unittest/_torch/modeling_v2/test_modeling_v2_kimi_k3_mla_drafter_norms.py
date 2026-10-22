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
"""`mla_drafter.K3MLADrafterNorms` on one GPU: its block decode with the residual adds and RMSNorms moved into the
all-reduces matches the stock MLA DSpark drafter's ``dflash_forward``, with a single-rank stand-in for the TP group's
collectives (the sum of one rank's partial is the partial). A block it does not take (no page table) runs the stock
forward."""

import importlib.util
import pathlib
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    mla_drafter,
)

_SEMANTICS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "speculative"
    / "hw_agnostic"
    / "test_kimi_k3_dspark_semantics.py"
)


def _semantics():
    name = "_k3_dspark_semantics_for_mla_drafter_norms"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _SEMANTICS)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class _OneRankComm:
    """``K3DecodeComm``'s fused all-reduce on a group of one: ``updated = residual + partial``, ``normed =
    norm(updated)``. The sandwich is never offered (no compiled forms)."""

    def __init__(self):
        self.calls = 0

    def takes_allreduce_norm(self, rows, hidden, one_shot_max_bytes=None):
        return rows > 0

    def takes_plain(self, *args, **kwargs):
        raise AssertionError("no k3_sandwich_plain form compiled, so none is asked for")

    def allreduce_norm(self, partial, residual, norm, one_shot_max_bytes=None):
        self.calls += 1
        updated = residual + partial
        return norm(updated), updated


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
@pytest.mark.parametrize("backend", ["TRTLLM", "CUTEDSL"])
def test_fused_norms_match_the_stock_block_decode(backend):
    sem = _semantics()
    if not sem._has_absorbed_mla_kernel():
        pytest.skip("no absorbed-MLA paged kernel on this GPU")
    if backend == "CUTEDSL":
        import tensorrt_llm._torch.models.modeling_dspark as md

        reason = md.cute_dsl_mla_decode_unavailability_reason()
        if reason is not None:
            pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")
    torch.manual_seed(0)
    drafter = sem._build_mla_drafter(sem._tiny_mla_weights(), dflash_attention_backend=backend)
    g = torch.Generator().manual_seed(42)
    requests = [
        (
            torch.randn(length, sem.MLA_HIDDEN * sem.NUM_CAPTURE, generator=g) * 0.5,
            torch.randn(sem.MLA_BLOCK, sem.MLA_HIDDEN, generator=g) * 0.5,
        )
        for length in (sem.MLA_CTX_LEN, sem.MLA_SHORT_CTX_LEN)
    ]
    stock = sem._run_mla_block_decode(drafter, requests, paged=True)

    comm = _OneRankComm()
    owner = mla_drafter.K3MLADrafterNorms(drafter, comm, None, drafter.dflash_forward, frozenset())
    drafter.dflash_forward = mla_drafter._Bound(owner)
    fused = sem._run_mla_block_decode(drafter, requests, paged=True)
    # Two fused all-reduces per layer: after the output projection and after the MLP.
    assert comm.calls == 2 * sem.MLA_LAYERS
    assert fused.shape == stock.shape
    # The stock fused add + RMSNorm and a bf16 add then RMSNorm can round the residual differently by one ULP.
    diff = (fused - stock).abs().max().item()
    assert diff < 0.03, f"fused block decode differs from the stock one by {diff}"

    # No page table: the stock forward, so no fused all-reduce.
    comm.calls = 0
    eager = sem._run_mla_block_decode(drafter, requests, paged=False)
    assert comm.calls == 0
    assert eager.shape == stock.shape


def test_layers_take_needs_row_parallel_projections():
    """A drafter whose output projection does not all-reduce (TP1) keeps its stock forward."""
    from types import SimpleNamespace

    norm = SimpleNamespace(
        weight=torch.ones(8, dtype=torch.bfloat16),
        variance_epsilon=1e-6,
        use_gemma=False,
        is_nvfp4=False,
        return_hp_output=False,
    )
    o_proj = SimpleNamespace(
        tp_mode=SimpleNamespace(name="ROW"), reduce_output=False, all_reduce=None
    )
    layer = SimpleNamespace(
        input_layernorm=norm,
        post_attention_layernorm=norm,
        self_attn=SimpleNamespace(o_proj=o_proj, hidden_size=8),
        mlp=SimpleNamespace(down_proj=o_proj),
    )
    drafter = SimpleNamespace(model=SimpleNamespace(norm=norm, layers=[layer]))
    assert not mla_drafter.K3MLADrafterNorms.layers_take(drafter)

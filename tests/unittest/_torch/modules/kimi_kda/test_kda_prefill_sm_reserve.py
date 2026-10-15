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
"""KDA prefill persistent grids that keep TRTLLM_CUTEDSL_SM_RESERVE SMs free
(TRTLLM_K3_KDA_PREFILL_SM_RESERVE=1) give bit-identical outputs and states."""

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (
    KDAKernelDispatch,
    is_kda_optimized_supported,
)

H, D = 96, 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_kda_optimized_supported(),
    reason="needs the optimized KDA prefill (SM100/SM103)",
)


def _run(seqlens, monkeypatch, reserve):
    if reserve:
        monkeypatch.setenv("TRTLLM_K3_KDA_PREFILL_SM_RESERVE", "1")
        monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", str(reserve))
    else:
        monkeypatch.delenv("TRTLLM_K3_KDA_PREFILL_SM_RESERVE", raising=False)
        monkeypatch.delenv("TRTLLM_CUTEDSL_SM_RESERVE", raising=False)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1)
    tokens = sum(seqlens)
    q = torch.nn.functional.normalize(
        torch.randn(1, tokens, H, D, device=dev, generator=gen), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, H, D, device=dev, generator=gen), dim=-1
    ).bfloat16()
    v = torch.randn(1, tokens, H, D, device=dev, generator=gen).bfloat16()
    g = torch.randn(1, tokens, H, D, device=dev, generator=gen).bfloat16()
    beta = torch.randn(1, tokens, H, device=dev, generator=gen).float()
    a_log = torch.randn(H, device=dev, generator=gen).float()
    dt_bias = torch.randn(H * D, device=dev, generator=gen).float()
    cu = torch.tensor([0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    n = len(seqlens)
    pool = torch.randn(n, H, D, D, dtype=torch.float32, device=dev, generator=gen) * 0.1
    idx = torch.arange(n, dtype=torch.int32, device=dev)
    out, _ = KDAKernelDispatch().prefill_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=D**-0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=cu,
        state_pool=pool,
        state_indices=idx,
    )
    torch.cuda.synchronize()
    return out.clone(), pool.clone()


@pytest.mark.parametrize("seqlens", [[16384], [4096, 4096, 4096, 4096], [3000, 64, 1, 9000]])
def test_prefill_grid_reserve_is_bit_exact(seqlens, monkeypatch):
    out0, pool0 = _run(seqlens, monkeypatch, 0)
    out1, pool1 = _run(seqlens, monkeypatch, 30)
    assert torch.equal(out0, out1)
    assert torch.equal(pool0, pool1)

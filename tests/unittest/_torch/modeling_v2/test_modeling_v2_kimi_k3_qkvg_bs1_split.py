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
"""``TRTLLM_K3LL_QKVG_BS1_SPLIT`` (``kimi_k3_nvfp4/flags.py``): ``fp8w_lowm.gemm`` on the fused KDA
[q | k | v | g | f_a | b] weight (3216 x 7168, a partial last row tile) at 8 or fewer rows runs ``k3_fp8w_lowm_gemm`` at
the forced split-K (3 or 4) instead of the default tactic's split 5. One GPU (SM100 / SM103).

* -1 (the default) resolves to 4 on SM103 and to 0 elsewhere (``qkvg_bs1_split``).
* 0: ``gemm`` is bit for bit ``trtllm::k3_fp8w_lowm_gemm`` at every row count and shape, and counts
  nothing.
* 3 / 4, 1-8 rows of the fused shape: that split (census ``dispatch:qkvg_bs1_split<S>_rows<M>``), within 1e-4
  (relative norm) of the default split (another FP32 summation order), bit for bit ``run_k3_fp8w_lowm_gemm`` at
  `qkvg_bs1_tactic`.
* 3 / 4, more rows: bit for bit the off arm, counted as ``qkvg_bs1_split_skip_rows<M>``; another shape: the off arm,
  nothing counted. Any other value raises.
* Load-time compile (``compile_qkvg_bs1_splits``, as the KDA layer calls it whatever the switch says): both splits;
  then a CUDA-graph capture at either value replays bit for bit the eager call (the warm-up coverage of the switch).
  Not compiled, the capture keeps the default split (bit for bit the off arm) and counts
  ``fallback:qkvg_bs1_split_not_compiled``.
"""

import functools

import pytest
import torch

K = 7168
N_FUSED = 3216
SPLITS = (3, 4)


def _modules():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        flags,
        fp8w_lowm,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

    return fp8w_lowm, flags, lowm


def test_default_is_auto():
    fp8w_lowm, flags, _ = _modules()
    assert flags._FLAGS["TRTLLM_K3LL_QKVG_BS1_SPLIT"][1] == -1
    assert fp8w_lowm.QKVG_BS1_SPLITS == SPLITS
    assert fp8w_lowm.QKVG_BS1_SPLIT_CAPABILITIES == ((10, 3),)
    assert fp8w_lowm.QKVG_BS1_DEFAULT_SPLIT == 4


@pytest.mark.parametrize(
    "value,capability,split",
    [
        (-1, (10, 3), 4),
        (-1, (10, 0), 0),
        (-1, (10, 7), 0),
        (0, (10, 3), 0),
        (3, (10, 3), 3),
        (4, (10, 7), 4),
    ],
)
def test_split_value(monkeypatch, value, capability, split):
    """The switch's resolution, no GPU needed (the device capability is patched)."""
    fp8w_lowm, flags, _ = _modules()
    monkeypatch.setattr(flags, "TRTLLM_K3LL_QKVG_BS1_SPLIT", value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    fp8w_lowm._qkvg_bs1_split_by_default.cache_clear()
    try:
        assert fp8w_lowm.qkvg_bs1_split(torch.device("cuda", 0)) == split
    finally:
        fp8w_lowm._qkvg_bs1_split_by_default.cache_clear()


gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the low-M FP8 GEMM runs on SM100 / SM103",
)


@functools.cache
def _weight(n):
    _, _, lowm = _modules()
    g = torch.Generator(device="cuda").manual_seed(n)
    n_pad = -(-n // 128) * 128
    w = torch.randn((n_pad, K), device="cuda", generator=g) * 0.02
    blocks = w.view(n_pad // 128, 128, K // 128, 128)
    scale = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0))
    )
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n_pad, K)
    return lowm.prepare_k3_fp8w_lowm_weight(w_fp8[:n].contiguous(), scale.float())


def _x(m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(1000 + 17 * m + seed)
    return (torch.randn((m, K), device="cuda", generator=g) * 3).to(torch.bfloat16)


@pytest.fixture
def census(monkeypatch):
    fp8w_lowm, _, _ = _modules()
    keys = []
    monkeypatch.setattr(fp8w_lowm, "_census_count", keys.append)
    return keys


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _forced(x, w, s, split):
    fp8w_lowm, _, lowm = _modules()
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    lowm.run_k3_fp8w_lowm_gemm(x, w, s, out, tactic=fp8w_lowm.qkvg_bs1_tactic(w.shape[0], K, split))
    return out


@gpu
@pytest.mark.parametrize("split", SPLITS)
def test_tactic(split):
    # The tactic's occupancy query needs a CUDA context (none yet in a fresh process).
    torch.zeros(1, device="cuda")
    fp8w_lowm, _, lowm = _modules()
    t = fp8w_lowm.qkvg_bs1_tactic(N_FUSED, K, split)
    assert t.mma_n == 8 and t.split_k == split
    lowm.validate_tactic(N_FUSED, K, t)
    assert lowm.default_tactic(8, N_FUSED, K).split_k == 5


@gpu
def test_other_value_raises(monkeypatch):
    fp8w_lowm, flags, _ = _modules()
    w, s = _weight(N_FUSED)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_QKVG_BS1_SPLIT", 5)
    with pytest.raises(ValueError):
        fp8w_lowm.gemm(_x(8), w, s)


@gpu
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("m", [1, 7, 8, 9, 16, 32])
@pytest.mark.parametrize("n", [N_FUSED, 3072])
def test_switch_arms(monkeypatch, census, split, m, n):
    fp8w_lowm, flags, _ = _modules()
    w, s = _weight(n)
    x = _x(m)
    stock = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, w, s)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_QKVG_BS1_SPLIT", 0)
    off = fp8w_lowm.gemm(x, w, s)
    assert torch.equal(off.view(torch.int16), stock.view(torch.int16))
    assert census == []
    monkeypatch.setattr(flags, "TRTLLM_K3LL_QKVG_BS1_SPLIT", split)
    on = fp8w_lowm.gemm(x, w, s)
    torch.cuda.synchronize()
    if n == N_FUSED and m <= 8:
        assert census == [f"dispatch:qkvg_bs1_split{split}_rows{m}"]
        assert torch.equal(on.view(torch.int16), _forced(x, w, s, split).view(torch.int16))
        assert _rel(on, off) < 1e-4
    else:
        assert census == ([f"qkvg_bs1_split_skip_rows{m}"] if n == N_FUSED else [])
        assert torch.equal(on.view(torch.int16), off.view(torch.int16))


@gpu
@pytest.mark.parametrize("compiled", [True, False])
def test_capture(monkeypatch, census, compiled):
    fp8w_lowm, flags, _ = _modules()
    w, s = _weight(N_FUSED)
    x = _x(8)
    monkeypatch.setattr(fp8w_lowm, "_QKVG_BS1_COMPILED", set())
    if compiled:
        assert fp8w_lowm.compile_qkvg_bs1_splits(w)
        index = torch.cuda.current_device()
        assert {(index, N_FUSED, K, split) for split in SPLITS} <= fp8w_lowm._QKVG_BS1_COMPILED
    assert not fp8w_lowm.compile_qkvg_bs1_splits(_weight(3072)[0])
    torch.ops.trtllm.k3_fp8w_lowm_gemm(
        x, w, s
    )  # the default split compiles eagerly, as the engine's warm-up does
    torch.cuda.synchronize()
    for split in SPLITS:
        census.clear()
        monkeypatch.setattr(flags, "TRTLLM_K3LL_QKVG_BS1_SPLIT", split)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fp8w_lowm.gemm(x, w, s)
        for seed in (1, 2):
            x.copy_(_x(8, seed))
            g.replay()
            torch.cuda.synchronize()
            ref = (
                _forced(x, w, s, split) if compiled else torch.ops.trtllm.k3_fp8w_lowm_gemm(x, w, s)
            )
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
        if compiled:
            assert census == [f"dispatch:qkvg_bs1_split{split}_rows8"]
        else:
            assert census == ["fallback:qkvg_bs1_split_not_compiled"]

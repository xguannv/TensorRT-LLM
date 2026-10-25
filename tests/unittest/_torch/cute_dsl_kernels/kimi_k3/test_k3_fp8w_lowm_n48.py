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
"""``TRTLLM_K3LL_FP8W_LOWM_N48``: the 64-token tile tactics of ``k3_fp8w_lowm_gemm`` (``_TUNED_N64``).

1 on, 0 off, -1 (default) on SM103 only. On, only the 64-token tile (33-64 token steps) of the table's shapes changes
tactic, and only its ring depths and quantization grouping (the split-K is kept), so every output is bitwise that of the
switch off.
"""

import pytest
import torch

GPU = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
needs_gpu = pytest.mark.skipif(not GPU, reason="needs an SM100/SM103 GPU")


def _mod():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm


def _clear(mod):
    mod.default_tactic.cache_clear()
    mod.gated_tactic.cache_clear()
    mod._gated_tactic_fits.cache_clear()
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8

    k3_fp8w_lowm_mxfp8.default_mx_tactic.cache_clear()


def _tactic(monkeypatch, value, m, n, k):
    mod = _mod()
    monkeypatch.setenv(mod.FP8W_LOWM_N48_ENV, str(value))
    _clear(mod)
    try:
        return mod.default_tactic(m, n, k)
    finally:
        _clear(mod)


@pytest.mark.parametrize(
    "value,capability,on",
    [
        ("1", (10, 3), True),
        ("1", (10, 0), True),
        ("0", (10, 3), False),
        ("-1", (10, 3), True),
        ("-1", (10, 0), False),
        ("-1", (10, 7), False),
        ("", (10, 3), True),
    ],
)
def test_switch_value(monkeypatch, value, capability, on):
    mod = _mod()
    monkeypatch.setenv(mod.FP8W_LOWM_N48_ENV, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    assert mod._n48_on() is on


def test_default_and_flag_doc():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    mod = _mod()
    assert flags._FLAGS["TRTLLM_K3LL_FP8W_LOWM_N48"][1] == -1
    assert mod.FP8W_LOWM_N48_CAPABILITIES == ((10, 3),)
    assert mod._TUNED_N64, "the table is empty"
    for (n, k), entry in mod._TUNED_N64.items():
        assert n % mod.MMA_M == 0 and k % mod.CTA_K == 0 and len(entry) == 5


@needs_gpu
@pytest.mark.parametrize("m", [1, 7, 8, 9, 16, 17, 31, 32])
def test_tactics_unchanged_up_to_32_tokens(monkeypatch, m):
    mod = _mod()
    for n, k in mod._TUNED_N64:
        assert _tactic(monkeypatch, 1, m, n, k) == _tactic(monkeypatch, 0, m, n, k)


@needs_gpu
@pytest.mark.parametrize("m", [33, 40, 48, 64])
def test_64_token_tile_takes_the_table(monkeypatch, m):
    mod = _mod()
    for (n, k), (split, x_stages, b_stages, quant_warps, quant_groups) in mod._TUNED_N64.items():
        t = _tactic(monkeypatch, 1, m, n, k)
        assert t.split_k == _tactic(monkeypatch, 0, m, n, k).split_k
        assert (t.mma_n, t.split_k, t.x_stages, t.b_stages, t.quant_warps, t.quant_groups) == (
            64,
            split,
            x_stages,
            b_stages,
            quant_warps,
            quant_groups,
        )
        mod.validate_tactic(n, k, t)


def _weight(n, k, seed):
    tiles = -(-n // 128)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((tiles * 128, k), device="cuda", generator=gen) * 0.02
    blocks = w.view(tiles, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(tiles * 128, k)
    return w_fp8[:n].contiguous(), scale.float()


def _run(monkeypatch, value, x, w, packed):
    mod = _mod()
    monkeypatch.setenv(mod.FP8W_LOWM_N48_ENV, str(value))
    _clear(mod)
    try:
        out = torch.empty((x.shape[0], w.shape[0]), device="cuda", dtype=torch.bfloat16)
        mod.run_k3_fp8w_lowm_gemm(x, w, packed, out)
        torch.cuda.synchronize()
        return out
    finally:
        _clear(mod)


@needs_gpu
@pytest.mark.parametrize(
    "nk", [(3216, 7168), (3072, 7168), (2112, 7168), (7168, 768), (18432, 1536), (768, 7168)]
)
def test_bitwise_and_accurate(monkeypatch, nk):
    mod = _mod()
    n, k = nk
    w_fp8, scale = _weight(n, k, seed=n + k)
    w, packed = mod.prepare_k3_fp8w_lowm_weight(w_fp8, scale)
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    for m in (1, 8, 16, 32, 40, 48):
        gen = torch.Generator(device="cuda").manual_seed(m)
        x = (torch.randn((m, k), device="cuda", generator=gen) * 3).to(torch.bfloat16)
        on, off = _run(monkeypatch, 1, x, w, packed), _run(monkeypatch, 0, x, w, packed)
        assert torch.equal(on, off), f"m={m}: the switch changed the output bits"
        ref = x.float() @ (w.float() * expanded).t()
        for out in (on, off):
            err = ((out.float() - ref).norm() / ref.norm()).item()
            assert err < 5e-2, f"m={m}: rel err {err:.3e}"

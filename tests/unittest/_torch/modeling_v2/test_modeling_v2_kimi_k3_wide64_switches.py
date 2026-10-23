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
"""The wide-step token limits of the Kimi K3 NVFP4 decode path (``kimi_k3_nvfp4/flags.py``) up to a 64-token step
(8 requests of 8 tokens), each arm against the other and the off arm against the chain it keeps, at the TP16 per-rank
shapes, one GPU. The collective parts (the residual updates inside ``KimiLinearDecoderLayer.forward``, the MoE wide
head) are in ``comm/_kimi_k3_lowm_switches_op_matrix.py``.

Checks:
  * Launch values (``test_launch_values``): with none of the switches in the environment, each holds its default,
    which is the behaviour before the switch: WIDE_ATTN_RES_MAX_TOKENS 32, WIDE_MOE_HEAD_ROWS 32, WIDE_MOE_TAIL_ROWS
    32, ROUTE_QUANT_FP32 0; ``_attn_res_max_tokens`` is the switch on a wide step and one token tile otherwise; the
    ``moe_head`` / ``moe_tail`` sites take the switches' row limits (``_row_limit``), no other site has one;
    ``K3DecodeGemvs.create`` warms those two sites at every wide tile up to 64 rows (``_warm_rows``; before the
    switches up to 32: a compile more at load, nothing else), the other sites at the rows it warmed before.
  * TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS (``test_wide_attn_res_max_tokens``): ``_apply_attn_res_add_and_rmsnorm`` and
    ``_apply_attn_res_and_rmsnorm`` at 16, 32, 48 and 64 rows, 1, 5 and 11 snapshots, with the limit at 32 (off) and
    64 (on). Off arm above 32 rows bit for bit the chain (the residual add, ``_apply_attn_res`` and the RMSNorm
    module). Up to 32 rows both arms run the same fused kernel: bit for bit. Above 32: ``updated`` bit for bit (the
    addends are exact payloads); ``normed`` NOT bitwise: the fused kernel normalizes the fp32 mixture, the chain the
    mixture rounded to bf16; within ``ATTN_RES_FUSED_TOL`` of its largest magnitude (max |err| / max |ref|), the
    measured error and ulp distance printed (``OPCHECK``).
  * TRTLLM_K3LL_WIDE_MOE_HEAD_ROWS / TRTLLM_K3LL_WIDE_MOE_TAIL_ROWS (``test_wide_moe_gemv_rows``): ``K3DecodeGemvs``
    on the ``moe_head`` (fp32 output) and ``moe_tail`` sites, one state (built with every tile up to 64 rows warmed),
    the switch (read at forward time) at 32 (off) and 64 (on). Off arm:
    ``project`` returns None above 32 rows, so the caller's stock GEMM runs as before. Up to 32 rows both arms the
    same bits. On arm above 32: ``k3_ctm_gemv_wide``, NOT bitwise the stock GEMM (another summation order): within
    ``GEMV_TOL`` of the fp32 reference and of the stock GEMM (``decode_moe._gemv`` without the decode state); a
    captured CUDA graph replayed with rewritten inputs bit for bit the eager call.
"""

import dataclasses
import os
import types

import pytest
import torch

H = 7168
ATTN_RES_ROWS = (16, 32, 48, 64)
SNAPSHOT_COUNTS = (1, 5, 11)
GEMV_ROWS = (9, 16, 32, 33, 48, 64)
LIMIT_OFF, LIMIT_ON = 32, 64
RES_EPS = 1e-5
# Fused against the chain above 32 rows: max |err| / max |ref| of normed (about one bf16 ulp of the largest values).
ATTN_RES_FUSED_TOL = 2e-2
# k3_ctm_gemv_wide against the fp32 reference / the stock GEMM: max |err| / max |ref|; the bf16 output (tail) rounds
# once, the fp32 output (head) only sums in another order.
GEMV_TOL = {"moe_head": 1e-4, "moe_tail": 1e-2}
SWITCHES = {
    "TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS": 32,
    "TRTLLM_K3LL_WIDE_MOE_HEAD_ROWS": 32,
    "TRTLLM_K3LL_WIDE_MOE_TAIL_ROWS": 32,
    "TRTLLM_K3LL_ROUTE_QUANT_FP32": 0,
}


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _mods():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_gemv,
        decode_moe,
        flags,
        implementation,
    )

    return types.SimpleNamespace(gemv=decode_gemv, moe=decode_moe, flags=flags, impl=implementation)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return (
        (got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-30)
    ).item()


def _exact_bf16(gen: torch.Generator, shape, lo: int, hi: int, scale: float) -> torch.Tensor:
    """bf16 integers in [lo, hi) times ``scale`` (a power of two): their sums are exact in bf16."""
    ints = torch.randint(lo, hi, tuple(shape), generator=gen, device="cuda")
    return (ints.float() * scale).bfloat16()


def test_launch_values():
    m = _mods()
    for name, default in SWITCHES.items():
        if os.environ.get(name):
            continue
        assert getattr(m.flags, name) == default, (name, getattr(m.flags, name))
        assert m.flags.flags()[name][1] == default
    wide, small = m.impl.DecodeStep(64, 8, 8), m.impl.DecodeStep(8, 1, 8)
    assert wide.wide and not small.wide
    assert m.impl._attn_res_max_tokens(wide) == m.flags.TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS
    assert m.impl._attn_res_max_tokens(small) == m.impl.DECODE_MAX_TOKENS
    sites, limit, warm = m.gemv.SITES, m.gemv._row_limit, m.gemv._warm_rows
    head, tail = sites["moe_head"], sites["moe_tail"]
    assert (head.rows_flag, tail.rows_flag) == (
        "TRTLLM_K3LL_WIDE_MOE_HEAD_ROWS",
        "TRTLLM_K3LL_WIDE_MOE_TAIL_ROWS",
    )
    assert limit(head) == min(64, m.flags.TRTLLM_K3LL_WIDE_MOE_HEAD_ROWS)
    assert limit(tail) == min(64, m.flags.TRTLLM_K3LL_WIDE_MOE_TAIL_ROWS)
    # Every wide tile up to 64 rows is warmed at load, so the switches can move after it.
    assert warm(head) == warm(tail) == (1, 16, 32, 64)
    assert warm(dataclasses.replace(tail, wide_rows=32)) == (1, 16, 32)
    assert warm(dataclasses.replace(tail, wide_rows=48)) == (1, 16, 32, 48)
    assert warm(sites["moe_shared_gate_up"]) == (1, 16, 32, 64)
    assert warm(sites["moe_up"]) == (1,)
    for name, spec in sites.items():
        if name not in ("moe_head", "moe_tail"):
            assert not spec.rows_flag and limit(spec) == spec.wide_rows, name


def _residual_modules(m):
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    g = torch.Generator(device="cuda").manual_seed(20261023)
    with torch.device("cuda"):
        proj = torch.nn.Linear(H, 1, bias=False, dtype=torch.bfloat16)
        norm = m.impl.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16)
        out_norm = RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16)
    with torch.no_grad():
        proj.weight.copy_(
            (0.05 * torch.randn(proj.weight.shape, generator=g, device="cuda")).bfloat16()
        )
        for module in (norm, out_norm):
            module.weight.copy_(
                (
                    1.0 + 0.1 * torch.randn(module.weight.shape, generator=g, device="cuda")
                ).bfloat16()
            )
    return proj, norm, out_norm


@pytest.mark.skipif(
    not _sm100(), reason="needs an SM100-family GPU (the K3 decode kernels' target)"
)
@pytest.mark.parametrize("snapshots", SNAPSHOT_COUNTS)
@pytest.mark.parametrize("rows", ATTN_RES_ROWS)
def test_wide_attn_res_max_tokens(rows, snapshots):
    m = _mods()
    proj, norm, out_norm = _residual_modules(m)
    g = torch.Generator(device="cuda").manual_seed(1000 * rows + snapshots)
    prefix = _exact_bf16(g, (rows, H), -64, 65, 1 / 16)
    addend = _exact_bf16(g, (rows, H), -64, 65, 1 / 16)
    bank = torch.randn(snapshots, rows, H, generator=g, device="cuda").bfloat16()
    add_fn, mix_fn = m.impl._apply_attn_res_add_and_rmsnorm, m.impl._apply_attn_res_and_rmsnorm
    with torch.no_grad():
        on = add_fn(prefix, addend, bank, proj, norm, out_norm, LIMIT_ON)
        off = add_fn(prefix, addend, bank, proj, norm, out_norm, LIMIT_OFF)
        on_mix = mix_fn(prefix, bank, proj, norm, out_norm, LIMIT_ON)
        off_mix = mix_fn(prefix, bank, proj, norm, out_norm, LIMIT_OFF)
        updated = prefix + addend
        chain = out_norm(m.impl._apply_attn_res(updated, bank, proj, norm))
        chain_mix = out_norm(m.impl._apply_attn_res(prefix, bank, proj, norm))
        exact = (prefix.float() + addend.float()).bfloat16()
    assert _same(off[0], exact) and _same(on[0], exact), "updated is not the exact sum"
    if rows <= LIMIT_OFF:
        assert _same(on[1], off[1]) and _same(on_mix, off_mix), "the same fused kernel differs"
        return
    assert _same(off[1], chain), "off arm is not the add -> attn_res -> RMSNorm chain"
    assert _same(off_mix, chain_mix), "off arm is not the attn_res -> RMSNorm chain"
    for name, got, want in (("add", on[1], off[1]), ("mix", on_mix, off_mix)):
        err = _rel_err(got, want)
        ulps = (_bits(got).int() - _bits(want).int()).abs().max().item()
        differ = int((_bits(got) != _bits(want)).sum())
        print(
            f"OPCHECK op=wide_attn_res_max_tokens kind={name} M={rows} S={snapshots} fused_vs_chain "
            f"max_rel_err={err:.3e} max_ulp={ulps} differ={differ}/{got.numel()}"
        )
        assert torch.isfinite(got.float()).all()
        assert err <= ATTN_RES_FUSED_TOL, f"{name}: fused vs chain rel err {err:.3e}"


def _project(m, state, site, limit, x, weight):
    """``state.project`` with the site's switch (read at forward time) at ``limit``."""
    name = m.gemv.SITES[site].rows_flag
    old = getattr(m.flags, name)
    setattr(m.flags, name, limit)
    try:
        return state.project(site, x, weight)
    finally:
        setattr(m.flags, name, old)


@pytest.mark.skipif(not _sm100(), reason="needs an SM100-family GPU (k3_ctm_gemv_wide)")
@pytest.mark.parametrize("site", ["moe_head", "moe_tail"])
def test_wide_moe_gemv_rows(site):
    m = _mods()
    spec = m.gemv.SITES[site]
    out_fp32 = spec.out_fp32
    g = torch.Generator(device="cuda").manual_seed(20261024)
    weight = (0.02 * torch.randn(spec.n, spec.k, generator=g, device="cuda")).bfloat16()
    state = m.gemv.K3DecodeGemvs.create(sites=(site,), device=torch.device("cuda"))
    for rows in GEMV_ROWS:
        x = (0.5 * torch.randn(rows, spec.k, generator=g, device="cuda")).bfloat16()
        off = _project(m, state, site, LIMIT_OFF, x, weight)
        on = _project(m, state, site, LIMIT_ON, x, weight)
        assert on is not None, (
            f"{site}: the wide kernel does not take {rows} rows at the 64-row limit"
        )
        assert on.dtype == (torch.float32 if out_fp32 else torch.bfloat16)
        if rows <= LIMIT_OFF:
            assert off is not None and _same(on, off), f"{site} {rows} rows: the arms differ"
            continue
        assert off is None, f"{site} {rows} rows: the 32-row limit took the call"
        stock = m.moe._gemv(None, site, x, weight, out_fp32=out_fp32)
        ref = x.float() @ weight.float().t()
        err_ref, err_stock = _rel_err(on, ref), _rel_err(on, stock)
        print(
            f"OPCHECK op=wide_moe_gemv_rows site={site} M={rows} vs_fp32={err_ref:.3e} "
            f"vs_stock={err_stock:.3e} stock_vs_fp32={_rel_err(stock, ref):.3e}"
        )
        assert err_ref <= GEMV_TOL[site] and err_stock <= GEMV_TOL[site], (err_ref, err_stock)
    # Under capture a fresh state takes a 64-row call: create warmed the 64-row tile whatever the switch's value.
    fresh = m.gemv.K3DecodeGemvs.create(sites=(site,), device=torch.device("cuda"))
    x = (0.5 * torch.randn(LIMIT_ON, spec.k, generator=g, device="cuda")).bfloat16()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _project(m, fresh, site, LIMIT_ON, x, weight)
        refused = _project(m, fresh, site, LIMIT_OFF, x, weight)
    assert captured is not None, f"{site}: the 64-row call was refused under capture"
    assert refused is None, f"{site}: the 32-row limit took a 64-row call under capture"
    for rep in range(2):
        x.copy_((0.5 * torch.randn(x.shape, generator=g, device="cuda")).bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        eager = _project(m, fresh, site, LIMIT_ON, x, weight)
        assert _same(captured, eager), f"{site}: replay {rep} differs from the eager call"

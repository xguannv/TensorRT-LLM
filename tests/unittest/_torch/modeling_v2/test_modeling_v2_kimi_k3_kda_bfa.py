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
"""Kimi K3 NVFP4 staircase, TRTLLM_K3LL_KDA_BFA_GEMV: the KDA layer's bf16 [f_a | b] projection.

* ``kda_bfa`` decode GEMV site (``K3DecodeGemvs.project`` on ``k3_ctm_gemv_wide``) against ``F.linear`` and an fp32
  reference, at the per-rank TP16 shape (136 x 7168) and 1 / 8 / 16 / 32 / 64 rows.
* ``kda_bfa_rt`` (TRTLLM_K3LL_KDA_BFA_ROW_TILES: the same GEMV over 34-row weight tiles) bit-identical to ``kda_bfa``.
* TRTLLM_K3LL_KDA_BFA_FP8: ``fp8w_lowm.append_rows`` (the [f_a | b] rows quantized to FP8 128x128 blocks below the
  QKVG weight) and one ``k3_fp8w_lowm_gemm`` over it: the QKVG columns against the QKVG-only call, the [f_a | b]
  columns against the bf16 product (FP8 error), the padding columns zero.
* ``K3DecodeKDA._k3_projections`` under every TRTLLM_K3LL_KDA_BFA_SIDE_FIRST / TRTLLM_K3LL_KDA_BFA_OVERLAP setting: the
  same bits; with OVERLAP the [f_a | b] projection on the projection stream, enqueued before QKVG with SIDE_FIRST and
  after it without; without OVERLAP both on the calling stream, QKVG first.
* ``trtllm::k3_fp8_kda_verify`` with the [f_a | b] rows passed beside the [q | k | v | g] rows (``bfa=``) against the
  same call on their concatenation: outputs, the pool state, the drafts' records and the conv caches bit for bit,
  over rounds with changing pending counts, fp32 and bf16 committed states.
"""

import types

import pytest
import torch
import torch.nn.functional as F

H = 6
K = V = 128
W = 4
HK = H * K
BFA = K + H + 2  # [f_a | b | pad]
POOL = 9
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = K**-0.5


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100-family (tcgen05, TMA, clusters)")


def _verify_op():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.verify import (
        k3_fp8_kda_verify,
    )

    return k3_fp8_kda_verify


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
def test_kda_bfa_site(rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_gemv,
    )

    g = torch.Generator(device="cuda").manual_seed(rows)
    w = (torch.randn(BFA, 7168, generator=g, device="cuda") * 0.02).bfloat16()
    x = torch.randn(rows, 7168, generator=g, device="cuda").bfloat16()
    with torch.inference_mode():
        gemvs = decode_gemv.K3DecodeGemvs.create(sites=("kda_bfa",))
        y = gemvs.project("kda_bfa", x, w)
        ref_bf16 = F.linear(x, w)
    assert y is not None, "k3_ctm_gemv_wide does not take the kda_bfa site"
    assert y.shape == (rows, BFA) and y.dtype == torch.bfloat16
    ref = x.float() @ w.float().t()
    scale = ref.abs().max().item()
    # bf16 output: one rounding of an fp32 accumulation; cuBLAS's split-K result rounds the same sum.
    assert (y.float() - ref).abs().max().item() <= 1e-2 * scale
    assert (ref_bf16.float() - ref).abs().max().item() <= 1e-2 * scale


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
def test_kda_bfa_row_tiles_site(rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_gemv,
    )

    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    w = (torch.randn(BFA, 7168, generator=g, device="cuda") * 0.02).bfloat16()
    x = torch.randn(rows, 7168, generator=g, device="cuda").bfloat16()
    with torch.inference_mode():
        gemvs = decode_gemv.K3DecodeGemvs.create(sites=("kda_bfa", "kda_bfa_rt"))
        y = gemvs.project("kda_bfa_rt", x, w)
        ref = gemvs.project("kda_bfa", x, w)
    assert y is not None and ref is not None
    assert torch.equal(y, ref)


def _fp8_block_weight(n, k, gen):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as m

    w = torch.randn((n, k), device="cuda", generator=gen) * 0.02
    blocks = w.view(n // 128, 128, k // 128, 128)
    scale = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0))
    )
    w8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    return m.prepare_k3_fp8w_lowm_weight(w8, scale.float())


def _rel(y, ref):
    return ((y.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
def test_bfa_fp8_rows(rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        fp8w_lowm,
    )

    g = torch.Generator(device="cuda").manual_seed(200 + rows)
    w_q, s_q = _fp8_block_weight(4 * HK, 7168, g)
    w_bfa = (torch.randn(BFA, 7168, generator=g, device="cuda") * 0.02).bfloat16()
    w_bfa[K + H :] = 0  # the [f_a | b] weight's padding rows
    x = torch.randn(rows, 7168, generator=g, device="cuda").bfloat16()
    with torch.inference_mode():
        fused = fp8w_lowm.append_rows(w_q, s_q, w_bfa)
        assert fused is not None
        w_f, s_f = fused
        assert w_f.shape == (3216, 7168) and torch.equal(w_f[: 4 * HK], w_q)
        assert s_f.shape == (3216, s_q.shape[1]) and s_f.stride() == (1, 3216)
        assert torch.equal(s_f[: 4 * HK], s_q)
        y = fp8w_lowm.gemm(x, w_f, s_f)
        ref_q = torch.ops.trtllm.k3_fp8w_lowm_gemm(x, w_q, s_q)
    assert y is not None and y.shape == (rows, 3216)
    # The QKVG columns: the same FP8 products and MXFP8 activation, summed at another split-K.
    assert _rel(y[:, : 4 * HK], ref_q.float()) < 1e-2
    # The [f_a | b] columns: FP8 weight and MXFP8 activation against the bf16 product.
    ref_b = x.float() @ w_bfa.float().t()
    err = _rel(y[:, 4 * HK : 4 * HK + K + H], ref_b[:, : K + H])
    assert err < 5e-2, err
    assert torch.count_nonzero(y[:, 4 * HK + K + H :]) == 0


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_projection_order(rows, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_gemv,
        flags,
        implementation,
    )
    from tensorrt_llm._torch.modules.multi_stream_utils import set_do_multi_stream

    kda = implementation.K3DecodeKDA
    g = torch.Generator(device="cuda").manual_seed(rows)
    w_bfa = (torch.randn(BFA, 7168, generator=g, device="cuda") * 0.02).bfloat16()
    # A bf16 stand-in for the FP8 QKVG module: the order and streams are under test, not its kernel.
    w_qkvg = (torch.randn(4 * HK, 7168, generator=g, device="cuda") * 0.02).bfloat16()
    x = torch.randn(rows, 7168, generator=g, device="cuda").bfloat16()
    calls = []
    mod = types.SimpleNamespace(
        _bfa_gemvs=None,
        _bfa_proj_weight=w_bfa,
        _projection_fork_event=torch.cuda.Event(),
        _projection_join_event=torch.cuda.Event(),
        _projection_aux_stream=torch.cuda.Stream(),
    )

    def qkvg_proj(t):
        calls.append(("qkvg", torch.cuda.current_stream()))
        return F.linear(t, w_qkvg)

    def k3_bfa(t):
        calls.append(("bfa", torch.cuda.current_stream()))
        return kda._k3_bfa(mod, t)

    mod.qkvg_proj, mod._k3_bfa = qkvg_proj, k3_bfa
    out = {}
    with torch.inference_mode():
        mod._bfa_gemvs = decode_gemv.K3DecodeGemvs.create(sites=("kda_bfa",))
        set_do_multi_stream(True)
        try:
            for side_first in (0, 1):
                for overlap in (0, 1):
                    monkeypatch.setattr(flags, "TRTLLM_K3LL_KDA_BFA_SIDE_FIRST", side_first)
                    monkeypatch.setattr(flags, "TRTLLM_K3LL_KDA_BFA_OVERLAP", overlap)
                    calls.clear()
                    main = torch.cuda.current_stream()
                    out[side_first, overlap] = kda._k3_projections(mod, x)
                    torch.cuda.synchronize()
                    order = [c[0] for c in calls]
                    streams = dict(calls)
                    assert streams["qkvg"] == main
                    if overlap:
                        assert order == (["bfa", "qkvg"] if side_first else ["qkvg", "bfa"])
                        assert streams["bfa"] == mod._projection_aux_stream
                    else:
                        assert order == ["qkvg", "bfa"] and streams["bfa"] == main
        finally:
            set_do_multi_stream(False)
    ref_q, ref_b = out[0, 0]
    assert torch.equal(ref_b, mod._bfa_gemvs.project("kda_bfa", x, w_bfa))
    for key, (q, b) in out.items():
        assert torch.equal(q, ref_q) and torch.equal(b, ref_b), (
            f"BFA_SIDE_FIRST, BFA_OVERLAP = {key}"
        )


def _inputs(seed: int, num_spec: int, ssm_dtype: torch.dtype):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=g, device="cuda") * scale

    wt = {
        "w_fb": (rnd(HK, K) * 0.05).bfloat16(),
        "w_q": rnd(HK, W, scale=0.3), "w_k": rnd(HK, W, scale=0.3), "w_v": rnd(HK, W, scale=0.3),
        "a_log": rnd(H, scale=0.5), "dt_bias": rnd(HK, scale=0.5), "onorm_w": (1 + 0.1 * rnd(V)).float(),
    }  # fmt: skip
    s = W - 1 + num_spec
    pools = {name: (rnd(POOL, s, HK) * 0.5).transpose(1, 2) for name in ("cs_q", "cs_k", "cs_v")}
    pools["ssm"] = (rnd(POOL, H, V, K) * 0.05).to(ssm_dtype)
    pools["state_tok"] = torch.zeros(POOL, 3, num_spec, H, K, device="cuda")
    pools["pending"] = torch.zeros(POOL, dtype=torch.int32, device="cuda")
    return wt, pools


def _clone(pools):
    return {
        n: (t.transpose(1, 2).clone().transpose(1, 2) if n.startswith("cs_") else t.clone())
        for n, t in pools.items()
    }


def _call(op, wt, p, slots, num_spec, proj, bfa=None):
    return op(
        proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        p["cs_q"], p["cs_k"], p["cs_v"], p["ssm"], p["state_tok"], slots, p["pending"], num_spec,
        LOWER_BOUND, SCALE, EPS, None, bfa,
    )  # fmt: skip


@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("num_requests,steps", [(1, 8), (2, 8), (4, 8), (3, 4)])
def test_verify_separate_bfa_rows(num_requests, steps, ssm_dtype):
    op = _verify_op()
    num_spec = steps - 1
    tokens = num_requests * steps
    wt, pools = _inputs(7 * num_requests + steps, num_spec, ssm_dtype)
    fused_p, sep_p = pools, _clone(pools)
    slots = torch.arange(1, 2 * num_requests, 2, dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(100 + tokens)
    with torch.inference_mode():
        for rnd in range(4):
            qkvg = torch.randn(tokens, 4 * HK, generator=g, device="cuda").bfloat16()
            bfa = torch.randn(tokens, BFA, generator=g, device="cuda").bfloat16()
            out_fused = _call(op, wt, fused_p, slots, num_spec, torch.cat((qkvg, bfa), dim=-1))
            out_sep = _call(op, wt, sep_p, slots, num_spec, qkvg, bfa)
            assert torch.equal(out_fused, out_sep), f"round {rnd}: outputs differ"
            for name in ("ssm", "state_tok", "cs_q", "cs_k", "cs_v"):
                assert torch.equal(fused_p[name], sep_p[name]), f"round {rnd}: {name} differs"
            # The sampler's accepted drafts for the next round, per slot.
            acc = torch.randint(
                0, steps, (num_requests,), generator=g, device="cuda", dtype=torch.int32
            )
            fused_p["pending"][slots.long()] = acc
            sep_p["pending"][slots.long()] = acc

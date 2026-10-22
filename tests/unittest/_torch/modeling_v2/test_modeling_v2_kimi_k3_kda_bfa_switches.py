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
"""The KDA [f_a | b] switches of the Kimi K3 NVFP4 decode path (``kimi_k3_nvfp4/flags.py``), each arm against the other
and the off arm against the op chain it replaced, at the TP16 per-rank shapes, one GPU.

What runs. ``K3DecodeKDA``'s projection methods (``_k3_projections``, ``_k3_bfa``, ``_k3_fused_rows``) on a stand-in
module holding what they read: the FP8 QKVG weight-read weight (3072 x 7168, 128x128 UE8M0 blocks, DeepGEMM-packed
scale) behind a callable that runs ``trtllm::k3_fp8w_lowm_gemm`` as the routed module does, the bf16 [f_a | b] weight
(136 x 7168, rows 134-135 zero), the decode GEMV state with the ``kda_bfa`` and ``kda_bfa_rt`` sites, and the
projection stream with its fork / join events. Multi-stream is on, as under CUDA-graph capture in the engine.

The chain the switches replaced (``_ref_projections``): ``k3_fp8w_lowm_gemm`` for QKVG and the [f_a | b] rows from
``k3_ctm_gemv_long`` at split 8, ring 6, push, 8 rows per call: the ``kda_bfa`` site's ``k3_ctm_gemv_wide`` at 136
rows runs split 8, whose rows its contract certifies bit-identical to the long kernel's at the same split.

Checks, at 8 / 16 / 32 rows, each eagerly and from a captured CUDA graph replayed with rewritten inputs (every replay
bit for bit the eager call on the same inputs):
  * TRTLLM_K3LL_KDA_BFA_SIDE_FIRST: off arm (OVERLAP on) bit for bit the reference chain; on arm bit for bit the off
    arm (the same kernels on the same inputs, another enqueue order).
  * TRTLLM_K3LL_KDA_BFA_ROW_TILES: off arm bit for bit the reference chain; on arm (``kda_bfa_rt``, 4 tiles of 34
    rows, split 8) bit for bit the off arm.
  * TRTLLM_K3LL_KDA_BFA_FP8: off arm: ``_k3_fused_rows`` returns None and launches nothing, so the projections are the
    reference chain; on arm: the QKVG columns within 1e-2 (relative norm) of the off arm's (the same FP8 products, a
    partial last row tile and another split-K), the [f_a | b] columns within 5e-2 of the off arm's bf16 ones (an FP8
    weight and the MXFP8 activation), the padding columns zero.
"""

import types

import pytest
import torch

H, K, HK, DIM = 6, 128, 768, 7168
BFA = K + H + 2
ROWS = (8, 16, 32)


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100-family (tcgen05, TMA, clusters)")


def _mods():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_gemv,
        flags,
        fp8w_lowm,
        implementation,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op as ctm_op

    return types.SimpleNamespace(
        gemv=decode_gemv,
        flags=flags,
        fp8w=fp8w_lowm,
        impl=implementation,
        lowm=k3_fp8w_lowm_gemm,
        ctm=ctm_op,
    )


@pytest.fixture(scope="module")
def stand_in():
    """The stand-in K3DecodeKDA (see the module docstring), built once."""
    m = _mods()
    g = torch.Generator(device="cuda").manual_seed(13)
    w = torch.randn((4 * HK, DIM), device="cuda", generator=g) * 0.02
    blocks = w.view(4 * HK // 128, 128, DIM // 128, 128)
    scale = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0))
    )
    w8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(4 * HK, DIM)
    w_q, s_q = m.lowm.prepare_k3_fp8w_lowm_weight(w8, scale.float())
    w_bfa = (torch.randn(BFA, DIM, generator=g, device="cuda") * 0.02).bfloat16()
    w_bfa[K + H :] = 0

    def qkvg_proj(x):
        return torch.ops.trtllm.k3_fp8w_lowm_gemm(x, w_q, s_q)

    qkvg_proj.weight, qkvg_proj.weight_scale = w_q, s_q
    mod = types.SimpleNamespace(
        qkvg_proj=qkvg_proj,
        _bfa_proj_weight=w_bfa,
        _qkvg_bfa_fp8=None,
        layer_idx=0,
        _projection_fork_event=torch.cuda.Event(),
        _projection_join_event=torch.cuda.Event(),
        _projection_aux_stream=torch.cuda.Stream(),
    )
    kda = m.impl.K3DecodeKDA
    for name in ("_k3_bfa", "_k3_projections", "_k3_fused_rows", "build_bfa_fp8"):
        setattr(mod, name, types.MethodType(getattr(kda, name), mod))
    with torch.inference_mode():
        mod._bfa_gemvs = m.gemv.K3DecodeGemvs.create(sites=("kda_bfa", "kda_bfa_rt"))
    return mod


def _x(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, DIM, generator=g, device="cuda").bfloat16()


def _ref_projections(mod, x):
    """The chain the switches replaced: QKVG on k3_fp8w_lowm_gemm, [f_a | b] on k3_ctm_gemv_long (split 8)."""
    m = _mods()
    qkvg = mod.qkvg_proj(x)
    bfa = torch.cat(
        [
            m.ctm.k3_ctm_gemv_long(x[i : i + 8], mod._bfa_proj_weight, split=8, ring=6, push=True)
            for i in range(0, x.shape[0], 8)
        ]
    )
    return qkvg, bfa


@pytest.fixture
def flags(monkeypatch):
    m = _mods()
    from tensorrt_llm._torch.modules.multi_stream_utils import set_do_multi_stream

    def set_(**values):
        for name, value in values.items():
            monkeypatch.setattr(m.flags, f"TRTLLM_K3LL_KDA_BFA_{name}", value)

    set_(OVERLAP=1, SIDE_FIRST=0, ROW_TILES=0, FP8=0, GEMV=1)
    set_do_multi_stream(True)
    yield set_
    set_do_multi_stream(False)


def _eager_and_replay(fn, rows: int):
    """``fn(x)`` eagerly on two inputs, and from one CUDA graph captured on the first and replayed on the second
    (``x`` rewritten in place): returns the eager results and asserts each replay matches them bit for bit."""
    xs = [_x(rows, 100 + rows), _x(rows, 200 + rows)]
    with torch.inference_mode():
        eager = [tuple(t.clone() for t in fn(x)) for x in xs]
        x_static = xs[0].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fn(x_static)
        for x, ref in zip(xs, eager):
            x_static.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(out, ref):
                assert torch.equal(got, want), "CUDA-graph replay differs from the eager call"
    return eager


def _assert_equal(a, b, what):
    for i, (x, y) in enumerate(zip(a, b)):
        for p, q in zip(x, y):
            assert torch.equal(p, q), f"{what}, input {i}"


@pytest.mark.parametrize("rows", ROWS)
def test_side_first(stand_in, flags, rows):
    ref = _eager_and_replay(lambda x: _ref_projections(stand_in, x), rows)
    off = _eager_and_replay(stand_in._k3_projections, rows)
    _assert_equal(off, ref, "SIDE_FIRST off vs the reference chain")
    flags(SIDE_FIRST=1)
    on = _eager_and_replay(stand_in._k3_projections, rows)
    _assert_equal(on, off, "SIDE_FIRST on vs off")


@pytest.mark.parametrize("rows", ROWS)
def test_row_tiles(stand_in, flags, rows):
    ref = _eager_and_replay(lambda x: _ref_projections(stand_in, x), rows)
    off = _eager_and_replay(stand_in._k3_projections, rows)
    _assert_equal(off, ref, "ROW_TILES off vs the reference chain")
    flags(ROW_TILES=1)
    on = _eager_and_replay(stand_in._k3_projections, rows)
    _assert_equal(on, off, "ROW_TILES on vs off")


def _rel(y, ref):
    return ((y.float() - ref.float()).norm() / ref.float().norm()).item()


@pytest.mark.parametrize("rows", ROWS)
def test_fp8(stand_in, flags, rows):
    with torch.inference_mode():
        assert stand_in._k3_fused_rows(_x(rows, 1)) is None, "FP8 off: the fused rows must not run"
    off = _eager_and_replay(stand_in._k3_projections, rows)
    ref = _eager_and_replay(lambda x: _ref_projections(stand_in, x), rows)
    _assert_equal(off, ref, "FP8 off vs the reference chain")
    flags(FP8=1)
    with torch.inference_mode():
        assert stand_in._k3_fused_rows(_x(rows, 1)) is not None  # eager: builds the fused weight
    on = _eager_and_replay(lambda x: (stand_in._k3_fused_rows(x),), rows)
    for (y,), (qkvg, bfa) in zip(on, off):
        assert y.shape == (rows, 3216)
        assert _rel(y[:, : 4 * HK], qkvg) < 1e-2
        assert _rel(y[:, 4 * HK : 4 * HK + K + H], bfa[:, : K + H]) < 5e-2
        assert torch.count_nonzero(y[:, 4 * HK + K + H :]) == 0

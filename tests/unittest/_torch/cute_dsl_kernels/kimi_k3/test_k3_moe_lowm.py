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
"""The NVFP4 MoE decode path of the Kimi K3 NVFP4 target (decode_moe_lowm.py), op by op, on SM100 / SM103.

  route_quant : trtllm::k3_route_quant with nvfp4_gsf against trtllm::fp4_quantize (sf_vec_size 16, linear scales)
                on the same bf16 latent, bit for bit, and the routing bit for bit the MXFP8 call's;
  lowm        : trtllm::k3_moe_nvfp4_lowm (blackwell/k3_moe_lowm_dyn.py) against the CuteDSL MoE backend's
                regular kernels on the same buffers (cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell, then
                cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell): FC1 rows and scales bit for bit, every FC2
                partial bit for bit (one non-zero routing weight per token), the full sum within the bf16 accumulation
                bound of the exact sum of the partials; through nvfp4_lowm_op.run (moe_sort first) as well, eagerly
                and from CUDA-graph replays, the counters zero after every launch; at the TP4 x EP4 slice (768), the
                TP8 x EP2 slice (384) and the TP16 slice (192, all 896 experts on one rank: a partial last FC2 K
                stage), also over the FC2 unit sizes and ring depths the microbenchmark sweeps, and against fp64 math
                on the dequantized operands; the 768 slice's kernel configuration as before partial stages existed;
  front (4 GPUs, one process each) : trtllm::k3_moe_front with nvfp4_gsf: the routing and the shared activation
                bit for bit the MXFP8 call's; the NVFP4 latent against fp4_quantize of the unfused chain's bf16
                latent (> 99.9 % equal codes and scales: the front sums the head in its own split-K order, so a code
                may move at a rounding tie), within one E2M1 step of the reference; then the routed experts on the
                front's outputs through nvfp4_lowm_op.run against the regular kernels on the same inputs.

Run under pytest (the front check takes a pool of 4 MPI workers) or directly:
  python3 test_k3_moe_lowm.py route_quant lowm
  srun -N1 -n4 --mpi=pmix python3 test_k3_moe_lowm.py front
"""

import pickle
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the multi-GPU check is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _trtllm():
    """``torch.ops.trtllm`` (named only in a nested function, see test_k3_moe_front.py)."""

    def namespace():
        return torch.ops.trtllm

    return namespace()


WORLD = 4
HIDDEN, LATENT, EXPERTS, TOP_K = 7168, 3584, 896, 16
SHARED_INTER = 6144
GATE_CAP, LINEAR_CAP = 4.0, 25.0
RSF = 2.827
I_TP, E_LOCAL = 768, 224  # one rank of the routed experts' TP4 x EP4
SF16 = 16


def _sm_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _sm_ok(), reason="needs sm_100 or sm_103"),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _fp4_ref(x: torch.Tensor, gsf: torch.Tensor):
    """fp4_quantize as the CuteDSL MoE backend quantizes its input: E4M3 scales per 16, linear layout."""
    q, s = _trtllm().fp4_quantize(x, gsf, SF16, False, False)
    m = x.shape[0]
    return _bits(q).reshape(m, -1), _bits(s).reshape(-1)[: m * (x.shape[1] // SF16)].reshape(m, -1)


# --------------------------------------------------------------------------------------------------------------------
# route_quant (one GPU)
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 7, 8, 9, 33, 64])
@pytest.mark.parametrize("amax_frac", [0.8, 1.0, 4.0], ids=["headroom", "calibrated", "saturating"])
def test_route_quant_nvfp4(m, amax_frac):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import (
        op,  # noqa: F401  (registers the op)
    )

    gen = torch.Generator(device="cuda").manual_seed(7 + m)
    logits = torch.randn(m, EXPERTS, device="cuda", generator=gen) * 2.0
    bias = (torch.randn(EXPERTS, device="cuda", generator=gen) * 0.05).float()
    h = (torch.randn(m, LATENT, device="cuda", generator=gen) * 0.3).bfloat16()
    h[0, :32] = 0  # a zero scale group
    if m > 1:
        h[1, 16:48] *= 64.0  # a few large groups
    gsf = torch.tensor([448.0 * 6.0 / (float(h.abs().max()) * amax_frac)], device="cuda")
    ids, w, q, s = _trtllm().k3_route_quant(logits, bias, h, RSF, False, gsf)
    ids8, w8, _, _ = _trtllm().k3_route_quant(logits, bias, h, RSF, False)
    r_q, r_s = _fp4_ref(h, gsf)
    torch.cuda.synchronize()
    assert q.shape == (m, LATENT // 2) and s.shape == (m, LATENT // SF16)
    assert _same(ids, ids8) and _same(w, w8), "routing differs from the MXFP8 call"
    assert _same(s, r_s), f"scales: {(s != r_s).sum().item()} of {s.numel()} differ"
    assert _same(q, r_q), f"codes: {(q != r_q).sum().item()} of {q.numel()} bytes differ"
    again = _trtllm().k3_route_quant(logits, bias, h, RSF, True, gsf)
    assert all(_same(a, b) for a, b in zip(again, (ids, w, q, s))), "early_trigger changed the bits"


# --------------------------------------------------------------------------------------------------------------------
# lowm (one GPU)
# --------------------------------------------------------------------------------------------------------------------
def _quantize(t):
    gsf = t.abs().max().float() / (448 * 6)
    q, sf = _trtllm().fp4_quantize(t, 1 / gsf, SF16, False)
    return q, sf, gsf


def _nvfp4_experts(num_local, hidden, inter_size, seed):
    """Random NVFP4 experts in the CuteDSL backend's buffer shapes (W1 [L, 2 I, H], W2 [L, H, I]; scales in the
    128x4 blocked layout) with per-expert global scales; and the FC2 input global scale."""
    torch.manual_seed(seed)
    w1 = torch.randn(num_local, 2 * inter_size, hidden, device="cuda", dtype=torch.bfloat16)
    w1, w1_sf, w1_gsf = _quantize(w1)
    w1_sf = w1_sf.view(num_local, 2 * inter_size, hidden // SF16)
    w2 = torch.randn(num_local, hidden, inter_size, device="cuda", dtype=torch.bfloat16)
    w2, w2_sf, w2_gsf = _quantize(w2)
    w2_sf = w2_sf.view(num_local, hidden, inter_size // SF16)
    fc2_input_scale = torch.tensor([448 * 6 / 4.0], dtype=torch.float32, device="cuda")
    spread = torch.linspace(0.5, 1.5, num_local, device="cuda")
    return SimpleNamespace(
        w1=w1.view(torch.uint8), w1_sf=w1_sf.view(torch.uint8), w2=w2.view(torch.uint8), w2_sf=w2_sf.view(torch.uint8),
        w1_gsf=w1_gsf, alpha2=(spread * (w2_gsf / fc2_input_scale[0])).float(), spread=spread,
        fc2_input_scale=fc2_input_scale,
    )  # fmt: skip


def _regular(
    x_q, x_sf, ex, alpha1, t2g, t2lim, p2e, nnet, scales, num_experts, top_k, offset, inter_size
):
    """The CuteDSL backend's regular kernels: gather FC1 + SiTU + NVFP4 quantization, then finalize FC2."""
    from tensorrt_llm._torch.utils import ActivationType

    num_local = ex.w1.shape[0]
    c, c_sf = _trtllm().cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        x_q.view(torch.float4_e2m1fn_x2), ex.w1.view(torch.float4_e2m1fn_x2), x_sf.view(torch.uint8), ex.w1_sf,
        alpha1, t2g, t2lim, p2e, nnet, ex.fc2_input_scale, num_experts=num_experts, top_k=top_k,
        num_local_experts=num_local, local_expert_offset=offset, tile_size=128, scaling_vector_size=SF16,
        activation_type=int(ActivationType.SiTu), situ_beta=GATE_CAP, situ_linear_beta=LINEAR_CAP,
    )  # fmt: skip
    c, c_sf = c.view(torch.uint8), c_sf.view(torch.uint8)

    def finalize(token_scales):
        out = torch.zeros(x_q.shape[0], ex.w2.shape[1], dtype=torch.bfloat16, device="cuda")
        _trtllm().cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
            input=c.view(torch.float4_e2m1fn_x2), weight=ex.w2.view(torch.float4_e2m1fn_x2),
            input_scale=c_sf.flatten(), weight_scale=ex.w2_sf, alpha=ex.alpha2, output=out,
            tile_idx_to_group_idx=t2g, tile_idx_to_mn_limit=t2lim, permuted_idx_to_expanded_idx=p2e,
            num_non_exiting_tiles=nnet, token_final_scales=token_scales, num_experts=num_experts, top_k=top_k,
            num_local_experts=num_local, local_expert_offset=offset, tile_size=128, output_dtype=torch.bfloat16,
        )  # fmt: skip
        return out

    return c, c_sf, finalize


def _sort(ids, scales, num_experts, top_k, offset, num_local):
    return _trtllm().moe_sort(
        token_selected_experts=ids, token_final_scales=scales, num_experts=num_experts, top_k=top_k,
        local_expert_offset=offset, local_num_experts=num_local, tile_tokens_dim=128,
    )  # fmt: skip


#: (hidden, intermediate slice, experts, local experts, local offset, top_k) of one rank's routed experts. The TP8 x EP2
#: (slice 384) and TP16 (slice 192) cases run FC2 with a partial last 256-wide K stage.
LOWM_SHAPES = {
    "k3_tp4ep4": (LATENT, I_TP, EXPERTS, E_LOCAL, E_LOCAL, TOP_K),
    "k3_tp8ep2": (LATENT, 384, EXPERTS, 448, 448, TOP_K),
    "k3_tp16": (LATENT, 192, EXPERTS, EXPERTS, 0, TOP_K),
    "small": (1024, 256, 64, 16, 0, 8),
}


def _lowm_case(shape, num_tokens, one_tile=False):
    """Routing, sorted tiles, NVFP4 tokens and experts of one rank, and the regular kernels' FC1 output and finalize.
    ``one_tile``: every token's first expert is the rank's first one and its others are not on the rank."""
    from tensorrt_llm._torch.utils import unswizzle_sf

    hidden, inter_size, num_experts, num_local, offset, top_k = shape
    torch.manual_seed(2000 + num_tokens)
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    scales, ids = logits.topk(top_k, dim=-1)
    ids = ids.to(torch.int32)
    scales = scales.softmax(dim=-1).float()
    ids[0, 0] = offset  # at least one token on this rank
    if one_tile:
        ids[:, 0] = offset
        ids[:, 1:] = offset + num_local + torch.arange(top_k - 1, dtype=torch.int32, device="cuda")
    t2g, t2lim, _, p2e, _, nnet = _sort(ids, scales, num_experts, top_k, offset, num_local)
    m = p2e.shape[0]
    tiles = int(nnet.item())
    limits = t2lim[:tiles].tolist()
    rows = torch.tensor([r for t in range(tiles) for r in range(t * 128, limits[t])], device="cuda")
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    x_q, x_sf, x_gsf = _quantize(x)
    x_sf = unswizzle_sf(x_sf, (num_tokens + 127) // 128 * 128, hidden)[:num_tokens].contiguous()
    ex = _nvfp4_experts(num_local, hidden, inter_size, 3000 + num_tokens)
    alpha1 = (ex.spread * (x_gsf * ex.w1_gsf)).float()
    c_ref, c_sf_ref, finalize = _regular(x_q, x_sf, ex, alpha1, t2g, t2lim, p2e, nnet, scales, num_experts, top_k,
                                         offset, inter_size)  # fmt: skip
    return SimpleNamespace(
        hidden=hidden, inter_size=inter_size, num_experts=num_experts, num_local=num_local, offset=offset, top_k=top_k,
        num_tokens=num_tokens, ids=ids, scales=scales, t2g=t2g, t2lim=t2lim, p2e=p2e, nnet=nnet, m=m, rows=rows,
        x_q=x_q, x_sf=x_sf, ex=ex, alpha1=alpha1, c_ref=c_ref, c_sf_ref=c_sf_ref, finalize=finalize,
        weights=(ex.w1, ex.w1_sf, alpha1, ex.w2, ex.w2_sf, ex.alpha2),
        counters=nvfp4_lowm_counters(),
        out=torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda"),
        h=torch.zeros(m, inter_size // 2, dtype=torch.uint8, device="cuda"),
        h_sf=torch.zeros(m * inter_size // SF16, dtype=torch.uint8, device="cuda"),
    )  # fmt: skip


def nvfp4_lowm_counters():
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    return nvfp4_lowm_op.new_counters(torch.device("cuda"))


def _lowm_call(c, token_scales, num_ctas, fc2_chunk=0, max_stages=0):
    """The op on the case's buffers (output pre-filled with garbage: the kernel zeroes it); counters zero after it."""
    c.out.fill_(7.0)
    _trtllm().k3_moe_nvfp4_lowm(
        _bits(c.x_q), _bits(c.x_sf), *c.weights[:3], *c.weights[3:], c.out, c.t2g, c.t2lim, c.p2e, c.nnet,
        c.ex.fc2_input_scale, token_scales, c.counters, c.top_k, GATE_CAP, LINEAR_CAP, num_ctas, c.h, c.h_sf,
        fc2_chunk, max_stages,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert int(c.counters.abs().sum().item()) == 0, "counters not reset"
    return c.out.clone()


def _check_partials(c, num_ctas, fc2_chunk=0, max_stages=0, slots=None):
    """FC1 rows and scales and every FC2 partial (one non-zero routing weight per token) bit for bit the regular
    kernels'; returns the exact sum of all partials and its bf16 accumulation bound."""
    from tensorrt_llm._torch.utils import unswizzle_sf

    partials = []
    for slot in range(c.top_k) if slots is None else slots:
        one_hot = torch.zeros_like(c.scales)
        one_hot[:, slot] = c.scales[:, slot]
        ref = c.finalize(one_hot)
        new = _lowm_call(c, one_hot, num_ctas, fc2_chunk, max_stages)
        assert torch.equal(ref.view(torch.int16), new.view(torch.int16)), f"FC2 slot {slot}"
        partials.append(ref.double())
        if len(partials) == 1:
            rows = c.rows
            assert torch.equal(c.c_ref[rows], c.h[rows]), "FC1 output rows differ"
            assert (c.h[rows] != 0).any(), "FC1 output is all zero"
            cols = c.inter_size // SF16
            sf_ref = unswizzle_sf(c.c_sf_ref, c.m, c.inter_size, SF16).view(c.m, cols)
            sf_new = unswizzle_sf(c.h_sf, c.m, c.inter_size, SF16).view(c.m, cols)
            assert torch.equal(sf_ref[rows], sf_new[rows]), "FC1 output scale factors differ"
    exact = torch.stack(partials).sum(0)
    bound = c.top_k * 2.0**-8 * torch.stack(partials).abs().sum(0) + 1e-30
    return exact, bound


def _check_sum(result, exact, bound, what):
    err = (result.double() - exact).abs()
    assert bool((err <= bound).all()), f"{what}: max err/bound {(err / bound).max()}"


@pytest.mark.parametrize("shape", list(LOWM_SHAPES.values()), ids=list(LOWM_SHAPES))
@pytest.mark.parametrize("num_tokens", [1, 7, 8, 14, 16, 32])
@pytest.mark.parametrize("num_ctas", [0, 100], ids=["ctas_max", "ctas_100"])
def test_moe_nvfp4_lowm(shape, num_tokens, num_ctas):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    ok, why = nvfp4_lowm_op.supported(torch.device("cuda"), 2 * shape[1], shape[0], num_tokens)
    assert ok, why
    c = _lowm_case(shape, num_tokens)
    exact, bound = _check_partials(c, num_ctas)
    _check_sum(_lowm_call(c, c.scales, num_ctas), exact, bound, "op")
    # nvfp4_lowm_op.run: moe_sort, then the op; bf16 routing weights converted to fp32 inside.
    run_args = (_bits(c.x_q), _bits(c.x_sf), c.ids, c.scales, c.weights, c.offset, c.num_experts,
                c.ex.fc2_input_scale, c.counters, GATE_CAP, LINEAR_CAP)  # fmt: skip
    y = nvfp4_lowm_op.run(*run_args, num_ctas=num_ctas)
    torch.cuda.synchronize()
    _check_sum(y, exact, bound, "run")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    static = torch.empty_like(y)
    with torch.cuda.graph(graph, stream=stream):
        nvfp4_lowm_op.run(*run_args, out=static, num_ctas=num_ctas)
    for replay in range(3):
        static.fill_(7.0)
        graph.replay()
        torch.cuda.synchronize()
        _check_sum(static, exact, bound, f"graph replay {replay}")
        assert int(c.counters.abs().sum().item()) == 0, "counters not reset after a replay"


@pytest.mark.parametrize(
    "shape", [LOWM_SHAPES["k3_tp8ep2"], LOWM_SHAPES["k3_tp16"]], ids=["k3_tp8ep2", "k3_tp16"]
)
@pytest.mark.parametrize("num_tokens", [8, 32])
@pytest.mark.parametrize("fc2_chunk, max_stages", [(1, 6), (2, 8), (4, 6), (7, 9), (28, 16)])
def test_moe_nvfp4_lowm_fc2_config(shape, num_tokens, fc2_chunk, max_stages):
    """The FC2 unit sizes and ring depths the microbenchmark sweeps at the partial-K slices: FC1 and two FC2 partials
    bit for bit the regular kernels', the full sum within the bound of the exact one (the partials of the default
    build)."""
    import tensorrt_llm  # noqa: F401

    c = _lowm_case(shape, num_tokens)
    exact, bound = _check_partials(c, 0)
    _check_partials(c, 0, fc2_chunk, max_stages, slots=(0, c.top_k - 1))
    _check_sum(
        _lowm_call(c, c.scales, 0, fc2_chunk, max_stages),
        exact,
        bound,
        f"chunk {fc2_chunk} stages {max_stages}",
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
def test_moe_nvfp4_lowm_one_tile(num_tokens):
    """One touched expert of FC2 K 192: three FC1 units, so unit 3, which zeroes the last output slice, is an FC2
    unit."""
    import tensorrt_llm  # noqa: F401

    c = _lowm_case((1024, 192, 64, 8, 0, 4), num_tokens, one_tile=True)
    assert int(c.nnet.item()) == 1
    exact, bound = _check_partials(c, 0)
    _check_sum(_lowm_call(c, c.scales, 0), exact, bound, "op")


def _situ_ref(up, gate):
    return (
        GATE_CAP
        * torch.tanh(gate / GATE_CAP)
        * torch.sigmoid(gate)
        * (LINEAR_CAP * torch.tanh(up / LINEAR_CAP))
    )


@pytest.mark.parametrize("shape", list(LOWM_SHAPES.values()), ids=list(LOWM_SHAPES))
@pytest.mark.parametrize("num_tokens", [8, 32])
def test_moe_nvfp4_lowm_dequant_ref(shape, num_tokens):
    """The kernel against fp64 math on the dequantized operands, independent of the regular kernels: FC1 output within
    one NVFP4 rounding (one scale unit) of SiTU of the exact FC1; the routed sum within the bf16 accumulation bound of
    the exact FC2 of the kernel's own FC1 output."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.utils import unswizzle_sf

    c = _lowm_case(shape, num_tokens)
    out = _lowm_call(c, c.scales, 0).double()
    hidden, inter = c.hidden, c.inter_size
    n1 = 2 * inter
    x_raw, _ = _dequant(_bits(c.x_q), _bits(c.x_sf), 1.0)
    h_sf = unswizzle_sf(c.h_sf, c.m, inter, SF16).view(c.m, inter // SF16)
    rows = c.rows.tolist()
    expanded = c.p2e[c.rows].tolist()
    ids = c.ids.tolist()
    by_expert = {}
    for r, e_idx in zip(rows, expanded):
        t, k = divmod(e_idx, c.top_k)
        by_expert.setdefault(ids[t][k] - c.offset, []).append((r, t, k))
    exact = torch.zeros(c.num_tokens, hidden, dtype=torch.float64, device="cuda")
    absum = torch.zeros_like(exact)
    for e, members in by_expert.items():
        rr = torch.tensor([r for r, _, _ in members], device="cuda")
        tt = torch.tensor([t for _, t, _ in members], device="cuda")
        kk = torch.tensor([k for _, _, k in members], device="cuda")
        w1, _ = _dequant(
            c.ex.w1[e], unswizzle_sf(c.ex.w1_sf[e], n1, hidden, SF16).view(n1, -1), 1.0
        )
        acc = (x_raw[tt].double() @ w1.double().t()).view(len(members), -1, 2, 64) * float(
            c.alpha1[e]
        )
        h_ref = _situ_ref(acc[:, :, 0], acc[:, :, 1]).reshape(len(members), inter)
        h_raw, h_scale = _dequant(c.h[rr], h_sf[rr], 1.0)
        h_val = h_raw.double() / float(c.ex.fc2_input_scale[0])
        unit = h_scale.double() / float(c.ex.fc2_input_scale[0])
        err = (h_val - h_ref).abs()
        # one scale unit: half the widest E2M1 step (4 -> 6); then slack for the MMA's FP32 accumulation and the
        # approximate exp2 / reciprocal of the SiTU, and for 16-groups below the smallest E4M3 scale
        tol = 1.01 * unit + 1e-3 * h_ref.abs() + 1e-6 * float(h_ref.abs().max())
        assert bool((err <= tol).all()), (
            f"expert {e}: FC1 output off by {(err / (unit + 1e-30)).max().item():.3f} scale units"
        )
        w2, _ = _dequant(
            c.ex.w2[e], unswizzle_sf(c.ex.w2_sf[e], hidden, inter, SF16).view(hidden, -1), 1.0
        )
        coef = (c.scales[tt, kk].double() * float(c.ex.alpha2[e])).unsqueeze(1)
        part = (h_raw.double() @ w2.double().t()) * coef
        exact.index_add_(0, tt, part)
        absum.index_add_(
            0,
            tt,
            part.abs() + 2.0**-12 * (h_raw.double().abs() @ w2.double().abs().t()) * coef.abs(),
        )
    assert bool((out != 0).any()), "output is all zero"
    bound = c.top_k * 2.0**-8 * absum + 1e-30
    err = (out - exact).abs()
    assert bool((err <= bound).all()), f"routed sum: max err/bound {(err / bound).max().item():.3f}"


@pytest.mark.parametrize("n_tok", [8, 16, 32])
def test_moe_nvfp4_lowm_k2_768_config_unchanged(n_tok):
    """The TP4 slice (FC2 K 768, every stage full) builds the kernel exactly as before partial FC2 stages existed: the
    op's default unit size and ring cap, three full FC2 stages, the token scales copied as three 2-KB stages, no
    K-step predicate (``k2_tail_kp`` 0 compiles none of the partial-stage code)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_moe_lowm_dyn as dyn
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    chunk, stages = nvfp4_lowm_op.kernel_config(n_tok, I_TP)
    assert (chunk, stages) == (1, {8: 6, 16: 6}.get(n_tok, 16))
    kernel = dyn.K3MoeLowMDynKernel(n_tok, 2 * I_TP, LATENT, TOP_K, GATE_CAP, LINEAR_CAP, fc2_chunk=chunk,
                                    max_ab_stage=stages)  # fmt: skip
    assert kernel.k2_tail_kp == 0 and kernel.nkb2 == I_TP // 256 == 3
    assert kernel.b2_sf_bytes == kernel.nkb2 * 2048
    b_stage = n_tok * 128
    old_slots = 1 if -(-3 * b_stage // 2048) * 2048 + 3 * 2048 <= 16384 else 2
    assert kernel.b2_slots == old_slots and kernel.fc2_chunk == 1


# --------------------------------------------------------------------------------------------------------------------
# front (4 GPUs)
# --------------------------------------------------------------------------------------------------------------------
def _context():
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op, nvfp4_lowm_op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op as moe_op
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    wl, we, inter = LATENT // world, EXPERTS // world, SHARED_INTER // world
    gen = torch.Generator(device="cuda").manual_seed(11)
    bias = (torch.randn(EXPERTS, device="cuda", generator=gen) * 0.05).float()
    x8 = torch.randn(8, HIDDEN, device="cuda", generator=gen).bfloat16()
    wgen = torch.Generator(device="cuda").manual_seed(111 + rank)
    head = (torch.randn(wl + we, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    head[wl:] *= 8.0
    gate_up = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    ws = moe_op.K3MoeHeadWorkspace.create(mapping, fabric_handle=True)
    # The 16-token build's own workspace, and 8 more rows for it (drawn after x8, which stays as it was).
    ws16 = moe_op.K3MoeHeadWorkspace.create(mapping, fabric_handle=True, m_max=16)
    x16 = torch.cat([x8, torch.randn(8, HIDDEN, device="cuda", generator=gen).bfloat16()])
    # The latent's calibrated global scale: the largest latent value of a few calls maps to 448 * 6.
    gsf = torch.tensor([448.0 * 6.0 / 0.5], device="cuda")
    ex = _nvfp4_experts(E_LOCAL, LATENT, I_TP, 4000 + rank)
    alpha1 = (ex.spread * (ex.w1_gsf / gsf[0])).float()
    return SimpleNamespace(
        comm=comm, rank=rank, world=world, wl=wl, inter=inter, bias=bias, x8=x8, head=head, gate_up=gate_up,
        front=front_op.front_weight(head, gate_up), ws=ws, ag=(ws.uc, ws.mc, ws.flags, ws.rank), gsf=gsf, ex=ex,
        x16=x16, ag16=(ws16.uc, ws16.mc, ws16.flags, ws16.rank), ws16=ws16,
        weights=(ex.w1, ex.w1_sf, alpha1, ex.w2, ex.w2_sf, ex.alpha2), offset=(rank % 4) * E_LOCAL,
        counters=nvfp4_lowm_op.new_counters(torch.device("cuda", torch.cuda.current_device())),
    )  # fmt: skip


def _front(ctx, x, nvfp4):
    ag = ctx.ag16 if x.shape[0] > 8 else ctx.ag
    return _trtllm().k3_moe_front(x, ctx.front, ctx.bias, RSF, ctx.inter, GATE_CAP, LINEAR_CAP, *ag, ctx.world,
                                  nvfp4_gsf=ctx.gsf if nvfp4 else None)  # fmt: skip


def _ref_latent(ctx, x):
    head = x.float() @ ctx.head.float().t()
    parts = [torch.from_numpy(a).cuda() for a in ctx.comm.allgather(head.cpu().numpy())]
    return torch.cat([p[:, : ctx.wl] for p in parts], dim=1).bfloat16().contiguous()


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dequant(q, s, gsf):
    """NVFP4 values (per 16 columns: E2M1 code * E4M3 scale / gsf) and the scale per column."""
    lo, hi = (q & 0xF).long(), (q >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], -1)
    table = _E2M1.to(q.device)
    vals = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(SF16, dim=1) / gsf
    return vals * scale, scale


def check_front(ctx):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    results = []
    for m in range(1, 9):
        x = ctx.x8[:m].contiguous()
        ids, w, q, s, shared = _front(ctx, x, True)
        ids8, w8, _, _, shared8 = _front(ctx, x, False)
        torch.cuda.synchronize()
        r_q, r_s = _fp4_ref(_ref_latent(ctx, x), ctx.gsf)
        dq, scale = _dequant(q, s, ctx.gsf[0])
        rdq, rscale = _dequant(r_q, r_s, ctx.gsf[0])
        # One E2M1 step is at most 2 units of the scale (4 -> 6); a code moved at a tie is one step.
        err_steps = ((dq - rdq).abs() / torch.maximum(scale, rscale)).max().item()
        y = nvfp4_lowm_op.run(q, s, ids, w, ctx.weights, ctx.offset, EXPERTS, ctx.ex.fc2_input_scale, ctx.counters,
                              GATE_CAP, LINEAR_CAP)  # fmt: skip
        scales = w.float()
        t2g, t2lim, _, p2e, _, nnet = _sort(ids, scales, EXPERTS, TOP_K, ctx.offset, E_LOCAL)
        _, _, finalize = _regular(q, s, ctx.ex, ctx.weights[2], t2g, t2lim, p2e, nnet, scales, EXPERTS, TOP_K,
                                  ctx.offset, I_TP)  # fmt: skip
        y_ref = finalize(scales)
        torch.cuda.synchronize()
        denom = y_ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        y_err_ulp = ((y.float() - y_ref.float()).abs() / denom).max().item() / 2.0**-8
        again = _front(ctx, x, True)
        gathered = ctx.comm.allgather([_bits(t).cpu() for t in (ids, w, q, s)])
        row = dict(
            op="k3_moe_front_nvfp4", case=f"tp{ctx.world}", M=m,
            routing_as_mxfp8=_same(ids, ids8) and _same(w, w8), shared_as_mxfp8=_same(shared, shared8),
            codes_equal=(q == r_q).float().mean().item(), scales_equal=(s == r_s).float().mean().item(),
            latent_err_scale_units=err_steps, experts_vs_regular_ulp=y_err_ulp,
            det=all(_same(a, b) for a, b in zip(again, (ids, w, q, s, shared))),
            ranks_agree=all(all(torch.equal(a, b) for a, b in zip(g, gathered[0])) for g in gathered),
            counters_zero=int(ctx.counters.abs().sum().item()) == 0,
        )  # fmt: skip
        good = (row["routing_as_mxfp8"] and row["shared_as_mxfp8"] and row["codes_equal"] > 0.999
                and row["scales_equal"] > 0.999 and err_steps <= 2.0 and y_err_ulp <= 8.0 and row["det"]
                and row["ranks_agree"] and row["counters_zero"])  # fmt: skip
        row["rank"], row["good"] = ctx.rank, bool(good)
        row["ok"] = all(ctx.comm.allgather(bool(good)))
        results.append(row)
    return results


def check_front16(ctx):
    """The 16-token build on 9-16 rows: every output bit for bit as the 8-token build's on rows [0, 8) and [8, M)
    (each token's head is its own dot products in the same split-K order, its routing and quantization its own)."""
    results = []
    for m in range(9, 17):
        x = ctx.x16[:m].contiguous()
        for nvfp4 in (True, False):
            out = _front(ctx, x, nvfp4)
            lo = _front(ctx, x[:8].contiguous(), nvfp4)
            hi = _front(ctx, x[8:].contiguous(), nvfp4)
            torch.cuda.synchronize()
            ref = [torch.cat([a, b]) for a, b in zip(lo, hi)]
            again = _front(ctx, x, nvfp4)
            # The routing and the latent are the same on every rank; the shared activation is this rank's shard.
            gathered = ctx.comm.allgather([_bits(t).cpu() for t in out[:4]])
            names = ("ids", "weights", "latent", "scales", "shared")
            row = dict(
                op="k3_moe_front16", case=f"tp{ctx.world}_{'nvfp4' if nvfp4 else 'mxfp8'}", M=m,
                **{f"{n}_as_8": _same(a, b) for n, a, b in zip(names, out, ref)},
                det=all(_same(a, b) for a, b in zip(again, out)),
                ranks_agree=all(all(torch.equal(a, b) for a, b in zip(g, gathered[0])) for g in gathered),
            )  # fmt: skip
            good = all(v for k, v in row.items() if k not in ("op", "case", "M"))
            row["rank"], row["good"] = ctx.rank, bool(good)
            row["ok"] = all(ctx.comm.allgather(bool(good)))
            results.append(row)
    return results


def _run_front(_=None):
    try:
        ctx = _context()
        with torch.inference_mode():
            return check_front(ctx) + check_front16(ctx)
    except Exception:
        traceback.print_exc()
        raise


def _report(per_rank):
    rows = list(per_rank[0]) + [row for rows in per_rank[1:] for row in rows if not row["good"]]
    for row in rows:
        fields = " ".join(f"{k}={(f'{v:.3e}' if isinstance(v, float) else v)}" for k, v in row.items()
                          if k not in ("op", "case", "M"))  # fmt: skip
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.skipif(
    MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} GPUs with MNNVL and mpi4py",
)
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_k3_moe_front_nvfp4(mpi_pool_executor):
    per_rank = list(mpi_pool_executor.map(_run_front, [None] * WORLD))
    _report(per_rank)
    assert all(row["ok"] for rows in per_rank for row in rows)


def main() -> int:
    names = sys.argv[1:] or ["route_quant", "lowm"]
    if "front" in names:
        rows = _run_front()
        per_rank = MPI.COMM_WORLD.gather(rows, root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            _report(per_rank)
            print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
        return 0 if all(r["ok"] for r in rows) else 1
    tests = {"route_quant": "test_route_quant_nvfp4", "lowm": "test_moe_nvfp4_lowm"}
    return pytest.main(["-q", __file__, "-k", " or ".join(tests[n] for n in names)])


if __name__ == "__main__":
    sys.exit(main())

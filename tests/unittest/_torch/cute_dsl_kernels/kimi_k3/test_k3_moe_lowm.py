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
                and from CUDA-graph replays, the counters zero after every launch; at most 16 tokens the same with
                the kernel's own sort of the ids (self_sort: no sort arguments read);
  front (4 GPUs, one process each) : trtllm::k3_moe_front with nvfp4_gsf: the routing and the shared activation
                bit for bit the MXFP8 call's; the NVFP4 latent against fp4_quantize of the unfused chain's bf16
                latent (> 99.9 % equal codes and scales: the front sums the head in its own split-K order, so a code
                may move at a rounding tie), within one E2M1 step of the reference; then the routed experts on the
                front's outputs through nvfp4_lowm_op.run against the regular kernels on the same inputs, with
                moe_sort and with the expert kernel's own sort at the rank's expert offset; the 16-token build
                (9-16 rows) as two 8-token calls; the 32-token build (17-32 rows) through FC1 + FC2 with moe_sort
                and with FC1 / FC2's own sort.

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
SHARED_INTER_32 = SHARED_INTER // 16  # the 32-token front's shared width per rank (TP16's)
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


@pytest.mark.parametrize(
    "shape",
    [(LATENT, I_TP, EXPERTS, E_LOCAL, E_LOCAL, TOP_K), (1024, 256, 64, 16, 0, 8)],
    ids=["k3_tp4ep4", "small"],
)
@pytest.mark.parametrize("num_tokens", [1, 7, 8, 14, 16, 32])
@pytest.mark.parametrize("num_ctas", [0, 100], ids=["ctas_max", "ctas_100"])
def test_moe_nvfp4_lowm(shape, num_tokens, num_ctas):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op
    from tensorrt_llm._torch.utils import unswizzle_sf

    hidden, inter_size, num_experts, num_local, offset, top_k = shape
    ok, why = nvfp4_lowm_op.supported(torch.device("cuda"), 2 * inter_size, hidden, num_tokens)
    assert ok, why
    torch.manual_seed(2000 + num_tokens)
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    scales, ids = logits.topk(top_k, dim=-1)
    ids = ids.to(torch.int32)
    scales = scales.softmax(dim=-1).float()
    ids[0, 0] = offset  # at least one token on this rank
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
    counters = nvfp4_lowm_op.new_counters(torch.device("cuda"))
    out = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    h = torch.zeros(m, inter_size // 2, dtype=torch.uint8, device="cuda")
    h_sf = torch.zeros(m * inter_size // SF16, dtype=torch.uint8, device="cuda")
    weights = (ex.w1, ex.w1_sf, alpha1, ex.w2, ex.w2_sf, ex.alpha2)

    def lowm(token_scales, fast=False):
        out.fill_(7.0)  # the kernel zeroes the output itself
        _trtllm().k3_moe_nvfp4_lowm(
            _bits(x_q), _bits(x_sf), *weights[:3], *weights[3:], out, t2g, t2lim, p2e, nnet, ex.fc2_input_scale,
            token_scales, counters, top_k, GATE_CAP, LINEAR_CAP, num_ctas, h, h_sf, static_first=fast, fast_tok=fast,
        )  # fmt: skip
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0, "counters not reset"
        return out.clone()

    partials = []
    for slot in range(top_k):
        one_hot = torch.zeros_like(scales)
        one_hot[:, slot] = scales[:, slot]
        ref = finalize(one_hot)
        new = lowm(one_hot)
        assert torch.equal(ref.view(torch.int16), new.view(torch.int16)), f"FC2 slot {slot}"
        partials.append(ref.double())
        if slot == 0:
            assert torch.equal(c_ref[rows], h[rows]), "FC1 output rows differ"
            assert (h[rows] != 0).any(), "FC1 output is all zero"
            cols = inter_size // SF16
            sf_ref = unswizzle_sf(c_sf_ref, m, inter_size, SF16).view(m, cols)
            sf_new = unswizzle_sf(h_sf, m, inter_size, SF16).view(m, cols)
            assert torch.equal(sf_ref[rows], sf_new[rows]), "FC1 output scale factors differ"
    exact = torch.stack(partials).sum(0)
    bound = top_k * 2.0**-8 * torch.stack(partials).abs().sum(0) + 1e-30

    def check(result, what):
        err = (result.double() - exact).abs()
        assert bool((err <= bound).all()), f"{what}: max err/bound {(err / bound).max()}"

    check(lowm(scales), "op")
    # nvfp4_lowm_op.run: moe_sort, then the op; bf16 routing weights converted to fp32 inside.
    run_args = (_bits(x_q), _bits(x_sf), ids, scales, weights, offset, num_experts, ex.fc2_input_scale, counters,
                GATE_CAP, LINEAR_CAP)  # fmt: skip
    y = nvfp4_lowm_op.run(*run_args, num_ctas=num_ctas)
    torch.cuda.synchronize()
    check(y, "run")
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
        check(static, f"graph replay {replay}")
        assert int(counters.abs().sum().item()) == 0, "counters not reset after a replay"

    def check_fast(op, self_sort):
        """``fast`` (TRTLLM_K3LL_MOE_LOWM_DYN_FAST: static_first + fast_tok): each one-hot slot bit for bit as the
        regular build, the op and `run` inside the bound, back to back (the last CTA resets the tile counters) and
        under graph replay."""
        tag = "self-sort fast" if self_sort else "fast"
        for slot in (0, top_k - 1):
            one_hot = torch.zeros_like(scales)
            one_hot[:, slot] = scales[:, slot]
            assert torch.equal(
                op(one_hot).view(torch.int16), op(one_hot, fast=True).view(torch.int16)
            ), f"{tag}: FC2 slot {slot}"
        for rep in range(3):
            check(op(scales, fast=True), f"{tag} op, call {rep}")
        for rep in range(2):
            y = nvfp4_lowm_op.run(*run_args, num_ctas=num_ctas, self_sort=self_sort, fast=True)
            torch.cuda.synchronize()
            assert int(counters.abs().sum().item()) == 0, f"{tag}: counters not reset"
            check(y, f"{tag} run, call {rep}")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            nvfp4_lowm_op.run(
                *run_args, out=static, num_ctas=num_ctas, self_sort=self_sort, fast=True
            )
        for replay in range(3):
            static.fill_(7.0)
            graph.replay()
            torch.cuda.synchronize()
            check(static, f"{tag} graph replay {replay}")
            assert int(counters.abs().sum().item()) == 0, (
                f"{tag}: counters not reset after a replay"
            )

    check_fast(lowm, self_sort=False)
    if not nvfp4_lowm_op.self_sort_takes(num_tokens, num_local):
        return

    # The kernel's own sort (TRTLLM_K3LL_MOE_SELF_SORT): the ids alone, the sort arguments never read.
    expanded = p2e.tolist()
    self_rows = torch.tensor(
        [t * 128 + sorted(expanded[t * 128 : limits[t]]).index(expanded[r]) for t in range(tiles)
         for r in range(t * 128, limits[t])], device="cuda",
    )  # fmt: skip

    def lowm_self(token_scales, fast=False):
        out.fill_(7.0)
        h.zero_()
        h_sf.zero_()
        _trtllm().k3_moe_nvfp4_lowm(
            _bits(x_q), _bits(x_sf), *weights[:3], *weights[3:], out, ids, ids, ids, ids, ex.fc2_input_scale,
            token_scales, counters, top_k, GATE_CAP, LINEAR_CAP, num_ctas, h, h_sf, topk_ids=ids, local_offset=offset,
            static_first=fast, fast_tok=fast,
        )  # fmt: skip
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0, "self-sort: counters not reset"
        return out.clone()

    for slot in (0, top_k - 1):
        one_hot = torch.zeros_like(scales)
        one_hot[:, slot] = scales[:, slot]
        new = lowm_self(one_hot)
        assert torch.equal(finalize(one_hot).view(torch.int16), new.view(torch.int16)), (
            f"self-sort FC2 slot {slot}"
        )
        if slot == 0:
            # Same tiles as moe_sort, the rows of a tile in token order (moe_sort's are in any order).
            assert torch.equal(c_ref[rows], h[self_rows]), "self-sort: FC1 output rows differ"
            cols = inter_size // SF16
            sf_ref = unswizzle_sf(c_sf_ref, m, inter_size, SF16).view(m, cols)
            sf_new = unswizzle_sf(h_sf, m, inter_size, SF16).view(m, cols)
            assert torch.equal(sf_ref[rows], sf_new[self_rows]), (
                "self-sort: FC1 output scale factors differ"
            )
    check(lowm_self(scales), "self-sort op")
    # bf16 routing weights (the front's output) are read by the kernel itself: one-hot slots bit for bit as their
    # fp32 widening.
    for slot in (0, top_k - 1):
        one_hot = torch.zeros_like(scales)
        one_hot[:, slot] = scales[:, slot]
        one_hot16 = one_hot.bfloat16()
        wide = lowm_self(one_hot16.float())
        assert torch.equal(lowm_self(one_hot16).view(torch.int16), wide.view(torch.int16)), (
            f"self-sort bf16 weights slot {slot}"
        )
    y = nvfp4_lowm_op.run(*run_args, num_ctas=num_ctas, self_sort=True)
    torch.cuda.synchronize()
    check(y, "run self_sort")
    bf16_args = run_args[:3] + (scales.bfloat16(),) + run_args[4:]
    y16 = nvfp4_lowm_op.run(*bf16_args, num_ctas=num_ctas, self_sort=True)
    torch.cuda.synchronize()
    exact16 = exact  # the bf16-rounded weights move the result by < 2^-8 relative per partial, inside the bound
    err16 = (y16.double() - exact16).abs()
    assert bool((err16 <= 2 * bound).all()), (
        f"run self_sort bf16: max err/bound {(err16 / bound).max()}"
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        nvfp4_lowm_op.run(*run_args, out=static, num_ctas=num_ctas, self_sort=True)
    for replay in range(3):
        static.fill_(7.0)
        graph.replay()
        torch.cuda.synchronize()
        check(static, f"self-sort graph replay {replay}")
        assert int(counters.abs().sum().item()) == 0, "self-sort: counters not reset after a replay"
    check_fast(lowm_self, self_sort=True)
    # no local expert: the output zeroed, the counters reset
    far = (
        ((offset + num_local + torch.arange(top_k, device="cuda")) % num_experts)
        .int()
        .expand(num_tokens, top_k)
    )
    for fast in (False, True):
        out.fill_(7.0)
        _trtllm().k3_moe_nvfp4_lowm(
            _bits(x_q), _bits(x_sf), *weights[:3], *weights[3:], out, ids, ids, ids, ids, ex.fc2_input_scale,
            scales, counters, top_k, GATE_CAP, LINEAR_CAP, num_ctas, h, h_sf, topk_ids=far.contiguous(),
            local_offset=offset, static_first=fast, fast_tok=fast,
        )  # fmt: skip
        torch.cuda.synchronize()
        assert int(out.view(torch.int16).abs().sum().item()) == 0, (
            f"no local expert (fast {fast}): output"
        )
        assert int(counters.abs().sum().item()) == 0, f"no local expert (fast {fast}): counters"


_REP_ROUNDS = (((0, 2), (1, 3)), ((0, 1), (2, 3)), ((0, 3), (1, 2)))


def _replica_choice(touched, table, home, replicas, groups=4):
    """S-1d's per-step choice, as ``K3MoeLowMDynKernel(replicas=...)`` and moss ``s1d_sim.py``'s ``pair`` rule:
    ``table[g R + c]`` the global slot copied into group ``g``'s copy slot ``c``; returns the moved entries."""
    hit = [e in touched for e in table]
    hg = [e // home for e in table]
    rg = [i // replicas for i in range(len(table))]
    k = {(a, b): sum(1 for i in range(len(table)) if hit[i] and hg[i] == a and rg[i] == b)
         for a in range(groups) for b in range(groups) if a != b}  # fmt: skip
    ld = [sum(1 for e in touched if e // home == g) - sum(k[(g, b)] for b in range(groups) if b != g)
          for g in range(groups)]  # fmt: skip
    mv = {}
    for rnd in _REP_ROUNDS:
        for a, b in rnd:
            la, lb = ld[a] + k[(a, b)], ld[b] + k[(b, a)]
            m_ab = min(k[(a, b)], (la - lb) // 2) if la >= lb else 0
            m_ba = min(k[(b, a)], (lb - la) // 2) if lb > la else 0
            ld[a], ld[b] = la - m_ab + m_ba, lb - m_ba + m_ab
            mv[(a, b)], mv[(b, a)] = m_ab, m_ba
    moved, taken = set(), {}
    for i in range(len(table)):
        if hit[i]:
            pair = (hg[i], rg[i])
            if taken.get(pair, 0) < mv[pair]:
                taken[pair] = taken.get(pair, 0) + 1
                moved.add(i)
    return moved


@pytest.mark.parametrize("home,top_k", [(224, 16), (32, 8)], ids=["k3_ep4", "small"])
@pytest.mark.parametrize("num_tokens", [1, 8, 13, 16])
@pytest.mark.parametrize("fast", [False, True], ids=["base", "fast"])
def test_moe_nvfp4_lowm_replicas(home, top_k, num_tokens, fast):
    """S-1d (``K3MoeLowMDynKernel(replicas=8)``): per EP group, the replica kernel (global ids, the group's home
    experts + 8 copies) bit for bit (one-hot routing slots) as the plain self-sort on the ids the reference choice
    (`_replica_choice`) assigns to the group, remapped to its local slots; FC1 rows bit for bit; summed over the 4
    groups, bit for bit the copy-free EP4 (each (token, expert) runs once)."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op
    from tensorrt_llm._torch.utils import unswizzle_sf

    hidden, inter_size, groups, R = 1024, 256, 4, 8
    num_experts = groups * home
    ok, why = nvfp4_lowm_op.supported(torch.device("cuda"), 2 * inter_size, hidden, num_tokens)
    assert ok, why
    gen = torch.Generator().manual_seed(7000 + 31 * num_tokens + home)
    # copies: group r holds R experts of other groups, biased to group 0 (the loaded one) so moves happen
    table, used = [], set()
    for r in range(groups):
        while len(table) < (r + 1) * R:
            src = (
                0
                if (r != 0 and torch.rand(1, generator=gen).item() < 0.6)
                else (r + 1 + int(torch.randint(groups - 1, (1,), generator=gen).item())) % groups
            )
            e = src * home + int(torch.randint(home, (1,), generator=gen).item())
            if src != r and e not in used:
                used.add(e)
                table.append(e)
    # routing: about half of each token's experts in group 0 (copied ones favoured), the rest anywhere
    hot = [e for e in table if e < home] + list(range(home))
    ids = torch.empty(num_tokens, top_k, dtype=torch.int32)
    for t in range(num_tokens):
        pick = []
        while len(pick) < top_k:
            if len(pick) < top_k // 2:
                e = hot[int(torch.randint(len(hot), (1,), generator=gen).item())]
            else:
                e = int(torch.randint(num_experts, (1,), generator=gen).item())
            if e not in pick:
                pick.append(e)
        ids[t] = torch.tensor(pick, dtype=torch.int32)
    scales = torch.rand(num_tokens, top_k, generator=gen).softmax(-1).float().cuda()
    ids = ids.cuda()
    touched = set(ids.flatten().tolist())
    moved = _replica_choice(touched, table, home, R)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    x_q, x_sf, x_gsf = _quantize(x)
    x_sf = unswizzle_sf(x_sf, (num_tokens + 127) // 128 * 128, hidden)[:num_tokens].contiguous()
    ex = _nvfp4_experts(num_experts, hidden, inter_size, 8000 + home)
    alpha1 = (ex.spread * (x_gsf * ex.w1_gsf)).float()
    counters = nvfp4_lowm_op.new_counters(torch.device("cuda"))
    tab = torch.tensor(table, dtype=torch.int32, device="cuda")

    def weights_of(slots):
        idx = torch.tensor(slots, device="cuda")
        return (ex.w1[idx].contiguous(), ex.w1_sf[idx].contiguous(), alpha1[idx].contiguous(), ex.w2[idx].contiguous(),
                ex.w2_sf[idx].contiguous(), ex.alpha2[idx].contiguous())  # fmt: skip

    def call(w, topk, token_scales, offset, replicas=0):
        n_local = w[0].shape[0]
        m = nvfp4_lowm_op.max_permuted_rows(num_tokens, top_k, n_local)
        out = torch.full((num_tokens, hidden), 7.0, dtype=torch.bfloat16, device="cuda")
        h = torch.zeros(m, inter_size // 2, dtype=torch.uint8, device="cuda")
        h_sf = torch.zeros(m * inter_size // SF16, dtype=torch.uint8, device="cuda")
        _trtllm().k3_moe_nvfp4_lowm(
            _bits(x_q), _bits(x_sf), *w, out, topk, topk, topk, topk, ex.fc2_input_scale, token_scales, counters,
            top_k, GATE_CAP, LINEAR_CAP, 0, h, h_sf, topk_ids=topk, local_offset=offset, static_first=fast,
            fast_tok=fast, replica_ids=tab if replicas else None, replicas=replicas,
        )  # fmt: skip
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0, "counters not reset"
        return out, h, h_sf

    far = num_experts + home + R  # an id no local slot takes
    sums_rep, sums_ep = None, None
    for slot in (0, top_k - 1, None):
        token_scales = scales
        if slot is not None:
            token_scales = torch.zeros_like(scales)
            token_scales[:, slot] = scales[:, slot]
        for g in range(groups):
            w = weights_of(list(range(g * home, (g + 1) * home)) + table[g * R : (g + 1) * R])
            y_rep, h_rep, hsf_rep = call(w, ids, token_scales, g * home, replicas=R)
            # reference: the group's share under the reference choice, remapped to its local slots
            away = {table[i] for i in moved if table[i] // home == g}
            local = {e: e - g * home for e in range(g * home, (g + 1) * home) if e not in away}
            local.update({table[g * R + c]: home + c for c in range(R) if g * R + c in moved})
            remap = ids.cpu().apply_(lambda e: local.get(e, far)).cuda()
            y_ref, h_ref, hsf_ref = call(w, remap.contiguous(), token_scales, 0)
            where = f"group {g} slot {slot} moved {sorted(moved)}"
            if slot is None:
                # several partials per token: bf16 reds in arrival order on both sides
                err = (y_rep.double() - y_ref.double()).abs().amax(-1)
                scale = y_ref.double().abs().amax(-1) + 1e-6
                assert bool((err <= top_k * 2.0**-7 * scale).all()), f"{where}: all slots"
                continue
            assert torch.equal(y_rep.view(torch.int16), y_ref.view(torch.int16)), f"{where}: output"
            assert torch.equal(h_rep, h_ref) and torch.equal(hsf_rep, hsf_ref), f"{where}: FC1 rows"
            y_ep, _, _ = call(
                weights_of(list(range(g * home, (g + 1) * home))), ids, token_scales, g * home
            )
            sums_rep = y_rep.float() if g == 0 else sums_rep + y_rep.float()
            sums_ep = y_ep.float() if g == 0 else sums_ep + y_ep.float()
        if slot is not None:
            assert torch.equal(sums_rep, sums_ep), f"slot {slot}: groups' sum vs copy-free EP4"
    assert moved or num_tokens == 1, "no copy moved: the routing does not exercise the choice"


@pytest.mark.parametrize("ceil_chunk", [False, True], ids=["fc2_equal_ranges", "fc2_ceil_chunk"])
@pytest.mark.parametrize("num_tokens", [17, 24, 28, 32])
def test_moe_nvfp4_lowm_fc12(num_tokens, ceil_chunk, monkeypatch):
    """trtllm::k3_moe_nvfp4_lowm_fc12 (K3MoeFc1LowMKernel + K3MoeFc2LowMKernel, TRTLLM_K3LL_MOE_LOWM_FC12; FC2's units
    in equal ranges or ceil chunks, TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK) against
    trtllm::k3_moe_nvfp4_lowm at K3 shapes: FC1 rows and scales bit for bit, every FC2 partial bit for bit (one
    non-zero routing weight per token), the full sum within the bf16 accumulation bound of the exact sum of the
    partials (both add a token's partials in hardware order); through run_sorted with the persistent buffers
    (nvfp4_lowm_op.new_fc12, the output zeroed in the call), and from CUDA-graph replays with the decode path's
    zeroing on its side stream (decode_moe_lowm.K3MoeFc12, forked before the experts as before the front, joined
    before FC1)."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        flags as k3ll_flags,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.decode_moe_lowm import (
        K3MoeFc12,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.utils import unswizzle_sf

    monkeypatch.setattr(k3ll_flags, "TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK", int(ceil_chunk))
    hidden, inter_size, num_experts, num_local, offset, top_k = (
        LATENT,
        I_TP,
        EXPERTS,
        E_LOCAL,
        E_LOCAL,
        TOP_K,
    )
    ok, why = nvfp4_lowm_op.supported(torch.device("cuda"), 2 * inter_size, hidden, num_tokens)
    assert ok, why
    torch.manual_seed(5000 + num_tokens)
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    scales, ids = logits.topk(top_k, dim=-1)
    ids = ids.to(torch.int32)
    scales = scales.softmax(dim=-1).float()
    ids[0, 0] = offset  # at least one token on this rank
    t2g, t2lim, _, p2e, _, nnet = _sort(ids, scales, num_experts, top_k, offset, num_local)
    m = p2e.shape[0]
    assert m <= nvfp4_lowm_op.max_permuted_rows(num_tokens, top_k, num_local)
    tiles = int(nnet.item())
    limits = t2lim[:tiles].tolist()
    rows = torch.tensor([r for t in range(tiles) for r in range(t * 128, limits[t])], device="cuda")
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    x_q, x_sf, x_gsf = _quantize(x)
    x_sf = unswizzle_sf(x_sf, (num_tokens + 127) // 128 * 128, hidden)[:num_tokens].contiguous()
    ex = _nvfp4_experts(num_local, hidden, inter_size, 6000 + num_tokens)
    alpha1 = (ex.spread * (x_gsf * ex.w1_gsf)).float()
    weights = (ex.w1, ex.w1_sf, alpha1, ex.w2, ex.w2_sf, ex.alpha2)
    counters = nvfp4_lowm_op.new_counters(torch.device("cuda"))
    out_dyn = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    out_fc12 = torch.empty_like(out_dyn)
    h_dyn = torch.zeros(m, inter_size // 2, dtype=torch.uint8, device="cuda")
    h_sf_dyn = torch.zeros(m * inter_size // SF16, dtype=torch.uint8, device="cuda")
    fc12 = nvfp4_lowm_op.new_fc12(torch.device("cuda"), 2 * inter_size, num_local, top_k)
    fc12 = fc12._replace(fc2_ceil_chunk=ceil_chunk)
    sorted_args = (t2g, t2lim, p2e, nnet, ex.fc2_input_scale)

    def dyn(token_scales):
        out_dyn.fill_(7.0)  # the kernel zeroes the output itself
        _trtllm().k3_moe_nvfp4_lowm(
            _bits(x_q), _bits(x_sf), *weights, out_dyn, *sorted_args, token_scales, counters, top_k, GATE_CAP,
            LINEAR_CAP, 0, h_dyn, h_sf_dyn,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out_dyn.clone()

    def two_kernels(token_scales):
        out_fc12.zero_()  # FC2 adds into its output
        _trtllm().k3_moe_nvfp4_lowm_fc12(
            _bits(x_q), _bits(x_sf), *weights, out_fc12, *sorted_args, token_scales, fc12.h, fc12.h_sf, top_k,
            GATE_CAP, LINEAR_CAP, fc2_ceil_chunk=ceil_chunk,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out_fc12.clone()

    partials = []
    for slot in range(top_k):
        one_hot = torch.zeros_like(scales)
        one_hot[:, slot] = scales[:, slot]
        ref = dyn(one_hot)
        new = two_kernels(one_hot)
        assert torch.equal(ref.view(torch.int16), new.view(torch.int16)), f"FC2 slot {slot}"
        partials.append(ref.double())
        if slot == 0:
            assert torch.equal(h_dyn[rows], fc12.h[rows]), "FC1 output rows differ"
            assert (fc12.h[rows] != 0).any(), "FC1 output is all zero"
            cols = inter_size // SF16
            sf_ref = unswizzle_sf(h_sf_dyn, m, inter_size, SF16).view(m, cols)
            sf_new = unswizzle_sf(fc12.h_sf[: m * cols], m, inter_size, SF16).view(m, cols)
            assert torch.equal(sf_ref[rows], sf_new[rows]), "FC1 output scale factors differ"
    exact = torch.stack(partials).sum(0)
    bound = top_k * 2.0**-8 * torch.stack(partials).abs().sum(0) + 1e-30

    def check(result, what):
        err = (result.double() - exact).abs()
        assert bool((err <= bound).all()), f"{what}: max err/bound {(err / bound).max()}"

    check(dyn(scales), "dyn")
    check(two_kernels(scales), "op")
    # run_sorted with fc12: the output (here not zero) zeroed on the current stream, then FC1 + FC2.
    static = torch.full_like(out_dyn, 7.0)
    y = nvfp4_lowm_op.run_sorted(_bits(x_q), _bits(x_sf), (t2g, t2lim, p2e, nnet, scales), weights,
                                 ex.fc2_input_scale, counters, GATE_CAP, LINEAR_CAP, out=static, fc12=fc12)  # fmt: skip
    torch.cuda.synchronize()
    assert y is static
    check(y, "run_sorted")

    # The decode path's launch: the output zeroed on K3MoeFc12's side stream (forked before the experts, joined
    # right before FC1), captured with multi-stream on as under the model's CUDA-graph capture.
    state = K3MoeFc12(fc12, torch.cuda.Stream(priority=-1), torch.cuda.Event(), torch.cuda.Event())
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream), with_multi_stream(True):
        step = state.begin(num_tokens, hidden, torch.device("cuda"))
        assert step.side
        launch = step.launch()
        assert launch.fc2_ceil_chunk == ceil_chunk, (
            "the decode launch reads TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK"
        )
        captured = nvfp4_lowm_op.run_sorted(
            _bits(x_q), _bits(x_sf), (t2g, t2lim, p2e, nnet, scales), weights, ex.fc2_input_scale, counters,
            GATE_CAP, LINEAR_CAP, out=step.out, fc12=launch,
        )  # fmt: skip
    for replay in range(3):
        captured.fill_(7.0)
        graph.replay()
        torch.cuda.synchronize()
        check(captured, f"graph replay {replay}")

    # FC1 and FC2's own per-CTA sort (TRTLLM_K3LL_MOE_SELF_SORT_FC12): the ids alone, the sort arguments never read.
    assert nvfp4_lowm_op.self_sort_takes(num_tokens, num_local, fc12=True)
    expanded = p2e.tolist()
    self_rows = torch.tensor(
        [t * 128 + sorted(expanded[t * 128 : limits[t]]).index(expanded[r]) for t in range(tiles)
         for r in range(t * 128, limits[t])], device="cuda",
    )  # fmt: skip

    def two_self(token_scales):
        out_fc12.zero_()
        fc12.h.zero_()
        fc12.h_sf.zero_()
        _trtllm().k3_moe_nvfp4_lowm_fc12(
            _bits(x_q), _bits(x_sf), *weights, out_fc12, ids, ids, ids, ids, ex.fc2_input_scale, token_scales,
            fc12.h, fc12.h_sf, top_k, GATE_CAP, LINEAR_CAP, topk_ids=ids, local_offset=offset,
            fc2_ceil_chunk=ceil_chunk,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out_fc12.clone()

    for slot in (0, 1, top_k - 1):
        one_hot = torch.zeros_like(scales)
        one_hot[:, slot] = scales[:, slot]
        ref = dyn(one_hot)
        new = two_self(one_hot)
        assert torch.equal(ref.view(torch.int16), new.view(torch.int16)), (
            f"self-sort FC2 slot {slot}"
        )
        if slot == 0:
            # Same tiles as moe_sort, the rows of a tile in token order (moe_sort's are in any order).
            assert torch.equal(h_dyn[rows], fc12.h[self_rows]), "self-sort: FC1 output rows differ"
            cols = inter_size // SF16
            sf_ref = unswizzle_sf(h_sf_dyn, m, inter_size, SF16).view(m, cols)
            sf_new = unswizzle_sf(fc12.h_sf[: m * cols], m, inter_size, SF16).view(m, cols)
            assert torch.equal(sf_ref[rows], sf_new[self_rows]), (
                "self-sort: FC1 output scale factors differ"
            )
        # bf16 routing weights (the front's output) read by FC2 itself: bit for bit as their fp32 widening
        one_hot16 = one_hot.bfloat16()
        wide = two_self(one_hot16.float())
        assert torch.equal(two_self(one_hot16).view(torch.int16), wide.view(torch.int16)), (
            f"self-sort bf16 weights slot {slot}"
        )
    check(two_self(scales), "self-sort op")
    run_args = (_bits(x_q), _bits(x_sf), ids, scales, weights, offset, num_experts, ex.fc2_input_scale, counters,
                GATE_CAP, LINEAR_CAP)  # fmt: skip
    static.fill_(7.0)
    y = nvfp4_lowm_op.run(*run_args, out=static, fc12=fc12, self_sort=True)
    torch.cuda.synchronize()
    check(y, "run self_sort fc12")
    bf16_args = run_args[:3] + (scales.bfloat16(),) + run_args[4:]
    static.fill_(7.0)
    y16 = nvfp4_lowm_op.run(*bf16_args, out=static, fc12=fc12, self_sort=True)
    torch.cuda.synchronize()
    err16 = (y16.double() - exact).abs()
    assert bool((err16 <= 2 * bound).all()), (
        f"run self_sort fc12 bf16: max err/bound {(err16 / bound).max()}"
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), with_multi_stream(True):
        step = state.begin(num_tokens, hidden, torch.device("cuda"))
        captured = nvfp4_lowm_op.run(*bf16_args, out=step.out, fc12=step.launch(), self_sort=True)
    for replay in range(3):
        captured.fill_(7.0)
        fc12.h.zero_()
        graph.replay()
        torch.cuda.synchronize()
        err16 = (captured.double() - exact).abs()
        assert bool((err16 <= 2 * bound).all()), (
            f"self-sort graph replay {replay}: {(err16 / bound).max()}"
        )


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
    # The 32-token build (bs4's wide step, 17-32 rows), with 16 more rows (drawn after x16). It takes the shared
    # expert's width of TP16 (its production world) per rank, SHARED_INTER // 16 columns, which also fits 4 ranks; the
    # 4-rank width does not fit its shared memory.
    ws32 = moe_op.K3MoeHeadWorkspace.create(mapping, fabric_handle=True, m_max=32)
    x32 = torch.cat([x16, torch.randn(16, HIDDEN, device="cuda", generator=gen).bfloat16()])
    gate_up32 = (
        torch.randn(2 * SHARED_INTER_32, HIDDEN, device="cuda", generator=wgen) * 0.02
    ).bfloat16()
    # The latent's calibrated global scale: the largest latent value of a few calls maps to 448 * 6.
    gsf = torch.tensor([448.0 * 6.0 / 0.5], device="cuda")
    ex = _nvfp4_experts(E_LOCAL, LATENT, I_TP, 4000 + rank)
    alpha1 = (ex.spread * (ex.w1_gsf / gsf[0])).float()
    return SimpleNamespace(
        comm=comm, rank=rank, world=world, wl=wl, inter=inter, bias=bias, x8=x8, head=head, gate_up=gate_up,
        front=front_op.front_weight(head, gate_up), ws=ws, ag=(ws.uc, ws.mc, ws.flags, ws.rank), gsf=gsf, ex=ex,
        x16=x16, ag16=(ws16.uc, ws16.mc, ws16.flags, ws16.rank), ws16=ws16,
        x32=x32, ag32=(ws32.uc, ws32.mc, ws32.flags, ws32.rank), ws32=ws32,
        front32=front_op.front_weight(head, gate_up32),
        weights=(ex.w1, ex.w1_sf, alpha1, ex.w2, ex.w2_sf, ex.alpha2), offset=(rank % 4) * E_LOCAL,
        counters=nvfp4_lowm_op.new_counters(torch.device("cuda", torch.cuda.current_device())),
        fc12=nvfp4_lowm_op.new_fc12(torch.device("cuda", torch.cuda.current_device()), 2 * I_TP, E_LOCAL, TOP_K),
    )  # fmt: skip


def _front(ctx, x, nvfp4):
    if x.shape[0] > 16:
        front, inter, ag = ctx.front32, SHARED_INTER_32, ctx.ag32
    else:
        front, inter, ag = ctx.front, ctx.inter, ctx.ag16 if x.shape[0] > 8 else ctx.ag
    return _trtllm().k3_moe_front(x, front, ctx.bias, RSF, inter, GATE_CAP, LINEAR_CAP, *ag, ctx.world,
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
        # The expert kernel's own sort of the front's ids (TRTLLM_K3LL_MOE_SELF_SORT) at this rank's offset.
        y_self = nvfp4_lowm_op.run(q, s, ids, w, ctx.weights, ctx.offset, EXPERTS, ctx.ex.fc2_input_scale,
                                   ctx.counters, GATE_CAP, LINEAR_CAP, self_sort=True)  # fmt: skip
        torch.cuda.synchronize()
        y_self_ulp = ((y_self.float() - y_ref.float()).abs() / denom).max().item() / 2.0**-8
        again = _front(ctx, x, True)
        gathered = ctx.comm.allgather([_bits(t).cpu() for t in (ids, w, q, s)])
        row = dict(
            op="k3_moe_front_nvfp4", case=f"tp{ctx.world}", M=m,
            routing_as_mxfp8=_same(ids, ids8) and _same(w, w8), shared_as_mxfp8=_same(shared, shared8),
            codes_equal=(q == r_q).float().mean().item(), scales_equal=(s == r_s).float().mean().item(),
            latent_err_scale_units=err_steps, experts_vs_regular_ulp=y_err_ulp,
            experts_self_sort_vs_regular_ulp=y_self_ulp,
            det=all(_same(a, b) for a, b in zip(again, (ids, w, q, s, shared))),
            ranks_agree=all(all(torch.equal(a, b) for a, b in zip(g, gathered[0])) for g in gathered),
            counters_zero=int(ctx.counters.abs().sum().item()) == 0,
        )  # fmt: skip
        good = (row["routing_as_mxfp8"] and row["shared_as_mxfp8"] and row["codes_equal"] > 0.999
                and row["scales_equal"] > 0.999 and err_steps <= 2.0 and y_err_ulp <= 8.0 and y_self_ulp <= 8.0
                and row["det"]
                and row["ranks_agree"] and row["counters_zero"])  # fmt: skip
        row["rank"], row["good"] = ctx.rank, bool(good)
        row["ok"] = all(ctx.comm.allgather(bool(good)))
        results.append(row)
    return results


def _experts_ulp(ctx, ids, w, q, s):
    """This rank's routed partial of the front's outputs through `run` with moe_sort and with the expert kernel's own
    sort (TRTLLM_K3LL_MOE_SELF_SORT), each against the regular kernels' (max error / row max, in bf16 ulps)."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    args = (q, s, ids, w, ctx.weights, ctx.offset, EXPERTS, ctx.ex.fc2_input_scale, ctx.counters, GATE_CAP,
            LINEAR_CAP)  # fmt: skip
    y = nvfp4_lowm_op.run(*args)
    y_self = nvfp4_lowm_op.run(*args, self_sort=True)
    scales = w.float()
    t2g, t2lim, _, p2e, _, nnet = _sort(ids, scales, EXPERTS, TOP_K, ctx.offset, E_LOCAL)
    _, _, finalize = _regular(q, s, ctx.ex, ctx.weights[2], t2g, t2lim, p2e, nnet, scales, EXPERTS, TOP_K,
                              ctx.offset, I_TP)  # fmt: skip
    y_ref = finalize(scales)
    torch.cuda.synchronize()
    denom = y_ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    return tuple(
        ((t.float() - y_ref.float()).abs() / denom).max().item() / 2.0**-8 for t in (y, y_self)
    )


def check_front16(ctx):
    """The 16-token build on 9-16 rows: every output bit for bit as the 8-token build's on rows [0, 8) and [8, M)
    (each token's head is its own dot products in the same split-K order, its routing and quantization its own);
    with the NVFP4 latent, also the routed partial through `run` with moe_sort and with the expert kernel's own sort,
    each within 8 bf16 ulps of the regular kernels'."""
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
            if nvfp4:
                y_ulp, y_self_ulp = _experts_ulp(ctx, *out[:4])
                row["experts_within_8ulp"] = y_ulp <= 8.0
                row["experts_self_sort_within_8ulp"] = y_self_ulp <= 8.0
                row["counters_zero"] = int(ctx.counters.abs().sum().item()) == 0
            good = all(v for k, v in row.items() if k not in ("op", "case", "M"))
            row["rank"], row["good"] = ctx.rank, bool(good)
            row["ok"] = all(ctx.comm.allgather(bool(good)))
            results.append(row)
    return results


def check_front32_fc12(ctx):
    """bs4's wide step: 17-32 rows through the 32-token front build (NVFP4 latent, bf16 routing weights), then this
    rank's routed partial through FC1 + FC2 (`run` with ``fc12``) with moe_sort, and with FC1 / FC2's own per-CTA sort
    (TRTLLM_K3LL_MOE_SELF_SORT_FC12): each within 8 bf16 ulps of the regular kernels' (FC2 adds a token's partials in
    hardware order, so repeated calls agree only within that bound)."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    results = []
    for m in (17, 24, 32):
        ids, w, q, s, _ = _front(ctx, ctx.x32[:m].contiguous(), True)
        args = (q, s, ids, w, ctx.weights, ctx.offset, EXPERTS, ctx.ex.fc2_input_scale, ctx.counters, GATE_CAP,
                LINEAR_CAP)  # fmt: skip
        assert nvfp4_lowm_op.self_sort_takes(m, E_LOCAL, fc12=True)
        y = nvfp4_lowm_op.run(*args, fc12=ctx.fc12).clone()
        y_self = nvfp4_lowm_op.run(*args, fc12=ctx.fc12, self_sort=True).clone()
        scales = w.float()
        t2g, t2lim, _, p2e, _, nnet = _sort(ids, scales, EXPERTS, TOP_K, ctx.offset, E_LOCAL)
        _, _, finalize = _regular(q, s, ctx.ex, ctx.weights[2], t2g, t2lim, p2e, nnet, scales, EXPERTS, TOP_K,
                                  ctx.offset, I_TP)  # fmt: skip
        y_ref = finalize(scales)
        torch.cuda.synchronize()
        denom = y_ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)

        def ulp(t):
            return ((t.float() - y_ref.float()).abs() / denom).max().item() / 2.0**-8

        gathered = ctx.comm.allgather([_bits(t).cpu() for t in (ids, w, q, s)])
        row = dict(
            op="k3_moe_front32_fc12", case=f"tp{ctx.world}", M=m,
            experts_within_8ulp=ulp(y) <= 8.0, self_sort_within_8ulp=ulp(y_self) <= 8.0,
            local_rows=bool(((ids >= ctx.offset) & (ids < ctx.offset + E_LOCAL)).any()),
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
            return check_front(ctx) + check_front16(ctx) + check_front32_fc12(ctx)
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

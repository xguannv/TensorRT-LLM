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
"""Kimi-K3 KDA verify kernel (``kda_mtp_decode_v2``) at NUM_SPEC = 8 (MTP8: nine new tokens).

Up to 17 chunk tokens: after a fully accepted round the kernel folds the first replayed draft
into the state (fp32 pre-step) and runs the 16-token chunk on the rest. Every variant of the
runtime contract is checked round by round against an fp64 torch recurrence (replay of the
accepted drafts, new tokens, in-place commit after the golden token, replay caches and conv
windows), with acceptance counts that cover 0..8: fp32 / bf16 state, forget gate from the
projection or computed in the kernel (``fa`` / ``w_fb``), plain or PDL launch, cluster-pair and
single-CTA grids. Bitwise: the in-kernel forget gate reproduces ``k3_kda_fab``'s, the PDL launch the
plain one, the MXFP8 side output the low-M GEMM's quantization of the BF16 output, and CUDA graph
replay the eager calls.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

H = 6
K = 128
W = 4
HIDDEN = 7168
N1 = 136
DIM = H * K
NUM_SPEC = 8
LOWER_BOUND = -5.0
EPS = 1e-5
POOLS = ("state", "cs_q", "cs_k", "cs_v", "k_cache", "g_cache", "v_cache", "beta_cache")


@pytest.fixture(autouse=True)
def _register_ops():
    """Registers trtllm::k3_kda_fab_proj / k3_kda_fab_fa and the verify ops."""
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_kda_fab  # noqa: F401


def _impl():
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import kda_mtp_decode_impl

    return kda_mtp_decode_impl


def _layer(batch, state_dtype, seed, num_spec=NUM_SPEC):
    """Weights and pools in the runtime's layouts (per-layer views of [layers, slots, ...] pools,
    dim-contiguous bf16 conv caches) with a state that already went through several rounds."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"

    def rn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=g) * scale).to(dtype).to(dev)

    pool = max(2 * batch, 8)
    s = W - 1 + num_spec
    w_bfa = torch.randn(N1, HIDDEN, generator=g) * 0.02
    w_bfa[K + H :] = 0
    state = torch.randn(3, pool, H, K, K, generator=g) * 0.05
    state *= torch.logspace(-1, 1, K).view(1, 1, 1, K, 1)
    return {
        "w_bfa": w_bfa.bfloat16().to(dev),
        "w_fb": rn(DIM, K, scale=0.05, dtype=torch.bfloat16),
        "w_q": rn(DIM, W, scale=0.5),
        "w_k": rn(DIM, W, scale=0.5),
        "w_v": rn(DIM, W, scale=0.5),
        "A_log": rn(H, scale=0.5),
        "dt_bias": rn(DIM, scale=0.5),
        "cs_q": rn(3, pool, s, DIM, dtype=torch.bfloat16).transpose(-1, -2)[1],
        "cs_k": rn(3, pool, s, DIM, dtype=torch.bfloat16).transpose(-1, -2)[1],
        "cs_v": rn(3, pool, s, DIM, dtype=torch.bfloat16).transpose(-1, -2)[1],
        "state": state.to(state_dtype).to(dev)[1],
        "k_cache": rn(3, pool, num_spec, DIM, scale=0.08, dtype=torch.bfloat16)[1],
        "g_cache": (-torch.rand(3, pool, num_spec, DIM, generator=g) * 0.5).to(dev)[1],
        "v_cache": rn(3, pool, num_spec, DIM, dtype=torch.bfloat16)[1],
        "beta_cache": torch.rand(3, pool, num_spec, H, generator=g).to(torch.bfloat16).to(dev)[1],
        "onorm_w": (0.75 + 0.5 * torch.rand(K, generator=g)).to(dev),
        "slots": torch.randperm(pool, generator=g)[:batch].to(torch.int32).to(dev),
        "accepted": torch.zeros(pool, dtype=torch.int32, device=dev),
        "gen": g,
    }


def _new_rows(lay, batch, num_spec=NUM_SPEC):
    """This round's projection inputs: hidden rows x (for the fab kernels), the [q | k | v |
    output gate] rows of the fused QKVG projection."""
    g = lay["gen"]
    m = batch * (num_spec + 1)
    chan = torch.ones(HIDDEN)
    chan[torch.randperm(HIDDEN, generator=g)[:16]] = 30.0  # activation outlier channels
    lay["x"] = (torch.randn(m, HIDDEN, generator=g) * chan).bfloat16().cuda()
    lay["qkvg"] = (torch.randn(m, 4 * DIM, generator=g) * 0.5).bfloat16().cuda()


def _set_acceptance(lay, accepted):
    lay["accepted"][lay["slots"].long()] = torch.tensor(accepted, dtype=torch.int32, device="cuda")


def _acceptance(batch, rnd):
    """Per-request accepted counts of round ``rnd``: every request sees 8 (the 17-token chunk)
    and 0 within four rounds, the others spread over 1..7."""
    return [
        8 if (n + rnd) % 4 == 0 else (0 if (n + rnd) % 4 == 2 else (3 * n + 5 * rnd) % 9)
        for n in range(batch)
    ]


def _clone(lay):
    out = dict(lay)
    for key in POOLS:
        t = lay[key]
        c = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        c.copy_(t)
        out[key] = c
    out["accepted"] = lay["accepted"].clone()
    return out


def _kwargs(lay, batch, num_spec, *, gate=None, beta=None, bfa=None, use_pdl=False):
    m = batch * (num_spec + 1)
    qkvg = lay["qkvg"]
    kw = dict(
        x_q=qkvg[:, :DIM].view(1, m, H, K),
        x_k=qkvg[:, DIM : 2 * DIM].view(1, m, H, K),
        x_v=qkvg[:, 2 * DIM : 3 * DIM].view(1, m, H, K),
        w_q=lay["w_q"],
        w_k=lay["w_k"],
        w_v=lay["w_v"],
        cs_q=lay["cs_q"],
        cs_k=lay["cs_k"],
        cs_v=lay["cs_v"],
        g=None if gate is None else gate.view(1, m, H, K),
        beta=(beta if bfa is None else bfa[:, K : K + H]).view(1, m, H),
        A_log=lay["A_log"],
        dt_bias=lay["dt_bias"],
        recurrent_state=lay["state"],
        k_cache=lay["k_cache"],
        g_cache=lay["g_cache"],
        v_cache=lay["v_cache"],
        beta_cache=lay["beta_cache"],
        ssm_state_indices=lay["slots"],
        cu_seqlens=None,
        num_spec=num_spec,
        num_accepted_tokens=lay["accepted"],
        lower_bound=LOWER_BOUND,
        scale=K**-0.5,
        packed_token_layout=True,
        onorm_g=qkvg[:, 3 * DIM :].view(1, m, H, K),
        onorm_weight=lay["onorm_w"],
        onorm_eps=EPS,
        fuse_output_norm=True,
        use_pdl=use_pdl,
    )
    if bfa is not None:
        kw["fa"] = bfa[:, :K].view(1, m, K)
        kw["w_fb"] = lay["w_fb"]
    return kw


def _run(lay, batch, *, fused_fb, use_pdl, num_spec=NUM_SPEC, mxfp8=False):
    """One verify round on ``lay``'s pools; returns (out, gate rows, beta rows) where the gate
    rows are what the kernel used (``k3_kda_fab``'s forget gate)."""
    beta, gate = torch.ops.trtllm.k3_kda_fab_proj(lay["x"], lay["w_bfa"], lay["w_fb"], H)
    if fused_fb:
        bfa = torch.ops.trtllm.k3_kda_fab_fa(lay["x"], lay["w_bfa"], H, lay["w_fb"])
        kw = _kwargs(lay, batch, num_spec, bfa=bfa, use_pdl=use_pdl)
    else:
        kw = _kwargs(lay, batch, num_spec, gate=gate, beta=beta, use_pdl=use_pdl)
    if mxfp8:
        m = batch * (num_spec + 1)
        xq = torch.empty((m, DIM), dtype=torch.float8_e4m3fn, device="cuda")
        xs = torch.empty((m, DIM // 32), dtype=torch.uint8, device="cuda")
        out = _impl()(**kw, mxfp8_out=(xq, xs))
        return out, gate, beta, (xq, xs)
    return _impl()(**kw), gate, beta


def _reference_round(lay, batch, gate, beta, num_spec=NUM_SPEC):
    """One verify round in fp64 from ``lay``'s pools (the kernel's semantics: replay of the
    accepted drafts from the caches, conv + SiLU + L2 norm of the new tokens, delta rule, commit
    after the golden token, spec tokens into the replay caches). Returns the raw outputs
    ``[m, H, V]`` and the expected pools."""
    f64 = torch.float64
    m = batch * (num_spec + 1)
    qkvg = lay["qkvg"].to(f64)
    xq = qkvg[:, :DIM]
    xk = qkvg[:, DIM : 2 * DIM]
    xv = qkvg[:, 2 * DIM : 3 * DIM]
    wq, wk, wv = (lay[n].to(f64) for n in ("w_q", "w_k", "w_v"))
    exp_a = torch.exp(lay["A_log"].to(f64))
    dt_bias = lay["dt_bias"].to(f64)
    gate = gate.to(f64).view(m, DIM)
    beta = beta.to(f64).view(m, H)
    exp = {n: lay[n].clone() for n in POOLS}
    out = torch.zeros(m, H, K, dtype=f64, device="cuda")
    scale = K**-0.5

    def silu(x):
        return x / (1.0 + torch.exp(-x))

    for n in range(batch):
        s = int(lay["slots"][n])
        a = int(lay["accepted"][s])
        hist = [lay[c][s, :, : W - 1].to(f64).clone() for c in ("cs_q", "cs_k", "cs_v")]
        st = lay["state"][s].to(f64).clone()  # [H, V, K]
        for i in range(a + 1 + num_spec):
            if i < a:
                raw = [lay[c][s, :, W - 1 + i].to(f64) for c in ("cs_q", "cs_k", "cs_v")]
                k_t = lay["k_cache"][s, i].to(f64).view(H, K)
                g_t = lay["g_cache"][s, i].to(f64).view(H, K)
                v_t = lay["v_cache"][s, i].to(f64).view(H, K)
                b_t = lay["beta_cache"][s, i].to(f64)
                q_t = None
            else:
                tok = n * (num_spec + 1) + i - a
                raw = [xq[tok], xk[tok], xv[tok]]
                conv = [
                    (torch.cat([hist[c], raw[c].unsqueeze(-1)], dim=-1) * w).sum(-1)
                    for c, w in zip(range(3), (wq, wk, wv))
                ]
                q_t = silu(conv[0]).view(H, K)
                q_t = q_t / torch.sqrt((q_t * q_t).sum(-1, keepdim=True) + 1e-6) * scale
                k_t = silu(conv[1]).view(H, K)
                k_t = k_t / torch.sqrt((k_t * k_t).sum(-1, keepdim=True) + 1e-6)
                v_t = silu(conv[2]).view(H, K)
                g_t = LOWER_BOUND * torch.sigmoid(
                    (gate[tok] + dt_bias).view(H, K) * exp_a.view(H, 1)
                )
                b_t = torch.sigmoid(beta[tok])
            hist = [torch.cat([hist[c][:, 1:], raw[c].unsqueeze(-1)], dim=-1) for c in range(3)]
            st = st * torch.exp(g_t).unsqueeze(1)
            v_new = (v_t - torch.einsum("hvk,hk->hv", st, k_t)) * b_t.unsqueeze(-1)
            st = st + torch.einsum("hv,hk->hvk", v_new, k_t)
            if q_t is not None:
                out[tok] = torch.einsum("hvk,hk->hv", st, q_t)
            if i == a:
                exp["state"][s] = st.to(exp["state"].dtype)
                for c, name in enumerate(("cs_q", "cs_k", "cs_v")):
                    exp[name][s, :, : W - 1] = hist[c].to(exp[name].dtype)
            if i > a:
                p = i - a - 1
                exp["k_cache"][s, p] = k_t.reshape(DIM).to(exp["k_cache"].dtype)
                exp["g_cache"][s, p] = g_t.reshape(DIM).to(exp["g_cache"].dtype)
                exp["v_cache"][s, p] = v_t.reshape(DIM).to(exp["v_cache"].dtype)
                exp["beta_cache"][s, p] = b_t.to(exp["beta_cache"].dtype)
                for c, name in enumerate(("cs_q", "cs_k", "cs_v")):
                    exp[name][s, :, W - 1 + p] = raw[c].to(exp[name].dtype)
    return out, exp


def _gated_norm(lay, o, batch, num_spec=NUM_SPEC):
    """The kernel's epilogue on the raw outputs: bf16(o) * w * sigmoid(gate) * rsqrt(mean(bf16(o)^2)
    + eps), in fp64."""
    m = batch * (num_spec + 1)
    val = o.to(torch.bfloat16).to(torch.float64)
    og = lay["qkvg"][:, 3 * DIM :].to(torch.float64).view(m, H, K)
    inv = torch.rsqrt((val * val).mean(-1, keepdim=True) + EPS)
    return val * lay["onorm_w"].to(torch.float64) * torch.sigmoid(og) * inv


def _assert_close(name, actual, expected, atol, rtol):
    a = actual.to(torch.float64)
    e = expected.to(torch.float64)
    diff = (a - e).abs()
    bad = diff > atol + rtol * e.abs()
    assert not bad.any(), (
        f"{name}: {int(bad.sum())} of {bad.numel()} elements off, max |diff| "
        f"{diff.max().item():.3e} (atol {atol}, rtol {rtol})"
    )


def _bits(t):
    return t.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _assert_same(name, actual, expected):
    assert actual.shape == expected.shape, name
    assert torch.equal(_bits(actual), _bits(expected)), (
        f"{name}: max |diff| {(actual.float() - expected.float()).abs().max().item():.3e}"
    )


def _check_round(lay_before, lay_after, out, gate, beta, batch, state_dtype):
    """Kernel round vs the fp64 recurrence run from the same pools."""
    o_ref, exp = _reference_round(lay_before, batch, gate, beta)
    m = batch * (NUM_SPEC + 1)
    _assert_close("out", out.view(m, H, K), _gated_norm(lay_before, o_ref, batch), 3e-2, 2e-2)
    if state_dtype == torch.float32:
        _assert_close("state", lay_after["state"], exp["state"], 2e-4, 2e-4)
    else:
        _assert_close("state", lay_after["state"], exp["state"], 2e-3, 8e-3)
    for name in ("cs_q", "cs_k", "cs_v"):
        _assert_same(name, lay_after[name], exp[name])
    _assert_close("k_cache", lay_after["k_cache"], exp["k_cache"], 1e-3, 8e-3)
    _assert_close("g_cache", lay_after["g_cache"], exp["g_cache"], 1e-5, 1e-5)
    _assert_close("v_cache", lay_after["v_cache"], exp["v_cache"], 1e-3, 8e-3)
    _assert_close("beta_cache", lay_after["beta_cache"], exp["beta_cache"], 1e-3, 8e-3)


@pytest.mark.parametrize("batch", [1, 2, 4, 16], ids=lambda b: f"b{b}")
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "fused_fb,use_pdl", [(True, True), (False, False)], ids=["fb_pdl", "gate_plain"]
)
def test_num_spec8_matches_recurrence(batch, state_dtype, fused_fb, use_pdl):
    """Four chained rounds with 0..8 accepted drafts per request (b1 / b2 / b4: cluster pairs,
    b16: single-CTA tiles), each against the fp64 recurrence from the same pools."""
    lay = _layer(batch, state_dtype, seed=31 + batch)
    for rnd in range(4):
        _new_rows(lay, batch)
        _set_acceptance(lay, _acceptance(batch, rnd))
        before = _clone(lay)
        out, gate, beta = _run(lay, batch, fused_fb=fused_fb, use_pdl=use_pdl)
        _check_round(before, lay, out, gate, beta, batch, state_dtype)


@pytest.mark.parametrize("batch", [1, 2, 4, 8], ids=lambda b: f"b{b}")
@pytest.mark.parametrize("use_pdl", [False, True], ids=["plain", "pdl"])
def test_num_spec8_fused_gate_bitwise(batch, use_pdl):
    """The in-kernel forget gate (second f_b tile for token 8) reproduces k3_kda_fab's gate:
    the fused verify equals the k3_kda_fab_proj + verify pair bit for bit over rounds with 0..8
    accepted drafts."""
    ref = _layer(batch, torch.bfloat16, seed=7 + batch)
    fused = _clone(ref)
    for rnd in range(4):
        _new_rows(ref, batch)
        fused["x"], fused["qkvg"] = ref["x"], ref["qkvg"]
        acc = _acceptance(batch, rnd)
        _set_acceptance(ref, acc)
        _set_acceptance(fused, acc)
        out_ref = _run(ref, batch, fused_fb=False, use_pdl=False)[0]
        out_fused = _run(fused, batch, fused_fb=True, use_pdl=use_pdl)[0]
        _assert_same(f"out (round {rnd})", out_fused, out_ref)
        for name in POOLS:
            _assert_same(f"{name} (round {rnd})", fused[name], ref[name])


@pytest.mark.parametrize("fused_fb", [False, True], ids=["gate", "fb"])
def test_num_spec8_pdl_bitwise(fused_fb):
    """PDL launch (state / replay / weight loads before griddepcontrol.wait) equals the plain
    launch bit for bit."""
    batch = 2
    plain = _layer(batch, torch.bfloat16, seed=5)
    pdl = _clone(plain)
    for rnd in range(4):
        _new_rows(plain, batch)
        pdl["x"], pdl["qkvg"] = plain["x"], plain["qkvg"]
        acc = _acceptance(batch, rnd)
        _set_acceptance(plain, acc)
        _set_acceptance(pdl, acc)
        a = _run(plain, batch, fused_fb=fused_fb, use_pdl=False)[0]
        b = _run(pdl, batch, fused_fb=fused_fb, use_pdl=True)[0]
        _assert_same(f"out (round {rnd})", b, a)
        for name in POOLS:
            _assert_same(f"{name} (round {rnd})", pdl[name], plain[name])


@pytest.mark.parametrize("batch", [1, 2], ids=lambda b: f"b{b}")
def test_num_spec8_mxfp8_side_output(batch):
    """MXFP8 side output (all nine tokens' rows) equals the low-M GEMM's quantization of the
    BF16 output, which itself matches the plain call."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

    lay = _layer(batch, torch.bfloat16, seed=9)
    plain = _clone(lay)
    m = batch * (NUM_SPEC + 1)
    for rnd in range(3):
        _new_rows(lay, batch)
        plain["x"], plain["qkvg"] = lay["x"], lay["qkvg"]
        acc = _acceptance(batch, rnd)
        _set_acceptance(lay, acc)
        _set_acceptance(plain, acc)
        out, _, _, (xq, xs) = _run(lay, batch, fused_fb=True, use_pdl=True, mxfp8=True)
        out_plain = _run(plain, batch, fused_fb=True, use_pdl=True)[0]
        _assert_same(f"out (round {rnd})", out, out_plain)
        ref_q, ref_s = mx.mxfp8_quantize_reference(out.view(m, DIM))
        _assert_same(f"xq (round {rnd})", xq, ref_q)
        _assert_same(f"xs (round {rnd})", xs, ref_s)


def test_num_spec8_cuda_graph():
    """Captured once and replayed over rounds with changing acceptance (0..8), the fused PDL
    verify equals eager calls bit for bit."""
    batch = 2
    eager = _layer(batch, torch.bfloat16, seed=13)
    graph = _clone(eager)
    _new_rows(eager, batch)
    graph["x"], graph["qkvg"] = eager["x"], eager["qkvg"]
    x_static = graph["x"].clone()
    qkvg_static = graph["qkvg"].clone()
    graph["x"], graph["qkvg"] = x_static, qkvg_static

    def step(lay):
        return _run(lay, batch, fused_fb=True, use_pdl=True)[0]

    warm = _clone(graph)
    step(warm)  # compile outside the capture
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        snapshot = _clone(graph)
        with torch.cuda.graph(g, stream=stream):
            out_static = step(graph)
    torch.cuda.synchronize()
    for name in POOLS:  # undo the capture-time pool writes (capture does not run the kernels)
        graph[name].copy_(snapshot[name])
    for rnd in range(4):
        _new_rows(eager, batch)
        x_static.copy_(eager["x"])
        qkvg_static.copy_(eager["qkvg"])
        acc = _acceptance(batch, rnd)
        _set_acceptance(eager, acc)
        _set_acceptance(graph, acc)
        g.replay()
        out_eager = step(eager)
        torch.cuda.synchronize()
        _assert_same(f"out (round {rnd})", out_static, out_eager)
        for name in POOLS:
            _assert_same(f"{name} (round {rnd})", graph[name], eager[name])


def test_num_spec8_needs_v2_contract(monkeypatch):
    """NUM_SPEC = 8 exists only on the v2 contract: the other kernels raise instead of running."""
    monkeypatch.setenv("TRTLLM_K3_KDA_MTP_V2", "0")
    batch = 1
    lay = _layer(batch, torch.bfloat16, seed=3)
    _new_rows(lay, batch)
    with pytest.raises(ValueError, match="num_spec=8 needs the v2 kernel contract"):
        _run(lay, batch, fused_fb=False, use_pdl=False)

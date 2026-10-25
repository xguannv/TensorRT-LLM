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
"""``k3_attn_res`` (CuTe DSL) against the C++ per-token attention-residual ops.

* The exact kernel reproduces the arithmetic of ``trtllm::attn_res_rmsnorm_fwd`` /
  ``trtllm::attn_res_add_rmsnorm_fwd`` instruction for instruction: every output is
  bit-identical.
* The fast (vectorized) kernel sums the statistics in another order: its updated prefix sum is
  bit-identical and its normalized output may differ from the C++ op by one BF16 ulp in a small
  fraction of the elements.
* With ``mnnvl_order`` the fast kernel runs the C++ MNNVL all-reduce epilogue's arithmetic
  (checked bitwise against that op by the multi-GPU o_proj push op matrix); here it is checked
  against the per-token C++ op like the fast kernel.
* Both run inside a CUDA graph behind PDL producers and consumers (DSL and C++).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

HIDDEN = 7168
EPS = 1e-6
OUT_EPS = 1e-5
# Fast kernel vs C++: an element differs by at most two BF16 rounding steps of its magnitude
# (plus a tiny absolute slack for values near zero), and few elements differ at all.
FAST_MAX_DIFF_FRACTION = 2e-2


def _k3():
    import tensorrt_llm  # noqa: F401  (registers the C++ trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    return k3_attn_res


def _inputs(m: int, n: int, scale: float, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return dict(
        block=rn(n - 1, m, HIDDEN, sc=scale),
        layer=rn(m, HIDDEN, sc=scale),
        addend=rn(m, HIDDEN, sc=0.5 * scale),
        shared=rn(m, HIDDEN, sc=0.5 * scale),
        proj=rn(HIDDEN, sc=0.05),
        norm_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
    )


def _weights(t):
    return (t["proj"], t["norm_w"], t["out_w"], EPS, OUT_EPS)


def _cpp(t, mode: int):
    """``(updated prefix or None, output)`` of the C++ path for add mode 0 / 1 / 2."""
    _k3()
    m, k = t["layer"].shape[0], t["block"].shape[0]
    block = t["block"].view(k, m, 1, HIDDEN)
    if mode == 1:
        updated, out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
            t["layer"].view(m, 1, HIDDEN), t["addend"].view(m, 1, HIDDEN), block, *_weights(t)
        )
        return updated.view(m, HIDDEN), out.view(m, HIDDEN)
    updated = None
    row = t["layer"]
    if mode == 2:
        # The MoE residual update, bf16(residual + bf16(routed + shared)): PyTorch's bf16 adds round once each.
        updated = row = t["layer"] + (t["addend"] + t["shared"])
    out = torch.ops.trtllm.attn_res_rmsnorm_fwd(row.view(m, 1, HIDDEN), block, *_weights(t))
    return updated, out.view(m, HIDDEN)


def _dsl(t, mode: int, exact: bool, cluster=None):
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    out = torch.empty_like(t["layer"])
    updated = torch.empty_like(t["layer"]) if mode else None
    _k3().run_k3_attn_res(
        t["layer"], addends, t["block"], *_weights(t), out, updated, exact=exact, cluster=cluster
    )
    return updated, out


def _mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


def _assert_fast_close(actual: torch.Tensor, expected: torch.Tensor, what):
    a, e = actual.float(), expected.float()
    diff = (a - e).abs()
    # 2^-6 relative is two BF16 ulps (a one-ulp change of the mixture can move the normalized
    # and the weighted rounding by one each). Where the mixture cancels to near zero, the two
    # summation orders differ by fp32 rounding of the (much larger) terms: 2^-8 of the row RMS.
    bound = (
        torch.maximum(a.abs(), e.abs()) * 2.0**-6 + 2.0**-8 * e.pow(2).mean(-1, keepdim=True).sqrt()
    )
    assert bool((diff <= bound).all()), (what, float((diff - bound).max()))
    frac = float((diff > 0).float().mean())
    assert frac <= FAST_MAX_DIFF_FRACTION, (what, frac)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", list(range(2, 13)))
@pytest.mark.parametrize("m", [1, 7, 32, 64])
def test_exact_bitwise(m: int, n: int, mode: int):
    for scale in (1.0, 30.0):
        t = _inputs(m, n, scale)
        ref_updated, ref_out = _cpp(t, mode)
        updated, out = _dsl(t, mode, exact=True)
        assert _mismatches(out, ref_out) == 0, f"scale {scale}"
        if mode:
            assert _mismatches(updated, ref_updated) == 0, f"scale {scale}"


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 9, 12])
@pytest.mark.parametrize("m", [1, 7, 32, 56, 64])
def test_fast_against_cpp(m: int, n: int, mode: int):
    clusters = (None, 2, 4, 8) if n > 8 else (None, 1, 2, 4, 8)
    for scale in (1.0, 30.0):
        t = _inputs(m, n, scale, seed=m * 13 + n)
        ref_updated, ref_out = _cpp(t, mode)
        for cluster in clusters:
            updated, out = _dsl(t, mode, exact=False, cluster=cluster)
            if mode:
                assert _mismatches(updated, ref_updated) == 0
            _assert_fast_close(out, ref_out, (cluster, scale))


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 4, 7, 12])
@pytest.mark.parametrize("m", [1, 8, 16, 64])
def test_mnnvl_order(m: int, n: int, mode: int):
    """``mnnvl_order`` (the C++ MNNVL epilogue's arithmetic; bitwise against that op in the
    multi-GPU ``_kimi_k3_oproj_push_op_matrix.py``): the same updated prefix, the per-token
    C++ op within the fast kernel's bound, deterministic, and the Lamport addend the same bits."""
    t = _inputs(m, n, 30.0, seed=7 * m + n)
    ref_updated, ref_out = _cpp(t, mode)
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    outs = []
    for _ in range(2):
        out = torch.empty_like(t["layer"])
        updated = torch.empty_like(t["layer"]) if mode else None
        _k3().run_k3_attn_res(
            t["layer"], addends, t["block"], *_weights(t), out, updated, mnnvl_order=True
        )
        outs.append(out)
    assert _mismatches(outs[0], outs[1]) == 0
    if mode:
        assert _mismatches(updated, ref_updated) == 0
    _assert_fast_close(outs[0], ref_out, "mnnvl_order")
    if mode == 1 and m <= 32:
        pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_lamport_ar")
        addend = t["addend"].clone()
        addend[addend == 0] = 0
        uc, flags, stage = _lamport_workspace(addend, dirty=1)
        stage.copy_(addend.view(torch.int32).flatten())
        got = [torch.empty_like(t["layer"]) for _ in range(2)]
        want = [torch.empty_like(t["layer"]) for _ in range(2)]
        k3 = _k3()
        k3.run_k3_attn_res(t["layer"], (addend,), t["block"], *_weights(t), *want, mnnvl_order=True)
        k3.run_k3_attn_res(
            t["layer"],
            (),
            t["block"],
            *_weights(t),
            *got,
            lamport_addend=(uc.data_ptr(), flags.data_ptr()),
            mnnvl_order=True,
        )
        for a, b in zip(want, got):
            assert _mismatches(a, b) == 0


@pytest.mark.parametrize("exact", [True, False])
def test_ops(exact: bool, monkeypatch):
    """The custom ops run the kernel ``TRTLLM_K3_ATTN_RES_PDL_EXACT`` selects."""
    k3 = _k3()
    monkeypatch.setattr(k3, "_EXACT", exact)
    t = _inputs(7, 9, 1.0)
    w = _weights(t)
    out = torch.ops.trtllm.k3_attn_res_rmsnorm(t["layer"], t["block"], *w)
    assert _mismatches(out, _dsl(t, 0, exact=exact)[1]) == 0
    updated, out = torch.ops.trtllm.k3_attn_res_add_rmsnorm(t["layer"], t["addend"], t["block"], *w)
    ref_updated, ref_out = _dsl(t, 1, exact=exact)
    assert _mismatches(updated, ref_updated) == 0 and _mismatches(out, ref_out) == 0
    prefix, out = torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    ref_prefix, ref_out = _dsl(t, 2, exact=exact)
    assert _mismatches(prefix, ref_prefix) == 0 and _mismatches(out, ref_out) == 0


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 6, 12])
@pytest.mark.parametrize("m", [1, 7, 14, 32, 64])
def test_mxfp8_side_output(m: int, n: int, mode: int, monkeypatch):
    """The ``_mx`` ops return the vectorized kernel's BF16 results and the MXFP8 form of the
    normalized output that the low-M GEMM's in-kernel quantization makes of it."""
    k3 = _k3()
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

    monkeypatch.setattr(k3, "_EXACT", False)
    t = _inputs(m, n, 1.0, seed=m + n)
    t["layer"][0, :64] = 0  # an all-zero 1x32 group of the mixture input
    w = _weights(t)
    ops = torch.ops.trtllm
    if mode == 0:
        ref = (ops.k3_attn_res_rmsnorm(t["layer"], t["block"], *w),)
        got = ops.k3_attn_res_rmsnorm_mx(t["layer"], t["block"], *w)
    elif mode == 1:
        ref = ops.k3_attn_res_add_rmsnorm(t["layer"], t["addend"], t["block"], *w)
        got = ops.k3_attn_res_add_rmsnorm_mx(t["layer"], t["addend"], t["block"], *w)
    else:
        ref = ops.k3_attn_res_moe_add_rmsnorm(t["layer"], t["addend"], t["shared"], t["block"], *w)
        got = ops.k3_attn_res_moe_add_rmsnorm_mx(
            t["layer"], t["addend"], t["shared"], t["block"], *w
        )
    for i, (a, b) in enumerate(zip(ref, got)):
        assert _mismatches(b, a) == 0, f"BF16 output {i}"
    xq, xs = got[len(ref) :]
    ref_q, ref_s = mx.mxfp8_quantize_reference(ref[-1])
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(xs, ref_s)


def _lamport_workspace(rows: torch.Tensor, dirty: int):
    """A one-rank stand-in for an MNNVL Lamport workspace (``k3_lamport_ar`` layout: three
    buffers of two stages) whose last all-reduce left ``rows`` in the broadcast stage of buffer
    ``dirty``; the rest holds the empty-slot sentinel."""
    m = rows.shape[0]
    stage_bytes = -(-m * HIDDEN * 2 // 1024) * 1024
    bytes_per_buffer = 2 * stage_bytes
    uc = torch.full((3 * bytes_per_buffer // 4,), -(2**31), dtype=torch.int32, device="cuda")
    flags = torch.zeros(9, dtype=torch.int32, device="cuda")
    flags[0], flags[1], flags[2] = (dirty + 1) % 3, dirty, bytes_per_buffer
    start = (dirty * 2 + 1) * stage_bytes // 4
    stage = uc[start : start + m * HIDDEN // 2]
    return uc, flags, stage


@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize("n", [2, 6, 12])
@pytest.mark.parametrize("m", [1, 7, 14, 32])
def test_lamport_addend(m: int, n: int, mx: bool, monkeypatch):
    """ADD_PREFIX with the addend polled from a Lamport workspace equals the tensor addend."""
    pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_lamport_ar")
    k3 = _k3()
    monkeypatch.setattr(k3, "_EXACT", False)
    t = _inputs(m, n, 1.0, seed=3 * m + n)
    addend = t["addend"].clone()
    # A Lamport row never holds -0 (the all-reduce stores +0), so no word is the sentinel.
    addend[addend == 0] = 0
    uc, flags, stage = _lamport_workspace(addend, dirty=2)
    stage.copy_(addend.view(torch.int32).flatten())
    w = _weights(t)
    ops = torch.ops.trtllm
    if mx:
        ref = ops.k3_attn_res_add_rmsnorm_mx(t["layer"], addend, t["block"], *w)
        got = ops.k3_attn_res_add_rmsnorm_lamport_mx(
            t["layer"], uc.data_ptr(), flags.data_ptr(), t["block"], *w
        )
    else:
        ref = ops.k3_attn_res_add_rmsnorm(t["layer"], addend, t["block"], *w)
        got = ops.k3_attn_res_add_rmsnorm_lamport(
            t["layer"], uc.data_ptr(), flags.data_ptr(), t["block"], *w
        )
    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"output {i}"


def test_lamport_addend_polls_until_rows_arrive():
    """The kernel waits for rows that land after it started (written from another stream)."""
    pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_lamport_ar")
    m, n = 14, 6
    t = _inputs(m, n, 1.0, seed=5)
    addend = t["addend"].clone()
    addend[addend == 0] = 0
    uc, flags, stage = _lamport_workspace(addend, dirty=0)
    w = _weights(t)
    ref = torch.ops.trtllm.k3_attn_res_add_rmsnorm(t["layer"], addend, t["block"], *w)
    src = addend.view(torch.int32).flatten()
    writer = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(writer):
        torch.cuda._sleep(2_000_000)
        stage.copy_(src)
    got = torch.ops.trtllm.k3_attn_res_add_rmsnorm_lamport(
        t["layer"], uc.data_ptr(), flags.data_ptr(), t["block"], *w
    )
    torch.cuda.synchronize()
    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"output {i}"


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("n", [4, 9])
def test_graph_replay_pdl_chain(n: int, exact: bool):
    """DSL -> DSL -> C++ -> DSL chain in one CUDA graph: every PDL hand-off is observed."""
    m = 32
    t = _inputs(m, n, 1.0)
    block4 = t["block"].view(n - 1, m, 1, HIDDEN)
    layer3 = t["layer"].view(m, 1, HIDDEN)
    addend = torch.empty_like(t["addend"])

    def dsl(a):
        out = torch.empty_like(a)
        updated = torch.empty_like(a)
        _k3().run_k3_attn_res(t["layer"], (a,), t["block"], *_weights(t), out, updated, exact=exact)
        return updated, out

    def cpp(a):
        u, o = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
            layer3, a.view(m, 1, HIDDEN), block4, *_weights(t)
        )
        return u.view(m, HIDDEN), o.view(m, HIDDEN)

    def chain(first, second, third, fourth):
        r1 = first(addend)
        r2 = second(r1[1])
        r3 = third(r2[1])
        r4 = fourth(r3[1])
        return [r1, r2, r3, r4]

    outs = []

    def body():
        outs[:] = chain(dsl, dsl, cpp, dsl)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        addend.copy_(t["addend"])
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        addend.copy_(torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        # The same chain run eagerly is the reference (same kernels, same inputs).
        refs = chain(dsl, dsl, cpp, dsl)
        for i, ((u, o), (ru, ro)) in enumerate(zip(outs, refs)):
            assert _mismatches(o, ro) == 0, f"replay {seed} stage {i} output"
            assert _mismatches(u, ru) == 0, f"replay {seed} stage {i} updated prefix"
        if exact:
            for i, ((u, o), (ru, ro)) in enumerate(zip(outs, chain(cpp, cpp, cpp, cpp))):
                assert _mismatches(o, ro) == 0, f"replay {seed} stage {i} output vs C++"


def _mixture_cpp(row: torch.Tensor, t) -> torch.Tensor:
    """The pre-norm mixture of ``trtllm::attn_res_fwd`` (what a separate attn_res computes for a tap)."""
    m, k = row.shape[0], t["block"].shape[0]
    out, _rsigma, _probs, _logits = torch.ops.trtllm.attn_res_fwd(
        row.reshape(m, 1, HIDDEN).contiguous(),
        t["block"].reshape(k, m, 1, HIDDEN).contiguous(),
        t["proj"],
        t["norm_w"],
        EPS,
    )
    return out.reshape(m, HIDDEN)


@pytest.mark.parametrize("mnnvl_order", [False, True])
@pytest.mark.parametrize("lamport", [False, True])
@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize("n", [2, 6, 12])
@pytest.mark.parametrize("m", [1, 14, 32])
def test_tap(m: int, n: int, mx: bool, lamport: bool, mnnvl_order: bool):
    """``tap``: the kernel also stores its pre-norm mixture into a strided view (a capture slot).
    The other outputs are the bits of the launch without it, the columns around the view are
    untouched, and the tap is the separate C++ attn_res's mixture within the fast kernel's bound."""
    if lamport:
        pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_lamport_ar")
    k3 = _k3()
    t = _inputs(m, n, 1.0, seed=11 * m + n)
    addend = t["addend"].clone()
    addend[addend == 0] = 0
    kwargs = dict(exact=False, mnnvl_order=mnnvl_order)
    if lamport:
        uc, flags, stage = _lamport_workspace(addend, dirty=1)
        stage.copy_(addend.view(torch.int32).flatten())
        addends = ()
        kwargs["lamport_addend"] = (uc.data_ptr(), flags.data_ptr())
    else:
        addends = (addend,)

    def run(tap=None):
        out = torch.empty_like(t["layer"])
        updated = torch.empty_like(t["layer"])
        mxo = None
        if mx:
            mxo = (
                torch.empty(m, HIDDEN, dtype=torch.float8_e4m3fn, device="cuda"),
                torch.empty(m, HIDDEN // 32, dtype=torch.uint8, device="cuda"),
            )
        k3.run_k3_attn_res(
            t["layer"], addends, t["block"], *_weights(t), out, updated, mxfp8_out=mxo, tap=tap,
            **kwargs,
        )  # fmt: skip
        return [out, updated] + (list(mxo) if mx else [])

    ref = run()
    # The capture buffer of three tapped layers; this layer's slot is the middle one.
    buf = torch.full((m + 3, 3 * HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    slot = buf[:m, HIDDEN : 2 * HIDDEN]
    got = run(slot)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"output {i}"
    rest = buf.clone()
    rest[:m, HIDDEN : 2 * HIDDEN] = 7.0
    assert bool((rest == 7.0).all()), "tap wrote outside its view"
    # The tap is the mixture of the updated row (bf16(layer + addend), output 1).
    expected = _mixture_cpp(got[1], t)
    _assert_fast_close(slot, expected, ("tap", mnnvl_order))
    a, e = slot.float(), expected.float()
    # One BF16 step of the expected value's binade (2^-7 of its leading power of two).
    ulp = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(2.0**-126))) - 7)
    print(
        f"TAPDIFF m={m} n={n} mx={mx} lamport={lamport} mnnvl_order={mnnvl_order} "
        f"max_bf16_steps={float(((a - e).abs() / ulp).max()):.3g} "
        f"differing={_mismatches(slot, expected)}/{slot.numel()}"
    )


@pytest.mark.parametrize("lamport", [False, True])
@pytest.mark.parametrize("n", [2, 6, 12])
@pytest.mark.parametrize("m", [14, 32])
def test_updated_into_bank_row(m: int, n: int, lamport: bool):
    """TRTLLM_K3LL_ATTN_RES_SNAPSHOT_INPLACE's call: ``block`` is rows ``[0, n - 1)`` of a snapshot bank and
    ``updated`` its row ``n - 1`` (the snapshot the step takes). Every output is the bits of the launch with a separate
    ``updated``, the row holds that ``updated``, and the rows the kernel reads are untouched."""
    if lamport:
        pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_lamport_ar")
    k3 = _k3()
    t = _inputs(m, n, 1.0, seed=13 * m + n)
    addend = t["addend"].clone()
    addend[addend == 0] = 0
    kwargs = dict(exact=False, mnnvl_order=True)
    if lamport:
        uc, flags, stage = _lamport_workspace(addend, dirty=1)
        stage.copy_(addend.view(torch.int32).flatten())
        addends = ()
        kwargs["lamport_addend"] = (uc.data_ptr(), flags.data_ptr())
    else:
        addends = (addend,)

    def run(block, updated):
        out = torch.empty_like(t["layer"])
        k3.run_k3_attn_res(t["layer"], addends, block, *_weights(t), out, updated, **kwargs)
        return out

    ref_updated = torch.empty_like(t["layer"])
    ref_out = run(t["block"], ref_updated)
    bank = torch.full((n + 1, m, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    bank[: n - 1].copy_(t["block"])
    before = bank[: n - 1].clone()
    out = run(bank[: n - 1], bank[n - 1])
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16)), "normed"
    assert torch.equal(bank[n - 1].view(torch.int16), ref_updated.view(torch.int16)), "updated row"
    assert torch.equal(bank[: n - 1].view(torch.int16), before.view(torch.int16)), (
        "read rows changed"
    )
    assert bool((bank[n] == 7.0).all()), "wrote past the row"


def test_tap_needs_vectorized_kernel():
    t = _inputs(2, 3, 1.0)
    out = torch.empty_like(t["layer"])
    with pytest.raises(ValueError):
        _k3().run_k3_attn_res(
            t["layer"],
            (),
            t["block"],
            *_weights(t),
            out,
            None,
            exact=True,
            tap=torch.empty_like(out),
        )

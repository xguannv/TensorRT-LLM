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
        from tensorrt_llm._torch.modules.kimi_k3_moe_residual import _fused_moe_residual_add

        updated = row = _fused_moe_residual_add(t["addend"], t["shared"], t["layer"])
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


@pytest.mark.parametrize("trigger", ["start", "wait", "end"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_trigger_placement(mode: int, trigger: str):
    """Where the fast kernel releases its dependents changes no bit: eagerly, and in a CUDA graph
    chain of three PDL launches that all trigger there, each consuming its predecessor's output as
    an input read after ``griddepcontrol.wait`` (the layer row without addends, else the first
    addend; the prefix sum and the snapshots are read before the wait)."""
    m, n = 14, 7
    t = _inputs(m, n, 1.0, seed=11 + mode)
    k3 = _k3()

    def run(prev, trig):
        layer = prev if mode == 0 else t["layer"]
        addends = ((), (prev,), (prev, t["shared"]))[mode]
        out = torch.empty_like(layer)
        updated = torch.empty_like(layer) if mode else None
        k3.run_k3_attn_res(
            layer, addends, t["block"], *_weights(t), out, updated, exact=False, trigger=trig
        )
        return updated, out

    ref = run(t["addend"], None)
    got = run(t["addend"], trigger)
    for a, b in zip(ref, got):
        if a is not None:
            assert _mismatches(b, a) == 0

    layer = torch.empty_like(t["layer"])
    outs = []

    def body():
        r1 = run(layer, trigger)
        r2 = run(r1[1], trigger)
        r3 = run(r2[1], trigger)
        outs[:] = [r1, r2, r3]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer.copy_(t["layer"])
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(100 + seed)
        layer.copy_(torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        r1 = run(layer, None)
        r2 = run(r1[1], None)
        r3 = run(r2[1], None)
        for i, ((u, o), (ru, ro)) in enumerate(zip(outs, (r1, r2, r3))):
            assert _mismatches(o, ro) == 0, f"replay {seed} stage {i} output"
            if ru is not None:
                assert _mismatches(u, ru) == 0, f"replay {seed} stage {i} updated prefix"


def _kernel_deps(graph, name_part: str):
    """Function names of the dependencies of every kernel node of ``graph`` (captured with
    ``keep_graph=True``) whose function name contains ``name_part``."""
    import cuda.bindings.driver as cu

    g = cu.CUgraph(graph.raw_cuda_graph())
    count = cu.cuGraphGetNodes(g, 0)[-1]
    nodes = cu.cuGraphGetNodes(g, count)[1]

    def name(node):
        err, kind = cu.cuGraphNodeGetType(node)
        if kind != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            return str(kind)
        _, params = cu.cuGraphKernelNodeGetParams(node)
        err, fname = cu.cuFuncGetName(params.func)
        return fname.decode() if err == cu.CUresult.CUDA_SUCCESS else "?"

    found = []
    for node in nodes:
        if name_part not in name(node):
            continue
        nd = cu.cuGraphNodeGetDependencies(node, 0)[-1]
        deps = cu.cuGraphNodeGetDependencies(node, nd)[1] if nd else []
        found.append([name(d) for d in deps])
    return found


@pytest.mark.parametrize("marker", [False, True])
def test_join_marker(marker: bool):
    """A side stream joined into a PDL attn_res launch (directly, or through ``run_join_marker``
    issued after the main stream's previous kernel): in a CUDA graph the consumer sees the side
    stream's late write, and with the marker it depends on the marker, not on the side kernel."""
    m, n = 14, 7
    t = _inputs(m, n, 1.0, seed=21)
    k3 = _k3()
    src = torch.empty_like(t["addend"])
    addend = torch.empty_like(t["addend"])
    side = torch.cuda.Stream()
    joined = torch.cuda.Event()
    fork = torch.cuda.Event()
    outs = []

    def body():
        fork.record()
        with torch.cuda.stream(side):
            fork.wait()
            torch.cuda._sleep(1_000_000)
            addend.copy_(src)
            if not marker:
                joined.record()
        # the main stream's own PDL kernel, captured after the side stream's work
        first = torch.empty_like(t["layer"])
        k3.run_k3_attn_res(t["layer"], (), t["block"], *_weights(t), first, None, exact=False)
        if marker:
            with torch.cuda.stream(side):
                k3.run_join_marker()
                joined.record()
        joined.wait()
        out = torch.empty_like(t["layer"])
        upd = torch.empty_like(t["layer"])
        k3.run_k3_attn_res(first, (addend,), t["block"], *_weights(t), out, upd, exact=False)
        outs[:] = [upd, out]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        src.copy_(t["addend"])
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        body()
    deps = _kernel_deps(graph, "K3AttnResFastKernel")
    consumer_deps = [d for d in deps if len(d) == 2]
    assert len(consumer_deps) == 1, deps
    assert any("K3JoinMarkerKernel" in d for d in consumer_deps[0]) == marker, consumer_deps
    for seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(200 + seed)
        src.copy_(torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16))
        addend.zero_()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        first = torch.empty_like(t["layer"])
        k3.run_k3_attn_res(t["layer"], (), t["block"], *_weights(t), first, None, exact=False)
        out = torch.empty_like(t["layer"])
        upd = torch.empty_like(t["layer"])
        k3.run_k3_attn_res(first, (src,), t["block"], *_weights(t), out, upd, exact=False)
        assert _mismatches(outs[0], upd) == 0, f"replay {seed} updated prefix"
        assert _mismatches(outs[1], out) == 0, f"replay {seed} output"

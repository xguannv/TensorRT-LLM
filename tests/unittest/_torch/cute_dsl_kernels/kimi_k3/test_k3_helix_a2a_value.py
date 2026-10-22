# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L0 per-op check and single-GPU timing of ``helix_a2a_value`` (experimental ``TRTLLM_K3LL_MK_ATTN``): the Helix
all-to-all receive + LSE combine + W_UV + output gate of one Kimi-K3 MLA layer in one kernel, against today's
staircase chain for the same layer:

    stats copy (the per-half (max, sum) expand)  ->  helix_a2a_fused (256-wide halves)  ->  k3_fp8_mla_output

One GPU stands in for the CP group: ``P`` = 16 receive segments in one local buffer; each source rank's records are
written by a send-only launch of the kernel under test (that rank's ``cp_rank``, its own partials), then rank ``RHO``
runs the receive-only launch. Partials are softmax averages of synthetic latent caches (channel scale from the layer's
real ``kv_a_layernorm``), with tokens empty on some ranks; W_UV / its block scales and the gate projection are the
layer's real FP8 weights when ``K3_CKPT`` names the NVFP4 checkpoint (else random FP8 weights).

Checks (rows 8 / 16 / 32, CUDA graph replays):
* the merged latent is bitwise today's combine output (``helix_a2a_fused``);
* the output against an fp64 reference (no intermediate rounding) and against today's chain (FP8 activation x FP8
  V-B), tolerances ``TOL_*`` below;
* replays reproduce the first replay bit for bit (Lamport re-arm, halves).

    pytest test_k3_helix_a2a_value.py
    python test_k3_helix_a2a_value.py --rows 8,16,32 --layers 3,47,92 --json out.json      # + timing table
"""

import argparse
import importlib.util
import json
import math
import os
import statistics

import pytest
import torch

P = 16  # CP size
HP = 6  # heads per rank
HEADS = P * HP
LATENT = 512
VDIM = 128
HIDDEN = 7168
GATE_COL = (
    1536 + 512 + 64
)  # the gate's first column in the fused in-projection output (q_a, kv_a, rope, gate)
AG_WIDTH = GATE_COL + HP * VDIM
RHO = 5  # the receiving rank
MLA_LAYERS = (3, 47, 92)
LOCAL_KV = 512  # synthetic latent rows per source rank
# Tolerances (relative L2 over the layer's [T, 768] output). Measured on GB300 (job in the 1022 retro): new vs fp64
# ~2e-3; today vs fp64 ~1-2e-2 (its FP8 activation quantization); new vs today the same as today vs fp64.
TOL_REF = 5e-3
TOL_TODAY = 5e-2

CKPT = os.environ.get(
    "K3_CKPT",
    "/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/nvidia_kimi-k3-nvfp4/hf/hf-1ec10fa_orig",
)


def _value_module():
    """``helix_a2a_value`` (``K3_MK_VALUE_PY`` = a path: load that file instead of the installed package's)."""
    path = os.environ.get("K3_MK_VALUE_PY")
    if not path:
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import helix_a2a_value

        return helix_a2a_value
    spec = importlib.util.spec_from_file_location(
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.helix_a2a_value", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mla_output_op():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (  # noqa: F401
        attention,
    )

    return torch.ops.trtllm.k3_fp8_mla_output


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _dequant_block(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    s = s.float().reshape(
        -(-w.shape[0] // 128), -(-w.shape[1] // 128)
    )  # checkpoint: [R / 128, 1, C / 128, 1]
    s = s.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]
    return w.float() * s


def load_layer(layer: int, rank: int = RHO):
    """This rank's W_UV ``[6, 128, 512]`` FP8, its block scales ``[6, 4]``, the gate projection rows ``[768, 7168]``
    (dequantized) and ``kv_a_layernorm`` of MLA layer ``layer``; random weights without the checkpoint."""
    heads = slice(HP * rank, HP * rank + HP)
    if CKPT and os.path.isdir(CKPT):
        from safetensors import safe_open

        with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
            index = json.load(f)["weight_map"]
        prefix = f"language_model.model.layers.{layer}.self_attn."
        names = ("kv_b_proj.weight", "kv_b_proj.weight_scale", "g_proj.weight", "g_proj.weight_scale",
                 "kv_a_layernorm.weight")  # fmt: skip
        t = {}
        for name in names:
            with safe_open(os.path.join(CKPT, index[prefix + name]), "pt", device="cuda") as f:
                t[name] = f.get_tensor(prefix + name)
        kvb, kvs = t["kv_b_proj.weight"], t["kv_b_proj.weight_scale"].float()
        assert kvb.dtype == torch.float8_e4m3fn and tuple(kvb.shape) == (HEADS * 256, LATENT), (
            kvb.shape
        )
        w_uv = kvb.view(HEADS, 256, LATENT)[heads, VDIM:].contiguous()
        w_scale = kvs.reshape(HEADS, 2, LATENT // 128)[heads, 1].contiguous()
        rows = slice(HP * VDIM * rank, HP * VDIM * (rank + 1))
        g = _dequant_block(t["g_proj.weight"], t["g_proj.weight_scale"])[rows]
        return w_uv, w_scale, g, t["kv_a_layernorm.weight"].float(), True
    gen = torch.Generator(device="cuda").manual_seed(1000 + layer)
    w_uv = (torch.randn(HP, VDIM, LATENT, generator=gen, device="cuda") * 64).to(
        torch.float8_e4m3fn
    )
    w_scale = torch.rand(HP, LATENT // 128, generator=gen, device="cuda") * 1e-3 + 1e-4
    g = torch.randn(HP * VDIM, HIDDEN, generator=gen, device="cuda") * 0.01
    return w_uv, w_scale, g, torch.ones(LATENT, device="cuda"), False


def load_w_uv_all(layer: int):
    """W_UV of all 96 heads, dequantized FP32 ``[96, 128, 512]`` (M1-o's V-up before the exchange)."""
    if CKPT and os.path.isdir(CKPT):
        from safetensors import safe_open

        with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
            index = json.load(f)["weight_map"]
        prefix = f"language_model.model.layers.{layer}.self_attn."
        t = {}
        for name in ("kv_b_proj.weight", "kv_b_proj.weight_scale"):
            with safe_open(os.path.join(CKPT, index[prefix + name]), "pt", device="cuda") as f:
                t[name] = f.get_tensor(prefix + name)
        w = _dequant_block(t["kv_b_proj.weight"], t["kv_b_proj.weight_scale"])
        return w.view(HEADS, 256, LATENT)[:, VDIM:].contiguous()
    return torch.cat([_dequant_rank(*load_layer(layer, r)[:2]) for r in range(P)])


def _dequant_rank(w_uv, w_scale):
    return w_uv.float() * w_scale.float().repeat_interleave(128, 1)[:, None, :]


def make_partials(rows: int, kv_norm: torch.Tensor, seed: int):
    """Per source rank: normalized partial outputs ``[T, 96 * 512]`` BF16 and ``(max, sum)`` ``[T, 96, 2]`` of a
    softmax over a synthetic latent cache, and the zero-KV mask ``[T]`` (rows empty on that rank: NaN partials,
    garbage statistics, as the split read leaves them)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for s in range(P):
        cache = torch.randn(LOCAL_KV, LATENT, generator=gen, device="cuda") * kv_norm
        logits = torch.randn(rows, HEADS, LOCAL_KV, generator=gen, device="cuda") * 2.5
        logits += (
            torch.randn(rows, HEADS, 1, generator=gen, device="cuda") * 3.0
        )  # per-source level
        mx = logits.amax(-1)
        p = torch.exp(logits - mx[..., None])
        total = p.sum(-1)
        po = (p / total[..., None]) @ cache
        stats = torch.stack((mx, total), -1).contiguous()
        mask = torch.zeros(rows, dtype=torch.bool, device="cuda")
        if s in (3, 11):  # a few tokens with no keys on these ranks
            mask[s % rows] = True
            mask[(s * 7 + 1) % rows] = True
        po = po.to(torch.bfloat16).reshape(rows, -1)
        po[mask] = float("nan")
        stats[mask] = 1.0e30
        out.append((po.contiguous(), stats, mask))
    return out


def make_gate(rows: int, g: torch.Tensor, seed: int) -> torch.Tensor:
    """``[T, AG_WIDTH]`` BF16 in-projection output; the gate columns are x @ g_proj^T of a random hidden state."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 77)
    x = torch.randn(rows, HIDDEN, generator=gen, device="cuda")
    ag = torch.randn(rows, AG_WIDTH, generator=gen, device="cuda").to(torch.bfloat16)
    ag[:, GATE_COL:] = (x @ g.t()).to(torch.bfloat16)
    return ag


def reference(srcs, rho, w_uv, w_scale, ag):
    """fp64 combine of rank ``rho``'s heads + W_UV (dequantized) + sigmoid gate, no intermediate rounding."""
    heads = slice(HP * rho, HP * rho + HP)
    o = torch.stack(
        [s[0].view(-1, HEADS, LATENT)[:, heads] for s in srcs]
    ).double()  # [P, T, 6, 512]
    st = torch.stack([s[1][:, heads] for s in srcs]).double()
    mask = torch.stack([s[2] for s in srcs])[..., None]  # [P, T, 1]
    mx = torch.where(mask, -math.inf, st[..., 0])
    w = torch.where(mask, 0.0, st[..., 1]) * torch.exp(mx - mx.amax(0))
    w = w / w.sum(0)
    merged = (w[..., None] * torch.nan_to_num(o, nan=0.0)).sum(0)  # [T, 6, 512]
    wd = w_uv.double() * w_scale.double().repeat_interleave(128, 1)[:, None, :]
    proj = torch.einsum("tak,ank->tan", merged, wd)
    gate = torch.sigmoid(ag[:, GATE_COL:].double().view(-1, HP, VDIM))
    return (proj * gate).reshape(-1, HP * VDIM), merged


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max())


# ---------------------------------------------------------------------------
# The single-GPU CP group
# ---------------------------------------------------------------------------


class Group:
    """Receive segments of ``P`` ranks in one local buffer (``rank_stride`` apart, two halves each, armed) and one
    launch counter per sending rank plus one for the receiver: every round runs each counter once, so all agree on
    the half."""

    def __init__(self, rows: int, record: int):
        self.half = -(-rows * P * record // 16) * 16
        self.stride = 2 * self.half
        self.buf = torch.full((P * self.stride,), 255, dtype=torch.uint8, device="cuda")
        self.addr = self.buf.data_ptr()
        self.state = torch.zeros(P + 1, 8, dtype=torch.int64, device="cuda")
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count


class Today:
    """stats copy -> ``helix_a2a_fused`` (halves) -> ``k3_fp8_mla_output``."""

    name = "today"

    def __init__(self, rows, srcs, w_uv, w_scale, ag):
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import helix_a2a_fused as fused

        self.f = fused
        self.rows, self.srcs, self.ag = rows, srcs, ag
        self.w_uv, self.w_scale = w_uv, w_scale.view(HP, 1, LATENT // 128)
        self.grp = Group(rows, fused.record_bytes(2 * HP, 256))
        self.halves = [self.stats_copy(s[1]) for s in srcs]
        self.latent = torch.empty(rows, HP * LATENT, dtype=torch.bfloat16, device="cuda")
        self.out = None
        self.mla_output = _mla_output_op()

    def stats_copy(self, stats):
        return (
            stats.view(self.rows, HEADS, 1, 2).expand(self.rows, HEADS, 2, 2).reshape(self.rows, -1)
        )

    def _launch(self, r, po, halves, mask, out, send):
        g = self.grp
        self.f.helix_a2a_fused(
            po, halves, mask, out, cp_rank=r, cp_size=P, num_heads=2 * HP, head_dim=256,
            ws_addr=g.addr, rank_stride=g.stride, half_bytes=g.half, state=g.state[r if send else P],
            max_send_ctas=2 * g.sms if send else 0, max_recv_ctas=0 if send else 2 * g.sms, use_pdl=True,
        )  # fmt: skip

    def send(self, r):
        po, _, mask = self.srcs[r]
        self._launch(r, po, self.halves[r], mask, self.latent, True)

    def receive(self):
        po, stats, mask = self.srcs[RHO]
        self.copied = self.stats_copy(stats)  # the per-rank copy of today's path (timed here)
        self._launch(RHO, po, self.halves[RHO], mask, self.latent, False)
        self.out = self.mla_output(self.latent, self.w_uv, self.w_scale, self.ag, GATE_COL)
        return self.out


class New:
    """``helix_a2a_value``."""

    name = "new"

    def __init__(self, rows, srcs, w_uv, w_scale, ag, k_split=4, debug=True):
        self.m = _value_module()
        self.rows, self.srcs, self.ag = rows, srcs, ag
        self.w_uv, self.w_scale, self.k_split = w_uv, w_scale, k_split
        self.grp = Group(rows, self.m.record_bytes(HP))
        self.out = torch.empty(rows, HP * VDIM, dtype=torch.bfloat16, device="cuda")
        self.latent = (
            torch.empty(rows, HP * LATENT, dtype=torch.bfloat16, device="cuda") if debug else None
        )
        self.name = f"new_ks{k_split}"

    def _launch(self, r, send):
        g = self.grp
        po, stats, mask = self.srcs[r]
        self.m.helix_a2a_value(
            po, stats, mask, self.ag[:, GATE_COL:], self.w_uv, self.w_scale, self.out,
            cp_rank=r, cp_size=P, num_heads=HP, ws_addr=g.addr, rank_stride=g.stride, half_bytes=g.half,
            state=g.state[r if send else P], max_send_ctas=2 * g.sms if send else 0, receive=not send,
            k_split=self.k_split, use_pdl=True, latent_out=None if send else self.latent,
        )  # fmt: skip

    def send(self, r):
        self._launch(r, True)

    def receive(self):
        self._launch(RHO, False)
        return self.out


class M1o:
    """M1-o's expected receive side: V-up before the exchange (each source's partials of all 96 heads projected to
    128 wide, FP32 then one BF16 rounding, standing in for a V-up in the split read's reducer), ``helix_a2a_fused``
    at 128 wide (one ``(max, sum)`` per head, no statistics copy), then the sigmoid gate (one Triton kernel; zero if
    M1-o folds it into the combine)."""

    name = "m1o"

    def __init__(self, rows, srcs, w_uv_all, ag):
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import helix_a2a_fused as fused

        self.f = fused
        self.rows, self.ag = rows, ag
        self.srcs = []
        for po, stats, mask in srcs:
            lat = po.view(rows, HEADS, LATENT).float()
            v = torch.einsum("tak,ank->tan", lat, w_uv_all).to(torch.bfloat16).reshape(rows, -1)
            self.srcs.append((v.contiguous(), stats, mask))
        self.grp = Group(rows, fused.record_bytes(HP, VDIM))
        self.combined = torch.empty(rows, HP * VDIM, dtype=torch.bfloat16, device="cuda")
        self.out = torch.empty(rows, HP * VDIM, dtype=torch.bfloat16, device="cuda")

    def _launch(self, r, send):
        g = self.grp
        po, stats, mask = self.srcs[r]
        self.f.helix_a2a_fused(
            po, stats, mask, self.combined, cp_rank=r, cp_size=P, num_heads=HP, head_dim=VDIM,
            ws_addr=g.addr, rank_stride=g.stride, half_bytes=g.half, state=g.state[r if send else P],
            max_send_ctas=2 * g.sms if send else 0, max_recv_ctas=0 if send else 2 * g.sms, use_pdl=True,
        )  # fmt: skip

    def send(self, r):
        self._launch(r, True)

    def receive(self):
        self._launch(RHO, False)
        _gate(self.combined, self.ag[:, GATE_COL:], self.out)
        return self.out


def _gate(x, gate, out):
    import triton
    import triton.language as tl

    global _GATE_KERNEL
    if _GATE_KERNEL is None:

        @triton.jit
        def kernel(X, G, OUT, GS, N: tl.constexpr, B: tl.constexpr):
            row = tl.program_id(0)
            k = tl.arange(0, B)
            m = k < N
            v = tl.load(X + row * N + k, m, 0.0).to(tl.float32)
            g = tl.load(G + row * GS + k, m, 0.0).to(tl.float32)
            tl.store(OUT + row * N + k, (v * tl.sigmoid(g)).to(tl.bfloat16), m)

        _GATE_KERNEL = kernel
    n = x.shape[1]
    _GATE_KERNEL[(x.shape[0],)](x, gate, out, gate.stride(0), n, triton.next_power_of_2(n))


_GATE_KERNEL = None


def round_once(impl):
    for r in range(P):
        impl.send(r)
    return impl.receive()


class Graphs:
    """Two graphs per implementation: ``setup`` (an optional L2 flush, then the 16 ranks' sends) and ``measure``
    (rank RHO's receive side). Replayed back to back."""

    def __init__(self, impl, flush: bool):
        self.impl = impl
        self.flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda") if flush else None
        for _ in range(3):  # warm up / compile, even number of rounds keeps the halves in step
            self._setup()
            impl.receive()
        torch.cuda.synchronize()
        self.setup = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.setup):
            self._setup()
        self.measure = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.measure):
            impl.receive()
        torch.cuda.synchronize()

    def _setup(self):
        if self.flush is not None:
            self.flush.add_(1)
        for r in range(P):
            self.impl.send(r)

    def replay(self, events=None):
        self.setup.replay()
        if events:
            events[0].record()
        self.measure.replay()
        if events:
            events[1].record()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def check(rows: int, layer: int, k_split: int = 4, replays: int = 4) -> dict:
    w_uv, w_scale, g, kv_norm, real = load_layer(layer)
    srcs = make_partials(rows, kv_norm, seed=rows * 1000 + layer)
    ag = make_gate(rows, g, seed=rows * 1000 + layer)
    ref, ref_latent = reference(srcs, RHO, w_uv, w_scale, ag)
    today, new = Today(rows, srcs, w_uv, w_scale, ag), New(rows, srcs, w_uv, w_scale, ag, k_split)
    res = {"rows": rows, "layer": layer, "real_weights": real, "k_split": k_split}
    out_t = round_once(today).clone()
    out_n = round_once(new).clone()
    res["latent_bitwise_today"] = bool(torch.equal(new.latent, today.latent))
    res["latent_vs_ref_rel_l2"] = rel_l2(new.latent, ref_latent.reshape(rows, -1))
    res["new_vs_ref_rel_l2"] = rel_l2(out_n, ref)
    res["new_vs_ref_max_rel"] = max_rel(out_n, ref)
    res["today_vs_ref_rel_l2"] = rel_l2(out_t, ref)
    res["today_vs_ref_max_rel"] = max_rel(out_t, ref)
    res["new_vs_today_rel_l2"] = rel_l2(out_n, out_t)
    res["new_vs_today_max_rel"] = max_rel(out_n, out_t)
    # graph replays reproduce the eager round bit for bit
    gr = Graphs(new, flush=False)
    same = True
    for _ in range(replays):
        new.out.zero_()
        gr.replay()
        torch.cuda.synchronize()
        same &= bool(torch.equal(new.out, out_n))
    res["graph_replays_bitwise"] = same
    m1o = M1o(rows, srcs, load_w_uv_all(layer), ag)
    out_m = round_once(m1o).clone()
    res["m1o_vs_ref_rel_l2"] = rel_l2(out_m, ref)
    res["m1o_vs_ref_max_rel"] = max_rel(out_m, ref)
    return res


@pytest.mark.parametrize("rows", [8, 16, 32])
@pytest.mark.parametrize("k_split", [4, 8])
def test_helix_a2a_value(rows, k_split):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 / SM103 only")
    r = check(rows, MLA_LAYERS[0], k_split)
    assert r["latent_bitwise_today"], r
    assert r["new_vs_ref_rel_l2"] < TOL_REF, r
    assert r["new_vs_today_rel_l2"] < TOL_TODAY, r
    assert r["graph_replays_bitwise"], r


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_impl(impl, flush: bool, iters: int) -> dict:
    """Kernel durations (CUPTI, per kernel position in a round) and the receive graph's wall time (events)."""
    gr = Graphs(impl, flush)
    for _ in range(20):
        gr.replay()
    torch.cuda.synchronize()
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for e in ev:
        gr.replay(e)
    torch.cuda.synchronize()
    wall = statistics.mean(
        a.elapsed_time(b) * 1e3 for a, b in ev
    )  # (event times are quantized: the mean)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            gr.replay()
        torch.cuda.synchronize()
    kernels = sorted(
        (
            e
            for e in prof.events()
            if e.device_type.name == "CUDA"
            and e.name
            and "Memcpy" not in e.name
            and "Memset" not in e.name
        ),
        key=lambda e: e.time_range.start,
    )
    per = len(kernels) // iters
    assert per * iters == len(kernels), (len(kernels), iters)
    slots: dict = {}
    for i, e in enumerate(kernels):
        slots.setdefault(i % per, []).append((e.name, e.time_range.end - e.time_range.start))
    rows = []
    for k in sorted(slots):
        names = {n for n, _ in slots[k]}
        rows.append((sorted(names)[0][:60], statistics.median(d for _, d in slots[k])))
    return {"wall_receive_graph_us": wall, "kernels": rows}


def label(impl_name: str, kernels, flush: bool):
    """Names of a round's kernel positions: [flush], 16 sends, then the receive side."""
    base = 1 if flush else 0
    out = {}
    for i, (name, us) in enumerate(kernels):
        if i < base:
            key = "flush"
        elif i < base + P:
            key = f"send_r{i - base}"
        else:
            key = {
                "today": ["stats_copy", "a2a_receive", "mla_output"],
                "m1o": ["a2a128_receive", "gate"],
            }.get(impl_name, ["value_receive"])[i - base - P]
        out[key] = {"us": round(us, 3), "kernel": name}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="8,16,32")
    ap.add_argument("--layers", default=",".join(map(str, MLA_LAYERS)))
    ap.add_argument("--ks", default="4,8")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--json", default="")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    print(
        f"[mk1] device {torch.cuda.get_device_name()} sms {torch.cuda.get_device_properties(0).multi_processor_count}"
    )
    results = {"checks": [], "timing": []}
    rows_list = [int(x) for x in args.rows.split(",")]
    for rows in rows_list:
        for layer in (int(x) for x in args.layers.split(",")):
            for ks in (int(x) for x in args.ks.split(",")):
                r = check(rows, layer, ks)
                results["checks"].append(r)
                print("[mk1 check]", json.dumps(r))
    if not args.no_timing:
        layer = int(args.layers.split(",")[0])
        w_uv, w_scale, g, kv_norm, _ = load_layer(layer)
        w_uv_all = load_w_uv_all(layer)
        for rows in rows_list:
            srcs = make_partials(rows, kv_norm, seed=rows)
            ag = make_gate(rows, g, seed=rows)
            impls = [Today(rows, srcs, w_uv, w_scale, ag), M1o(rows, srcs, w_uv_all, ag)]
            impls += [
                New(rows, srcs, w_uv, w_scale, ag, int(ks), debug=False)
                for ks in args.ks.split(",")
            ]
            for flush in (False, True):
                for impl in impls:
                    t = time_impl(impl, flush, args.iters)
                    lab = label(impl.name, t["kernels"], flush)
                    rec = {"rows": rows, "impl": impl.name, "l2_flush": flush,
                           "wall_receive_graph_us": round(t["wall_receive_graph_us"], 3), "kernels": lab}  # fmt: skip
                    results["timing"].append(rec)
                    recv = {
                        k: v["us"]
                        for k, v in lab.items()
                        if not k.startswith("send_r") and k != "flush"
                    }
                    print(f"[mk1 time] rows {rows} {impl.name:9s} flush {int(flush)} recv-graph wall "
                          f"{t['wall_receive_graph_us']:.2f} us  kernels {recv}  send_r{RHO} "
                          f"{lab[f'send_r{RHO}']['us']}")  # fmt: skip
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

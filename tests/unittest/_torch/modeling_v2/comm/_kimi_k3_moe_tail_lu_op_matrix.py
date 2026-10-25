# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_K3LL_MOE_TAIL_LU (``kimi_k3_nvfp4/moe_tail_lu.py``): a wide decode step's MoE tail with the full latent
all-reduce + RMSNorm on its own workspace, the replicated FP8 latent_up on its MXFP8 form, the shared expert's
all-reduce on a side stream and the attn_res kernel's MoE add, at W ranks (default 4) with the TP16 per-rank shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_moe_tail_lu_op_matrix.py [--world-size 4]

Not a pytest module: one fixed sequence of checks inside one W-rank job (the cases share the collective state). The
collected entry point is ``test_modeling_v2_kimi_k3_moe_tail_lu_op_matrix.py``.

What runs: the TP group's ``K3DecodeComm`` (as post_load_weights builds it), the decode GEMVs' state with the shared
site, ``K3TailLuState.create`` and ``K3TailLuLayer.create`` on a stand-in MoE (a replicated bf16 latent_up and latent
norm, this rank's shared down shard), ``moe_tail_lu.warm_up`` as ``_build_decode_moe_lowm`` runs it. Every rank draws
every rank's inputs from per-rank seeds, so each computes the references of the sums itself.

Checks:
  * build: the warm-up compiled the latent all-reduce, the shared push for both tiles, the MXFP8 GEMM for 9..32 rows
    and the MoE-add attn_res kernel for every candidate count; the weight is the stand-in's latent_up quantized
    (dequantized within E4M3's rounding) and the scale DeepGEMM's packed layout.
  * takes: per switch, token count (9..32 only), a missing state / layer / GEMV state.
  * pieces (9, 16, 17, 24, 32 rows, eager): the latent all-reduce + RMSNorm against the fp32 sum and RMSNorm of the
    ranks' rows (exact sums), its MXFP8 form bit for bit ``mxfp8_quantize_reference`` of its bf16 output; latent_up
    against ``F.linear`` of the dequantized MXFP8 latent and FP8 weight (fp32), and against the bf16 latent_up of the
    bf16 latent within the FP8 bound; the shared all-reduce equal to the sum of the ranks' plain GEMV partials in rank
    order; ``run`` bit for bit the pieces; the consumer's ``tail_lu_attn_res``: ``updated`` bit for bit
    ``bf16(prefix + bf16(routed + shared))``, ``normed`` bit for bit the same kernel without the add on that row, and
    the fp32 reference within the fast kernel's bound; every output the same on every rank.
  * decoder layer (16 and 32 rows, a snapshotting layer and not): ``KimiLinearDecoderLayer.forward`` with the
    ``LuTail`` handed on runs ``tail_lu_attn_res`` (the attention's input its ``normed``), and with that refused the
    torch add + fused attn_res (``updated`` the same bits; the census key counted once per fused call).
  * graphs (16, 24, 32 rows): two engaged layers with the decode workspace's other users between them
    (``allreduce_attn_res`` two-shot) captured with multi-stream on (the shared expert on the side stream), replayed
    alternately with new inputs, one rank late on every other replay, with eager steps in between: bit for bit the
    eager run (in order on one stream), the other users' outputs their eager values.
"""

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "TRTLLM_K3LL_MOE_TAIL_LU requires CUDA devices"

DEADLINE_S = 1800
H = 7168
L = 3584  # the latent
S_ACT = 384  # the shared expert's width per rank at TP16
K_O = 768
SNAPSHOTS = 3
ATTN_RES_BLOCK = 4
RES_EPS = 1e-5
LAT_EPS = 1e-6
ROWS = (9, 16, 17, 24, 32)
GRAPH_ROWS = (16, 24, 32)
REPLAY_ORDER = (16, 32, 24, 16, 24, 32)
# Fast attn_res kernel vs the fp32 reference (as the kernel's own tests): max |diff| relative to the output's max.
NORMED_REL = 2e-2
# The RMSNorm'd latent vs fp32: a bf16 rounding or two.
LATENT_REL = 1e-2
# latent_up from the MXFP8 latent vs F.linear of the dequantized operands: fp32 sums in another order, bf16 output.
LU_REL = 1e-2
# latent_up from the MXFP8 latent and FP8 weight vs the bf16 latent_up of the bf16 latent: E4M3's rounding of both
# operands (at most 2^-4 relative each, averaging out over 3584 terms).
LU_FP8_REL = 1e-1

R = None
IMPL = None
DC = None
DG = None
LU = None
FLAGS = None
CENSUS = None
MSU = None
BUILT = None
STATS = {}


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0, offset=0.0):
    return (offset + scale * torch.randn(tuple(shape), generator=g, device="cuda")).bfloat16()


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(t.shape[0], -1)


def _expect_same(where: str, got: torch.Tensor, want: torch.Tensor, replicated: bool = False):
    same = got.shape == want.shape and torch.equal(_bytes(got), _bytes(want))
    if not same:
        differ = int((_bytes(got) != _bytes(want)).sum()) if got.shape == want.shape else -1
        raise AssertionError(
            f"{where}: not bit for bit ({differ} bytes differ, shapes {got.shape} {want.shape})"
        )
    if replicated:
        assert R.same_on_ranks(got), f"{where}: the ranks differ"


def _expect_close(where: str, got: torch.Tensor, want: torch.Tensor, rel: float):
    err = ls.rel_err(got, want)
    key = where.split(": ", 1)[-1]
    STATS[key] = max(STATS.get(key, 0.0), err)
    assert err <= rel, f"{where}: relative error {err:.3e} > {rel:.1e}"


@contextlib.contextmanager
def _switch(value: int):
    old = FLAGS.TRTLLM_K3LL_MOE_TAIL_LU
    FLAGS.TRTLLM_K3LL_MOE_TAIL_LU = int(value)
    try:
        yield
    finally:
        FLAGS.TRTLLM_K3LL_MOE_TAIL_LU = old


def _census(key: str, mode: str) -> int:
    return CENSUS.counts().get(f"{key}:{mode}", 0)


def _mx_dequant(xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    m, k = xq.shape
    scale = torch.exp2(xs.float() - 127.0)
    return (xq.float().view(m, k // 32, 32) * scale[..., None]).view(m, k)


# ----------------------------------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------------------------------
def _shared_down(rank: int) -> torch.Tensor:
    return _normal(_gen(100 + rank), (H, S_ACT), 0.03)


def check_build():
    """The decode state, the switch's state and this rank's layer weights, as the target builds them."""
    global BUILT
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if R.world not in (4, 8, 16):
        raise AssertionError(f"the MNNVL ops run 4, 8 or 16 ranks, not {R.world}")
    comm = DC.K3DecodeComm.create(R.mapping, None)
    gemvs = DG.K3DecodeGemvs.create(sites=(LU.SHARED_SITE,), device=torch.device("cuda"))
    g = _gen(7)
    with torch.device("cuda"):
        moe = SimpleNamespace(
            routed_expert_up_proj=nn.Linear(L, H, bias=False, dtype=torch.bfloat16),
            routed_expert_norm=RMSNorm(hidden_size=L, eps=LAT_EPS, dtype=torch.bfloat16),
            shared_experts=SimpleNamespace(down_proj=SimpleNamespace(weight=_shared_down(R.rank))),
        )
    with torch.no_grad():
        moe.routed_expert_up_proj.weight.copy_(_normal(g, (H, L), 0.02))
        # Some 128 x 128 blocks of another magnitude, so the block scales differ.
        moe.routed_expert_up_proj.weight[:128, :128] *= 16
        moe.routed_expert_up_proj.weight[1024:1152, 2048:2176] *= 1 / 64
        moe.routed_expert_norm.weight.copy_(_normal(g, (L,), 0.1, 1.0))
    assert LU.layer_gaps(moe, R.world) == [], LU.layer_gaps(moe, R.world)
    state = LU.K3TailLuState.create(R.mapping, comm.oproj_push.lamport)
    layer = LU.K3TailLuLayer.create(moe)
    assert state.latent.world == R.world and state.latent.rank == R.rank
    assert state.latent.flags is state.latent_ws.buffer_flags
    assert state.push.flags is comm.mnnvl.buffer_flags, (
        "the shared all-reduce must use the decode workspace"
    )
    # The FP8 weight: E4M3 codes and power-of-two block scales within E4M3's rounding of the bf16 weight; the packed
    # scale DeepGEMM's layout of those scales.
    w = moe.routed_expert_up_proj.weight.float()
    codes, scale = LU.quantize_block_fp8(moe.routed_expert_up_proj.weight)
    assert torch.equal(_bytes(codes), _bytes(layer.up_weight))
    assert bool((torch.log2(scale) == torch.round(torch.log2(scale))).all()), (
        "scales are not powers of two"
    )
    deq = LU.dequantize_block_fp8(codes, scale)
    assert bool(((deq - w).abs() <= w.abs() * 2.0**-4 + scale.repeat_interleave(128, 0).repeat_interleave(
        128, 1) * 2.0**-9).all()), "the FP8 weight is not E4M3's rounding of the bf16 weight"  # fmt: skip
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    want_scale = transform_sf_into_required_layout(
        scale, mn=H, k=L, recipe=(1, 128, 128), is_sfa=False
    )
    assert torch.equal(layer.up_scale.cpu(), want_scale.cpu()), "packed scale layout"
    assert layer.up_scale.stride() == (1, H), layer.up_scale.stride()
    assert lowm.can_run(
        torch.zeros(9, L, dtype=torch.bfloat16, device="cuda"), layer.up_weight, layer.up_scale
    )
    LU.warm_up(state, layer, gemvs, SNAPSHOTS + 1)
    assert state.ready, "the warm-up did not compile the collectives"
    assert state.mx_rows == frozenset(range(LU.MIN_ROWS, LU.MAX_ROWS + 1)), sorted(state.mx_rows)
    assert state.attn_res_candidates == frozenset(range(2, SNAPSHOTS + 2)), (
        state.attn_res_candidates
    )
    with torch.device("cuda"):
        res = SimpleNamespace(
            sa_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            mlp_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            sa_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            mlp_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            in_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
            post_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
        )
    g = _gen(7000)
    with torch.no_grad():
        for proj in (res.sa_proj, res.mlp_proj):
            proj.weight.copy_(_normal(g, proj.weight.shape, 0.05))
        for norm in (res.sa_norm, res.mlp_norm, res.in_norm, res.post_norm):
            norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
    torch.cuda.synchronize()
    BUILT = SimpleNamespace(comm=comm, gemvs=gemvs, state=state, layer=layer, moe=moe, res=res,
                            up_bf16=moe.routed_expert_up_proj.weight.detach())  # fmt: skip
    if R.rank == 0:
        print(
            f"[rank 0] built at world {R.world}: {layer.extra_bytes} bytes of FP8 latent_up",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------
class _Call:
    """One engaged layer's inputs for every rank (drawn from per-rank seeds): the routed experts' latent partial and
    the shared activation; this rank's in persistent buffers. The consumer's prefix sum and snapshot bank (the same on
    every rank)."""

    def __init__(self, seed: int, rows: int):
        self.rows = rows
        empty = dict(dtype=torch.bfloat16, device="cuda")
        self.routed = torch.empty(rows, L, **empty)
        self.act = torch.empty(rows, S_ACT, **empty)
        self.prefix = torch.empty(rows, H, **empty)
        self.bank = torch.empty(SNAPSHOTS + 1, rows, H, **empty)
        self.draw(seed)

    def inputs(self, seed: int, rank: int):
        g = _gen(seed * 64 + rank + 1)
        # Small integers times a power of two: the ranks' sums are exact in fp32 and bf16.
        routed = ls.exact_bf16(g, (self.rows, L), -8, 8, 0.125)
        act = _normal(g, (self.rows, S_ACT), 0.5)
        return routed, act

    def draw(self, seed: int):
        self.seed = seed
        routed, act = self.inputs(seed, R.rank)
        self.routed.copy_(routed)
        self.act.copy_(act)
        g = _gen(seed)
        self.prefix.copy_(_normal(g, self.prefix.shape))
        self.bank.copy_(_normal(g, self.bank.shape))

    def references(self):
        """``(latent_sum, shared)``: the fp32 sum of every rank's routed rows, and the shared all-reduce as the C++
        one-shot order sums the ranks' plain GEMV partials (fp32, rank order, one bf16 rounding)."""
        latent = torch.zeros(self.rows, L, device="cuda")
        shared = torch.zeros(self.rows, H, device="cuda")
        for r in range(R.world):
            routed, act = self.inputs(self.seed, r)
            latent += routed.float()
            partial = BUILT.gemvs.project(LU.SHARED_SITE, act, _shared_down(r))
            assert partial is not None, "the shared site's plain GEMV did not take the call"
            shared += partial.float()
        return latent, shared.bfloat16()


def _lat_norm_ref(latent_sum: torch.Tensor) -> torch.Tensor:
    x = latent_sum.bfloat16().float()
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + LAT_EPS)
    return ((x * r).bfloat16().float() * BUILT.moe.routed_expert_norm.weight.float()).bfloat16()


# ----------------------------------------------------------------------------------------------------------------------
# takes
# ----------------------------------------------------------------------------------------------------------------------
def check_takes():
    st, layer, gemvs = BUILT.state, BUILT.layer, BUILT.gemvs
    for rows in (8, 9, 16, 17, 24, 32, 33, 48):
        with _switch(0):
            assert not LU.takes(st, layer, rows, S_ACT, gemvs), f"switch 0 took {rows} rows"
        with _switch(1):
            want = LU.MIN_ROWS <= rows <= LU.MAX_ROWS
            assert LU.takes(st, layer, rows, S_ACT, gemvs) == want, f"switch 1 at {rows} rows"
            assert not LU.takes(None, layer, rows, S_ACT, gemvs)
            assert not LU.takes(st, None, rows, S_ACT, gemvs)
            assert not LU.takes(st, layer, rows, S_ACT, None)
    with _switch(1):
        assert not LU.takes(st, layer, 16, S_ACT + 128, gemvs), (
            "took a shared activation of another width"
        )
        # A capture whose shared site's plain wide GEMV has not run eagerly: refused, as the plain path refuses it.
        fresh = DG.K3DecodeGemvs()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            refused = not LU.takes(st, layer, 16, S_ACT, fresh)
        assert refused, "took a capture whose shared site's GEMV had not run eagerly"
        del graph


# ----------------------------------------------------------------------------------------------------------------------
# Pieces and the consumer
# ----------------------------------------------------------------------------------------------------------------------
def _consumer(tail, call):
    res = BUILT.res
    return BUILT.comm.tail_lu_attn_res(
        tail, call.prefix, call.bank[:SNAPSHOTS], res.sa_proj, res.sa_norm, res.in_norm
    )


def check_pieces():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res, k3_fp8w_lowm_mxfp8

    st, layer, gemvs, res = BUILT.state, BUILT.layer, BUILT.gemvs, BUILT.res
    for rows in ROWS:
        where = f"{rows} rows"
        call = _Call(300 + rows, rows)
        latent_sum, shared_ref = call.references()
        latent, xq, xs = LU.latent_all_reduce(st, layer, call.routed)
        _expect_close(f"{where}: latent RMSNorm", latent, _lat_norm_ref(latent_sum), LATENT_REL)
        rq, rs = k3_fp8w_lowm_mxfp8.mxfp8_quantize_reference(latent)
        _expect_same(f"{where}: MXFP8 codes", xq, rq, replicated=True)
        _expect_same(f"{where}: MXFP8 scales", xs, rs, replicated=True)
        _expect_same(f"{where}: latent", latent, latent, replicated=True)
        up = LU.latent_up(layer, xq, xs)
        w_deq = _mx_dequant(layer.up_weight, _w_exponents(layer))
        _expect_close(f"{where}: latent_up vs F.linear (dequantized)", up, F.linear(_mx_dequant(xq, xs), w_deq),
                      LU_REL)  # fmt: skip
        _expect_close(f"{where}: latent_up vs bf16 latent_up", up,
                      F.linear(latent.float(), BUILT.up_bf16.float()), LU_FP8_REL)  # fmt: skip
        shared = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
        LU.shared_all_reduce(st, gemvs, layer.shared_down, call.act, shared)
        assert torch.equal(shared.float(), shared_ref.float()), (
            f"{where}: shared all-reduce vs the rank-order sum"
        )
        _expect_same(f"{where}: shared", shared, shared, replicated=True)
        with _switch(1):
            tail = LU.run(st, layer, gemvs, call.act, lambda c=call: c.routed)
        _expect_same(f"{where}: run routed vs pieces", tail.routed, up, replicated=True)
        _expect_same(f"{where}: run shared vs pieces", tail.shared, shared)
        assert tail.candidates == st.attn_res_candidates
        assert BUILT.comm.takes_tail_lu(tail, call.prefix, call.bank[:SNAPSHOTS], res.sa_proj, res.sa_norm,
                                        res.in_norm), f"{where}: the consumer refused"  # fmt: skip
        normed, updated = _consumer(tail, call)
        want_updated = call.prefix + (tail.routed + tail.shared)
        _expect_same(f"{where}: updated", updated, want_updated, replicated=True)
        plain = torch.empty_like(normed)
        k3_attn_res.run_k3_attn_res(
            want_updated, (), call.bank[:SNAPSHOTS], res.sa_proj.weight.reshape(-1), res.sa_norm.weight,
            res.in_norm.weight, RES_EPS, RES_EPS, plain, None, exact=False, mnnvl_order=True,
        )  # fmt: skip
        _expect_same(
            f"{where}: normed vs the kernel on the updated row", normed, plain, replicated=True
        )
        ref = ls.residual_update_ref(updated, call.bank[:SNAPSHOTS], res.sa_proj.weight.reshape(-1),
                                     res.sa_norm.weight, RES_EPS, res.in_norm.weight, RES_EPS)  # fmt: skip
        _expect_close(f"{where}: normed vs fp32", normed, ref, NORMED_REL)
        # The zero prefix (a layer that has just snapshotted).
        normed0, updated0 = BUILT.comm.tail_lu_attn_res(
            tail, None, call.bank[:SNAPSHOTS], res.sa_proj, res.sa_norm, res.in_norm
        )
        _expect_same(f"{where}: updated without prefix", updated0, tail.routed + tail.shared)
    if R.rank == 0:
        print(
            "[rank 0] pieces: latent AR + RMSNorm + MXFP8, FP8 latent_up, shared AR, MoE-add attn_res; max rel err "
            + ", ".join(f"{k} {v:.2e}" for k, v in sorted(STATS.items())),
            flush=True,
        )


def _w_exponents(layer) -> torch.Tensor:
    """The UE8M0 exponent of every weight element's 128 x 128 block, as ``[N, K / 32]`` (the MXFP8 dequant's
    layout), read back from the packed scale."""
    words = layer.up_scale.contiguous()  # [N, K / 512] int32, four K blocks per word
    blocks = words.view(torch.uint8).view(H, -1)[:, : L // 128]
    return blocks.repeat_interleave(4, dim=1)


# ----------------------------------------------------------------------------------------------------------------------
# The decoder layer's pre-attention step
# ----------------------------------------------------------------------------------------------------------------------
class _Attention:
    def __init__(self, partial: torch.Tensor):
        self.partial, self.seen = partial, None

    def will_run_decode_branch(self, attn_metadata, step) -> bool:
        return True

    def __call__(
        self, hidden_states, attn_metadata, step=None, reduce_output=True, project_output=True
    ):
        self.seen = hidden_states
        assert project_output and not reduce_output
        return self.partial


class _Recorder:
    def __init__(self, out):
        self.out, self.seen = out, None

    def __call__(self, hidden_states, *args, **kwargs):
        self.seen = hidden_states
        return self.out


def _decoder(call: _Call, snapshot_layer: bool):
    res = BUILT.res
    partial = _normal(_gen(call.seed * 64 + R.rank + 33), (call.rows, H), 0.25)
    attention = _Attention(partial)
    layer = SimpleNamespace(
        layer_idx=ATTN_RES_BLOCK if snapshot_layer else 1, attn_res_block_size=ATTN_RES_BLOCK,
        decode_comm=BUILT.comm, sandwich_oproj=False, is_kda=True, linear_attn=attention, self_attn=None,
        self_attention_res_proj=res.sa_proj, self_attention_res_norm=res.sa_norm, input_layernorm=res.in_norm,
        mlp_res_proj=res.mlp_proj, mlp_res_norm=res.mlp_norm, post_attention_layernorm=res.post_norm,
        is_moe=True, block_sparse_moe=_Recorder(torch.zeros(call.rows, H, dtype=torch.bfloat16, device="cuda")),
        attention=attention,
    )  # fmt: skip
    for name in ("_o_proj", "_takes_oproj_push", "_oproj_push_attn_res"):
        setattr(layer, name, types.MethodType(getattr(IMPL.KimiLinearDecoderLayer, name), layer))
    return layer


def _decode_step(rows: int):
    return IMPL.DecodeStep(rows, max(1, rows // 8), min(rows, 8))


def check_decoder():
    st, layer, gemvs = BUILT.state, BUILT.layer, BUILT.gemvs
    comm = BUILT.comm
    for rows in (16, 32):
        for snapshot_layer in (False, True):
            where = f"decoder {rows} rows snapshot {snapshot_layer}"
            call = _Call(500 + rows + int(snapshot_layer), rows)
            with _switch(1):
                tail = LU.run(st, layer, gemvs, call.act, lambda c=call: c.routed)
            want_normed, want_updated = (t.clone() for t in _consumer(tail, call))
            outs = {}
            for fused in (True, False):
                dec = _decoder(call, snapshot_layer)
                call.draw(call.seed)
                before = _census("dispatch:moe_tail_lu_attn_res", "eager")
                if not fused:
                    comm.takes_tail_lu = lambda *a, **k: False
                try:
                    # TRTLLM_K3LL_OPROJ_PUSH off: the post-attention step is the two-shot all-reduce of the partial.
                    old = FLAGS.TRTLLM_K3LL_OPROJ_PUSH
                    FLAGS.TRTLLM_K3LL_OPROJ_PUSH = 0
                    prefix, snapshots, _ = IMPL.KimiLinearDecoderLayer.forward(
                        dec, call.prefix, call.bank, SNAPSHOTS, None, _decode_step(rows),
                        pending_moe_partial=tail, defer_moe_tail=True,
                    )  # fmt: skip
                finally:
                    FLAGS.TRTLLM_K3LL_OPROJ_PUSH = old
                    comm.__dict__.pop("takes_tail_lu", None)
                assert snapshots == SNAPSHOTS + int(snapshot_layer)
                if CENSUS.ENABLED:
                    assert _census("dispatch:moe_tail_lu_attn_res", "eager") - before == int(
                        fused
                    ), where
                outs[fused] = SimpleNamespace(attn_in=dec.attention.seen.clone(), prefix=prefix.clone(),
                                              snap=call.bank[SNAPSHOTS].clone())  # fmt: skip
            _expect_same(f"{where}: attention input = tail_lu_attn_res normed", outs[True].attn_in, want_normed,
                         replicated=True)  # fmt: skip
            if snapshot_layer:
                _expect_same(f"{where}: snapshot row = updated", outs[True].snap, want_updated)
                _expect_same(f"{where}: fallback snapshot row", outs[False].snap, want_updated)
            _expect_close(f"{where}: fallback normed", outs[False].attn_in, want_normed, NORMED_REL)
            _expect_same(f"{where}: fallback output prefix", outs[False].prefix, outs[True].prefix)
    if R.rank == 0:
        print(
            "[rank 0] decoder layer: LuTail through tail_lu_attn_res, and the fallback", flush=True
        )


# ----------------------------------------------------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------------------------------------------------
class _Step:
    """Two engaged layers with a two-shot ``allreduce_attn_res`` on the decode workspace between them."""

    def __init__(self, rows: int, seed: int):
        self.rows = rows
        self.a, self.b, self.ar = _Call(seed, rows), _Call(seed + 1, rows), _Call(seed + 2, rows)
        self.ar_partial = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
        self.draw(seed)

    def draw(self, seed: int):
        for i, call in enumerate((self.a, self.b, self.ar)):
            call.draw(seed + 17 * i)
        self.ar_partial.copy_(_normal(_gen(seed * 64 + R.rank + 7), self.ar_partial.shape, 0.25))

    def run(self):
        st, layer, gemvs, res = BUILT.state, BUILT.layer, BUILT.gemvs, BUILT.res
        out = SimpleNamespace()
        with _switch(1):
            ta = LU.run(st, layer, gemvs, self.a.act, lambda: self.a.routed)
            out.a_normed, out.a_updated = _consumer(ta, self.a)
            out.ar_normed, out.ar_updated = BUILT.comm.allreduce_attn_res(
                self.ar_partial, self.ar.prefix, self.ar.bank[:SNAPSHOTS], res.mlp_proj, res.mlp_norm,
                res.post_norm, two_shot=True,
            )  # fmt: skip
            tb = LU.run(st, layer, gemvs, self.b.act, lambda: self.b.routed)
            out.b_normed, out.b_updated = _consumer(tb, self.b)
            out.b_routed, out.b_shared = tb.routed, tb.shared
        return out


def _cloned(ns):
    return {k: v.clone() for k, v in vars(ns).items()}


def check_graphs():
    steps = {rows: _Step(rows, 600 + 10 * rows) for rows in GRAPH_ROWS}
    for step in steps.values():
        step.run()
    torch.cuda.synchronize()
    graphs = {}
    # Where the shared all-reduce runs: the side stream under multi-stream capture, the current stream eagerly.
    streams = []
    original = LU.shared_all_reduce

    def tapped(*args, **kwargs):
        streams.append(torch.cuda.current_stream().cuda_stream)
        return original(*args, **kwargs)

    LU.shared_all_reduce = tapped
    try:
        for rows, step in steps.items():
            R.barrier()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with MSU.with_multi_stream(True), torch.cuda.graph(graph, stream=stream):
                out = step.run()
            R.barrier()
            graphs[rows] = (graph, out)
            assert streams == [BUILT.state.stream.cuda_stream] * 2, (
                "the captured shared all-reduces' streams"
            )
            streams.clear()
        step.run()
        current = torch.cuda.current_stream().cuda_stream
        assert streams == [current] * 2, "eagerly the shared all-reduce runs on the current stream"
    finally:
        LU.shared_all_reduce = original
    for i, rows in enumerate(REPLAY_ORDER):
        step = steps[rows]
        where = f"graph {rows} rows replay {i}"
        step.draw(700 + 13 * i)
        graph, out = graphs[rows]
        R.barrier()
        R.late(i % R.world if i % 2 else None)
        graph.replay()
        torch.cuda.synchronize()
        got = _cloned(out)
        eager = _cloned(step.run())
        for name, value in got.items():
            _expect_same(f"{where}: graph vs eager {name}", value, eager[name], replicated=True)
        # An eager engaged layer of another token count between replays.
        extra = _Call(800 + i, ROWS[i % len(ROWS)])
        with _switch(1):
            tail = LU.run(BUILT.state, BUILT.layer, BUILT.gemvs, extra.act, lambda: extra.routed)
        _expect_same(f"{where}: eager {extra.rows} rows", tail.routed, tail.routed, replicated=True)
    del graphs
    torch.cuda.synchronize()
    if R.rank == 0:
        print(
            f"[rank 0] graphs {GRAPH_ROWS} (side stream) replayed {len(REPLAY_ORDER)}x = eager",
            flush=True,
        )


def check_pre_clear():
    """TRTLLM_K3LL_LATENT_AR_PRE_CLEAR on: the latent all-reduce clears the previous call's buffer before its PDL wait;
    the pieces (the all-reduce against its fp32 reference) and the graphs (replays with a late rank, bit for bit the
    eager run) again, and the switch counted in both modes."""
    old = FLAGS.TRTLLM_K3LL_LATENT_AR_PRE_CLEAR
    FLAGS.TRTLLM_K3LL_LATENT_AR_PRE_CLEAR = 1
    try:
        eager0, capture0 = (
            _census("dispatch:latent_ar_pre_clear", "eager"),
            _census("dispatch:latent_ar_pre_clear", "capture"),
        )
        check_pieces()
        check_graphs()
        assert _census("dispatch:latent_ar_pre_clear", "eager") > eager0, (
            "pre_clear: eager calls not counted"
        )
        assert _census("dispatch:latent_ar_pre_clear", "capture") > capture0, (
            "pre_clear: captured calls not counted"
        )
    finally:
        FLAGS.TRTLLM_K3LL_LATENT_AR_PRE_CLEAR = old


CHECKS = [check_takes, check_pieces, check_decoder, check_graphs, check_pre_clear]


def _run_one_rank(args) -> int:
    global R, IMPL, DC, DG, LU, FLAGS, CENSUS, MSU
    R = ls.Rank(args)
    import importlib

    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    IMPL = importlib.import_module(f"{package}.implementation")
    DC = importlib.import_module(f"{package}.decode_comm")
    DG = importlib.import_module(f"{package}.decode_gemv")
    LU = importlib.import_module(f"{package}.moe_tail_lu")
    FLAGS = importlib.import_module(f"{package}.flags")
    # the matrix covers MOE_TAIL_LU's whole 9-32 row range; the product default caps it at 16 rows
    FLAGS.TRTLLM_K3LL_MOE_TAIL_LU_MAX_ROWS = 32
    CENSUS = importlib.import_module(f"{package}.census")
    MSU = importlib.import_module("tensorrt_llm._torch.modules.multi_stream_utils")
    CENSUS.ENABLED = True
    code = ls.run_checks(R, [check_build])
    if code:
        return code
    with torch.inference_mode():
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print("[rank 0] MOE_TAIL_LU checks passed", flush=True)
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

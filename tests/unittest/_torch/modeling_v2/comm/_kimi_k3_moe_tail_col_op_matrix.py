# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_K3LL_MOE_TAIL_COL (``kimi_k3_nvfp4/moe_tail_col.py``): a wide decode step's MoE tail with TAIL_LU's full
latent all-reduce + RMSNorm and side-stream shared all-reduce, then the column-sharded BF16 latent_up publishing each
rank's ``7168 / W`` output columns (plus the shared output's) into the decode workspace's Lamport rows, which the
consumer's attn_res kernel polls with no finish kernel; at W ranks (default 4) with the TP16 per-rank shared width.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_moe_tail_col_op_matrix.py [--world-size 4]

Not a pytest module: one fixed sequence of checks inside one W-rank job (the cases share the collective state). The
collected entry point is ``test_modeling_v2_kimi_k3_moe_tail_col_op_matrix.py``.

What runs: the TP group's ``K3DecodeComm`` (as post_load_weights builds it), the decode GEMVs' state with the shared
site, ``K3TailLuState.create`` + ``K3TailColState`` and ``K3TailColLayer.create`` on a stand-in MoE whose latent norm
is folded first (``decode_moe.fold_latent_norm``, as ``_build_decode_moe_lowm`` does before ``_build_tail_col``), and
``moe_tail_col.warm_up``. Every rank draws every rank's inputs from per-rank seeds, so each computes the references
itself.

Error bound (lossless: BF16 weight, fp32 sums). Each published element is ``bf16(sum_k latent[t, k] up[c, k] +
shared[t, c])`` of the bf16 operands; against the float64 value ``ref`` of the same operands it is within
``2^-8 |ref| + 2^-10 (sum_k |latent[t, k] up[c, k]| + |shared[t, c]|)``: the bf16 output rounding (half an ulp,
2^-9 relative, doubled for margin) and the fp32 accumulation over 3584 products in the MMA's split-K partials (at most
3584 roundings of 2^-24 relative each, 2^-12.2 of the absolute sum, quadrupled).

Checks:
  * build: the warm-up ran the shared all-reduce, the latent all-reduce, the publish at both token tiles and the
    Lamport attn_res kernel for every candidate count; the slice is rows ``[W r, W r + W)`` of the folded latent_up.
  * takes: per switch, token count (9..32 only), a missing state / layer / GEMV state, a capture whose shared site's
    GEMV did not run eagerly.
  * pieces (9, 16, 17, 24, 32 rows, eager): the latent all-reduce + RMSNorm against the fp32 reference; the shared
    all-reduce equal to the rank-order sum of the plain GEMV partials; ``run``'s published rows
    (``ColTail.materialize``) within the error bound of the float64 reference and within BF16 tolerance of the bf16
    replicated latent_up + shared (the switch-off arithmetic's operands); every rank the same bits; the consumer's
    ``tail_col_attn_res``: ``updated`` bit for bit ``bf16(prefix + rows)``, ``normed`` bit for bit the same kernel
    on that row, and within the fast kernel's bound of the fp32 reference; the zero prefix.
  * decoder layer (16 and 32 rows, a snapshotting layer and not): ``KimiLinearDecoderLayer.forward`` with the
    ``ColTail`` handed on runs ``tail_col_attn_res`` (the attention's input its ``normed``); with that refused the
    rows are copied out and the torch add + fused attn_res runs (``updated`` the same bits; census key counted once
    per fused call).
  * graphs (16, 24, 28, 32 rows): two engaged layers with the decode workspace's other users between them
    (``allreduce_attn_res`` two-shot), captured with multi-stream on (the shared expert on the side stream), replayed
    6 times alternately with new inputs (the Lamport rotation goes round twice per replay), one rank late on every
    other replay, with eager steps in between: bit for bit the eager run, the other users' outputs their eager values.
  * latent all-reduce re-arm CTAs (``TRTLLM_K3LL_LAR_CLEAR_SPLIT`` off and on): eager at every row count, bit for
    bit the plain kernel; a chain of 5 back-to-back latent all-reduces (no other kernel between them: the re-arm CTAs
    of one call and the push of the next are as close as PDL lets them) captured once per value and replayed 6 times
    with new inputs, the two graphs replayed one after the other on the same workspace (so a re-arm by one variant
    precedes the push of the other), one rank late on every other replay: every output bit for bit the plain eager
    chain; the dispatch census counts the switch once per captured call; and the two-layer step graphs (16 / 32 rows)
    captured with the switch on, replayed between the plain eager steps: bit for bit.
  * publish addend early (``TRTLLM_K3LL_MOE_TAIL_COL_ADDEND_EARLY`` off and on): ``run``'s published rows bit for bit
    at every row count, the dispatch census counted once per engaged call; the two-layer step graphs (16 / 24 / 32
    rows) captured with the switch on and replayed with new inputs, one rank late on every other replay: bit for bit
    the plain eager step.
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

assert torch.cuda.is_available(), "TRTLLM_K3LL_MOE_TAIL_COL requires CUDA devices"

DEADLINE_S = 1800
H = 7168
L = 3584  # the latent
S_ACT = 384  # the shared expert's width per rank at TP16
SNAPSHOTS = 3
ATTN_RES_BLOCK = 4
RES_EPS = 1e-5
LAT_EPS = 1e-6
ROWS = (9, 16, 17, 24, 28, 32)  # 28: MTP6 bs4
GRAPH_ROWS = (16, 24, 28, 32)
REPLAY_ORDER = (16, 32, 24, 28, 16, 24, 32, 28)
NORMED_REL = 2e-2  # fast attn_res kernel vs fp32 (as the kernel's own tests)
LATENT_REL = 1e-2  # the RMSNorm'd latent vs fp32: a bf16 rounding or two
OUT_ROUND = 2.0**-8  # the published element's bf16 rounding, relative (module docstring)
OUT_SUM = 2.0**-10  # its fp32 accumulation, relative to the sum of |terms| (module docstring)
OUT_REL = 1e-2  # against the bf16 replicated latent_up + shared, max |diff| / max |ref|

R = None
IMPL = None
DC = None
DG = None
DM = None
LU = None
COL = None
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


def _expect_bound(where: str, got: torch.Tensor, ref: torch.Tensor, terms: torch.Tensor):
    """``|got - ref| <= OUT_ROUND |ref| + OUT_SUM terms`` everywhere (float64 ``ref`` and ``terms``)."""
    diff = (got.double() - ref).abs()
    allowed = OUT_ROUND * ref.abs() + OUT_SUM * terms + 1e-30
    ratio = float((diff / allowed).max())
    STATS["published rows / error bound"] = max(
        STATS.get("published rows / error bound", 0.0), ratio
    )
    assert ratio <= 1.0, f"{where}: outside the error bound ({ratio:.3f} of it)"


@contextlib.contextmanager
def _switch(value: int):
    old = FLAGS.TRTLLM_K3LL_MOE_TAIL_COL
    FLAGS.TRTLLM_K3LL_MOE_TAIL_COL = int(value)
    try:
        yield
    finally:
        FLAGS.TRTLLM_K3LL_MOE_TAIL_COL = old


def _census(key: str, mode: str) -> int:
    return CENSUS.counts().get(f"{key}:{mode}", 0)


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
        moe.routed_expert_norm.weight.copy_(_normal(g, (L,), 0.1, 1.0))
    DM.fold_latent_norm(moe)
    assert COL.layer_gaps(moe, R.world) == [], COL.layer_gaps(moe, R.world)
    state = COL.K3TailColState(LU.K3TailLuState.create(R.mapping, comm.oproj_push.lamport))
    layer = COL.K3TailColLayer.create(moe, R.rank, R.world)
    width = COL.slice_width(R.world)
    up = moe.routed_expert_up_proj.weight.detach()
    assert state.base.push is comm.oproj_push.lamport, "the publish must use the decode workspace"
    assert layer.col0 == R.rank * width and tuple(layer.up_slice.shape) == (width, L)
    assert torch.equal(layer.up_slice, up[layer.col0 : layer.col0 + width])
    assert layer.up_slice.data_ptr() != up.data_ptr(), "the slice is a copy"
    assert bool((layer.gamma == 1).all()), "the folded latent norm's weight is ones"
    assert layer.tiles == COL.row_tiles(width)
    COL.warm_up(state, layer, gemvs, SNAPSHOTS + 1)
    assert state.ready, "the warm-up did not run the publish"
    assert state.tiles == frozenset((16, 32)), sorted(state.tiles)
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
    BUILT = SimpleNamespace(
        comm=comm, gemvs=gemvs, state=state, layer=layer, moe=moe, res=res, up=up
    )
    if R.rank == 0:
        print(
            f"[rank 0] built at world {R.world}: [{width}, {L}] latent_up slice, {layer.tiles} row tiles, "
            f"{layer.extra_bytes} bytes",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------
class _Call:
    """One engaged layer's inputs for every rank (per-rank seeds): the routed experts' latent partial and the shared
    activation, this rank's in persistent buffers; the consumer's prefix sum and snapshot bank (the same on every
    rank)."""

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
    return ((x * r).bfloat16().float() * BUILT.layer.gamma.float()).bfloat16()


# ----------------------------------------------------------------------------------------------------------------------
# takes
# ----------------------------------------------------------------------------------------------------------------------
def check_takes():
    st, layer, gemvs = BUILT.state, BUILT.layer, BUILT.gemvs
    for rows in (8, 9, 16, 17, 24, 32, 33, 48):
        with _switch(0):
            assert not COL.takes(st, layer, rows, S_ACT, gemvs), f"switch 0 took {rows} rows"
        with _switch(1):
            want = LU.MIN_ROWS <= rows <= LU.MAX_ROWS
            assert COL.takes(st, layer, rows, S_ACT, gemvs) == want, f"switch 1 at {rows} rows"
            assert not COL.takes(None, layer, rows, S_ACT, gemvs)
            assert not COL.takes(st, None, rows, S_ACT, gemvs)
            assert not COL.takes(st, layer, rows, S_ACT, None)
    with _switch(1):
        assert not COL.takes(st, layer, 16, S_ACT + 128, gemvs), (
            "took a shared activation of another width"
        )
        fresh = DG.K3DecodeGemvs()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            refused = not COL.takes(st, layer, 16, S_ACT, fresh)
        assert refused, "took a capture whose shared site's GEMV had not run eagerly"
        del graph


# ----------------------------------------------------------------------------------------------------------------------
# Pieces and the consumer
# ----------------------------------------------------------------------------------------------------------------------
def _consumer(tail, call):
    res = BUILT.res
    return BUILT.comm.tail_col_attn_res(
        tail, call.prefix, call.bank[:SNAPSHOTS], res.sa_proj, res.sa_norm, res.in_norm
    )


def check_pieces():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    st, layer, gemvs, res = BUILT.state, BUILT.layer, BUILT.gemvs, BUILT.res
    for rows in ROWS:
        where = f"{rows} rows"
        call = _Call(300 + rows, rows)
        latent_sum, shared_ref = call.references()
        latent = COL.latent_all_reduce(st.base, layer, call.routed)
        _expect_close(f"{where}: latent RMSNorm", latent, _lat_norm_ref(latent_sum), LATENT_REL)
        _expect_same(f"{where}: latent", latent, latent, replicated=True)
        shared = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
        LU.shared_all_reduce(st.base, gemvs, layer.shared_down, call.act, shared)
        assert torch.equal(shared.float(), shared_ref.float()), (
            f"{where}: shared all-reduce vs the rank-order sum"
        )
        with _switch(1):
            tail = COL.run(st, layer, gemvs, call.act, lambda c=call: c.routed)
        assert isinstance(tail, COL.ColTail) and tail.rows == rows
        assert tail.lamport is BUILT.comm.oproj_push.lamport
        assert tail.candidates == st.attn_res_candidates
        out = tail.materialize()
        _expect_same(f"{where}: published rows", out, out, replicated=True)
        # The error bound against the float64 value of the same bf16 operands (every column: every rank's slice).
        up = BUILT.up.double()
        ref = latent.double() @ up.T + shared.double()
        terms = latent.double().abs() @ up.abs().T + shared.double().abs()
        _expect_bound(f"{where}: published rows vs float64", out, ref, terms)
        # Against the switch-off arithmetic's replicated form: bf16 latent_up, then the bf16 add.
        _expect_close(
            f"{where}: published rows vs bf16 latent_up + shared",
            out,
            F.linear(latent, BUILT.up) + shared,
            OUT_REL,
        )
        # The consumer: run again (the rows were read out by the gather; a new call leaves them in place).
        with _switch(1):
            tail = COL.run(st, layer, gemvs, call.act, lambda c=call: c.routed)
        assert BUILT.comm.takes_tail_col(tail, call.prefix, call.bank[:SNAPSHOTS], res.sa_proj, res.sa_norm,
                                         res.in_norm), f"{where}: the consumer refused"  # fmt: skip
        normed, updated = _consumer(tail, call)
        _expect_same(f"{where}: updated", updated, call.prefix + out, replicated=True)
        plain = torch.empty_like(normed)
        k3_attn_res.run_k3_attn_res(
            updated.clone(), (), call.bank[:SNAPSHOTS], res.sa_proj.weight.reshape(-1), res.sa_norm.weight,
            res.in_norm.weight, RES_EPS, RES_EPS, plain, None, exact=False, mnnvl_order=True,
        )  # fmt: skip
        _expect_same(
            f"{where}: normed vs the kernel on the updated row", normed, plain, replicated=True
        )
        ref_n = ls.residual_update_ref(updated, call.bank[:SNAPSHOTS], res.sa_proj.weight.reshape(-1),
                                       res.sa_norm.weight, RES_EPS, res.in_norm.weight, RES_EPS)  # fmt: skip
        _expect_close(f"{where}: normed vs fp32", normed, ref_n, NORMED_REL)
        # The zero prefix (a layer that has just snapshotted): the rows themselves; the same rows polled again.
        normed0, updated0 = BUILT.comm.tail_col_attn_res(
            tail, None, call.bank[:SNAPSHOTS], res.sa_proj, res.sa_norm, res.in_norm
        )
        _expect_same(f"{where}: updated without prefix", updated0, out)
    if R.rank == 0:
        print(
            "[rank 0] pieces: latent AR + RMSNorm, shared AR, column publish, Lamport attn_res; max "
            + ", ".join(f"{k} {v:.2e}" for k, v in sorted(STATS.items())),
            flush=True,
        )


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
                tail = COL.run(st, layer, gemvs, call.act, lambda c=call: c.routed)
            want_normed, want_updated = (t.clone() for t in _consumer(tail, call))
            outs = {}
            for fused in (True, False):
                dec = _decoder(call, snapshot_layer)
                call.draw(call.seed)
                with _switch(1):
                    tail = COL.run(st, layer, gemvs, call.act, lambda c=call: c.routed)
                before = _census("dispatch:moe_tail_col_attn_res", "eager")
                if not fused:
                    comm.takes_tail_col = lambda *a, **k: False
                try:
                    # TRTLLM_K3LL_OPROJ_PUSH off: the post-attention step is the two-shot all-reduce of the partial,
                    # the next call on the decode workspace after the rows were read.
                    old = FLAGS.TRTLLM_K3LL_OPROJ_PUSH
                    FLAGS.TRTLLM_K3LL_OPROJ_PUSH = 0
                    prefix, snapshots, _ = IMPL.KimiLinearDecoderLayer.forward(
                        dec, call.prefix, call.bank, SNAPSHOTS, None, _decode_step(rows),
                        pending_moe_partial=tail, defer_moe_tail=True,
                    )  # fmt: skip
                finally:
                    FLAGS.TRTLLM_K3LL_OPROJ_PUSH = old
                    comm.__dict__.pop("takes_tail_col", None)
                assert snapshots == SNAPSHOTS + int(snapshot_layer)
                if CENSUS.ENABLED:
                    assert _census("dispatch:moe_tail_col_attn_res", "eager") - before == int(
                        fused
                    ), where
                outs[fused] = SimpleNamespace(attn_in=dec.attention.seen.clone(), prefix=prefix.clone(),
                                              snap=call.bank[SNAPSHOTS].clone())  # fmt: skip
            _expect_same(f"{where}: attention input = tail_col_attn_res normed", outs[True].attn_in, want_normed,
                         replicated=True)  # fmt: skip
            if snapshot_layer:
                _expect_same(f"{where}: snapshot row = updated", outs[True].snap, want_updated)
                _expect_same(f"{where}: fallback snapshot row", outs[False].snap, want_updated)
            _expect_close(f"{where}: fallback normed", outs[False].attn_in, want_normed, NORMED_REL)
            _expect_same(f"{where}: fallback output prefix", outs[False].prefix, outs[True].prefix)
    if R.rank == 0:
        print(
            "[rank 0] decoder layer: ColTail through tail_col_attn_res, and the copied-out fallback",
            flush=True,
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
            ta = COL.run(st, layer, gemvs, self.a.act, lambda: self.a.routed)
            out.a_normed, out.a_updated = _consumer(ta, self.a)
            out.ar_normed, out.ar_updated = BUILT.comm.allreduce_attn_res(
                self.ar_partial, self.ar.prefix, self.ar.bank[:SNAPSHOTS], res.mlp_proj, res.mlp_norm,
                res.post_norm, two_shot=True,
            )  # fmt: skip
            tb = COL.run(st, layer, gemvs, self.b.act, lambda: self.b.routed)
            out.b_rows = tb.materialize()
            out.b_normed, out.b_updated = _consumer(tb, self.b)
        return out


def _cloned(ns):
    return {k: v.clone() for k, v in vars(ns).items()}


def check_graphs():
    steps = {rows: _Step(rows, 600 + 10 * rows) for rows in GRAPH_ROWS}
    for step in steps.values():
        step.run()
    torch.cuda.synchronize()
    graphs = {}
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
            assert streams == [BUILT.state.base.stream.cuda_stream] * 2, (
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
        # An eager engaged layer of another token count between replays, its rows read out.
        extra = _Call(800 + i, ROWS[i % len(ROWS)])
        with _switch(1):
            tail = COL.run(BUILT.state, BUILT.layer, BUILT.gemvs, extra.act, lambda: extra.routed)
        rows_out = tail.materialize()
        _expect_same(f"{where}: eager {extra.rows} rows", rows_out, rows_out, replicated=True)
    del graphs
    torch.cuda.synchronize()
    if R.rank == 0:
        print(
            f"[rank 0] graphs {GRAPH_ROWS} (side stream) replayed {len(REPLAY_ORDER)}x = eager",
            flush=True,
        )


LAR_VARIANTS = (0, 1)  # TRTLLM_K3LL_LAR_CLEAR_SPLIT
LAR_CHAIN = 5


@contextlib.contextmanager
def _lar(clear: int):
    old = FLAGS.TRTLLM_K3LL_LAR_CLEAR_SPLIT
    FLAGS.TRTLLM_K3LL_LAR_CLEAR_SPLIT = int(clear)
    try:
        yield
    finally:
        FLAGS.TRTLLM_K3LL_LAR_CLEAR_SPLIT = old


def check_lar_variants():
    st, layer = BUILT.state, BUILT.layer
    # Eager, every row count: each pair bit for bit the plain kernel (and so within check_pieces' reference bound).
    for rows in ROWS:
        call = _Call(900 + rows, rows)
        outs = {}
        for v in LAR_VARIANTS:
            with _lar(v):
                outs[v] = COL.latent_all_reduce(st.base, layer, call.routed).clone()
        for v in LAR_VARIANTS[1:]:
            _expect_same(
                f"latent AR {rows} rows eager switch {v}", outs[v], outs[0], replicated=True
            )
    # Back-to-back chains under graph capture, the pairs' graphs interleaved on the one workspace.
    for rows in (16, 24, 32):
        calls = [_Call(950 + 7 * k + rows, rows) for k in range(LAR_CHAIN)]
        graphs = {}
        for v in LAR_VARIANTS:
            torch.cuda.synchronize()
            R.barrier()
            before = _census("dispatch:lar_clear_split", "capture")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with _lar(v), torch.cuda.graph(graph, stream=stream):
                outs = [COL.latent_all_reduce(st.base, layer, c.routed) for c in calls]
            R.barrier()
            counted = _census("dispatch:lar_clear_split", "capture") - before
            assert counted == LAR_CHAIN * v, f"latent AR {rows} rows switch {v}: census {counted}"
            graphs[v] = (graph, outs)
        for i in range(6):
            for k, c in enumerate(calls):
                c.draw(1000 + 31 * i + k)
            got = {}
            for v in LAR_VARIANTS if i % 2 == 0 else LAR_VARIANTS[::-1]:
                graph, outs = graphs[v]
                R.barrier()
                R.late(i % R.world if i % 2 else None)
                graph.replay()
                torch.cuda.synchronize()
                got[v] = [o.clone() for o in outs]
            with _lar(0):
                eager = [COL.latent_all_reduce(st.base, layer, c.routed).clone() for c in calls]
            for v in LAR_VARIANTS:
                for k in range(LAR_CHAIN):
                    _expect_same(
                        f"latent AR {rows} rows chain replay {i} {v} call {k}",
                        got[v][k],
                        eager[k],
                        replicated=True,
                    )
        del graphs
    # The two-layer step (other users of the decode workspace between the layers, shared expert on the side stream)
    # captured with both switches on: bit for bit the plain eager step.
    for rows in (16, 32):
        step = _Step(rows, 1200 + rows)
        with _lar(1):
            step.run()
        torch.cuda.synchronize()
        R.barrier()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with _lar(1), MSU.with_multi_stream(True), torch.cuda.graph(graph, stream=stream):
            out = step.run()
        R.barrier()
        for i in range(4):
            step.draw(1300 + 11 * i)
            R.barrier()
            R.late(i % R.world if i % 2 else None)
            graph.replay()
            torch.cuda.synchronize()
            got = _cloned(out)
            with _lar(0):
                eager = _cloned(step.run())
            for name, value in got.items():
                _expect_same(
                    f"step {rows} rows replay {i} LAR on: {name}",
                    value,
                    eager[name],
                    replicated=True,
                )
        del graph
    torch.cuda.synchronize()
    if R.rank == 0:
        print(
            f"[rank 0] latent AR re-arm CTAs {LAR_VARIANTS}: eager, {LAR_CHAIN}-call chains and step graphs = plain",
            flush=True,
        )


@contextlib.contextmanager
def _early(value: int):
    old = FLAGS.TRTLLM_K3LL_MOE_TAIL_COL_ADDEND_EARLY
    FLAGS.TRTLLM_K3LL_MOE_TAIL_COL_ADDEND_EARLY = int(value)
    try:
        yield
    finally:
        FLAGS.TRTLLM_K3LL_MOE_TAIL_COL_ADDEND_EARLY = old


def check_addend_early():
    st, layer, gemvs = BUILT.state, BUILT.layer, BUILT.gemvs
    # Eager, every row count: the publish with its addend loaded after the grid wait publishes the same rows.
    for rows in ROWS:
        call = _Call(1400 + rows, rows)
        outs = {}
        for v in (0, 1):
            before = _census("dispatch:tail_col_addend_early", "eager")
            with _switch(1), _early(v):
                outs[v] = COL.run(st, layer, gemvs, call.act, lambda c=call: c.routed).materialize()
            counted = _census("dispatch:tail_col_addend_early", "eager") - before
            assert counted == v, f"addend early {rows} rows switch {v}: census {counted}"
        _expect_same(f"addend early {rows} rows eager", outs[1], outs[0], replicated=True)
    # The two-layer step (other users of the decode workspace between the layers, shared expert on the side stream)
    # captured with the switch on: bit for bit the plain eager step.
    for rows in (16, 24, 32):
        step = _Step(rows, 1500 + rows)
        with _early(1):
            step.run()
        torch.cuda.synchronize()
        R.barrier()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with _early(1), MSU.with_multi_stream(True), torch.cuda.graph(graph, stream=stream):
            out = step.run()
        R.barrier()
        for i in range(4):
            step.draw(1600 + 11 * i)
            R.barrier()
            R.late(i % R.world if i % 2 else None)
            graph.replay()
            torch.cuda.synchronize()
            got = _cloned(out)
            with _early(0):
                eager = _cloned(step.run())
            for name, value in got.items():
                _expect_same(
                    f"step {rows} rows replay {i} addend early: {name}",
                    value,
                    eager[name],
                    replicated=True,
                )
        del graph
    torch.cuda.synchronize()
    if R.rank == 0:
        print("[rank 0] publish addend early: eager rows and step graphs = plain", flush=True)


CHECKS = [
    check_takes,
    check_pieces,
    check_decoder,
    check_graphs,
    check_lar_variants,
    check_addend_early,
]


def _run_one_rank(args) -> int:
    global R, IMPL, DC, DG, DM, LU, COL, FLAGS, CENSUS, MSU
    R = ls.Rank(args)
    import importlib

    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    IMPL = importlib.import_module(f"{package}.implementation")
    DC = importlib.import_module(f"{package}.decode_comm")
    DG = importlib.import_module(f"{package}.decode_gemv")
    DM = importlib.import_module(f"{package}.decode_moe")
    LU = importlib.import_module(f"{package}.moe_tail_lu")
    COL = importlib.import_module(f"{package}.moe_tail_col")
    FLAGS = importlib.import_module(f"{package}.flags")
    CENSUS = importlib.import_module(f"{package}.census")
    MSU = importlib.import_module("tensorrt_llm._torch.modules.multi_stream_utils")
    CENSUS.ENABLED = True
    code = ls.run_checks(R, [check_build])
    if code:
        return code
    with torch.inference_mode():
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print("[rank 0] MOE_TAIL_COL checks passed", flush=True)
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

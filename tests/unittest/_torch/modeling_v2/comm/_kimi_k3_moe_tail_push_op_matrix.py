# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_K3LL_MOE_TAIL_PUSH (``kimi_k3_nvfp4/moe_tail_push.py``): a wide decode step's MoE tail GEMV pushing its
partial into the consumer's all-reduce, each arm against the other and the off arm against the chain it replaces, at W
ranks (default 4) with the TP16 per-rank tail shape [7168, 640].

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_moe_tail_push_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _kimi_k3_moe_tail_push_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (the arms share the decode path's collective
state). The collected entry point is ``test_modeling_v2_kimi_k3_moe_tail_push_op_matrix.py``.

What runs. ``KimiLinearDecoderLayer.forward`` itself on a stand-in layer (as the o_proj push matrix): the TP group's
``K3DecodeComm`` (``create``, as post_load_weights builds it), the decode GEMVs' state with the ``moe_tail`` site
(``K3DecodeGemvs.create``, then ``compile_push`` as ``_build_decode_moe_lowm`` runs it, whatever the switch), the
residual modules, an attention stand-in recording its input (the pre-attention ``normed``) and handing over a fixed
o_proj input, an FP8 o_proj stand-in on ``k3_fp8w_lowm_gemm`` (so ``TRTLLM_K3LL_OPROJ_PUSH`` runs too) and a MoE
stand-in. The previous layer's handed-on tail is what ``decode_moe_lowm``'s wide step returns: with the switch, a
``WideTail`` of this rank's tail input and weight (a row-parallel shard each); without it, ``_gemv(gemvs, "moe_tail",
...)``. Both switches are set on the flags module per arm (read at forward time), on one built state.

Checks:
  * build: the push GEMV compiled at load for the 16- and 32-token tiles, its Lamport view the workspace's.
  * takes: ``K3DecodeComm.takes_tail_push`` per switch, token count (9..32 on the wide GEMV; 33..64, where the plain
    tail is ``F.linear``, refused), step kind, ``TRTLLM_K3LL_WIDE_AR_ATTN_RES``, an empty bank, and a capture whose
    plain wide GEMV has not run eagerly.
  * eager (9, 16, 17, 24, 32, 33, 48, 64 rows; a layer that snapshots and one that does not; KDA and MLA; the four
    combinations of TAIL_PUSH x OPROJ_PUSH, the same ``WideTail`` objects handed to both tail arms): the off arm's
    pre-attention step bit for bit the pre-change chain (the tail GEMV, then ``allreduce_attn_res(two_shot=True)``);
    every arm bit for bit every other (pre-attention normed and updated, the bank's snapshot row, the post-attention
    normed, the layer's output prefix sum), the same bits on every rank; which collective ran is tapped. The final
    norm's consumer (``output_attn_res_*`` and the model norm) on and off, bit for bit. Census counts the switch's
    dispatch key.
  * graphs (16, 24, 32 rows): one step per token count captured three times on one built state (tail on / oproj on,
    tail off / oproj on, tail on / oproj off; the switches flipped between captures): two layers with the workspace's
    other users between them (``mnnvl_allgather_split``, ``allreduce_attn_res``) and the final norm's consumer. Replayed
    alternately with new inputs, one rank late on every other replay, with eager steps (both arms) in between: every
    graph's outputs bit for bit every other graph's and the eager arms', the other users' outputs their exact
    references.
  * ATTN_RES_MXFP8 (``TRTLLM_K3LL_ATTN_RES_MXFP8``, tail push and o_proj push on; 9, 16, 24, 32 rows; KDA with the fused
    ``[q | k | v | g | f_a | b]`` FP8 weight of the TP16 per-rank shape [3328, 7168], MLA with an FP8 fused q_a / kv_a
    module [2112, 7168], partial last row tile): the K3AttnResFast that finishes the pushed tail also writes the MXFP8
    form of its ``normed``, offered to the attention's first FP8 projection (``fp8w_lowm.gemm``), which then runs
    ``k3_fp8w_lowm_mxfp8``. On vs off: the pre-attention normed and updated, the projection's output and the layer's
    outputs bit for bit; the offered ``(xq, xs)`` equal ``mxfp8_quantize_reference(normed)``; the on arm's projection
    ran the MXFP8 GEMM (``fp8w_lowm.MX_STATS``) and the off arm's did not. A CUDA graph of the on arm (compiled
    eagerly first) replayed with new inputs gives the eager on arm's bits.
"""

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "TRTLLM_K3LL_MOE_TAIL_PUSH requires CUDA devices"

DEADLINE_S = 1800
H = 7168
K_TAIL = 640  # [latent up 224 | padding 32 | shared down 384] at TP16
PAD = (224, 256)  # the padding columns: zero in the weight and the input
K_O = 768  # o_proj's input per rank
SNAPSHOTS = 3
ATTN_RES_BLOCK = 4
RES_EPS = 1e-5
PUSH_ROWS = (9, 16, 17, 24, 32)
EAGER_ROWS = PUSH_ROWS + (33, 48, 64)
GRAPH_ROWS = (16, 24, 32)
GRAPH_MODES = ((1, 1), (0, 1), (1, 0))  # (TAIL_PUSH, OPROJ_PUSH) of each captured graph
REPLAY_ORDER = (16, 32, 24, 16, 24, 32, 32, 16, 24)
AG_BF16, AG_FP32 = 64, 16
CENSUS_KEY = "dispatch:moe_tail_push"

R = None
IMPL = None  # kimi_k3_nvfp4/implementation.py
DC = None  # decode_comm
DG = None  # decode_gemv
DM = None  # decode_moe
TP = None  # moe_tail_push
FLAGS = None
CENSUS = None
FW = None  # fp8w_lowm
MX = None  # k3_fp8w_lowm_mxfp8
BUILT = None
STATS = {"tail_push_calls": 0}


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0, offset=0.0):
    return (offset + scale * torch.randn(tuple(shape), generator=g, device="cuda")).bfloat16()


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(t.shape[0], -1)


def _ulp_note(got: torch.Tensor, want: torch.Tensor) -> str:
    if got.dtype != torch.bfloat16 or got.shape != want.shape:
        return ""
    a, e = got.float(), want.float()
    diff = (a - e).abs()
    ulp = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(2.0**-126))) - 7)
    return f"; {float((diff > 0).float().mean()):.4%} of the elements, max {float((diff / ulp).max()):.2f} ulp"


def _expect_same(where: str, got: torch.Tensor, want: torch.Tensor, replicated: bool = False):
    """``got`` bit for bit ``want``; with ``replicated``, also the same bits on every rank."""
    same = got.shape == want.shape and torch.equal(_bytes(got), _bytes(want))
    agree = R.same_on_ranks(got) if replicated else True
    if not same:
        differ = int((_bytes(got) != _bytes(want)).sum()) if got.shape == want.shape else -1
        raise AssertionError(
            f"{where}: not bit for bit ({differ} bytes differ, shapes {got.shape} {want.shape}"
            f"{_ulp_note(got, want)})"
        )
    assert agree, f"{where}: the ranks differ"


@contextlib.contextmanager
def _switches(tail: int, oproj: int, wide_ar: int = 1):
    """TRTLLM_K3LL_MOE_TAIL_PUSH, TRTLLM_K3LL_OPROJ_PUSH and TRTLLM_K3LL_WIDE_AR_ATTN_RES on the flags module (read at
    forward time) for the block."""
    names = ("TRTLLM_K3LL_MOE_TAIL_PUSH", "TRTLLM_K3LL_OPROJ_PUSH", "TRTLLM_K3LL_WIDE_AR_ATTN_RES")
    old = [getattr(FLAGS, n) for n in names]
    for n, v in zip(names, (tail, oproj, wide_ar)):
        setattr(FLAGS, n, int(v))
    try:
        yield
    finally:
        for n, v in zip(names, old):
            setattr(FLAGS, n, v)


@contextlib.contextmanager
def _tap():
    """Record the collectives the layer calls on the decode state, in order, with their outputs (instance attributes,
    removed on exit)."""
    comm = BUILT.comm
    seen = []
    names = ("tail_push_attn_res", "allreduce_attn_res", "oproj_push_attn_res")
    for name in names:

        def tapped(*args, _method=getattr(comm, name), _name=name, **kwargs):
            out = _method(*args, **kwargs)
            seen.append((_name, out))
            return out

        setattr(comm, name, tapped)
    try:
        yield seen
    finally:
        for name in names:
            comm.__dict__.pop(name, None)


def _decode_step(rows: int):
    """A pure decode step of ``rows`` tokens."""
    return IMPL.DecodeStep(rows, max(1, rows // 8), min(rows, 8))


def _flat(ns, prefix=""):
    out = {}
    for key, value in vars(ns).items():
        if isinstance(value, SimpleNamespace):
            out.update(_flat(value, f"{prefix}{key}."))
        elif value is not None:
            out[f"{prefix}{key}"] = value
    return out


def _cloned(ns):
    return {k: v.clone() for k, v in _flat(ns).items()}


def _census(mode: str) -> int:
    return CENSUS.counts().get(f"{CENSUS_KEY}:{mode}", 0)


# ----------------------------------------------------------------------------------------------------------------------
# Stand-ins
# ----------------------------------------------------------------------------------------------------------------------
class _Fp8OProj(nn.Module):
    """What ``fp8w_lowm.install`` leaves on a routed FP8 o_proj (see the o_proj push matrix)."""

    def __init__(self, seed: int):
        super().__init__()
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

        g = _gen(seed)
        w = (torch.randn(H, K_O, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        scale = torch.exp2(
            torch.randint(-8, -2, (H // 128, K_O // 128), generator=g, device="cuda").float()
        )
        weight, packed = lowm.prepare_k3_fp8w_lowm_weight(w, scale)
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", packed)
        self.bias = None
        self._k3ll_fp8w_lowm = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.k3_fp8w_lowm_gemm(x, self.weight, self.weight_scale)


class _Attention:
    """An attention module's decode branch: records its input (the pre-attention ``normed``), hands over the fixed
    o_proj input (``project_output=False``) or o_proj's partial of it (``reduce_output=False``)."""

    def __init__(self, core: torch.Tensor, o_proj: nn.Module):
        self.core, self.o_proj, self.seen = core, o_proj, None
        self.mixer = SimpleNamespace(o_proj=o_proj)

    def will_run_decode_branch(self, attn_metadata, step) -> bool:
        return True

    def __call__(
        self, hidden_states, attn_metadata, step=None, reduce_output=True, project_output=True
    ):
        self.seen = hidden_states
        if not project_output:
            return self.core
        assert not reduce_output, "the decode path asks for the unreduced partial"
        return self.o_proj(self.core)


class _Recorder:
    """The MoE stand-in: keeps its last input (the post-attention ``normed``), returns a fixed tensor."""

    def __init__(self, out):
        self.out, self.seen = out, None

    def __call__(self, hidden_states, *args, **kwargs):
        self.seen = hidden_states
        return self.out


class _Call:
    """One layer's step: this rank's tail input and o_proj input (its own), the prefix sum and the snapshot bank
    (``SNAPSHOTS`` valid rows, one spare; the same on every rank); rewritten in place for a replay."""

    def __init__(self, seed: int, rows: int, snapshot_layer: bool):
        self.rows, self.snapshot_layer, self.step = rows, snapshot_layer, _decode_step(rows)
        empty = dict(dtype=torch.bfloat16, device="cuda")
        self.tail_x = torch.empty(rows, K_TAIL, **empty)
        self.core = torch.empty(rows, K_O, **empty)
        self.prefix = torch.empty(rows, H, **empty)
        self.bank = torch.empty(SNAPSHOTS + 1, rows, H, **empty)
        self.moe_next = torch.zeros(rows, H, **empty)
        self.draw(seed)

    def draw(self, seed: int):
        mine = _gen(seed * 64 + R.rank + 1)
        self.tail_x.copy_(_normal(mine, self.tail_x.shape))
        self.tail_x[:, PAD[0] : PAD[1]] = 0
        self.core.copy_(_normal(mine, self.core.shape, 0.25))
        g = _gen(seed)
        self.prefix.copy_(_normal(g, self.prefix.shape))
        self.bank.copy_(_normal(g, self.bank.shape))

    def pending(self, tail: int):
        """What the previous MoE layer's wide step hands on: a ``WideTail`` with the switch, else the tail's
        partial (``decode_moe_lowm``'s ``_gemv(gemvs, "moe_tail", ...)``)."""
        if tail:
            return TP.WideTail(self.tail_x, BUILT.tail_w, BUILT.gemvs)
        return DM._gemv(BUILT.gemvs, TP.SITE, self.tail_x, BUILT.tail_w)


def _decoder(call: _Call, kind: str):
    """``KimiLinearDecoderLayer``'s attributes its decode forward reads, with its own methods bound."""
    res = BUILT.res
    attention = _Attention(call.core, BUILT.o_proj)
    layer = SimpleNamespace(
        layer_idx=ATTN_RES_BLOCK if call.snapshot_layer else 1, attn_res_block_size=ATTN_RES_BLOCK,
        decode_comm=BUILT.comm, sandwich_oproj=False, is_kda=kind == "kda",
        linear_attn=attention if kind == "kda" else None, self_attn=attention if kind == "mla" else None,
        self_attention_res_proj=res.sa_proj, self_attention_res_norm=res.sa_norm, input_layernorm=res.in_norm,
        mlp_res_proj=res.mlp_proj, mlp_res_norm=res.mlp_norm, post_attention_layernorm=res.post_norm,
        is_moe=True, block_sparse_moe=_Recorder(call.moe_next), attention=attention,
    )  # fmt: skip
    for name in ("_o_proj", "_takes_oproj_push", "_oproj_push_attn_res"):
        setattr(layer, name, types.MethodType(getattr(IMPL.KimiLinearDecoderLayer, name), layer))
    return layer


def _arm(layer, call: _Call, tail: int, oproj: int):
    """The layer's decode forward with the previous layer's tail handed on, at TAIL_PUSH = ``tail`` and OPROJ_PUSH =
    ``oproj``: the pre-attention ``normed`` (the attention's input), the bank's snapshot row (a snapshotting layer),
    the post-attention ``normed`` (the MoE's input) and the output prefix sum."""
    with _switches(tail, oproj):
        prefix, snapshots, _ = IMPL.KimiLinearDecoderLayer.forward(
            layer, call.prefix, call.bank, SNAPSHOTS, None, call.step,
            pending_moe_partial=call.pending(tail), defer_moe_tail=True,
        )  # fmt: skip
    assert snapshots == SNAPSHOTS + int(call.snapshot_layer), snapshots
    return SimpleNamespace(
        attn_in=layer.attention.seen,
        snapshot=call.bank[SNAPSHOTS] if call.snapshot_layer else None,
        moe_in=layer.block_sparse_moe.seen,
        prefix=prefix,
    )


def _final(call: _Call, tail: int):
    """The final norm's consumer of the last MoE layer's handed-on tail (``KimiLinearModel.forward`` after the
    layers): the push where ``takes_tail_push`` holds, else the tail GEMV and the two-shot ``allreduce_attn_res``."""
    comm, res = BUILT.comm, BUILT.res
    pending = call.pending(tail)
    out_mods = (res.out_proj, res.out_attn_norm, res.model_norm)
    valid = call.bank[:SNAPSHOTS]
    with _switches(tail, 1):
        if isinstance(pending, DC.WideTail):
            if comm.takes_tail_push(pending, call.step, call.prefix, valid, *out_mods):
                normed, _ = comm.tail_push_attn_res(pending, call.prefix, valid, *out_mods)
                return normed
            pending = pending.partial()
        assert comm.takes_wide(pending, call.step)
        normed, _ = comm.allreduce_attn_res(pending, call.prefix, valid, *out_mods, two_shot=True)
    return normed


def _ref_pre_attention(call: _Call):
    """The pre-attention step before the switch: the tail GEMV on its decode GEMV site, then
    ``allreduce_attn_res(two_shot=True)``: ``(normed, updated)``."""
    res = BUILT.res
    partial = DM._gemv(BUILT.gemvs, TP.SITE, call.tail_x, BUILT.tail_w)
    return BUILT.comm.allreduce_attn_res(
        partial, call.prefix, call.bank[:SNAPSHOTS], res.sa_proj, res.sa_norm, res.in_norm,
        two_shot=True,
    )  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------------------------------
def check_build():
    """The TP group's ``K3DecodeComm`` and decode GEMVs as the target builds them, the push compiled at load, this
    rank's tail weight and o_proj stand-in, the residual modules."""
    global BUILT
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if R.world not in (4, 8, 16):
        raise AssertionError(f"the MNNVL ops run 4, 8 or 16 ranks, not {R.world}")
    comm = DC.K3DecodeComm.create(R.mapping, None)
    view = comm.oproj_push.lamport
    assert view.world == R.world and view.rank == R.rank
    assert view.flags is comm.mnnvl.buffer_flags, "the view must share the workspace's rotation"
    gemvs = DG.K3DecodeGemvs.create(sites=(TP.SITE,), device=torch.device("cuda"))
    tail_w = _normal(_gen(100 + R.rank), (H, K_TAIL), 0.02)
    tail_w[:, PAD[0] : PAD[1]] = 0
    tiles = gemvs.compile_push(TP.SITE, tail_w, view.world)
    assert tiles == [16, 32], f"the push compiled for {tiles}"
    g = _gen(7000)
    with torch.device("cuda"):
        res = SimpleNamespace(
            sa_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            mlp_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            out_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            sa_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            mlp_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            out_attn_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            in_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
            post_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
            model_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
        )
    with torch.no_grad():
        for proj in (res.sa_proj, res.mlp_proj, res.out_proj):
            proj.weight.copy_(_normal(g, proj.weight.shape, 0.05))
        for norm in (res.sa_norm, res.mlp_norm, res.out_attn_norm, res.in_norm, res.post_norm,
                     res.model_norm):  # fmt: skip
            norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
    torch.cuda.synchronize()
    BUILT = SimpleNamespace(
        comm=comm, gemvs=gemvs, tail_w=tail_w, o_proj=_Fp8OProj(200 + R.rank), res=res
    )
    if R.rank == 0:
        print(
            f"[rank 0] built at world {R.world}: push compiled for token tiles {tiles}", flush=True
        )


# ----------------------------------------------------------------------------------------------------------------------
# takes
# ----------------------------------------------------------------------------------------------------------------------
def check_takes():
    """``takes_tail_push`` per switch, token count and step, and its refusals."""
    comm, res = BUILT.comm, BUILT.res
    norms = (res.sa_proj, res.sa_norm, res.in_norm)

    def takes(tail, rows, step="call", snapshots=SNAPSHOTS, wide_ar=1, gemvs=None):
        call = calls[rows]
        pending = TP.WideTail(call.tail_x, BUILT.tail_w, gemvs or BUILT.gemvs)
        with _switches(tail, 1, wide_ar):
            return comm.takes_tail_push(
                pending, call.step if step == "call" else step, call.prefix,
                call.bank[:snapshots], *norms,
            )  # fmt: skip

    calls = {rows: _Call(300 + rows, rows, False) for rows in EAGER_ROWS}
    for rows in EAGER_ROWS:
        assert not takes(0, rows), f"switch 0 took {rows} rows"
        assert takes(1, rows) == (rows in PUSH_ROWS), f"switch 1 at {rows} rows"
    assert not takes(1, 16, step=IMPL.DecodeStep(16)), "took a step that is not a pure decode step"
    assert not takes(1, 16, step=None), "took a step of None"
    assert not takes(1, 16, step=_decode_step(24)), "took 16 rows of a 24-token step"
    assert not takes(1, 16, wide_ar=0), (
        "took a step whose switch-off consumer is not the two-shot attn_res op"
    )
    assert not takes(1, 16, snapshots=0), "took an empty snapshot bank"
    # A capture whose plain wide GEMV has not run eagerly: the plain path would refuse the kernel, so must the push.
    fresh = DG.K3DecodeGemvs()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        refused = not takes(1, 16, gemvs=fresh)
    assert refused, "took a capture whose plain wide GEMV had not run eagerly"
    del graph


# ----------------------------------------------------------------------------------------------------------------------
# Eager
# ----------------------------------------------------------------------------------------------------------------------
ARMS = ((0, 0), (1, 0), (0, 1), (1, 1))  # (TAIL_PUSH, OPROJ_PUSH)


def check_eager():
    """The off arm's pre-attention step = the pre-change chain; every arm = every other (module docstring)."""
    for rows in EAGER_ROWS:
        for snapshot_layer in (False, True):
            for kind in ("kda", "mla"):
                where = f"eager {rows} rows {kind} snapshot {snapshot_layer}"
                seed = 400 + 2 * rows + int(snapshot_layer)
                call = _Call(seed, rows, snapshot_layer)
                layer = _decoder(call, kind)
                ref_normed, ref_updated = (t.clone() for t in _ref_pre_attention(call))
                arms, pre = {}, {}
                for tail, oproj in ARMS:
                    call.draw(seed)
                    before = _census("eager")
                    with _tap() as seen:
                        arms[tail, oproj] = _cloned(_arm(layer, call, tail, oproj))
                    pushed = bool(tail) and rows in PUSH_ROWS
                    first, (pre_normed, pre_updated) = seen[0]
                    assert first == ("tail_push_attn_res" if pushed else "allreduce_attn_res"), (
                        f"{where}: arm {(tail, oproj)} ran {[name for name, _ in seen]}"
                    )
                    assert [name for name, _ in seen[1:]] == [
                        "oproj_push_attn_res" if oproj else "allreduce_attn_res"
                    ], (
                        f"{where}: arm {(tail, oproj)} post-attention ran {[name for name, _ in seen]}"
                    )
                    if CENSUS.ENABLED:
                        assert _census("eager") - before == int(pushed), f"{where}: census"
                    STATS["tail_push_calls"] += int(pushed)
                    pre[tail, oproj] = (pre_normed.clone(), pre_updated.clone())
                for (tail, oproj), (normed, updated) in pre.items():
                    _expect_same(f"{where} arm {(tail, oproj)}: pre-attention normed vs pre-change",
                                 normed, ref_normed, replicated=True)  # fmt: skip
                    _expect_same(f"{where} arm {(tail, oproj)}: pre-attention updated vs pre-change",
                                 updated, ref_updated, replicated=True)  # fmt: skip
                for arm in ARMS[1:]:
                    for name, value in arms[arm].items():
                        _expect_same(f"{where}: arm {arm} vs (0, 0) {name}", value,
                                     arms[0, 0][name], replicated=True)  # fmt: skip
                # The final norm's consumer.
                call.draw(seed)
                off = _final(call, 0).clone()
                on = _final(call, 1).clone()
                _expect_same(f"{where}: final norm on vs off", on, off, replicated=True)
    if R.rank == 0:
        print(
            "[rank 0] eager: off = pre-change chain; every TAIL_PUSH x OPROJ_PUSH arm bit for bit; final norm on = off",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------------------------------------------------
class _Step:
    """One captured step at ``rows`` tokens: layer a (KDA), the all-gather, layer b (MLA, snapshotting), a two-shot
    attn_res all-reduce, the final norm's consumer; every input a persistent buffer, rewritten by ``draw``."""

    def __init__(self, rows: int, seed: int):
        self.rows = rows
        self.a = _Call(seed, rows, False)
        self.b = _Call(seed + 1, rows, True)
        self.f = _Call(seed + 2, rows, False)
        self.la, self.lb = _decoder(self.a, "kda"), _decoder(self.b, "mla")
        self.ar = _Call(seed + 3, rows, False)
        self.ar_partial = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
        self.ag_in = torch.empty(rows, AG_BF16 + AG_FP32, device="cuda")
        self.draw(seed)

    def draw(self, seed: int):
        for i, call in enumerate((self.a, self.b, self.f, self.ar)):
            call.draw(seed + 17 * i)
        self.ar_partial.copy_(_normal(_gen(seed * 64 + R.rank + 7), self.ar_partial.shape, 0.25))
        g = _gen(seed + 99)
        self.ag_all = torch.randn(R.world, self.rows, AG_BF16 + AG_FP32, generator=g, device="cuda")
        self.ag_in.copy_(self.ag_all[R.rank])

    def run(self, tail: int, oproj: int):
        from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
            mnnvl_allgather_split,
        )

        comm, res = BUILT.comm, BUILT.res
        out = SimpleNamespace()
        out.a = _arm(self.la, self.a, tail, oproj)
        out.ag_bf16, out.ag_fp32 = mnnvl_allgather_split(self.ag_in, AG_BF16, comm.mnnvl)
        out.b = _arm(self.lb, self.b, tail, oproj)
        normed, updated = comm.allreduce_attn_res(
            self.ar_partial, self.ar.prefix, self.ar.bank[:SNAPSHOTS], res.mlp_proj,
            res.mlp_norm, res.post_norm, two_shot=True,
        )  # fmt: skip
        out.ar = SimpleNamespace(normed=normed, updated=updated)
        out.final = _final(self.f, tail)
        return out

    def verify_refs(self, where: str, got: dict):
        want_bf16 = self.ag_all[:, :, :AG_BF16].bfloat16().permute(1, 0, 2).reshape(self.rows, -1)
        want_fp32 = self.ag_all[:, :, AG_BF16:].permute(1, 0, 2).reshape(self.rows, -1)
        _expect_same(f"{where}: all-gather bf16", got["ag_bf16"], want_bf16, replicated=True)
        _expect_same(f"{where}: all-gather fp32", got["ag_fp32"], want_fp32, replicated=True)


def _capture(fn):
    R.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn()
    R.barrier()
    return graph, out


def check_graphs():
    """Graphs of 16, 24 and 32 tokens, each captured with the switches flipped between captures on one built state,
    replayed alternately among eager steps (module docstring)."""
    steps = {rows: _Step(rows, 600 + 10 * rows) for rows in GRAPH_ROWS}
    for step in steps.values():
        for tail, oproj in ARMS:
            step.run(tail, oproj)  # every kernel compiles eagerly, before any capture
    torch.cuda.synchronize()
    graphs = {}
    for rows, step in steps.items():
        for tail, oproj in GRAPH_MODES:
            before = _census("capture")
            with _tap() as seen:
                graphs[rows, tail, oproj] = _capture(lambda s=step, t=tail, o=oproj: s.run(t, o))
            pushes = sum(name == "tail_push_attn_res" for name, _ in seen)
            # The two layers' pre-attention steps and the final norm's consumer; the census counts the layers' only
            # (the final norm's block is the model's forward, which _final stands in for).
            assert pushes == 3 * tail, (
                f"captured {rows} rows {(tail, oproj)}: {[n for n, _ in seen]}"
            )
            if CENSUS.ENABLED:
                assert _census("capture") - before == 2 * tail, (
                    f"census at capture {rows} {(tail, oproj)}"
                )
    for i, rows in enumerate(REPLAY_ORDER):
        step = steps[rows]
        where = f"graph {rows} rows replay {i}"
        step.draw(700 + 13 * i)
        got = {}
        for tail, oproj in GRAPH_MODES:
            graph, out = graphs[rows, tail, oproj]
            R.barrier()
            R.late(i % R.world if i % 2 else None)
            graph.replay()
            torch.cuda.synchronize()
            got[tail, oproj] = _cloned(out)
        on = _cloned(step.run(1, 1))
        off = _cloned(step.run(0, 0))
        for mode, outs in got.items():
            for name, value in outs.items():
                _expect_same(f"{where}: graph {mode} vs eager on {name}", value, on[name])
                _expect_same(f"{where}: graph {mode} vs eager off {name}", value, off[name],
                             replicated=True)  # fmt: skip
            step.verify_refs(f"{where} graph {mode}", outs)
        # An eager push of another token count between replays.
        extra = _Call(800 + i, PUSH_ROWS[i % len(PUSH_ROWS)], i % 2 == 1)
        a = _cloned(_arm(_decoder(extra, "kda"), extra, 1, 1))
        extra.draw(800 + i)
        b = _cloned(_arm(_decoder(extra, "kda"), extra, 0, 1))
        for name in a:
            _expect_same(
                f"{where} eager {extra.rows} rows {name}", a[name], b[name], replicated=True
            )
    del graphs
    torch.cuda.synchronize()
    if R.rank == 0:
        print(
            f"[rank 0] graphs {GRAPH_ROWS} x modes {GRAPH_MODES} replayed {len(REPLAY_ORDER)}x alternately with "
            "eager steps: every graph = eager on = eager off; other Lamport users unchanged",
            flush=True,
        )


MX_ROWS = (9, 16, 24, 32)
N_KDA_FUSED, N_MLA_FUSED_A = 3328, 2112


def _fp8_weight(n: int, seed: int):
    """An FP8 weight-read weight ``[n, H]`` and its packed scale (``prepare_k3_fp8w_lowm_weight``); ``n`` padded to a
    row tile for the scale blocks, then the weight cut back to ``n`` rows (a partial last tile)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

    g = _gen(seed)
    n_pad = -(-n // 128) * 128
    w = (torch.randn(n_pad, H, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scale = torch.exp2(
        torch.randint(-8, -2, (n_pad // 128, H // 128), generator=g, device="cuda").float()
    )
    return lowm.prepare_k3_fp8w_lowm_weight(w[:n].contiguous(), scale)


class _ProjAttention(_Attention):
    """``_Attention`` that first runs its first FP8 projection on its input through ``fp8w_lowm.gemm`` (where an
    offered MXFP8 form is taken) and records the projection's output."""

    def __init__(self, core, o_proj, kind: str, weight, packed):
        super().__init__(core, o_proj)
        self.kind, self.weight, self.packed, self.proj = kind, weight, packed, None
        if kind == "kda":
            self._qkvg_bfa_fp8 = (weight, packed)
        else:
            self.mixer = SimpleNamespace(
                o_proj=o_proj,
                kv_a_proj_with_mqa=SimpleNamespace(
                    weight=weight,
                    weight_scale=packed,
                    _k3ll_fp8w_lowm=True,
                    _k3ll_fp8w_partial_n_tile=True,
                ),
            )

    def __call__(self, hidden_states, *args, **kwargs):
        self.proj = FW.gemm(
            hidden_states.reshape(-1, hidden_states.shape[-1]), self.weight, self.packed
        )
        return super().__call__(hidden_states, *args, **kwargs)


def _mx_switches(on: int):
    return _flag_block(
        {
            "TRTLLM_K3LL_ATTN_RES_MXFP8": on,
            "TRTLLM_K3LL_KDA_BFA_FP8": 1,
            "TRTLLM_K3LL_FP8W_LOWM_FUSED_A": 1,
        }
    )


@contextlib.contextmanager
def _flag_block(values):
    old = {n: getattr(FLAGS, n) for n in values}
    for n, v in values.items():
        setattr(FLAGS, n, int(v))
    try:
        yield
    finally:
        for n, v in old.items():
            setattr(FLAGS, n, v)


def _mx_layer(call: _Call, kind: str, weights):
    layer = _decoder(call, kind)
    attention = _ProjAttention(call.core, BUILT.o_proj, kind, *weights[kind])
    layer.attention = attention
    if kind == "kda":
        layer.linear_attn = attention
    else:
        layer.self_attn = attention
    layer.hidden_size = H
    return layer


def _mx_arm(layer, call: _Call, on: int):
    with _mx_switches(on):
        out = _arm(layer, call, 1, 1)
    out.proj = layer.attention.proj
    return out


def check_attn_res_mxfp8():
    """ATTN_RES_MXFP8 on vs off on the tail-push path (module docstring)."""
    weights = {"kda": _fp8_weight(N_KDA_FUSED, 71), "mla": _fp8_weight(N_MLA_FUSED_A, 72)}
    for rows in MX_ROWS:
        for kind in ("kda", "mla"):
            for snapshot_layer in (False, True):
                where = f"attn_res_mxfp8 {rows} rows {kind} snapshot {snapshot_layer}"
                seed = 900 + 2 * rows + int(snapshot_layer)
                call = _Call(seed, rows, snapshot_layer)
                layer = _mx_layer(call, kind, weights)
                outs = {}
                for on in (0, 1):
                    call.draw(seed)
                    before = FW.MX_STATS["mxfp8"]
                    seen_offer = []
                    real_offer = FW.offer_mxfp8

                    def spy(x, xq, xs, _seen=seen_offer, _real=real_offer):
                        _seen.append((x.clone(), xq.clone(), xs.clone()))
                        return _real(x, xq, xs)

                    FW.offer_mxfp8 = spy
                    try:
                        outs[on] = _cloned(_mx_arm(layer, call, on))
                    finally:
                        FW.offer_mxfp8 = real_offer
                    ran = FW.MX_STATS["mxfp8"] - before
                    assert ran == on, f"{where}: on={on} ran the MXFP8 GEMM {ran} times"
                    assert len(seen_offer) == on, (
                        f"{where}: on={on} offered {len(seen_offer)} times"
                    )
                    if on:
                        x, xq, xs = seen_offer[0]
                        rq, rs = MX.mxfp8_quantize_reference(x.reshape(-1, H))
                        assert torch.equal(xq.view(torch.uint8), rq.view(torch.uint8)), (
                            f"{where}: xq"
                        )
                        assert torch.equal(xs, rs), f"{where}: xs"
                for name, value in outs[1].items():
                    _expect_same(
                        f"{where}: on vs off {name}",
                        value,
                        outs[0][name],
                        replicated=name != "proj",
                    )
    # A CUDA graph of the on arm, compiled eagerly above, replayed with new inputs.
    rows, kind = 16, "kda"
    call = _Call(950, rows, False)
    layer = _mx_layer(call, kind, weights)
    _mx_arm(layer, call, 1)
    torch.cuda.synchronize()
    graph, out = _capture(lambda: _mx_arm(layer, call, 1))
    for i in range(3):
        call.draw(960 + i)
        graph.replay()
        torch.cuda.synchronize()
        replayed = _cloned(out)
        call.draw(960 + i)
        eager = _cloned(_mx_arm(layer, call, 1))
        for name, value in replayed.items():
            _expect_same(
                f"attn_res_mxfp8 graph replay {i} {name}",
                value,
                eager[name],
                replicated=name != "proj",
            )
    if R.rank == 0:
        print("[rank 0] attn_res_mxfp8: on = off bit for bit (normed, updated, projection, outputs); xq / xs = "
              "reference; MXFP8 GEMM ran on the on arm only; graph replay = eager", flush=True)  # fmt: skip


CHECKS = [check_takes, check_eager, check_graphs, check_attn_res_mxfp8]


def _run_one_rank(args) -> int:
    global R, IMPL, DC, DG, DM, TP, FLAGS, CENSUS, FW, MX
    R = ls.Rank(args)
    import importlib

    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    IMPL = importlib.import_module(f"{package}.implementation")
    DC = importlib.import_module(f"{package}.decode_comm")
    DG = importlib.import_module(f"{package}.decode_gemv")
    DM = importlib.import_module(f"{package}.decode_moe")
    TP = importlib.import_module(f"{package}.moe_tail_push")
    FLAGS = importlib.import_module(f"{package}.flags")
    CENSUS = importlib.import_module(f"{package}.census")
    FW = importlib.import_module(f"{package}.fp8w_lowm")
    MX = importlib.import_module(
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_fp8w_lowm_mxfp8"
    )
    CENSUS.ENABLED = True  # the switch's dispatch key is checked as the gates read it
    code = ls.run_checks(R, [check_build])
    if code:
        return code
    with torch.inference_mode():
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print(
            f"[rank 0] MOE_TAIL_PUSH on vs off: bit for bit; {STATS['tail_push_calls']} eager layer push calls; "
            f"census {CENSUS_KEY}: eager={_census('eager')} capture={_census('capture')}",
            flush=True,
        )
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

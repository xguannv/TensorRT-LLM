# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_K3LL_OPROJ_PUSH (``kimi_k3_nvfp4/oproj_push.py``): a decode layer's post-attention step with o_proj pushing
its partial into the all-reduce, each arm against the other and the off arm against the chain it replaced, at W ranks
(default 4) with the TP16 per-rank o_proj shape [7168, 768] (the KDA and the MLA layers' alike).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_oproj_push_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _kimi_k3_oproj_push_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (the arms share the decode path's collective
state). The collected entry point is ``test_modeling_v2_kimi_k3_oproj_push_op_matrix.py``.

What runs. ``KimiLinearDecoderLayer.forward`` itself on a stand-in layer: the target's residual modules, the TP group's
``K3DecodeComm`` (``create``, as post_load_weights builds it), an attention stand-in that hands over a fixed o_proj
input (``project_output=False``) or o_proj's partial (``reduce_output=False``), and an FP8 o_proj stand-in holding what
``fp8w_lowm.install`` leaves on a routed module (the UE8M0 weight, the DeepGEMM-packed scale, ``_k3ll_fp8w_lowm``) whose
forward is the routed ``k3_fp8w_lowm_gemm``. Every rank holds its own o_proj weight and input (a row-parallel shard);
the prefix sum and the snapshot bank are the same on every rank. The switch is set on the flags module per arm (read at
forward time): 0 off, 1 every pure decode step, 2 wide steps only.

Checks:
  * takes: ``K3DecodeComm.takes_oproj_push`` per switch value, step kind and token count, and its refusals (a module
    not on the low-M GEMM, a step that is not a pure decode step, a token count other than the step's).
  * eager (1, 8, 9, 16, 32, 64 rows; a layer that snapshots and one that does not; the layer as KDA and as MLA): the
    off arm (the one-shot ``mnnvl_allreduce_attn_res`` up to 8 rows, two-shot on wide steps) bit for bit the pre-change
    chain (o_proj on ``k3_fp8w_lowm_gemm``, then ``K3DecodeComm.allreduce_attn_res``), on every rank alike. The on arm
    against the off arm, bit for bit and the same bits on every rank: ``updated`` (the push GEMM rounds as the plain
    GEMM, the finish kernel sums in the C++ kernels' order, the residual add is the same bf16 add) and ``normed``
    (``K3AttnResFast`` with ``mnnvl_order`` runs the C++ epilogue's layout, statistics order, softmax and mixture).
    Switch 2 equals switch 1 on wide steps and the off arm elsewhere, bit for bit. Which collective ran is tapped, not
    assumed.
  * graphs (8, 16, 32 rows): one step per token count captured in a CUDA graph with the switch on: three layers on the
    push path (KDA, MLA snapshotting, MLA) with the workspace's other users between them: ``mnnvl_allgather_split`` and
    ``allreduce_attn_res`` on the same ``MnnvlWorkspace`` (they share its Lamport rotation), ``sandwich_plain`` on the
    sandwich workspace and two pushes + ``k3_latent_reduce`` on a ``K3LatentExchange``. The three graphs are replayed
    alternately with new inputs, one rank late on every other replay, with eager steps (both arms) in between; every
    replay bit for bit the eager on-arm step on the same inputs, the eager on arm against the eager off arm as above,
    every other collective's output the same bits on both arms and equal to its exact reference (the all-gather's
    concatenation, the latent sum of exact payloads).

TRTLLM_K3LL_FINISH_FOLD (the attn_res kernel running the Lamport finish itself): every eager case also runs the on
arm with the fold off and on (whether ``finish_lamport_push`` ran is tapped), both bit for bit the off arm; the graph
check runs twice, its graphs captured with the fold off, then on.

Where ``normed`` differs, the failure names the share of differing elements and the largest difference in bf16 ulps.
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

assert torch.cuda.is_available(), "TRTLLM_K3LL_OPROJ_PUSH requires CUDA devices"

DEADLINE_S = 1800
H = 7168
K_O = 768  # o_proj's input per rank (kda_o / mla_o at TP16)
SNAPSHOTS = 3
ATTN_RES_BLOCK = 4
RES_EPS = 1e-5
EAGER_ROWS = (1, 8, 9, 16, 32, 64)
GRAPH_ROWS = (8, 16, 32)
REPLAY_ORDER = (8, 32, 16, 8, 16, 32, 32, 8, 16)
PLAIN_K = 384  # the sandwich_plain stand-in's projection (the drafter's TP16 o_proj)
PLAIN_ROWS = 8
LATENT = 3584
LATENT_ROWS = 8
LATENT_WORDS = LATENT // 2
AG_BF16, AG_FP32 = 64, 16  # the all-gather stand-in's columns per rank
NEG_ZERO = -(2**15)

R = None
IMPL = None  # kimi_k3_nvfp4/implementation.py
DC = None  # decode_comm
OP = None  # oproj_push
FLAGS = None  # flags
BUILT = None
STATS = {"push_calls": 0, "fold_calls": 0}


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0, offset=0.0):
    return (offset + scale * torch.randn(tuple(shape), generator=g, device="cuda")).bfloat16()


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(t.shape[0], -1)


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


def _ulp_note(got: torch.Tensor, want: torch.Tensor) -> str:
    """For a bf16 mismatch: the share of differing elements and the largest difference in bf16 ulps of the element."""
    if got.dtype != torch.bfloat16 or got.shape != want.shape:
        return ""
    a, e = got.float(), want.float()
    diff = (a - e).abs()
    ulp = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(2.0**-126))) - 7)
    return f"; {float((diff > 0).float().mean()):.4%} of the elements, max {float((diff / ulp).max()):.2f} ulp"


@contextlib.contextmanager
def _switch(value: int):
    """Set TRTLLM_K3LL_OPROJ_PUSH on the flags module (read at forward time) for the block."""
    old = FLAGS.TRTLLM_K3LL_OPROJ_PUSH
    FLAGS.TRTLLM_K3LL_OPROJ_PUSH = int(value)
    try:
        yield
    finally:
        FLAGS.TRTLLM_K3LL_OPROJ_PUSH = old


@contextlib.contextmanager
def _fold(value: int):
    """Set TRTLLM_K3LL_FINISH_FOLD on the flags module (read at forward time) for the block."""
    old = FLAGS.TRTLLM_K3LL_FINISH_FOLD
    FLAGS.TRTLLM_K3LL_FINISH_FOLD = int(value)
    try:
        yield
    finally:
        FLAGS.TRTLLM_K3LL_FINISH_FOLD = old


@contextlib.contextmanager
def _finish_tap():
    """Count the separate Lamport finish launches (``oproj_push.finish_lamport_push``) in the block."""
    seen = {"finish": 0}
    orig = OP.finish_lamport_push

    def tapped(*args, **kwargs):
        seen["finish"] += 1
        return orig(*args, **kwargs)

    OP.finish_lamport_push = tapped
    try:
        yield seen
    finally:
        OP.finish_lamport_push = orig


@contextlib.contextmanager
def _path_tap():
    """Count the post-attention collectives the layer calls on the decode state (instance attributes, removed on
    exit)."""
    comm = BUILT.comm
    seen = {"push": 0, "allreduce": 0}
    names = {"push": "oproj_push_attn_res", "allreduce": "allreduce_attn_res"}
    for key, name in names.items():

        def tapped(*args, _method=getattr(comm, name), _key=key, **kwargs):
            seen[_key] += 1
            return _method(*args, **kwargs)

        setattr(comm, name, tapped)
    try:
        yield seen
    finally:
        for name in names.values():
            comm.__dict__.pop(name, None)


def _decode_step(rows: int):
    """A pure decode step of ``rows`` tokens (8 per request; one request below 8)."""
    return IMPL.DecodeStep(rows, max(1, rows // 8), min(rows, 8))


def _flat(ns, prefix=""):
    """{name: tensor} of a (nested) namespace of outputs."""
    out = {}
    for key, value in vars(ns).items():
        if isinstance(value, SimpleNamespace):
            out.update(_flat(value, f"{prefix}{key}."))
        elif value is not None:
            out[f"{prefix}{key}"] = value
    return out


def _cloned(ns):
    return {k: v.clone() for k, v in _flat(ns).items()}


# ----------------------------------------------------------------------------------------------------------------------
# Stand-ins
# ----------------------------------------------------------------------------------------------------------------------
class _Fp8OProj(nn.Module):
    """What ``fp8w_lowm.install`` leaves on a routed FP8 o_proj: the UE8M0-resmoothed E4M3 weight ``[H, K]``, its
    DeepGEMM-packed scale, ``_k3ll_fp8w_lowm``; its forward the routed ``k3_fp8w_lowm_gemm``."""

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
    """An attention module's decode branch as the layer calls it: ``project_output=False`` hands over the fixed o_proj
    input, ``reduce_output=False`` o_proj's partial of it."""

    def __init__(self, core: torch.Tensor, o_proj: nn.Module):
        self.core, self.o_proj = core, o_proj
        self.mixer = SimpleNamespace(o_proj=o_proj)

    def will_run_decode_branch(self, attn_metadata, step) -> bool:
        return True

    def __call__(
        self, hidden_states, attn_metadata, step=None, reduce_output=True, project_output=True
    ):
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
    """One layer's step: this rank's o_proj input (its own), the prefix sum and the snapshot bank (``SNAPSHOTS``
    valid rows, one spare; the same on every rank); rewritten in place for a replay."""

    def __init__(self, seed: int, rows: int, snapshot_layer: bool):
        self.rows, self.snapshot_layer, self.step = rows, snapshot_layer, _decode_step(rows)
        empty = dict(dtype=torch.bfloat16, device="cuda")
        self.core = torch.empty(rows, K_O, **empty)
        self.prefix = torch.empty(rows, H, **empty)
        self.bank = torch.empty(SNAPSHOTS + 1, rows, H, **empty)
        self.moe_next = torch.zeros(rows, H, **empty)
        self.draw(seed)

    def draw(self, seed: int):
        self.core.copy_(_normal(_gen(seed * 64 + R.rank + 1), self.core.shape, 0.25))
        g = _gen(seed)
        self.prefix.copy_(_normal(g, self.prefix.shape))
        self.bank.copy_(_normal(g, self.bank.shape))


def _decoder(call: _Call, kind: str):
    """``KimiLinearDecoderLayer``'s attributes its decode forward reads, with its own methods bound: the TP group's
    ``K3DecodeComm``, the residual modules, a KDA (``linear_attn``) or MLA (``self_attn``) attention stand-in and a
    MoE stand-in."""
    res = BUILT.res
    attention = _Attention(call.core, BUILT.o_proj)
    layer = SimpleNamespace(
        layer_idx=ATTN_RES_BLOCK if call.snapshot_layer else 1, attn_res_block_size=ATTN_RES_BLOCK,
        decode_comm=BUILT.comm, sandwich_oproj=False, is_kda=kind == "kda",
        linear_attn=attention if kind == "kda" else None, self_attn=attention if kind == "mla" else None,
        self_attention_res_proj=res.sa_proj, self_attention_res_norm=res.sa_norm, input_layernorm=res.in_norm,
        mlp_res_proj=res.mlp_proj, mlp_res_norm=res.mlp_norm, post_attention_layernorm=res.post_norm,
        is_moe=True, block_sparse_moe=_Recorder(call.moe_next),
    )  # fmt: skip
    for name in ("_o_proj", "_takes_oproj_push", "_oproj_push_attn_res"):
        setattr(layer, name, types.MethodType(getattr(IMPL.KimiLinearDecoderLayer, name), layer))
    return layer


def _arm(layer, call: _Call, mode: int):
    """The layer's decode forward with TRTLLM_K3LL_OPROJ_PUSH = ``mode``: ``normed`` (the MoE's input) and ``updated``
    (the prefix sum after the attention)."""
    with _switch(mode):
        prefix, snapshots, _ = IMPL.KimiLinearDecoderLayer.forward(
            layer, call.prefix, call.bank, SNAPSHOTS, None, call.step, defer_moe_tail=True
        )
    assert snapshots == SNAPSHOTS + int(call.snapshot_layer), snapshots
    return SimpleNamespace(normed=layer.block_sparse_moe.seen, updated=prefix)


def _ref_chain(call: _Call):
    """The post-attention step before the switch: the layer's snapshot of its prefix sum (a snapshotting layer), o_proj
    on the routed ``k3_fp8w_lowm_gemm``, then ``K3DecodeComm.allreduce_attn_res`` (one-shot up to 8 rows, two-shot on
    a wide step)."""
    prefix, valid = call.prefix, call.bank[:SNAPSHOTS]
    if call.snapshot_layer:
        call.bank[SNAPSHOTS].copy_(call.prefix)
        prefix, valid = None, call.bank[: SNAPSHOTS + 1]
    partial = torch.ops.trtllm.k3_fp8w_lowm_gemm(
        call.core, BUILT.o_proj.weight, BUILT.o_proj.weight_scale
    )
    res = BUILT.res
    normed, updated = BUILT.comm.allreduce_attn_res(
        partial, prefix, valid, res.mlp_proj, res.mlp_norm, res.post_norm, two_shot=call.step.wide
    )
    return SimpleNamespace(normed=normed, updated=updated)


# ----------------------------------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------------------------------
def check_build():
    """The TP group's ``K3DecodeComm`` as post_load_weights builds it (its o_proj push state included), this rank's
    o_proj stand-in, the residual modules, the sandwich_plain stand-in (compiled) and a latent exchange."""
    global BUILT
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_latent_reduce import (
        K3LatentExchange,
    )
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if R.world not in (4, 8, 16):
        raise AssertionError(f"the MNNVL ops run 4, 8 or 16 ranks, not {R.world}")
    comm = DC.K3DecodeComm.create(R.mapping, None)
    push = comm.oproj_push
    assert isinstance(push, OP.K3OprojPush), (
        "K3DecodeComm.create did not build the o_proj push state"
    )
    view = push.lamport
    assert view.world == R.world and view.rank == R.rank
    assert view.uc_local == comm.mnnvl.lamport.data_ptr()
    assert view.flags is comm.mnnvl.buffer_flags, (
        "the view must share the workspace's rotation state"
    )
    assert view.bytes_per_buffer == comm.mnnvl.buffer_bytes
    assert view.peer_ptrs.tolist()[R.rank] == view.uc_local
    assert bool((push.zeros == 0).all()) and tuple(push.zeros.shape) == (OP.MAX_TOKENS, H)
    g = _gen(7000)
    with torch.device("cuda"):
        res = SimpleNamespace(
            sa_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            mlp_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            sa_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            mlp_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            in_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
            post_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
            plain_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
        )
    with torch.no_grad():
        for proj in (res.sa_proj, res.mlp_proj):
            proj.weight.copy_(_normal(g, proj.weight.shape, 0.05))
        for norm in (res.sa_norm, res.mlp_norm, res.in_norm, res.post_norm, res.plain_norm):
            norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
    o_proj = _Fp8OProj(100 + R.rank)
    plain_w = _normal(_gen(200 + R.rank), (H, PLAIN_K), 0.02)
    assert comm.compile_plain(plain_w), "sandwich_plain does not take the stand-in's shape"
    exchange = K3LatentExchange.create(R.mapping, fabric_handle=R.fabric)
    torch.cuda.synchronize()
    BUILT = SimpleNamespace(
        comm=comm, o_proj=o_proj, res=res, plain_w=plain_w, exchange=exchange, latent_calls=0
    )
    if R.rank == 0:
        print(
            f"[rank 0] built at world {R.world}: Lamport view {view.bytes_per_buffer} B per buffer",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# takes
# ----------------------------------------------------------------------------------------------------------------------
def check_takes():
    """``takes_oproj_push`` per switch value and step, and its refusals."""
    res, comm = BUILT.res, BUILT.comm
    norms = (res.mlp_proj, res.mlp_norm, res.post_norm)

    def takes(mode, step, rows, o_proj=None, prefix="call", snapshots=SNAPSHOTS):
        call = calls[rows]
        p = call.prefix if prefix == "call" else prefix
        with _switch(mode):
            return comm.takes_oproj_push(
                step, rows, o_proj or BUILT.o_proj, p, call.bank[:snapshots], *norms
            )

    calls = {rows: _Call(300 + rows, rows, False) for rows in EAGER_ROWS}
    for rows in EAGER_ROWS:
        step = _decode_step(rows)
        assert not takes(0, step, rows), f"switch 0 took {rows} rows"
        assert takes(1, step, rows), f"switch 1 did not take {rows} rows"
        assert takes(2, step, rows) == step.wide, f"switch 2 at {rows} rows"
        assert takes(1, step, rows, prefix=None, snapshots=SNAPSHOTS + 1), f"zero prefix, {rows}"
    step = _decode_step(8)
    assert not takes(1, IMPL.DecodeStep(8), 8), "took a step that is not a pure decode step"
    assert not takes(1, None, 8), "took a step of None"
    assert not takes(1, _decode_step(16), 8), "took 8 rows of a 16-token step"
    unrouted = _Fp8OProj(100 + R.rank)
    unrouted._k3ll_fp8w_lowm = False
    assert not takes(1, step, 8, o_proj=unrouted), "took an o_proj not on k3_fp8w_lowm_gemm"
    assert not takes(1, step, 8, snapshots=0), "took an empty snapshot bank"


# ----------------------------------------------------------------------------------------------------------------------
# Eager
# ----------------------------------------------------------------------------------------------------------------------
def _verify_arms(where, on, off):
    for name in ("updated", "normed"):
        _expect_same(f"{where}: on vs off {name}", on[name], off[name], replicated=True)


def check_eager():
    """Off arm = pre-change chain, on arm against the off arm, switch 2 = the arm it selects (module docstring)."""
    for rows in EAGER_ROWS:
        for snapshot_layer in (False, True):
            for kind in ("kda", "mla"):
                where = f"eager {rows} rows {kind} snapshot {snapshot_layer}"
                call = _Call(400 + 2 * rows + int(snapshot_layer), rows, snapshot_layer)
                layer = _decoder(call, kind)
                ref = _cloned(_ref_chain(call))
                arms = {}
                for mode in (0, 1, 2):
                    with _path_tap() as seen:
                        arms[mode] = _cloned(_arm(layer, call, mode))
                    pushed = mode == 1 or (mode == 2 and call.step.wide)
                    assert seen == {"push": int(pushed), "allreduce": int(not pushed)}, (
                        f"{where}: switch {mode} ran {seen}"
                    )
                    STATS["push_calls"] += seen["push"]
                for name in ("normed", "updated"):
                    _expect_same(f"{where}: off arm vs pre-change {name}", arms[0][name], ref[name],
                                 replicated=True)  # fmt: skip
                _verify_arms(where, arms[1], arms[0])
                for fold in (0, 1):
                    with _fold(fold), _finish_tap() as fin:
                        folded = _cloned(_arm(layer, call, 1))
                    assert fin["finish"] == int(not fold), f"{where}: fold {fold} ran {fin}"
                    _verify_arms(f"{where} fold {fold}", folded, arms[0])
                    STATS["fold_calls"] += int(fold)
                for name in ("normed", "updated"):
                    _expect_same(f"{where}: switch 2 {name}", arms[2][name],
                                 arms[1 if call.step.wide else 0][name])  # fmt: skip
    if R.rank == 0:
        print(
            "[rank 0] eager: off = pre-change chain; on = off (updated and normed bit for bit)",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------------------------------------------------
class _Step:
    """One captured step at ``rows`` tokens (module docstring): three push-path layers and the other Lamport users
    between them; every input a persistent buffer, rewritten by ``draw``."""

    def __init__(self, rows: int, seed: int):
        self.rows = rows
        self.a = _Call(seed, rows, False)
        self.b = _Call(seed + 1, rows, True)
        self.c = _Call(seed + 2, rows, False)
        self.la, self.lb, self.lc = (
            _decoder(self.a, "kda"),
            _decoder(self.b, "mla"),
            _decoder(self.c, "mla"),
        )
        self.ar = _Call(seed + 3, rows, False)
        self.ar_partial = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
        self.ag_in = torch.empty(rows, AG_BF16 + AG_FP32, device="cuda")
        self.lat = torch.empty(2, LATENT_ROWS, LATENT, dtype=torch.bfloat16, device="cuda")
        self.plain_x = torch.empty(PLAIN_ROWS, PLAIN_K, dtype=torch.bfloat16, device="cuda")
        self.plain_res = torch.empty(PLAIN_ROWS, H, dtype=torch.bfloat16, device="cuda")
        self.draw(seed)

    def draw(self, seed: int):
        for i, call in enumerate((self.a, self.b, self.c, self.ar)):
            call.draw(seed + 17 * i)
        self.ar_partial.copy_(_normal(_gen(seed * 64 + R.rank + 7), self.ar_partial.shape, 0.25))
        g = _gen(seed + 99)
        self.ag_all = torch.randn(R.world, self.rows, AG_BF16 + AG_FP32, generator=g, device="cuda")
        self.ag_in.copy_(self.ag_all[R.rank])
        # Exact payloads (multiples of 1/16 in [-1/4, 1/4]): any summation order gives the same sum.
        self.lat_all = ls.exact_bf16(g, (2, R.world, LATENT_ROWS, LATENT), -4, 5, 1 / 16)
        self.lat.copy_(self.lat_all[:, R.rank])
        self.plain_x.copy_(_normal(_gen(seed * 64 + R.rank + 9), self.plain_x.shape, 0.25))
        self.plain_res.copy_(_normal(g, self.plain_res.shape))

    def _latent(self, i: int):
        """What a push-only producer stores (this rank's rows, -0 as +0, into slot [rank] of the call's half of every
        rank's exchange, through the multicast mapping), then ``k3_latent_reduce``."""
        from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_latent_reduce import (
            k3_latent_reduce,
        )

        ex = BUILT.exchange
        words = self.lat[i].view(torch.int16)
        words = words.masked_fill(words == NEG_ZERO, 0).view(torch.int32)
        dest = ex.mc.view(2, 8, R.world, LATENT_WORDS)
        dest[BUILT.latent_calls & 1, :LATENT_ROWS, R.rank].copy_(words)
        out = k3_latent_reduce(LATENT_ROWS, ex)
        BUILT.latent_calls += 1
        return out

    def run(self, mode: int):
        from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
            mnnvl_allgather_split,
        )

        comm, res = BUILT.comm, BUILT.res
        out = SimpleNamespace()
        out.a = _arm(self.la, self.a, mode)
        out.ag_bf16, out.ag_fp32 = mnnvl_allgather_split(self.ag_in, AG_BF16, comm.mnnvl)
        out.lat1 = self._latent(0)
        out.b = _arm(self.lb, self.b, mode)
        normed, updated = comm.allreduce_attn_res(
            self.ar_partial, self.ar.prefix, self.ar.bank[:SNAPSHOTS], res.mlp_proj,
            res.mlp_norm, res.post_norm, two_shot=self.rows > 8,
        )  # fmt: skip
        out.ar = SimpleNamespace(normed=normed, updated=updated)
        normed, updated = comm.sandwich_plain(
            self.plain_x, BUILT.plain_w, self.plain_res, res.plain_norm
        )
        out.plain = SimpleNamespace(normed=normed, updated=updated)
        out.lat2 = self._latent(1)
        out.c = _arm(self.lc, self.c, mode)
        return out

    def verify_refs(self, where: str, got: dict):
        """The other users' outputs against their exact references."""
        want_bf16 = self.ag_all[:, :, :AG_BF16].bfloat16().permute(1, 0, 2).reshape(self.rows, -1)
        want_fp32 = self.ag_all[:, :, AG_BF16:].permute(1, 0, 2).reshape(self.rows, -1)
        _expect_same(f"{where}: all-gather bf16", got["ag_bf16"], want_bf16, replicated=True)
        _expect_same(f"{where}: all-gather fp32", got["ag_fp32"], want_fp32, replicated=True)
        for i in range(2):
            want = self.lat_all[i].float().sum(0).bfloat16()
            _expect_same(f"{where}: latent {i}", got[f"lat{i + 1}"], want, replicated=True)


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
    """The graph check with TRTLLM_K3LL_FINISH_FOLD off, then on."""
    for fold in (0, 1):
        with _fold(fold):
            _check_graphs(fold)


def _check_graphs(fold: int):
    """Graphs of 8, 16 and 32 tokens replayed alternately among eager steps (module docstring)."""
    steps = {rows: _Step(rows, 600 + 10 * rows) for rows in GRAPH_ROWS}
    for step in steps.values():
        # Every kernel compiles eagerly, before any capture.
        step.run(1)
        step.run(0)
    torch.cuda.synchronize()
    graphs = {}
    for rows, step in steps.items():
        with _path_tap() as seen:
            graphs[rows] = _capture(lambda step=step: step.run(1))
        assert seen == {"push": 3, "allreduce": 1}, f"captured {rows} rows: {seen}"
    for i, rows in enumerate(REPLAY_ORDER):
        step = steps[rows]
        graph, out = graphs[rows]
        where = f"graph {rows} rows replay {i}"
        step.draw(700 + 13 * i)
        R.barrier()
        R.late(i % R.world if i % 2 else None)
        graph.replay()
        torch.cuda.synchronize()
        got = _cloned(out)
        on = _cloned(step.run(1))
        off = _cloned(step.run(0))
        for name, value in got.items():
            _expect_same(f"{where}: replay vs eager {name}", value, on[name])
        for layer in ("a", "b", "c"):
            _verify_arms(f"{where} layer {layer}", {k: on[f"{layer}.{k}"] for k in ("normed", "updated")},
                         {k: off[f"{layer}.{k}"] for k in ("normed", "updated")})  # fmt: skip
        for name in ("ag_bf16", "ag_fp32", "lat1", "lat2", "ar.normed", "ar.updated", "plain.normed",
                     "plain.updated"):  # fmt: skip
            _expect_same(f"{where}: {name} on vs off arm", on[name], off[name], replicated=True)
        step.verify_refs(where, got)
        # An eager push of another token count between replays.
        extra = _Call(800 + i, 9 + i, i % 2 == 1)
        _verify_arms(f"{where} eager {extra.rows} rows",
                     _cloned(_arm(_decoder(extra, "kda"), extra, 1)),
                     _cloned(_arm(_decoder(extra, "kda"), extra, 0)))  # fmt: skip
    del graphs
    torch.cuda.synchronize()
    if R.rank == 0:
        print(
            f"[rank 0] graphs {GRAPH_ROWS} (fold {fold}) replayed {len(REPLAY_ORDER)}x alternately with eager steps: "
            "replay = eager = off arm; other Lamport users unchanged",
            flush=True,
        )


CHECKS = [check_takes, check_eager, check_graphs]


def _run_one_rank(args) -> int:
    global R, IMPL, DC, OP, FLAGS
    R = ls.Rank(args)
    import importlib

    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    IMPL = importlib.import_module(f"{package}.implementation")
    DC = importlib.import_module(f"{package}.decode_comm")
    OP = importlib.import_module(f"{package}.oproj_push")
    FLAGS = importlib.import_module(f"{package}.flags")
    code = ls.run_checks(R, [check_build])
    if code:
        return code
    with torch.inference_mode():
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print(
            f"[rank 0] OPROJ_PUSH on vs off: normed and updated bit for bit; {STATS['push_calls']} eager push calls; "
            f"FINISH_FOLD on = off: {STATS['fold_calls']} folded eager calls",
            flush=True,
        )
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

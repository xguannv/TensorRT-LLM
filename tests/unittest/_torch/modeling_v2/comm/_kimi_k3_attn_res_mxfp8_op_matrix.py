# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_ATTN_RES_MXFP8`` and ``TRTLLM_K3LL_MX_SF_EARLY_RELEASE`` (``kimi_k3_nvfp4/flags.py``) in
``KimiLinearDecoderLayer.forward``, at W ranks (default 4) with the TP16 per-rank shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_attn_res_mxfp8_op_matrix.py [--world-size 4]
    srun -N1 -n4 --mpi=pmix python _kimi_k3_attn_res_mxfp8_op_matrix.py --launcher srun --world-size 4

Not a pytest module: one fixed sequence of checks inside one W-rank job (the arms share the decode path's collective
state). The collected entry point is ``test_modeling_v2_kimi_k3_attn_res_mxfp8_op_matrix.py``.

What runs. ``KimiLinearDecoderLayer.forward`` itself on a stand-in layer: the TP group's ``K3DecodeComm`` (built and
its sandwich tail compiled as ``post_load_weights`` does), the residual modules, an attention stand-in that runs the
layer's first FP8 projection on its input as the real attention does and returns a fixed partial, and a MoE stand-in.
The projection is one of the three consumers of the pre-attention output: the KDA fused ``[q | k | v | g | f_a | b]``
GEMM (``fp8w_lowm.gemm``, partial last row tile, N 3216), the KDA QKVG module (N 3072) and the MLA fused q_a / kv_a
module (N 2112, partial last row tile), each routed to ``k3_fp8w_lowm_gemm`` the way ``fp8w_lowm.install`` routes a
weight-read module. The pre-attention step is a small decode step's ``k3_sandwich_tail`` (M 1, 5, 8: a
``PendingTail``) or a wide step's two-shot ``mnnvl_allreduce_attn_res`` (M 16, 32: an unreduced MoE partial; with a
library that has ``mnnvl_allreduce_attn_res_mx``), in a layer that snapshots and one that does not. The post-attention
step is the one-shot / two-shot ``allreduce_attn_res`` (``TRTLLM_K3LL_OPROJ_PUSH`` 0: the stand-in has no o_proj).

Checks (every output compared bit for bit: the attention input, the projection output, the MoE input, the returned
prefix sum and the snapshot bank row; the shared outputs also across the ranks):
  * off arm (both switches 0) against the pre-change chain: the collectives without ``mxfp8_out`` and
    ``k3_fp8w_lowm_gemm`` on the attention input, called directly;
  * on arm (``TRTLLM_K3LL_ATTN_RES_MXFP8`` 1, early release 0 and 1) against the off arm, and that the projection
    actually ran ``k3_fp8w_lowm_mxfp8`` on the offered MXFP8 input (``fp8w_lowm.MX_STATS``: one GEMM per call, no
    offer left unconsumed); the MXFP8 copy itself against ``mxfp8_quantize_reference`` of the attention input;
  * one built layer flipped at run time, off -> on -> off -> on (early) -> off, every call against the first;
  * CUDA graphs captured per arm (after the eager calls compiled every kernel) and replayed with rewritten inputs,
    each replay against the eager off arm on the same inputs, the on arm's capture holding the MXFP8 GEMM.
"""

import contextlib
import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the Kimi K3 MXFP8 hand-off requires CUDA devices"

DEADLINE_S = 1800
H = 7168
LATENT = 3584
TAIL_SLICE = 224
TAIL_PAD = 256
TAIL_ACT = 384
RES_EPS = 1e-5
LAT_EPS = 1e-5
ATTN_RES_BLOCK = 4
SNAPSHOTS = 3
SMALL_ROWS = (1, 5, 8)
WIDE_ROWS = (16, 32)
REPLAYS = 2
# The consumers: (kind, N, partial last row tile).
CONSUMERS = (("kda_fused", 3216, True), ("kda", 3072, False), ("mla", 2112, True))

R = None
IMPL = None  # kimi_k3_nvfp4/implementation.py
DC = None  # decode_comm
FL = None  # fp8w_lowm
FLAGS = None  # flags
BUILT = None


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


@contextlib.contextmanager
def _switches(**values):
    """Set switches on the flags module (read at forward time as ``_flags.<name>``) for the block."""
    old = {name: getattr(FLAGS, name) for name in values}
    for name, value in values.items():
        setattr(FLAGS, name, int(value))
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(FLAGS, name, value)


def _arm(mx: int, early: int = 0):
    return _switches(
        TRTLLM_K3LL_ATTN_RES_MXFP8=mx,
        TRTLLM_K3LL_MX_SF_EARLY_RELEASE=early,
        TRTLLM_K3LL_OPROJ_PUSH=0,
    )


def _fp8_weight(n, k, seed):
    """An FP8 projection weight as the low-M GEMM holds it: N(0, 0.02), power-of-two 128x128 block scales, the
    DeepGEMM-packed UE8M0 scale."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

    g = _gen(seed)
    n_pad = -(-n // 128) * 128
    w = torch.randn((n_pad, k), device="cuda", generator=g) * 0.02
    blocks = w.view(n_pad // 128, 128, k // 128, 128)
    scale = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0))
    )
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n_pad, k)
    return lowm.prepare_k3_fp8w_lowm_weight(w_fp8[:n].contiguous(), scale.float())


class _Projection(nn.Module):
    """An FP8 weight-read projection routed as ``fp8w_lowm.install`` routes one: its forward is ``fp8w_lowm._forward``
    over a stock forward that must not run."""

    def __init__(self, weight, scale, partial):
        super().__init__()
        self.weight, self.weight_scale = weight, scale
        self._locality_domain_weight_shards = None
        self._k3ll_fp8w_lowm = True
        self._k3ll_fp8w_partial_n_tile = partial

        def stock(x, **kwargs):
            raise AssertionError("the stand-in projection's stock GEMM ran")

        self.forward = functools.partial(FL._forward, self, stock)


class _Attention:
    """An attention stand-in: runs its consumer projection on the step's rows of its input (kept with the output) and
    returns a fixed partial, as ``attention(..., reduce_output=False)`` does."""

    def __init__(self, kind, proj, fused, partial):
        self.kind, self.proj, self.fused, self.partial = kind, proj, fused, partial
        self._qkvg_bfa_fp8 = fused
        self.qkvg_proj = proj if kind == "kda" else None
        self.mixer = SimpleNamespace(kv_a_proj_with_mqa=proj if kind == "mla" else None)
        self.seen = self.out = None

    def will_run_decode_branch(self, attn_metadata, step):
        return True

    def __call__(
        self, hidden_states, attn_metadata=None, step=None, reduce_output=True, project_output=True
    ):
        assert not reduce_output and project_output
        x = hidden_states[: step.num_tokens]
        self.seen = hidden_states
        if self.kind == "kda_fused":
            self.out = FL.gemm(x.reshape(-1, x.shape[-1]), *self.fused)
        else:
            self.out = self.proj(x)
        return self.partial


class _Recorder:
    """The MoE stand-in: keeps its input, returns a fixed partial."""

    def __init__(self, out):
        self.out, self.seen = out, None

    def __call__(self, hidden_states, *args, **kwargs):
        self.seen = hidden_states
        return self.out


class _Call:
    """One decode step's inputs (this rank's buffers, rewritten in place for a replay): the previous layer's MoE
    output (a ``PendingTail`` on a small step, an unreduced partial on a wide one), the running prefix sum, the snapshot
    bank (``SNAPSHOTS`` valid rows) and the attention partial."""

    def __init__(self, seed, rows):
        self.rows = rows
        self.wide = rows > 8
        self.step = IMPL.DecodeStep(rows, max(1, rows // 8), min(rows, 8))
        bf = dict(dtype=torch.bfloat16, device="cuda")
        self.latent = torch.empty(rows, LATENT, **bf)
        self.act = torch.empty(rows, TAIL_ACT, **bf)
        self.moe_partial = torch.empty(rows, H, **bf)
        self.attn_partial = torch.empty(rows, H, **bf)
        self.prefix = torch.empty(rows, H, **bf)
        self.bank = torch.empty(SNAPSHOTS + 1, rows, H, **bf)
        self.moe_next = torch.zeros(rows, H, **bf)
        self.draw(seed)

    def draw(self, seed):
        g = _gen(seed)
        self.latent.copy_(
            _normal(g, self.latent.shape, 0.8)
        )  # the reduced latent: the same on every rank
        g_rank = _gen(seed + 7919 * (R.rank + 1))
        self.act.copy_(_normal(g_rank, self.act.shape, 0.5))
        self.moe_partial.copy_(_normal(g_rank, self.moe_partial.shape, 0.25))
        self.attn_partial.copy_(_normal(g_rank, self.attn_partial.shape, 0.25))
        self.prefix.copy_(_normal(g, self.prefix.shape))
        self.bank.copy_(_normal(g, self.bank.shape))

    def pending(self):
        if self.wide:
            return self.moe_partial
        return DC.PendingTail(
            self.latent, self.act, BUILT.tail_weight, R.rank * TAIL_SLICE, LAT_EPS
        )


def _decoder(call, consumer, layer_idx):
    """``KimiLinearDecoderLayer``'s attributes its forward reads on a decode step, with the attention stand-in for
    ``consumer``."""
    kind, n, partial = consumer
    w, ws = BUILT.weights[kind]
    proj = None if kind == "kda_fused" else _Projection(w, ws, partial)
    fused = (w, ws) if kind == "kda_fused" else None
    attention = _Attention(kind, proj, fused, call.attn_partial)
    res = BUILT.res
    return SimpleNamespace(
        layer_idx=layer_idx, attn_res_block_size=ATTN_RES_BLOCK, decode_comm=BUILT.comm, hidden_size=H,
        self_attention_res_proj=res.sa_proj, self_attention_res_norm=res.sa_norm, input_layernorm=res.in_norm,
        mlp_res_proj=res.mlp_proj, mlp_res_norm=res.mlp_norm, post_attention_layernorm=res.post_norm,
        is_kda=kind != "mla", linear_attn=attention, self_attn=attention, sandwich_oproj=False,
        is_moe=True, block_sparse_moe=_Recorder(call.moe_next), _o_allreduce=lambda: BUILT.all_reduce,
    )  # fmt: skip


def _forward(layer, call):
    """One ``KimiLinearDecoderLayer.forward`` (handing its MoE output on); its outputs, the graph's own tensors under
    capture."""
    prefix, snapshots, _ = IMPL.KimiLinearDecoderLayer.forward(
        layer, call.prefix, call.bank, SNAPSHOTS, None, call.step,
        pending_moe_partial=call.pending(), defer_moe_tail=True,
    )  # fmt: skip
    snapshot = layer.layer_idx % ATTN_RES_BLOCK == 0
    assert snapshots == SNAPSHOTS + int(snapshot), snapshots
    attention = layer.linear_attn
    return SimpleNamespace(
        normed1=attention.seen, proj=attention.out, bank_row=call.bank[SNAPSHOTS] if snapshot else None,
        normed2=layer.block_sparse_moe.seen, updated2=prefix,
    )  # fmt: skip


def _cloned(ns):
    return SimpleNamespace(**{k: None if v is None else v.clone() for k, v in vars(ns).items()})


def _ref_chain(layer, call):
    """The pre-change chain: the pre-attention collective without ``mxfp8_out``, ``k3_fp8w_lowm_gemm`` on its output,
    the snapshot, the post-attention collective."""
    comm, valid = layer.decode_comm, call.bank[:SNAPSHOTS]
    snapshot = layer.layer_idx % ATTN_RES_BLOCK == 0
    if call.wide:
        normed1, updated1 = comm.allreduce_attn_res(
            call.moe_partial, call.prefix, valid, layer.self_attention_res_proj, layer.self_attention_res_norm,
            layer.input_layernorm, two_shot=True,
        )  # fmt: skip
        if snapshot:
            call.bank[SNAPSHOTS].copy_(updated1)
    else:
        normed1, updated1 = comm.sandwich_tail(
            call.pending(), call.prefix, valid, layer.self_attention_res_proj, layer.self_attention_res_norm,
            layer.input_layernorm, updated_out=call.bank[SNAPSHOTS] if snapshot else None,
        )  # fmt: skip
    w, ws = BUILT.weights[layer.linear_attn.kind]
    proj = torch.ops.trtllm.k3_fp8w_lowm_gemm(normed1, w, ws)
    prefix = None if snapshot else updated1
    valid = call.bank[: SNAPSHOTS + 1] if snapshot else valid
    normed2, updated2 = comm.allreduce_attn_res(
        call.attn_partial, prefix, valid, layer.mlp_res_proj, layer.mlp_res_norm, layer.post_attention_layernorm,
        two_shot=call.wide,
    )  # fmt: skip
    return SimpleNamespace(normed1=normed1, proj=proj, bank_row=call.bank[SNAPSHOTS] if snapshot else None,
                           normed2=normed2, updated2=updated2)  # fmt: skip


def _verify(where, got, want):
    for name in ("normed1", "proj", "bank_row", "normed2", "updated2"):
        a, b = getattr(got, name), getattr(want, name)
        if b is None:
            assert a is None, f"{where}: {name}"
            continue
        _expect_same(f"{where}: {name}", a, b, replicated=name != "proj")


def _mx_expected(call) -> bool:
    """Whether the on arm hands the MXFP8 form over at this step (a wide step needs the library's op)."""
    return not call.wide or DC.wide_mxfp8_out()


def _on_arm(layer, call, early, where):
    """The on arm, eagerly: its outputs, after checking that the projection ran k3_fp8w_lowm_mxfp8 on the offer and
    that the offer was consumed (where the step hands one over)."""
    before = dict(FL.MX_STATS)
    captured = {}
    offer = FL.offer_mxfp8

    def keep(x, xq, xs):
        captured["mx"] = (x, xq, xs)
        offer(x, xq, xs)

    FL.offer_mxfp8 = keep
    try:
        with _arm(1, early):
            out = _cloned(_forward(layer, call))
    finally:
        FL.offer_mxfp8 = offer
    gemms = FL.MX_STATS["mxfp8"] - before.get("mxfp8", 0)
    unconsumed = FL.MX_STATS["unconsumed"] - before.get("unconsumed", 0)
    if _mx_expected(call):
        assert gemms == 1 and unconsumed == 0 and "mx" in captured, (
            f"{where}: the projection did not run on the offered MXFP8 input (gemms {gemms}, unconsumed {unconsumed})"
        )
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

        x, xq, xs = captured["mx"]
        rq, rs = mx.mxfp8_quantize_reference(x)
        _expect_same(
            f"{where}: the MXFP8 codes", xq.view(torch.uint8), rq.view(torch.uint8), replicated=True
        )
        _expect_same(f"{where}: the MXFP8 exponents", xs, rs, replicated=True)
    else:
        assert gemms == 0 and "mx" not in captured, f"{where}: an offer without the library's op"
    return out


def check_build():
    """The TP group's ``K3DecodeComm`` (and the sandwich tail compiled with and without the MXFP8 copy), the residual
    modules, the consumers' FP8 weights and the tail weight, outside inference mode."""
    global BUILT
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.functional import AllReduceStrategy

    if R.world not in (4, 8, 16):
        raise AssertionError(f"the MNNVL ops run 4, 8 or 16 ranks, not {R.world}")
    all_reduce = AllReduce(
        mapping=R.mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
    )
    DC.use_decode_one_shot(nn.ModuleList([all_reduce]))
    comm = DC.K3DecodeComm.create(R.mapping, None)
    g = _gen(5000 + R.rank)
    tail_weight = _normal(g, (H, TAIL_PAD + TAIL_ACT), 0.03)
    tail_weight[:, TAIL_SLICE:TAIL_PAD] = 0
    comm.compile_tail(LATENT, TAIL_ACT, tail_weight)
    g = _gen(7000)
    with torch.device("cuda"):
        res = SimpleNamespace(
            sa_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            mlp_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            sa_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            mlp_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            in_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
            post_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
        )
    with torch.no_grad():
        for proj in (res.sa_proj, res.mlp_proj):
            proj.weight.copy_(_normal(g, proj.weight.shape, 0.05))
        for norm in (res.sa_norm, res.mlp_norm, res.in_norm, res.post_norm):
            norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
    weights = {kind: _fp8_weight(n, H, 900 + n) for kind, n, _ in CONSUMERS}
    torch.cuda.synchronize()
    BUILT = SimpleNamespace(
        comm=comm, all_reduce=all_reduce, res=res, weights=weights, tail_weight=tail_weight
    )
    if R.rank == 0:
        print(
            f"[rank 0] built at world {R.world}; wide steps hand MXFP8 over: {DC.wide_mxfp8_out()}",
            flush=True,
        )


def check_arms_eager():
    """Every consumer x M x snapshotting or not: off arm = pre-change chain, on arm (early release 0 / 1) = off arm,
    the projection on the offered MXFP8 input and the copy the GEMM's own quantization."""
    for consumer in CONSUMERS:
        for rows in SMALL_ROWS + WIDE_ROWS:
            for snapshot in (False, True):
                where = f"{consumer[0]} M {rows} snapshot {snapshot}"
                call = _Call(100 * rows + 10 * len(consumer[0]) + int(snapshot), rows)
                layer = _decoder(call, consumer, ATTN_RES_BLOCK if snapshot else 1)
                with _arm(0):
                    off = _cloned(_forward(layer, call))
                _verify(
                    f"{where}: off arm vs the pre-change chain",
                    off,
                    _cloned(_ref_chain(layer, call)),
                )
                for early in (0, 1):
                    on = _on_arm(layer, call, early, f"{where} early {early}")
                    _verify(f"{where}: on arm (early release {early}) vs off arm", on, off)
    if R.rank == 0:
        print(f"[rank 0] arms eager: MX_STATS {dict(FL.MX_STATS)}", flush=True)


def check_runtime_flip():
    """One built layer, the switch flipped between calls: off -> on -> off -> on (early release) -> off, each call
    bit for bit the first (the same state, the arm picked per call)."""
    for consumer in CONSUMERS:
        for rows in (8, 16):
            where = f"flip {consumer[0]} M {rows}"
            call = _Call(4000 + rows + len(consumer[0]), rows)
            layer = _decoder(call, consumer, 1)
            with _arm(0):
                first = _cloned(_forward(layer, call))
            for i, (mx, early) in enumerate(((1, 0), (0, 0), (1, 1), (0, 0))):
                if mx:
                    got = _on_arm(layer, call, early, f"{where} call {i + 1}")
                else:
                    with _arm(0):
                        got = _cloned(_forward(layer, call))
                _verify(f"{where}: call {i + 1} (mx {mx} early {early}) vs call 0", got, first)


def check_graph_replay():
    """Each arm captured (after the eager calls above compiled every kernel) and replayed with rewritten inputs; each
    replay bit for bit the eager off arm on the same inputs. The on arm's capture holds the MXFP8 GEMM."""
    for consumer in (CONSUMERS[0], CONSUMERS[2]):
        for rows in (8, 32):
            for snapshot in (False, True):
                where = f"graph {consumer[0]} M {rows} snapshot {snapshot}"
                call = _Call(6000 + rows + int(snapshot), rows)
                layer = _decoder(call, consumer, ATTN_RES_BLOCK if snapshot else 1)
                graphs = {}
                for mx, early in ((0, 0), (1, 0), (1, 1)):
                    with _arm(mx, early):
                        _forward(layer, call)  # eager first: every kernel of the arm compiled
                    R.barrier()
                    before = FL.MX_STATS["mxfp8"]
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with _arm(mx, early), torch.cuda.graph(graph, stream=stream):
                        out = _forward(layer, call)
                    R.barrier()
                    held = FL.MX_STATS["mxfp8"] - before
                    assert held == (1 if mx and _mx_expected(call) else 0), (
                        f"{where}: captured MXFP8 GEMMs {held}"
                    )
                    graphs[(mx, early)] = (graph, out)
                for rep in range(REPLAYS):
                    replayed = {}
                    for key, (graph, out) in graphs.items():
                        call.draw(6100 + 10 * rows + 2 * rep + int(snapshot))
                        R.barrier()
                        graph.replay()
                        torch.cuda.synchronize()
                        replayed[key] = _cloned(out)
                    call.draw(6100 + 10 * rows + 2 * rep + int(snapshot))
                    with _arm(0):
                        eager = _cloned(_forward(layer, call))
                    for key, got in replayed.items():
                        _verify(f"{where} replay {rep}: arm {key} vs eager off arm", got, eager)
                del graphs


CHECKS = [check_arms_eager, check_runtime_flip, check_graph_replay]


def _run_one_rank(args) -> int:
    global R, IMPL, DC, FL, FLAGS
    R = ls.Rank(args)
    import importlib

    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    IMPL = importlib.import_module(f"{package}.implementation")
    DC = importlib.import_module(f"{package}.decode_comm")
    FL = importlib.import_module(f"{package}.fp8w_lowm")
    FLAGS = importlib.import_module(f"{package}.flags")
    code = ls.run_checks(R, [check_build])
    if code:
        return code
    with torch.inference_mode():
        code = ls.run_checks(R, CHECKS)
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

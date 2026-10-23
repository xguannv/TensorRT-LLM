# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA DSpark drafter's context path on a side stream (``kimi_k3_nvfp4/flags.py``:
``TRTLLM_K3LL_DRAFTER_CTX_SIDE``), on the TP group's decode collectives, at W ranks (default 4) with the TP16 per-rank
shapes, 8 to 64 rows (B x (K + 1) of a decode step, B <= 8, K = 7).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_drafter_input_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _kimi_k3_drafter_input_op_matrix.py --launcher srun --world-size 4

Not a pytest module (the ranks share the TP group's decode collectives); the collected entry point is
``test_modeling_v2_kimi_k3_drafter_input_op_matrix.py``.

What runs. The context path the worker forks (``KimiK3MLADSparkWorker._k3_ctx_latents``): the target's split ``fc``
(this rank's columns of a [7168, 3 x 7168] projection, ``K3FcSlice``), its partial products summed with
``hidden_norm`` in one ``comm.allreduce_norm`` on the group's MNNVL workspace (the call ``_gate_mla_drafter_fc``'s
``project_target_hidden`` makes, at ``mla_drafter.ar_ceiling``), and the 5-layer latent GEMM [2880, 7168]. Forked with
``mla_drafter_prep.CtxSide`` while the main stream runs its own work (a GEMM of the size of the target LM head shard,
standing in for the head and the acceptance), then joined.

Checks (each eagerly and from a captured CUDA graph replayed with rewritten inputs, one rank starting late on some
calls):
  * the side-stream latents bit for bit the same chain run on the main stream (the off arm), and equal on every rank;
  * the comm workspace stays in step: a main-stream ``allreduce_norm`` after the join (the drafter's first all-reduce)
    is bit for bit its eager reference.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the Kimi K3 drafter input checks require CUDA devices"

DEADLINE_S = 1800
H = 7168
CAPTURED = 3  # captured target layers
LAYERS = 5
LATENT = 576
ROWS = (8, 16, 32, 56, 64)
GRAPH_ROWS = (16, 64)
REPLAYS = 3

R = None
MD = None  # mla_drafter
DC = None  # decode_comm
PREP = None  # mla_drafter_prep
B = None  # built state


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0):
    return (scale * torch.randn(tuple(shape), generator=g, device="cuda")).bfloat16()


def _same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    )


def build():
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    cols = CAPTURED * H // R.world
    g = _gen(1000 + R.rank)
    fc = _normal(g, (H, cols), (CAPTURED * H) ** -0.5)
    latent_w = _normal(_gen(11), (LAYERS * LATENT, H), H**-0.5)
    norm = RMSNorm(hidden_size=H, eps=1e-5, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_((1 + 0.1 * torch.rand(H, generator=_gen(7), device="cuda")).bfloat16())
    comm = DC.K3DecodeComm.create(R.mapping, None)
    zeros = torch.zeros(max(ROWS), H, dtype=torch.bfloat16, device="cuda")
    head = _normal(_gen(13), (8192, H), H**-0.5)
    return SimpleNamespace(
        fc=fc, start=R.rank * cols, cols=cols, latent_w=latent_w, norm=norm, comm=comm, zeros=zeros,
        head=head, side=PREP.CtxSide(),
    )  # fmt: skip


def _takes(rows) -> bool:
    """Whether the MNNVL workspace takes ``rows`` rows (the worker's ``k3_comm_only``: only then the side stream)."""
    return B.comm.takes_allreduce_norm(rows, H, MD.ar_ceiling(rows))


def _project(hs):
    """``_gate_mla_drafter_fc``'s ``project_target_hidden`` on rows its MNNVL all-reduce takes."""
    rows = hs.shape[0]
    partial = torch.nn.functional.linear(hs[:, B.start : B.start + B.cols], B.fc)
    normed, _ = B.comm.allreduce_norm(partial, B.zeros[:rows], B.norm, MD.ar_ceiling(rows))
    return normed


def _latents(hs):
    """The worker's context path: the projection, then the latent GEMM (``raw_context_latents``)."""
    projected = _project(hs)
    latent = torch.nn.functional.linear(projected, B.latent_w)
    return latent.view(latent.shape[0], LAYERS, 1, LATENT)


def _hidden(rows, seed):
    """The captured rows (replicated: every rank's target computed the same hidden states)."""
    return _normal(_gen(seed), (rows, CAPTURED * H))


def _main_work(x):
    return torch.nn.functional.linear(x, B.head)


def _side_step(hs, x, after):
    """Fork the context path, run the main stream's work beside it, join, then one more all-reduce on the main
    stream (the drafter's first)."""
    latents = B.side.fork(lambda: _latents(hs))
    main = _main_work(x)
    B.side.join(latents)
    next_ar, _ = B.comm.allreduce_norm(
        after, B.zeros[: after.shape[0]], B.norm, MD.ar_ceiling(after.shape[0])
    )
    return latents, main, next_ar


def _inline_step(hs, x, after):
    main = _main_work(x)
    latents = _latents(hs)
    next_ar, _ = B.comm.allreduce_norm(
        after, B.zeros[: after.shape[0]], B.norm, MD.ar_ceiling(after.shape[0])
    )
    return latents, main, next_ar


def _inputs(rows, seed):
    return (
        _hidden(rows, seed),
        _normal(_gen(seed + 1), (rows, H)),
        _normal(_gen(seed * 7 + 3 + R.rank), (rows, H)),
    )


def check_eager():
    for i, rows in enumerate(ROWS):
        if not _takes(rows):
            if R.rank == 0:
                print(
                    f"[side eager] {rows} rows: not on the MNNVL workspace at {R.world} ranks, skipped",
                    flush=True,
                )
            continue
        ins = _inputs(rows, 100 + rows)
        want = _inline_step(*ins)
        R.late(i % R.world)
        got = _side_step(*ins)
        torch.cuda.synchronize()
        for name, a, b in zip(("latents", "main", "next all-reduce"), got, want):
            assert _same(a, b), f"eager, {rows} rows: {name} differs from the main-stream chain"
        assert R.same_on_ranks(got[0]), f"eager, {rows} rows: latents differ across ranks"
        if R.rank == 0:
            print(f"[side eager] {rows} rows: bit for bit", flush=True)


def check_graphs():
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    for rows in GRAPH_ROWS:
        if not _takes(rows):
            continue
        ins = [t.clone() for t in _inputs(rows, 200 + rows)]
        with with_multi_stream(True):
            _side_step(*ins)  # warm up outside the capture
            torch.cuda.synchronize()
            R.barrier()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    outs = _side_step(*ins)
            torch.cuda.current_stream().wait_stream(stream)
        for r in range(REPLAYS):
            for t, n in zip(ins, _inputs(rows, 300 + 10 * r + rows)):
                t.copy_(n)
            R.barrier()
            R.late(r % R.world)
            graph.replay()
            torch.cuda.synchronize()
            got = [o.clone() for o in outs]
            want = _inline_step(*ins)
            torch.cuda.synchronize()
            for name, a, b in zip(("latents", "main", "next all-reduce"), got, want):
                assert _same(a, b), (
                    f"graph, {rows} rows, replay {r}: {name} differs from the eager chain"
                )
            assert R.same_on_ranks(got[0]), f"graph, {rows} rows, replay {r}: ranks differ"
        del graph
        if R.rank == 0:
            print(f"[side graph] {rows} rows: {REPLAYS} replays bit for bit", flush=True)


CHECKS = [check_eager, check_graphs]


def _run_one_rank(args) -> int:
    global R, MD, DC, PREP, B
    R = ls.Rank(args)
    import importlib

    pkg = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    MD = importlib.import_module(pkg + ".mla_drafter")
    DC = importlib.import_module(pkg + ".decode_comm")
    PREP = importlib.import_module(pkg + ".mla_drafter_prep")
    with torch.inference_mode():
        B = build()
        return ls.run_checks(R, CHECKS)


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

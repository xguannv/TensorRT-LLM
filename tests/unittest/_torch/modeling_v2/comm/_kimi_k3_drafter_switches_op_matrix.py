# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA DSpark drafter's switches on its projections and all-reduces (``kimi_k3_nvfp4/flags.py``:
``TRTLLM_K3LL_DRAFTER_WIDE_AR``, ``TRTLLM_K3LL_DRAFTER_GEMM``), each arm against the other and the off arm against
the call it replaced, at W ranks (default 4) with the TP16 per-rank shapes, 8 / 16 / 32 rows.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_drafter_switches_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _kimi_k3_drafter_switches_op_matrix.py --launcher srun --world-size 4

Not a pytest module (the ranks share the TP group's decode collectives); the collected entry point is
``test_modeling_v2_kimi_k3_drafter_switches_op_matrix.py``.

What runs. ``mla_drafter.K3MLADrafterNorms``' per-layer steps after the attention, on the group's ``K3DecodeComm``:
``_project_norm`` (the output projection [7168, 512], a stand-in returning this rank's unreduced product) and
``_mlp_norm`` (a stock ``GatedMLP`` of intermediate 896 per rank, its gate_up [1792, 7168] and down [7168, 896]), with
the stock RMSNorm; no sandwich form and no decode GEMV site (rows past one token tile, where neither takes the call);
and ``K3MLADrafterNorms._gemm`` on each pinned role's weight ([q_a | kv_a] [2112, 7168], q_b [768, 1536], gate_up).
The switches are set on the flags module per arm (both are read at each call).

Checks (each eagerly and, where marked, from a captured CUDA graph replayed with rewritten inputs, each replay bit for
bit the eager call on the same inputs):
  * WIDE_AR (8, 16, 32 rows; graph at 16, 32): off arm bit for bit ``comm.allreduce_norm`` at the decode ceiling (the
    call it replaced), for both steps. On arm: at 8 rows bit for bit the off arm (one token tile: unchanged); above it,
    the output projection on exact payloads (small multiples of powers of two: every sum exact in any order) has
    ``updated`` bit for bit and ``normed`` within ``NORMED_TOL`` of the off arm's; the MLP step within ``MLP_TOL``.
    Outputs equal on every rank.
  * GEMM (8, 16, 32 rows; graph at 16, 32): off arm: ``_gemm`` returns None and ``_mlp_norm`` is bit for bit the
    GatedMLP + ``allreduce_norm`` chain. On arm: each role's output against an fp32 product within ``GEMM_TOL`` of
    its largest magnitude and no worse than ``WORSE`` times cuBLAS's; ``_mlp_norm`` within ``MLP_TOL`` of the off
    arm; a ``dispatch:drafter_gemm_<role>`` count per call.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the Kimi K3 drafter switches require CUDA devices"

DEADLINE_S = 1800
H = 7168
INTER = 896  # the MLP's intermediate columns per rank (TP16's)
O_IN = 512  # 4 heads x 128 per rank
ROWS = (8, 16, 32)
GRAPH_ROWS = (16, 32)
REPLAYS = 2
NORMED_TOL = 2e-2
MLP_TOL = 2e-2
GEMM_TOL = 1e-2
WORSE = 2.0

R = None
MD = None  # mla_drafter
FLAGS = None
CENSUS = None
DC = None  # decode_comm
B = None  # built state


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0):
    return (scale * torch.randn(tuple(shape), generator=g, device="cuda")).bfloat16()


def _same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    )


class _Proj:
    """A row-parallel projection without its all-reduce: this rank's ``x @ weight.T``."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, all_reduce_params=None):
        return torch.nn.functional.linear(x, self.weight)


def build():
    from types import SimpleNamespace

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    mlp = GatedMLP(
        hidden_size=H,
        intermediate_size=INTER * R.world,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(mapping=R.mapping),
    ).cuda()
    g = _gen(1000 + R.rank)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(_normal(g, mlp.gate_up_proj.weight.shape, H**-0.5))
        mlp.down_proj.weight.copy_(_normal(g, mlp.down_proj.weight.shape, INTER**-0.5))
    assert tuple(mlp.gate_up_proj.weight.shape) == (2 * INTER, H)
    assert tuple(mlp.down_proj.weight.shape) == (H, INTER)
    norm = RMSNorm(hidden_size=H, eps=1e-5, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_((1 + 0.1 * torch.rand(H, generator=_gen(7), device="cuda")).bfloat16())
    comm = DC.K3DecodeComm.create(R.mapping, None)
    o_exact = ls.exact_bf16(_gen(2000 + R.rank), (H, O_IN), -4, 5, 1 / 16)
    weights = {
        "q_kv_a": _normal(_gen(11), (2112, H), H**-0.5),
        "q_b": _normal(_gen(12), (768, 1536), 1536**-0.5),
        "gate_up": mlp.gate_up_proj.weight,
    }
    return SimpleNamespace(mlp=mlp, norm=norm, comm=comm, o_exact=_Proj(o_exact), weights=weights)


def _owner(gemm: bool):
    return MD.K3MLADrafterNorms(
        None, B.comm, None, None, frozenset(), gemm=MD._gemm.DrafterGemm() if gemm else None
    )


def _inputs(rows, seed):
    """(exact o_proj input, exact residual, MLP input, residual): the residuals replicated, the inputs per rank."""
    g, gr = _gen(seed), _gen(seed * 31 + 7 + R.rank)
    return (
        ls.exact_bf16(gr, (rows, O_IN), -4, 5, 1 / 8),
        ls.exact_bf16(g, (rows, H), -8, 9, 1 / 4),
        _normal(gr, (rows, H)),
        _normal(g, (rows, H)),
    )


def _steps(owner, x_o, res_o, x_m, res_m):
    n1, u1 = owner._project_norm(x_o, B.o_exact, res_o, B.norm)
    n2, u2 = owner._mlp_norm(B.mlp, x_m, res_m, B.norm)
    return n1, u1, n2, u2


def _replaced(x_o, res_o, x_m, res_m):
    """The calls the switches replaced: the projection without its all-reduce, then ``allreduce_norm`` at the decode
    ceiling."""
    skip = DC.skip_all_reduce()
    n1, u1 = B.comm.allreduce_norm(B.o_exact(x_o), res_o, B.norm)
    n2, u2 = B.comm.allreduce_norm(B.mlp(x_m, final_all_reduce_params=skip), res_m, B.norm)
    return n1, u1, n2, u2


def _set(wide_ar, gemm):
    FLAGS.TRTLLM_K3LL_DRAFTER_WIDE_AR = wide_ar
    FLAGS.TRTLLM_K3LL_DRAFTER_GEMM = gemm


def check_wide_ar():
    owner = _owner(gemm=False)
    for rows in ROWS:
        ins = _inputs(rows, 100 + rows)
        _set(0, 0)
        off = _steps(owner, *ins)
        want = _replaced(*ins)
        assert all(_same(a, b) for a, b in zip(off, want)), (
            f"WIDE_AR off, {rows} rows: not the replaced call"
        )
        _set(1, 0)
        on = _steps(owner, *ins)
        if rows <= DC.SANDWICH_MAX_TOKENS:
            assert all(_same(a, b) for a, b in zip(on, off)), (
                f"WIDE_AR on, {rows} rows: changed within a tile"
            )
        else:
            assert _same(on[1], off[1]), f"WIDE_AR on, {rows} rows: exact updated differs"
            e_n = ls.rel_err(on[0], off[0])
            e_m = max(ls.rel_err(on[2], off[2]), ls.rel_err(on[3], off[3]))
            assert e_n <= NORMED_TOL, (rows, e_n)
            assert e_m <= MLP_TOL, (rows, e_m)
            if R.rank == 0:
                print(f"[wide_ar] {rows} rows: normed {e_n:.2e}, mlp {e_m:.2e}", flush=True)
        assert R.same_on_ranks(*on), f"WIDE_AR on, {rows} rows: ranks differ"
    _set(0, 0)


def check_gemm():
    owner = _owner(gemm=True)
    for rows in ROWS:
        for role, weight in B.weights.items():
            x = _normal(_gen(300 + rows + R.rank), (rows, weight.shape[1]))
            _set(0, 0)
            assert owner._gemm(x, weight, role) is None
            _set(0, 1)
            before = CENSUS.counts().get(f"dispatch:drafter_gemm_{role}:eager", 0)
            got = owner._gemm(x, weight, role)
            assert got is not None, f"GEMM {role} {rows} rows: not taken"
            assert CENSUS.counts().get(f"dispatch:drafter_gemm_{role}:eager", 0) == before + 1
            ref = x.float() @ weight.float().t()
            cublas = torch.nn.functional.linear(x, weight)
            e_on, e_off = ls.rel_err(got, ref), ls.rel_err(cublas, ref)
            assert e_on <= GEMM_TOL and e_on <= max(WORSE * e_off, 2**-8), (role, rows, e_on, e_off)
            if R.rank == 0:
                print(f"[gemm] {role} {rows} rows: {e_on:.2e} (cuBLAS {e_off:.2e})", flush=True)
        ins = _inputs(rows, 400 + rows)
        _set(0, 0)
        off = _steps(owner, *ins)
        want = _replaced(*ins)
        assert all(_same(a, b) for a, b in zip(off, want)), (
            f"GEMM off, {rows} rows: not the replaced chain"
        )
        _set(0, 1)
        on = _steps(owner, *ins)
        e = max(ls.rel_err(on[2], off[2]), ls.rel_err(on[3], off[3]))
        assert e <= MLP_TOL, (rows, e)
        assert all(_same(a, b) for a, b in zip(on[:2], off[:2]))
    _set(0, 0)


def check_graphs():
    for wide_ar, gemm in ((0, 0), (1, 0), (0, 1), (1, 1)):
        owner = _owner(gemm=True)
        for rows in GRAPH_ROWS:
            _set(wide_ar, gemm)
            ins = [t.clone() for t in _inputs(rows, 500 + rows)]
            _steps(owner, *ins)
            torch.cuda.synchronize()
            R.barrier()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    outs = _steps(owner, *ins)
            torch.cuda.current_stream().wait_stream(stream)
            for r in range(REPLAYS):
                for t, n in zip(ins, _inputs(rows, 600 + 10 * r + rows)):
                    t.copy_(n)
                R.barrier()
                graph.replay()
                torch.cuda.synchronize()
                got = [o.clone() for o in outs]
                want = _steps(owner, *ins)
                assert all(_same(a, b) for a, b in zip(got, want)), (
                    f"graph WIDE_AR {wide_ar} GEMM {gemm}, {rows} rows, replay {r}: not the eager call"
                )
            del graph
    _set(0, 0)


CHECKS = [check_wide_ar, check_gemm, check_graphs]


def _run_one_rank(args) -> int:
    global R, MD, FLAGS, CENSUS, DC, B
    R = ls.Rank(args)
    import importlib

    pkg = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    MD = importlib.import_module(pkg + ".mla_drafter")
    FLAGS = importlib.import_module(pkg + ".flags")
    CENSUS = importlib.import_module(pkg + ".census")
    DC = importlib.import_module(pkg + ".decode_comm")
    CENSUS.ENABLED = True
    with torch.inference_mode():
        B = build()
        return ls.run_checks(R, CHECKS)


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

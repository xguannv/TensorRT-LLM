# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/mnnvl_allreduce_rmsnorm_slice`` catalog entry on its ``MnnvlWorkspace``.

The op takes one turn of the workspace's Lamport rotation (two stages), which every MNNVL op of the TP group advances.
Beyond single calls this drives call *sequences*: decode steps whose token count dips and grows back with a random
rank late, the op interleaved with ``comm/mnnvl_allreduce_attn_res`` (one- and two-shot) and
``comm/mnnvl_allgather_split`` on one workspace, and CUDA-graph capture and replay mixed with eager calls.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_allreduce_rmsnorm_slice_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _mnnvl_allreduce_rmsnorm_slice_op_matrix.py --launcher srun \
        --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces and their
rotation). The collected entry point is ``test_modeling_v2_mnnvl_allreduce_rmsnorm_slice_op_matrix.py``.

Every rank draws every rank's inputs from one seed, so each rank holds the whole reference. The addends are random
bf16 (not exact): the reference sums them in fp32 in rank order, the two-shot all-reduce's order, so the reduced row
``r`` is the op's bit for bit, and the slice is compared with ``r * rsqrt(mean(r * r) + eps)`` within one bf16 ulp
(the op sums the squares in another order). Padding and tail are compared bit for bit. Every rank's slice is also
gathered, so the check covers the whole normalized row.
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "mnnvl_allreduce_rmsnorm_slice requires CUDA devices"

DEADLINE_S = 900
SLICE = 224  # Kimi K3's MoE latent columns per rank (3584 over 16 ranks)
BUFFER_BYTES = 4 << 20  # the Lamport buffer size Kimi K3 decodes with
EPS = 1e-6
PAD = 32
TAIL = 192
TOKENS = (1, 2, 7, 8, 9, 16, 17, 32, 33, 48, 64)
DIP_STEPS = (16, 32, 8, 64, 1, 16, 9, 32, 2, 32)
LAYERS = 6
ULP_REL = 2.0**-7  # one bf16 ulp, relative

R = None
entry = None
fits = None
WS_A = None
WS_B = None
STATS = {"ulp_mismatch": 0, "elements": 0}


class Call:
    """One call's arguments on every rank and its reference."""

    def __init__(self, seed, tokens, hidden=None, pad=PAD, tail=TAIL):
        hidden = SLICE * R.world if hidden is None else hidden
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.inputs = [
            (torch.randn(tokens, hidden, generator=g, device="cuda") * 0.5).bfloat16()
            for _ in range(R.world)
        ]
        self.tail = (
            torch.randn(tokens, tail, generator=g, device="cuda").bfloat16() if tail else None
        )
        self.pad = pad

    def reduced(self):
        acc = torch.zeros_like(self.inputs[0], dtype=torch.float32)
        for x in self.inputs:
            acc = acc + x.float()
        return acc.bfloat16()

    def run(self, ws):
        return entry(self.inputs[R.rank], self.tail, self.pad, EPS, ws)


def verify(call: Call, got: torch.Tensor, where: str) -> None:
    r = call.reduced().float()
    want = (r * (r.square().mean(dim=-1, keepdim=True) + EPS).rsqrt()).bfloat16()
    tokens, hidden = r.shape
    width = hidden // R.world
    tail_cols = 0 if call.tail is None else call.tail.shape[1]
    assert got.shape == (tokens, width + call.pad + tail_cols), f"{where}: shape {tuple(got.shape)}"
    torch.cuda.synchronize()
    # The whole normalized row: every rank's slice, in rank order.
    every = R.comm.allgather(got[:, :width].float().cpu())
    full = torch.cat(every, dim=1)
    ref = want.float().cpu()
    diff = (full - ref).abs()
    allowed = ref.abs() * ULP_REL + 1e-30
    assert bool((diff <= allowed).all()), (
        f"{where}: slice off by more than one bf16 ulp (max rel {(diff / allowed).max().item() * ULP_REL:.3e})"
    )
    STATS["ulp_mismatch"] += int((diff > 0).sum())
    STATS["elements"] += diff.numel()
    assert torch.equal(got[:, width : width + call.pad], got.new_zeros(tokens, call.pad)), (
        f"{where}: padding not zero"
    )
    if call.tail is not None:
        assert torch.equal(got[:, width + call.pad :], call.tail), f"{where}: tail differs"


def check_single_calls() -> None:
    for t in TOKENS:
        for pad, tail in ((PAD, TAIL), (0, TAIL), (PAD, 0), (0, 0)):
            call = Call(1000 + 7 * t + pad + tail, t, pad=pad, tail=tail)
            verify(call, call.run(WS_A), f"T {t} pad {pad} tail {tail}")


def check_other_hidden_sizes() -> None:
    """Slices of 8 and 256 columns (the bounds), and a hidden size that is not Kimi K3's."""
    for hidden in (8 * R.world, 256 * R.world, 1024 * R.world // 8):
        for t in (1, 16, 33):
            call = Call(1500 + hidden + t, t, hidden=hidden)
            verify(call, call.run(WS_A), f"hidden {hidden} T {t}")


def check_unsupported_calls_raise_on_every_rank() -> None:
    """``fits`` refuses a call over half a Lamport buffer or a slice that is not whole packets; the op itself raises on
    such a call on every rank and the rotation does not move."""
    big = BUFFER_BYTES // 2 // (SLICE * R.world * 2) + 1
    assert not fits(big, SLICE * R.world, PAD, TAIL, WS_A)
    assert not fits(4, 4 * R.world, PAD, TAIL, WS_A)  # 4-column slices
    assert not fits(4, 512 * R.world, PAD, TAIL, WS_A)  # 512-column slices
    call = Call(1900, big)
    try:
        torch.ops.trtllm.mnnvl_allreduce_rmsnorm_slice(
            call.inputs[R.rank], call.tail, PAD, EPS, R.world,
            WS_A.comm_buffer(torch.bfloat16), WS_A.buffer_flags,
        )  # fmt: skip
        raised = False
    except RuntimeError as exc:
        raised = "exceed half a Lamport buffer" in str(exc)
    assert R.all_true(raised), "a call over half a Lamport buffer did not raise on every rank"
    call = Call(1901, 16)
    verify(call, call.run(WS_A), "after the rejected call")


def check_dip_and_regrow_sequence() -> None:
    """Steps of LAYERS calls whose token count dips and grows back, a random rank late at every call: a call after a
    smaller one must not read what an older, larger call left in the buffer."""
    rng = random.Random(7)
    for step, t in enumerate(DIP_STEPS):
        for layer in range(LAYERS):
            call = Call(3000 + 100 * step + layer, t)
            R.barrier()
            R.late(rng.randrange(R.world))
            verify(call, call.run(WS_A), f"step {step} T {t} layer {layer}")


def check_interleaved_with_the_other_mnnvl_ops() -> None:
    """The op between ``mnnvl_allreduce_attn_res`` (one- and two-shot) and ``mnnvl_allgather_split`` calls on one
    workspace, in a decode-like order, with no host synchronization between the calls of a step."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
        mnnvl_allgather_split,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce_attn_res import (
        mnnvl_allreduce_attn_res,
    )

    hidden = 7168
    for step, t in enumerate((16, 32, 8, 16, 1, 32)):
        g = torch.Generator(device="cuda").manual_seed(4000 + step)
        ar_in = [ls.exact_bf16(g, (t, hidden), -4, 5, 1 / 16) for _ in range(R.world)]
        ones = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
        block = torch.zeros(0, t, hidden, dtype=torch.bfloat16, device="cuda")
        ag_in = [torch.randn(t, 64, generator=g, device="cuda") for _ in range(R.world)]
        calls = [Call(4100 + 10 * step + k, t) for k in range(3)]
        R.barrier()
        outs = []
        for k, call in enumerate(calls):
            normed, updated = mnnvl_allreduce_attn_res(
                ar_in[R.rank],
                None,
                block,
                ones,
                ones,
                ones,
                EPS,
                EPS,
                WS_B,
                two_shot=(k % 2 == 1) and t > 8,
            )
            ag_bf16, ag_fp32 = mnnvl_allgather_split(ag_in[R.rank], 32, WS_B)
            outs.append((call.run(WS_B), updated, ag_bf16, ag_fp32))
        want_updated = sum(x.float() for x in ar_in).bfloat16()
        want_bf16 = torch.cat([x[:, :32].bfloat16() for x in ag_in], dim=1)
        want_fp32 = torch.cat([x[:, 32:] for x in ag_in], dim=1)
        for k, (call, (got, updated, ag_bf16, ag_fp32)) in enumerate(zip(calls, outs)):
            where = f"interleaved step {step} T {t} call {k}"
            verify(call, got, where)
            assert torch.equal(updated, want_updated), f"{where}: the all-reduce's sum differs"
            assert torch.equal(ag_bf16, want_bf16) and torch.equal(ag_fp32, want_fp32), (
                f"{where}: the all-gather differs"
            )


def check_graph_capture_and_replay() -> None:
    """A captured step of LAYERS calls (T 16) replayed with rewritten inputs, eager calls of other shapes on the same
    workspace between replays."""
    t = 16
    calls = [Call(5000 + layer, t) for layer in range(LAYERS)]
    bufs = [(c.inputs[R.rank].clone(), c.tail.clone()) for c in calls]

    def step():
        return [entry(x, tail, PAD, EPS, WS_A) for x, tail in bufs]

    step()
    R.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = step()
    R.barrier()
    for rep in range(6):
        fresh = [Call(6000 + 100 * rep + layer, t) for layer in range(LAYERS)]
        for (x, tail), f, c in zip(bufs, fresh, calls):
            x.copy_(f.inputs[R.rank])
            tail.copy_(f.tail)
            c.inputs, c.tail = f.inputs, f.tail
        R.barrier()
        graph.replay()
        for layer, (c, got) in enumerate(zip(calls, outs)):
            verify(c, got, f"replay {rep} layer {layer}")
        eager = Call(7000 + rep, (9, 1, 32, 64)[rep % 4])
        verify(eager, eager.run(WS_A), f"eager after replay {rep}")
    del graph


CHECKS = [
    check_single_calls,
    check_other_hidden_sizes,
    check_unsupported_calls_raise_on_every_rank,
    check_dip_and_regrow_sequence,
    check_interleaved_with_the_other_mnnvl_ops,
    check_graph_capture_and_replay,
]


def _run_one_rank(args) -> int:
    global R, entry, fits, WS_A, WS_B
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_allreduce_rmsnorm_slice as module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
        MnnvlWorkspace,
    )

    entry = module.mnnvl_allreduce_rmsnorm_slice
    fits = module.fits
    with torch.inference_mode():
        WS_A = MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        WS_B = MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print(
            f"[rank 0] world {R.world}; slice elements off by one ulp: {STATS['ulp_mismatch']} of "
            f"{STATS['elements']}",
            flush=True,
        )
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

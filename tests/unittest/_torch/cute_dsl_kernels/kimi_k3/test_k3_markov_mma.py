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
"""``TRTLLM_K3LL_DRAFTER_MARKOV_V2``: ``trtllm::k3_markov``'s MMA build (``k3_markov_mma_kernel``) against its other
build (``k3_markov_kernel``, the off arm) on W GPUs of one NVLink domain (MPI, one rank per GPU), bf16 block logits of
the TP16 vocab slice (S = 10240), B in 1, 2, 4, 8 requests x K in 1, 7, 8 positions.

Checks (``report``):

- the switch in the model (``KimiK3DSparkWorker._k3_markov_chain`` on a stand-in worker): off, bit for bit the off
  arm's op called directly (corrected, tokens, next_new, the rewound KV lengths); on, bit for bit the MMA build's op
  called directly, and one ``dispatch:drafter_markov_v2`` count per call; ``TRTLLM_K3LL_DRAFTER_MARKOV_FAST`` off
  leaves both arms as they are; on with greedy drafting, tokens, next_new and the rewind bit for bit the MMA build's
  default and the corrected logits a NaN stand-in of their shape; on under advanced draft sampling (or a lilicorr
  drafter), the corrected logits too bit for bit the default's;
- the fast build's variants (``ldsm`` and/or no ``corrected`` store) against the MMA build's default, same inputs:
  tokens, next_new and the rewind bit for bit, the stored corrected logits bit for bit;
- MMA build against the off arm, same inputs: tokens equal, or where they part, the two tokens within the bias's
  rounding (one bf16 ulp of the bias plus 2^-16 of the sum of the products' magnitudes) of each other in the off
  arm's corrected logits (a near tie; the rest of that request is not compared, its anchors differ); corrected within
  that tolerance up to the first parting; next_new = [accepted[rows[b], num_accepted[b] - 1], tokens[b]]; the KV
  rewind identical; the tokens identical on every rank; a rerun bit for bit;
- 40 eager calls with B and K varying, interleaved with MNNVL all-reduces and calls of the off arm (both builds'
  exchange rotations);
- CUDA graphs of [all-reduce, MMA build, MMA build with the rewind, fast build (ldsm, no store) with the rewind] at
  (B, K) in (1, 7), (2, 7), (4, 7), replayed with rewritten inputs, each replay bit for bit the eager calls on the
  same inputs.

Timing (``time``): CUDA graphs of back-to-back calls of each build (the other build, the MMA build, the MMA build with
``TRTLLM_K3LL_DRAFTER_MARKOV_FAST``'s ldsm and no store) on bf16 logits, (B, K) in (1, 7), (2, 7), (4, 7),
markov_w2 cold in L2 (a copy per call), ABBA rounds behind an MPI barrier; the max over the ranks of the median us per
call. The exchange is over the W ranks only (TP16 has 16 x clusters slots).

    srun -n W --mpi=pmix python3 test_k3_markov_mma.py [report | time] [--rounds R]

Under pytest (4 GPUs visible) one test runs ``report`` on 4 ranks under a local ``mpirun -n 4``.
"""

import argparse
import os
import signal
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch

VOCAB = 163840
MARKOV_RANK = 256
SHARD = VOCAB // 16
HIDDEN = 7168
SPLITS = [(b, k) for b in (1, 2, 4, 8) for k in (1, 7, 8)]
GRAPH_SPLITS = ((1, 7), (2, 7), (4, 7))
TIME_SPLITS = ((1, 7), (2, 7), (4, 7))
WORLD = 4
DEADLINE_S = 2400


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not _sm100() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} SM100 / SM103 GPUs (MNNVL multicast, clusters, PDL)",
)


def bits(t):
    return t.contiguous().view(torch.int32)


def identical(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


class Group:
    def __init__(self):
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.rank, self.world = self.comm.Get_rank(), self.comm.Get_size()
        gpus = torch.cuda.device_count()
        torch.cuda.set_device(self.rank % gpus)
        import tensorrt_llm  # noqa: F401
        from tensorrt_llm._torch.cute_dsl_kernels.k3_markov import op
        from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
        from tensorrt_llm.mapping import Mapping

        self.op = op
        self.mapping = Mapping(
            world_size=self.world, rank=self.rank, gpus_per_node=gpus, tp_size=self.world
        )
        self.all_reduce_module = MNNVLAllReduce(self.mapping, torch.bfloat16)
        g = torch.Generator(device="cuda").manual_seed(7)
        self.w1 = (torch.randn(VOCAB, MARKOV_RANK, generator=g, device="cuda") * 0.25).bfloat16()
        self.w2 = (torch.randn(VOCAB, MARKOV_RANK, generator=g, device="cuda") * 0.25).bfloat16()
        self.w2s = self.w2[self.rank * SHARD : (self.rank + 1) * SHARD]
        local = torch.Generator(device="cuda").manual_seed(100 + self.rank)
        self.ar_input = torch.randn(8, HIDDEN, generator=local, device="cuda").bfloat16()
        self.ar_params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            norm_weight=torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda"),
            eps=1e-5,
        )

    def say(self, *a):
        if self.rank == 0:
            print(*a, flush=True)

    def all_ranks(self, ok) -> bool:
        return all(self.comm.allgather(bool(ok)))

    def all_reduce(self):
        return self.all_reduce_module(self.ar_input, all_reduce_params=self.ar_params)

    def inputs(self, batch, block, seed):
        """(base bf16 [B, K, S] of this rank, first_prev int64 [B], (accepted, num_accepted, accepted_rows))."""
        shared = torch.Generator(device="cuda").manual_seed(seed)
        local = torch.Generator(device="cuda").manual_seed(seed + 7919 * (self.rank + 1))
        base = (torch.randn(batch, block, SHARD, generator=local, device="cuda") * 2).bfloat16()
        first = torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda")
        rows = batch + 1
        accepted = torch.randint(
            0, VOCAB, (rows, block + 1), generator=shared, device="cuda", dtype=torch.int32
        )
        num_accepted = torch.randint(
            1, block + 2, (batch,), generator=shared, device="cuda", dtype=torch.int32
        )
        acc_rows = torch.randperm(rows, generator=shared, device="cuda")[:batch].to(torch.int32)
        return base, first, (accepted, num_accepted, acc_rows)

    def rewind(self, batch, seed):
        gen = torch.Generator().manual_seed(seed)
        first = 1
        kv = torch.randint(0, 50, (first + batch + 1,), generator=gen, dtype=torch.int32)
        amounts = torch.randint(-6, 9, (batch,), generator=gen, dtype=torch.int32)
        amounts[0] = kv[first] + 5
        return kv.cuda(), amounts.cuda(), first

    def v1(self, base, first, acc, rewind=None):
        kv, amounts, rf = (None, None, 0) if rewind is None else rewind
        return self.op.markov_chain(
            self.mapping, base, first, self.w1, self.w2s, self.rank * SHARD, *acc,
            kv_lens=kv, rewind=amounts, rewind_first=rf,
        )  # fmt: skip

    def v2(self, base, first, acc, rewind=None, **fast):
        kv, amounts, rf = (None, None, 0) if rewind is None else rewind
        return self.op.markov_chain_mma(
            self.mapping, base, first, self.w1, self.w2s, self.rank * SHARD, *acc,
            kv_lens=kv, rewind=amounts, rewind_first=rf, **fast,
        )  # fmt: skip


#: The fast build's variants: (ldsm, store_corrected).
FAST_VARIANTS = ((True, True), (True, False), (False, False))


def compare(g: Group, base, first, acc, ref, out):
    """(failures, partings, near-tie partings): the MMA build ``out`` against the off arm ``ref``."""
    corrected_ref, tokens_ref, _ = ref
    corrected, tokens, next_new = out
    batch, block = tokens.shape
    fail = 0
    partings = 0
    near = 0
    lo = g.rank * SHARD
    for b in range(batch):
        prev = int(first[b])
        for k in range(block):
            bias = corrected_ref[b, k] - base[b, k].float()
            mag = torch.nn.functional.linear(
                g.w1[prev].float().abs().unsqueeze(0), g.w2s.float().abs()
            )[0]
            tol = bias.abs() * 2.0**-7 + mag * 2.0**-16
            close = bool(((corrected[b, k] - corrected_ref[b, k]).abs() <= tol).all())
            fail += int(not g.all_ranks(close))
            t_ref, t_out = int(tokens_ref[b, k]), int(tokens[b, k])
            if t_ref == t_out:
                prev = t_out
                continue
            partings += 1
            # A near tie: the two tokens' values in the off arm's corrected logits within the tolerance.
            vals = []
            for t in (t_ref, t_out):
                mine = lo <= t < lo + SHARD
                v = float(corrected_ref[b, k, t - lo]) if mine else float("-inf")
                s = float(tol[t - lo]) if mine else 0.0
                vals.append((v, s))
            every = g.comm.allgather(vals)
            v_ref = max(e[0][0] for e in every)
            v_out = max(e[1][0] for e in every)
            slack = max(max(e[0][1], e[1][1]) for e in every)
            if abs(v_ref - v_out) <= 2 * slack:
                near += 1
            else:
                fail += 1
            break
    # next_new from the MMA build's own tokens.
    accepted, num_accepted, rows = acc
    want = torch.cat([accepted[rows.long(), num_accepted.long() - 1].unsqueeze(1), tokens], dim=1)
    fail += int(not identical(next_new, want))
    every = g.comm.allgather(tokens.cpu())
    fail += int(not all(torch.equal(every[0], t) for t in every[1:]))
    return fail, partings, near


def report(g: Group) -> int:
    failures = 0
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        census,
        flags,
        spec_worker,
    )

    # 1. The switch in the model: (MARKOV_V2, MARKOV_FAST, advanced draft sampling, lilicorr drafter).
    census.ENABLED = True
    cases = [(0, 0, False, False), (1, 0, False, False), (0, 1, False, False), (1, 1, False, False),
             (1, 1, True, False), (1, 1, False, True)]  # fmt: skip
    for b, k in ((1, 7), (2, 7), (4, 7)):
        base, first, acc = g.inputs(b, k, 11 + b)
        for v2, fast, advanced, lilicorr in cases:
            flags.TRTLLM_K3LL_DRAFTER_MARKOV_V2 = v2
            flags.TRTLLM_K3LL_DRAFTER_MARKOV_FAST = fast
            kv, amounts, rf = g.rewind(b, 5)
            kv2 = kv.clone()
            worker = SimpleNamespace(
                mapping=g.mapping,
                _k3_acceptance=(
                    acc[0],
                    torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), acc[1]]),
                    1,
                    SimpleNamespace(
                        batch_indices_cuda=torch.cat(
                            [torch.zeros(1, dtype=torch.int32, device="cuda"), acc[2]]
                        ),
                        wants_advanced_draft_sampling=advanced,
                    ),
                    SimpleNamespace(kv_lens_cuda=kv),
                ),
                _kv_rewind_amount=amounts,
                _kv_rewind_pending=True,
                _kv_rewind_bs=rf + b,
                _kv_rewind_nc=rf,
            )
            before = census.counts().get("dispatch:drafter_markov_v2:eager", 0)
            draft_model = SimpleNamespace(markov_w1=g.w1, markov_w2=g.w2)
            if lilicorr:
                draft_model.lilicorr = object()
            got = spec_worker.KimiK3DSparkWorker._k3_markov_chain(
                worker, draft_model, base, first, slice(g.rank * SHARD, (g.rank + 1) * SHARD)
            )
            chain = worker._k3_markov
            # The reference is always the default build of the arm (the fast build must match it bit for bit).
            direct = (g.v2 if v2 else g.v1)(base, first, acc, rewind=(kv2, amounts, rf))
            after = census.counts().get("dispatch:drafter_markov_v2:eager", 0)
            unread = bool(v2 and fast and not advanced and not lilicorr)
            if unread:
                corrected_ok = (
                    tuple(chain[0].shape) == tuple(direct[0].shape)
                    and chain[0].dtype == torch.float32
                    and bool(torch.isnan(chain[0]).all())
                )
            else:
                corrected_ok = identical(chain[0], direct[0])
            ok = (
                got is chain[0]
                and corrected_ok
                and all(identical(x, y) for x, y in zip(chain[1:3], direct[1:]))
                and identical(kv, kv2)
                and after - before == v2
            )
            arm = "MMA build" if v2 else "off arm"
            g.say(
                f"switch v2 {v2} fast {fast} advanced {int(advanced)} lilicorr {int(lilicorr)} B {b} K {k}: "
                + (f"bit for bit the {arm}" + (" (corrected unread: NaN stand-in)" if unread else "")
                   if ok else "FAIL")
            )  # fmt: skip
            failures += int(not g.all_ranks(ok))
    flags.TRTLLM_K3LL_DRAFTER_MARKOV_V2 = 0
    flags.TRTLLM_K3LL_DRAFTER_MARKOV_FAST = 0

    # 1b. The fast build's variants against the MMA build's default.
    g.say("| B | K | ldsm | store | tokens / next_new / rewind | corrected |")
    g.say("| --: | --: | :-- | :-- | :-- | :-- |")
    for b, k in SPLITS:
        base, first, acc = g.inputs(b, k, 3000 + 10 * b + k)
        kv, amounts, rf = g.rewind(b, 31 + b)
        kv_ref = kv.clone()
        ref = g.v2(base, first, acc, rewind=(kv_ref, amounts, rf))
        for ldsm, store in FAST_VARIANTS:
            kv_out = kv.clone()
            out = g.v2(
                base, first, acc, rewind=(kv_out, amounts, rf), ldsm=ldsm, store_corrected=store
            )
            same = (
                identical(out[1], ref[1])
                and identical(out[2], ref[2])
                and identical(kv_out, kv_ref)
            )
            if store:
                corr = identical(out[0], ref[0])
            else:
                corr = tuple(out[0].shape) == (b, k, 0)
            ok = g.all_ranks(same and corr)
            g.say(f"| {b} | {k} | {ldsm} | {store} | {same} | {corr} |")
            failures += int(not ok)

    # 2. MMA build against the off arm.
    g.say("| B | K | failures | partings (near ties) | rerun | rewind |")
    g.say("| --: | --: | --: | --: | :-- | :-- |")
    for b, k in SPLITS:
        base, first, acc = g.inputs(b, k, 1000 + 10 * b + k)
        ref = g.v1(base, first, acc)
        out = g.v2(base, first, acc)
        again = g.v2(base, first, acc)
        rerun = all(identical(x, y) for x, y in zip(out, again))
        kv, amounts, rf = g.rewind(b, 77 + b)
        kv_ref = kv.clone()
        g.v1(base, first, acc, rewind=(kv_ref, amounts, rf))
        rewound = g.v2(base, first, acc, rewind=(kv, amounts, rf))
        rewind_ok = identical(kv, kv_ref) and all(identical(x, y) for x, y in zip(rewound, out))
        fail, partings, near = compare(g, base, first, acc, ref, out)
        fail += int(not g.all_ranks(rerun and rewind_ok))
        partings = sum(g.comm.allgather(partings)) // g.world
        g.say(f"| {b} | {k} | {fail} | {partings} ({near}) | {rerun} | {rewind_ok} |")
        failures += fail

    # 3. Interleaved eager calls.
    ok = True
    for i in range(40):
        b, k = SPLITS[i % len(SPLITS)]
        base, first, acc = g.inputs(b, k, 5000 + i)
        g.all_reduce()
        r1 = g.v1(base, first, acc)
        r2 = g.v2(base, first, acc)
        r2b = g.v2(base, first, acc)
        ok &= all(identical(x, y) for x, y in zip(r2, r2b))
        f, _, _ = compare(g, base, first, acc, r1, r2)
        ok &= f == 0
    ok = g.all_ranks(ok)
    g.say(f"interleaved eager calls: {'PASS' if ok else 'FAIL'}")
    failures += int(not ok)

    # 4. CUDA graphs.
    for b, k in GRAPH_SPLITS:
        base, first, acc = g.inputs(b, k, 9000 + b)
        kv, amounts, rf = g.rewind(b, 9)
        fast = dict(ldsm=True, store_corrected=False)
        g.v2(base, first, acc)
        g.v2(base, first, acc, rewind=(kv.clone(), amounts, rf))
        g.v2(base, first, acc, rewind=(kv.clone(), amounts, rf), **fast)
        kv_f = kv.clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                g.all_reduce()
                o1 = g.v2(base, first, acc)
                o2 = g.v2(base, first, acc, rewind=(kv, amounts, rf))
                o3 = g.v2(base, first, acc, rewind=(kv_f, amounts, rf), **fast)
        torch.cuda.current_stream().wait_stream(stream)
        ok = True
        for r in range(5):
            nb, nf, nacc = g.inputs(b, k, 9100 + 10 * r + b)
            base.copy_(nb)
            first.copy_(nf)
            for x, y in zip(acc, nacc):
                x.copy_(y)
            kv0 = kv.clone()
            kv_f.copy_(kv0)
            graph.replay()
            torch.cuda.synchronize()
            got = [t.clone() for t in (*o1, *o2, *o3[1:])]
            kv_graph = kv.clone()
            kv_graph_f = kv_f.clone()
            kv.copy_(kv0)
            e1 = g.v2(base, first, acc)
            e2 = g.v2(base, first, acc, rewind=(kv, amounts, rf))
            kv_f.copy_(kv0)
            e3 = g.v2(base, first, acc, rewind=(kv_f, amounts, rf), **fast)
            ok &= all(identical(x, y) for x, y in zip(got, (*e1, *e2, *e3[1:])))
            ok &= identical(kv_graph, kv) and identical(kv_graph_f, kv_f) and identical(kv_f, kv)
            # The fast build's tokens / next_new are the default build's.
            ok &= identical(e3[1], e2[1]) and identical(e3[2], e2[2])
        ok = g.all_ranks(ok)
        g.say(f"graph B {b} K {k}: {'PASS' if ok else 'FAIL'}")
        failures += int(not ok)

    g.say("ALL PASS" if failures == 0 else f"FAILURES {failures}")
    return int(failures != 0)


def timing(g: Group, rounds: int):
    g.say("| B | K | k3_markov us | MMA build us | MMA fast us | MMA - other | fast - MMA |")
    g.say("| --: | --: | --: | --: | --: | --: | --: |")
    calls = 8
    for b, k in TIME_SPLITS:
        base, first, acc = g.inputs(b, k, 4242)
        copies = [g.w2s.clone() for _ in range(calls)]
        graphs = {}

        def fast_fn(*args):
            return g.op.markov_chain_mma(*args, ldsm=True, store_corrected=False)

        for name, fn in (
            ("v1", g.op.markov_chain),
            ("v2", g.op.markov_chain_mma),
            ("v2f", fast_fn),
        ):
            for w in copies[:1]:
                fn(g.mapping, base, first, g.w1, w, g.rank * SHARD, *acc)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    for w in copies:
                        fn(g.mapping, base, first, g.w1, w, g.rank * SHARD, *acc)
            torch.cuda.current_stream().wait_stream(stream)
            graphs[name] = graph
        times = {"v1": [], "v2": [], "v2f": []}
        for r in range(rounds):
            for name in ("v1", "v2", "v2f", "v2f", "v2", "v1"):
                g.comm.Barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graphs[name].replay()
                end.record()
                torch.cuda.synchronize()
                times[name].append(start.elapsed_time(end) * 1000 / calls)
        med = {n: max(g.comm.allgather(statistics.median(v))) for n, v in times.items()}
        g.say(
            f"| {b} | {k} | {med['v1']:.1f} | {med['v2']:.1f} | {med['v2f']:.1f} | "
            f"{med['v2'] - med['v1']:+.1f} | {med['v2f'] - med['v2']:+.1f} |"
        )


def launch() -> int:
    command = ["mpirun", "-n", str(WORLD), sys.executable, os.path.abspath(__file__), "report"]
    print(f"[launch] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print(f"[launch] the {WORLD}-rank run did not finish in {DEADLINE_S} s", flush=True)
        return 1


def test_k3_markov_mma():
    """``report`` on WORLD ranks, from a fresh interpreter (pytest's process has initialized MPI)."""
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "launch"],
        capture_output=True,
        text=True,
        timeout=DEADLINE_S + 300,
    )
    print(done.stdout, flush=True)
    assert done.returncode == 0 and "ALL PASS" in done.stdout, (
        done.stdout[-20000:] + done.stderr[-20000:]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="k3_markov MMA build check (module docstring)")
    parser.add_argument("mode", nargs="?", choices=("report", "time", "launch"))
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "launch":
        return launch()
    with torch.inference_mode():
        g = Group()
        status = 0 if args.mode == "time" else report(g)
        if args.mode in (None, "time"):
            timing(g, args.rounds)
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        from mpi4py import MPI

        MPI.COMM_WORLD.Abort(1)

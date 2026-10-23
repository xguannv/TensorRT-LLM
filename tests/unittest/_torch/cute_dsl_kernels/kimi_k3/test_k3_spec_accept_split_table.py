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
"""``trtllm::k3_spec_accept``'s split block table (``split_table=True``, switch ``TRTLLM_K3LL_SPEC_ACCEPT_PAR``)
against the serial build (CTA 0 decodes the draft pool's block table one request after another), bit for bit, on one
GPU.

* Eager: B = 1, 2, 4, 8 requests x K = 7 drafts (block 7, DSpark's), block tables of 300, 2048, 2049 and 32768 columns
  (one partial chunk, one whole chunk, a chunk plus one column, 16 chunks of ``TABLE_COLS`` = 2048), greedy, sampled
  (``samp_*``) and forced-fractional acceptance; 6 consecutive calls with the state carried and fresh block offsets
  per call whose allocated prefix ends at the chunk and thread boundaries (2047, 2048, 2049, 255, 256, 0, 1, the whole
  row) with placeholders (-1), other negative offsets and zero offsets inside it. Every output and every in-place
  state tensor of the split build equals the serial build's, and the table and counts equal the torch decode.
* CUDA graphs: one call of each build captured (block 7 and 8, sampled, 4100 and 32768 columns) and replayed 8 times
  with the logits, drafts, sampling inputs, dummy mask, block offsets and context lengths rewritten in place.
* The switch: one state, a warm eager call with the switch off (``spec_worker.spec_accept_build_kwargs``, which
  compiles both builds), a capture with it off, the switch flipped, a capture with it on and no eager call in between;
  the two graphs replayed on clones of the same state agree bit for bit, and the census counts
  ``dispatch:spec_accept_par:capture`` once.

Timing (us per call, both builds, CUDA graphs of 20 back-to-back calls, median over 12 replays in alternating order):
``python3 test_k3_spec_accept_split_table.py time [--widths 4096,32768]``, or under pytest with
``K3_SPEC_ACCEPT_SPLIT_TIME=1`` (``-s`` to see the table). The vocabulary-sharded build (the model's): the split checks
and the ``split table`` timing column of test_k3_spec_accept_sharded.py.
"""

import argparse
import collections
import importlib.util
import os
import pathlib
import statistics
import sys

import pytest
import torch

_BASE = pathlib.Path(__file__).resolve().parent / "test_k3_spec_accept.py"


def _base():
    name = "_k3_spec_accept_base_for_split_table"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _BASE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="needs SM100 (CTM kernel)",
)

DRAFTS = 7
BLOCK = 7  # DSpark's drafter block (K)
BATCHES = (1, 2, 4, 8)
TABLE_COLS = 2048  # the kernel's TABLE_COLS (test_geometry checks it)
WIDTHS = (300, 2048, 2049, 32768)
GRAPH_WIDTHS = (4100, 32768)
MODES = ("greedy", "sampled", "forced")
KINDS = ("plain", "ties", "nan", "signed")
STEPS = 6
REPLAYS = 8
TP16_COLUMNS = 10240  # a TP16 rank's vocabulary shard: the model's 10 argmax CTAs
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"


# ----------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------


def edge_offsets(gen, max_blocks: int, step: int) -> torch.Tensor:
    """Encoded block offsets [pools, seqs, K / V, max_blocks] (int32, up to 2^31 - 2). In the draft pool's K plane
    request s holds a prefix of (2047, 2048, 2049, 0, 1, 255, 256, max_blocks)[(s + step) % 8] columns (clamped to the
    row), placeholders (-1) after it, and inside it about 1 % placeholders, 0.5 % offsets of -7 and 0.5 % zeros."""
    base = _base()
    off = torch.randint(0, 2**31 - 1, (base.POOLS, base.SEQS, 2, max_blocks), generator=gen, device="cuda",
                        dtype=torch.int32)  # fmt: skip
    edges = torch.tensor(
        [TABLE_COLS - 1, TABLE_COLS, TABLE_COLS + 1, 0, 1, 255, 256, max_blocks], device="cuda"
    )
    seq = torch.arange(base.SEQS, device="cuda")
    used = edges[(seq + step) % len(edges)].clamp(max=max_blocks)
    cols = torch.arange(max_blocks, device="cuda")
    plane = off[base.POOL_IDX, :, 0]
    u = torch.rand(base.SEQS, max_blocks, generator=gen, device="cuda")
    plane.masked_fill_(u < 0.01, -1)
    plane.masked_fill_((u >= 0.01) & (u < 0.015), -7)
    plane.masked_fill_((u >= 0.015) & (u < 0.02), 0)
    plane.masked_fill_(cols[None, :] >= used[:, None], -1)
    return off


def make_state(gen, batch: int, max_blocks: int) -> dict:
    st = _base().make_state(gen, batch, max_blocks)
    st["block_off"] = edge_offsets(gen, max_blocks, 0)
    return st


def step_inputs(gen, batch: int, vocab: int, kind: str, step: int):
    """fp32 logits [B (K + 1), vocab] of ``kind`` and drafts with mixed accepted prefixes."""
    base = _base()
    logits = torch.randn(batch * (DRAFTS + 1), vocab, generator=gen, device="cuda")
    base.shape_logits(logits, gen, kind, step)
    return logits, base.drafts_for(gen, logits, batch, DRAFTS, step)


def sampling(gen, rows: int) -> dict:
    """The op's sampling keywords: every third row greedy (T = 0), the others at 0.7 / 1.0 / 1.3."""
    temps = [0.0 if r % 3 == 0 else (0.7, 1.0, 1.3)[r % 3] for r in range(rows)]
    return dict(
        samp_temp=torch.tensor(temps, dtype=torch.float32, device="cuda"),
        samp_seed=torch.randint(0, 2**62, (rows,), generator=gen, device="cuda", dtype=torch.int64),
        samp_off=torch.randint(0, 2**40, (rows,), generator=gen, device="cuda", dtype=torch.int64),
    )


def table_reference(st: dict, batch: int):
    """DFlashWorker._refresh_ctx_block_tables: (counts int64 [B], table int32 [B, max_blocks])."""
    encoded = st["block_off"][st["pool_idx"], :batch, 0].to(torch.int64)
    counts = (encoded >= 0).sum(dim=1)
    table = encoded.clamp(min=0).div(st["divisor"], rounding_mode="floor").to(torch.int32)
    return counts, table


def call(
    st: dict, logits, draft, force: float, block: int, split: bool, samp=None, **extra
) -> list:
    return _base().fused(
        st, logits, draft, force, block, split_table=split, **(samp or {}), **extra
    )


def compare(res: dict, step: int, batch: int, got, want, st_split: dict, st_serial: dict) -> None:
    """One step: the split build's outputs and state against the serial build's, and its table against torch."""
    base = _base()
    counts, table = table_reference(st_split, batch)
    base.record(res, step, {
        "outputs": (base.OUTPUTS, got, want),
        "state": (base.STATE_KEYS, base.state_of(st_split), base.state_of(st_serial)),
        "table": (("block_counts", "block_tables"),
                  [st_split["block_counts"][:batch], st_split["block_tables"][:batch]], [counts, table]),
    })  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------------------------


def measure(batch: int, max_blocks: int, mode: str, vocab: int = None, steps: int = STEPS) -> dict:
    """``steps`` eager calls of both builds, each on its own copy of one state, the state carried."""
    base = _base()
    base._op()
    vocab = vocab or base.V
    gen = torch.Generator(device="cuda").manual_seed(
        20261023 + 1000 * batch + max_blocks + MODES.index(mode)
    )
    st_serial = make_state(gen, batch, max_blocks)
    st_split = base.clone_state(st_serial)
    force = base.force_value(DRAFTS, "frac") if mode == "forced" else 0.0
    res = base.new_result(
        split=f"{batch}x{DRAFTS + 1}", block=BLOCK, case=f"{mode}, {max_blocks} columns"
    )
    for step in range(steps):
        base.set_dummies((st_serial, st_split), batch, step)
        offsets = edge_offsets(gen, max_blocks, step)
        for st in (st_serial, st_split):
            st["block_off"].copy_(offsets)
        logits, draft = step_inputs(gen, batch, vocab, KINDS[step % len(KINDS)], step)
        samp = sampling(gen, logits.shape[0]) if mode == "sampled" else None
        want = call(st_serial, logits, draft, force, BLOCK, False, samp)
        got = call(st_split, logits, draft, force, BLOCK, True, samp)
        compare(res, step, batch, got, want, st_split, st_serial)
    return base.finish(res)


def measure_graph(batch: int, max_blocks: int, block: int, replays: int = REPLAYS) -> dict:
    """One sampled call of each build captured in its own CUDA graph (its own copy of one state, shared input
    buffers), both replayed ``replays`` times with every input rewritten in place."""
    base = _base()
    base._op()
    gen = torch.Generator(device="cuda").manual_seed(4242 + 100 * batch + max_blocks + block)
    st_serial = make_state(gen, batch, max_blocks)
    st_split = base.clone_state(st_serial)
    logits, draft = step_inputs(gen, batch, base.V, "plain", 0)
    samp = sampling(gen, logits.shape[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outs = [], []
    with torch.cuda.stream(stream):
        for st, split in ((st_serial, False), (st_split, True)):
            call(
                base.clone_state(st), logits, draft, 0.0, block, split, samp
            )  # compiled outside capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outs.append(call(st, logits, draft, 0.0, block, split, samp))
            graphs.append(graph)
    torch.cuda.synchronize()
    res = base.new_result(split=f"{batch}x{DRAFTS + 1}", block=block,
                          case=f"graph x{replays}, sampled, {max_blocks} columns")  # fmt: skip
    for rep in range(replays):
        new_logits, new_draft = step_inputs(gen, batch, base.V, KINDS[rep % len(KINDS)], rep)
        logits.copy_(new_logits)
        draft.copy_(new_draft)
        for key, value in sampling(gen, logits.shape[0]).items():
            samp[key].copy_(value)
        base.set_dummies((st_serial, st_split), batch, rep)
        offsets = edge_offsets(gen, max_blocks, rep)
        ctx = base.context_lengths(gen, st_serial["batch_to_slot"], batch)
        for st in (st_serial, st_split):
            st["block_off"].copy_(offsets)
            st["ctx_len"].copy_(ctx)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        compare(res, rep, batch, outs[1], outs[0], st_split, st_serial)
    del graphs
    return base.finish(res)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("max_blocks", WIDTHS, ids=[f"w{w}" for w in WIDTHS])
@pytest.mark.parametrize("batch", BATCHES, ids=[f"bs{b}" for b in BATCHES])
def test_split_table_eager(batch, max_blocks, mode):
    with torch.inference_mode():
        res = measure(batch, max_blocks, mode)
    assert res["ok"], _base().failures([res])


@pytest.mark.parametrize("block", [BLOCK, DRAFTS + 1], ids=["block=K", "block=K+1"])
@pytest.mark.parametrize("max_blocks", GRAPH_WIDTHS, ids=[f"w{w}" for w in GRAPH_WIDTHS])
@pytest.mark.parametrize("batch", BATCHES, ids=[f"bs{b}" for b in BATCHES])
def test_split_table_graph_replay(batch, max_blocks, block):
    with torch.inference_mode():
        res = measure_graph(batch, max_blocks, block)
    assert res["ok"], _base().failures([res])


@pytest.mark.parametrize("batch", BATCHES, ids=[f"bs{b}" for b in BATCHES])
def test_switch_flips_between_captures(batch, monkeypatch):
    """One built state; the switch read per call by ``spec_worker.spec_accept_build_kwargs``: warm eager call with it
    off, capture A (off), flip, capture B (on) with no eager call in between (both builds already compiled), then A and
    B replayed on clones of the state with the same rewritten inputs agree bit for bit."""
    import importlib

    base = _base()
    base._op()
    flags = importlib.import_module(f"{_PKG}.flags")
    census = importlib.import_module(f"{_PKG}.census")
    spec_worker = importlib.import_module(f"{_PKG}.spec_worker")
    monkeypatch.setattr(census, "ENABLED", True)
    monkeypatch.setattr(census, "_counts", collections.Counter())
    monkeypatch.setattr(flags, "TRTLLM_K3LL_SPEC_ACCEPT_PAR", 0)
    max_blocks = 32768
    gen = torch.Generator(device="cuda").manual_seed(31337 + batch)
    with torch.inference_mode():
        built = make_state(gen, batch, max_blocks)
        st_off, st_on = base.clone_state(built), base.clone_state(built)
        logits, draft = step_inputs(gen, batch, TP16_COLUMNS, "plain", 0)
        samp = sampling(gen, logits.shape[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs, outs = [], []
        with torch.cuda.stream(stream):
            # The warm call (switch off): compiles both builds of the shape.
            base.fused(base.clone_state(built), logits, draft, 0.0, BLOCK, **samp,
                       **spec_worker.spec_accept_build_kwargs())  # fmt: skip
            torch.cuda.synchronize()
            for st, switch in ((st_off, 0), (st_on, 1)):
                monkeypatch.setattr(flags, "TRTLLM_K3LL_SPEC_ACCEPT_PAR", switch)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outs.append(base.fused(st, logits, draft, 0.0, BLOCK, **samp,
                                           **spec_worker.spec_accept_build_kwargs()))  # fmt: skip
                graphs.append(graph)
        torch.cuda.synchronize()
        counts = census.counts()
        assert counts.get("dispatch:spec_accept_par:capture") == 1, counts
        assert "dispatch:spec_accept_par:eager" not in counts, counts
        res = base.new_result(
            split=f"{batch}x{DRAFTS + 1}", block=BLOCK, case="switch flipped between captures"
        )
        for rep in range(REPLAYS):
            new_logits, new_draft = step_inputs(
                gen, batch, TP16_COLUMNS, KINDS[rep % len(KINDS)], rep
            )
            logits.copy_(new_logits)
            draft.copy_(new_draft)
            for key, value in sampling(gen, logits.shape[0]).items():
                samp[key].copy_(value)
            base.set_dummies((st_off, st_on), batch, rep)
            offsets = edge_offsets(gen, max_blocks, rep)
            for st in (st_off, st_on):
                st["block_off"].copy_(offsets)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            compare(res, rep, batch, outs[1], outs[0], st_on, st_off)
        del graphs
    assert res["ok"], base.failures([res])


def test_geometry():
    """``table_geometry``: ``TABLE_COLS`` columns per table CTA, every request at least one."""
    kern = _base()._op()._kernel_module()
    assert kern.TABLE_COLS == TABLE_COLS == kern.THREADS * kern.TABLE_UNROLL
    assert kern.table_geometry(1, 72) == (1, 1)
    assert kern.table_geometry(2, 2048) == (2, 1)
    assert kern.table_geometry(2, 2049) == (4, 2)
    assert kern.table_geometry(4, 32768) == (64, 16)
    assert kern.table_geometry(8, 32768) == (128, 16)


def test_arrival_counter_at_rest_after_split_calls():
    """Split and serial calls of several shapes back to back on one stream, no host synchronization between them:
    every arrival counter is zero once the device is idle (the split build's last CTA counts the table CTAs too)."""
    base = _base()
    op = base._op()
    gen = torch.Generator(device="cuda").manual_seed(20261024)
    with torch.inference_mode():
        calls = []
        for batch, max_blocks, split in [(1, 32768, True), (8, 2049, True), (4, 300, False), (2, 32768, True),
                                         (8, 32768, False)]:  # fmt: skip
            st = make_state(gen, batch, max_blocks)
            for step in range(3):
                logits, draft = step_inputs(gen, batch, base.V, "plain", step)
                calls.append((st, logits, draft, split))
        torch.cuda.synchronize()
        for st, logits, draft, split in calls:
            call(st, logits, draft, 0.0, BLOCK, split)
        torch.cuda.synchronize()
    device = torch.cuda.current_device()
    counters = [scratch[1] for (dev, _), scratch in op._scratch.items() if dev.index == device]
    assert counters and all(int(c.item()) == 0 for c in counters), (
        "an arrival counter is not at rest"
    )


# ----------------------------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------------------------


def timing(widths=(4096, 32768), vocabs=(TP16_COLUMNS,), calls: int = 20, rounds: int = 12) -> list:
    """us per call of the serial and the split build at B = 1, 2, 4, 8 (K = 7, block 7), greedy and sampled."""
    base = _base()
    base._op()
    rows = []
    print(f"{torch.cuda.get_device_name()}; K {DRAFTS}, block {BLOCK}; CUDA graphs of {calls} back-to-back calls, "
          f"median over {rounds} replays in alternating order: us per call")  # fmt: skip
    print(
        "| vocab columns (CTAs) | block-table columns | mode | split | serial | split table | split - serial |"
    )
    print("| --: | --: | :-- | :-- | --: | --: | --: |")
    with torch.inference_mode():
        for vocab in vocabs:
            for max_blocks in widths:
                for mode in ("greedy", "sampled"):
                    for batch in BATCHES:
                        gen = torch.Generator(device="cuda").manual_seed(99 + batch + max_blocks)
                        st_serial = make_state(gen, batch, max_blocks)
                        base.set_dummies((st_serial,), batch, 0)
                        st_split = base.clone_state(st_serial)
                        logits, draft = step_inputs(gen, batch, vocab, "plain", 3)
                        samp = sampling(gen, logits.shape[0]) if mode == "sampled" else None
                        graphs = [
                            base.capture(
                                lambda: call(st_serial, logits, draft, 0.0, BLOCK, False, samp),
                                calls,
                            ),
                            base.capture(
                                lambda: call(st_split, logits, draft, 0.0, BLOCK, True, samp), calls
                            ),
                        ]
                        per_call = [[], []]
                        for rd in range(rounds):
                            for arm in (0, 1) if rd % 2 == 0 else (1, 0):
                                per_call[arm].append(base.replay_us(graphs[arm], calls))
                        serial_us, split_us = (statistics.median(t) for t in per_call)
                        rows.append(dict(vocab=vocab, max_blocks=max_blocks, mode=mode, batch=batch,
                                         serial=serial_us, split=split_us))  # fmt: skip
                        print(f"| {vocab} ({vocab // 1024}) | {max_blocks} | {mode} | {batch}x{DRAFTS + 1} | "
                              f"{serial_us:.2f} | {split_us:.2f} | {split_us - serial_us:+.2f} |",
                              flush=True)  # fmt: skip
                        del graphs
    return rows


@pytest.mark.skipif(
    os.environ.get("K3_SPEC_ACCEPT_SPLIT_TIME") != "1", reason="timing: K3_SPEC_ACCEPT_SPLIT_TIME=1"
)
def test_timing():
    rows = timing()
    assert rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        parser = argparse.ArgumentParser(
            description="k3_spec_accept: serial vs split block table, us per call"
        )
        parser.add_argument("mode", choices=("time",))
        parser.add_argument(
            "--widths", default="4096,32768", help="block-table columns, comma-separated"
        )
        parser.add_argument(
            "--vocabs", default=str(TP16_COLUMNS), help="logits columns, comma-separated"
        )
        parser.add_argument("--rounds", type=int, default=12)
        args = parser.parse_args()
        timing(tuple(int(w) for w in args.widths.split(",")), tuple(int(v) for v in args.vocabs.split(",")),
               rounds=args.rounds)  # fmt: skip
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))

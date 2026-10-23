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
"""``TRTLLM_K3LL_SPEC_HEAD_FUSED``: the spec decode step's head bookkeeping in ``DecoderRunner._preprocess_inputs``
(the overlap scheduler's ``position_ids`` / ``kv_lens_cuda`` correction, the Helix ``helix_position_offsets``
correction and ``recompute_helix_spec_buffers``) as one ``fused_spec_head_prep`` launch, bit for bit (int32).

The runner's own ``_preprocess_inputs`` / ``_fused_spec_head_prep`` and the attention metadata's own
``recompute_helix_spec_buffers`` / ``helix_local_len_vec`` run on stand-in objects that carry only the buffers and
sizes they read, on engine-sized int32 buffers filled at random (what an arm must not touch included) and compared
whole:

* switch off (unset or 0): the fused kernel is never launched and every buffer equals the torch ops' result;
* switch on: every buffer equals the switch-off arm's, eagerly and from a CUDA graph of each arm replayed with every
  input rewritten in place; a replay of the same inputs is bit-identical;
* switch on, a step the kernel does not cover (context tokens in the batch): the torch path runs;
* the runner's fused head-prep count (what its log reports) counts each fused ``_preprocess_inputs`` call Python
  runs, a CUDA graph capture among them, and no graph replay;
* the switch is the K3 flags module's attribute, read on every ``_preprocess_inputs`` call (``k3ll_flag``): on only
  at 1, off with any other value or without the K3 target loaded; registered in ``flags.py`` with default 1 and bound
  by value nowhere else (what the one-process A/B's switch tool requires);
* one runner, built once, the switch flipped between eager steps and between CUDA graph captures (what the
  one-process A/B does), equals a runner of the other arm on every step and replay, and the K3 census key
  ``dispatch:spec_head_fused:<mode>`` counts each fused eager call (``eager``) and capture (``capture``), never a
  replay or a switch-off call.

Batch 1, 2, 4 and 8 at draft length 7 (8 tokens per row); Helix16 at CP ranks 0, 5 and 15 with 64-token pages and
positions straddling page and ledger boundaries, the overlap scheduler on and off, and the TP16 shape (no Helix).
"""

import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    census,
    flags,
)
from tensorrt_llm._torch.pyexecutor import k3ll_flags

DRAFT_LEN = 7
TOKENS = DRAFT_LEN + 1
MAX_BATCH = 8
MAX_NUM_TOKENS = 512
PAGE = 64
CP_SIZE = 16
BATCHES = [1, 2, 4, 8]
NAME = "TRTLLM_K3LL_SPEC_HEAD_FUSED"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def _k3_switch(monkeypatch):
    """The switch at 0 on the K3 flags module (as at launch with its default) and the K3 census on; both restored
    afterwards. ``Step.run`` sets the switch to its arm right before the call."""
    monkeypatch.setattr(flags, NAME, 0)
    monkeypatch.setattr(census, "ENABLED", True)


def dispatches():
    """The census counts of ``dispatch:spec_head_fused:<mode>``: (eager, capture)."""
    c = census.counts()
    return c.get("dispatch:spec_head_fused:eager", 0), c.get("dispatch:spec_head_fused:capture", 0)


# (label, helix, cp_rank, overlap)
SHAPES = [
    ("helix16_rank0_overlap", True, 0, True),
    ("helix16_rank5_overlap", True, 5, True),
    ("helix16_rank15_overlap", True, 15, True),
    ("helix16_rank5_no_overlap", True, 5, False),
    ("tp16_overlap", False, 0, True),
]
CASES = [(b, *s) for b in BATCHES for s in SHAPES]
IDS = [f"bs{b}-{s[0]}" for b, *s in CASES]


def _runner_cls():
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

    return DecoderRunner


def _metadata_cls():
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    return TrtllmAttentionMetadata


class Step:
    """One decode step's runner and attention-metadata stand-ins over int32 device buffers.

    ``batch`` generation rows of TOKENS tokens (after ``num_ctx`` context rows of ``ctx_tokens`` tokens each).
    """

    def __init__(self, batch, helix, cp_rank, overlap, fused, seed, num_ctx=0, ctx_tokens=0):
        self.batch, self.helix, self.overlap = batch, helix, overlap
        self.num_ctx, self.ctx_tokens = num_ctx, ctx_tokens
        self.g = torch.Generator().manual_seed(seed)
        names = {
            "position_ids": MAX_NUM_TOKENS,
            "pos_off": MAX_NUM_TOKENS,
            "kv_off": MAX_BATCH,
            "kv_lens": MAX_BATCH,
            "helix_pos": MAX_NUM_TOKENS,
            "helix_slots": MAX_NUM_TOKENS,
            "helix_bounds": MAX_NUM_TOKENS,
        }
        self.bufs = {k: torch.empty(n, dtype=torch.int32, device="cuda") for k, n in names.items()}
        self.num_tokens = num_ctx * ctx_tokens + batch * TOKENS
        b = self.bufs
        mapping = types.SimpleNamespace(
            cp_size=CP_SIZE, cp_rank=cp_rank, has_cp_helix=lambda: helix
        )
        md = types.SimpleNamespace(
            kv_cache_manager=types.SimpleNamespace(tokens_per_block=PAGE),
            mapping=mapping,
            num_seqs=num_ctx + batch,
            num_contexts=num_ctx,
            num_generations=batch,
            num_ctx_tokens=num_ctx * ctx_tokens,
            num_chunked_ctx_requests=0,
            kv_lens_cuda=b["kv_lens"],
            helix_position_offsets=b["helix_pos"] if helix else None,
            helix_local_slots=b["helix_slots"] if helix else None,
            helix_kv_bounds=b["helix_bounds"] if helix else None,
            _helix_spec_tokens_valid=helix,
            on_update_kv_lens=lambda: None,
        )
        meta = _metadata_cls()
        md.helix_local_len_vec = types.MethodType(meta.helix_local_len_vec, md)
        md.recompute_helix_spec_buffers = types.MethodType(meta.recompute_helix_spec_buffers, md)
        # Every DecoderRunner attribute _preprocess_inputs reads on this path, as __init__ sets it. The switch
        # itself is read per call from the K3 flags module (set by run()).
        runner = types.SimpleNamespace(
            _config=types.SimpleNamespace(disable_overlap_scheduler=not overlap),
            mapping=mapping,
            previous_pos_id_offsets_cuda=b["pos_off"],
            previous_kv_lens_offsets_cuda=b["kv_off"],
            get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
            guided_decoder=None,
            _spec_head_fusable=True,
            _spec_head_fused_steps=0,
            _spec_head_fused_captures=0,
        )
        self.runner = runner
        self.fused = fused
        cls = _runner_cls()
        runner._fused_spec_head_prep = types.MethodType(cls._fused_spec_head_prep, runner)
        self.preprocess = types.MethodType(cls._preprocess_inputs, runner)
        self.inputs = {
            "attn_metadata": md,
            "input_ids": torch.zeros(self.num_tokens, dtype=torch.int32, device="cuda"),
            "position_ids": b["position_ids"][: self.num_tokens].unsqueeze(0),
        }
        self.rewrite()

    def rewrite(self):
        """New random contents for every buffer, in place: each generation row's positions run on from a base
        within a token of a 64-token page or a 1024-token ledger boundary (so a group straddles them), its position
        offsets are its accepted length 1 .. 8 on each token, its KV-length offset that minus 8."""
        g, b = self.g, self.bufs
        for t in b.values():
            t.copy_(torch.randint(-(1 << 20), 1 << 20, t.shape, generator=g, dtype=torch.int32))
        gen = self.batch * TOKENS
        first = self.num_ctx * self.ctx_tokens
        boundary = torch.randint(1, 200, (self.batch,), generator=g) * PAGE
        boundary[::2] = torch.randint(1, 12, (len(boundary[::2]),), generator=g) * PAGE * CP_SIZE
        base = boundary - torch.randint(1, TOKENS + 1, (self.batch,), generator=g)
        positions = (base[:, None] + torch.arange(TOKENS)[None, :]).flatten().to(torch.int32)
        accepted = torch.randint(1, TOKENS + 1, (self.batch,), generator=g, dtype=torch.int32)
        b["position_ids"][first : first + gen].copy_(positions)
        b["helix_pos"][:gen].copy_(positions)
        b["pos_off"][:gen].copy_(accepted.repeat_interleave(TOKENS))
        b["kv_off"][: self.batch].copy_(accepted - TOKENS)
        b["kv_lens"][self.num_ctx : self.num_ctx + self.batch].copy_(
            (base + TOKENS).to(torch.int32)
        )

    def run(self):
        """``_preprocess_inputs`` with the switch at this step's ``fused`` (read by the runner during the call)."""
        setattr(flags, NAME, int(self.fused))
        self.preprocess(self.inputs, enable_spec_decode=True, runtime_draft_len=DRAFT_LEN)

    def fused_counts(self):
        """The runner's count of fused head preps: (all, under CUDA graph capture)."""
        return self.runner._spec_head_fused_steps, self.runner._spec_head_fused_captures


def torch_reference(st: Step):
    """The torch ops of ``_preprocess_inputs`` for this step, written out (the overlap correction, then the Helix
    position correction and the metadata's own ``recompute_helix_spec_buffers``)."""
    b, md = st.bufs, st.inputs["attn_metadata"]
    first = md.num_ctx_tokens
    gen = st.num_tokens - first
    if st.overlap:
        st.inputs["position_ids"][0, first:] += b["pos_off"][:gen]
        b["kv_lens"][md.num_contexts : md.num_seqs] += b["kv_off"][: md.num_generations]
    if st.helix:
        if st.overlap:
            b["helix_pos"][:gen] += b["pos_off"][:gen]
        md.recompute_helix_spec_buffers(gen, TOKENS)


def clone(bufs):
    return {k: t.clone() for k, t in bufs.items()}


def set_bufs(st, values):
    for k, t in st.bufs.items():
        t.copy_(values[k])


def diff(x, y):
    return [k for k in x if not torch.equal(x[k], y[k])]


def _forbid_fused(monkeypatch):
    from tensorrt_llm._torch.speculative import spec_head_kernels

    def forbidden(*args, **kwargs):
        raise AssertionError("fused_spec_head_prep launched with the switch off")

    monkeypatch.setattr(spec_head_kernels, "fused_spec_head_prep", forbidden)


@pytest.mark.parametrize("value", [None, 0, 1, 2])
def test_switch_read(monkeypatch, value):
    """The runner reads the flags module's value at the call: fused only at 1; 0, 2 and no K3 target loaded (the
    flags module not imported) keep the torch path."""
    st = Step(2, True, 5, True, fused=False, seed=3)
    if value is None:
        monkeypatch.delitem(sys.modules, k3ll_flags.K3LL_FLAGS_MODULE)
    else:
        monkeypatch.setattr(flags, NAME, value)
    if value != 1:
        _forbid_fused(monkeypatch)
    eager0, capture0 = dispatches()
    st.preprocess(st.inputs, enable_spec_decode=True, runtime_draft_len=DRAFT_LEN)
    torch.cuda.synchronize()
    on = value == 1
    assert st.fused_counts() == (int(on), 0)
    assert dispatches() == (eager0 + on, capture0)


def test_switch_registered_and_not_bound():
    """Registered in the K3 flags with default 1, and bound by value in no other loaded module of the K3 package nor
    in the runner (the one-process A/B's switch tool refuses a switch bound at import)."""
    assert flags.flags()[NAME][1] == 1
    runner_module = sys.modules[_runner_cls().__module__]
    package = k3ll_flags.K3LL_FLAGS_MODULE.rsplit(".", 1)[0]
    bound = [
        name
        for name, module in list(sys.modules.items())
        if module is not None
        and (name.startswith(package) or module is runner_module)
        and name != k3ll_flags.K3LL_FLAGS_MODULE
        and NAME in getattr(module, "__dict__", {})
    ]
    assert not bound, bound


@pytest.mark.parametrize("batch,label,helix,cp_rank,overlap", CASES, ids=IDS)
def test_switch_off_unchanged(monkeypatch, batch, label, helix, cp_rank, overlap):
    """Switch off: no fused launch, every buffer as the torch ops leave it."""
    _forbid_fused(monkeypatch)
    st = Step(
        batch, helix, cp_rank, overlap, fused=False, seed=11 * batch + cp_rank + 100 * overlap
    )
    start = clone(st.bufs)
    torch_reference(st)
    want = clone(st.bufs)
    set_bufs(st, start)
    st.run()
    torch.cuda.synchronize()
    assert not diff(want, st.bufs), diff(want, st.bufs)
    assert st.fused_counts() == (0, 0)


@pytest.mark.parametrize("batch,label,helix,cp_rank,overlap", CASES, ids=IDS)
def test_switch_on_equals_off(batch, label, helix, cp_rank, overlap):
    """Switch on vs off, eagerly and from a CUDA graph of each arm replayed with every input rewritten in place."""
    seed = 7 * batch + cp_rank + 100 * overlap
    off = Step(batch, helix, cp_rank, overlap, fused=False, seed=seed)
    on = Step(batch, helix, cp_rank, overlap, fused=True, seed=seed)
    assert not diff(off.bufs, on.bufs)
    start = clone(on.bufs)
    off.run()
    on.run()  # also compiles the kernel, outside capture
    torch.cuda.synchronize()
    assert not diff(off.bufs, on.bufs), ("eager", diff(off.bufs, on.bufs))
    set_bufs(on, start)
    on.run()
    torch.cuda.synchronize()
    assert not diff(off.bufs, on.bufs), ("eager rerun", diff(off.bufs, on.bufs))
    assert on.fused_counts() == (2, 0) and off.fused_counts() == (0, 0)

    graphs = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for st in (off, on):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                st.run()
            graphs.append(graph)
    torch.cuda.synchronize()
    # The capture runs Python once; the replays below rerun the captured launch without it.
    assert on.fused_counts() == (3, 1) and off.fused_counts() == (0, 0)
    for replay in range(4):
        off.rewrite()
        on.rewrite()
        start = clone(on.bufs)
        assert not diff(off.bufs, on.bufs)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        assert not diff(off.bufs, on.bufs), (f"replay {replay}", diff(off.bufs, on.bufs))
        eager = Step(batch, helix, cp_rank, overlap, fused=False, seed=0)
        set_bufs(eager, start)
        eager.run()
        torch.cuda.synchronize()
        assert not diff(eager.bufs, on.bufs), (
            f"replay {replay} vs eager",
            diff(eager.bufs, on.bufs),
        )
    after = clone(on.bufs)
    set_bufs(on, start)
    graphs[1].replay()
    torch.cuda.synchronize()
    assert not diff(after, on.bufs), ("same inputs replayed", diff(after, on.bufs))
    assert on.fused_counts() == (3, 1)


@pytest.mark.parametrize("helix", [True, False], ids=["helix16", "tp16"])
def test_switch_on_context_step_takes_torch_path(monkeypatch, helix):
    """A step with context tokens is outside what the kernel covers: with the switch on, nothing is launched and the
    torch path writes what it writes with the switch off."""
    off = Step(2, helix, 5, True, fused=False, seed=31, num_ctx=1, ctx_tokens=5)
    on = Step(2, helix, 5, True, fused=True, seed=31, num_ctx=1, ctx_tokens=5)
    _forbid_fused(monkeypatch)
    off.run()
    on.run()
    torch.cuda.synchronize()
    assert not diff(off.bufs, on.bufs), diff(off.bufs, on.bufs)
    assert on.fused_counts() == (0, 0)


@pytest.mark.parametrize("batch,label,helix,cp_rank,overlap", CASES, ids=IDS)
def test_flip_between_steps_and_captures(batch, label, helix, cp_rank, overlap):
    """One runner, built once, the switch flipped on the K3 flags module between eager steps and between CUDA graph
    captures (what the one-process A/B does: set the switch on the flags module, release the graphs, recapture).
    Every step and every replay equals a runner of the other arm on the same inputs; a graph keeps the arm current at
    its capture whatever the switch says at replay; the census key and the runner's counts see each fused eager call
    and capture, no replay and no switch-off call."""
    seed = 13 * batch + cp_rank + 100 * overlap
    flip = Step(batch, helix, cp_rank, overlap, fused=False, seed=seed)
    # The other arm's runner for each value of flip's switch: off for a fused flip, on for a torch-path flip.
    other = {
        True: Step(batch, helix, cp_rank, overlap, fused=False, seed=seed),
        False: Step(batch, helix, cp_rank, overlap, fused=True, seed=seed),
    }
    eager0, capture0 = dispatches()

    def step_pair(arm, run_flip):
        flip.rewrite()
        ref = other[arm]
        for st in other.values():
            st.rewrite()
        run_flip()
        ref.run()
        torch.cuda.synchronize()
        return ref

    def check(ref, what):
        assert not diff(ref.bufs, flip.bufs), (what, diff(ref.bufs, flip.bufs))

    def expected_dispatches(captures):
        # Every fused Python call of flip and of the switch-on reference; captures are flip's only.
        eager = flip.fused_counts()[0] - captures + other[False].fused_counts()[0]
        return eager0 + eager, capture0 + captures

    # Eager steps, the switch flipped between them: off, on, on, off, on.
    flip_on = 0
    for i, arm in enumerate((False, True, True, False, True)):
        flip.fused = arm
        ref = step_pair(arm, flip.run)
        check(ref, f"eager step {i} switch {int(arm)}")
        flip_on += arm
        assert flip.fused_counts() == (flip_on, 0)
        assert dispatches() == expected_dispatches(0), (i, dispatches())
    assert other[True].fused_counts() == (0, 0)

    stream = torch.cuda.Stream()
    captures = 0
    # Capture with the switch off, then on, then off again; each graph replayed with rewritten inputs, the switch
    # flipped after capture (a captured graph does not read it).
    for arm in (False, True, False):
        flip.fused = arm
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                flip.run()
        torch.cuda.synchronize()
        flip_on += arm
        captures += arm
        assert flip.fused_counts() == (flip_on, captures)
        assert dispatches() == expected_dispatches(captures)
        for replay in range(3):
            setattr(flags, NAME, int(not arm))
            ref = step_pair(arm, graph.replay)
            check(ref, f"graph switch {int(arm)} replay {replay}")
        assert flip.fused_counts() == (flip_on, captures), "a replay ran Python"
        assert dispatches() == expected_dispatches(captures), "a replay ran Python"
        del graph
    assert flip.fused_counts() == (4, 1)
    assert other[True].fused_counts() == (0, 0)

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
"""``TRTLLM_K3LL_SPEC_STEP_COPIES_SM103``: the spec step-copy kernels (``SlotScatter``, ``StepInputGather``) on SM 103
against the torch path SM 103 takes without the switch, bit for bit (int32).

The store update runs through ``SpecSampler._update_stores`` of two real samplers, one built with the switch off and
one with it on, over the sampler's own stores (draft length 7: [8, 8 slots, 1], [8, 8 slots, 1], [8], [8, 7]); the
overlap gathers of the next step read those stores into engine-sized buffers, by ``StepInputGather`` and by the
engine's torch ops (``torch_gather`` of ``test_spec_step_copies``, a copy of ``_prepare_tp_inputs``'s fallback).

* switch off (unset or 0): ``is_supported()`` is False on SM 103 (True on SM 100, as before), the sampler builds no
  ``SlotScatter`` and its store update is the torch ops';
* switch on, batch 1, 2 and 4 x 8 tokens: every gathered buffer and every store equals the switch-off arm's, except
  ``new_tokens`` past a row's accepted length, which the kernel stores as zeros where the torch ops store the
  forward's raw columns (no reader looks there); eagerly over chained steps, and from a CUDA graph of the scatter
  and the gather replayed with every input rewritten in place, a replay of the same inputs bit-identical;
* a run-time reader installed (``set_sm103_switch``, what the K3 model installs: its flags module): the sampler
  builds its state and compiles the kernel at load whatever the switch says, the environment is no longer read,
  and one sampler, built once, with the switch flipped between chained steps (the gathers chosen per step as
  ``_prepare_tp_inputs`` chooses them) equals the switch-off arm on every step; with the switch off it touches
  neither the kernel nor its slot table, and the installed ``dispatched`` callback (the K3 census key
  ``dispatch:spec_step_copies_sm103:eager``) fires once per launch, only with the switch on.
"""

import pytest
import torch

DRAFT_LEN = 7
TOKENS = DRAFT_LEN + 1
NUM_SLOTS = 8
MAX_NUM_TOKENS = 8192
DRAFT_BUFFER = DRAFT_LEN * NUM_SLOTS
BATCHES = [1, 2, 4]
ENV = "TRTLLM_K3LL_SPEC_STEP_COPIES_SM103"


def _op():
    from tensorrt_llm._torch.cute_dsl_kernels.spec_step_copies import op

    return op


def _sm() -> int:
    from tensorrt_llm._utils import get_sm_version

    return get_sm_version()


def _runnable() -> bool:
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    return torch.cuda.is_available() and IS_CUTLASS_DSL_AVAILABLE and _sm() in (100, 103)


pytestmark = pytest.mark.skipif(
    not _runnable(), reason="the step-copy kernels run on SM 100 / 103 with the CuTe DSL"
)


def _sampler():
    from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    return SpecSampler(
        TorchSampler.Args(
            max_seq_len=4096,
            max_draft_len=DRAFT_LEN,
            max_num_sequences=NUM_SLOTS,
            max_beam_width=1,
            max_total_draft_tokens=DRAFT_LEN,
        )
    )


@pytest.fixture(autouse=True)
def _no_reader():
    """Tests start and end with the environment deciding (no run-time reader installed)."""
    _op().set_sm103_switch(None)
    yield
    _op().set_sm103_switch(None)


def _set_switch(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)


def torch_gather(store, slots, pos_index, rows, bufs):
    """``_prepare_tp_inputs``'s torch overlap gathers of ``rows`` rows (no request without a previous batch first)."""
    previous_slots = slots[:rows]
    tokens = rows * TOKENS
    new_tokens = store.next_new_tokens.transpose(0, 1)[previous_slots, :TOKENS].flatten()
    bufs["input_ids"][:tokens].copy_(new_tokens, non_blocking=True)
    bufs["draft"][: rows * DRAFT_LEN].copy_(
        store.next_draft_tokens[previous_slots, :DRAFT_LEN].flatten(), non_blocking=True
    )
    kv_len_offsets = store.new_tokens_lens - TOKENS
    bufs["pos_off"][:tokens].copy_(store.new_tokens_lens[pos_index[:tokens]], non_blocking=True)
    bufs["kv_off"][:rows].copy_(kv_len_offsets[previous_slots], non_blocking=True)


def kernel_gather(store, slots, pos_index, rows, bufs):
    """The same gathers by ``StepInputGather``, which must cover them."""
    assert _op().StepInputGather().gather(
        store.next_new_tokens, store.next_draft_tokens, store.new_tokens_lens, slots, pos_index, rows, TOKENS,
        DRAFT_LEN, bufs["input_ids"], 0, bufs["draft"], 0, bufs["pos_off"], 0, bufs["kv_off"], 0,
    ), "gather declined a covered call"  # fmt: skip


class Arm:
    """One arm's sampler stores and the engine's gather buffers, all random at the start."""

    def __init__(self, sampler, g):
        self.sampler = sampler
        self.store = sampler.store
        shapes = {
            "input_ids": MAX_NUM_TOKENS,
            "draft": DRAFT_BUFFER,
            "pos_off": MAX_NUM_TOKENS,
            "kv_off": NUM_SLOTS,
        }
        self.bufs = {k: torch.empty(n, dtype=torch.int32, device="cuda") for k, n in shapes.items()}
        for t in self.tensors().values():
            t.copy_(torch.randint(-(1 << 30), 1 << 30, t.shape, generator=g, dtype=torch.int32))

    def tensors(self):
        s = self.store
        return dict(self.bufs, new_tokens=s.new_tokens, next_new_tokens=s.next_new_tokens,
                    new_tokens_lens=s.new_tokens_lens, next_draft_tokens=s.next_draft_tokens)  # fmt: skip

    def snapshot(self):
        return {k: t.clone() for k, t in self.tensors().items()}

    def restore(self, values):
        for k, t in self.tensors().items():
            t.copy_(values[k])


def step_inputs(rows, g):
    """A pure-generation step's forward outputs (every column written, as the one-model worker writes a generation
    row; accepted lengths 1 .. 8), its distinct slots in a shuffled order, and the next step's slots (another order)
    and per-token index list (each row's slot, once per token)."""
    rnd = lambda *shape: torch.randint(0, 1 << 20, shape, generator=g, dtype=torch.int32).cuda()  # noqa: E731
    outputs = {
        "new_tokens": rnd(rows, TOKENS),
        "next_new_tokens": rnd(rows, TOKENS),
        "new_tokens_lens": torch.randint(
            1, TOKENS + 1, (rows,), generator=g, dtype=torch.int32
        ).cuda(),
        "next_draft_tokens": rnd(rows, DRAFT_LEN),
    }
    slots = torch.randperm(NUM_SLOTS, generator=g)[:rows].tolist()
    nxt = slots[1:] + slots[:1]
    prev_slots = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int32)
    prev_slots[:rows] = torch.tensor(nxt, dtype=torch.int32)
    pos_index = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int32)
    pos_index[: rows * TOKENS] = torch.tensor(nxt, dtype=torch.int32).repeat_interleave(TOKENS)
    return outputs, slots, prev_slots.cuda(), pos_index.cuda()


def compare(off, on, slots, outputs):
    """Names of what differs between the arms: whole buffers, except the store's ``new_tokens`` of each scattered
    slot past its accepted length, which must be zero in the switch-on arm."""
    a, b = off.snapshot(), on.snapshot()
    lens = outputs["new_tokens_lens"].cpu()
    new_a, new_b = a.pop("new_tokens")[:, :, 0].clone(), b.pop("new_tokens")[:, :, 0].clone()
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    for r, s in enumerate(slots):
        n = int(lens[r])
        if bool((new_b[n:, s] != 0).any()):
            bad.append(f"new_tokens[{n}:, {s}] not zero")
        new_a[n:, s] = 0
    if not torch.equal(new_a, new_b):
        bad.append("new_tokens")
    return bad


@pytest.mark.parametrize("value", [None, "0"], ids=["unset", "0"])
def test_switch_off_keeps_torch_path(monkeypatch, value):
    """Switch off: SM 103 keeps the torch path (SM 100 the kernels, as before), and the sampler's store update is
    ``_scatter_to_stores``'s, bit for bit."""
    _set_switch(monkeypatch, value)
    sampler = _sampler()
    if _sm() == 100:
        assert _op().is_supported() and sampler._store_scatter is not None
        return
    assert not _op().is_supported()
    assert sampler._store_scatter is None
    g = torch.Generator().manual_seed(3)
    off = Arm(sampler, g)
    ref = Arm(_sampler(), torch.Generator().manual_seed(3))
    for rows in BATCHES:
        outputs, slots, _, _ = step_inputs(rows, g)
        off.sampler._update_stores(outputs, 0, slots)
        ref.sampler._scatter_to_stores(outputs, 0, slots)
        torch.cuda.synchronize()
        assert not [k for k, t in off.snapshot().items() if not torch.equal(t, ref.snapshot()[k])]


def _arms(monkeypatch, seed):
    _set_switch(monkeypatch, "0")
    off_sampler = _sampler()
    _set_switch(monkeypatch, "1")
    assert _op().is_supported()
    on_sampler = _sampler()
    assert on_sampler._store_scatter is not None
    if _sm() == 103:
        assert off_sampler._store_scatter is None
    else:  # SM 100: the off arm also runs the kernel; compare against the torch ops instead
        off_sampler._store_scatter = None
    return Arm(off_sampler, torch.Generator().manual_seed(seed)), Arm(
        on_sampler, torch.Generator().manual_seed(seed)
    )


@pytest.mark.parametrize("rows", BATCHES, ids=[f"bs{r}" for r in BATCHES])
def test_switch_on_equals_off_eager(monkeypatch, rows):
    """Switch on vs off over four chained steps (each step's gather reads the stores the scatter just wrote). A slot
    keeps the dead ``new_tokens`` columns of its last scatter, so after each step's comparison the switch-off arm takes
    the switch-on arm's zeros there (everything else having compared equal) and the next step compares that step."""
    off, on = _arms(monkeypatch, 40 + rows)
    assert not [k for k, t in off.snapshot().items() if not torch.equal(t, on.snapshot()[k])]
    g = torch.Generator().manual_seed(rows)
    for step in range(4):
        outputs, slots, prev_slots, pos_index = step_inputs(rows, g)
        off.sampler._update_stores(outputs, 0, slots)
        torch_gather(off.store, prev_slots, pos_index, rows, off.bufs)
        on.sampler._update_stores(outputs, 0, slots)
        kernel_gather(on.store, prev_slots, pos_index, rows, on.bufs)
        torch.cuda.synchronize()
        bad = compare(off, on, slots, outputs)
        assert not bad, (f"step {step}", bad)
        off.store.new_tokens.copy_(on.store.new_tokens)


@pytest.mark.parametrize("rows", BATCHES, ids=[f"bs{r}" for r in BATCHES])
def test_switch_on_equals_off_graph_replay(monkeypatch, rows):
    """A CUDA graph of the switch-on arm's scatter and gather (the launches ``_update_stores`` and the engine make),
    replayed with every input rewritten in place, against the switch-off arm run eagerly on the same inputs."""
    off, on = _arms(monkeypatch, 80 + rows)
    g = torch.Generator().manual_seed(100 + rows)
    outputs, slots, prev_slots, pos_index = step_inputs(rows, g)
    on.sampler._slot_table_buffer.update(torch.tensor(slots, dtype=torch.int32))
    slot_table = on.sampler._slot_table_buffer.tensor
    scatter = _op().SlotScatter()

    def body():
        s = on.store
        assert scatter.scatter(outputs, 0, rows, slot_table, s.new_tokens, s.next_new_tokens, s.new_tokens_lens,
                               s.next_draft_tokens)  # fmt: skip
        kernel_gather(on.store, prev_slots, pos_index, rows, on.bufs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()  # compiles outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
    torch.cuda.synchronize()
    start = None
    for replay in range(4):
        new_outputs, new_slots, new_prev, new_pos = step_inputs(rows, g)
        for k, t in outputs.items():
            t.copy_(new_outputs[k])
        slot_table[:rows].copy_(torch.tensor(new_slots, dtype=torch.int32))
        prev_slots.copy_(new_prev)
        pos_index.copy_(new_pos)
        start = on.snapshot()
        off.restore(start)
        graph.replay()
        off.sampler._update_stores(outputs, 0, new_slots)
        torch_gather(off.store, prev_slots, pos_index, rows, off.bufs)
        torch.cuda.synchronize()
        bad = compare(off, on, new_slots, outputs)
        assert not bad, (f"replay {replay}", bad)
    after = on.snapshot()
    on.restore(start)
    graph.replay()
    torch.cuda.synchronize()
    again = on.snapshot()
    assert not [k for k in after if not torch.equal(after[k], again[k])], "same inputs replayed"


needs_sm103 = pytest.mark.skipif(
    not _runnable() or _sm() != 103,
    reason="the switch gates SM 103 only (SM 100 always runs the kernels)",
)


class Reader:
    """A run-time switch reader (as the K3 model installs) and the launches it was told of."""

    def __init__(self):
        self.on = False
        self.dispatched = 0

    def install(self):
        _op().set_sm103_switch(lambda: self.on, self._dispatched)

    def _dispatched(self):
        self.dispatched += 1


@needs_sm103
@pytest.mark.parametrize("env", [None, "0", "1"], ids=["unset", "0", "1"])
def test_reader_builds_at_load_and_overrides_environment(monkeypatch, env):
    """With a reader installed: the sampler builds its scatter state and compiles the kernel at construction even
    with the switch off, and the switch is the reader's, whatever the environment says."""
    op = _op()
    _set_switch(monkeypatch, env)
    monkeypatch.setattr(op.SlotScatter, "_kernel", None)
    reader = Reader()
    reader.install()
    assert op.build_at_load() and op.precompile_at_load()
    sampler = _sampler()
    assert sampler._store_scatter is not None and sampler._slot_table_buffer is not None
    assert op.SlotScatter._kernel is not None, "not compiled at load"
    for on in (False, True, False):
        reader.on = on
        assert op.is_enabled() is on and op.is_supported() is on
    op.set_sm103_switch(None)
    assert op.is_enabled() == (env == "1")
    assert not op.precompile_at_load()


@needs_sm103
def test_switch_off_with_reader_touches_nothing(monkeypatch):
    """Built at load, switch off: ``_update_stores`` neither updates the slot table nor launches the kernel, and
    the stores are the torch ops', bit for bit."""
    reader = Reader()
    reader.install()
    sampler = _sampler()

    def forbidden(*args, **kwargs):
        raise AssertionError("step-copy state used with the switch off")

    monkeypatch.setattr(sampler._slot_table_buffer, "update", forbidden)
    monkeypatch.setattr(sampler._store_scatter, "scatter", forbidden)
    g = torch.Generator().manual_seed(5)
    off = Arm(sampler, g)
    ref = Arm(_sampler(), torch.Generator().manual_seed(5))
    for rows in BATCHES:
        outputs, slots, _, _ = step_inputs(rows, g)
        off.sampler._update_stores(outputs, 0, slots)
        ref.sampler._scatter_to_stores(outputs, 0, slots)
        torch.cuda.synchronize()
        assert not [k for k, t in off.snapshot().items() if not torch.equal(t, ref.snapshot()[k])]
    assert reader.dispatched == 0


@needs_sm103
@pytest.mark.parametrize("rows", BATCHES, ids=[f"bs{r}" for r in BATCHES])
def test_flip_between_steps_equals_off_arm(rows):
    """One sampler built once with the reader installed, the switch flipped between chained steps (off, on, on, off,
    on, off): its store update and the gathers chosen as ``_prepare_tp_inputs`` chooses them (``StepInputGather``
    where ``is_enabled()``, the torch ops otherwise) equal a switch-off arm's on every step (the switch-on steps up
    to the dead ``new_tokens`` columns, as above). The dispatch callback counts two launches per switch-on step."""
    op = _op()
    reader = Reader()
    reader.install()
    flip = Arm(_sampler(), torch.Generator().manual_seed(120 + rows))
    ref_sampler = _sampler()
    ref_sampler._store_scatter = None  # the torch ops, every step
    ref = Arm(ref_sampler, torch.Generator().manual_seed(120 + rows))
    assert not [k for k, t in ref.snapshot().items() if not torch.equal(t, flip.snapshot()[k])]
    g = torch.Generator().manual_seed(rows)
    expect = 0
    for step, on in enumerate((False, True, True, False, True, False)):
        reader.on = on
        outputs, slots, prev_slots, pos_index = step_inputs(rows, g)
        flip.sampler._update_stores(outputs, 0, slots)
        gather = kernel_gather if op.is_enabled() else torch_gather
        gather(flip.store, prev_slots, pos_index, rows, flip.bufs)
        ref.sampler._update_stores(outputs, 0, slots)
        torch_gather(ref.store, prev_slots, pos_index, rows, ref.bufs)
        torch.cuda.synchronize()
        expect += 2 * on
        assert reader.dispatched == expect, (step, reader.dispatched, expect)
        if on:
            bad = compare(ref, flip, slots, outputs)
        else:
            a, b = ref.snapshot(), flip.snapshot()
            bad = [k for k in a if not torch.equal(a[k], b[k])]
        assert not bad, (f"step {step} switch {int(on)}", bad)
        ref.store.new_tokens.copy_(flip.store.new_tokens)

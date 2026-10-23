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
"""``TRTLLM_K3LL_DRAFTER_PREP_FUSED`` and ``TRTLLM_K3LL_DRAFTER_CTX_SIDE`` (``kimi_k3_nvfp4/flags.py``), one GPU.

Both switches are read per step and set on the flags module per arm; neither changes a value, so every check is bit
for bit. The worker is the MLA DSpark worker (`KimiK3MLADSparkWorker`) on the stock semantics test's tiny MLA drafter
(its real context projection, latent GEMM, RoPE table and embedding), with the context latent path installed
(`_k3_install_ctx`) and the worker state a decode step reads: context lengths, slots (padding rows on the dummy
slot), the draft pool's block table and counts, the paged pools. K = 7 drafts, a block of 7 (the DSpark shift-label
width) or 8.

* One worker, one built state: each step runs the switch-off arm (the stock ``prepare_1st_drafter_inputs``), restores
  the state, flips the switches and runs the on arm; every input the drafter reads (noise block, query positions,
  advertised context, anchor tokens, page table, rows) and every state the next step reads (``_ctx_len`` of every
  slot, every layer's pool, ``_seq_lens_cuda`` and the host ``_seq_lens``) is compared. Four steps on one worker (the
  pools carry over): first step (empty contexts), mixed acceptance, full rejection (n = 1), full acceptance, across
  page boundaries, at the context
  ceiling, with an allocation one page short (the store's clamp; destinations the stock store writes more than once
  are excluded from the pool comparison, as its winner is not defined) and the advertised-context clamp. B = 1 / 2 / 4
  / 8, with a vocabulary-split embedding (a one-GPU stand-in for its all-reduce) and a trained mask row.
* The same chain with the side stream on (multi-stream enabled, as in CUDA-graph warmup), and from a captured CUDA
  graph (side stream on and off) replayed with rewritten inputs and state, each replay against the off arm's eager
  step on the same inputs.
* The block state the worker hands the drafter (`BlockPrep`) against ``K3MLADrafterNorms._fixup`` /
  ``_build_mla_block_fixup`` of the off arm's outputs, the pool slots and ``freqs_cis[query_positions]``; and the
  block decode (`K3MLADrafterNorms.forward`, cute-dsl backend, SM100 / SM103) with a `BlockPrep` against without.
* The masked store (``k3_drafter_mla_ctx_kv`` with ``mask``) against the stock bf16 multiply followed by the unmasked
  store, signed zeros included.
"""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_HERE = pathlib.Path(__file__).resolve().parent
K = 7
PAGE = 32
MAXB = 6  # pages per block-table row
SLOTS = 9  # 8 requests + the dummy slot
DUMMY = SLOTS - 1
MAX_CTX = 150
POOL_PAGES = 64
ROWS_MAX = 8 * (K + 1)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the drafter's cute-dsl decode needs SM100 / SM103",
)


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _semantics():
    return _load(
        "_k3_dspark_semantics_for_drafter_input",
        _HERE.parent / "speculative" / "hw_agnostic" / "test_kimi_k3_dspark_semantics.py",
    )


def _pkg(name):
    import importlib

    return importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4." + name
    )


def _set(monkeypatch, prep, side, pdl=None):
    flags = _pkg("flags")
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_PREP_FUSED", prep)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_CTX_SIDE", side)
    if pdl is not None:
        monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_PDL", pdl)


def _bits(t):
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.complex64:
        return torch.view_as_real(t).view(torch.int32)
    return t


def _same(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, (
        f"{what}: {a.dtype} {tuple(a.shape)} vs {b.dtype} {tuple(b.shape)}"
    )
    assert torch.equal(_bits(a), _bits(b)), f"{what}: not bit for bit"


def test_switches_are_registered_on_by_default():
    table = _pkg("flags").flags()
    for name in ("TRTLLM_K3LL_DRAFTER_PREP_FUSED", "TRTLLM_K3LL_DRAFTER_CTX_SIDE"):
        assert name in table
        assert table[name][1] == 1, f"{name} must default to 1"


# ----------------------------------------------------------------------------------------------------------------------
# The worker on the tiny MLA drafter
# ----------------------------------------------------------------------------------------------------------------------


class _FakeAllReduce(nn.Module):
    """A one-GPU stand-in for the vocabulary-split embedding's all-reduce: elementwise and deterministic, so its outputs
    are equal exactly when its inputs are."""

    def forward(self, x):
        return (x.float() * 0.75 + 0.125).to(x.dtype)


class _CommOnly:
    """The drafter's ``project_target_hidden`` declaring, as the target's split ``fc`` does, that its collective is
    the decode collectives' (here: none), so the side stream may run it."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, hidden_states):
        return self.fn(hidden_states)

    @staticmethod
    def k3_comm_only(rows):
        return True


class _SpecMetadata:
    def __init__(self, captured, hidden):
        self.captured = captured
        self.runtime_draft_len = K
        self.hidden_size = hidden

    def get_hidden_states(self, num_tokens):
        return self.captured[:num_tokens]


def _drafter(split: bool, trained: bool):
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    sem = _semantics()
    torch.manual_seed(0)
    drafter = sem._build_mla_drafter(sem._tiny_mla_weights())
    drafter.project_target_hidden = _CommOnly(drafter.project_target_hidden)
    embed = drafter.draft_model_full.model.embed_tokens
    if split:
        # This rank holds ids [start, end): the bonus tokens outside it look up +0 rows before the all-reduce.
        start, end = sem.VOCAB // 3, sem.VOCAB - 5
        embed.weight = nn.Parameter(embed.weight.data[start:end].clone(), requires_grad=False)
        embed.tp_size, embed.tp_rank, embed.tp_mode = 2, 0, TensorParallelMode.COLUMN
        embed.vocab_start_index, embed.vocab_end_index = start, end
        embed.all_reduce = _FakeAllReduce()
    g = torch.Generator(device="cuda").manual_seed(5)
    drafter.mask_token_embedding = (
        (torch.randn(sem.MLA_HIDDEN, generator=g, device="cuda") * 0.05).bfloat16()
        if trained
        else None
    )
    return sem, drafter


def _worker(drafter, block):
    spec_worker = _pkg("spec_worker")
    mla_drafter = _pkg("mla_drafter")
    prep = _pkg("mla_drafter_prep")
    w = object.__new__(spec_worker.KimiK3MLADSparkWorker)
    w.spec_config = SimpleNamespace(max_draft_len=K, mask_token_id=None)
    w._resolved_mask_token_id = None
    w._compute_block_size = block
    w._ctx_paged = True
    w._ctx_page_size = PAGE
    w._max_ctx = MAX_CTX
    w._dummy_slot = DUMMY
    w._ctx_k_buf = None
    w._ctx_v_buf = None
    w._ctx_page_table = None
    w._ctx_kv_buf = [
        torch.zeros(POOL_PAGES, 1, 1, PAGE, 576, dtype=torch.bfloat16, device="cuda")
        for _ in range(len(drafter.model.layers))
    ]
    w._ctx_block_tables = torch.zeros(SLOTS, MAXB, dtype=torch.int32, device="cuda")
    w._ctx_block_counts = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
    w._ctx_len = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
    w._batch_to_slot_buffer = SimpleNamespace(
        tensor=torch.zeros(8, dtype=torch.long, device="cuda")
    )
    w._k3_side = prep.CtxSide()
    w._k3_install_ctx(drafter)
    assert w._k3_ctx, "the context latent path did not install on the tiny drafter"
    # The block decode's owner, so the worker hands over its BlockPrep (the stock forward is not run here).
    owner = object.__new__(mla_drafter.K3MLADrafterNorms)
    owner._consts = {}
    owner.block_prep = None
    drafter.dflash_forward = mla_drafter._Bound(owner)
    return w, owner


class _Inputs:
    """One step's inputs and the state it starts from (static tensors, so a captured graph reads them)."""

    def __init__(self, sem, batch):
        dev = "cuda"
        self.batch = batch
        self.accepted = torch.zeros(batch, K + 1, dtype=torch.int32, device=dev)
        self.num_accepted = torch.ones(batch, dtype=torch.int32, device=dev)
        self.captured = torch.zeros(
            ROWS_MAX, sem.MLA_HIDDEN * sem.NUM_CAPTURE, dtype=torch.bfloat16, device=dev
        )
        self.seq_lens_cuda = torch.zeros(batch, dtype=torch.int32, device=dev)
        self.seq_lens = torch.zeros(batch, dtype=torch.int32)
        self.spec = _SpecMetadata(self.captured, sem.MLA_HIDDEN)
        self.attn = SimpleNamespace(
            num_contexts=0,
            num_seqs=batch,
            num_ctx_tokens=0,
            _seq_lens_cuda=self.seq_lens_cuda,
            _seq_lens=self.seq_lens,
        )


def _scenario(w, inp, sem, seed, kind, n_kind, pad):
    """Write a step's inputs and state: ``kind`` the context lengths and allocations, ``n_kind`` the acceptance."""
    g = torch.Generator().manual_seed(seed)
    batch = inp.batch
    slots = torch.randperm(DUMMY, generator=g)[:batch]
    if pad and batch > 1:
        slots[-1] = DUMMY
    ctx = torch.zeros(SLOTS, dtype=torch.int64)
    choices = {
        "first": [0],
        "boundary": [PAGE - 1, PAGE - 4, 2 * PAGE - 1, 2 * PAGE, PAGE - K - 1],
        "ceiling": [MAX_CTX - 3, MAX_CTX, MAX_CTX - 1],
        "random": list(range(0, MAX_CTX + 1)),
    }[kind]
    for s in range(DUMMY):
        ctx[s] = choices[int(torch.randint(len(choices), (1,), generator=g))]
    ctx[DUMMY] = 0
    counts = torch.zeros(SLOTS, dtype=torch.int64)
    table = torch.zeros(SLOTS, MAXB, dtype=torch.int32)
    pages = torch.randperm(POOL_PAGES, generator=g)
    for b in range(batch):
        need = -(-(int(ctx[slots[b]]) + K + 1) // PAGE)
        short = int(torch.randint(4, (1,), generator=g)) == 0  # one page short: the store clamps
        counts[b] = max(0, min(MAXB, need - (1 if short else 0)))
        table[b] = pages[b * MAXB : (b + 1) * MAXB].to(torch.int32)
    if n_kind == "reject":
        n = torch.ones(batch, dtype=torch.int32)
    elif n_kind == "accept":
        n = torch.full((batch,), K + 1, dtype=torch.int32)
    else:
        n = torch.randint(1, K + 2, (batch,), generator=g, dtype=torch.int32)
    w._ctx_len.copy_(ctx)
    w._ctx_block_counts.copy_(counts)
    w._ctx_block_tables.copy_(table)
    w._batch_to_slot_buffer.tensor[:batch].copy_(slots)
    inp.num_accepted.copy_(n)
    inp.accepted.copy_(torch.randint(0, sem.VOCAB, (batch, K + 1), generator=g, dtype=torch.int32))
    inp.captured.copy_((torch.randn(inp.captured.shape, generator=g) * 0.5).to(torch.bfloat16))
    inp.seq_lens_cuda.zero_()
    inp.seq_lens.zero_()


def _randomize_pools(w, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    for p in w._ctx_kv_buf:
        p.copy_((torch.randn(p.shape, generator=g, device="cuda") * 0.1).bfloat16())


def _state(w, inp):
    return dict(
        ctx_len=w._ctx_len.clone(),
        pools=[p.clone() for p in w._ctx_kv_buf],
        seq_lens_cuda=inp.seq_lens_cuda.clone(),
        seq_lens=inp.seq_lens.clone(),
        counts=w._ctx_block_counts.clone(),
        table=w._ctx_block_tables.clone(),
        slots=w._batch_to_slot_buffer.tensor.clone(),
    )


def _restore(w, inp, st):
    w._ctx_len.copy_(st["ctx_len"])
    for p, q in zip(w._ctx_kv_buf, st["pools"]):
        p.copy_(q)
    inp.seq_lens_cuda.copy_(st["seq_lens_cuda"])
    inp.seq_lens.copy_(st["seq_lens"])
    w._ctx_block_counts.copy_(st["counts"])
    w._ctx_block_tables.copy_(st["table"])
    w._batch_to_slot_buffer.tensor.copy_(st["slots"])


def _contested(w, inp):
    """(page, offset) pairs the stock store writes more than once this step (its clamp folds rows together)."""
    batch = inp.batch
    slots = w._batch_to_slot_buffer.tensor[:batch]
    j = torch.arange(K + 1, device="cuda")
    col = w._ctx_len[slots].unsqueeze(1) + j
    cap = w._ctx_block_counts[:batch] * PAGE
    col = torch.minimum(col, (cap - 1).unsqueeze(1)).clamp_(min=0)
    pages = torch.gather(w._ctx_block_tables[:batch].long(), 1, col // PAGE)
    keys = (pages * PAGE + col % PAGE).reshape(-1)
    uniq, counts = torch.unique(keys, return_counts=True)
    return uniq[counts > 1]


def _run(w, drafter, inp):
    """What a step does from the target LM head to the drafter call: the decision, then the drafter's inputs."""
    w._k3_input = w._k3_input_begin(drafter, inp.spec, inp.attn)
    return w.prepare_1st_drafter_inputs(
        input_ids=None,
        position_ids=None,
        hidden_states=inp.captured,
        accepted_tokens=inp.accepted,
        num_accepted_tokens=inp.num_accepted,
        attn_metadata=inp.attn,
        spec_metadata=inp.spec,
        draft_model=drafter,
        total_target_tokens=inp.batch * (K + 1),
    )


_OUT_KEYS = ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens")


def _snapshot(out, w, inp):
    return dict(
        out={k: out[k].clone() for k in _OUT_KEYS},
        rows=out["ctx_cache_batch_idx"].clone(),
        page_table=out["ctx_page_table"],
        state=_state(w, inp),
    )


def _compare(got, want, contested, what, host=True):
    for k in _OUT_KEYS:
        _same(got["out"][k], want["out"][k], f"{what}: {k}")
    _same(got["rows"], want["rows"], f"{what}: ctx_cache_batch_idx")
    assert got["page_table"] is want["page_table"], f"{what}: ctx_page_table"
    _same(got["state"]["ctx_len"], want["state"]["ctx_len"], f"{what}: _ctx_len")
    _same(got["state"]["seq_lens_cuda"], want["state"]["seq_lens_cuda"], f"{what}: _seq_lens_cuda")
    if host:
        _same(got["state"]["seq_lens"], want["state"]["seq_lens"], f"{what}: _seq_lens")
    for layer, (a, b) in enumerate(zip(got["state"]["pools"], want["state"]["pools"])):
        a, b = a.clone(), b.clone()
        if contested.numel():
            flat_a = a.view(POOL_PAGES * PAGE, -1)
            flat_b = b.view(POOL_PAGES * PAGE, -1)
            flat_a[contested] = 0
            flat_b[contested] = 0
        _same(a, b, f"{what}: layer {layer} pool")


def _check_block_prep(owner, want, w, drafter, inp, block, what):
    """The worker's `BlockPrep` against the drafter's own fixup of the off arm's outputs."""
    from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup

    prep = owner.block_prep
    assert prep is not None, f"{what}: no BlockPrep handed to the drafter"
    owner.block_prep = None
    num_ctx, qpos = want["out"]["num_ctx_per_req"], want["out"]["query_positions"]
    table = w._ctx_block_tables
    rows = torch.arange(inp.batch, device="cuda")
    glue = owner._fixup(num_ctx, table, rows, block, PAGE)
    stock = _build_mla_block_fixup(num_ctx, table.index_select(0, rows).long(), block, PAGE)
    for ref, name in ((glue, "glue fixup"), (stock, "stock fixup")):
        _same(prep.pages, ref.pages, f"{what}: {name} pages")
        _same(prep.offsets, ref.offsets, f"{what}: {name} offsets")
        _same(prep.seq_lens_i32, ref.seq_lens_i32, f"{what}: {name} seq_lens")
        _same(prep.kv_bounds_i32, ref.kv_bounds_i32, f"{what}: {name} kv_bounds")
        _same(table[: inp.batch], ref.page_tables_i32.to(torch.int32), f"{what}: {name} tables")
    pool = w._ctx_kv_buf[0]
    assert tuple(prep.pool_strides) == (pool.stride(0), pool.stride(3))
    _same(
        prep.slots,
        (glue.pages * pool.stride(0) + glue.offsets * pool.stride(3)).reshape(-1),
        f"{what}: pool slots",
    )
    _same(prep.blk_freqs, drafter._mla_freqs_cis(qpos.device)[qpos.long()], f"{what}: freqs rows")
    assert prep.rows_start == 0 and prep.page_table is table


_STEPS = (("first", "mixed"), ("boundary", "reject"), ("ceiling", "accept"), ("random", "mixed"))


@needs_cuda
@pytest.mark.parametrize("pdl", [1, 0])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("block", [7, 8])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_worker_steps_flip_bit_identical(batch, block, split, trained, side, pdl, monkeypatch):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    census = _pkg("census")
    monkeypatch.setattr(census, "ENABLED", True)
    sem, drafter = _drafter(split, trained)
    w, owner = _worker(drafter, block)
    inp = _Inputs(sem, batch)
    _randomize_pools(w, 1)
    for step, (kind, n_kind) in enumerate(_STEPS):
        # Each step sets its own context lengths, allocations and acceptance; the pools carry over from the step
        # before.
        _scenario(w, inp, sem, 100 * batch + step, kind, n_kind, pad=step % 2 == 1)
        what = f"B {batch} block {block} split {split} trained {trained} step {step} ({kind}, {n_kind})"
        start = _state(w, inp)
        contested = _contested(w, inp)
        with with_multi_stream(bool(side)):
            _set(monkeypatch, 0, 0, pdl)
            off = _snapshot(_run(w, drafter, inp), w, inp)
            assert owner.block_prep is None, f"{what}: the off arm handed over a BlockPrep"
            _restore(w, inp, start)
            before = census.counts().get("dispatch:drafter_prep_fused:eager", 0)
            sides = census.counts().get("dispatch:drafter_ctx_side:eager", 0)
            _set(monkeypatch, 1, side, pdl)
            on = _snapshot(_run(w, drafter, inp), w, inp)
        torch.cuda.synchronize()
        assert census.counts().get("dispatch:drafter_prep_fused:eager", 0) == before + 1, what
        assert census.counts().get("dispatch:drafter_ctx_side:eager", 0) == sides + side, what
        _compare(on, off, contested, what)
        _check_block_prep(owner, off, w, drafter, inp, block, what)
        # The next step starts from this step's (equal) state.
        _restore(w, inp, on["state"])


@needs_cuda
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_worker_graph_replay_matches_the_off_arm(batch, side, monkeypatch):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    census = _pkg("census")
    monkeypatch.setattr(census, "ENABLED", True)
    block = 7
    sem, drafter = _drafter(split=True, trained=False)
    w, owner = _worker(drafter, block)
    inp = _Inputs(sem, batch)
    _randomize_pools(w, 2)
    _scenario(w, inp, sem, 7, "boundary", "mixed", pad=True)
    start = _state(w, inp)
    with with_multi_stream(True):
        _set(monkeypatch, 1, side)
        _run(
            w, drafter, inp
        )  # eager first: compiles the kernels and builds the store's tables outside the capture
        owner.block_prep = None
        torch.cuda.synchronize()
        _restore(w, inp, start)
        captures = census.counts().get("dispatch:drafter_prep_fused:capture", 0)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                out = _run(w, drafter, inp)
        torch.cuda.current_stream().wait_stream(stream)
    assert census.counts().get("dispatch:drafter_prep_fused:capture", 0) == captures + 1
    if side:
        assert census.counts().get("dispatch:drafter_ctx_side:capture", 0) >= 1
    prep = owner.block_prep
    owner.block_prep = None
    for r, (kind, n_kind) in enumerate(_STEPS * 2):
        _scenario(w, inp, sem, 500 + r, kind, n_kind, pad=r % 2 == 0)
        _randomize_pools(w, 600 + r)
        start = _state(w, inp)
        contested = _contested(w, inp)
        graph.replay()
        torch.cuda.synchronize()
        got = _snapshot(out, w, inp)
        got_prep = [
            t.clone() for t in (prep.pages, prep.offsets, prep.seq_lens_i32, prep.kv_bounds_i32)
        ]
        _restore(w, inp, start)
        _set(monkeypatch, 0, 0)
        want = _snapshot(_run(w, drafter, inp), w, inp)
        _set(monkeypatch, 1, side)
        what = f"graph B {batch} side {side} replay {r} ({kind}, {n_kind})"
        # A replay runs no Python: the host _seq_lens is the eager arm's alone.
        _compare(got, want, contested, what, host=False)
        fix = owner._fixup(
            want["out"]["num_ctx_per_req"], w._ctx_block_tables, torch.arange(batch, device="cuda"),
            block, PAGE,
        )  # fmt: skip
        for t, ref, name in zip(
            got_prep, (fix.pages, fix.offsets, fix.seq_lens_i32, fix.kv_bounds_i32),
            ("pages", "offsets", "seq_lens", "kv_bounds"),
        ):  # fmt: skip
            _same(t, ref, f"{what}: BlockPrep {name}")
        _restore(w, inp, got["state"])


@needs_cuda
def test_off_arm_is_the_stock_path(monkeypatch):
    """With PREP_FUSED off no step decision is made, the side stream is not used and no BlockPrep is handed over,
    whatever CTX_SIDE says."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    sem, drafter = _drafter(split=False, trained=False)
    w, owner = _worker(drafter, 7)
    inp = _Inputs(sem, 2)
    _scenario(w, inp, sem, 3, "random", "mixed", pad=False)
    with with_multi_stream(True):
        _set(monkeypatch, 0, 1)
        assert w._k3_input_begin(drafter, inp.spec, inp.attn) is None
        _run(w, drafter, inp)
    assert owner.block_prep is None


# ----------------------------------------------------------------------------------------------------------------------
# The masked store against the stock multiply + store
# ----------------------------------------------------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [8, 32, 64])
def test_masked_store_matches_the_stock_multiply(rows, pdl):
    _pkg("mla_drafter_ctx")
    dev = "cuda"
    layers, page, max_pos = 5, 64, 4096
    g = torch.Generator(device=dev).manual_seed(rows)
    latent = (torch.randn(rows, layers * 576, generator=g, device=dev)).bfloat16()
    latent[0, :7] = -0.0  # signed zeros survive the multiply
    norm = (
        torch.rand(layers, 512, generator=g, device=dev) - 0.3
    ).bfloat16()  # negative weights flip zero signs
    angles = torch.rand(max_pos, 32, generator=g, device=dev) * 6.28
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).contiguous()
    positions = torch.randperm(max_pos, generator=g, device=dev)[:rows]
    pages = torch.arange(rows, device=dev) // 2
    offsets = (torch.arange(rows, device=dev) % 2) * 32 + torch.arange(rows, device=dev) % 32
    mask = (torch.rand(rows, generator=g, device=dev) < 0.5).float()

    def pools():
        return [torch.full((rows, 1, 1, page, 576), 3.0, dtype=torch.bfloat16, device=dev)
                for _ in range(layers)]  # fmt: skip

    stock, fused = pools(), pools()
    stock_ptrs = torch.tensor([p.data_ptr() for p in stock], dtype=torch.int64, device=dev)
    fused_ptrs = torch.tensor([p.data_ptr() for p in fused], dtype=torch.int64, device=dev)
    masked = latent.view(rows, layers, 1, 576).clone()
    masked.mul_(mask.view(-1, 1, 1, 1).to(torch.bfloat16))
    torch.ops.trtllm.k3_drafter_mla_ctx_kv(
        masked.reshape(rows, -1), norm, freqs, positions, pages, offsets, stock_ptrs, stock, 1e-6,
        pdl,
    )  # fmt: skip
    torch.ops.trtllm.k3_drafter_mla_ctx_kv(
        latent, norm, freqs, positions, pages, offsets, fused_ptrs, fused, 1e-6, pdl, mask
    )
    for layer in range(layers):
        _same(fused[layer], stock[layer], f"rows {rows} pdl {pdl} layer {layer}")


# ----------------------------------------------------------------------------------------------------------------------
# The block decode with a BlockPrep
# ----------------------------------------------------------------------------------------------------------------------


@needs_sm100
@pytest.mark.parametrize("q_lora", [16, 1536])
def test_block_decode_takes_the_block_prep(q_lora, monkeypatch):
    """`K3MLADrafterNorms.forward` with a `BlockPrep` of its inputs (the values the worker's kernel produces, checked
    against the same references above) against the same call without one."""
    pdl_glue = _load(
        "_k3_drafter_pdl_glue_for_drafter_input",
        _HERE / "test_modeling_v2_kimi_k3_drafter_pdl_glue.py",
    )
    mla_drafter = _pkg("mla_drafter")
    census = _pkg("census")
    from tensorrt_llm._torch.models import modeling_dspark as md

    sem = _semantics()
    reason = md.cute_dsl_mla_decode_unavailability_reason()
    if reason is not None:
        pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")
    monkeypatch.setattr(sem, "MLA_Q_LORA", q_lora)
    monkeypatch.setattr(census, "ENABLED", True)
    torch.manual_seed(0)
    drafter = sem._build_mla_drafter(sem._tiny_mla_weights(), dflash_attention_backend="CUTEDSL")
    a_weights = None
    if q_lora == 1536:
        drafter._build_fused_kv_buffers()
        a_weights = mla_drafter.K3MLADrafterNorms._fuse_a_weights(drafter.model.layers)
    stock = drafter.dflash_forward
    g = torch.Generator().manual_seed(42)
    requests = [
        (
            torch.randn(length, sem.MLA_HIDDEN * sem.NUM_CAPTURE, generator=g) * 0.5,
            torch.randn(sem.MLA_BLOCK, sem.MLA_HIDDEN, generator=g) * 0.5,
        )
        for length in (sem.MLA_CTX_LEN, sem.MLA_SHORT_CTX_LEN)
    ]
    results = []
    for with_prep in (False, True):
        owner = mla_drafter.K3MLADrafterNorms(
            drafter, pdl_glue._OneRankComm(), None, stock, frozenset(), a_weights
        )
        kwargs = pdl_glue._forward_inputs(sem, drafter, requests)
        if with_prep:
            qpos, num_ctx = kwargs["query_positions"], kwargs["num_ctx_per_req"]
            table, rows = kwargs["ctx_page_table"], kwargs["ctx_cache_batch_idx"]
            pool = kwargs["ctx_kv_cache"][0]
            block = sem.MLA_BLOCK
            fix = owner._fixup(num_ctx, table, rows, block, pool.shape[-2])
            owner.block_prep = mla_drafter.BlockPrep(
                query_positions=qpos,
                num_ctx=num_ctx,
                page_table=table,
                rows_start=0,
                pages=fix.pages.clone(),
                offsets=fix.offsets.clone(),
                seq_lens_i32=fix.seq_lens_i32.clone(),
                kv_bounds_i32=fix.kv_bounds_i32.clone(),
                slots=(fix.pages * pool.stride(0) + fix.offsets * pool.stride(3)).reshape(-1),
                pool_strides=(pool.stride(0), pool.stride(3)),
                blk_freqs=drafter._mla_freqs_cis(qpos.device)[qpos.long()].clone(),
            )
        before = census.counts().get("dispatch:drafter_block_prep:eager", 0)
        out = owner.forward(**kwargs)
        taken = census.counts().get("dispatch:drafter_block_prep:eager", 0) - before
        assert taken == int(with_prep), f"q_lora {q_lora}: BlockPrep taken {taken} times"
        assert owner.block_prep is None, "the BlockPrep outlived its call"
        results.append((out, kwargs["ctx_kv_cache"]))
    (out0, pools0), (out1, pools1) = results
    _same(out1, out0, f"q_lora {q_lora}: block output")
    for layer, (p1, p0) in enumerate(zip(pools1, pools0)):
        _same(p1, p0, f"q_lora {q_lora}: layer {layer} pool")


@needs_cuda
def test_block_prep_of_another_call_is_dropped():
    """A `BlockPrep` whose tensors are not the call's is ignored (and cleared)."""
    mla_drafter = _pkg("mla_drafter")
    owner = object.__new__(mla_drafter.K3MLADrafterNorms)
    owner._consts = {}
    t = torch.zeros(1, dtype=torch.int64, device="cuda")
    owner.block_prep = mla_drafter.BlockPrep(t, t, t, 0, t, t, t, t, t, (0, 0), t)
    owner.takes = lambda *args: False
    owner.stock = lambda *args: "stock"
    noise = torch.zeros(1, 7, 8, dtype=torch.bfloat16, device="cuda")
    assert owner.forward(noise, t, t, None, None, t, None, None) == "stock"
    assert owner.block_prep is None

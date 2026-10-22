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
* switch on, a step the kernel does not cover (context tokens in the batch): the torch path runs.

Batch 1, 2 and 4 at draft length 7 (8 tokens per row); Helix16 at CP ranks 0, 5 and 15 with 64-token pages and
positions straddling page and ledger boundaries, the overlap scheduler on and off, and the TP16 shape (no Helix).
"""

import types

import pytest
import torch

DRAFT_LEN = 7
TOKENS = DRAFT_LEN + 1
MAX_BATCH = 8
MAX_NUM_TOKENS = 512
PAGE = 64
CP_SIZE = 16
BATCHES = [1, 2, 4]
ENV = "TRTLLM_K3LL_SPEC_HEAD_FUSED"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

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
        runner = types.SimpleNamespace(
            _config=types.SimpleNamespace(disable_overlap_scheduler=not overlap),
            mapping=mapping,
            previous_pos_id_offsets_cuda=b["pos_off"],
            previous_kv_lens_offsets_cuda=b["kv_off"],
            get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
            guided_decoder=None,
            _spec_head_fused=fused,
        )
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
        self.preprocess(self.inputs, enable_spec_decode=True, runtime_draft_len=DRAFT_LEN)


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


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_switch_read(monkeypatch, value):
    """The runner's switch: on only for ``1``; unset and ``0`` keep the torch path."""
    from tensorrt_llm._torch.speculative import spec_head_kernels

    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert spec_head_kernels.spec_head_fusion_enabled() == (value == "1")


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

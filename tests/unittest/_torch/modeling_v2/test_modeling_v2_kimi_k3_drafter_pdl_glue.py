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
"""``TRTLLM_K3LL_DRAFTER_PDL`` and ``TRTLLM_K3LL_DRAFTER_GLUE`` (``kimi_k3_nvfp4/flags.py``), one GPU (the CPU tests
need none). Both switches are read per call and set on the flags module per arm. Neither changes a value, so every
check is bit for bit:

* One MLA DSpark drafter layer's attention (`K3MLADrafterNorms._attention_qkv`) at the K3 drafter's per-rank shapes
  (``test_modeling_v2_kimi_k3_drafter_gemm.py``'s cases: 1 / 2 / 4 requests of a 7- or 8-token block, 4 or 6
  heads), with ``TRTLLM_K3LL_DRAFTER_GEMM`` on (the default) and off: every (PDL, GLUE) arm against the (0, 0) arm,
  the output and the pool; and from a captured CUDA graph replayed with rewritten inputs, each replay against the
  (0, 0) arm's eager call on the same inputs.
* The whole block decode (`K3MLADrafterNorms.forward`) of the stock semantics test's tiny MLA drafter on the cute-dsl
  backend, with a single-rank stand-in for the collectives and ``TRTLLM_K3LL_DRAFTER_GEMM`` on and off: with the stock
  q_a rank (`_attention`, the paged attention's cute-dsl decode) and with the K3 one (`_attention_qkv`), every arm
  against the (0, 0) arm.
* GLUE's block state (`K3MLADrafterNorms._fixup`) against ``_build_mla_block_fixup``, field by field (values, dtype,
  ``kv_bounds`` contiguous), for an int32 and an int64 table, eagerly and from a replayed graph.
* PDL's kernels alone: ``k3_drafter_mla_ctx_kv`` and ``cute_dsl_dspark_rope_into`` with and without PDL, eagerly and
  from a graph in which a kernel right before rewrites their inputs (a grid that read before its wait would see the
  previous replay's).
* GLUE in the MLA worker (CPU): the V2 manager's KDA replay record (``update_mamba_states``) applied after
  ``k3_spec_accept``'s leaves the record unchanged, and the worker skips the call only on a fused step with the
  switch on.
"""

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ARMS = ((0, 0), (1, 0), (0, 1), (1, 1))

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the drafter's kernels (cute-dsl decode, low-M GEMM) need SM100 / SM103",
)


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _gemm_cases():
    """``test_modeling_v2_kimi_k3_drafter_gemm.py``: the per-rank attention cases and their owner."""
    return _load(
        "_k3_drafter_gemm_cases_for_pdl_glue", _HERE / "test_modeling_v2_kimi_k3_drafter_gemm.py"
    )


def _semantics():
    return _load(
        "_k3_dspark_semantics_for_pdl_glue",
        _HERE.parent / "speculative" / "hw_agnostic" / "test_kimi_k3_dspark_semantics.py",
    )


def _flags():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    return flags


def _set(monkeypatch, pdl, glue):
    flags = _flags()
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_PDL", pdl)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_GLUE", glue)


def _bits(t):
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


def _same(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, (
        f"{what}: {a.dtype} {a.shape} vs {b.dtype} {b.shape}"
    )
    assert torch.equal(_bits(a), _bits(b)), f"{what}: not bit for bit"


def test_switches_are_registered_off_by_default():
    flags = _flags()
    table = flags.flags()
    for name in ("TRTLLM_K3LL_DRAFTER_PDL", "TRTLLM_K3LL_DRAFTER_GLUE"):
        assert name in table
        assert table[name][1] == 0, f"{name} must default to 0"


# ----------------------------------------------------------------------------------------------------------------------
# One layer's attention at the K3 drafter's shapes (_attention_qkv): a_post, rope, the bf16 decode, the V up-projection
# ----------------------------------------------------------------------------------------------------------------------


@needs_sm100
@pytest.mark.parametrize("gemm", [1, 0])
@pytest.mark.parametrize("heads", [4, 6])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("block", [7, 8])
def test_attention_arms_bit_identical(block, batch, heads, gemm, monkeypatch):
    m = _gemm_cases()
    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_DRAFTER_GEMM", gemm)
    results = {}
    for pdl, glue in _ARMS:
        _set(monkeypatch, pdl, glue)
        c = m._case(batch, heads, block=block)
        owner = m._owner(heads, gemm=bool(gemm))
        results[pdl, glue] = (m._run(owner, c), c.pool)
    out0, pool0 = results[0, 0]
    for arm, (out, pool) in results.items():
        _same(out, out0, f"arm {arm} gemm {gemm}: output")
        _same(pool, pool0, f"arm {arm} gemm {gemm}: pool")


@needs_sm100
@pytest.mark.parametrize("gemm", [1, 0])
@pytest.mark.parametrize("arm", _ARMS[1:])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_attention_graph_replay_matches_the_off_arm(batch, arm, gemm, monkeypatch):
    m = _gemm_cases()
    heads = 6
    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_DRAFTER_GEMM", gemm)
    _set(monkeypatch, *arm)
    c = m._case(batch, heads)
    owner = m._owner(heads, gemm=bool(gemm))
    m._run(owner, c)  # eager first: compiles and tunes outside the capture
    torch.cuda.synchronize()
    src = c.normed.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            c.normed.copy_(src)  # a kernel right before the chain writes its input
            out = m._run(owner, c)
    torch.cuda.current_stream().wait_stream(stream)
    off = m._owner(heads, gemm=bool(gemm))
    for r in range(2):
        m._rewrite(c, 300 + r)
        src.copy_(c.normed)
        pool0 = c.pool.clone()
        graph.replay()
        torch.cuda.synchronize()
        got, pool_after = out.clone(), c.pool.clone()
        c.pool.copy_(pool0)
        _set(monkeypatch, 0, 0)
        want = m._run(off, c)
        _set(monkeypatch, *arm)
        _same(got, want, f"arm {arm} gemm {gemm} replay {r}: output")
        _same(pool_after, c.pool, f"arm {arm} gemm {gemm} replay {r}: pool")


# ----------------------------------------------------------------------------------------------------------------------
# The whole block decode of a tiny MLA drafter (K3MLADrafterNorms.forward)
# ----------------------------------------------------------------------------------------------------------------------


class _OneRankComm:
    """``K3DecodeComm``'s fused all-reduce on a group of one: ``updated = residual + partial``, ``normed =
    norm(updated)``. The sandwich is never offered (no compiled forms)."""

    def takes_allreduce_norm(self, rows, hidden, one_shot_max_bytes=None):
        return rows > 0

    def takes_plain(self, *args, **kwargs):
        raise AssertionError("no k3_sandwich_plain form compiled, so none is asked for")

    def allreduce_norm(self, partial, residual, norm, one_shot_max_bytes=None):
        updated = residual + partial
        return norm(updated), updated


def _forward_inputs(sem, drafter, requests):
    """``dflash_forward``'s arguments on a paged pool (``_run_mla_block_decode``'s paged layout: each request owns a
    disjoint page range, an int32 page table)."""
    dev = "cuda"
    batch = len(requests)
    ctx_lens = [c.shape[0] for c, _ in requests]
    per_req = -(-(max(ctx_lens) + sem.MLA_BLOCK) // sem.MLA_PAGE)
    pool = [
        torch.zeros(
            batch * per_req, 1, 1, sem.MLA_PAGE, sem.MLA_LATENT, dtype=torch.bfloat16, device=dev
        )
        for _ in range(sem.MLA_LAYERS)
    ]
    for b, (captured, _) in enumerate(requests):
        proj = drafter.project_target_hidden(captured.to(dev, torch.bfloat16))
        latent, _ = drafter.precompute_context_kv(proj, torch.arange(captured.shape[0], device=dev))
        pos = torch.arange(ctx_lens[b], device=dev)
        for layer_idx in range(sem.MLA_LAYERS):
            pool[layer_idx][b * per_req + pos // sem.MLA_PAGE, 0, 0, pos % sem.MLA_PAGE] = latent[
                :, layer_idx, 0
            ]
    return dict(
        noise_embedding=torch.stack([n.to(dev, torch.bfloat16) for _, n in requests]),
        query_positions=torch.stack(
            [torch.arange(n, n + sem.MLA_BLOCK, device=dev) for n in ctx_lens]
        ),
        num_ctx_per_req=torch.tensor(ctx_lens, device=dev),
        ctx_k_cache=pool[0],
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(batch, device=dev),
        ctx_kv_cache=pool,
        ctx_page_table=torch.arange(batch * per_req, device=dev, dtype=torch.int32).view(
            batch, per_req
        ),
    )


@needs_sm100
@pytest.mark.parametrize("gemm", [1, 0])
@pytest.mark.parametrize("q_lora", [16, 1536])
def test_block_decode_arms_bit_identical(q_lora, gemm, monkeypatch):
    """q_lora 16: the stock projections and the paged attention's cute-dsl decode (`_attention`); 1536: the K3 one,
    ``[q_a | kv_a]`` and ``k3_drafter_mla_a_post`` (`_attention_qkv`). ``gemm``: ``TRTLLM_K3LL_DRAFTER_GEMM`` (the
    default 1 builds the pinned low-M GEMMs, which take the roles their shapes allow)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_drafter,
        mla_drafter_gemm,
    )
    from tensorrt_llm._torch.models import modeling_dspark as md

    sem = _semantics()
    reason = md.cute_dsl_mla_decode_unavailability_reason()
    if reason is not None:
        pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")
    monkeypatch.setattr(sem, "MLA_Q_LORA", q_lora)
    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_DRAFTER_GEMM", gemm)
    torch.manual_seed(0)
    drafter = sem._build_mla_drafter(sem._tiny_mla_weights(), dflash_attention_backend="CUTEDSL")
    a_weights = None
    if q_lora == 1536:
        drafter._build_fused_kv_buffers()
        a_weights = mla_drafter.K3MLADrafterNorms._fuse_a_weights(drafter.model.layers)
        assert a_weights is not None
    stock = drafter.dflash_forward
    g = torch.Generator().manual_seed(42)
    requests = [
        (
            torch.randn(length, sem.MLA_HIDDEN * sem.NUM_CAPTURE, generator=g) * 0.5,
            torch.randn(sem.MLA_BLOCK, sem.MLA_HIDDEN, generator=g) * 0.5,
        )
        for length in (sem.MLA_CTX_LEN, sem.MLA_SHORT_CTX_LEN)
    ]
    results = {}
    for pdl, glue in _ARMS:
        _set(monkeypatch, pdl, glue)
        owner = mla_drafter.K3MLADrafterNorms(
            drafter,
            _OneRankComm(),
            None,
            stock,
            frozenset(),
            a_weights,
            gemm=mla_drafter_gemm.DrafterGemm() if gemm else None,
        )
        kwargs = _forward_inputs(sem, drafter, requests)
        out = owner.forward(**kwargs)
        results[pdl, glue] = (out, kwargs["ctx_kv_cache"])
    out0, pools0 = results[0, 0]
    for arm, (out, pools) in results.items():
        _same(out, out0, f"q_lora {q_lora} gemm {gemm} arm {arm}: block output")
        for layer, (p, p0) in enumerate(zip(pools, pools0)):
            _same(p, p0, f"q_lora {q_lora} gemm {gemm} arm {arm}: layer {layer} pool")


# ----------------------------------------------------------------------------------------------------------------------
# GLUE's block state (_fixup) against _build_mla_block_fixup
# ----------------------------------------------------------------------------------------------------------------------


def _fixup_case(batch, block, table_dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    page = 64
    table = torch.randint(0, 500, (12, 9), generator=g, dtype=torch.int64).to(table_dtype).cuda()
    rows = torch.randperm(12, generator=g)[:batch].cuda()
    ctx_len = torch.randint(0, 9 * page - block, (batch,), generator=g).cuda()
    return ctx_len, table, rows, block, page


def _stock_fixup(ctx_len, table, rows, block, page):
    from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup

    return _build_mla_block_fixup(ctx_len, table.index_select(0, rows.long()).long(), block, page)


def _same_fixup(got, want, what):
    for field in want._fields:
        _same(getattr(got, field), getattr(want, field), f"{what}: {field}")
    assert got.kv_bounds_i32.is_contiguous(), f"{what}: kv_bounds not contiguous"


@needs_cuda
@pytest.mark.parametrize("table_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("block", [7, 8])
def test_fixup_matches_the_stock_fixup(block, batch, table_dtype):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_drafter,
    )

    owner = object.__new__(mla_drafter.K3MLADrafterNorms)
    owner._consts = {}
    case = _fixup_case(batch, block, table_dtype)
    got = owner._fixup(*case)
    _same_fixup(got, _stock_fixup(*case), f"block {block} batch {batch} {table_dtype}")
    again = owner._fixup(*case)
    assert again.valid.data_ptr() == got.valid.data_ptr(), "the mask is not cached"

    # From a graph replayed with new lengths and rows.
    ctx_len, table, rows, _, page = case
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out = owner._fixup(ctx_len, table, rows, block, page)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (5, 6):
        new = _fixup_case(batch, block, table_dtype, seed)
        ctx_len.copy_(new[0])
        rows.copy_(new[2])
        graph.replay()
        torch.cuda.synchronize()
        _same_fixup(out, _stock_fixup(*case), f"block {block} batch {batch} replay {seed}")


# ----------------------------------------------------------------------------------------------------------------------
# PDL's kernels alone
# ----------------------------------------------------------------------------------------------------------------------


def _graph(fn):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


@needs_cuda
@pytest.mark.parametrize("rows", [7, 24, 56])
def test_ctx_kv_pdl_bit_identical(rows):
    dev = "cuda"
    layers, page, max_pos = 5, 64, 4096

    def case(seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        latent = (torch.randn(rows, layers * 576, generator=g, device=dev)).bfloat16()
        positions = torch.randperm(max_pos, generator=g, device=dev)[:rows]
        return latent, positions

    g = torch.Generator(device=dev).manual_seed(1)
    norm = (torch.rand(layers, 512, generator=g, device=dev) + 1.0).bfloat16()
    angles = torch.rand(max_pos, 32, generator=g, device=dev) * 6.28
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).contiguous()
    pages = torch.arange(rows, device=dev) // 2
    offsets = (torch.arange(rows, device=dev) % 2) * 32 + torch.arange(rows, device=dev) % 32

    class Arm:
        def __init__(self, pdl):
            self.pdl = pdl
            self.src, self.positions = case(2)
            self.latent = torch.empty_like(self.src)
            self.pools = [torch.zeros(rows, 1, 1, page, 576, dtype=torch.bfloat16, device=dev)
                          for _ in range(layers)]  # fmt: skip
            self.ptrs = torch.tensor(
                [p.data_ptr() for p in self.pools], dtype=torch.int64, device=dev
            )

        def __call__(self):
            self.latent.copy_(self.src)  # the kernel right before writes the latents
            torch.ops.trtllm.k3_drafter_mla_ctx_kv(
                self.latent, norm, freqs, self.positions, pages, offsets, self.ptrs, self.pools,
                1e-6, self.pdl,
            )  # fmt: skip

    arms = {pdl: Arm(pdl) for pdl in (False, True)}
    for arm in arms.values():
        arm()
    for layer in range(layers):
        _same(arms[True].pools[layer], arms[False].pools[layer], f"eager layer {layer}")
    graphs = {pdl: _graph(arm) for pdl, arm in arms.items()}
    for seed in (3, 4):
        for pdl, arm in arms.items():
            arm.src.copy_(case(seed)[0])
            for p in arm.pools:
                p.zero_()
            graphs[pdl].replay()
        torch.cuda.synchronize()
        for layer in range(layers):
            _same(arms[True].pools[layer], arms[False].pools[layer], f"replay {seed} layer {layer}")


@needs_sm100
@pytest.mark.parametrize("rows", [7, 16, 28])
@pytest.mark.parametrize("heads", [4, 6])
def test_rope_into_pdl_bit_identical(rows, heads):
    from tensorrt_llm._torch.models.modeling_dspark import _mla_rope_into

    dev = "cuda"
    nope, rope, latent = 128, 64, 512

    class Arm:
        def __init__(self, pdl):
            g = torch.Generator(device=dev).manual_seed(0)
            self.pdl = pdl
            self.src = torch.randn(rows, 1, heads, nope + rope, generator=g, device=dev).bfloat16()
            self.q = torch.empty_like(self.src)
            angles = torch.rand(rows, 1, rope // 2, generator=g, device=dev) * 6.28
            self.freqs = torch.polar(torch.ones_like(angles), angles)
            self.out = torch.zeros(rows, 1, heads, latent + rope, dtype=torch.bfloat16, device=dev)

        def __call__(self):
            self.q.copy_(self.src)  # the kernel right before writes the query rows
            assert _mla_rope_into(
                self.q, self.freqs, self.out, heads, rope, latent, use_pdl=self.pdl
            )

    arms = {pdl: Arm(pdl) for pdl in (False, True)}
    for arm in arms.values():
        arm()
    _same(arms[True].out, arms[False].out, "eager")
    graphs = {pdl: _graph(arm) for pdl, arm in arms.items()}
    for seed in (3, 4):
        g = torch.Generator(device=dev).manual_seed(seed)
        new = torch.randn(arms[False].src.shape, generator=g, device=dev).bfloat16()
        for pdl, arm in arms.items():
            arm.src.copy_(new)
            arm.out.zero_()
            graphs[pdl].replay()
        torch.cuda.synchronize()
        _same(arms[True].out, arms[False].out, f"replay {seed}")


# ----------------------------------------------------------------------------------------------------------------------
# GLUE in the MLA worker: no second KDA replay record after k3_spec_accept (CPU)
# ----------------------------------------------------------------------------------------------------------------------


def _kernel_record(prev, state_indices, num_accepted, dummy):
    """``k3_spec_accept``'s KDA replay record: ``prev[si[b]] = dummy[b] ? prev[si[b]] : max(n[b] - 1, 0)``."""
    out = prev.clone()
    for b in range(num_accepted.numel()):
        if not bool(dummy[b]):
            out[state_indices[b]] = max(int(num_accepted[b]) - 1, 0)
    return out


@pytest.mark.parametrize("seed", range(4))
def test_update_mamba_states_after_the_kernel_record_is_a_noop(seed):
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )

    g = torch.Generator().manual_seed(seed)
    batch, slots, drafts = 8, 16, 7
    state_indices = torch.randperm(slots, generator=g)[:batch].to(torch.int32)
    num_accepted = torch.randint(1, drafts + 2, (batch,), generator=g).to(torch.int32)
    dummy = torch.rand(batch, generator=g) < 0.3
    prev = torch.randint(0, drafts + 1, (slots,), generator=g).to(torch.int32)
    after_kernel = _kernel_record(prev, state_indices, num_accepted, dummy)

    mgr = SimpleNamespace(
        local_num_mamba_layers=3,
        use_kda_replay_update=True,
        _dummy_request_mask=dummy,
        prev_num_accepted_tokens=after_kernel.clone(),
    )
    mgr._record_kda_replay_acceptance = types.MethodType(
        MambaHybridCacheManagerV2._record_kda_replay_acceptance, mgr
    )
    attn_metadata = SimpleNamespace(num_seqs=batch, num_contexts=0)
    MambaHybridCacheManagerV2.update_mamba_states(mgr, attn_metadata, num_accepted, state_indices)
    assert torch.equal(mgr.prev_num_accepted_tokens, after_kernel)


@pytest.mark.parametrize("glue", [0, 1])
@pytest.mark.parametrize("fused", [False, True])
def test_mla_worker_skips_the_update_only_on_fused_glue_steps(fused, glue, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.spec_worker import (
        KimiK3MLADSparkWorker,
    )

    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_DRAFTER_GLUE", glue)
    calls = []
    mgr = SimpleNamespace(update_mamba_states=lambda **kw: calls.append(kw))
    attn_metadata = SimpleNamespace(
        kv_cache_manager=mgr, mamba_metadata=SimpleNamespace(state_indices="si")
    )
    worker = object.__new__(KimiK3MLADSparkWorker)
    worker._k3_prep = {"rewind": None} if fused else None
    worker._update_mamba_states(attn_metadata, "n")
    if fused and glue:
        assert calls == []
    else:
        assert calls == [
            dict(attn_metadata=attn_metadata, num_accepted_tokens="n", state_indices="si")
        ]


@needs_cuda
@pytest.mark.parametrize("glue", [0, 1])
def test_mla_worker_step_arange(glue, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.spec_worker import (
        KimiK3MLADSparkWorker,
    )

    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_DRAFTER_GLUE", glue)
    worker = object.__new__(KimiK3MLADSparkWorker)
    for start, end in ((0, 8), (2, 6), (0, 1)):
        a = worker._step_arange(start, end)
        b = worker._step_arange(start, end)
        want = torch.arange(start, end, dtype=torch.long, device="cuda")
        _same(a, want, f"arange({start}, {end})")
        assert (a.data_ptr() == b.data_ptr()) == bool(glue), "cached exactly when the switch is on"

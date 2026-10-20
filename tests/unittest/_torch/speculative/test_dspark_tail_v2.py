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
"""``TLLM_DSPARK_TAIL_V2`` kernels write bitwise what the launches they replace write."""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="needs an SM100-family GPU"
)

KV_LORA = 512
ROPE = 64
NOPE = 128
HEAD_DIM = KV_LORA + ROPE
EPS = 1e-5


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _freqs_table(num_positions: int, gen) -> torch.Tensor:
    """A ``[P, ROPE // 2, 2]`` FP32 (cos, sin) table with YaRN-like magnitudes."""
    angle = torch.rand(num_positions, ROPE // 2, device="cuda", generator=gen) * 6.3
    scale = 1.0 + 0.3 * torch.rand(num_positions, 1, device="cuda", generator=gen)
    return torch.stack([angle.cos() * scale, angle.sin() * scale], dim=-1).contiguous()


def _rows(n, width, gen, row_stride=None):
    """BF16 rows with a spread of magnitudes; optionally a row-strided view."""
    row_stride = row_stride or width
    base = torch.randn(n, row_stride, device="cuda", generator=gen)
    base = base * torch.logspace(-2, 2, n, device="cuda").unsqueeze(1)
    return base.to(torch.bfloat16)[:, :width]


@pytest.mark.parametrize("num_tokens", [1, 7, 14, 28, 33])
def test_ctx_norm_rope_matches_per_layer_kernel(num_tokens):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        _compile_fused_dspark_rmsnorm_rope,
        dspark_mla_ctx_norm_rope,
    )

    layers = 5
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    latent = _rows(num_tokens, layers * HEAD_DIM, gen).view(num_tokens, layers, HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(layers, KV_LORA, device="cuda", generator=gen)).to(
        torch.bfloat16
    )
    table = _freqs_table(4096, gen)
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda", generator=gen)

    out = dspark_mla_ctx_norm_rope(latent, torch.nn.Parameter(weight), positions, table, EPS)

    compiled = _compile_fused_dspark_rmsnorm_rope(
        HEAD_DIM, ROPE, 1, EPS, True, True, False, KV_LORA
    )
    gathered = table[positions].contiguous()
    ref = torch.empty_like(out)
    for layer in range(layers):
        compiled(latent[:, layer], weight[layer], gathered, ref[layer])
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("num_tokens", [7, 14, 28, 32])
@pytest.mark.parametrize("page_size", [32, 64])
def test_block_rope_matches_separate_launches(num_tokens, page_size):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        cute_dsl_dspark_rope_into,
        dspark_mla_block_rope,
    )
    from tensorrt_llm._torch.speculative.spec_tail_kernels import scatter_rows_to_pool

    heads = 4
    gen = torch.Generator(device="cuda").manual_seed(100 * num_tokens + page_size)
    q = _rows(num_tokens * heads, NOPE + ROPE, gen).view(num_tokens, heads, NOPE + ROPE)
    # The block latents are the kv half of the fused [q_a | kv_a] GEMM output.
    kv = _rows(num_tokens, HEAD_DIM, gen, row_stride=1536 + HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(KV_LORA, device="cuda", generator=gen)).to(torch.bfloat16)
    table = _freqs_table(8192, gen)
    positions = torch.randint(0, 8192, (num_tokens,), device="cuda", generator=gen)
    # An interleaved multi-layer page space, like the draft KV manager's view.
    num_pages = 40
    space = torch.randn(num_pages, 3, page_size, HEAD_DIM, device="cuda", generator=gen)
    pool = space.to(torch.bfloat16)[:, 1]
    pages = torch.randperm(num_pages, device="cuda", generator=gen)[:num_tokens].long()
    offsets = torch.randint(0, page_size, (num_tokens,), device="cuda", generator=gen)

    query = torch.full((num_tokens, heads, HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda")
    pool_new = pool.clone()
    dspark_mla_block_rope(
        q, query, kv, torch.nn.Parameter(weight), positions, table, pool_new, pages, offsets, EPS
    )

    gathered = table[positions].contiguous()
    query_ref = torch.full_like(query, 7.0)
    cute_dsl_dspark_rope_into(q, gathered, query_ref, heads, ROPE, KV_LORA)
    blk_kv = cute_dsl_dspark_rmsnorm_rope(
        kv, weight, gathered, 1, ROPE, EPS, True, True, False, KV_LORA
    )
    pool_ref = pool.clone()
    scatter_rows_to_pool(blk_kv, pool_ref, pages, offsets)
    torch.cuda.synchronize()
    assert torch.equal(_bits(query), _bits(query_ref))
    assert torch.equal(_bits(pool_new), _bits(pool_ref))


@pytest.mark.parametrize(
    "rows,n_blocks,holes",
    [(1, 31, False), (2, 2048, False), (3, 4097, True), (8, 30000, False), (64, 6000, True)],
)
def test_block_table_decode_parallel_matches(rows, n_blocks, holes):
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
        BLOCK_TABLE_CHUNK,
        decode_draft_block_table,
        decode_draft_block_table_parallel,
    )

    gen = torch.Generator(device="cuda").manual_seed(rows * 7 + n_blocks)
    divisor = 10
    src_full = torch.randint(0, 50000, (2, rows + 1, 2, n_blocks), device="cuda", generator=gen)
    src_full = src_full.to(torch.int32)
    lengths = torch.randint(0, n_blocks + 1, (rows + 1,), device="cuda", generator=gen)
    cols = torch.arange(n_blocks, device="cuda")
    src_full[1] = torch.where(cols < lengths.view(-1, 1, 1), src_full[1], -1)
    if holes:
        src_full[1, :, :, ::7] = -1
    src = src_full[1, :rows, 0]

    table_ref = torch.full((rows + 1, n_blocks), -5, dtype=torch.int32, device="cuda")
    counts_ref = torch.full((rows + 1,), -5, dtype=torch.long, device="cuda")
    decode_draft_block_table(src, divisor, table_ref[:rows], counts_ref[:rows])

    table = torch.full_like(table_ref, -5)
    counts = torch.full_like(counts_ref, -5)
    partial = torch.full(
        (rows * -(-n_blocks // BLOCK_TABLE_CHUNK),), -1, dtype=torch.int32, device="cuda"
    )
    decode_draft_block_table_parallel(src, divisor, table[:rows], counts[:rows], partial)
    torch.cuda.synchronize()
    assert torch.equal(table, table_ref)
    assert torch.equal(counts, counts_ref)


def _pairs(rows, n_slots, gen, ties=False, nans=False):
    """An exchanged pair buffer: ``[rows, width]`` FP32, ``(value, index)`` per slot."""
    width = triton_next_pow2(2 * n_slots)
    buf = torch.zeros(rows, width, device="cuda")
    vals = torch.randn(rows, n_slots, device="cuda", generator=gen)
    if ties:
        vals = vals.round()
    if nans:
        vals[0, 3] = float("nan")
        vals[1, 7] = float("nan")
        vals[1, 2] = float("nan")
    idx = torch.randint(0, 163840, (rows, n_slots), device="cuda", generator=gen).float()
    buf[:, 0 : 2 * n_slots : 2] = vals
    buf[:, 1 : 2 * n_slots : 2] = idx
    return buf, width


def triton_next_pow2(x):
    return 1 << (x - 1).bit_length()


@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (0, 2), (0, 4), (2, 3), (3, 0)])
@pytest.mark.parametrize("force", [None, "const", "frac", "state", "state_ties"])
@pytest.mark.parametrize("ties,nans", [(False, False), (True, True)])
def test_fused_select_accept_matches_reduce_then_accept(num_contexts, num_gens, force, ties, nans):
    import types
    from types import SimpleNamespace

    import triton

    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import KDAHybridCacheManagerV2
    from tensorrt_llm._torch.speculative import vocab_parallel as vp
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
        FORCE_ACCEPT_TIE_STRIDE,
        ForceAcceptKey,
        fused_select_accept,
        fused_strict_accept,
    )

    K, n_slots = 6, 80
    batch = num_contexts + num_gens
    rows = num_contexts + num_gens * (K + 1)
    gen = torch.Generator(device="cuda").manual_seed(rows * 13 + int(ties) + len(str(force)))
    buf, width = _pairs(max(rows, 2), n_slots, gen, ties=ties, nans=nans and rows >= 2)
    buf = buf[:rows].contiguous()
    targets = torch.empty(rows, dtype=torch.int64, device="cuda")
    vp._reduce_pairs_kernel[(triton.cdiv(max(rows, 1), vp._REDUCE_BLOCK_ROWS),)](
        buf,
        width,
        2,
        rows,
        n_slots,
        targets,
        BLOCK_ROWS=vp._REDUCE_BLOCK_ROWS,
        BLOCK_SLOTS=triton.next_power_of_2(n_slots),
    )
    # Drafts that match a random-length prefix of the targets.
    t_gen = targets[num_contexts:].view(num_gens, K + 1) if num_gens else targets[:0].view(0, K + 1)
    drafts = torch.randint(0, 163840, (num_gens, K), device="cuda", generator=gen).to(torch.int32)
    for g in range(num_gens):
        keep = int(torch.randint(0, K + 1, (1,), device="cuda", generator=gen))
        drafts[g, :keep] = t_gen[g, :keep].to(torch.int32)
    pool = torch.rand(1 << 10, device="cuda", generator=gen)
    # State key: each generation request's newest committed position and token head its group.
    kgen = torch.Generator(device="cuda").manual_seed(rows)
    key_pos = torch.randint(0, 1 << 20, (num_gens * (K + 1),), device="cuda", generator=kgen)
    key_tok = torch.randint(0, 163840, (num_gens * (K + 1),), device="cuda", generator=kgen)
    if force == "state_ties" and num_gens > 1:
        # The last request is in the same state as the first: told apart by the tie count.
        key_pos[(num_gens - 1) * (K + 1)] = key_pos[0]
        key_tok[(num_gens - 1) * (K + 1)] = key_tok[0]
    key = ForceAcceptKey(key_pos.to(torch.int32), key_tok.to(torch.int32), K + 1)
    force_args = {
        None: None,
        "const": (3,),
        "frac": (2, 0.6, K + 1, pool, torch.zeros(1, dtype=torch.int64, device="cuda"), 7, 3),
        "state": (2, 0.6, K + 1, pool, key, 7, 3),
        "state_ties": (2, 0.6, K + 1, pool, key, 7, 3),
    }[force]
    ref_force = force_args
    if force == "frac":
        ref_force = force_args[:4] + (force_args[4].clone(),) + force_args[5:]
    ref_acc, ref_n = fused_strict_accept(targets, drafts, num_contexts, batch, ref_force)

    # KDA replay record and drafter KV prep, eager reference.
    slots = torch.randperm(64, device="cuda", generator=gen)[: max(batch, 1)].to(torch.int32)
    dummy = torch.zeros(64, dtype=torch.bool, device="cuda")
    if num_gens > 1:
        dummy[num_contexts + num_gens - 1] = True
    prev0 = torch.randint(0, 7, (64,), device="cuda", generator=gen).to(torch.int32)
    kv0 = torch.randint(0, 1000, (64,), device="cuda", generator=gen).to(torch.int32)
    mgr = SimpleNamespace(
        _use_kda_replay_update=True,
        local_num_mamba_layers=1,
        prev_num_accepted_tokens=prev0.clone(),
        _dummy_request_mask=dummy,
    )
    mgr._record_kda_replay_acceptance = types.MethodType(
        KDAHybridCacheManagerV2._record_kda_replay_acceptance, mgr
    )
    md = SimpleNamespace(num_contexts=num_contexts, num_seqs=batch)
    if num_gens > 0:
        KDAHybridCacheManagerV2.update_mamba_states(
            mgr, attn_metadata=md, num_accepted_tokens=ref_n, state_indices=slots
        )
    ref_kv = kv0.clone()
    ref_rewind = 1 - ref_n[num_contexts:batch]
    if batch > num_contexts:
        ref_kv[num_contexts:batch] += 1

    prev = prev0.clone()
    kv = kv0.clone()
    acc, n, rewind = fused_select_accept(
        (buf, width, 2, n_slots),
        drafts,
        num_contexts,
        batch,
        force_args,
        kda_replay=(prev, slots, dummy) if num_gens > 0 else None,
        kv_lens=kv,
    )
    torch.cuda.synchronize()
    gen_rows = slice(num_contexts, batch)
    assert torch.equal(acc[gen_rows], ref_acc[gen_rows])
    assert torch.equal(acc[:num_contexts, 0], ref_acc[:num_contexts, 0])
    assert torch.equal(n, ref_n)
    assert torch.equal(prev, mgr.prev_num_accepted_tokens)
    assert torch.equal(kv, ref_kv)
    assert torch.equal(rewind, ref_rewind)
    if force == "frac":
        assert torch.equal(force_args[4], ref_force[4])
    if force in ("state", "state_ties") and num_gens > 0:
        p, t = key_pos[:: K + 1], key_tok[:: K + 1]
        rows_ = torch.arange(num_gens, device="cuda")
        ties = (
            (p[:, None] == p[None, :])
            & (t[:, None] == t[None, :])
            & (rows_[None, :] < rows_[:, None])
        ).sum(1)
        assert int(ties.sum()) == (1 if force == "state_ties" and num_gens > 1 else 0)
        idx = (p * 7 + t * 3 + ties * FORCE_ACCEPT_TIE_STRIDE) & (pool.numel() - 1)
        expected = (2 + (pool[idx] < 0.6).to(torch.int32)).clamp(max=K + 1)
        assert torch.equal(n[gen_rows], expected)


@pytest.mark.parametrize("num_contexts,num_gens", [(0, 2), (1, 3), (0, 16)])
def test_next_new_tokens_rewind_matches(num_contexts, num_gens):
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
        fused_next_new_tokens,
        fused_next_new_tokens_rewind,
    )

    K = 6
    batch = num_contexts + num_gens
    gen = torch.Generator(device="cuda").manual_seed(batch)
    accepted = torch.randint(0, 1000, (batch, K + 1), device="cuda", generator=gen).to(torch.int32)
    n = torch.randint(1, K + 2, (batch,), device="cuda", generator=gen).to(torch.int32)
    drafts = torch.randint(0, 1000, (batch, K), device="cuda", generator=gen).to(torch.int32)
    kv0 = torch.randint(0, 9, (32,), device="cuda", generator=gen).to(torch.int32)
    rewind = (1 - n[num_contexts:]).contiguous()
    ref = fused_next_new_tokens(accepted, drafts, n)
    ref_kv = kv0.clone()
    ref_kv[num_contexts:batch] -= rewind
    ref_kv[num_contexts:batch].clamp_(min=0)
    kv = kv0.clone()
    out = fused_next_new_tokens_rewind(accepted, drafts, n, kv, rewind, num_contexts)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert torch.equal(kv, ref_kv)


@pytest.mark.parametrize(
    "role,n,k",
    [
        ("q_kv_a", 2112, 7168),
        ("q_b", 768, 1536),
        ("gate_up", 1792, 7168),
        ("down", 7168, 896),
        ("ctx_latent", 2880, 7168),
        ("lm_head", 10240, 7168),
    ],
)
@pytest.mark.parametrize("m", [7, 14, 28, 32, 40])
def test_draft_gemm_pins(monkeypatch, role, n, k, m):
    """The pinned drafter GEMMs compute x @ W^T (to BF16 rounding), and step aside
    above the pinned row range or with the switch off."""
    from tensorrt_llm._torch.speculative.dspark_draft_gemm import DRAFT_GEMM_ENV_VAR, draft_gemm

    gen = torch.Generator(device="cuda").manual_seed(m + n)
    x = (torch.randn(m, k, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).to(torch.bfloat16)
    monkeypatch.setenv(DRAFT_GEMM_ENV_VAR, "0")
    assert draft_gemm(x, w, role) is None
    monkeypatch.setenv(DRAFT_GEMM_ENV_VAR, "1")
    # Module weights are parameters that require grad, also in the server.
    out = draft_gemm(x, torch.nn.Parameter(w), role)
    if m > 32:
        assert out is None
        return
    ref = x.float() @ w.float().t()
    assert out is not None and out.shape == (m, n) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_gens", [1, 2, 4, 16])
def test_ctx_positions_match_gen_prepare(num_gens):
    """The side stream's context positions are the ones dflash_gen_prepare assigns."""
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
        dflash_ctx_positions,
        dflash_gen_prepare,
    )

    width, block, page, max_blocks = 7, 7, 32, 64
    gen = torch.Generator(device="cuda").manual_seed(num_gens)
    slots = torch.randperm(32, device="cuda", generator=gen)[:num_gens].long()
    ctx_len = torch.randint(0, 1500, (33,), device="cuda", generator=gen).long()
    positions = dflash_ctx_positions(ctx_len, slots, width)
    prep = dflash_gen_prepare(
        torch.randint(1, width + 1, (num_gens,), device="cuda", generator=gen).to(torch.int32),
        torch.randint(0, 1000, (num_gens, width), device="cuda", generator=gen).to(torch.int32),
        slots,
        ctx_len.clone(),
        torch.full((num_gens,), max_blocks, dtype=torch.long, device="cuda"),
        torch.arange(num_gens * max_blocks, device="cuda", dtype=torch.int32).view(num_gens, -1),
        torch.zeros(num_gens, dtype=torch.int32, device="cuda"),
        0,
        num_gens,
        32,
        1 << 20,
        999,
        page,
        block,
        width,
    )
    torch.cuda.synchronize()
    assert torch.equal(positions, prep["ctx_positions"])


def test_draft_l2_prefetch_mode(monkeypatch):
    """TLLM_DSPARK_L2_PREFETCH: ``1`` forks after [q_a | kv_a] from 64 CTAs, ``qb[:<CTAs>]`` after
    q_b from 16 (or the given) CTAs."""
    from tensorrt_llm._torch.speculative import dspark_draft_gemm as m

    for value, mode in (
        ("0", None),
        ("", None),
        ("1", ("q_kv_a", 64)),
        ("qb", ("q_b", 16)),
        ("qb:12", ("q_b", 12)),
    ):
        monkeypatch.setenv(m.L2_PREFETCH_ENV_VAR, value)
        assert m.l2_prefetch_mode() == mode
        assert m.l2_prefetch_enabled() == (mode is not None)


def test_draft_l2_prefetch_weights_after_qb():
    """The weights prefetched after q_b: gate_up (the SwiGLU copy once built), down, then the next
    layer's fused [q_a | kv_a] weight (or its two weights)."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    def lin(rows):
        return SimpleNamespace(weight=torch.empty(rows, 8))

    layers = []
    for _ in range(2):
        mlp = SimpleNamespace(gate_up_proj=lin(4), down_proj=lin(2))
        attn = SimpleNamespace(q_a_proj=lin(3), kv_a_proj_with_mqa=lin(5))
        layers.append(SimpleNamespace(mlp=mlp, self_attn=attn))
    fused = lin(8)
    stub = SimpleNamespace(
        model=SimpleNamespace(layers=layers), _draft_fused_a_proj=lambda idx: fused
    )
    fn = MLADSparkForCausalLM._draft_l2_prefetch_weights_after_qb
    got = fn(stub, 0, 2, True)
    assert [w.shape[0] for w in got] == [4, 2, 8]
    got = fn(stub, 0, 2, False)
    assert [w.shape[0] for w in got] == [4, 2, 3, 5]
    layers[1].mlp.__dict__["_dspark_swiglu_weight"] = (None, torch.empty(6, 8))
    got = fn(stub, 1, 2, True)
    assert [w.shape[0] for w in got] == [6, 2]


@pytest.mark.parametrize("max_ctas", [64, 16])
def test_draft_l2_prefetch_in_cuda_graph(max_ctas):
    """The drafter's side-stream L2 prefetch forks and joins inside a CUDA graph
    capture and leaves the results bit for bit unchanged; a first use inside a
    capture skips the prefetch."""
    from tensorrt_llm._torch.speculative.dspark_draft_gemm import DraftL2Prefetch

    gen = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(14, 1024, device="cuda", generator=gen).to(torch.bfloat16)
    w1 = (torch.randn(2048, 1024, device="cuda", generator=gen) / 32).to(torch.bfloat16)
    w2 = torch.nn.Parameter(
        (torch.randn(512, 2048, device="cuda", generator=gen) / 45).to(torch.bfloat16)
    )
    w3 = (torch.randn(1024, 512, device="cuda", generator=gen) / 23).to(torch.bfloat16)

    def body(pf):
        y = x @ w1.t()
        if pf is not None:
            pf.prefetch([w2, None, w3], max_ctas=max_ctas)
        z = (y @ w2.detach().t()) @ w3.t()
        if pf is not None:
            pf.join()
        return z

    with torch.inference_mode():
        ref = body(None)
        pf = DraftL2Prefetch()
        eager = body(pf)  # creates the side stream outside the capture
        late = DraftL2Prefetch()
        outs = {}
        for name, p in (("pf", pf), ("late", late)):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs[name] = body(p)
            g.replay()
        torch.cuda.synchronize()
    assert late._stream is None  # first use inside a capture: nothing forked
    assert torch.equal(eager, ref)
    assert torch.equal(outs["pf"], ref)
    assert torch.equal(outs["late"], ref)


@pytest.mark.parametrize("tp", [2, 4, 8, 16])
def test_markov_kernels_compile_at_decode_shapes(tp):
    """The fused Markov kernels compile at the decode graph's constexprs for every
    TP size (the multi-GPU tests only reach TP 2 and 4): a 10240-column vocab
    shard, markov rank 256, 6 or 7 steps, 1-4 requests, and rank-dependent
    integer arguments (Triton specializes integers equal to 1)."""
    import triton

    from tensorrt_llm._torch.speculative import vocab_parallel as vp

    shard, rank, vocab = 10240, 256, 163840
    programs = triton.cdiv(shard, vp._MARKOV_BLOCK_V)
    set_floats = vp._CHAIN_MAX_STEPS * vp._CHAIN_MAX_REQS * tp * 2
    width = max(vp._MARKOV_EXCHANGE_WIDTH, triton.next_power_of_2(2 * tp))
    f32, i32, bf16 = torch.float32, torch.int32, torch.bfloat16
    w1 = torch.empty(vocab, rank, dtype=bf16, device="cuda")
    w2 = torch.empty(shard, rank, dtype=bf16, device="cuda")
    partial = torch.empty(
        vp._CHAIN_MAX_STEPS * programs * vp._CHAIN_MAX_REQS * 2, dtype=f32, device="cuda"
    )
    tickets = torch.zeros(vp._CHAIN_MAX_STEPS, dtype=i32, device="cuda")
    launch = torch.zeros(1, dtype=i32, device="cuda")
    xbuf = torch.empty(3 * set_floats, dtype=f32, device="cuda")
    for steps, num_reqs, tp_rank in ((6, 2, 1), (7, 1, 0), (7, 4, min(5, tp - 1))):
        local = torch.empty(num_reqs, steps, shard, dtype=bf16, device="cuda")
        anchors = torch.zeros(num_reqs, dtype=i32, device="cuda")
        tokens = torch.empty(num_reqs, steps, dtype=i32, device="cuda")
        vp._markov_chain_kernel.warmup(
            anchors, w1, w2, w2.stride(0), local, local.stride(0), local.stride(1), tokens,
            tokens.stride(0), partial, tickets, launch, xbuf, xbuf, num_reqs, vocab, shard,
            tp_rank * shard, tp_rank, STEPS=steps, RANK=rank, TP=tp,
            BLOCK_B=vp._CHAIN_MAX_REQS, BLOCK_V=vp._MARKOV_BLOCK_V,
            BLOCK_P=triton.next_power_of_2(programs), SET_FLOATS=set_floats, num_warps=4,
            grid=(programs,),
        )  # fmt: skip
        pairs = torch.empty(num_reqs, width, dtype=f32, device="cuda")
        for step in (0, 1, steps - 1):
            vp._markov_round_kernel.warmup(
                pairs, width, tp, anchors, w1, w2, w2.stride(0), local[:, 0], local.stride(0),
                tokens, tokens.stride(0), step, partial, tickets, pairs, width, tp_rank,
                num_reqs, vocab, shard, tp_rank * shard, RANK=rank, FROM_ANCHOR=step == 0,
                ZERO_WIDTH=width, ZERO_CHUNK=triton.next_power_of_2(triton.cdiv(width, programs)),
                BLOCK_B=16, BLOCK_SLOTS=triton.next_power_of_2(tp), BLOCK_V=vp._MARKOV_BLOCK_V,
                BLOCK_P=triton.next_power_of_2(programs), num_warps=4, grid=(programs,),
            )  # fmt: skip


@pytest.mark.parametrize("num_tokens", [7, 14, 32])
def test_block_rope_pdl_matches(monkeypatch, num_tokens):
    """``TLLM_DSPARK_ROPE_PDL=1`` writes what the plain launch writes, also when it follows a
    PDL GEMM in a CUDA graph."""
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import dspark_mla_block_rope
    from tensorrt_llm._torch.speculative.dspark_draft_gemm import DRAFT_GEMM_ENV_VAR, draft_gemm

    heads = 4
    gen = torch.Generator(device="cuda").manual_seed(31 * num_tokens)
    x = (torch.randn(num_tokens, 1536, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(heads * (NOPE + ROPE), 1536, device="cuda", generator=gen) / 40).to(
        torch.bfloat16
    )
    kv = _rows(num_tokens, HEAD_DIM, gen, row_stride=1536 + HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(KV_LORA, device="cuda", generator=gen)).to(torch.bfloat16)
    table = _freqs_table(4096, gen)
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda", generator=gen)
    pool = torch.randn(48, 64, HEAD_DIM, device="cuda", generator=gen).to(torch.bfloat16)
    pages = torch.randperm(48, device="cuda", generator=gen)[:num_tokens].long()
    offsets = torch.randint(0, 64, (num_tokens,), device="cuda", generator=gen)
    monkeypatch.setenv(DRAFT_GEMM_ENV_VAR, "1")

    def run(pdl: str):
        monkeypatch.setenv("TLLM_DSPARK_ROPE_PDL", pdl)
        q = draft_gemm(x, w, "q_b").view(num_tokens, heads, NOPE + ROPE)
        query = torch.zeros((num_tokens, heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        out_pool = pool.clone()
        dspark_mla_block_rope(q, query, kv, weight, positions, table, out_pool, pages, offsets, EPS)
        return query, out_pool

    ref_query, ref_pool = run("0")
    query, out_pool = run("1")
    torch.cuda.synchronize()
    assert torch.equal(_bits(query), _bits(ref_query))
    assert torch.equal(_bits(out_pool), _bits(ref_pool))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_query, g_pool = run("1")
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(g_query), _bits(ref_query))
    assert torch.equal(_bits(g_pool), _bits(ref_pool))


@pytest.mark.parametrize("role,n,k", [("q_kv_a", 2112, 7168), ("gate_up", 1792, 7168),
                                      ("lm_head", 10240, 7168)])  # fmt: skip
@pytest.mark.parametrize("m", [7, 14, 28, 32])
def test_draft_gemm_prefetch_matches_pins(monkeypatch, role, n, k, m):
    """``TLLM_DSPARK_DRAFT_PREFETCH`` / ``TLLM_DSPARK_DRAFT_EXACT_SMEM`` (deeper ring, L2
    prefetch of the following K tiles, exact shared memory) give the pinned kernels' values."""
    from tensorrt_llm._torch.speculative.dspark_draft_gemm import (
        DRAFT_GEMM_ENV_VAR,
        DRAFT_PREFETCH_ENV_VAR,
        draft_gemm,
    )

    gen = torch.Generator(device="cuda").manual_seed(7 * m + n)
    x = (torch.randn(m, k, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).to(torch.bfloat16)
    monkeypatch.setenv(DRAFT_GEMM_ENV_VAR, "1")
    monkeypatch.setenv(DRAFT_PREFETCH_ENV_VAR, "0")
    monkeypatch.setenv("TLLM_DSPARK_DRAFT_EXACT_SMEM", "0")
    ref = draft_gemm(x, w, role)
    monkeypatch.setenv(DRAFT_PREFETCH_ENV_VAR, "1")
    monkeypatch.setenv("TLLM_DSPARK_DRAFT_EXACT_SMEM", "1")
    out = draft_gemm(x, w, role)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))


def test_block_logit_window(monkeypatch):
    """``TLLM_DSPARK_LMHEAD_ALL_ROWS``: the per-request slot window of the draft gather ids."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_draft_slot_ids

    def window(ids, num_gens, block, k):
        return DFlashWorker._block_logit_window(SimpleNamespace(), ids, num_gens, block, k)

    monkeypatch.setenv("TLLM_DSPARK_LMHEAD_ALL_ROWS", "0")
    assert window(dflash_draft_slot_ids(2, 7, 6, True), 2, 7, 6) is None
    monkeypatch.setenv("TLLM_DSPARK_LMHEAD_ALL_ROWS", "1")
    assert window(dflash_draft_slot_ids(2, 7, 6, True), 2, 7, 6) == 0
    assert window(dflash_draft_slot_ids(4, 8, 7, True), 4, 8, 7) == 0
    assert window(dflash_draft_slot_ids(2, 8, 6, False), 2, 8, 6) == 1
    # Slots 1..K of a K-slot block overrun the block: not a window.
    ids = dflash_draft_slot_ids(3, 6, 6, False).clamp(max=17)
    assert window(ids, 3, 6, 6) is None


@pytest.mark.parametrize("batch,ctx", [(1, 300), (2, 1024), (4, 2500)])
def test_mla_decode_fp16_pdl_matches(monkeypatch, batch, ctx):
    """``TRTLLM_MLA_FP16_PDL=1`` (split-KV kernel and reduction as programmatic dependents)
    gives the drafter's block decode the same output, eagerly and in a CUDA graph."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
    from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup

    heads, block, page = 4, 7, 64
    gen = torch.Generator(device="cuda").manual_seed(batch * 1000 + ctx)
    pages_per_req = (ctx + block + page - 1) // page + 1
    pool = (torch.randn(batch * pages_per_req + 1, page, HEAD_DIM, device="cuda", generator=gen)
            * 0.5).to(torch.bfloat16)  # fmt: skip
    tables = torch.arange(batch * pages_per_req, device="cuda").view(batch, pages_per_req)
    ctx_len = torch.full((batch,), ctx, dtype=torch.long, device="cuda")
    fixup = _build_mla_block_fixup(ctx_len, tables, block, page)
    query = (torch.randn(batch, block, heads, HEAD_DIM, device="cuda", generator=gen)).to(
        torch.bfloat16
    )
    ws_numel = CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        heads, block, KV_LORA, batch, __import__("cutlass").Float32
    )
    workspace = torch.zeros(max(ws_numel, 1 << 20), dtype=torch.uint8, device="cuda")

    def decode(pdl: str):
        monkeypatch.setenv("TRTLLM_MLA_FP16_PDL", pdl)
        out = torch.empty(batch, block, heads, KV_LORA, dtype=torch.bfloat16, device="cuda")
        torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell(
            query[..., :KV_LORA].permute(2, 3, 1, 0),
            query[..., KV_LORA:].permute(2, 3, 1, 0),
            pool[..., :KV_LORA].permute(1, 2, 0),
            pool[..., KV_LORA:].permute(1, 2, 0),
            fixup.page_tables_i32.t(),
            fixup.seq_lens_i32,
            out.permute(2, 3, 1, 0),
            workspace,
            heads,
            block,
            page,
            0.07,
            1.0,
            batch,
            None,
            fixup.kv_bounds_i32,
        )
        return out

    ref = decode("0")
    out = decode("1")
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = decode("1")
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(g_out), _bits(ref))


@pytest.mark.parametrize("m", [7, 14, 28, 32])
@pytest.mark.parametrize("h,row_stride", [(1536, 2112), (1536, 1536), (7168, 7168)])
def test_rmsnorm_early_trigger_matches_flashinfer(monkeypatch, m, h, row_stride):
    """``TLLM_DSPARK_NORM_EARLY_TRIGGER``: the early-trigger RMSNorm gives the values of the
    RMSNorm module's FlashInfer kernel (also on the strided q_a rows of [q_a | kv_a])."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm._torch.speculative.dspark_norm import (
        NORM_EARLY_TRIGGER_ENV_VAR,
        draft_rmsnorm,
    )

    gen = torch.Generator(device="cuda").manual_seed(m * h + row_stride)
    x = _rows(m, h, gen, row_stride=row_stride)
    norm = RMSNorm(hidden_size=h, eps=1e-6, dtype=torch.bfloat16).cuda()
    norm.weight.data.copy_(1 + 0.3 * torch.randn(h, device="cuda", generator=gen))
    monkeypatch.setenv(NORM_EARLY_TRIGGER_ENV_VAR, "0")
    assert draft_rmsnorm(norm, x) is None
    ref = norm(x)
    monkeypatch.setenv(NORM_EARLY_TRIGGER_ENV_VAR, "1")
    out = draft_rmsnorm(norm, x)
    torch.cuda.synchronize()
    assert out is not None
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("m", [1, 7, 14, 16, 28, 32])
def test_gate_up_swiglu_matches(monkeypatch, m):
    """``TLLM_DSPARK_GATE_UP_SWIGLU``: the pinned gate_up GEMM with the SwiGLU epilogue gives the
    values of the pinned GEMM followed by ``trtllm::silu_and_mul`` (signed zeros, large and tiny
    gate values included)."""
    import torch.nn.functional as F

    from tensorrt_llm._torch.speculative.dspark_draft_gemm import (
        DRAFT_GEMM_ENV_VAR,
        GATE_UP_SWIGLU_ENV_VAR,
        draft_gemm,
        draft_gemm_swiglu,
    )

    class _Lin:
        def __init__(self, w):
            self.weight = w

    class _Mlp:
        def __init__(self, w):
            self.gate_up_proj = _Lin(w)
            self.activation = F.silu
            self.swiglu_limit = None
            self.swiglu_alpha = None
            self.swiglu_beta = None

    gen = torch.Generator(device="cuda").manual_seed(m)
    hidden, inter = 7168, 896
    x = (torch.randn(m, hidden, device="cuda", generator=gen) * 2).to(torch.bfloat16)
    w = (torch.randn(2 * inter, hidden, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w[:8] *= 64  # large gate values (saturating sigmoid)
    w[inter : inter + 8] = 0  # up rows of exact zeros
    mlp = _Mlp(w)
    monkeypatch.setenv(DRAFT_GEMM_ENV_VAR, "1")
    monkeypatch.setenv(GATE_UP_SWIGLU_ENV_VAR, "0")
    assert draft_gemm_swiglu(x, mlp) is None
    ref = torch.ops.trtllm.silu_and_mul(draft_gemm(x, w, "gate_up"))
    monkeypatch.setenv(GATE_UP_SWIGLU_ENV_VAR, "1")
    out = draft_gemm_swiglu(x, mlp)
    torch.cuda.synchronize()
    assert out is not None
    assert torch.equal(_bits(out), _bits(ref))

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
@pytest.mark.parametrize("force", [None, "const", "frac", "state"])
@pytest.mark.parametrize("ties,nans", [(False, False), (True, True)])
def test_fused_select_accept_matches_reduce_then_accept(num_contexts, num_gens, force, ties, nans):
    import types
    from types import SimpleNamespace

    import triton

    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import KDAHybridCacheManagerV2
    from tensorrt_llm._torch.speculative import vocab_parallel as vp
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
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
    key = ForceAcceptKey(key_pos.to(torch.int32), key_tok.to(torch.int32), K + 1)
    force_args = {
        None: None,
        "const": (3,),
        "frac": (2, 0.6, K + 1, pool, torch.zeros(1, dtype=torch.int64, device="cuda"), 7, 3),
        "state": (2, 0.6, K + 1, pool, key, 7, 3),
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
    if force == "state" and num_gens > 0:
        idx = (key_pos[:: K + 1] * 7 + key_tok[:: K + 1] * 3) & (pool.numel() - 1)
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


def test_draft_l2_prefetch_in_cuda_graph():
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
            pf.prefetch([w2, None, w3])
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A`` of the Kimi K3 NVFP4 decode path, each arm against the other, the off arm
against the chain it replaced and both against fp64, at W ranks of a Helix mapping (tp 1 x cp W, 6 heads per rank:
the target's per-rank shape at W = 16).

    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _kimi_k3_helix_v_up_op_matrix.py --launcher srun --world-size 16
    srun -N 1 --ntasks-per-node 4 --mpi=pmix python _kimi_k3_helix_v_up_op_matrix.py --launcher srun --world-size 4
    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_helix_v_up_op_matrix.py            # W = 4, mpirun

Not a pytest module (one W-rank job; the arms share the split read's exchange state).

What runs. Every rank builds the target's ``MlaSplitRead`` on the Helix mapping with both MNNVL exchange switches at 1
(``build_helix_comm``: the query all-gather on an ``MnnvlWorkspace``, ``helix_a2a_fused``) and drives the op chain of
``K3DecodeMLA._quantized_decode`` after the query projection: ``K3DecodeMLA._split_read_output`` itself, on a stand-in
module holding what it reads (the split read, the layer's attention stand-in, this rank's FP8 V-B head group and the
BF16 W_UV of every head as ``set_helix_v_up`` builds them), with ``TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A`` set on the
flags module per arm, as the forward reads it. The metadata is a stand-in holding what the split read reads: this
rank's Helix-sharded FP8 pages (only its round-robin pages of a global cache, every other page garbage), its local
block table and lengths, the step's global positions. Requests of 1, 2 and 4 x 8 tokens (8, 16, 32 rows per head);
one request short enough that some ranks hold none of its pages.

Checks:
  * each arm with the gate from the side stream (``TRTLLM_K3LL_MLA_GATE_SIDE``, the default) and from ``ag`` in
    place: bit for bit the same;
  * off arm against ``_ref_chain`` (the chain before the switch: the query all-gather, ``make_helix_plan``,
    ``split_read_attention``, ``exchange_combine_fused`` of the 512-wide latent, ``k3_fp8_mla_output``) bit for bit;
  * the on arm against itself built from parts (the V-up decode op, ``helix_a2a_fused`` at 128 values per head, then
    ``k3_mla_gate``) and against the NCCL exchange (``TRTLLM_K3LL_HELIX_A2A_FUSED=0``) within ``GATE_TOL``;
  * on arm against the off arm within ``ON_OFF_TOL`` of each (token, head)'s norm, and both against fp64 (attention
    over the global cache, the FP8 V-B codes times their scales, the gate): the on arm within ``FP64_TOL`` per
    (token, head); the median / max of both arms are printed (the current path quantizes the latent to FP8 in groups
    of 128 before its FP8 V-B);
  * a step of two layers per arm captured in a CUDA graph and replayed with new queries and gates, an eager call of
    another row count between replays, a random rank late before each: every replay bit for bit the eager arm on the
    same inputs;
  * the switch at 1 on a GPU it does not build on (``HELIX_SWITCH_CAPABILITIES`` forced empty): ``set_helix_v_up``
    keeps no W_UV and the on arm is bit for bit the off arm.
"""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

DEADLINE_S = 1800
HP = 6
LATENT, ROPE, V, PAGE = 512, 64, 128, 64
Q_LORA = 1536
GATE_COL = Q_LORA + LATENT + ROPE
Q_LEN = 8
# per row count, the global lengths of the requests (including the step's Q_LEN tokens)
CASES = {8: [20000 + Q_LEN], 16: [20000 + Q_LEN, 300], 32: [20000 + Q_LEN, 300, 9000, 4100]}
LAYERS = 2
REPLAYS = 2
WORKSPACE_BYTES = 4 << 20
# Per (token, head) relative L2. On GB300 at W 4 / 16 these inputs give: on vs off up to 4.1e-2 (the current
# path's FP8 latent x FP8 V-B), on vs fp64 up to 3.5e-2 (FP8 attention with BF16 W_UV), off vs fp64 up to 5.5e-2.
ON_OFF_TOL = 8e-2
FP64_TOL = 6e-2
GATE_TOL = (
    2.0**-6
)  # per (token, head): the gate in the combine vs k3_mla_gate (sigmoid approximations, roundings)
STATS = {"on": [], "off": []}

R = None
DCP = None
IMPL = None
FLAGS = None
SR = None
WS = None
TP_MAPPING = None
LOG2_E = 1.4426950408889634


def global_inputs(seed: int, lens):
    """The global FP8 cache, block rows and every head's FP8 queries (the same on every rank), the V-B codes and
    scales of every head, and this rank's gate input rows."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    heads = HP * R.world
    blocks = [-(-n // PAGE) for n in lens]
    num_pages = sum(blocks)
    perm = torch.randperm(num_pages, generator=gen)
    rows = torch.zeros(len(lens), max(blocks), dtype=torch.long)
    k = 0
    for b, nb in enumerate(blocks):
        rows[b, :nb] = perm[k : k + nb]
        k += nb
    pool = (torch.randn(num_pages, PAGE, LATENT + ROPE, generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    q = (torch.randn(len(lens), Q_LEN, heads, LATENT + ROPE, generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    tokens = len(lens) * Q_LEN
    ag = torch.randn(R.world, tokens, GATE_COL + HP * V, generator=gen).bfloat16()
    return pool.cuda(), rows.cuda(), q.cuda(), ag[R.rank].cuda().contiguous()


def v_b(seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    heads = HP * R.world
    codes = (torch.randn(heads, V, LATENT, generator=gen) * 64).clamp(-448, 448)
    scales = torch.rand(heads, 1, LATENT // 128, generator=gen) * 2e-3 + 5e-4
    return codes.to(torch.float8_e4m3fn).cuda(), scales.cuda()


def local_meta(pool, rows, lens, num_layers=LAYERS):
    """This rank's Helix storage and the metadata stand-in ``MlaSplitRead.forward`` reads."""
    w, r = R.world, R.rank
    local = [list(range(r, -(-n // PAGE), w)) for n in lens]
    width = max(1, max(len(x) for x in local))
    gen = torch.Generator(device="cpu").manual_seed(1000 + r)
    num_local = sum(len(x) for x in local) + 3
    local_pool = (
        (torch.randn(num_local, PAGE, LATENT + ROPE, generator=gen) * 50)
        .to(torch.float8_e4m3fn)
        .cuda()
    )
    table = torch.zeros(len(lens), width, dtype=torch.int32)
    k = 1  # page 0 stays garbage
    for b, pages in enumerate(local):
        for j, g in enumerate(pages):
            table[b, j] = k
            local_pool[k] = pool[rows[b, g]]
            k += 1
    from tensorrt_llm._torch.utils import helix_local_len_tensor

    lens_t = torch.tensor(lens, dtype=torch.int32, device="cuda")
    positions = (lens_t.view(-1, 1) - Q_LEN + torch.arange(Q_LEN, device="cuda")).reshape(-1)
    kv_pool = local_pool.view(num_local, 1, PAGE, 1, LATENT + ROPE)
    meta = SimpleNamespace(
        num_contexts=0,
        num_generations=len(lens),
        kv_cache_manager=SimpleNamespace(get_buffers=lambda layer_idx: kv_pool),
        host_kv_cache_pool_mapping=torch.zeros(num_layers, 2, dtype=torch.int32),
        kv_cache_block_offsets=table.cuda()
        .view(1, len(lens), 1, width)
        .expand(1, len(lens), 2, width),
        kv_lens_cuda_runtime=helix_local_len_tensor(lens_t, PAGE, w, r).to(torch.int32),
        helix_position_offsets=positions.to(torch.int32),
    )
    return meta


def stand_in(layer_idx: int, codes, scales, w_uv):
    """What ``K3DecodeMLA._split_read_output`` reads of the module: the split read, the attention (layer index), this
    rank's FP8 V-B head group, the BF16 W_UV of every head, the MLA widths."""
    group = slice(R.rank * HP, (R.rank + 1) * HP)
    mqa = SimpleNamespace(layer_idx=layer_idx, get_local_layer_idx=lambda meta: 0)
    return SimpleNamespace(
        _split_read=SR,
        _helix_v_up_weight=w_uv,
        mqa=mqa,
        v_b_proj=codes[group].contiguous(),
        v_b_proj_scale=scales[group].contiguous(),
        q_lora_rank=Q_LORA,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        _decode_heads=HP,
        v_head_dim=V,
    )


def scales_of(softmax_scale: float):
    bmm1 = torch.tensor([0.0, softmax_scale * LOG2_E], device="cuda")
    bmm2 = torch.tensor([1.0], device="cuda")
    return bmm1, bmm2


def run_arm(on, module, quant_q, meta, ag, bmm, gate_side: bool = True):
    """``_split_read_output`` with the switch at ``on``. ``gate_side`` (``TRTLLM_K3LL_MLA_GATE_SIDE``, the default):
    the gate comes from the module's side-stream gate (here a stand-in whose join copies ``ag``'s gate columns, so
    both gate forms carry the same values); else from ``ag`` at ``GATE_COL``."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_v_up,
    )

    # "proto": the on arm with the prototype send from the fused reduction (K3LL_PROTO_A2A_SEND)
    mla_v_up.PROTO_A2A_SEND = on == "proto"
    FLAGS.TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A = int(bool(on))
    module._gate_side = SimpleNamespace(
        join=lambda: ag[:, GATE_COL : GATE_COL + HP * V].contiguous()
    )
    return IMPL.K3DecodeMLA._split_read_output(
        module, quant_q, meta, Q_LEN, *bmm, ag, gate_side, GATE_COL
    )


def ref_chain(module, quant_q, meta, ag, softmax_scale):
    """The chain before the switch (``MlaSplitRead.forward`` + ``k3_fp8_mla_output`` as they were)."""
    tokens = quant_q.shape[0]
    requests = tokens // Q_LEN
    heads = HP * R.world
    q_local = quant_q.view(torch.float8_e4m3fn).view(tokens, HP, LATENT + ROPE)
    q_all = DCP.gather_q_mnnvl(q_local.view(torch.uint8), WS)
    q_all = q_all.view(torch.float8_e4m3fn).view(requests, Q_LEN, heads, LATENT + ROPE)
    pages = meta.kv_cache_manager.get_buffers(0).view(torch.float8_e4m3fn)[
        :, 0, :, 0, : LATENT + ROPE
    ]
    plan = DCP.make_helix_plan(
        meta.kv_cache_block_offsets[0, :, 0, :],
        meta.kv_lens_cuda_runtime,
        meta.helix_position_offsets[:tokens],
        PAGE,
        R.world,
        R.rank,
    )
    out = torch.empty(requests, Q_LEN, heads, LATENT, dtype=torch.bfloat16, device="cuda")
    stats = torch.empty(tokens, heads, 2, dtype=torch.float32, device="cuda")
    ws = torch.empty(DCP.workspace_bytes(heads, DCP.MAX_Q_LEN, DCP.MAX_REQUESTS), dtype=torch.uint8,
                     device="cuda")  # fmt: skip
    DCP.split_read_attention(
        q_all, pages, plan, out, stats, ws, softmax_scale, 1.0, DCP.MAX_REQUESTS
    )
    latent = DCP.exchange_combine_fused(out.view(tokens, -1), stats, plan.empty, SR._a2a_fused, HP)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        attention,
    )

    return attention.mla_output(latent, module.v_b_proj, module.v_b_proj_scale, ag, GATE_COL)


def ref_on_chain(module, quant_q, meta, ag, softmax_scale, w_uv):
    """The on arm from its parts: the V-up decode op, ``helix_a2a_fused`` at 128 values per head (no gate), then
    ``k3_mla_gate``."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_v_up,
    )

    tokens = quant_q.shape[0]
    requests = tokens // Q_LEN
    heads = HP * R.world
    q_local = quant_q.view(torch.float8_e4m3fn).view(tokens, HP, LATENT + ROPE)
    q_all = DCP.gather_q_mnnvl(q_local.view(torch.uint8), WS)
    q_all = q_all.view(torch.float8_e4m3fn).view(requests, Q_LEN, heads, LATENT + ROPE)
    pages = meta.kv_cache_manager.get_buffers(0).view(torch.float8_e4m3fn)[
        :, 0, :, 0, : LATENT + ROPE
    ]
    plan = DCP.make_helix_plan(meta.kv_cache_block_offsets[0, :, 0, :], meta.kv_lens_cuda_runtime,
                               meta.helix_position_offsets[:tokens], PAGE, R.world, R.rank)  # fmt: skip
    if FLAGS.TRTLLM_K3LL_MLA_VAR_SPLIT:  # the split counts MlaSplitRead._step_plan gives the on arm
        plan = DCP.with_var_split(plan, heads, Q_LEN)
    latent = torch.empty(requests, Q_LEN, heads, LATENT, dtype=torch.bfloat16, device="cuda")
    stats = torch.empty(tokens, heads, 2, dtype=torch.float32, device="cuda")
    v_part = torch.empty(tokens, heads, V, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(DCP.workspace_bytes(heads, DCP.MAX_Q_LEN, DCP.MAX_REQUESTS), dtype=torch.uint8,
                     device="cuda")  # fmt: skip
    DCP.split_read_attention_v_up(q_all, pages, plan, latent, stats, ws, softmax_scale, 1.0,
                                  DCP.MAX_REQUESTS, w_uv, v_part)  # fmt: skip
    values = SR._a2a_fused(v_part.view(tokens, -1), stats, HP, V, zero_kv_mask=plan.empty)
    return mla_v_up.mla_gate(values, ag, GATE_COL)


def fp64_reference(pool, rows, q, lens, codes, scales, ag, softmax_scale):
    """This rank's heads: attention over the global cache, the FP8 V-B codes times their scales, the gate."""
    group = slice(R.rank * HP, (R.rank + 1) * HP)
    outs = []
    for b, n in enumerate(lens):
        kv = pool[rows[b, : -(-n // PAGE)]].double().reshape(-1, LATENT + ROPE)[:n]
        qb = q[b, :, group].double()
        s = torch.einsum("tjd,nd->tjn", qb, kv) * softmax_scale
        limit = n - Q_LEN + 1 + torch.arange(Q_LEN, device="cuda")
        s = s.masked_fill(
            torch.arange(n, device="cuda").view(1, 1, -1) >= limit.view(-1, 1, 1), float("-inf")
        )
        outs.append(torch.softmax(s, -1) @ kv[:, :LATENT])
    latent = torch.cat(outs)  # [T, HP, 512]
    w = codes[group].double() * scales[group].double().repeat_interleave(128, -2).repeat_interleave(
        128, -1
    )
    v = torch.einsum("thk,hvk->thv", latent, w)
    gate = torch.sigmoid(ag[:, GATE_COL : GATE_COL + HP * V].double()).view(-1, HP, V)
    return (v * gate).reshape(-1, HP * V)


def per_head_err(got, want):
    g = got.double().view(-1, HP, V)
    w = want.double().view(-1, HP, V)
    return (g - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)


def case_inputs(rows_per_head: int, seed: int):
    lens = CASES[rows_per_head]
    pool, rows, q, ag = global_inputs(seed, lens)
    codes, scales = v_b(7)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_v_up,
    )

    w_uv = mla_v_up.dequantize_v_b(codes, scales)
    group = slice(R.rank * HP, (R.rank + 1) * HP)
    quant_q = q[:, :, group].reshape(-1, HP, LATENT + ROPE).contiguous().view(torch.uint8)
    meta = local_meta(pool, rows, lens)
    return SimpleNamespace(lens=lens, pool=pool, rows=rows, q=q, ag=ag, codes=codes, scales=scales,
                           w_uv=w_uv, quant_q=quant_q, meta=meta)  # fmt: skip


def check_arms() -> None:
    bmm = scales_of(DCP.default_softmax_scale())
    # the scale the split read takes from the bmm1 scale tensor (MlaSplitRead._layer_scales)
    softmax_scale = float(bmm[0][1].item()) / LOG2_E
    for i, rows_per_head in enumerate(CASES):
        x = case_inputs(rows_per_head, 300 + i)
        module = stand_in(0, x.codes, x.scales, x.w_uv)
        R.late(random.Random(i).randrange(R.world))
        off = run_arm(0, module, x.quant_q, x.meta, x.ag, bmm)
        R.late(random.Random(50 + i).randrange(R.world))
        ref = ref_chain(module, x.quant_q, x.meta, x.ag, softmax_scale)
        assert torch.equal(off, ref), (
            f"{rows_per_head} rows: off arm differs from the chain it replaced"
        )
        R.late(random.Random(100 + i).randrange(R.world))
        on = run_arm(1, module, x.quant_q, x.meta, x.ag, bmm)
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
            mla_v_up,
        )

        sent = mla_v_up.proto_send_count
        R.late(random.Random(150 + i).randrange(R.world))
        proto = run_arm("proto", module, x.quant_q, x.meta, x.ag, bmm)
        assert mla_v_up.proto_send_count == sent + 1, (
            f"{rows_per_head} rows: the prototype send did not engage"
        )
        assert torch.equal(proto, on), (
            f"{rows_per_head} rows: prototype send differs from the on arm"
        )
        again = run_arm(1, module, x.quant_q, x.meta, x.ag, bmm)
        assert torch.equal(again, on), f"{rows_per_head} rows: on arm after the prototype differs"
        if R.rank == 0:
            print(
                f"[rank 0] {rows_per_head} rows: prototype a2a send = on arm bit for bit",
                flush=True,
            )
        # the gate read from ag in place (MLA_GATE_SIDE=0) gives the same bits in both arms
        for arm, want in ((0, off), (1, on)):
            got = run_arm(arm, module, x.quant_q, x.meta, x.ag, bmm, gate_side=False)
            assert torch.equal(got, want), (
                f"{rows_per_head} rows, arm {arm}: gate from ag differs from the side gate"
            )
        assert on.shape == off.shape and on.dtype == off.dtype and torch.isfinite(on).all()
        # the gate in the combine against the ungated exchange of the same projected partials and the gate kernel
        ungated = ref_on_chain(module, x.quant_q, x.meta, x.ag, softmax_scale, x.w_uv)
        gate_err = per_head_err(on, ungated).max().item()
        assert gate_err <= GATE_TOL, (
            f"{rows_per_head} rows: gate in the combine off k3_mla_gate by {gate_err:.3e}"
        )
        # the NCCL exchange (TRTLLM_K3LL_HELIX_A2A_FUSED=0) of the same partials, then k3_mla_gate
        FLAGS.TRTLLM_K3LL_HELIX_A2A_FUSED = 0
        try:
            nccl = run_arm(1, module, x.quant_q, x.meta, x.ag, bmm)
        finally:
            FLAGS.TRTLLM_K3LL_HELIX_A2A_FUSED = 1
        nccl_err = per_head_err(on, nccl).max().item()
        assert nccl_err <= GATE_TOL, (
            f"{rows_per_head} rows: NCCL exchange off the fused one by {nccl_err:.3e}"
        )
        on_off = per_head_err(on, off).max().item()
        assert on_off <= ON_OFF_TOL, f"{rows_per_head} rows: on arm off the off arm by {on_off:.3e}"
        exact = fp64_reference(x.pool, x.rows, x.q, x.lens, x.codes, x.scales, x.ag, softmax_scale)
        e_on, e_off = per_head_err(on, exact), per_head_err(off, exact)
        STATS["on"].append(e_on.flatten())
        STATS["off"].append(e_off.flatten())
        assert e_on.max().item() <= FP64_TOL, (
            f"{rows_per_head} rows: on arm off fp64 by {e_on.max().item():.3e} per (token, head)"
        )
        if R.rank == 0:
            print(f"[rank 0] {rows_per_head} rows: gated combine vs exchange + k3_mla_gate {gate_err:.2e}, vs NCCL "
                  f"{nccl_err:.2e}; on vs off max {on_off:.3e}; vs fp64 on median "
                  f"{e_on.median():.2e} max {e_on.max():.2e}, off median {e_off.median():.2e} max "
                  f"{e_off.max():.2e}", flush=True)  # fmt: skip


def check_graph() -> None:
    bmm = scales_of(DCP.default_softmax_scale())
    for k, rows_per_head in enumerate(CASES):
        x = case_inputs(rows_per_head, 500 + k)
        modules = [stand_in(layer, x.codes, x.scales, x.w_uv) for layer in range(LAYERS)]
        q_in, ag_in = x.quant_q.clone(), x.ag.clone()
        for on in (0, 1, "proto"):

            def step():
                return [run_arm(on, m, q_in, x.meta, ag_in, bmm) for m in modules]

            R.barrier()
            step()  # warm: every variant compiled before capture
            R.barrier()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    static = step()
            torch.cuda.current_stream().wait_stream(stream)
            R.barrier()
            for replay in range(REPLAYS):
                y = case_inputs(rows_per_head, 700 + 10 * k + replay)
                q_in.copy_(y.quant_q)
                ag_in.copy_(y.ag)
                R.late(random.Random(900 + replay).randrange(R.world))
                graph.replay()
                torch.cuda.synchronize()
                got = [t.clone() for t in static]
                R.barrier()
                want = step()
                torch.cuda.synchronize()
                for layer in range(LAYERS):
                    assert torch.equal(got[layer], want[layer]), (
                        f"{rows_per_head} rows, arm {on}, replay {replay}, layer {layer}: replay differs from eager"
                    )
                # an eager call of another row count between replays
                other = list(CASES)[(k + 1 + replay) % len(CASES)]
                z = case_inputs(other, 800 + replay)
                run_arm(on, stand_in(0, z.codes, z.scales, z.w_uv), z.quant_q, z.meta, z.ag, bmm)
                R.barrier()
            del graph
    FLAGS.TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A = 0


def check_unsupported_gpu() -> None:
    """The switch at 1 on a GPU it does not build on (``HELIX_SWITCH_CAPABILITIES`` without this GPU's capability):
    ``set_helix_v_up`` keeps no W_UV and the on arm is bit for bit the off arm. On this GPU's capability it keeps
    bit for bit the W_UV the other checks use."""
    bmm = scales_of(DCP.default_softmax_scale())
    saved = IMPL.HELIX_SWITCH_CAPABILITIES
    try:
        for i, rows_per_head in enumerate(CASES):
            x = case_inputs(rows_per_head, 1100 + i)
            module = stand_in(0, x.codes, x.scales, None)
            module.mapping = SimpleNamespace(has_cp_helix=lambda: True)
            module.num_heads_tp = HP * R.world
            IMPL.HELIX_SWITCH_CAPABILITIES = (torch.cuda.get_device_capability(),)
            assert IMPL.K3DecodeMLA.set_helix_v_up(module, x.codes, x.scales) is None
            assert torch.equal(module._helix_v_up_weight, x.w_uv), (
                "W_UV kept differs from dequantize_v_b's"
            )
            module._helix_v_up_weight = None
            IMPL.HELIX_SWITCH_CAPABILITIES = ()
            why = IMPL.K3DecodeMLA.set_helix_v_up(module, x.codes, x.scales)
            assert why is not None and module._helix_v_up_weight is None, why
            R.late(random.Random(1200 + i).randrange(R.world))
            off = run_arm(0, module, x.quant_q, x.meta, x.ag, bmm)
            R.late(random.Random(1300 + i).randrange(R.world))
            on = run_arm(1, module, x.quant_q, x.meta, x.ag, bmm)
            assert torch.equal(on, off), (
                f"{rows_per_head} rows: switch at 1 on an unsupported GPU differs from off"
            )
            if R.rank == 0:
                print(
                    f"[rank 0] {rows_per_head} rows: unsupported GPU ({why}): no W_UV, on == off",
                    flush=True,
                )
    finally:
        IMPL.HELIX_SWITCH_CAPABILITIES = saved
        FLAGS.TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A = 0


def rank_main(args) -> int:
    global R, DCP, IMPL, FLAGS, SR, WS, TP_MAPPING
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
        MnnvlWorkspace,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
        flags,
        implementation,
    )
    from tensorrt_llm.mapping import CpType, Mapping

    flags.TRTLLM_K3LL_HELIX_Q_AG_MNNVL = 1
    flags.TRTLLM_K3LL_HELIX_A2A_FUSED = 1
    DCP, IMPL, FLAGS = decode_mla_dcp, implementation, flags
    per_node = torch.cuda.device_count()
    cp_mapping = Mapping(world_size=R.world, rank=R.rank, gpus_per_node=per_node, tp_size=1, cp_size=R.world,
                         cp_config={"cp_type": CpType.HELIX})  # fmt: skip
    TP_MAPPING = cp_mapping.repurpose_helix_cp_to_tp()
    WS = MnnvlWorkspace.create(TP_MAPPING, WORKSPACE_BYTES, fabric_handle=R.fabric)
    SR = decode_mla_dcp.MlaSplitRead(cp_mapping, HP, 1, gather_mapping=TP_MAPPING)
    for layer in range(LAYERS):
        SR.register(layer)
    built = SR.build_helix_comm(WS, TP_MAPPING)
    assert SR._q_gather_ws is WS and SR._a2a_fused is not None, built
    code = ls.run_checks(R, [check_arms, check_graph, check_unsupported_gpu])
    if R.rank == 0:
        on, off = torch.cat(STATS["on"]), torch.cat(STATS["off"])
        print(f"[rank 0] W={R.world}: per (token, head) vs fp64 over {on.numel()} rows: V-up before the exchange "
              f"median {on.median():.3e} p99 {on.quantile(0.99):.3e} max {on.max():.3e}; FP8 V-B after it median "
              f"{off.median():.3e} p99 {off.quantile(0.99):.3e} max {off.max():.3e}", flush=True)  # fmt: skip
    return code


def main(argv) -> int:
    args = ls.parse_args(argv)
    if args.launcher == "mpirun" and not args.rank_worker:
        ls.spawn(__file__, args, DEADLINE_S)
        return 0
    return rank_main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

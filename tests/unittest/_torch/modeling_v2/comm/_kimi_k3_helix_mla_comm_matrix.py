# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU check of the Kimi K3 staircase's Helix MLA decode exchange on MNNVL (``decode_mla_dcp.py``).

Under Helix the split read all-gathers the FP8 absorbed queries over the CP group and, after attending with every
head over this rank's pages, sends each head group's partials to its owner and combines them. With
``TRTLLM_K3LL_HELIX_Q_AG_MNNVL`` the all-gather is ``gather_q_mnnvl`` (``comm/mnnvl_allgather_split`` on the decode
collectives' ``MnnvlWorkspace``, the bytes carried as fp32 words); with ``TRTLLM_K3LL_HELIX_A2A_FUSED`` the exchange
and combine are ``exchange_combine_fused`` (one ``helix_a2a_fused`` launch, each 512-wide latent head as two 256-wide
halves with the same statistics). This drives both through ``MlaSplitRead.build_helix_comm`` on a W-rank Helix
mapping (tp 1 x cp W) and checks:

* the all-gather: bit for bit the ranks' query bytes in rank order (``allgather`` along the head dim), with the one
  documented rewrite: a 4-byte group equal to the fp32 ``-0.0`` word arrives as zeros (one FP8 ``-0.0`` -> ``+0.0``);
  the rows hold such groups and FP8 NaN bytes on purpose;
* the fused exchange + combine against the production NCCL path's combine (``sanitize_empty`` +
  ``trtllm::helix_post_process`` on the same partials, every rank holding every rank's data from one seed) and against
  an fp64 reference, with tokens empty on some ranks (the combine's identity) and tokens seen by one rank only;
* a decode step's call sequence (gather, combine per layer) captured in a CUDA graph and replayed with new inputs,
  eager calls of other token counts between replays, a random rank late before each;
* ``TRTLLM_K3LL_HELIX_STATS_PAIRS``: the fused kernel reading each head's statistics once for both halves
  (``pair_stats``) gives the bits of the call on the per-half copy.

    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _kimi_k3_helix_mla_comm_matrix.py --launcher srun --world-size 16
    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_helix_mla_comm_matrix.py            # W = 4, mpirun

Not a pytest module (one W-rank job).
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

DEADLINE_S = 900
HP = 6  # heads per rank: Kimi K3's 96 heads over 16 ranks
LATENT = 512
Q_BYTES = 576
TOKENS = (8, 16, 32, 64, 1, 7, 24, 8, 2, 4)
LAYERS = 6
REPLAYS = 4
WORKSPACE_BYTES = 4 << 20  # decode_comm.MNNVL_BUFFER_BYTES
INT32_MIN = torch.iinfo(torch.int32).min
STATS = {"q_neg_zero_words": 0, "fused_vs_stock_max": 0.0, "fused_vs_stock_bitwise": [0, 0],
         "fused_err": 0.0, "stock_err": 0.0, "pair_stats_bitwise": [0, 0]}  # fmt: skip

R = None
MOD = None  # decode_mla_dcp
SR = None  # MlaSplitRead under test


def all_ranks_q(seed: int, tokens: int) -> torch.Tensor:
    """Every rank's query bytes ``[W, T, HP, 576]`` uint8 from one seed: random bytes, FP8 NaN bytes, and 4-byte
    groups equal to the fp32 ``-0.0`` word (bytes 0, 0, 0, 0x80) at random places."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, 256, (R.world, tokens, HP, Q_BYTES), generator=gen, device="cuda",
                      dtype=torch.int32).to(torch.uint8)  # fmt: skip
    words = q.view(torch.int32)
    neg_zero = torch.rand(words.shape, generator=gen, device="cuda") < 0.02
    words[neg_zero] = INT32_MIN
    nan_bytes = torch.rand(q.shape, generator=gen, device="cuda") < 0.01
    q[nan_bytes] = 0x7F
    return q


def q_reference(q_all: torch.Tensor) -> torch.Tensor:
    tokens = q_all.shape[1]
    ref = q_all.permute(1, 0, 2, 3).reshape(tokens, R.world * HP, Q_BYTES).contiguous()
    words = ref.view(torch.int32)
    STATS["q_neg_zero_words"] += int((words == INT32_MIN).sum())
    return torch.where(words == INT32_MIN, torch.zeros_like(words), words).view(torch.uint8)


def all_ranks_partials(seed: int, tokens: int):
    """Every rank's partials: ``out`` ``[W, T, W * HP * 512]`` bf16, ``stats`` ``[W, T, W * HP, 2]`` fp32 ``(max,
    sum)``, ``empty`` ``[W, T]`` bool (rank s sees no key of token t). Every token is seen by at least one rank; token
    0 by rank ``T % W`` only."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = R.world
    out = torch.randn((w, tokens, w * HP * LATENT), generator=gen, device="cuda").bfloat16()
    mx = torch.randn((w, tokens, w * HP), generator=gen, device="cuda") * 4.0
    sm = torch.rand((w, tokens, w * HP), generator=gen, device="cuda") * 8.0 + 0.5
    stats = torch.stack((mx, sm), dim=-1).contiguous()
    empty = torch.rand((w, tokens), generator=gen, device="cuda") < 0.3
    empty[0, empty.all(dim=0)] = False
    empty[:, 0] = True
    empty[tokens % w, 0] = False
    # garbage in the empty rows: the sender must replace it
    out[empty] = float("nan")
    return out, stats, empty


def combine_references(out, stats, empty):
    """This rank's head group: the production NCCL path's combine (``sanitize_empty`` + ``helix_post_process`` on
    the gathered chunks) and an fp64 reference."""
    r, w = R.rank, R.world
    tokens = out.shape[1]
    o = out[:, :, r * HP * LATENT : (r + 1) * HP * LATENT].clone()  # [W, T, HP * 512]
    s = stats[:, :, r * HP : (r + 1) * HP].clone()  # [W, T, HP, 2]
    for src in range(w):
        MOD.sanitize_empty(o[src].reshape(tokens, HP, LATENT), s[src], empty[src])
    stock = torch.ops.trtllm.helix_post_process(o.contiguous(), s.contiguous(), 1.0)
    m = s[..., 0].double()
    gmax = m.max(dim=0).values
    weight = s[..., 1].double() * torch.exp(m - gmax)  # [W, T, HP]
    weight = weight / weight.sum(dim=0)
    exact = (
        (o.view(w, tokens, HP, LATENT).double() * weight[..., None])
        .sum(dim=0)
        .view(tokens, HP * LATENT)
    )
    return stock, exact


def check_build() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    assert SR._q_gather_ws is None and SR._a2a_fused is None
    built = SR.build_helix_comm(WS, TP_MAPPING)
    assert SR._q_gather_ws is WS and SR._a2a_fused is not None, built
    assert flags.TRTLLM_K3LL_HELIX_Q_AG_MNNVL and flags.TRTLLM_K3LL_HELIX_A2A_FUSED
    if R.rank == 0:
        print(f"[rank 0] build_helix_comm: {built}", flush=True)


def check_q_gather() -> None:
    for i, tokens in enumerate(TOKENS):
        q_all = all_ranks_q(100 + i, tokens)
        R.late(random.Random(i).randrange(R.world))
        got = MOD.gather_q_mnnvl(q_all[R.rank].contiguous(), WS)
        want = q_reference(q_all)
        assert got.shape == want.shape and torch.equal(got, want), (
            f"q all-gather differs at T={tokens}"
        )
    assert R.all_true(STATS["q_neg_zero_words"] > 0), "the -0.0 word rule was not exercised"


def run_fused(out, stats, empty):
    return MOD.exchange_combine_fused(out[R.rank], stats[R.rank], empty[R.rank], SR._a2a_fused, HP)


def compare_combine(got, out, stats, empty, what: str) -> None:
    stock, exact = combine_references(out, stats, empty)
    assert torch.isfinite(got).all(), f"{what}: non-finite combine"
    scale = exact.abs().max().item()
    diff = (got.float() - stock.float()).abs().max().item()
    fused_err = (got.double() - exact).abs().max().item() / scale
    stock_err = (stock.double() - exact).abs().max().item() / scale
    STATS["fused_vs_stock_max"] = max(STATS["fused_vs_stock_max"], diff / scale)
    STATS["fused_vs_stock_bitwise"][0] += int(torch.equal(got, stock))
    STATS["fused_vs_stock_bitwise"][1] += 1
    STATS["fused_err"] = max(STATS["fused_err"], fused_err)
    STATS["stock_err"] = max(STATS["stock_err"], stock_err)
    # bf16 rounding of the result: one ulp at the largest magnitude is 2^-8 relative
    assert fused_err <= 2.0**-7, (
        f"{what}: fused combine off the fp64 reference by {fused_err:.3e} (relative)"
    )
    assert diff / scale <= 2.0**-7, (
        f"{what}: fused combine off the NCCL path's by {diff / scale:.3e} (relative)"
    )


def check_fused_combine() -> None:
    for i, tokens in enumerate(TOKENS):
        out, stats, empty = all_ranks_partials(200 + i, tokens)
        R.late(random.Random(1000 + i).randrange(R.world))
        got = run_fused(out, stats, empty)
        assert got.shape == (tokens, HP * LATENT) and got.dtype == torch.bfloat16
        compare_combine(got, out, stats, empty, f"T={tokens}")


def check_pair_stats() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    try:
        for i, tokens in enumerate(TOKENS):
            out, stats, empty = all_ranks_partials(300 + i, tokens)
            flags.TRTLLM_K3LL_HELIX_STATS_PAIRS = 0
            R.late(random.Random(3000 + i).randrange(R.world))
            want = run_fused(out, stats, empty)
            flags.TRTLLM_K3LL_HELIX_STATS_PAIRS = 1
            R.late(random.Random(4000 + i).randrange(R.world))
            got = run_fused(out, stats, empty)
            torch.cuda.synchronize()
            same = torch.equal(got.view(torch.int16), want.view(torch.int16))
            STATS["pair_stats_bitwise"][0] += int(same)
            STATS["pair_stats_bitwise"][1] += 1
            assert same, f"pair_stats differs at T={tokens}"
            compare_combine(got, out, stats, empty, f"pair_stats T={tokens}")
    finally:
        flags.TRTLLM_K3LL_HELIX_STATS_PAIRS = 0


def check_graph_sequence() -> None:
    """One captured decode step: per layer the query all-gather then the fused exchange + combine, both on their
    workspaces; replayed with new inputs, an eager call of another token count between replays."""
    tokens = 16
    q_in = torch.empty((tokens, HP, Q_BYTES), dtype=torch.uint8, device="cuda")
    o_in = [
        torch.empty((tokens, R.world * HP * LATENT), dtype=torch.bfloat16, device="cuda")
        for _ in range(LAYERS)
    ]
    s_in = [
        torch.empty((tokens, R.world * HP, 2), dtype=torch.float32, device="cuda")
        for _ in range(LAYERS)
    ]
    e_in = [torch.empty((tokens,), dtype=torch.bool, device="cuda") for _ in range(LAYERS)]

    def step():
        outs = []
        for layer in range(LAYERS):
            outs.append(MOD.gather_q_mnnvl(q_in, WS))
            outs.append(
                MOD.exchange_combine_fused(o_in[layer], s_in[layer], e_in[layer], SR._a2a_fused, HP)
            )
        return outs

    def fill(seed):
        q_all = all_ranks_q(seed, tokens)
        q_in.copy_(q_all[R.rank])
        parts = []
        for layer in range(LAYERS):
            out, stats, empty = all_ranks_partials(seed * 100 + layer, tokens)
            o_in[layer].copy_(out[R.rank])
            s_in[layer].copy_(stats[R.rank])
            e_in[layer].copy_(empty[R.rank])
            parts.append((out, stats, empty))
        return q_reference(q_all), parts

    fill(1)
    R.barrier()
    step()  # warm (compiles nothing new: every shape ran eagerly above)
    R.barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            static_outs = step()
    torch.cuda.current_stream().wait_stream(stream)
    R.barrier()
    for replay in range(REPLAYS):
        q_want, parts = fill(10 + replay)
        R.late(random.Random(2000 + replay).randrange(R.world))
        graph.replay()
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            assert torch.equal(static_outs[2 * layer], q_want), (
                f"replay {replay} layer {layer}: q all-gather"
            )
            compare_combine(
                static_outs[2 * layer + 1], *parts[layer], f"replay {replay} layer {layer}"
            )
        # an eager pair of another token count on the same workspaces between replays
        eager_tokens = (1, 8, 64, 3)[replay]
        q_all = all_ranks_q(50 + replay, eager_tokens)
        assert torch.equal(MOD.gather_q_mnnvl(q_all[R.rank].contiguous(), WS), q_reference(q_all))
        out, stats, empty = all_ranks_partials(60 + replay, eager_tokens)
        compare_combine(run_fused(out, stats, empty), out, stats, empty, f"eager T={eager_tokens}")
        R.barrier()


def rank_main(args) -> int:
    global R, MOD, SR, WS, TP_MAPPING
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
        MnnvlWorkspace,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
        flags,
    )
    from tensorrt_llm.mapping import CpType, Mapping

    # the load-time values build_helix_comm reads (as the model's launch with both switches at 1)
    flags.TRTLLM_K3LL_HELIX_Q_AG_MNNVL = 1
    flags.TRTLLM_K3LL_HELIX_A2A_FUSED = 1

    MOD = decode_mla_dcp
    per_node = torch.cuda.device_count()
    cp_mapping = Mapping(world_size=R.world, rank=R.rank, gpus_per_node=per_node, tp_size=1, cp_size=R.world,
                         cp_config={"cp_type": CpType.HELIX})  # fmt: skip
    TP_MAPPING = cp_mapping.repurpose_helix_cp_to_tp()
    assert TP_MAPPING.tp_rank == cp_mapping.cp_rank and list(TP_MAPPING.tp_group) == list(
        cp_mapping.cp_group
    )
    WS = MnnvlWorkspace.create(TP_MAPPING, WORKSPACE_BYTES, fabric_handle=R.fabric)
    SR = decode_mla_dcp.MlaSplitRead(cp_mapping, HP, 1, gather_mapping=TP_MAPPING)
    code = ls.run_checks(
        R,
        [check_build, check_q_gather, check_fused_combine, check_graph_sequence, check_pair_stats],
    )
    if R.rank == 0:
        hit, total = STATS["fused_vs_stock_bitwise"]
        print(
            f"[rank 0] W={R.world}: q -0.0 words rewritten {STATS['q_neg_zero_words']}; fused vs NCCL-path combine: "
            f"bitwise {hit}/{total} calls, max |diff| {STATS['fused_vs_stock_max']:.3e} (relative); vs fp64: "
            f"fused {STATS['fused_err']:.3e}, NCCL path {STATS['stock_err']:.3e}; pair_stats vs per-half copy: "
            f"bitwise {STATS['pair_stats_bitwise'][0]}/{STATS['pair_stats_bitwise'][1]}",
            flush=True,
        )
    return code


def main(argv) -> int:
    args = ls.parse_args(argv)
    if args.launcher == "mpirun" and not args.rank_worker:
        ls.spawn(__file__, args, DEADLINE_S)
        return 0
    return rank_main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

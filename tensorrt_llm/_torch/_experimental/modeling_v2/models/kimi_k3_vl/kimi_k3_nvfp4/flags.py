# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switches of the Kimi K3 NVFP4 low-latency decode path.

One ``_flag(...)`` line per switch: environment variable name, default, one sentence. A switch is read once, at
import, from the environment of every rank; ``log_flags`` prints every switch's effective value when the model is
built, so a log always states which variant ran. With every switch at its default the target behaves as before the
switch was added. A switch whose value is a path is a ``_path_flag(...)``: unset, empty or ``0`` is off (so a
harness that sets a switch's off arm to 0 works for it too).
"""

import os
from typing import Dict, Tuple

from tensorrt_llm.logger import logger

_FLAGS: Dict[str, Tuple[int, int, str]] = {}
_PATH_FLAGS: Dict[str, Tuple[str, str]] = {}


def _flag(name: str, default: int, doc: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as err:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from err
    _FLAGS[name] = (value, default, doc)
    return value


def _path_flag(name: str, doc: str) -> str:
    """A switch whose value is a file or directory path; ``""`` (unset, empty or ``0``) is off."""
    value = os.environ.get(name) or ""
    if value == "0":
        value = ""
    _PATH_FLAGS[name] = (value, doc)
    return value


TRTLLM_K3LL_CENSUS = _flag(
    "TRTLLM_K3LL_CENSUS", 0, "count and log generic fallbacks in decode steps"
)
TRTLLM_K3LL_MOE_LOWM = _flag(
    "TRTLLM_K3LL_MOE_LOWM",
    1,
    "MoE decode path: NVFP4-latent front + k3_moe_lowm_dyn on CuteDSL NVFP4 buffers (decode_moe_lowm.py)",
)
TRTLLM_K3LL_MOE_FRONT_SORT = _flag(
    "TRTLLM_K3LL_MOE_FRONT_SORT",
    1,
    "with TRTLLM_K3LL_MOE_LOWM: the routing's moe_sort over the local experts in k3_moe_front, not its own kernel",
)
TRTLLM_K3LL_MOE_SELF_SORT = _flag(
    "TRTLLM_K3LL_MOE_SELF_SORT",
    -1,
    "with TRTLLM_K3LL_MOE_LOWM, steps of at most 16 tokens over at most 256 local experts on K3MoeLowMDynKernel (the "
    "front's steps, and the wide head's k3_route_quant steps): the expert kernel sorts the step's routing over the "
    "local experts itself, in every CTA after its wait (the same tiles and rows, in token order, as k3_moe_front's "
    "sort, in shared memory); the front skips its sort (TRTLLM_K3LL_MOE_FRONT_SORT's last route CTA hand-off), or "
    "moe_sort does not run; the kernel reads the front's bf16 routing weights itself (widened exactly). FC1 rows "
    "bit for bit; each (token, expert) partial bit for bit; a token's sum of partials in FC2's red order, as "
    "without it. Read at forward time; 1 on, 0 off, -1 (default): on on SM103 only "
    "(decode_moe_lowm.MOE_SELF_SORT_CAPABILITIES; the lowm path runs on SM100 / SM103 only; L1 bs1 -0.150 ms, bs2 "
    "-0.223 ms, moss-0444; the token-mask sort, here and in FC1 / FC2, vs the earlier scan sort: L1 1x -0.049 ms, "
    "2x -0.061 ms, 4x -0.142 ms, moss-0471)",
)
TRTLLM_K3LL_MOE_SELF_SORT_FC12 = _flag(
    "TRTLLM_K3LL_MOE_SELF_SORT_FC12",
    -1,
    "with TRTLLM_K3LL_MOE_LOWM_FC12, wide steps of 17-32 tokens over at most 256 local experts (the front's, and the "
    "wide head's): K3MoeFc1LowMKernel (after its wait) and K3MoeFc2LowMKernel (at its start) each build the tile "
    "metadata of their own CTA's tiles from the step's top-k ids (the same tiles and rows, in token order, as "
    "k3_moe_front's sort, in shared memory), so the front skips its sort (or moe_sort does not run) and FC2 reads "
    "the front's bf16 routing weights (widened exactly). FC1 rows and each (token, expert) partial bit for bit; a "
    "token's sum in FC2's red order, as without it. Read at forward time; 1 on, 0 off, -1 (default): on on SM103 "
    "only (decode_moe_lowm.MOE_SELF_SORT_CAPABILITIES; L1 bs4 -0.102 ms, moss-0452; the token-mask sort, here and in "
    "K3MoeLowMDynKernel, vs the earlier scan sort: L1 1x -0.049 ms, 2x -0.061 ms, 4x -0.142 ms, moss-0471)",
)
TRTLLM_K3LL_MLA_DCP = _flag(
    "TRTLLM_K3LL_MLA_DCP",
    1,
    "FP8-cache MLA decode attention: 0 stock, 1 CuTe DSL kernel on this rank's heads, 2 split-read DCP over TP",
)
TRTLLM_K3LL_DRAFTER_FC_TP = _flag(
    "TRTLLM_K3LL_DRAFTER_FC_TP",
    1,
    "MLA DSpark drafter: fc split by input feature over TP, hidden_norm in the MNNVL all-reduce",
)
TRTLLM_K3LL_MLA_SPEC_WORKER = _flag(
    "TRTLLM_K3LL_MLA_SPEC_WORKER",
    1,
    "MLA DSpark drafter: stock worker plus sharded draft logits and the k3_markov chain (KimiK3MLADSparkWorker)",
)
TRTLLM_K3LL_MLA_DRAFTER_NORMS = _flag(
    "TRTLLM_K3LL_MLA_DRAFTER_NORMS",
    1,
    "MLA DSpark drafter: residual adds and RMSNorms in the o_proj / down all-reduces (k3_sandwich_plain)",
)
TRTLLM_K3LL_MLA_DRAFTER_CTX = _flag(
    "TRTLLM_K3LL_MLA_DRAFTER_CTX",
    1,
    "MLA DSpark drafter (with TRTLLM_K3LL_MLA_SPEC_WORKER): context latents of all layers in one k3_drafter_mla_ctx_kv",
)
TRTLLM_K3LL_MLA_SPEC_ACCEPT = _flag(
    "TRTLLM_K3LL_MLA_SPEC_ACCEPT",
    1,
    "MLA DSpark worker: vocab-sharded target logits + k3_spec_accept (greedy, or temperature via no_topk_no_topp)",
)

TRTLLM_K3LL_MLA_DRAFTER_QKV = _flag(
    "TRTLLM_K3LL_MLA_DRAFTER_QKV",
    1,
    "MLA DSpark drafter (with TRTLLM_K3LL_MLA_DRAFTER_NORMS): [q_a | kv_a] GEMV + k3_drafter_mla_a_post per layer",
)


TRTLLM_K3LL_MLA_QKV_SPLIT = _flag(
    "TRTLLM_K3LL_MLA_QKV_SPLIT",
    1,
    "target MLA decode: q_a prep, Q-B in 32-column slices and the absorption as three launches, not one per head",
)
TRTLLM_K3LL_MLA_QKV_PDL = _flag(
    "TRTLLM_K3LL_MLA_QKV_PDL",
    0,
    "with MLA_QKV_SPLIT, target MLA decode on a quantized cache: its three query kernels launched with PDL, each "
    "loading its first weights before the dependency wait",
)
TRTLLM_K3LL_DRAFTER_ATTN_CUTEDSL = _flag(
    "TRTLLM_K3LL_DRAFTER_ATTN_CUTEDSL",
    1,
    "MLA DSpark drafter: block decode on its cute-dsl implementation when speculative_config leaves it at AUTO",
)
TRTLLM_K3LL_DRAFTER_MARKOV_V2 = _flag(
    "TRTLLM_K3LL_DRAFTER_MARKOV_V2",
    1,
    "MLA DSpark worker (with TRTLLM_K3LL_MLA_SPEC_WORKER): trtllm::k3_markov's MMA build, every request's bias in "
    "one MMA per position and a cluster-reduced exchange (draft tokens only: another bias summation order)",
)
TRTLLM_K3LL_DRAFTER_GEMM = _flag(
    "TRTLLM_K3LL_DRAFTER_GEMM",
    1,
    "MLA DSpark drafter (with TRTLLM_K3LL_MLA_DRAFTER_NORMS): [q_a | kv_a] and gate_up above one token tile and q_b "
    "on the low-M split-K GEMM at pinned tactics, not cuBLAS (draft tokens only: split-K partial sums)",
)
TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU = _flag(
    "TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU",
    -1,
    "MLA DSpark drafter (with TRTLLM_K3LL_DRAFTER_GEMM): where gate_up runs on its pinned low-M GEMM, the SwiGLU in "
    "that GEMM's epilogue (low_m_bf16_splitk.run_splitk_dense_swiglu, the weight read through a gate / up interleaving "
    "view, nothing copied) instead of a separate silu_and_mul launch; bit for bit the two launches. -1 (default): on "
    "on SM103 only (mla_drafter_gemm.GATE_UP_SWIGLU_CAPABILITIES, where the gate_up tactics are pinned); read per "
    "call",
)
TRTLLM_K3LL_DRAFTER_WIDE_AR = _flag(
    "TRTLLM_K3LL_DRAFTER_WIDE_AR",
    1,
    "MLA DSpark drafter: its fused all-reduce + RMSNorm calls above one token tile two-shot above the wide decode "
    "steps' 1 MiB one-shot ceiling, not the decode path's 4 MiB (draft tokens only: another reduction order)",
)
TRTLLM_K3LL_DRAFTER_PDL = _flag(
    "TRTLLM_K3LL_DRAFTER_PDL",
    1,
    "MLA DSpark drafter: k3_drafter_mla_a_post, the query RoPE (cute_dsl_dspark_rope_into), the bf16 block decode "
    "(split + reduction, k3_mla_decode_fp16_pdl) and k3_drafter_mla_ctx_kv launched with programmatic dependent "
    "launch, norm weights loaded before the dependency wait (the same bits; read per call)",
)
TRTLLM_K3LL_DRAFTER_GLUE = _flag(
    "TRTLLM_K3LL_DRAFTER_GLUE",
    1,
    "MLA DSpark drafter and worker: fewer glue launches (no second KDA replay record after k3_spec_accept, cached "
    "arange tensors and block mask, kv_bounds without repeat_interleave, the V up-projection's bmm into the o_proj "
    "input layout); the same values (read per call)",
)
TRTLLM_K3LL_DRAFT_GATHER_TABLE = _flag(
    "TRTLLM_K3LL_DRAFT_GATHER_TABLE",
    1,
    "MLA DSpark worker: the draft gather's block-row ids as a view of a slot table built outside capture "
    "(`_draft_gather_ids`) instead of the stock per-step arange / add / clamp kernels; the same ids (read per call)",
)
TRTLLM_K3LL_DRAFTER_PREP_FUSED = _flag(
    "TRTLLM_K3LL_DRAFTER_PREP_FUSED",
    1,
    "MLA DSpark worker (with MLA_SPEC_ACCEPT and MLA_DRAFTER_CTX), on a k3_spec_accept step: the drafter's inputs in "
    "one k3_drafter_input_prep launch (bonus, positions, context-store targets and mask, ctx_len, seq_lens, the "
    "block fixup, pool slots and query RoPE rows, the noise rows' lookup), the noise block in one copy kernel, the "
    "store mask in k3_drafter_mla_ctx_kv (the same values; read per step)",
)
TRTLLM_K3LL_DRAFTER_PREP_SPLIT = _flag(
    "TRTLLM_K3LL_DRAFTER_PREP_SPLIT",
    1,
    "MLA DSpark worker (with TRTLLM_K3LL_DRAFTER_PREP_FUSED): k3_drafter_input_prep runs its bookkeeping in a program "
    "of its own instead of after request 0's embedding-row copy, and that kernel and k3_noise_block copy each hidden "
    "row in one tile instead of 1024-element tiles (the same values; read per call)",
)
TRTLLM_K3LL_DRAFTER_CTX_SIDE = _flag(
    "TRTLLM_K3LL_DRAFTER_CTX_SIDE",
    1,
    "with TRTLLM_K3LL_DRAFTER_PREP_FUSED, under multi-stream (CUDA-graph warmup and capture): the context projection "
    "(fc and its all-reduce + hidden_norm) and the latent GEMM on a side stream forked before the target LM head, "
    "joined before the context store (the same values; read per step)",
)
TRTLLM_K3LL_KDA_BFA_OVERLAP = _flag(
    "TRTLLM_K3LL_KDA_BFA_OVERLAP",
    1,
    "KDA decode / verify: the bf16 [f_a | b] projection on the projection stream beside the FP8 QKVG projection",
)
TRTLLM_K3LL_KDA_BFA_SIDE_FIRST = _flag(
    "TRTLLM_K3LL_KDA_BFA_SIDE_FIRST",
    0,
    "with TRTLLM_K3LL_KDA_BFA_OVERLAP: the [f_a | b] projection enqueued on the projection stream before the FP8 "
    "QKVG projection is enqueued, not after it",
)
TRTLLM_K3LL_KDA_BFA_ROW_TILES = _flag(
    "TRTLLM_K3LL_KDA_BFA_ROW_TILES",
    0,
    "with TRTLLM_K3LL_KDA_BFA_GEMV: the [f_a | b] projection on decode_gemv site kda_bfa_rt (k3_ctm_gemv_wide over "
    "equal row tiles, more CTAs, the same bits); the site is built at load when this is 1",
)
TRTLLM_K3LL_KDA_BFA_FP8 = _flag(
    "TRTLLM_K3LL_KDA_BFA_FP8",
    1,
    "KDA decode / verify: [q | k | v | g | f_a | b] in one k3_fp8w_lowm_gemm, the bf16 [f_a | b] weight quantized "
    "to FP8 128x128 blocks and appended to the QKVG weight (changes numerics); the fused weight is built at load "
    "(the target's post_load_weights) when 1; launched at 0, switching it on later keeps the unfused path",
)
TRTLLM_K3LL_FP8W_LOWM = _flag(
    "TRTLLM_K3LL_FP8W_LOWM",
    1,
    "FP8 attention projections in DeepGEMM layout (weight reads) and on k3_fp8w_lowm_gemm at decode size",
)
TRTLLM_K3LL_FP8W_LOWM_FUSED_A = _flag(
    "TRTLLM_K3LL_FP8W_LOWM_FUSED_A",
    1,
    "with FP8W_LOWM, also the weight reads whose N is not a multiple of 128 (MLA fused q_a/kv_a, partial row tile); "
    "read at forward time, the routing is installed at load when the launch value is 1",
)

TRTLLM_K3LL_MOE_FRONT16 = _flag(
    "TRTLLM_K3LL_MOE_FRONT16",
    1,
    "lowm MoE on 9-16 token steps: the head, routing, NVFP4 latent and shared gate_up in k3_moe_front's 16-token build",
)
TRTLLM_K3LL_MOE_FRONT32 = _flag(
    "TRTLLM_K3LL_MOE_FRONT32",
    1,
    "lowm MoE on 17-32 token steps: the head, routing, NVFP4 latent and shared gate_up in k3_moe_front's 32-token "
    "build",
)
TRTLLM_K3LL_KDA_BFA_GEMV = _flag(
    "TRTLLM_K3LL_KDA_BFA_GEMV",
    1,
    "KDA decode / verify: the bf16 [f_a | b] projection on k3_ctm_gemv_wide (decode_gemv site kda_bfa), and the "
    "verify kernel reading it beside the FP8 QKVG rows instead of a concatenated copy",
)


TRTLLM_K3LL_WIDE_AR_ATTN_RES = _flag(
    "TRTLLM_K3LL_WIDE_AR_ATTN_RES",
    1,
    "wide decode steps: post-attention / MoE-tail all-reduce + residual add + attn_res + RMSNorm in one two-shot call",
)
TRTLLM_K3LL_WIDE_LATENT_TAIL = _flag(
    "TRTLLM_K3LL_WIDE_LATENT_TAIL",
    1,
    "wide decode steps: MoE latent all-reduce + latent RMSNorm + the tail GEMV's input concat in one MNNVL call",
)

TRTLLM_K3LL_OPROJ_PUSH = _flag(
    "TRTLLM_K3LL_OPROJ_PUSH",
    1,
    "decode steps up to 64 tokens (1; 2: wide steps only): o_proj's GEMM pushes its partial into the post-attention "
    "all-reduce, K3LamportFinish + K3AttnResFast run the rest (oproj_push.py; read at forward time)",
)
TRTLLM_K3LL_MOE_TAIL_PUSH = _flag(
    "TRTLLM_K3LL_MOE_TAIL_PUSH",
    1,
    "wide decode steps whose MoE tail runs on k3_ctm_gemv_wide (9-32 tokens): the tail GEMV pushes its partial into "
    "the next pre-attention step's (or final norm's) all-reduce, K3LamportFinish + K3AttnResFast run the rest "
    "(moe_tail_push.py; bitwise; read at forward time)",
)
TRTLLM_K3LL_MOE_TAIL_LU = _flag(
    "TRTLLM_K3LL_MOE_TAIL_LU",
    0,
    "wide decode steps of 9-32 tokens on the MoE front's 16- / 32-token builds: the shared expert's down projection "
    "and its all-reduce on a high-priority side stream under the routed experts, the full latent all-reduce + "
    "RMSNorm on its own MNNVL workspace (K3LatentArRowKernel, with the latent's MXFP8 form), latent_up replicated on "
    "k3_fp8w_lowm_mxfp8 with an FP8 block-scale weight, and K3AttnResFast adding routed + shared (moe_tail_lu.py). "
    "LOSSY (FP8 latent_up). 1 at launch builds the FP8 latent_up of every MoE layer (+25.9 MB per layer per rank, "
    "~2.4 GB per rank over 92 layers) and a 12 MiB MNNVL workspace; engagement is read at forward time. -1: on on "
    "SM103 only (moe_tail_lu.MOE_TAIL_LU_CAPABILITIES). 0 (default): off; TRTLLM_K3LL_MOE_TAIL_COL takes the same "
    "steps losslessly",
)
TRTLLM_K3LL_MOE_TAIL_LU_MAX_ROWS = _flag(
    "TRTLLM_K3LL_MOE_TAIL_LU_MAX_ROWS",
    16,
    "with TRTLLM_K3LL_MOE_TAIL_LU: the widest step (tokens) it takes, read at forward time; 16 (default) = bs2 at "
    "MTP7, 32 = up to bs4. At bs4 (32 rows) the switch measured no gain (nsys wall +0.012 ms, ABBA unresolved "
    "without one outlier round; the tail K3AttnResFast runs longer there, cause not isolated), so the lossy path "
    "stays off there",
)
TRTLLM_K3LL_MOE_LOWM_FC12 = _flag(
    "TRTLLM_K3LL_MOE_LOWM_FC12",
    -1,
    "lowm MoE on wide decode steps of 17-32 tokens: the routed experts as K3MoeFc1LowMKernel then "
    "K3MoeFc2LowMKernel (one CTA per SM each, FC2 streaming its weights under FC1 by PDL; "
    "nvfp4_lowm_op.k3_moe_nvfp4_lowm_fc12), the output zeroed on a side stream forked before the MoE front, instead "
    "of K3MoeLowMDynKernel; steps of at most 16 tokens keep it. Not bitwise: the FC1 output and every FC2 partial "
    "are the single kernel's bits, only the order of the bf16 red.add of a token's partials differs (hardware order "
    "in both). 1 at launch builds the FC1 output buffer (12.5 MB per rank at K3 TP4 x EP4) and compiles both "
    "kernels; engagement is read at forward time. -1 (default): on on SM103 only "
    "(decode_moe_lowm.MOE_LOWM_FC12_CAPABILITIES; L1 bs4 -0.229 ms, moss-0361)",
)
TRTLLM_K3LL_MOE_TAIL_COL = _flag(
    "TRTLLM_K3LL_MOE_TAIL_COL",
    -1,
    "wide decode steps of 9-32 tokens on the MoE front's 16- / 32-token builds: TRTLLM_K3LL_MOE_TAIL_LU's side-stream "
    "shared all-reduce and full latent all-reduce + RMSNorm, then a column-sharded BF16 latent_up (each rank its "
    "7168 / TP output columns, k3_ctm_gemv_wide) adding the shared output's columns and multicasting its slice into "
    "the next K3AttnResFast's Lamport rows: no FP8, no finish kernel (moe_tail_col.py). Lossless (BF16 weight, fp32 "
    "sums). 1 at launch builds the latent_up row slice of every MoE layer (+3.2 MB per layer per rank) and TAIL_LU's "
    "12 MiB MNNVL workspace; engagement is read at forward time. Takes precedence over TRTLLM_K3LL_MOE_TAIL_LU. -1 "
    "(default): on on SM103 only (moe_tail_col.MOE_TAIL_COL_CAPABILITIES)",
)
TRTLLM_K3LL_LAR_CLEAR_SPLIT = _flag(
    "TRTLLM_K3LL_LAR_CLEAR_SPLIT",
    0,
    "TRTLLM_K3LL_MOE_TAIL_COL's latent all-reduce (K3LatentArRowKernel): the previous call's Lamport buffer is "
    "re-armed by extra CTAs of the same grid after its griddepcontrol.wait (k3_latent_ar_norm.ROW_CLEAR_CTAS) instead "
    "of by the row CTAs between their push and their poll, which is on the last rank's critical path. Bitwise (the "
    "same stores, other CTAs, the same window between the grid's wait and its completion). Read at forward time. "
    "0 (default): off; 1: on",
)
TRTLLM_K3LL_ATTN_RES_SNAPSHOT_INPLACE = _flag(
    "TRTLLM_K3LL_ATTN_RES_SNAPSHOT_INPLACE",
    -1,
    "a snapshot layer (layer_idx % attn_res_block_size == 0) whose pre-attention step is TRTLLM_K3LL_MOE_TAIL_COL's "
    "K3AttnResFast: the kernel stores the updated prefix sum straight into the snapshot bank row, and the bank's "
    "device-to-device copy after the step (a graph memcpy node that also breaks the PDL chain) is skipped. Bitwise "
    "(the same stores, another destination). Acts only on TAIL_COL's fused steps (9-32 rows; not bs1 / bs8) where the "
    "bank's row count is the step's. Read at forward time. -1 (default): on on SM103 only "
    "(implementation.ATTN_RES_SNAPSHOT_INPLACE_CAPABILITIES; L1 bs2 -0.033 ms, moss-0411)",
)
TRTLLM_K3LL_ATTN_RES_TAP = _flag(
    "TRTLLM_K3LL_ATTN_RES_TAP",
    -1,
    "a drafter-tapped layer whose pre-attention step is TRTLLM_K3LL_MOE_TAIL_COL's K3AttnResFast: the kernel also "
    "stores the pre-norm attention-residual mixture into the layer's capture slot (spec_worker.capture_view), and "
    "the separate attn_res for the tap after the step is skipped. The target's outputs are unchanged; the tap is the "
    "fast kernel's mixture (the C++ MNNVL epilogue's arithmetic), within test_k3_attn_res's fast-kernel bound of the "
    "separate attn_res (2^-6 relative + 2^-8 of the row RMS, at most 2% of elements differing; test_tap prints the "
    "measured maximum). Acts only on TAIL_COL's fused steps (9-32 rows) with KIMI_K3_AUX_ATTN_RES_STREAM=1 and a "
    "drafter capture buffer. Read at forward time. 1 on, 0 off, -1 (default): on on SM103 only "
    "(implementation.ATTN_RES_TAP_CAPABILITIES; L1 bs2 -0.059 ms, GSM8K PASS, natural AL +0.014 in band: "
    "moss-0402/0412/0424)",
)

TRTLLM_K3LL_HELIX_Q_AG_MNNVL = _flag(
    "TRTLLM_K3LL_HELIX_Q_AG_MNNVL",
    1,
    "Helix MLA decode: the FP8 absorbed-query all-gather on the decode collectives' MNNVL workspace "
    "(mnnvl_allgather_split), not NCCL",
)
TRTLLM_K3LL_HELIX_A2A_FUSED = _flag(
    "TRTLLM_K3LL_HELIX_A2A_FUSED",
    1,
    "Helix MLA decode: the partials' exchange and combine in one Lamport kernel (helix_a2a_fused), not NCCL "
    "alltoall_helix + helix_post_process",
)
TRTLLM_K3LL_HELIX_Q_ABSORB = _flag(
    "TRTLLM_K3LL_HELIX_Q_ABSORB",
    1,
    "Helix MLA decode (FP8 cache): every rank computes all heads' FP8 queries (q_b through the module's FP8 "
    "projection, k3_fp8w_lowm_gemm at decode size; the BF16 W_UK absorption in K3MlaQAbsorb), no query all-gather; "
    "the BF16 W_UK is built at load when the launch value is 1, on SM103 only (elsewhere the switch-off path)",
)
TRTLLM_K3LL_MLA_GATE_SIDE = _flag(
    "TRTLLM_K3LL_MLA_GATE_SIDE",
    1,
    "MLA decode (quantized cache): the output gate projection on a side stream beside the attention, read in place "
    "by the V-B / gate kernel (no [kv_a | g] concatenation)",
)
TRTLLM_K3LL_MLA_KV_SIDE = _flag(
    "TRTLLM_K3LL_MLA_KV_SIDE",
    1,
    "Helix all-head MLA decode (HELIX_Q_ABSORB): the latent row and the KV append (mla_latent, mla_rope_generation "
    "with q_rope_applied) on their own side stream from right after the fused q_a / kv_a, beside the query path "
    "(q_a norm, q_b, K3MlaQAbsorb); the main stream joins it before the split read, so every later reader of the "
    "KV pages runs after the append (same kernels and bytes; read at forward time). Decode steps of 1 to 4 requests "
    "only (decode_mla_q.KV_SIDE_MAX_REQUESTS); from 5 requests the main stream runs them in order as with 0. SM103 "
    "only: the all-head path needs HELIX_Q_ABSORB's W_UK, built on implementation.HELIX_SWITCH_CAPABILITIES; on "
    "SM107 (Rubin) the decode takes the switch-off path and this switch has no effect",
)
TRTLLM_K3LL_MLA_DECODE_PDL = _flag(
    "TRTLLM_K3LL_MLA_DECODE_PDL",
    1,
    "MLA decode split read: the CuTe DSL FP8 split-KV kernel and its reduction with programmatic dependent launch "
    "(k3_mla_decode_fp8_pdl)",
)
TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A = _flag(
    "TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A",
    1,
    "Helix MLA decode: every head's BF16 V up-projection inside the split-KV reduction, the exchange at 128 values "
    "per head, then the gate (mla_v_up.py); 1 at launch builds the BF16 W_UV of all heads (12.6 MB per MLA layer), "
    "on SM103 only (elsewhere the switch-off path)",
)
TRTLLM_K3LL_MLA_KV_JOIN_EARLY = _flag(
    "TRTLLM_K3LL_MLA_KV_JOIN_EARLY",
    -1,
    "with TRTLLM_K3LL_MLA_KV_SIDE (and so on SM103 only): the main stream joins the KV side stream right before the "
    "query absorption instead of before the split read, so the split read's only dependency is the absorption and it "
    "keeps its programmatic early launch; the output gate (TRTLLM_K3LL_MLA_GATE_SIDE, forked after the absorption) "
    "then runs on the SMs the split read leaves instead of pushing split CTAs out (same kernels and bytes; read at "
    "forward time). 1 on, 0 off, -1 (default) on SM103 only (implementation.MLA_KV_JOIN_EARLY_CAPABILITIES)",
)
TRTLLM_K3LL_MLA_QNORM_PDL = _flag(
    "TRTLLM_K3LL_MLA_QNORM_PDL",
    1,
    "Helix all-head MLA queries: the q_a RMSNorm releases q_b at launch and reads its weight before the PDL wait "
    "(modules/kimi_k3_mla/q_norm_pdl.py, bitwise FlashInfer's kernel; read at forward time), for decode steps of 1 "
    "to decode_mla_q.QNORM_PDL_MAX_REQUESTS requests. SM103 only, like "
    "TRTLLM_K3LL_MLA_KV_SIDE: it acts on the HELIX_Q_ABSORB all-head path (implementation.HELIX_SWITCH_CAPABILITIES)",
)
TRTLLM_K3LL_MLA_PREWAIT_L2 = _flag(
    "TRTLLM_K3LL_MLA_PREWAIT_L2",
    0,
    "MLA decode weights before the PDL wait: K3MlaQAbsorb stages / L2-prefetches all of its W_UK tiles and releases "
    "its dependents after the wait; the V-up split-KV kernel L2-prefetches W_UV and releases reduce_v_up after the "
    "wait (bitwise; read at forward time)",
)
TRTLLM_K3LL_MLA_VAR_SPLIT = _flag(
    "TRTLLM_K3LL_MLA_VAR_SPLIT",
    1,
    "Helix MLA split read with the V up-projection: each request's own split-KV count from its KV length on the "
    "rank (computed on device once per step, decode_mla_dcp.var_split_counts), so a batch of mixed lengths is not "
    "bound by its longest request; equal lengths keep the static split's count (not bitwise in general: the split "
    "boundaries move; read at forward time). Only the V-up split read (TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A, SM103) "
    "takes the counts, and only for 4 or more requests; elsewhere (SM107, the V-up off, or 1-3 requests) the split "
    "read keeps the static split and no counts are computed",
)
TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY = _flag(
    "TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY",
    -1,
    "with per-request split-KV counts (TRTLLM_K3LL_MLA_VAR_SPLIT) on the fused V-up path: the split kernel's "
    "clusters past their request's count (the grid is sized for the largest count) take no warp role and exit "
    "right after the PDL wait instead of allocating tensor memory and walking an empty tile (bitwise: they write "
    "nothing the reduction reads; read at forward time). 1 on, 0 off, -1 (default) on SM103 only "
    "(mla_v_up.MLA_VAR_SPLIT_SKIP_EMPTY_CAPABILITIES)",
)
TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE = _flag(
    "TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE",
    1,
    "with TRTLLM_K3LL_MLA_VAR_SPLIT: the per-request split counts from one Triton program that scores every candidate "
    "at once (decode_mla_dcp._var_split_counts_tile_kernel) instead of one candidate after the other: the same counts "
    "(an integer choice; bitwise), ~2 us instead of 3.5 (4 requests) on GB300. 0: the sequential kernel. Read at "
    "forward time; the counts only run where TRTLLM_K3LL_MLA_VAR_SPLIT does (SM103)",
)
TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL = _flag(
    "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL",
    -1,
    "with TRTLLM_K3LL_MLA_VAR_SPLIT: per-request split-KV counts from 2 requests on instead of 4, so a short request "
    "beside a long one gives its splits to the long one (equal lengths keep the static split; not bitwise when the "
    "lengths differ: the split boundaries move; read at forward time). 1 on, 0 off, -1 (default) on SM103 only "
    "(decode_mla_dcp.MLA_VAR_SPLIT_SMALL_CAPABILITIES)",
)
TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT = _flag(
    "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT",
    -1,
    "with per-request split counts on the fused V-up path: the split kernel's grid holds the batch's splits back to "
    "back (the one-wave budget of slots per query tile, each mapped to its (request, split) on device after the PDL "
    "wait) instead of the largest count per request, so no cluster is launched past a request's count (same work "
    "and partials: bitwise; read at forward time). 1 on, 0 off, -1 (default) on SM103 only "
    "(mla_v_up.MLA_VAR_SPLIT_FLAT_CAPABILITIES)",
)
TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED = _flag(
    "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED",
    -1,
    "Helix MLA decode: the step's split-read plan (decode_mla_dcp.make_helix_plan: int32 page table, lengths, "
    "per-token bounds and empty flags, built by the step's first MLA layer) in one Triton launch instead of about ten "
    "int torch kernels in a row before that layer's split read (bitwise; read at forward time). 1 on, 0 off, -1 "
    "(default): on on SM103 only (decode_mla_dcp.MLA_HELIX_PLAN_FUSED_CAPABILITIES; L1 2x -0.023 ms, 4x -0.029 ms)",
)
TRTLLM_K3LL_FORCE_REPLAY = _flag(
    "TRTLLM_K3LL_FORCE_REPLAY",
    0,
    "measurement only, with forced acceptance: generation rows that follow a stream of TRTLLM_K3LL_FORCE_REPLAY_FILE "
    "commit its tokens and take its continuation as drafts, draws keyed by the stream offset (force_replay.py; MLA "
    "DSpark worker; read at forward time)",
)
TRTLLM_K3LL_SLICED_HELIX_PARTITION = _flag(
    "TRTLLM_K3LL_SLICED_HELIX_PARTITION",
    1,
    "Helix generation intake: merge_helix_requests reads the prompt from the request's serialized int32 bytes and "
    "slices only this rank's blocks (pyexecutor/request_utils.py; same lists; read per call)",
)
TRTLLM_K3LL_DISAGG_GEN_FAST_SYNC = _flag(
    "TRTLLM_K3LL_DISAGG_GEN_FAST_SYNC",
    0,
    "disagg generation (no attention DP, PP 1): the per-iteration non-blocking KV-receive poll is skipped when one "
    "int64 all-gather shows no receive session on any rank (disaggregation/transceiver.py; read per call)",
)
TRTLLM_K3LL_SPEC_STEP_COPIES_SM10X = _flag(
    "TRTLLM_K3LL_SPEC_STEP_COPIES_SM10X",
    1,
    "the speculative step's overlap gathers and sampler store scatters as the two spec_step_copies kernels (one "
    "launch each, bit for bit the torch ops): 1 on SM 103, 2 on SM 103 and SM 107 (SM 100 always uses them; read per "
    "step)",
)
TRTLLM_K3LL_KDA_SEED_STACKED = _flag(
    "TRTLLM_K3LL_KDA_SEED_STACKED",
    1,
    "disagg generation intake: MambaHybridCacheManagerV2 seeds every KDA layer's replay conv caches of the new "
    "requests at once from the stacked conv-state view (~10 launches, not ~7 per layer; same values; read per call)",
)
TRTLLM_K3LL_SPEC_ACCEPT_WIDE_TABLE = _flag(
    "TRTLLM_K3LL_SPEC_ACCEPT_WIDE_TABLE",
    -1,
    "trtllm::k3_spec_accept: the draft pool's block table (B x max_blocks entries, max_blocks = max_seq_len / 64) "
    "written by every CTA over its column slice with all B rows' loads in flight, the counts summed by the last CTA, "
    "instead of CTA 0 walking it alone; the same bits (read per call, so per CUDA graph capture). 1 on, 0 off, -1 "
    "(default) on SM103 only (spec_worker.SPEC_ACCEPT_WIDE_TABLE_CAPABILITIES; SM107 / Rubin keeps CTA 0's table "
    "unless forced with 1)",
)
TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT = _flag(
    "TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT",
    -1,
    "trtllm::k3_spec_accept: N > 0 reduces each 1024-column slice of the target logits with B (K + 1) / N CTAs of N "
    "rows each (N rounded down to a divisor of the row count) instead of one CTA walking all B (K + 1) rows; the "
    "same bits, greedy or sampled. The TP16 shard has 10 such CTAs, and with temperature sampling every element "
    "also draws its Gumbel noise: 1 takes the bs4 MTP7 sampled call from 53 to 13 us (read per call, so per CUDA "
    "graph capture). 0 off; -1 (default) 1 on SM103 only (spec_worker.SPEC_ACCEPT_ROW_SPLIT_CAPABILITIES; SM107 / "
    "Rubin keeps one CTA per slice unless set > 0)",
)
TRTLLM_K3LL_SPEC_HEAD_FUSED = _flag(
    "TRTLLM_K3LL_SPEC_HEAD_FUSED",
    1,
    "speculative decode step without context tokens: _preprocess_inputs' head bookkeeping (overlap position / "
    "KV-length correction, Helix position correction and spec buffers) as one fused_spec_head_prep launch, bit for "
    "bit the torch ops (pyexecutor/engine/runners/decoder/runner.py; read per call, so per CUDA graph capture)",
)
TRTLLM_K3LL_EXPERT_PLACEMENT = _path_flag(
    "TRTLLM_K3LL_EXPERT_PLACEMENT",
    "routed MoE, at load: a kimi_k3_expert_placement/v1 JSON whose expert_of_slot[layer][slot] names the checkpoint "
    "expert each slot (EP rank slot // 224) holds; the router rows and score bias are permuted to slot order, decode "
    "unchanged (expert_placement.py; not bitwise: the EP groups' partial sums and top-k tie order change)",
)
TRTLLM_K3LL_ROUTING_STATS_DIR = _path_flag(
    "TRTLLM_K3LL_ROUTING_STATS_DIR",
    "measurement only, rank 0: per-step token counts of every MoE layer's checkpoint experts on the lowm decode path "
    "(<= 32 tokens), counted on the GPU inside the CUDA graphs and written as routing_*.npz chunks to this directory "
    "(routing_stats.py; adds small kernels per MoE layer on rank 0, so do not time with it)",
)
TRTLLM_K3LL_ROUTING_STATS_STRIDE = _flag(
    "TRTLLM_K3LL_ROUTING_STATS_STRIDE",
    0,
    "with TRTLLM_K3LL_ROUTING_STATS_DIR: tokens per request of a decode step (1 + draft length); > 0 adds a second "
    "count plane of every request's first token",
)
TRTLLM_K3LL_ROUTING_STATS_CHUNK = _flag(
    "TRTLLM_K3LL_ROUTING_STATS_CHUNK",
    1024,
    "with TRTLLM_K3LL_ROUTING_STATS_DIR: steps per .npz chunk (one is also written every 30 s and at exit)",
)

TRTLLM_K3LL_ATTN_RES_MXFP8 = _flag(
    "TRTLLM_K3LL_ATTN_RES_MXFP8",
    -1,
    "decode steps whose MoE tail pushes into the all-reduce (TRTLLM_K3LL_MOE_TAIL_PUSH): the K3AttnResFast that "
    "finishes it before an attention also writes its output's MXFP8 form, and the attention's first FP8 projection "
    "(KDA fused [q | k | v | g | f_a | b] or QKVG, MLA fused q_a / kv_a) runs k3_fp8w_lowm_mxfp8 on it instead of "
    "quantizing inside k3_fp8w_lowm_gemm (bitwise; read at forward time). 1 on, 0 off, -1 (default) on SM103 only "
    "(implementation.ATTN_RES_MXFP8_CAPABILITIES); k3_fp8w_lowm_mxfp8 itself takes SM100 / SM103 only, so on SM107 / "
    "Rubin nothing changes even with 1",
)
TRTLLM_K3LL_MX_SF_EARLY_RELEASE = _flag(
    "TRTLLM_K3LL_MX_SF_EARLY_RELEASE",
    -1,
    "k3_fp8w_lowm_mxfp8 (TRTLLM_K3LL_ATTN_RES_MXFP8): where every K tile has its own activation stage, release each "
    "K tile to the MMA once its scales are folded, not after all of them (bitwise; read at forward time). 1 on, 0 "
    "off, -1 (default) on SM103 only (fp8w_lowm.MX_SF_EARLY_RELEASE_CAPABILITIES); k3_fp8w_lowm_mxfp8 itself takes "
    "SM100 / SM103 only",
)
TRTLLM_K3LL_KDA_VERIFY_HOIST = _flag(
    "TRTLLM_K3LL_KDA_VERIFY_HOIST",
    -1,
    "k3_fp8_kda_verify: issue every prologue load while the cluster forms, two round trips deep (the slot, then the "
    "accepted-draft count, pool state, weights and, by bulk copies into shared memory, the conv history at every "
    "cache column and every draft's records), instead of one load round trip per accepted draft (bitwise; read at "
    "forward time). 1 on, 0 off, -1 (default) on SM103 only (verify.KDA_VERIFY_HOIST_CAPABILITIES)",
)
TRTLLM_K3LL_KDA_VERIFY_ODD_CLAMP = _flag(
    "TRTLLM_K3LL_KDA_VERIFY_ODD_CLAMP",
    -1,
    "k3_fp8_kda_verify at a verify width that is not a multiple of 4 (MTP6: 7 tokens): run the q / k pre-compute "
    "without a branch, the warp's token past the last recomputing the last one (bitwise; read at forward time). 1 "
    "on, 0 off, -1 (default) on SM103 only (verify.KDA_VERIFY_ODD_CLAMP_CAPABILITIES)",
)
TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS = _flag(
    "TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS",
    -1,
    "k3_fp8_kda_verify with the hoisted prologue (TRTLLM_K3LL_KDA_VERIFY_HOIST) and the folded f_b: wait for the grid "
    "dependency and issue this step's projection-row loads before the conv-history pick and the replay of the "
    "accepted drafts, so the rows' load round trip overlaps them (bitwise; read at forward time). 1 on, 0 off, -1 "
    "(default) on SM103 only (verify.KDA_VERIFY_EARLY_ROWS_CAPABILITIES)",
)
TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS = _flag(
    "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS",
    -1,
    "Helix MLA split read with per-request split counts on the FLAT grid (at most 32 rows): size the split workspace "
    "for every count the budget allows (decode_mla_dcp._flat_units) instead of the static split's bound, so one long "
    "request among short ones takes up to budget - B + 1 splits (bs4 MTP7: 9, was 6). The static split's units are "
    "queried at run time (148 or 152 on GB300) and the widened units capped at 1.5x them (222 or 228): MTP7 needs "
    "216 units (+16.8..17.9 MB per rank); the allocation covers query lengths 1..8, whose one-token case reaches the "
    "cap (+19.4..20.0 MB). Only where the split read takes per-request counts (Helix V-up split read, "
    "TRTLLM_K3LL_MLA_VAR_SPLIT, FLAT on); elsewhere nothing is allocated. Equal-length batches keep their counts "
    "(bitwise); mixed ones move split boundaries. Read at forward time; widens only where the allocated workspace "
    "holds it. 1 on, 0 off, -1 (default) on SM103 only (decode_mla_dcp.MLA_VAR_SPLIT_WIDE_WS_CAPABILITIES)",
)
TRTLLM_K3LL_DRAFTER_MARKOV_FAST = _flag(
    "TRTLLM_K3LL_DRAFTER_MARKOV_FAST",
    1,
    "with TRTLLM_K3LL_DRAFTER_MARKOV_V2: the MMA build's bias fragments by ldmatrix, and no store of the corrected "
    "logits on steps where only its tokens are read (greedy drafts, no lilicorr / DFlash 2 selector); the same "
    "tokens and next_new bit for bit (read at forward time)",
)

TRTLLM_K3LL_INPUT_CHAIN = _flag(
    "TRTLLM_K3LL_INPUT_CHAIN",
    -1,
    "decode-step input chain (pyexecutor/input_commit.py, set at model construction by input_chain.py): the defaults "
    "TRTLLM_BATCHED_INPUT_COMMIT=1 (H2D input copies and overlap gathers of a CUDA-graph decode step in one "
    "_input_commit_kernel), TRTLLM_KV_BLOCK_OFFSETS_TRITON=2 (target and draft block-offset tables in one "
    "_copy_block_offsets_pair_kernel), TRTLLM_STATE_INDEX_COPY_KERNEL=1 (state indices + dummy mask in one "
    "_small_copy_pair_kernel) and TRTLLM_SPEC_SAMPLER_FUSED_SCATTER=1 (store scatters in one "
    "_spec_store_scatter_kernel, before the spec_step_copies scatter); each of those set in the environment wins "
    "(the same values). 1 on, 0 off, -1 (default) on SM103 only (input_chain.INPUT_CHAIN_CAPABILITIES). Not "
    "covered: TRTLLM_FUSED_HELIX_SPEC_PREPROCESS and TRTLLM_BATCHED_INPUT_COMMIT_VERIFY (environment only)",
)


def flags() -> Dict[str, Tuple[int, int, str]]:
    """Every integer switch: name -> (value, default, doc)."""
    return dict(_FLAGS)


def path_flags() -> Dict[str, Tuple[str, str]]:
    """Every path switch: name -> (value, doc); ``""`` is off."""
    return dict(_PATH_FLAGS)


def log_flags() -> None:
    """Print every switch's effective value (``*`` marks a non-default value)."""
    parts = [
        f"{name}={value}{'*' if value != default else ''}"
        for name, (value, default, _) in _FLAGS.items()
    ]
    parts += [f"{name}={value!r}{'*' if value else ''}" for name, (value, _) in _PATH_FLAGS.items()]
    logger.info("Kimi K3 low-latency switches: " + " ".join(parts))

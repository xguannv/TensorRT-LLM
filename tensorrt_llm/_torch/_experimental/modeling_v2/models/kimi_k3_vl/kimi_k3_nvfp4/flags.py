# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switches of the Kimi K3 NVFP4 low-latency decode path.

One ``_flag(...)`` line per switch: environment variable name, default, one sentence. A switch is read once, at
import, from the environment of every rank; ``log_flags`` prints every switch's effective value when the model is
built, so a log always states which variant ran. With every switch at its default the target behaves as before the
switch was added.
"""

import os
from typing import Dict, Tuple

from tensorrt_llm.logger import logger

_FLAGS: Dict[str, Tuple[int, int, str]] = {}


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
TRTLLM_K3LL_DRAFTER_PREP_FUSED = _flag(
    "TRTLLM_K3LL_DRAFTER_PREP_FUSED",
    1,
    "MLA DSpark worker (with MLA_SPEC_ACCEPT and MLA_DRAFTER_CTX), on a k3_spec_accept step: the drafter's inputs in "
    "one k3_drafter_input_prep launch (bonus, positions, context-store targets and mask, ctx_len, seq_lens, the "
    "block fixup, pool slots and query RoPE rows, the noise rows' lookup), the noise block in one copy kernel, the "
    "store mask in k3_drafter_mla_ctx_kv (the same values; read per step)",
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
TRTLLM_K3LL_MLA_QNORM_PDL = _flag(
    "TRTLLM_K3LL_MLA_QNORM_PDL",
    0,
    "Helix all-head MLA queries: the q_a RMSNorm releases q_b at launch and reads its weight before the PDL wait "
    "(modules/kimi_k3_mla/q_norm_pdl.py, bitwise FlashInfer's kernel; read at forward time)",
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
    "takes the counts; elsewhere (SM107, or the V-up off) the split read keeps the static split and no counts are "
    "computed",
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
TRTLLM_K3LL_SPEC_HEAD_FUSED = _flag(
    "TRTLLM_K3LL_SPEC_HEAD_FUSED",
    1,
    "speculative decode step without context tokens: _preprocess_inputs' head bookkeeping (overlap position / "
    "KV-length correction, Helix position correction and spec buffers) as one fused_spec_head_prep launch, bit for "
    "bit the torch ops (pyexecutor/engine/runners/decoder/runner.py; read per call, so per CUDA graph capture)",
)


def flags() -> Dict[str, Tuple[int, int, str]]:
    """Every switch: name -> (value, default, doc)."""
    return dict(_FLAGS)


def log_flags() -> None:
    """Print every switch's effective value (``*`` marks a non-default value)."""
    parts = [
        f"{name}={value}{'*' if value != default else ''}"
        for name, (value, default, _) in _FLAGS.items()
    ]
    logger.info("Kimi K3 low-latency switches: " + " ".join(parts))

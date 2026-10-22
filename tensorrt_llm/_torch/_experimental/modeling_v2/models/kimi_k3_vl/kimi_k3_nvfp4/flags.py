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
TRTLLM_K3LL_DRAFTER_ATTN_CUTEDSL = _flag(
    "TRTLLM_K3LL_DRAFTER_ATTN_CUTEDSL",
    1,
    "MLA DSpark drafter: block decode on its cute-dsl implementation when speculative_config leaves it at AUTO",
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
    0,
    "KDA decode / verify: [q | k | v | g | f_a | b] in one k3_fp8w_lowm_gemm, the bf16 [f_a | b] weight quantized "
    "to FP8 128x128 blocks and appended to the QKVG weight (changes numerics); the weight is built at load when 1",
)
TRTLLM_K3LL_FP8W_LOWM = _flag(
    "TRTLLM_K3LL_FP8W_LOWM",
    1,
    "FP8 attention projections in DeepGEMM layout (weight reads) and on k3_fp8w_lowm_gemm at decode size",
)
TRTLLM_K3LL_FP8W_LOWM_FUSED_A = _flag(
    "TRTLLM_K3LL_FP8W_LOWM_FUSED_A",
    0,
    "with FP8W_LOWM, also the weight reads whose N is not a multiple of 128 (MLA fused q_a/kv_a, partial row tile)",
)

TRTLLM_K3LL_MOE_FRONT16 = _flag(
    "TRTLLM_K3LL_MOE_FRONT16",
    1,
    "lowm MoE on 9-16 token steps: the head, routing, NVFP4 latent and shared gate_up in k3_moe_front's 16-token build",
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

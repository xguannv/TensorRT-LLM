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
    0,
    "MLA DSpark drafter: residual adds and RMSNorms in the o_proj / down all-reduces (k3_sandwich_plain)",
)
TRTLLM_K3LL_MLA_DRAFTER_CTX = _flag(
    "TRTLLM_K3LL_MLA_DRAFTER_CTX",
    0,
    "MLA DSpark drafter (with TRTLLM_K3LL_MLA_SPEC_WORKER): context latents of all layers in one k3_drafter_mla_ctx_kv",
)
TRTLLM_K3LL_MLA_SPEC_ACCEPT = _flag(
    "TRTLLM_K3LL_MLA_SPEC_ACCEPT",
    0,
    "MLA DSpark worker: vocab-sharded target logits + k3_spec_accept (greedy, or temperature via no_topk_no_topp)",
)


TRTLLM_K3LL_FP8W_LOWM = _flag(
    "TRTLLM_K3LL_FP8W_LOWM",
    0,
    "FP8 attention projections in DeepGEMM layout (weight reads) and on k3_fp8w_lowm_gemm at decode size",
)
TRTLLM_K3LL_FP8W_LOWM_FUSED_A = _flag(
    "TRTLLM_K3LL_FP8W_LOWM_FUSED_A",
    0,
    "with FP8W_LOWM, also the weight reads whose N is not a multiple of 128 (MLA fused q_a/kv_a, partial row tile)",
)
TRTLLM_K3LL_WIDE_ONESHOT = _flag(
    "TRTLLM_K3LL_WIDE_ONESHOT",
    0,
    "wide decode steps (9-16 tokens): one-shot fused AR + attn_res + RMSNorm and one-shot latent AR, not two-shot",
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

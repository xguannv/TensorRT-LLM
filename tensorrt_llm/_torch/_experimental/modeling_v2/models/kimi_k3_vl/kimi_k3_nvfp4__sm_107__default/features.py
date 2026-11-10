# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which optimizations the Kimi K3 SM107 target runs, per deployment. Nothing here reads the environment.

``routing.py`` maps each deployment shape to one profile (its ``parallel`` value: ``tp1``, ``helix8``,
``helix16_moetp4ep4``, ``dep16``), and each profile turns on a fixed set of optimizations. Only optimizations that
some deployment able to reach them leaves off are listed here; the others run wherever their code is reached.
:data:`RETIRED_ENV` lists the environment variables the target does not read; :func:`select` warns about any that is
set.

``route()`` calls :func:`select` with the deployment's profile before it imports the target's modules, so every
module of the target reads :func:`active` after the selection, at import (module-level constants) or when called.
Before any selection :func:`active` is ``K3Features()``: every optimization off, the plain code path. Unit tests that
import the modules directly get that; they change a field with ``monkeypatch.setattr(features, "_ACTIVE",
dataclasses.replace(features.active(), <field>=<value>))`` (fields read when called) or by patching the constant in
the module that reads it (fields read at import).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Dict, Optional


@dataclasses.dataclass(frozen=True)
class K3Features:
    """One deployment's settings. Each default is the plain code path."""

    # Token embedding and the speculative decode step's head (modeling.py)
    embedding_gather: bool = False  # TLLM_SPEC_HEAD_FUSION
    # Decode step inputs: one upload and one scatter at the graph replay (modeling.py)
    pack_step_inputs: bool = False  # TRTLLM_K3_ULL_STEP_PREP and TRTLLM_BATCHED_INPUT_COMMIT
    # MoE (moe_runtime.py, fp8_weight_read.py, modeling.py)
    fuse_moe_residual: bool = False  # TRTLLM_K3_FUSE_MOE_RESIDUAL
    fp8w_lowm_gemm: bool = False  # TRTLLM_K3_FP8W_LOWM_GEMM
    ld_fused: bool = False  # TRTLLM_K3_LD_FUSED=1
    shared_sm: int = 0  # TRTLLM_K3_SHARED_SM
    fc12_die_major: bool = False  # TRTLLM_K3_FC12_DIE_LAYOUT and TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR
    router_exact_kda: bool = False  # TRTLLM_K3_ROUTER_EXACT=kda
    latent_ar_own_ws: bool = False  # TRTLLM_K3_LATENT_AR_OWN_WS
    # Attention residual and the decoder layer's all-reduces (attn_res.py, modeling.py, moe_runtime.py, kernels/)
    attn_res_pdl: bool = False  # TRTLLM_K3_ATTN_RES_PDL=2
    attn_res_pdl_max_m: int = 64  # TRTLLM_K3_ATTN_RES_PDL_MAX_M
    attn_res_mxfp8: bool = False  # TRTLLM_K3_ATTN_RES_MXFP8
    attn_res_early_trigger: bool = False  # TRTLLM_K3_ATTN_RES_EARLY_TRIGGER
    cute_carveout_fix: bool = False  # TRTLLM_K3_CUTE_CARVEOUT_FIX
    ull_oproj_sandwich: bool = False  # TRTLLM_K3_ULL_OPROJ_SANDWICH
    ull_tail_sandwich: bool = False  # TRTLLM_K3_ULL_TAIL_SANDWICH
    ull2_coll_cp_tail: bool = False  # TRTLLM_K3_ULL2_COLL_CP_TAIL and TRTLLM_K3_ULL_TAIL1
    ull_hygiene: bool = False  # TRTLLM_K3_ULL_HYGIENE
    ull_front: bool = False  # TRTLLM_K3_ULL_FRONT and TRTLLM_K3_ULL_FRONT_EARLY
    ull4_mla_plan_early: bool = False  # TRTLLM_K3_ULL4_MLA_PLAN_EARLY
    # KDA (kda.py)
    kda_prepad_prefill: bool = False  # TLLM_KDA_PREPAD_PREFILL
    kda_fused_conv_prefill: bool = False  # TRTLLM_KDA_FUSED_CONV_PREFILL
    kda_l2_prefetch: bool = False  # TRTLLM_K3_KDA_L2_PREFETCH
    # MLA (mla.py, mla.py, mla.py, mla.py)
    ull2_mla_tail: bool = False  # TRTLLM_K3_ULL2_MLA_TAIL
    mla_q_norm_pdl: bool = False  # TRTLLM_K3_MLA_Q_NORM_PDL
    ull3_mla_gate_at: str = "decode"  # TRTLLM_K3_ULL3_MLA_GATE_AT
    ull2_mla_decode_pdl: bool = False  # TRTLLM_K3_ULL2_MLA_DEC_PDL
    # FP4 MLA generation decode (ops/attention/_fp4_mla_decode.py, kernels/fp4_mla_packed_reduce.py)
    fp4_mla_page_plan: bool = True  # TRTLLM_FP4_MLA_PAGE_PLAN (=0: False)
    fp4_mla_reduce_pdl: bool = False  # TRTLLM_FP4_MLA_REDUCE_PDL
    fp4_mla_f16_partials: bool = False  # TRTLLM_FP4_MLA_F16_PARTIALS
    # DSpark (drafter.py, dflash.py)
    draft3_glue: bool = False  # TRTLLM_K3_DRAFT3_GLUE
    dspark_draft_gemm: bool = False  # TLLM_DSPARK_DRAFT_GEMM
    ull2_draft_a2a: bool = False  # TRTLLM_K3_ULL2_DRAFT_A2A and TRTLLM_K3_ULL2_DRAFT_TAIL
    ull2_draft_store: bool = False  # TRTLLM_K3_ULL2_DRAFT_STORE
    spec_tail_fusion: bool = False  # TLLM_SPEC_TAIL_FUSION
    dspark_tail_v2: bool = False  # TLLM_DSPARK_TAIL_V2

    def changed(self) -> Dict[str, object]:
        """The fields that differ from the plain path, by name."""
        base = K3Features()
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(base, f.name)
        }


# One profile per deployment (the names are routing.py's ``parallel`` values).
PROFILES: Dict[str, K3Features] = {
    # Context servers (DWDP, one rank per process).
    "tp1": K3Features(
        kda_fused_conv_prefill=True,
        kda_prepad_prefill=True,
    ),
    # Generation over Helix CP8.
    "helix8": K3Features(
        attn_res_early_trigger=True,
        attn_res_mxfp8=True,
        attn_res_pdl=True,
        attn_res_pdl_max_m=128,
        dspark_tail_v2=True,
        embedding_gather=True,
        fc12_die_major=True,
        fp4_mla_reduce_pdl=True,
        fp8w_lowm_gemm=True,
        fuse_moe_residual=True,
        kda_l2_prefetch=True,
        latent_ar_own_ws=True,
        ld_fused=True,
        mla_q_norm_pdl=True,
        router_exact_kda=True,
        shared_sm=16,
        spec_tail_fusion=True,
    ),
    # Generation over Helix CP16, routed experts MoE TP4 x EP4 (lowest latency).
    "helix16_moetp4ep4": K3Features(
        attn_res_early_trigger=True,
        attn_res_mxfp8=True,
        attn_res_pdl=True,
        cute_carveout_fix=True,
        draft3_glue=True,
        dspark_draft_gemm=True,
        dspark_tail_v2=True,
        embedding_gather=True,
        fc12_die_major=True,
        fp4_mla_f16_partials=True,
        fp4_mla_reduce_pdl=True,
        fp8w_lowm_gemm=True,
        fuse_moe_residual=True,
        ld_fused=True,
        mla_q_norm_pdl=True,
        pack_step_inputs=True,
        router_exact_kda=True,
        spec_tail_fusion=True,
        ull2_coll_cp_tail=True,
        ull2_draft_a2a=True,
        ull2_draft_store=True,
        ull2_mla_decode_pdl=True,
        ull2_mla_tail=True,
        ull3_mla_gate_at="exchange:16,start",
        ull4_mla_plan_early=True,
        ull_front=True,
        ull_hygiene=True,
        ull_oproj_sandwich=True,
        ull_tail_sandwich=True,
    ),
    # Generation under attention DP over 16 ranks.
    "dep16": K3Features(
        attn_res_pdl=True,
        fp4_mla_page_plan=False,
        fp8w_lowm_gemm=True,
        fuse_moe_residual=True,
        spec_tail_fusion=True,
    ),
}

_FIELD_ENV = {
    "embedding_gather": "TLLM_SPEC_HEAD_FUSION",
    "pack_step_inputs": "TRTLLM_K3_ULL_STEP_PREP",
    "fuse_moe_residual": "TRTLLM_K3_FUSE_MOE_RESIDUAL",
    "fp8w_lowm_gemm": "TRTLLM_K3_FP8W_LOWM_GEMM",
    "ld_fused": "TRTLLM_K3_LD_FUSED",
    "shared_sm": "TRTLLM_K3_SHARED_SM",
    "fc12_die_major": "TRTLLM_K3_FC12_DIE_LAYOUT",
    "router_exact_kda": "TRTLLM_K3_ROUTER_EXACT",
    "latent_ar_own_ws": "TRTLLM_K3_LATENT_AR_OWN_WS",
    "attn_res_pdl": "TRTLLM_K3_ATTN_RES_PDL",
    "attn_res_pdl_max_m": "TRTLLM_K3_ATTN_RES_PDL_MAX_M",
    "attn_res_mxfp8": "TRTLLM_K3_ATTN_RES_MXFP8",
    "attn_res_early_trigger": "TRTLLM_K3_ATTN_RES_EARLY_TRIGGER",
    "cute_carveout_fix": "TRTLLM_K3_CUTE_CARVEOUT_FIX",
    "ull_oproj_sandwich": "TRTLLM_K3_ULL_OPROJ_SANDWICH",
    "ull_tail_sandwich": "TRTLLM_K3_ULL_TAIL_SANDWICH",
    "ull2_coll_cp_tail": "TRTLLM_K3_ULL2_COLL_CP_TAIL",
    "ull_hygiene": "TRTLLM_K3_ULL_HYGIENE",
    "ull_front": "TRTLLM_K3_ULL_FRONT",
    "ull4_mla_plan_early": "TRTLLM_K3_ULL4_MLA_PLAN_EARLY",
    "kda_prepad_prefill": "TLLM_KDA_PREPAD_PREFILL",
    "kda_fused_conv_prefill": "TRTLLM_KDA_FUSED_CONV_PREFILL",
    "kda_l2_prefetch": "TRTLLM_K3_KDA_L2_PREFETCH",
    "ull2_mla_tail": "TRTLLM_K3_ULL2_MLA_TAIL",
    "mla_q_norm_pdl": "TRTLLM_K3_MLA_Q_NORM_PDL",
    "ull3_mla_gate_at": "TRTLLM_K3_ULL3_MLA_GATE_AT",
    "ull2_mla_decode_pdl": "TRTLLM_K3_ULL2_MLA_DEC_PDL",
    "fp4_mla_page_plan": "TRTLLM_FP4_MLA_PAGE_PLAN",
    "fp4_mla_reduce_pdl": "TRTLLM_FP4_MLA_REDUCE_PDL",
    "fp4_mla_f16_partials": "TRTLLM_FP4_MLA_F16_PARTIALS",
    "draft3_glue": "TRTLLM_K3_DRAFT3_GLUE",
    "dspark_draft_gemm": "TLLM_DSPARK_DRAFT_GEMM",
    "ull2_draft_a2a": "TRTLLM_K3_ULL2_DRAFT_A2A",
    "ull2_draft_store": "TRTLLM_K3_ULL2_DRAFT_STORE",
    "spec_tail_fusion": "TLLM_SPEC_TAIL_FUSION",
    "dspark_tail_v2": "TLLM_DSPARK_TAIL_V2",
}

# Variables nothing reads: the code they switched runs wherever it is reached, or no deployment runs it.
_FIXED_ENV = (
    "KIMI_K3_ATTN_RES_TOPOLOGY",
    "KIMI_K3_FUSED_ATTN_RES",
    "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS",
    "TLLM_DSPARK_AR_PUSH",
    "TLLM_DSPARK_CTX_SIDE_STREAM",
    "TLLM_DSPARK_DRAFT_FUSION",
    "TLLM_DSPARK_FUSED_MARKOV",
    "TLLM_DSPARK_MARKOV_V2",
    "TLLM_DSPARK_MARKOV_V3",
    "TLLM_K3_MLA_GEN_BACKEND",
    "TLLM_KDA_BUF_CACHE_ENTRIES",
    "TLLM_SPEC_VOCAB_PARALLEL",
    "TLLM_SPEC_VOCAB_PARALLEL_COMM",
    "TLLM_SPEC_VOCAB_PARALLEL_SAMPLING",
    "TRTLLM_BATCHED_INPUT_COMMIT",
    "TRTLLM_BATCHED_INPUT_COMMIT_VERIFY",
    "TRTLLM_FP4_MLA_LPT_PLAN",
    "TRTLLM_FP4_MLA_PACKED_WAVE_PLAN",
    "TRTLLM_FP4_MLA_REDUCE_SPREAD_STATS",
    "TRTLLM_FP4_MLA_SPLIT_KV",
    "TRTLLM_HELIX_V_UP_BEFORE_A2A",
    "TRTLLM_K3_AR_FINISH_FUSED",
    "TRTLLM_K3_AR_PUSH",
    "TRTLLM_K3_AR_PUSH_AMID",
    "TRTLLM_K3_AR_PUSH_ONE_SHOT_MAX_BYTES",
    "TRTLLM_K3_ATTN_RES_SM_RESERVE",
    "TRTLLM_K3_BF16_OVER_PREQUANTIZED_MAX_M",
    "TRTLLM_K3_DRAFT3_QHEAD",
    "TRTLLM_K3_DRAFT3_REDUCE_UNROLL",
    "TRTLLM_K3_FC12_DIE_SCHED",
    "TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS",
    "TRTLLM_K3_FC12_PREZERO",
    "TRTLLM_K3_FC12_SCHED_LOOKAHEAD",
    "TRTLLM_K3_FC12_SM_RESERVE",
    "TRTLLM_K3_FP4_MLA_SHARED_DECODE",
    "TRTLLM_K3_FP8W_LOWM_FUSED_A",
    "TRTLLM_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES",
    "TRTLLM_K3_HELIX_TAIL_CTAS",
    "TRTLLM_K3_HELIX_TAIL_FUSED",
    "TRTLLM_K3_HELIX_TAIL_GATE",
    "TRTLLM_K3_HELIX_TAIL_MB",
    "TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER",
    "TRTLLM_K3_KDA_DECODE_FUSED_FAB",
    "TRTLLM_K3_KDA_L2_PREFETCH_MAX_ROWS",
    "TRTLLM_K3_KDA_MTP_V2",
    "TRTLLM_K3_KDA_MX_PDL",
    "TRTLLM_K3_KDA_QKVG_STAGE",
    "TRTLLM_K3_KDA_QKVG_STAGE_RING",
    "TRTLLM_K3_LATENT_AR_CUTE",
    "TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES",
    "TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M",
    "TRTLLM_K3_LATENT_AR_ROW",
    "TRTLLM_K3_LATENT_AR_ROW_MAX_M",
    "TRTLLM_K3_LATENT_STREAM_PRIORITY",
    "TRTLLM_K3_LD_FUSED_MAX_M",
    "TRTLLM_K3_LD_FUSED_MIN_M",
    "TRTLLM_K3_LOWM_SF_SLOTS",
    "TRTLLM_K3_LU_MID_MAX_M",
    "TRTLLM_K3_LU_MID_MXFP8",
    "TRTLLM_K3_MID_AR_PUSH",
    "TRTLLM_K3_MID_AR_PUSH_MAX_M",
    "TRTLLM_K3_MID_MXFP8_IN",
    "TRTLLM_K3_MID_OPROJ_PC",
    "TRTLLM_K3_MLA_CTX_FP8_KV_FUSED",
    "TRTLLM_K3_MLA_CTX_TRITON_QUANT",
    "TRTLLM_K3_MLA_GATE_SIDE",
    "TRTLLM_K3_MLA_GATE_SIDE_MAX_M",
    "TRTLLM_K3_MLA_Q_FUSED",
    "TRTLLM_K3_MLA_Q_FUSED_KV_PRIO",
    "TRTLLM_K3_MLA_Q_FUSED_MAX_TOKENS",
    "TRTLLM_K3_MLA_Q_FUSED_NORM",
    "TRTLLM_K3_OPROJ_PC",
    "TRTLLM_K3_ROUTER_FIRST",
    "TRTLLM_K3_ROUTER_FIRST_MAX_M",
    "TRTLLM_K3_ROUTE_APPROX",
    "TRTLLM_K3_ROUTE_GEMV_MAX_M",
    "TRTLLM_K3_ROUTE_TAIL",
    "TRTLLM_K3_ROUTE_TAIL_MAX_M",
    "TRTLLM_K3_ROUTE_TAIL_ORDER",
    "TRTLLM_K3_ROUTE_TAIL_ORDER_MIN_ROWS",
    "TRTLLM_K3_ROUTING_STATS_CHUNK",
    "TRTLLM_K3_ROUTING_STATS_DIR",
    "TRTLLM_K3_ROUTING_STATS_EVERY",
    "TRTLLM_K3_ROUTING_STATS_STRIDE",
    "TRTLLM_K3_RP_TAIL",
    "TRTLLM_K3_RP_TAIL_MAX_M",
    "TRTLLM_K3_SHARED_STREAM_PRIORITY",
    "TRTLLM_K3_SHARED_TAIL_MAX_M",
    "TRTLLM_K3_SHARED_TAIL_MIN_M",
    "TRTLLM_K3_SHARED_TAIL_PRIO",
    "TRTLLM_K3_ULL2_COLL_AR_EARLY_STATE",
    "TRTLLM_K3_ULL2_DRAFT_TAIL",
    "TRTLLM_K3_ULL3_MLA_LATE_TAIL_MAX_TOKENS",
    "TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER",
    "TRTLLM_K3_ULL_EXPERTS",
    "TRTLLM_K3_ULL_EXPERTS_COMBINE",
    "TRTLLM_K3_ULL_EXPERTS_CTAS",
    "TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR",
    "TRTLLM_K3_ULL_EXPERTS_TRIGGER",
    "TRTLLM_K3_ULL_FRONT_EARLY",
    "TRTLLM_K3_ULL_FRONT_MAX_M",
    "TRTLLM_K3_ULL_MOEBLK",
    "TRTLLM_K3_ULL_MOETP",
    "TRTLLM_K3_ULL_SHARED_PERSIST",
    "TRTLLM_K3_ULL_TAIL1",
)

# Variables read under another name -> that name.
_RENAMED_ENV = {
    "TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES": "TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES",
}

#: Environment variables the target does not read -> the field that replaces them (None: no field;
#: ``_RENAMED_ENV`` names the variable read instead).
RETIRED_ENV: Dict[str, Optional[str]] = {
    **{env: field for field, env in _FIELD_ENV.items()},
    **{env: None for env in _FIXED_ENV},
    **{env: None for env in _RENAMED_ENV},
}

_ACTIVE = K3Features()
_SELECTED: Optional[str] = None
_PACKAGE = __name__.rpartition(".")[0]


def active() -> K3Features:
    """The selected target's settings (``K3Features()`` before :func:`select`)."""
    return _ACTIVE


def selected() -> Optional[str]:
    """The selected target, or None."""
    return _SELECTED


def select(target: str) -> K3Features:
    """Make ``target``'s profile the active one; once per process, before any other module of the target.

    Raises if another profile was selected, or if a module of the target was imported first (its
    import-time settings would be the plain path's, not ``target``'s).
    """
    global _ACTIVE, _SELECTED
    if target not in PROFILES:
        raise KeyError(f"no Kimi K3 feature profile {target!r}; have {sorted(PROFILES)}")
    if _SELECTED is not None:
        if _SELECTED != target:
            raise RuntimeError(
                f"Kimi K3 features: {_SELECTED!r} is selected in this process, cannot select {target!r}"
            )
        return _ACTIVE
    early = sorted(m for m in sys.modules if m.startswith(_PACKAGE + ".") and m != __name__)
    if early:
        raise RuntimeError(
            f"Kimi K3 features: modules of the target were imported before the {target!r} profile was "
            f"selected, with the plain path's settings: {', '.join(early)}"
        )
    _ACTIVE, _SELECTED = PROFILES[target], target

    from tensorrt_llm.logger import logger

    changed = _ACTIVE.changed()
    logger.info(
        f"Kimi K3 features: {target} ({len(changed)} set): "
        + ", ".join(f"{name}={value}" for name, value in sorted(changed.items()))
    )
    stale = sorted(name for name in RETIRED_ENV if name in os.environ)
    if stale:
        names = ", ".join(
            f"{name} (set {_RENAMED_ENV[name]} instead)" if name in _RENAMED_ENV else name
            for name in stale
        )
        logger.warning(
            "Kimi K3 features: the target does not read these environment variables (its "
            f"profile in features.py decides): {names}"
        )
    return _ACTIVE

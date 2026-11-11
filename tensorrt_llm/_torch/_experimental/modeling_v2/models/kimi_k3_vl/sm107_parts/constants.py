# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The constants a Kimi K3 SM107 target hands to the family's SM107 code.

Each target (``kimi_k3_nvfp4__sm_107__<layout>/constants.py``) writes every field out; there are
no defaults. The family code reads them from the instance it is given, never from the environment
or a process-wide selection, so two targets can be imported in one process. Kernels and
operations do not read this class: their callers pass the values they need as arguments.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class K3Sm107Constants:
    """What one Kimi K3 SM107 target runs. A field is a fixed property of the target."""

    # The speculative decode step's head in one kernel (``RuntimeHooks.decode_prologue``) and the
    # token embedding as a vectorized row copy (formerly TLLM_SPEC_HEAD_FUSION).
    spec_head_fusion: bool
    # MoE (moe_runtime.py, fp8_weight_read.py, modeling.py).
    fuse_moe_residual: bool
    fp8w_lowm_gemm: bool
    ld_fused: bool
    shared_sm: int
    # FC12 die-major expert layout and schedule (the routed experts' kernel set).
    fc12_die_major: bool
    router_exact_kda: bool
    latent_ar_own_ws: bool
    # Attention residual (attn_res.py, kernels/k3r_attn_res).
    attn_res_pdl: bool
    attn_res_pdl_max_m: int
    attn_res_mxfp8: bool
    attn_res_early_trigger: bool
    # KDA (kda.py).
    kda_prepad_prefill: bool
    kda_fused_conv_prefill: bool
    kda_l2_prefetch: bool
    # MLA and its FP4 generation decode (mla.py, ops/attention/_fp4_mla_decode.py).
    mla_q_norm_pdl: bool
    fp4_mla_page_plan: bool
    fp4_mla_reduce_pdl: bool
    # DSpark (drafter.py, dflash.py, kernels/k3r_spec/spec_tail_kernels.py).
    spec_tail_fusion: bool
    dspark_tail_v2: bool
    # Device memory, in bytes, the engine leaves outside the PyTorch memory cap
    # (``RuntimeHooks.torch_mem_safety_margin``).
    torch_mem_safety_margin: int

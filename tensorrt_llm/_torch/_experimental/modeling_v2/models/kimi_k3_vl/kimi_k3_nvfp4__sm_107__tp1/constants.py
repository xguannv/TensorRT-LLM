# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the tp1 target runs: the constants it hands to the family's SM107 code.

Taken from every recipe's context servers (c383 977198, c240 977200, c24 932149 @ 69055f20a9): profile ``tp1``.
"""

from ..sm107_parts.constants import K3Sm107Constants

CONSTANTS = K3Sm107Constants(
    spec_head_fusion=False,
    fuse_moe_residual=False,
    fp8w_lowm_gemm=False,
    ld_fused=False,
    shared_sm=0,
    fc12_die_major=False,
    router_exact_kda=False,
    latent_ar_own_ws=False,
    attn_res_pdl=False,
    attn_res_pdl_max_m=64,
    attn_res_mxfp8=False,
    attn_res_early_trigger=False,
    kda_prepad_prefill=True,
    kda_fused_conv_prefill=True,
    kda_l2_prefetch=False,
    mla_q_norm_pdl=False,
    fp4_mla_page_plan=True,
    fp4_mla_reduce_pdl=False,
    spec_tail_fusion=False,
    dspark_tail_v2=False,
    torch_mem_safety_margin=4 * 1024**3,
)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the dep16 target runs: the constants it hands to the family's SM107 code.

Taken from 084745cde4's ``dep16`` profile (c383 977198, post arm), with the speculative decode head
fusion on as on the other generation targets.
"""

from ..sm107_parts.constants import K3Sm107Constants

CONSTANTS = K3Sm107Constants(
    spec_head_fusion=True,
    fuse_moe_residual=True,
    fp8w_lowm_gemm=True,
    ld_fused=False,
    shared_sm=0,
    fc12_die_major=False,
    router_exact_kda=False,
    latent_ar_own_ws=False,
    attn_res_pdl=True,
    attn_res_pdl_max_m=64,
    attn_res_mxfp8=False,
    attn_res_early_trigger=False,
    kda_prepad_prefill=False,
    kda_fused_conv_prefill=False,
    kda_l2_prefetch=False,
    mla_q_norm_pdl=False,
    fp4_mla_page_plan=False,
    fp4_mla_reduce_pdl=False,
    spec_tail_fusion=True,
    dspark_tail_v2=False,
    torch_mem_safety_margin=4 * 1024**3,
)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the helix8_moeep8 target runs: the constants it hands to the family's SM107 code.

Taken from 932149 (c24 @ 69055f20a9) exported the same values as 084745cde4's ``helix8`` profile (checked
against its start_server_cmds.sh, see notes lines/split.md); the two helix8 targets differ only in
the MoE split.
"""

from ..sm107_parts.constants import K3Sm107Constants

CONSTANTS = K3Sm107Constants(
    spec_head_fusion=True,
    fuse_moe_residual=True,
    fp8w_lowm_gemm=True,
    ld_fused=True,
    shared_sm=16,
    fc12_die_major=True,
    router_exact_kda=True,
    latent_ar_own_ws=True,
    attn_res_pdl=True,
    attn_res_pdl_max_m=128,
    attn_res_mxfp8=True,
    attn_res_early_trigger=True,
    kda_prepad_prefill=False,
    kda_fused_conv_prefill=False,
    kda_l2_prefetch=True,
    mla_q_norm_pdl=True,
    fp4_mla_page_plan=True,
    fp4_mla_reduce_pdl=True,
    spec_tail_fusion=True,
    dspark_tail_v2=True,
    torch_mem_safety_margin=4 * 1024**3,
)

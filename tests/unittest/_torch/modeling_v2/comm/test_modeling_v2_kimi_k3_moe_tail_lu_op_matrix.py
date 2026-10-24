# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the wide MoE tail with the replicated FP8 latent_up of the Kimi K3 NVFP4 decode path
(TRTLLM_K3LL_MOE_TAIL_LU).

The checks are ``_kimi_k3_moe_tail_lu_op_matrix.py`` beside this file, its own W-rank launcher (the cases share the
decode path's collective state, so one job, not independent cases); see ``_rank_job`` for why that is left intact.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "TRTLLM_K3LL_MOE_TAIL_LU requires CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    # The latent all-reduce, the MXFP8 GEMM, the push GEMV and the attn_res kernel run on sm_100 / sm_103 only.
    pytest.skip("the MoE tail LU kernels run on sm_100 / sm_103 only", allow_module_level=True)

if torch.cuda.device_count() < _rank_job.WORLD_SIZE:
    # One rank per device: fewer visible devices cannot host the check's world size.
    pytest.skip(
        f"the check runs {_rank_job.WORLD_SIZE} ranks, one per device; "
        f"{torch.cuda.device_count()} visible",
        allow_module_level=True,
    )


# The case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_kimi_k3_moe_tail_lu_op_matrix() -> None:
    _rank_job.run("kimi_k3_moe_tail_lu")

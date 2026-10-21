# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the mnnvl_allreduce_rmsnorm_slice op's certification matrix.

The matrix is ``_mnnvl_allreduce_rmsnorm_slice_op_matrix.py`` beside this file, its own W-rank launcher (call
sequences over caller-owned workspaces, so one job, not independent cases); see ``_rank_job`` for why that is left
intact.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "mnnvl_allreduce_rmsnorm_slice requires CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip(
        "mnnvl_allreduce_rmsnorm_slice is certified on sm_100 / sm_103 only",
        allow_module_level=True,
    )


# Each case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_mnnvl_allreduce_rmsnorm_slice_op_matrix() -> None:
    _rank_job.run("mnnvl_allreduce_rmsnorm_slice")

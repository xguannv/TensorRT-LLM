# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the MLA DSpark drafter's context path on a side stream with the TP group's decode
collectives (Kimi K3 NVFP4 decode path).

The checks are ``_kimi_k3_drafter_input_op_matrix.py`` beside this file, its own W-rank launcher (the side stream's
all-reduce shares the decode path's collective state, so one job, not independent cases); see ``_rank_job`` for why
that is left intact.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "the Kimi K3 drafter input checks require CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    # The MNNVL collectives run on sm_100 / sm_103 only.
    pytest.skip(
        "the Kimi K3 drafter input checks run on sm_100 / sm_103 only", allow_module_level=True
    )

if torch.cuda.device_count() < _rank_job.WORLD_SIZE:
    # One rank per device: fewer visible devices cannot host the check's world size.
    pytest.skip(
        f"the check runs {_rank_job.WORLD_SIZE} ranks, one per device; "
        f"{torch.cuda.device_count()} visible",
        allow_module_level=True,
    )


# The case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_kimi_k3_drafter_input_op_matrix() -> None:
    _rank_job.run("kimi_k3_drafter_input")

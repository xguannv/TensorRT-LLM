# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the MoE-path switches flipped after warm-up and first run under capture.

The checks are ``_kimi_k3_lowm_warm_up_capture_op_matrix.py`` beside this file, its own W-rank launcher; one fresh
job per warm-up state (``K3LL_WARM_UP_STATE``), since a build compiled earlier in the same process would hide a gap.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "the Kimi K3 lowm MoE switches require CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    # The lowm expert kernel and the MoE front run on sm_100 / sm_103 only (nvfp4_lowm_op.supported).
    pytest.skip("the Kimi K3 lowm MoE path runs on sm_100 / sm_103 only", allow_module_level=True)

if torch.cuda.device_count() < _rank_job.WORLD_SIZE:
    # One rank per device: fewer visible devices cannot host the check's world size.
    pytest.skip(
        f"the check runs {_rank_job.WORLD_SIZE} ranks, one per device; "
        f"{torch.cuda.device_count()} visible",
        allow_module_level=True,
    )


# The case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
@pytest.mark.parametrize("state", ["on", "off"])
def test_kimi_k3_lowm_warm_up_capture_op_matrix(state, monkeypatch) -> None:
    monkeypatch.setenv("K3LL_WARM_UP_STATE", state)
    _rank_job.run("kimi_k3_lowm_warm_up_capture")

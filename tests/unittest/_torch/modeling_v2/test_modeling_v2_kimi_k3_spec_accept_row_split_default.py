# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT`` (``kimi_k3_nvfp4/flags.py``): the ``rows_per_cta`` the Kimi K3 spec workers
pass to ``trtllm::k3_spec_accept``. N > 0 is passed as given and 0 is off on any GPU; the default (-1) gives
``SPEC_ACCEPT_ROW_SPLIT_DEFAULT_ROWS`` (1) only on the compute capabilities of
``spec_worker.SPEC_ACCEPT_ROW_SPLIT_CAPABILITIES`` (SM103), so SM107 / Rubin keeps one CTA per column slice. No GPU
needed: the device capability is patched.
"""

import pytest
import torch


def _sw():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        spec_worker,
    )

    return spec_worker


@pytest.mark.parametrize("sample", [False, True], ids=["greedy", "sampled"])
@pytest.mark.parametrize(
    "value,capability,rows",
    [
        (1, (10, 3), 1),
        (2, (10, 7), 2),
        (0, (10, 3), 0),
        (0, (10, 7), 0),
        (-1, (10, 3), 1),
        (-1, (10, 7), 0),
        (-1, (10, 0), 0),
    ],
)
def test_row_split_value(monkeypatch, value, capability, rows, sample):
    sw = _sw()
    monkeypatch.setattr(sw._flags, "TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT", value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    sw._row_split_by_default.cache_clear()
    try:
        assert sw._k3_rows_per_cta(sample, torch.device("cuda", 0)) == rows
    finally:
        sw._row_split_by_default.cache_clear()


def test_default_is_auto():
    sw = _sw()
    assert sw._flags._FLAGS["TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT"][1] == -1
    assert sw.SPEC_ACCEPT_ROW_SPLIT_CAPABILITIES == ((10, 3),)
    assert sw.SPEC_ACCEPT_ROW_SPLIT_DEFAULT_ROWS == 1

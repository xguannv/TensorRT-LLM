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
"""``TRTLLM_K3LL_SPEC_ACCEPT_WIDE_TABLE`` (``kimi_k3_nvfp4/flags.py``): which ``wide_table`` the Kimi K3 spec
workers pass to ``trtllm::k3_spec_accept``. 1 forces it on and 0 off on any GPU; the default (-1) turns it on only
on the compute capabilities of ``spec_worker.SPEC_ACCEPT_WIDE_TABLE_CAPABILITIES`` (SM103), so SM107 / Rubin keeps
CTA 0's block table. No GPU needed: the device capability is patched.
"""

import pytest
import torch


def _sw():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        spec_worker,
    )

    return spec_worker


@pytest.mark.parametrize(
    "value,capability,wide",
    [
        (1, (10, 3), True),
        (1, (10, 7), True),
        (0, (10, 3), False),
        (0, (10, 7), False),
        (-1, (10, 3), True),
        (-1, (10, 7), False),
        (-1, (10, 0), False),
    ],
)
def test_wide_table_value(monkeypatch, value, capability, wide):
    sw = _sw()
    monkeypatch.setattr(sw._flags, "TRTLLM_K3LL_SPEC_ACCEPT_WIDE_TABLE", value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    sw._wide_table_by_default.cache_clear()
    try:
        assert sw._k3_wide_table(torch.device("cuda", 0)) is wide
    finally:
        sw._wide_table_by_default.cache_clear()


def test_default_is_auto():
    sw = _sw()
    assert sw._flags._FLAGS["TRTLLM_K3LL_SPEC_ACCEPT_WIDE_TABLE"][1] == -1
    assert sw.SPEC_ACCEPT_WIDE_TABLE_CAPABILITIES == ((10, 3),)

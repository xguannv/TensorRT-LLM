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
"""``TRTLLM_K3LL_MLA_KV_JOIN_EARLY`` (``kimi_k3_nvfp4/flags.py``): where the MLA decode joins the KV side stream. 1
forces the early join on and 0 off; the default (-1) turns it on only on the compute capabilities of
``implementation.MLA_KV_JOIN_EARLY_CAPABILITIES`` (SM103). No GPU needed: the device capability is patched.
"""

import pytest
import torch


def _impl():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        implementation,
    )

    return implementation


@pytest.mark.parametrize(
    "value,capability,on",
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
def test_kv_join_early_value(monkeypatch, value, capability, on):
    impl = _impl()
    monkeypatch.setattr(impl._flags, "TRTLLM_K3LL_MLA_KV_JOIN_EARLY", value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    impl._kv_join_early_by_default.cache_clear()
    try:
        assert impl._kv_join_early_on(torch.device("cuda", 0)) is on
    finally:
        impl._kv_join_early_by_default.cache_clear()


def test_default_is_auto():
    impl = _impl()
    assert impl._flags._FLAGS["TRTLLM_K3LL_MLA_KV_JOIN_EARLY"][1] == -1
    assert impl.MLA_KV_JOIN_EARLY_CAPABILITIES == ((10, 3),)

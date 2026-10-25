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
"""``TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS`` (``kimi_k3_nvfp4/flags.py``): whether ``trtllm::k3_fp8_kda_verify`` waits for
the grid dependency and loads the projection rows before the history pick and the replay (the same bits). Only with
the hoisted prologue and the folded f_b; otherwise 1 forces it on and 0 off, and the default (-1) turns it on only on
the compute capabilities of ``verify.KDA_VERIFY_EARLY_ROWS_CAPABILITIES`` (SM103). No GPU needed: the device
capability is patched.
"""

import pytest
import torch


def _verify():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import verify

    return verify


def _flags():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    return flags


@pytest.mark.parametrize(
    "value,capability,hoist,fold_fb,on",
    [
        (1, (10, 3), True, True, True),
        (1, (10, 7), True, True, True),
        (0, (10, 3), True, True, False),
        (-1, (10, 3), True, True, True),
        (-1, (10, 7), True, True, False),
        (-1, (10, 0), True, True, False),
        (1, (10, 3), False, True, False),
        (-1, (10, 3), False, True, False),
        (1, (10, 3), True, False, False),
    ],
)
def test_early_rows_value(monkeypatch, value, capability, hoist, fold_fb, on):
    vf = _verify()
    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS", value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    vf._early_rows_by_default.cache_clear()
    try:
        assert vf._early_rows(torch.device("cuda", 0), hoist, fold_fb) is on
    finally:
        vf._early_rows_by_default.cache_clear()


def test_default_is_auto():
    assert _flags()._FLAGS["TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS"][1] == -1
    assert _verify().KDA_VERIFY_EARLY_ROWS_CAPABILITIES == ((10, 3),)

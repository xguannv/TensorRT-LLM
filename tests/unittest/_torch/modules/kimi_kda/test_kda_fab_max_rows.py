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
"""Row cap of the fused KDA gate projection in the verify front end (TRTLLM_K3_KDA_FAB_MAX_ROWS).

``k3_kda_fab`` accepts up to ``MAX_M`` rows; the mixer routes the verify ``[f_a | b] -> f_b``
chain to it (full or f_a mode) only up to ``_KDA_FAB_FUSED_MAX_ROWS`` rows (256 unless the
environment raises it), so that the default routing is the one of a 256-row kernel.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

H = 6
HIDDEN = 7168


def _mixer():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer

    return kimi_kda_mixer


def _weights():
    w_bfa = (torch.randn(136, HIDDEN, device="cuda") * 0.02).bfloat16()
    w_fb = (torch.randn(H * 128, 128, device="cuda") * 0.05).bfloat16()
    return w_bfa, w_fb


def _x(batch, steps):
    return torch.randn(batch, steps, HIDDEN, device="cuda").bfloat16()


def test_default_cap_is_256():
    mix = _mixer()
    if "TRTLLM_K3_KDA_FAB_MAX_ROWS" not in os.environ:
        assert mix._KDA_FAB_FUSED_MAX_ROWS == 256


@pytest.mark.parametrize(
    "cap,batch,steps,expect",
    [
        (256, 32, 8, True),
        (256, 64, 6, False),
        (384, 64, 6, True),
        (512, 64, 8, True),
        (512, 65, 8, False),
        (1024, 128, 7, False),  # above the kernel's MAX_M
    ],
)
def test_fused_fab_respects_cap(monkeypatch, cap, batch, steps, expect):
    mix = _mixer()
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED", True)
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED_MAX_ROWS", cap)
    w_bfa, w_fb = _weights()
    assert mix._use_fused_fab(_x(batch, steps), w_bfa, w_fb, H) is expect


def test_fab_fa_respects_cap(monkeypatch):
    """f_a mode (single-wave verify, no two-CTA kernel) never runs above the cap either."""
    mix = _mixer()
    monkeypatch.setattr(mix, "_KDA_DECODE_FUSED_FAB", True)
    monkeypatch.setattr(mix, "_KDA_DECODE_2CTA", False)
    w_bfa, w_fb = _weights()
    x = _x(48, 6)  # 288 rows, 288 verify tiles
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED_MAX_ROWS", 256)
    assert not mix._use_fab_fa(x, w_bfa, w_fb, H)
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED_MAX_ROWS", 512)
    assert mix._use_fab_fa(x, w_bfa, w_fb, H)

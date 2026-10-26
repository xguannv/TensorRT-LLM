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
"""The per-module weight-load threads run on the model's device, not on ``local_mpi_rank()``.

A launcher that assigns devices itself can give a rank a device other than its MPI rank modulo the
device count; a load thread on another device fails its device-to-device copies.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models import modeling_utils


def _param(index=None):
    if index is None:
        return SimpleNamespace(is_cuda=False)
    return SimpleNamespace(is_cuda=True, device=SimpleNamespace(index=index))


@pytest.mark.cpu_only
def test_the_models_cuda_device_wins_over_local_mpi_rank(monkeypatch):
    monkeypatch.setattr(modeling_utils, "local_mpi_rank", lambda: 1)
    model = SimpleNamespace(parameters=lambda: iter([_param(), _param(2), _param(3)]))
    assert modeling_utils._weight_loading_device(model) == 2


@pytest.mark.cpu_only
def test_without_cuda_parameters_the_current_device_is_used(monkeypatch):
    monkeypatch.setattr(modeling_utils, "local_mpi_rank", lambda: 1)
    monkeypatch.setattr(modeling_utils.torch.cuda, "current_device", lambda: 3)
    model = SimpleNamespace(parameters=lambda: iter([_param()]))
    assert modeling_utils._weight_loading_device(model) == 3

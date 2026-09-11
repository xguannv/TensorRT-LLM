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
"""KVCacheManagerV2 must describe its per-pool-group sizes at construction.

`get_kv_cache_stats` sums `total` and `available` over every pool group, so the
one number a worker reports cannot distinguish "evenly half full" from "one
pool group at its ceiling while the rest are empty". Heterogeneous-attention
models reach the latter state routinely: a pool group whose per-request cost
does not scale with sequence length stays small next to the token-scaled
groups, so it fills first while the aggregate still reads low.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only

_MODULE = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"


def _manager(stats=None, side_effect=None) -> KVCacheManagerV2:
    """A manager carrying only the surface the size formatter reads."""
    manager = object.__new__(KVCacheManagerV2)
    manager._get_storage_statistics = Mock(return_value=stats, side_effect=side_effect)
    return manager


def test_reports_one_line_per_pool_group() -> None:
    """Every group is listed, so a small one cannot hide behind a large one."""
    stats = [
        SimpleNamespace(total=207735, slot_sizes=[512]),
        SimpleNamespace(total=136, slot_sizes=[31457280, 7864320]),
    ]
    entries = _manager(stats)._format_kv_cache_pool_group_sizes()

    assert len(entries) == 2
    assert "pool_group_id=0" in entries[0]
    assert "num_slots=207735" in entries[0]
    # The second group is the one worth finding: a slot count this close to a
    # plausible max_batch_size has no headroom for per-request variation.
    assert "pool_group_id=1" in entries[1]
    assert "num_slots=136" in entries[1]
    assert "slot_size=[31457280, 7864320]" in entries[1]
    assert f"bytes={136 * (31457280 + 7864320)}" in entries[1]


def test_accepts_either_backend_spelling() -> None:
    """The C++ binding exposes `slot_sizes`; the Python one `slot_size`.

    Both carry one size per pool in the group, so neither needs wrapping. The
    default backend is C++, which means a mistake here is invisible until
    someone runs the Python implementation.
    """
    entries = _manager(
        [SimpleNamespace(total=64, slot_size=[4096, 1024])]
    )._format_kv_cache_pool_group_sizes()

    assert len(entries) == 1
    assert "num_slots=64" in entries[0]
    assert "slot_size=[4096, 1024]" in entries[0]
    assert f"bytes={64 * (4096 + 1024)}" in entries[0]


def test_introspection_failure_does_not_break_startup() -> None:
    """A description of the layout must never be able to stop a worker."""
    manager = _manager(side_effect=RuntimeError("no storage"))
    assert manager._format_kv_cache_pool_group_sizes() == []


def test_sizes_are_logged_with_the_lifecycle_mapping() -> None:
    """The two are emitted together, so one never appears without the other."""
    manager = _manager()
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=[SimpleNamespace(layer_id=0, buffers=[SimpleNamespace(role="KEY")])]
    )
    manager._format_kv_cache_pool_lifecycle_entry = Mock(return_value="role=KEY, pool_group_id=0")

    with (
        patch.object(
            KVCacheManagerV2,
            "_format_kv_cache_pool_group_sizes",
            return_value=["pool_group_id=0, num_slots=7"],
        ),
        patch(f"{_MODULE}.logger") as mock_logger,
    ):
        manager._log_kv_cache_pool_lifecycle_mapping()

    logged = [call.args[0] for call in mock_logger.info.call_args_list]
    assert any("role=KEY" in line for line in logged)
    assert any("num_slots=7" in line for line in logged)

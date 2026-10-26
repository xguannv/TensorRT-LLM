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
"""``TransferWorker._register_kv_cache`` registers each KV pool on its own.

The transfer agent requires the descriptors of one registration to share a VMM chunk size, and pools of different
quotas (a target pool and an external drafter pool) get different chunk sizes.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch.disaggregation.native import transfer


@pytest.mark.cpu_only
def test_each_pool_is_registered_on_its_own():
    pools = [(0x1000, 1 << 30, 0, "pool_0"), (0x80000000, 1 << 28, 0, "pool_1")]
    agent = mock.Mock()
    worker = SimpleNamespace(
        _rank_info=SimpleNamespace(page_table=object(), device_id=0),
        _agent=agent,
        _registered_mem=[],
    )
    with mock.patch.object(transfer, "get_unique_pool_memory_descs", return_value=pools):
        transfer.TransferWorker._register_kv_cache(worker)
    registered = [call.args[0] for call in agent.register_memory.call_args_list]
    assert [r.descs for r in registered] == [[pool] for pool in pools]
    assert worker._registered_mem == registered


@pytest.mark.cpu_only
def test_no_pools_registers_nothing():
    agent = mock.Mock()
    worker = SimpleNamespace(
        _rank_info=SimpleNamespace(page_table=object(), device_id=0),
        _agent=agent,
        _registered_mem=[],
    )
    with mock.patch.object(transfer, "get_unique_pool_memory_descs", return_value=[]):
        transfer.TransferWorker._register_kv_cache(worker)
    agent.register_memory.assert_not_called()
    assert worker._registered_mem == []

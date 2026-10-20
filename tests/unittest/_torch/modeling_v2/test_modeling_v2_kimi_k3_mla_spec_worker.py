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
"""The MLA DSpark drafter's shardable logits processor (spec_worker._K3ShardableDraftLogits) on CPU: it can replace a
model's child-module processor, calls through to the processor unless its worker shards the step, returns the head
shard (``lm_head_shard``) when it does, and forwards the processor's other attributes."""

from types import SimpleNamespace

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    spec_worker,
)


class _Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gemvs = "gemvs"
        self.calls = []

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        self.calls.append(("full", return_context_logits))
        return hidden_states.float() * 2

    def lm_head_shard(self, rows, lm_head):
        self.calls.append(("shard",))
        return rows[:, :2]


class _Drafter(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.logits_processor = processor


def test_shardable_draft_logits_wraps_a_child_module():
    processor = _Processor()
    drafter = _Drafter(processor)
    worker = SimpleNamespace(_k3_sharded_block_logits=False)
    wrapper = spec_worker._K3ShardableDraftLogits(processor, worker)
    drafter.logits_processor = wrapper  # a child module: must be an nn.Module
    rows = torch.ones(3, 4, dtype=torch.bfloat16)
    full = drafter.logits_processor(rows, None, None, True)
    assert full.dtype == torch.float32 and tuple(full.shape) == (3, 4)
    worker._k3_sharded_block_logits = True
    shard = drafter.logits_processor(rows, None, None, True)
    assert tuple(shard.shape) == (3, 2)
    assert processor.calls == [("full", True), ("shard",)]
    assert drafter.logits_processor.gemvs == "gemvs"
    assert "_k3_worker" not in dict(drafter.named_modules())


def test_mla_worker_wraps_once():
    processor = _Processor()
    drafter = _Drafter(processor)
    worker = spec_worker.KimiK3MLADSparkWorker.__new__(spec_worker.KimiK3MLADSparkWorker)
    worker.wrap_draft_logits(drafter)
    first = drafter.logits_processor
    worker.wrap_draft_logits(drafter)
    assert drafter.logits_processor is first
    assert isinstance(first, spec_worker._K3ShardableDraftLogits)

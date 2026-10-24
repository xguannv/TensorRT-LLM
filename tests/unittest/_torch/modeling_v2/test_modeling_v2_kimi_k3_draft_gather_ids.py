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
"""``TRTLLM_K3LL_DRAFT_GATHER_TABLE``: the MLA DSpark worker's draft gather ids
(``spec_worker.KimiK3MLADSparkWorker._draft_gather_ids``), one GPU. With the switch off, the stock ids and no table
view (the switch-off path unchanged). With it on, a view of the slot table built outside capture, equal to the stock ids (``DFlashWorker._draft_gather_ids``:
``dflash_draft_slot_ids`` clamped to the last row) for every request count, both slot conventions and the K3
drafter's block shapes; the stock path where an id would reach past the rows (the clamp is not the identity) or
before the table exists; inside a CUDA-graph capture the same table storage (no per-step index kernels); and the
block-logits sharding decision taken on the fast path as on the stock one."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags, spec_worker
from tensorrt_llm._torch.speculative.dflash import dflash_draft_slot_ids

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

MAX_GENS = 16


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFT_GATHER_TABLE", 1)


def _worker(sharded=False, max_gens=MAX_GENS):
    worker = spec_worker.KimiK3MLADSparkWorker.__new__(spec_worker.KimiK3MLADSparkWorker)
    if max_gens:
        worker._k3_max_gens = max_gens
    worker.k3_decode = sharded
    worker._k3_spec_metadata = SimpleNamespace() if sharded else None
    worker._k3_acceptance = object() if sharded else None
    worker._keep_draft_logits_sharded = lambda draft_model, spec_metadata, num_gens: True
    worker._k3_sharded_block_logits = False
    return worker


def _stock(num_gens, block_size, k, shift, rows):
    return dflash_draft_slot_ids(num_gens, block_size, k, shift, device="cuda").clamp(max=rows - 1)


@pytest.mark.parametrize("shift", [True, False])
@pytest.mark.parametrize("block_size,k", [(8, 7), (8, 6), (7, 7), (16, 7)])
def test_gather_ids_match_stock(shift, block_size, k):
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=shift)
    for num_gens in range(1, MAX_GENS + 1):
        for rows in (num_gens * block_size, num_gens * block_size + 3):
            got = worker._draft_gather_ids(drafter, num_gens, block_size, k, rows)
            assert torch.equal(got, _stock(num_gens, block_size, k, shift, rows)), (num_gens, rows)
    table = worker._k3_slot_table(drafter, block_size, k)
    assert table is not None and table.numel() == MAX_GENS * k


def test_overrun_and_oversize_take_the_stock_path():
    # block_size == K with shift_label off: the last request's last slot is past the rows, so the clamp acts.
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=False)
    table = worker._k3_slot_table(drafter, 7, 7)
    got = worker._draft_gather_ids(drafter, 3, 7, 7, 21)
    assert torch.equal(got, _stock(3, 7, 7, False, 21))
    assert got.data_ptr() != table.data_ptr()
    # More requests than the table holds.
    got = worker._draft_gather_ids(drafter, MAX_GENS + 2, 8, 7, (MAX_GENS + 2) * 8)
    assert torch.equal(got, _stock(MAX_GENS + 2, 8, 7, False, (MAX_GENS + 2) * 8))


def test_no_table_before_max_gens_is_known():
    worker = _worker(max_gens=0)
    drafter = SimpleNamespace(_dspark_shift_label=True)
    assert worker._k3_slot_table(drafter, 8, 7) is None
    got = worker._draft_gather_ids(drafter, 2, 8, 7, 16)
    assert torch.equal(got, _stock(2, 8, 7, True, 16))


def test_capture_reads_the_table():
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=True)
    table = worker._k3_slot_table(drafter, 8, 7)
    hidden = torch.randn(4 * 8, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(4 * 7, 64, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out.copy_(hidden[worker._draft_gather_ids(drafter, 4, 8, 7, 32)])  # warm-up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    seen = []
    with torch.cuda.graph(graph):
        ids = worker._draft_gather_ids(drafter, 4, 8, 7, 32)
        seen.append(ids.data_ptr())
        out.copy_(hidden[ids])
    assert seen == [table.data_ptr()]
    hidden.copy_(torch.randn_like(hidden))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, hidden[_stock(4, 8, 7, True, 32)])
    # A shape first seen during capture builds no table there and takes the stock path.
    assert (8, 6, True) not in worker._k3_slot_tables
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        worker._draft_gather_ids(drafter, 4, 8, 6, 32)
    assert (8, 6, True) not in worker._k3_slot_tables


@pytest.mark.parametrize("sharded", [True, False])
def test_fast_path_takes_the_sharding_decision(sharded):
    drafter = SimpleNamespace(_dspark_shift_label=True)
    fast, stock = _worker(sharded), _worker(sharded)
    fast._draft_gather_ids(drafter, 2, 8, 7, 16)
    stock._draft_slot_ids(drafter, 2, 8, 7)
    assert fast._k3_sharded_block_logits is stock._k3_sharded_block_logits is sharded


def test_switch_off_is_the_stock_path(monkeypatch):
    monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFT_GATHER_TABLE", 0)
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=True)
    for num_gens in (1, 2, 4):
        got = worker._draft_gather_ids(drafter, num_gens, 8, 7, num_gens * 8)
        assert torch.equal(got, _stock(num_gens, 8, 7, True, num_gens * 8))
    assert not worker.__dict__.get("_k3_slot_tables")

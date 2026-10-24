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
"""DWDP fabric gather: the op copies exactly the planned bytes, and its switch stays opt-in.

The kernel choice (TRTLLM_DWDP_GATHER_KERNEL) and cluster size are read once per process, so each
kernel is covered by running this file in a process with that variable set; within one process the
copies are checked against the source on one GPU (local addresses stand in for peer mappings).
"""

import os

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp import fabric_gather

needs_op = pytest.mark.skipif(
    not torch.cuda.is_available() or fabric_gather.gather_op() is None,
    reason="needs CUDA and trtllm::dwdp_fabric_gather",
)


def _plan(src: torch.Tensor, dst: torch.Tensor, pieces, rows: int):
    """``pieces`` = [(src_offset, dst_offset, size)] in bytes -> descriptor groups, ``rows`` per peer-like split."""
    descs = [(dst.data_ptr() + d, src.data_ptr() + s, n) for s, d, n in pieces]
    per = -(-len(descs) // rows)
    return [descs[i : i + per] for i in range(0, len(descs), per)]


@needs_op
@pytest.mark.parametrize("rows_env", ["0", "3"])
@pytest.mark.parametrize("blocks", [2, 16])
def test_gather_copies_exactly_the_planned_bytes(monkeypatch, rows_env, blocks):
    monkeypatch.setenv("TRTLLM_DWDP_GATHER_ROWS", rows_env)
    monkeypatch.setenv("TRTLLM_DWDP_GATHER_BLOCKS", str(blocks))
    torch.manual_seed(0)
    src = torch.randint(0, 256, (8 << 20,), dtype=torch.uint8, device="cuda")
    dst = torch.zeros(8 << 20, dtype=torch.uint8, device="cuda")
    # Uneven, 16-byte aligned pieces, out of order, with gaps left untouched in dst.
    pieces = [
        (0, 4096, 1 << 20),
        (3 << 20, 2 << 20, 48),
        (5 << 20, 0, 16),
        (6 << 20, 5 << 20, (1 << 20) + 16400),
        (2 << 20, 7 << 20, 65536),
    ]
    groups = _plan(src, dst, pieces, rows=4)
    plan = fabric_gather.build_plan(groups, src.device)
    assert plan is not None
    dsts, srcs, lines, row_starts, blocks_per_row = plan
    fabric_gather.gather_op()(dsts, srcs, lines, row_starts, blocks_per_row, fabric_gather.THREADS)
    torch.cuda.synchronize()
    expected = torch.zeros_like(dst)
    for s, d, n in pieces:
        expected[d : d + n] = src[s : s + n]
    assert torch.equal(dst, expected)


def test_unaligned_descriptor_falls_back():
    groups = [[(4096, 8192, 32)], [(4096 + 8, 0, 16)]]
    assert fabric_gather.build_plan(groups, torch.device("cpu")) is None


def test_switch_is_opt_in(monkeypatch):
    monkeypatch.delenv("TRTLLM_K3LL_DWDP_GATHER", raising=False)
    assert not fabric_gather.is_enabled()
    # Dongxu's switch name, which the K3 sweeps and harness already export, must not turn it on.
    monkeypatch.setenv("TRTLLM_DWDP_PREFETCH_GATHER", "1")
    assert not fabric_gather.is_enabled()
    monkeypatch.setenv("TRTLLM_K3LL_DWDP_GATHER", "1")
    assert fabric_gather.is_enabled()


def test_compute_side_defaults_keep_explicit_values(monkeypatch):
    names = [k for k, _ in fabric_gather.compute_side_defaults()]
    for k in names:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("TRTLLM_DWDP_GATHER_BLOCKS", raising=False)
    monkeypatch.setenv("TRTLLM_K3LL_DEEPGEMM_SM_RESERVE", "8")
    fabric_gather.apply_compute_side_defaults()
    assert os.environ["TRTLLM_K3LL_CUTEDSL_SM_RESERVE"] == str(fabric_gather.TARGET_BLOCKS)
    assert os.environ["TRTLLM_K3LL_DEEPGEMM_SM_RESERVE"] == "8"
    assert os.environ["TRTLLM_K3LL_KDA_PREFILL_SM_RESERVE"] == "1"
    assert os.environ["TRTLLM_K3LL_ROUTING_DISABLE_COOP_LAUNCH"] == "1"
    assert os.environ["TRTLLM_DWDP_GATHER_KERNEL"] == "tma"


def test_sm_reserve_hook_reads_the_k3ll_name(monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels import sm_reserve

    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", "16")
    monkeypatch.delenv("TRTLLM_K3LL_CUTEDSL_SM_RESERVE", raising=False)
    assert sm_reserve() == 0
    monkeypatch.setenv("TRTLLM_K3LL_CUTEDSL_SM_RESERVE", "16")
    assert sm_reserve() == 16

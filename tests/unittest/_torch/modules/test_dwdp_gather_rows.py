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
"""DWDP fabric-gather plan re-cut into fewer grid rows (TRTLLM_DWDP_GATHER_ROWS)."""

import pytest

from tensorrt_llm._torch.modules.dwdp.fabric_gather import ROW_CUT_BYTES, regroup_rows

MB = 1 << 20


def _peer_groups(num_peers=15, weights=(11 * MB + 16 * 1000, 5 * MB + 48)):
    groups = []
    dst = 1 << 40
    for p in range(num_peers):
        src = (p + 1) << 36
        descs = []
        for w in weights:
            descs.append((dst, src, w))
            dst += w
            src += w
        groups.append(descs)
    return groups


def _bytes_map(groups):
    """dst byte ranges -> src byte ranges, merged, to compare two plans."""
    pieces = sorted((d, s, n) for g in groups for d, s, n in g)
    merged = []
    for d, s, n in pieces:
        if merged and merged[-1][0] + merged[-1][2] == d and merged[-1][1] + merged[-1][2] == s:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + n)
        else:
            merged.append((d, s, n))
    return merged


@pytest.mark.parametrize("rows", [1, 4, 7, 8])
def test_regroup_keeps_every_byte_and_balances(rows):
    groups = _peer_groups()
    out = regroup_rows(groups, rows)
    assert len(out) == rows
    assert _bytes_map(out) == _bytes_map(groups)
    sizes = [sum(n for _, _, n in g) for g in out]
    total = sum(sizes)
    per_row = -(-(-(-total // rows)) // ROW_CUT_BYTES) * ROW_CUT_BYTES
    assert all(s <= per_row for s in sizes)
    assert all(s == per_row for s in sizes[:-1])
    for g in out:
        for d, s, n in g:
            assert d % 16 == 0 and s % 16 == 0 and n % 16 == 0 and n > 0


@pytest.mark.parametrize("rows", [0, 15, 16, 30])
def test_regroup_noop_when_not_fewer_rows(rows):
    groups = _peer_groups()
    assert regroup_rows(groups, rows) is groups

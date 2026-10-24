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
"""``TRTLLM_K3LL_MLA_KV_JOIN_EARLY`` (``kimi_k3_nvfp4/flags.py``): with ``TRTLLM_K3LL_MLA_KV_SIDE``, the main stream
joins the KV side stream right before the query absorption instead of before the split read. On the stand-in module
and the paging cache kernel of ``test_modeling_v2_kimi_k3_mla_kv_side`` (the append lands after a ~20 us device
sleep), 8 / 16 / 32 rows:

* switch on = KV_SIDE off, eagerly: output, the latent row the cache kernel received, the query bytes the split read
  received and the pages read right after the layer, bit for bit; the cache kernel ran on ``kv_side_stream``; the
  census key ``mla_kv_join_early`` counts one per forking forward (``mla_kv_side`` too);
* a captured layer plus a read of the pages right after it, replayed with new inputs, equals the off arm's eager pages;
* with KV_SIDE at 0 the switch changes nothing and counts nothing.
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_modeling_v2_kimi_k3_mla_kv_side as kvs  # noqa: E402

pytestmark = kvs.pytestmark

ROWS = kvs.ROWS
mla = kvs.mla
census = kvs.census


@contextlib.contextmanager
def _switches(monkeypatch, kv_side, join_early):
    flags = kvs._pkg().flags
    with monkeypatch.context() as m:
        values = {
            **kvs.TRUNK_DEFAULTS,
            "TRTLLM_K3LL_MLA_KV_SIDE": kv_side,
            "TRTLLM_K3LL_MLA_KV_JOIN_EARLY": join_early,
        }
        for name, value in values.items():
            m.setattr(flags, name, int(value))
        yield


def _joins():
    return sum(v for k, v in kvs._census().counts().items() if k.startswith("mla_kv_join_early:"))


def _arm(mla, monkeypatch, kv_side, join_early, x, meta):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    qs = kvs._qs()
    mla.mqa.pages.zero_()
    torch.cuda.synchronize()
    mla.mqa.streams.clear()
    with _switches(monkeypatch, kv_side, join_early), with_multi_stream(True):
        mla.record.clear()
        out = qs._decode(mla)(x, meta)
        pages = mla.mqa.pages.clone()
        torch.cuda.synchronize()
    order = [r[0] for r in mla.record]
    rope = [r for r in mla.record if r[0] == "rope"]
    split = [r for r in mla.record if r[0] == "split"]
    assert len(rope) == 1 and len(split) == 1
    return out, rope[0], split[0], pages, list(mla.mqa.streams), order


@pytest.mark.parametrize("rows", ROWS)
def test_join_early_equals_off_eager(mla, monkeypatch, census, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )

    qs = kvs._qs()
    meta = qs._meta(rows)
    for seed in (20, 21):
        x = qs._x(rows, seed + rows)
        joins = _joins()
        off, off_rope, off_split, off_pages, _, _ = _arm(mla, monkeypatch, 0, 0, x, meta)
        assert _joins() == joins
        on, on_rope, on_split, on_pages, on_streams, _ = _arm(mla, monkeypatch, 1, 1, x, meta)
        assert _joins() == joins + 1
        assert on_streams == [decode_mla_q.kv_side_stream(x.device)]
        assert kvs._bits(on, off)
        assert kvs._bits(on_rope[1], off_rope[1]) and kvs._bits(on_split[1], off_split[1])
        assert kvs._bits(on_pages, off_pages) and bool(on_pages[:rows].abs().sum() > 0)


@pytest.mark.parametrize("rows", ROWS)
def test_join_early_graph_replay(mla, monkeypatch, census, rows):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    qs = kvs._qs()
    meta = qs._meta(rows)
    x = qs._x(rows, 2)
    with _switches(monkeypatch, 1, 1), with_multi_stream(True):
        qs._decode(mla)(x, meta)  # compiles every kernel outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured = kvs._census().counts().get("mla_kv_join_early:capture", 0)
        with torch.cuda.graph(graph):
            out = qs._decode(mla)(x, meta)
            pages = mla.mqa.pages.clone()
    assert kvs._census().counts().get("mla_kv_join_early:capture", 0) == captured + 1
    for replay in range(kvs.REPLAYS):
        x.copy_(qs._x(rows, 400 + replay))
        mla.mqa.pages.zero_()
        pages.fill_(0)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed_out, replayed_pages = out.clone(), pages.clone()
        ref, _, _, ref_pages, _, _ = _arm(mla, monkeypatch, 0, 0, x.clone(), meta)
        assert kvs._bits(replayed_out, ref)
        assert kvs._bits(replayed_pages, ref_pages) and bool(ref_pages[:rows].abs().sum() > 0)


@pytest.mark.parametrize("rows", ROWS)
def test_join_early_without_kv_side_is_a_no_op(mla, monkeypatch, census, rows):
    qs = kvs._qs()
    meta = qs._meta(rows)
    x = qs._x(rows, 30 + rows)
    joins = _joins()
    off, off_rope, off_split, off_pages, off_streams, _ = _arm(mla, monkeypatch, 0, 0, x, meta)
    on, on_rope, on_split, on_pages, on_streams, _ = _arm(mla, monkeypatch, 0, 1, x, meta)
    assert _joins() == joins
    assert on_streams == off_streams
    assert kvs._bits(on, off) and kvs._bits(on_pages, off_pages)
    assert kvs._bits(on_rope[1], off_rope[1]) and kvs._bits(on_split[1], off_split[1])

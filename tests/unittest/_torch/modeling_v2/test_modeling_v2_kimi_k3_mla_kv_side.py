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
"""``TRTLLM_K3LL_MLA_KV_SIDE`` (``kimi_k3_nvfp4/flags.py``): in the Helix all-head MLA decode, the latent row and the KV
append on their own stream (``decode_mla_q.kv_side_stream``) beside the query path, joined by the main stream before
the split read. One GPU, ``K3DecodeMLA._quantized_decode`` on the stand-in module of
``test_modeling_v2_kimi_k3_mla_q_switches`` at the trunk defaults, 8 / 16 / 32 rows (1 / 2 / 4 requests); 40 / 64 rows
(5 / 8 requests) for the request-count bound.

The stand-in cache kernel here also appends: it writes the latent row into a page buffer that stands in for the KV
pages, after a ~20 us device sleep, so a reader that is not ordered after the append sees the old pages.

* switch on = switch off, eagerly with the side stream on: output, the latent row the cache kernel received, the query
  bytes the split read received and the pages, bit for bit; the cache kernel ran on ``kv_side_stream`` with the switch
  on and on the main stream with it off;
* the pages read right after the layer: the layer and a copy of the pages captured in one CUDA graph (side stream on),
  replayed with new inputs, gives the off arm's eager pages bit for bit (the append completes before every later
  reader on the main stream, not just the split read);
* from 5 requests (``decode_mla_q.KV_SIDE_MAX_REQUESTS`` is 4) the switch at 1 runs the cache kernel on the main
  stream, and the output, records and pages equal the switch at 0 bit for bit;
* the instrument fires: with the main stream's join removed, the same eager read right after the layer sees pages
  that differ from the appended ones (so the check above would catch a missing join).
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="k3_fp8w_lowm_gemm and K3MlaQAbsorb run on SM100 / SM103",
)

ROWS = (8, 16, 32)
ROWS_ABOVE = (40, 64)
REPLAYS = 2
SLEEP_CYCLES = 40_000  # ~20 us at GB300 clocks
# Negative control: ~25 ms, longer than the main stream's eager launches of the rest of the layer, so the append
# lands after a read that does not wait for it.
SLOW_SLEEP_CYCLES = 50_000_000
TRUNK_DEFAULTS = {
    "TRTLLM_K3LL_HELIX_Q_ABSORB": 1,
    "TRTLLM_K3LL_MLA_GATE_SIDE": 1,
    "TRTLLM_K3LL_FP8W_LOWM_FUSED_A": 1,
    "TRTLLM_K3LL_MLA_QKV_PDL": 0,
    "TRTLLM_K3LL_MLA_DECODE_PDL": 1,
    "TRTLLM_K3LL_MLA_QNORM_PDL": 0,
    "TRTLLM_K3LL_MLA_PREWAIT_L2": 0,
}


def _qs():
    import test_modeling_v2_kimi_k3_mla_q_switches as qs

    return qs


def _pkg():
    return _qs()._pkg()


class _PagingBackend:
    """The stand-in cache kernel (``q_switches._Backend``) that also appends: the latent row goes into ``pages`` after
    a device sleep, on whatever stream the decode enqueues it."""

    def __init__(self, inner, rows_max):
        self.inner = inner
        self.num_heads, self.layer_idx = inner.num_heads, inner.layer_idx
        self.has_fp8_kv_cache, self.has_fp4_kv_cache = True, False
        self.pages = torch.zeros(rows_max, 576, dtype=torch.bfloat16, device="cuda")
        self.streams = []
        self.sleep_cycles = SLEEP_CYCLES

    def mla_rope_generation(self, fused_q, q_pe, latent, *args, **kwargs):
        self.streams.append(torch.cuda.current_stream())
        torch.cuda._sleep(self.sleep_cycles)
        self.pages[: latent.shape[0]].copy_(latent.view(latent.shape[0], -1)[:, :576])
        return self.inner.mla_rope_generation(fused_q, q_pe, latent, *args, **kwargs)


@pytest.fixture(scope="module")
def mla():
    torch.manual_seed(0)
    qs = _qs()
    module = qs._standin(qs._random_weights(torch.Generator(device="cuda").manual_seed(1234)))
    module.mqa = _PagingBackend(module.mqa, max(ROWS + ROWS_ABOVE))
    return module


@contextlib.contextmanager
def _switches(monkeypatch, kv_side):
    flags = _pkg().flags
    with monkeypatch.context() as m:
        for name, value in {**TRUNK_DEFAULTS, "TRTLLM_K3LL_MLA_KV_SIDE": kv_side}.items():
            m.setattr(flags, name, int(value))
        yield


def _bits(a, b) -> bool:
    return _qs()._bits(a, b)


def _arm(mla, monkeypatch, kv_side, x, meta, read_pages=True):
    """Output, rope record, split record, the pages read on the main stream right after the layer, and the streams
    the cache kernel ran on."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    qs = _qs()
    mla.mqa.pages.zero_()
    torch.cuda.synchronize()
    mla.mqa.streams.clear()
    with _switches(monkeypatch, kv_side), with_multi_stream(True):
        mla.record.clear()
        out = qs._decode(mla)(x, meta)
        pages = mla.mqa.pages.clone() if read_pages else None
        torch.cuda.synchronize()
    rope = [r for r in mla.record if r[0] == "rope"]
    split = [r for r in mla.record if r[0] == "split"]
    assert len(rope) == 1 and len(split) == 1
    return out, rope[0], split[0], pages, list(mla.mqa.streams)


@pytest.mark.parametrize("rows", ROWS)
def test_kv_side_equals_off_eager(mla, monkeypatch, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )

    qs = _qs()
    meta = qs._meta(rows)
    for seed in (10, 11):
        x = qs._x(rows, seed + rows)
        off, off_rope, off_split, off_pages, off_streams = _arm(mla, monkeypatch, 0, x, meta)
        on, on_rope, on_split, on_pages, on_streams = _arm(mla, monkeypatch, 1, x, meta)
        assert (
            on_split[2] is True and on_rope[2] is True
        )  # the all-head path, the cache kernel with q_rope_applied
        side = decode_mla_q.kv_side_stream(x.device)
        assert on_streams == [side] and off_streams and off_streams[0] != side
        assert _bits(on, off)
        assert _bits(on_rope[1], off_rope[1]) and _bits(on_split[1], off_split[1])
        assert _bits(on_pages, off_pages) and bool(on_pages[:rows].abs().sum() > 0)


@pytest.mark.parametrize("rows", ROWS)
def test_pages_read_after_the_layer_graph_replay(mla, monkeypatch, rows):
    """The layer and a read of the pages right after it, captured together with the switch on, replayed with new
    inputs: the read sees the appended pages (the off arm's eager ones), bit for bit."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    qs = _qs()
    meta = qs._meta(rows)
    x = qs._x(rows, 1)
    with _switches(monkeypatch, 1), with_multi_stream(True):
        qs._decode(mla)(x, meta)  # compiles every kernel outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = qs._decode(mla)(x, meta)
            pages = mla.mqa.pages.clone()
    for replay in range(REPLAYS):
        x.copy_(qs._x(rows, 300 + replay))
        mla.mqa.pages.zero_()
        pages.fill_(0)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed_out, replayed_pages = out.clone(), pages.clone()
        ref, _, _, ref_pages, _ = _arm(mla, monkeypatch, 0, x.clone(), meta)
        assert _bits(replayed_out, ref)
        assert _bits(replayed_pages, ref_pages) and bool(ref_pages[:rows].abs().sum() > 0)


def test_kv_side_applies_to_one_to_four_requests():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )

    assert decode_mla_q.KV_SIDE_MAX_REQUESTS == 4
    assert [n for n in range(0, 10) if decode_mla_q.kv_side_applies(n)] == [1, 2, 3, 4]


@pytest.mark.parametrize("rows", ROWS_ABOVE)
def test_kv_side_main_stream_from_five_requests(mla, monkeypatch, rows):
    """5 and 8 requests with the switch at 1: the cache kernel stays on the main stream (no side stream, no join), and
    everything equals the switch at 0."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )

    qs = _qs()
    meta = qs._meta(rows)
    assert not decode_mla_q.kv_side_applies(meta.num_generations)
    x = qs._x(rows, 50 + rows)
    off, off_rope, off_split, off_pages, off_streams = _arm(mla, monkeypatch, 0, x, meta)
    on, on_rope, on_split, on_pages, on_streams = _arm(mla, monkeypatch, 1, x, meta)
    side = decode_mla_q.kv_side_stream(x.device)
    assert on_streams == off_streams and len(on_streams) == 1 and on_streams[0] != side
    assert _bits(on, off)
    assert _bits(on_rope[1], off_rope[1]) and _bits(on_split[1], off_split[1])
    assert _bits(on_pages, off_pages) and bool(on_pages[:rows].abs().sum() > 0)


def test_missing_join_is_seen(mla, monkeypatch):
    """Negative control: with ``KvSide.join`` not waiting, the read right after the layer sees the pages before the
    append (zeros), so the page checks above detect a missing join."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )

    qs = _qs()
    rows = 32
    meta = qs._meta(rows)
    x = qs._x(rows, 7)
    _, _, _, ref_pages, _ = _arm(mla, monkeypatch, 0, x, meta)

    def join_without_wait(self):
        self._marked = self._side = False

    with monkeypatch.context() as m:
        m.setattr(decode_mla_q.KvSide, "join", join_without_wait)
        m.setattr(mla.mqa, "sleep_cycles", SLOW_SLEEP_CYCLES)
        _, _, _, early_pages, streams = _arm(mla, monkeypatch, 1, x, meta)
    assert streams == [decode_mla_q.kv_side_stream(x.device)]
    assert not _bits(early_pages, ref_pages)

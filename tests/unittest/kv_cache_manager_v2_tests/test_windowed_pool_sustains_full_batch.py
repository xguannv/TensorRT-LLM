# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A windowed pool group pinned at its constraint floor must sustain a full batch.

Constraint floors are derived from declared batches, and the declared batch for
CUDA-graph generation warmup is one long request plus `max_batch_size - 1`
minimal decode requests of a few tokens each. Charging each of those its
instantaneous demand gives a windowed life cycle one block per request, so the
floor lands a hair above `max_batch_size` -- and a windowed life cycle needs one
*or two* blocks per request depending on where its window currently sits
relative to a block boundary. Steady state therefore needs more slots than the
floor reserves, and since the resume gate takes the maximum utilization over
pool groups, the smallest group turns that shortfall into a refusal for the
whole worker while the token-scaled groups beside it are nearly empty.

Reported against DeepSeek-V4-Pro on a disaggregated generation worker: with
`max_batch_size=128` only ~110-117 requests per rank could be activated, the
executor reported insufficient KV cache, and aggregate utilization was ~13%.
The binding pool group held 136 slots, which is `ceil((128 + 1) / 0.95)`.

These tests are storage-manager level: one windowed life cycle, one token-scaled
one, no model and no attention kernels. The compressor state of a
heterogeneous-attention model is one instance of the shape, not the cause.
"""

import math
import os
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from typing import TYPE_CHECKING

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BatchDesc,
        BufferConfig,
        GpuCacheTierConfig,
        KVCacheDesc,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        _introspection,
        _KVCache,
    )
    from kv_cache_manager_v2._common import GPU_LEVEL
    from kv_cache_manager_v2._utils import CachedCudaStream, init_cuda_once, temporary_sys_path

    with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
        from fake_engine import Role

_HAS_V2 = find_spec("kv_cache_manager_v2") is not None

TOKENS_PER_BLOCK = 32
WINDOW = 32
MAX_UTIL_FOR_RESUME = 0.95
MAX_SEQ_LEN = 2048
# `1 + max_draft_len + num_extra_kv_tokens` in the executor: a few tokens, well
# under one block, which is what makes the warmup batch charge one block each.
MIN_DECODE_CAPACITY = 8


def _pinned_quota(max_batch_size: int) -> int:
    """A quota that covers both floors but leaves the windowed group pinned.

    The windowed life cycle holds one or two blocks per request while the
    token-scaled one holds a whole sequence, so the byte ratio hands the
    windowed group a few percent of the quota -- well under its constraint
    floor. It is therefore pinned at that floor for any quota the manager
    accepts, and the floor is what decides admission. The quota still has to
    clear the sum of both floors, or construction raises before the question is
    ever asked.
    """
    return ((8 * max_batch_size) + 128) << 20


def _warmup_constraint_batch(max_batch_size: int) -> "BatchDesc":
    """The batch the executor declares for CUDA-graph generation warmup."""
    return BatchDesc(
        kv_caches=[KVCacheDesc(capacity=MAX_SEQ_LEN, history_length=MAX_SEQ_LEN - 1)]
        + [KVCacheDesc(capacity=MIN_DECODE_CAPACITY, history_length=0)] * (max_batch_size - 1)
    )


def _make_config(max_batch_size: int, gpu_quota: int) -> "KVCacheManagerConfig":
    """One windowed life cycle and one token-scaled one, in separate pool groups.

    The buffer sizes differ so the two cannot coalesce, and they are comparable
    so the token-scaled group carries enough byte weight to squeeze the windowed
    group down to its floor.
    """
    return KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
        max_util_for_resume=MAX_UTIL_FOR_RESUME,
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=(1 << 20) + 1)],
                sliding_window_size=WINDOW,
                num_sink_tokens=0,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(role=Role.KEY, size=1 << 20)],
                sliding_window_size=None,
            ),
        ],
        # Steady state, so the ratio model reflects requests that have been
        # resident for a while rather than a single fresh context.
        typical_step=BatchDesc(
            kv_caches=[KVCacheDesc(capacity=MAX_SEQ_LEN, history_length=0)]
            + [KVCacheDesc(capacity=MAX_SEQ_LEN, history_length=MAX_SEQ_LEN - 4)]
            * (max_batch_size - 1)
        ),
        constraints=[_warmup_constraint_batch(max_batch_size)],
    )


SUSTAIN_ENV = "TLLM_KV_CACHE_MANAGER_V2_SUSTAIN_WINDOWED_FLOOR"


@contextmanager
def _sustain_windowed_floor(enabled: bool):
    """Select the sizing rule for the manager constructed inside the block.

    The rule is read once, while sizing, so scoping the variable to construction
    is enough -- and keeps the two arms comparable within a single process.
    """
    previous = os.environ.get(SUSTAIN_ENV)
    os.environ[SUSTAIN_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            del os.environ[SUSTAIN_ENV]
        else:
            os.environ[SUSTAIN_ENV] = previous


class _Batch:
    """Drive `max_batch_size` long-lived requests through the resume gate."""

    def __init__(
        self, max_batch_size: int, gpu_quota: int | None = None, sustain: bool = True
    ) -> None:
        gpu_quota = _pinned_quota(max_batch_size) if gpu_quota is None else gpu_quota
        init_cuda_once()
        self.max_batch_size = max_batch_size
        with _sustain_windowed_floor(sustain):
            self.manager = KVCacheManager(_make_config(max_batch_size, gpu_quota))
        self._stream_holder = CachedCudaStream()
        self.caches: list = []

    def close(self) -> None:
        for kv_cache in self.caches:
            if kv_cache.status != _KVCache.Status.CLOSED:
                kv_cache.close()
        self.manager.shutdown()

    def admit(self) -> int:
        """Resume a batch of generating requests; returns how many got in.

        Each request is given a history almost as long as its capacity, which is
        what makes it a *generating* request: a windowed life cycle only drops
        blocks once history moves past them, so a request resized without
        advancing history holds its whole capacity and never exercises the
        window at all.

        Lengths are staggered on purpose. In the field each request sits at a
        different point in its sequence, so across the batch the windowed life
        cycle's live block count spans both phases of the window slide. One
        uniform length would put every request in the same phase and hide the
        effect this test is about.
        """
        stream = self._stream_holder.handle
        admitted = 0
        for i in range(self.max_batch_size):
            kv_cache = self.manager.create_kv_cache()
            self.caches.append(kv_cache)
            if not kv_cache.resume(stream):
                break
            length = min(TOKENS_PER_BLOCK * (4 + (i % 2)) + (i % TOKENS_PER_BLOCK), MAX_SEQ_LEN)
            if not kv_cache.resize(length, history_length=length - 1):
                break
            admitted += 1
        return admitted

    def pool_slots(self) -> list:
        return [stat.total for stat in _introspection.storage_statistics(self.manager, GPU_LEVEL)]


@unittest.skipUnless(_HAS_V2, "kv_cache_manager_v2 is not importable")
class TestWindowedPoolSustainsFullBatch(unittest.TestCase):
    def test_floor_budgets_more_than_one_slot_per_request(self) -> None:
        """The floor has to cover the slide, not just the declared instant.

        Before the fix the windowed group was sized at roughly
        `ceil((max_batch_size + 1) / max_util_for_resume)`, i.e. about one slot
        per request with a few percent of slack.
        """
        for max_batch_size in (8, 16, 32):
            with self.subTest(max_batch_size=max_batch_size):
                batch = _Batch(max_batch_size)
                try:
                    windowed = min(batch.pool_slots())
                    naive_floor = math.ceil((max_batch_size + 1) / MAX_UTIL_FOR_RESUME)
                    self.assertGreater(
                        windowed,
                        naive_floor,
                        f"windowed pool got {windowed} slots for {max_batch_size} "
                        f"requests, which is still the instantaneous-demand floor",
                    )
                finally:
                    batch.close()

    def test_full_batch_is_admitted(self) -> None:
        """The declared batch size has to be reachable, at every size."""
        for max_batch_size in (8, 16, 32):
            with self.subTest(max_batch_size=max_batch_size):
                batch = _Batch(max_batch_size)
                try:
                    self.assertEqual(batch.admit(), max_batch_size)
                finally:
                    batch.close()

    def test_generous_quota_also_admits_the_full_batch(self) -> None:
        """Control: the pinned regime is not the only one that has to work."""
        batch = _Batch(16, gpu_quota=1 << 30)
        try:
            self.assertEqual(batch.admit(), 16)
        finally:
            batch.close()

    def test_instantaneous_sizing_still_refuses_the_full_batch(self) -> None:
        """The opt-out reproduces the defect, which is what makes it an A/B.

        Keeping the reported behavior reachable from one environment variable
        means a field report can be confirmed or excluded on the spot, without a
        rebuild, and it keeps this test honest about what the fix changed.
        """
        for max_batch_size in (16, 32):
            with self.subTest(max_batch_size=max_batch_size):
                sustained = _Batch(max_batch_size)
                try:
                    sustained_slots = min(sustained.pool_slots())
                finally:
                    sustained.close()

                batch = _Batch(max_batch_size, sustain=False)
                try:
                    # Compared against the other arm rather than against a
                    # closed form, so pool granularity rounding cannot make this
                    # assertion about something other than the sizing rule.
                    self.assertLess(min(batch.pool_slots()), sustained_slots)
                    self.assertLess(batch.admit(), max_batch_size)
                finally:
                    batch.close()


if __name__ == "__main__":
    unittest.main()

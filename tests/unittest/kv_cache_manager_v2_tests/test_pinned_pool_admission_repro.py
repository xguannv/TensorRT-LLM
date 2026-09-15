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
"""A pool group pinned at its constraint floor cannot admit max_batch_size requests.

Reported against DeepSeek-V4-Pro on a disaggregated GEN worker: with
`max_batch_size=128`, only ~110-117 requests per rank could be activated, and
the executor reported `Insufficient KV cache for gen-only benchmark mode`.
Aggregate KV utilization at that moment was ~13%; the binding resource was a
small auxiliary pool group holding the compressor state, sized at 136 slots.

The arithmetic behind that 136 is what these tests pin down, and it contains
no reference to the model, the parallelism, or the prompt length:

    min_slots = ceil(demand(constraint_batch) / max_util_for_resume)

`constraint_batch` is the one built for CUDA-graph generation warmup -- one
request at `max_seq_len` plus `max_batch_size - 1` *minimal decode* requests
of a few tokens each. For a windowed life cycle every one of those minimal
requests needs exactly one block, so the floor lands a hair above
`max_batch_size`. Steady state is a different batch: `max_batch_size`
*long-lived* requests, each holding one or two window blocks depending on
where its window currently sits relative to a block boundary. The floor was
dimensioned for the feasibility of one warmup batch and is then used as the
admission ceiling for a workload it never modelled.

Two things follow, and both are tested here:

1.  The shortfall is a *ratio*, not an absolute -- it does not depend on how
    many requests per rank the deployment happens to produce. DEP16 at batch
    128 and DEP32 at batch 128 fail identically per rank; the reported
    "DEP32 might work" is about requests per rank, not about the defect.
2.  It only bites while the pool group is *pinned* at the floor. A pool whose
    per-request cost is independent of sequence length has a naturally tiny
    byte share, so it is the one that gets pinned -- which is why an
    auxiliary pool fails while the token-scaled pool beside it is 87% empty.

These are storage-manager level tests: a windowed life cycle and a
token-scaled one, no model and no attention kernels. The DeepSeek-V4
compressor state is one instance of the shape, not the cause.
"""

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

    # `fake_engine` is a sibling module, not an installed package.
    with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
        from fake_engine import Role

_HAS_V2 = find_spec("kv_cache_manager_v2") is not None

# Mirrors the executor's own defaults so the numbers here can be compared
# against a real run without a translation step.
TOKENS_PER_BLOCK = 32
# A window exactly one block wide is the DeepSeek-V4 compressor shape, which is
# what the incident was, and it is also a blind spot: there "demand + 1" and
# "what the window spans" are the same number, so any test using only this
# window passes under either sizing rule and cannot defend the default.
WINDOW = 32
WIDE_WINDOW = 4 * TOKENS_PER_BLOCK
MAX_UTIL_FOR_RESUME = 0.95
MAX_SEQ_LEN = 2048
# `min_decode_capacity = 1 + max_draft_len + num_extra_kv_tokens` in
# tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py; any value well below
# one block reproduces the "one block per warmup request" accounting.
MIN_DECODE_CAPACITY = 8


def _warmup_constraint_batch(max_batch_size: int) -> "BatchDesc":
    """The batch the executor declares for CUDA-graph generation warmup.

    Copied in shape from `kv_cache_manager_v2.py`'s `constraints.append(...)`:
    one request at `max_seq_len` and `max_batch_size - 1` minimal decode
    requests. This is the batch whose demand becomes `min_slots`.
    """
    return BatchDesc(
        kv_caches=[KVCacheDesc(capacity=MAX_SEQ_LEN, history_length=MAX_SEQ_LEN - 1)]
        + [KVCacheDesc(capacity=MIN_DECODE_CAPACITY, history_length=0)] * (max_batch_size - 1)
    )


def _make_config(
    max_batch_size: int, gpu_quota: int, window: int = WINDOW
) -> "KVCacheManagerConfig":
    """Two life cycles: one windowed (the pinned pool), one token-scaled.

    The buffer sizes differ so the two cannot coalesce into a single pool
    group (`_storage/_config.py` merges only identical `slot_size_list`s).
    The windowed buffer is the large one for the same reason it is large in
    the field: its slot holds per-request state rather than per-token state.
    """
    return KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
        max_util_for_resume=MAX_UTIL_FOR_RESUME,
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=(1 << 20) + 1)],
                sliding_window_size=window,
                num_sink_tokens=0,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(role=Role.KEY, size=1 << 20)],
                sliding_window_size=None,
            ),
        ],
        # The ratio model must look like steady state. A single
        # `history_length=0` request leaves nothing stale, so the windowed
        # group would claim a full-length share and never reach its floor --
        # the regime the incident was in would not be reproduced at all.
        typical_step=BatchDesc(
            kv_caches=[KVCacheDesc(capacity=MAX_SEQ_LEN, history_length=0)]
            + [KVCacheDesc(capacity=MAX_SEQ_LEN, history_length=MAX_SEQ_LEN - 4)]
            * (max_batch_size - 1)
        ),
        constraints=[_warmup_constraint_batch(max_batch_size)],
    )


SUSTAIN_ENV = "TLLM_KV_CACHE_MANAGER_V2_SUSTAIN_WINDOWED_FLOOR"
MODE_OFF = "off"
MODE_WORST = "worst"

# A Gemma-shaped cache: sliding-window layers are the MAJORITY and the window is
# many blocks wide. The rest of this file uses the DeepSeek-V4 shape, where the
# windowed group is byte-tiny and its window is one block -- the shape where the
# new floor is nearly free however it is written. This is the opposite one, and
# therefore the one that has to demonstrate the bound.
#
# Ratios, not absolutes. 5 sliding layers per full-attention layer with a
# 32-block window is what Gemma3 and Gemma4 ship (both 5:1, window 512 or 1024
# at tokens_per_block 32); Gemma4 additionally cannot opt out of this manager,
# because `_util.py::_non_hybrid_kv_cache_manager_cls` forces V2 whenever
# `config_utils.is_gemma4_hybrid` holds. The per-layer byte size is scaled down
# to keep the test cheap -- only the ratio between the two groups and the
# window-to-block ratio decide which side of the band a quota lands on.
GEMMA_WINDOW = 32 * TOKENS_PER_BLOCK
GEMMA_SLIDING_LAYERS = 50
GEMMA_FULL_LAYERS = 10
GEMMA_BLOCK_BYTES_PER_LAYER = 8 << 10
# Tight: below the sum of the two floors under `worst`, so the rules diverge.
# Generous: far enough above that each group's ratio share covers its floor and
# neither rule is consulted. Both were checked against the measured band rather
# than picked -- a quota that drifts to the wrong side of it turns this test
# into one that measures nothing, which is how an earlier test in this file
# passed six arms while proving nothing.
GEMMA_TIGHT_QUOTA = 64 << 20
GEMMA_GENEROUS_QUOTA = 2 << 30


def _make_windowed_dominant_config(max_batch_size: int, gpu_quota: int) -> "KVCacheManagerConfig":
    """Sliding-window layers in the majority, against a full-attention minority."""
    layers = [
        AttentionLayerConfig(
            layer_id=LayerId(i),
            buffers=[BufferConfig(role=Role.KEY, size=GEMMA_BLOCK_BYTES_PER_LAYER)],
            sliding_window_size=GEMMA_WINDOW,
            num_sink_tokens=0,
        )
        for i in range(GEMMA_SLIDING_LAYERS)
    ] + [
        AttentionLayerConfig(
            layer_id=LayerId(GEMMA_SLIDING_LAYERS + i),
            buffers=[BufferConfig(role=Role.KEY, size=GEMMA_BLOCK_BYTES_PER_LAYER)],
            sliding_window_size=None,
        )
        for i in range(GEMMA_FULL_LAYERS)
    ]
    return KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
        max_util_for_resume=MAX_UTIL_FOR_RESUME,
        layers=layers,
        typical_step=BatchDesc(
            kv_caches=[KVCacheDesc(capacity=MAX_SEQ_LEN, history_length=MAX_SEQ_LEN - 4)]
            * max_batch_size
        ),
        constraints=[_warmup_constraint_batch(max_batch_size)],
    )


@contextmanager
def _floor_mode(mode: str | None):
    """Pick the sizing rule for the manager built inside the block.

    The rule is read once, while sizing, so scoping it to construction is
    enough. `None` leaves the variable unset so the block exercises whatever
    ships as the default -- naming the current default here would make those
    tests immune to a change of default, which is the one thing they must not be.
    """
    previous = os.environ.get(SUSTAIN_ENV)
    if mode is None:
        os.environ.pop(SUSTAIN_ENV, None)
    else:
        os.environ[SUSTAIN_ENV] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SUSTAIN_ENV, None)
        else:
            os.environ[SUSTAIN_ENV] = previous


class _AdmissionProbe:
    """Drive `max_batch_size` long-lived requests through the resume gate."""

    def __init__(
        self,
        max_batch_size: int,
        gpu_quota: int,
        mode: str | None = None,
        window: int = WINDOW,
    ) -> None:
        init_cuda_once()
        self.max_batch_size = max_batch_size
        self.window = window
        self.cfg = _make_config(max_batch_size, gpu_quota, window)
        with _floor_mode(mode):
            self.manager = KVCacheManager(self.cfg)
        self._stream_holder = CachedCudaStream()
        self.caches: list = []

    def close(self) -> None:
        for kv_cache in self.caches:
            if kv_cache.status != _KVCache.Status.CLOSED:
                kv_cache.close()
        self.manager.shutdown()

    def admit_batch(self, phase: int | None = None) -> int:
        """Resume requests until one is refused. Returns how many got in.

        `history_length` is what makes these *generating* requests. A windowed
        life cycle only drops a block once history moves past it, so a request
        resized without advancing history is never stale anywhere: it holds its
        whole capacity, exactly like a full-attention one, and the window never
        slides at all.

        This loop used to omit it while its own docstring claimed the live-block
        count "oscillates between one and two across the batch" -- describing a
        behaviour the code did not produce. Measured, the windowed pool cost
        about five blocks per request instead of one or two, and admission was
        capped by plain capacity rather than by the floor. That made the
        admission tests unable to show the defect either way.

        `phase=None` staggers lengths across a block boundary, which samples the
        phases of the slide roughly evenly. An integer pins every request to the
        same phase: admission is decided by the maximum utilization over pool
        groups, so the worst phase is what matters, and a live workload can drift
        into alignment on its own.
        """
        stream = self._stream_holder.handle
        admitted = 0
        for i in range(self.max_batch_size):
            kv_cache = self.manager.create_kv_cache()
            self.caches.append(kv_cache)
            if not kv_cache.resume(stream):
                break
            if phase is None:
                length = TOKENS_PER_BLOCK * (4 + (i % 2)) + (i % TOKENS_PER_BLOCK)
            else:
                length = 2 * self.window + phase
            length = min(length, MAX_SEQ_LEN)
            if not kv_cache.resize(length, history_length=length - 1):
                break
            admitted += 1
        return admitted

    def pool_slots(self) -> list:
        return [stat.total for stat in _introspection.storage_statistics(self.manager)]

    def utilizations(self) -> list:
        return list(_introspection.storage_utilization(self.manager, GPU_LEVEL))


@unittest.skipUnless(_HAS_V2, "kv_cache_manager_v2 is not importable")
class TestPinnedPoolAdmission(unittest.TestCase):
    """The floor is a warmup feasibility bound, used as a steady-state ceiling."""

    # Measured, not derived. The windowed group is only pinned at its floor
    # inside a band, and the band's two edges move apart with max_batch_size:
    # above it the group's ratio share exceeds the floor and it is not pinned;
    # below it the whole cache is full and the shortfall is plain exhaustion.
    # Swept on 2026-09-14 (`probe_pinned_band.py`, `off` arm), reading
    # slots-per-request and the MINIMUM utilization across pool groups:
    #
    #   B=8    16-64 MiB   /req 1.12  umin 0.34      unpins at 96 MiB
    #   B=16   16-192 MiB  /req 1.06  umin 0.57-0.28 unpins at 256 MiB
    #   B=32   192-256 MiB /req 1.09  umin 0.63-0.45 unpins at >=384 MiB
    #
    # (B=32 below 192 MiB reads /req 1.09 *and* umin 0.96 -- pinned and
    # exhausted at once, where the two failures cannot be told apart.)
    #
    # 8 MiB per declared request lands at 64/128/256, pinned in all three with
    # utilization far from the resume threshold and 1.5-2x of headroom to the
    # unpinning edge. An earlier version added a flat 128 MiB on top; at B=8
    # that pushed 64 to 192, past the unpinning edge, so the defect the file
    # exists to show simply did not occur.
    @staticmethod
    def pinned_quota(max_batch_size: int) -> int:
        return (8 * max_batch_size) << 20

    def assert_not_merely_exhausted(self, probe) -> None:
        """The shortfall must be one pinned group, not a full cache.

        Both look like "fewer requests than declared". Only the first is what
        this file is about, and the field's signature was precisely that the
        cache was 87% empty while admission stopped. Without this check a
        harness that has drifted out of the band produces a confident failure
        that describes something else.
        """
        umin = min(probe.utilizations())
        self.assertLess(
            umin,
            MAX_UTIL_FOR_RESUME,
            f"every pool group is at {umin:.2f} utilization, so the cache is "
            f"simply full -- this is exhaustion, not a pinned pool group, and "
            f"nothing here is evidence about the floor",
        )

    # The floor buys roughly one slot per declared request and nothing more:
    # the warmup batch it is derived from has `max_batch_size - 1` requests of
    # a few tokens each, and a windowed life cycle needs exactly one block for
    # each of those. Field value: 136 slots for max_batch_size 128, or 1.06
    # per request. Asserting a band rather than `ceil((B+1)/max_util)` keeps
    # the test on the property that matters -- the floor does not budget for
    # the window-slide oscillation -- instead of on grain-rounding details
    # that differ with slot size.
    SLOTS_PER_REQUEST_CEILING = 1.25

    def test_floor_gives_about_one_slot_per_request_without_the_fix(self) -> None:
        """`min_slots` tracks `max_batch_size`, not memory and not sequence length.

        Reproducing the sizing from `max_batch_size` alone is what rules out
        the model, the topology and the 32K prompts as causes.
        """
        for max_batch_size in (8, 16, 32):
            with self.subTest(max_batch_size=max_batch_size):
                probe = _AdmissionProbe(
                    max_batch_size, self.pinned_quota(max_batch_size), mode=MODE_OFF
                )
                try:
                    windowed = min(probe.pool_slots())
                    per_request = windowed / max_batch_size
                    self.assertGreaterEqual(
                        per_request,
                        1.0,
                        f"windowed pool got {windowed} slots for "
                        f"{max_batch_size} requests, which is below the "
                        f"declared floor -- the harness is not in the pinned "
                        f"regime this test is about",
                    )
                    self.assertLessEqual(
                        per_request,
                        self.SLOTS_PER_REQUEST_CEILING,
                        f"windowed pool got {windowed} slots for "
                        f"{max_batch_size} requests ({per_request:.2f} per "
                        f"request); the floor is supposed to budget about one",
                    )
                finally:
                    probe.close()

    def test_admission_falls_short_of_max_batch_size_without_the_fix(self) -> None:
        """The defect itself: a full batch cannot be made resident."""
        for max_batch_size in (8, 16, 32):
            with self.subTest(max_batch_size=max_batch_size):
                probe = _AdmissionProbe(
                    max_batch_size, self.pinned_quota(max_batch_size), mode=MODE_OFF
                )
                try:
                    admitted = probe.admit_batch()
                    self.assertLess(
                        admitted,
                        max_batch_size,
                        "a pool group pinned at the warmup floor should not "
                        "fit a full steady-state batch",
                    )
                    # The token-scaled pool beside it is nowhere near full --
                    # this is the "aggregate utilization was 13%" confusion.
                    self.assert_not_merely_exhausted(probe)
                finally:
                    probe.close()

    def test_shortfall_is_scale_invariant_without_the_fix(self) -> None:
        """Doubling the batch does not change the admitted *fraction*.

        The field report asked whether DEP32 would work where DEP16 did not.
        Per rank both run the same `max_batch_size`, and this is the test that
        says the answer cannot come from the parallelism: the floor and the
        demand scale together, so the shortfall ratio is flat.
        """
        fractions = []
        for max_batch_size in (8, 16, 32):
            probe = _AdmissionProbe(
                max_batch_size, self.pinned_quota(max_batch_size), mode=MODE_OFF
            )
            try:
                fractions.append(probe.admit_batch() / max_batch_size)
            finally:
                probe.close()
        spread = max(fractions) - min(fractions)
        self.assertLess(
            spread,
            0.15,
            f"admitted fraction should be roughly constant across batch sizes, got {fractions}",
        )

    def test_unpinned_pool_admits_the_full_batch(self) -> None:
        """Negative control: the floor is the cause, not the resume gate.

        With enough quota the windowed pool group escapes `min_slots` and is
        sized by its ratio share instead. If this also failed, the finding
        would be about the 0.95 threshold rather than about the floor, and
        raising `free_gpu_memory_fraction` would be the fix.
        """
        max_batch_size = 16
        probe = _AdmissionProbe(max_batch_size, 1 << 30, mode=MODE_OFF)
        try:
            self.assertEqual(probe.admit_batch(), max_batch_size)
        finally:
            probe.close()

    def test_pinned_pool_admits_the_full_batch_with_the_fix(self) -> None:
        """The same pinned quota, under whatever ships: the batch fits.

        This is the pair to `test_admission_falls_short_of_max_batch_size_
        without_the_fix`. Same configuration, same quota, only the sizing rule
        differs, so the two together say the floor was the cause rather than
        the memory.
        """
        for max_batch_size in (8, 16, 32):
            with self.subTest(max_batch_size=max_batch_size):
                probe = _AdmissionProbe(max_batch_size, self.pinned_quota(max_batch_size))
                try:
                    admitted = probe.admit_batch()
                    self.assert_not_merely_exhausted(probe)
                finally:
                    probe.close()
                self.assertEqual(
                    admitted,
                    max_batch_size,
                    f"admitted {admitted} of {max_batch_size} at the pinned "
                    f"quota; the floor is still short of steady state",
                )

    def test_the_shipping_rule_does_not_vary_with_the_phase_of_the_slide(self) -> None:
        """Admission must not depend on where the requests sit in their window.

        This is the test that justifies the default, so it compares the two
        rules rather than asserting an absolute count. An absolute count would
        really be asserting that the quota happens to sit in the band where the
        windowed group is pinned -- and that band moves with the window width,
        the batch size and the rule itself. Chasing it cost four wrong quotas
        here; the invariant underneath never moved.

        The invariant: charging a windowed request its demand at the history
        the descriptor happens to name leaves admission to rise and fall as the
        requests drift through the phases of the slide. Charging what the window
        spans is the residency itself, so it does not move with phase. Since the
        resume gate takes the maximum utilization over pool groups, a rule that
        varies is a rule that refuses intermittently under load -- and both arms
        here share one quota, so a quota that drifts moves both and cannot
        manufacture the result.

        The comparison arm is ``off``, the pre-fix rule, so this measures the
        change the MR actually makes. It was a third "straddle block only" rule
        until that rule was removed: measured, it admitted fewer than ``worst``
        at every phase and was bit-identical to ``off`` on a shape where
        windowed layers dominate, so it was neither correct nor cheaper.

        Uses WIDE_WINDOW: at a window one block wide the two rules are the same
        number and this can prove nothing.
        """
        max_batch_size = 16
        admitted = {}
        for mode in (MODE_OFF, MODE_WORST):
            counts = []
            for phase in range(TOKENS_PER_BLOCK):
                probe = _AdmissionProbe(
                    max_batch_size,
                    self.pinned_quota(max_batch_size),
                    mode=mode,
                    window=WIDE_WINDOW,
                )
                try:
                    counts.append(probe.admit_batch(phase=phase))
                    self.assert_not_merely_exhausted(probe)
                finally:
                    probe.close()
            admitted[mode] = counts

        worst_spread = max(admitted[MODE_WORST]) - min(admitted[MODE_WORST])
        off_spread = max(admitted[MODE_OFF]) - min(admitted[MODE_OFF])
        # Always report both arms. An earlier version asserted the shipping
        # rule's spread was exactly zero and printed only that arm when it
        # failed -- so a comparison test degraded into a one-sided one at the
        # moment the comparison was needed. (It also asserted the wrong thing:
        # zero spread was measured on a different shape, and carrying it over
        # as a universal is the same mistake as carrying over a quota.)
        detail = f"off={admitted[MODE_OFF]} worst={admitted[MODE_WORST]}"

        # First, that this configuration can tell the rules apart at all. If
        # the pre-fix rule does not vary with phase here, nothing below means
        # anything -- which is the silent pass that made the first version of
        # this test worthless when it ran at a one-block window.
        self.assertGreater(
            off_spread,
            0,
            f"the pre-fix rule did not vary with phase, so this configuration "
            f"cannot distinguish the two rules: {detail}",
        )
        # The shipping rule must be no more phase-sensitive than the pre-fix
        # one. This is the property that matters: the resume gate takes the
        # maximum utilization over pool groups, so a rule whose admission moves
        # with phase refuses intermittently as live requests drift. Measured on
        # SM107 this passes by equality (both spreads 1), not by a margin --
        # it was 1 <= 2 against the removed third rule. A shape change that
        # makes `worst` one step more phase-sensitive will trip it.
        self.assertLessEqual(
            worst_spread,
            off_spread,
            f"the shipping rule varies more across phases than the pre-fix one: {detail}",
        )
        # And its worst phase must be no worse. Both arms share one quota, so a
        # quota that has drifted moves both and cannot manufacture this.
        self.assertGreaterEqual(
            min(admitted[MODE_WORST]),
            min(admitted[MODE_OFF]),
            f"the pre-fix rule admitted more at its worst phase: {detail}",
        )

    def _windowed_dominant_slots(self, max_batch_size: int, quota: int, mode: str) -> list[int]:
        """Slot counts for the Gemma-shaped cache, sorted.

        Sorted because `storage_statistics` returns a bare list whose order is
        not the same under the two backends -- reading it positionally once
        produced a report with the windowed and full-attention groups swapped.
        Only set equality is needed here, so sorting sidesteps the question.
        """
        with _floor_mode(mode):
            manager = KVCacheManager(_make_windowed_dominant_config(max_batch_size, quota))
        try:
            return sorted(st.total for st in _introspection.storage_statistics(manager, GPU_LEVEL))
        finally:
            manager.shutdown()

    def test_the_rule_costs_nothing_where_the_floor_does_not_bind(self) -> None:
        """Outside the pinned band the new floor changes no allocation at all.

        This is the bound on the change's blast radius, and it is the claim
        every model other than the reported one rests on. The rest of this file
        measures the DeepSeek-V4 shape, where the windowed group is byte-tiny
        and its window is a single block; there the new floor is nearly free
        whatever happens. The shape that could actually cost something is the
        opposite one -- sliding-window layers in the majority, window many
        blocks wide -- so that is the shape this asserts on.

        `max(ratio share, floor)` gives the guarantee by construction: while the
        share is above both floors, neither rule is consulted. The test exists
        because "by construction" was the same reasoning that said the fix would
        be a no-op on this shape *inside* the band, and a measurement said
        otherwise -- there the allocation reaches 1.4-2.7x the configured quota
        and the full-attention group is starved to a few hundred tokens per
        request. That cost is real and is argued in the MR description; what
        must not happen is for it to leak outside the band, and nothing tested
        that.

        The tight-quota arm is a canary, not a second guarantee: it only
        establishes that this shape can tell the two rules apart, so that the
        generous-quota equality cannot pass by measuring nothing. Deliberately
        NOT asserting the overrun factor -- pinning that number would freeze the
        behaviour the MR describes as the open question.
        """
        for max_batch_size in (8, 16):
            with self.subTest(max_batch_size=max_batch_size):
                tight_off = self._windowed_dominant_slots(
                    max_batch_size, GEMMA_TIGHT_QUOTA, MODE_OFF
                )
                tight_worst = self._windowed_dominant_slots(
                    max_batch_size, GEMMA_TIGHT_QUOTA, MODE_WORST
                )
                self.assertNotEqual(
                    tight_off,
                    tight_worst,
                    f"the two rules allocate identically at the tight quota, so "
                    f"this shape cannot tell them apart and the equality below "
                    f"proves nothing: off={tight_off} worst={tight_worst}",
                )

                loose_off = self._windowed_dominant_slots(
                    max_batch_size, GEMMA_GENEROUS_QUOTA, MODE_OFF
                )
                loose_worst = self._windowed_dominant_slots(
                    max_batch_size, GEMMA_GENEROUS_QUOTA, MODE_WORST
                )
                self.assertEqual(
                    loose_off,
                    loose_worst,
                    f"the windowed floor changed the allocation on a shape "
                    f"whose ratio share already covers it, so the change is not "
                    f"confined to the pinned band: off={loose_off} "
                    f"worst={loose_worst}",
                )


if __name__ == "__main__":
    if os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp") == "python":
        print(
            "NOTE: running the python backend; the sizing arithmetic under "
            "test is mirrored line-for-line in storageManager.cpp"
        )
    unittest.main()

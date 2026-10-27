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
"""The tail-only draft pool of a TRTLLM_K3LL_DRAFTER_PROMPT_CTX context worker.

A context worker's paired draft pool keeps W plus at most one page per request (a
sliding-window pool of W + 1 tokens, ``drafter_prompt_tail`` = W). These tests run real
KVCacheManagerV2 target and draft pools through the real V2 scheduler's joint claim and check:

- tail eviction: pressure on the draft pool evicts drafter pages before any stored prompt's
  window, and never touches the target pool;
- the pairing rule: on a branch hit below the window start, the target reuses exactly what it
  reuses with a full-length draft pool (the draft starts with an absent prefix);
- window completeness: after the draft lost a stored prompt entirely, a longer prompt that
  extends it still gets every page of its drafter window, and the draft publishes only the
  pages it wrote.
"""

import gc

import pytest
import torch

from tensorrt_llm._torch.disaggregation.transceiver import drafter_window
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    DFlashDecodingConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

TPB = 32
WINDOW = 64  # the drafter prompt window W
TAIL = WINDOW + 1  # the tail pool's sliding window: W plus at most one page per request
MAX_SEQ_LEN = 512
# Fillers are one page long, so each one takes exactly one draft page when it runs.
FILLER_LEN = TPB
# The pool's page count follows from its byte quota; fillers stop as soon as eviction shows.
MAX_FILLERS = 1024


def _manager(*, is_draft: bool, window, max_tokens: int) -> KVCacheManagerV2:
    config = dict(max_tokens=max_tokens, enable_block_reuse=True)
    if window is not None:
        config["max_attention_window"] = [window]
    return KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(**config),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=TPB,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
        execution_stream=torch.cuda.Stream(),
        spec_config=DFlashDecodingConfig(max_draft_len=2, speculative_model="draft-model"),
        is_draft=is_draft,
        joint_kv_cache_reuse=True,
    )


class _Pair:
    """A joint target / draft pool pair behind the real V2 scheduler.

    ``tail``: the draft pool is the tail-only pool (window W + 1, ``drafter_prompt_tail`` = W, a
    small pool so fillers evict from it); otherwise a full-length draft pool as large as
    the target, so nothing is evicted from it.
    """

    def __init__(self, *, tail: bool):
        torch.cuda.init()
        self.target = _manager(is_draft=False, window=None, max_tokens=8192)
        self.draft = _manager(
            is_draft=True,
            window=TAIL if tail else None,
            max_tokens=512 if tail else 8192,
        )
        if tail:
            self.draft.drafter_prompt_tail = WINDOW
        self.scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=MAX_SEQ_LEN,
            kv_cache_manager=self.target,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            draft_kv_cache_manager=self.draft,
        )
        self._next_id = 1
        self._next_filler_token = 20000

    def close(self):
        for mgr in (self.target, self.draft):
            mgr._stream.synchronize()
            mgr.shutdown()
        gc.collect()
        torch.cuda.empty_cache()

    def request(self, tokens):
        req = LlmRequest(
            request_id=self._next_id,
            max_new_tokens=4,
            input_tokens=list(tokens),
            sampling_config=SamplingConfig(1),
            is_streaming=False,
        )
        self._next_id += 1
        return req

    def probes(self, tokens):
        """``(draft, target)`` reusable depths for a prompt, without claiming."""
        req = self.request(tokens)
        return self.draft.probe_context_reuse(req), self.target.probe_context_reuse(req)

    def prefill(self, tokens, *, keep=False):
        """Run a whole prompt through the joint claim, allocation and context update, as one
        chunk. Returns ``(request, target reuse)``; the request is freed unless ``keep``."""
        req = self.request(tokens)
        assert self.scheduler._prepare_context_pair(req)
        reused = req.context_current_position
        assert self.scheduler._try_allocate_context(req, req.context_remaining_length)
        req.move_to_next_context_chunk()
        batch = ScheduledRequests()
        batch.append_context_request(req)
        self.target.update_context_resources(batch)
        self.draft.update_context_resources(batch)
        assert req.context_remaining_length == 0
        if not keep:
            self.free(req)
        return req, reused

    def free(self, req):
        self.target.free_resources(req)
        self.draft.free_resources(req)

    def filler(self):
        """A one-page prompt no other prompt shares."""
        first = self._next_filler_token
        self._next_filler_token += FILLER_LEN
        self.prefill(range(first, first + FILLER_LEN))

    def evict_draft_until(self, done):
        """Run fillers until ``done()``; each takes one draft page, evicting at most one."""
        for _ in range(MAX_FILLERS):
            if done():
                return
            self.filler()
        raise AssertionError(f"the draft pool did not evict within {MAX_FILLERS} fillers")


@pytest.fixture
def pairs():
    made = []

    def make(tail):
        pair = _Pair(tail=tail)
        made.append(pair)
        return pair

    yield make
    for pair in made:
        pair.close()


def _draft_pages(pair, req):
    kv_cache = pair.draft.kv_cache_map[req.py_request_id]
    return [int(p) for p in kv_cache.get_aggregated_page_indices(0, valid_only=False)]


def test_tail_eviction_frees_draft_blocks_only(pairs):
    """Draft pressure evicts a stored prompt's pages before its window, and the target keeps all.

    The prompt's 6 drafter pages: 3 evicted first (blocks 0-2) and 3 kept for its endpoint
    window (blocks 3-5: the window of the prompt matched without its last token). One-page
    fillers take one draft page each, so the pool evicts one page at a time; the first eviction
    that reaches the prompt's prefix must leave its window whole, and the target's match is
    never affected.
    """
    pair = pairs(tail=True)
    prompt = list(range(6 * TPB))
    prefix_probe = prompt[: 2 * TPB] + list(range(10000, 10000 + TPB))
    req, _ = pair.prefill(prompt, keep=True)
    # While the request is live its draft holds only its window's pages.
    held = [p for p in _draft_pages(pair, req) if p != BAD_PAGE_INDEX]
    assert len(held) <= WINDOW // TPB + 1
    pair.free(req)

    draft_full, target_full = pair.probes(prompt)
    draft_prefix, target_prefix = pair.probes(prefix_probe)
    window_start, _ = drafter_window(len(prompt), WINDOW, TPB)
    assert draft_full >= window_start and target_full == draft_full
    assert draft_prefix == target_prefix == 2 * TPB

    pair.evict_draft_until(lambda: pair.probes(prefix_probe)[0] < 2 * TPB)

    # The window survived the eviction of the pages before it; the target lost nothing.
    assert pair.probes(prompt) == (draft_full, target_full)
    assert pair.probes(prefix_probe)[1] == target_prefix


@pytest.mark.parametrize("branch_blocks", [1, 3, 6])
def test_pairing_keeps_target_reuse_on_branch_hit(pairs, branch_blocks):
    """A branch hit at c <= P - W reuses c tokens of the target with the tail pool, as with a
    full-length draft pool, although the tail pool lost the drafter pages at c."""
    prompt = list(range(8 * TPB))
    branch = prompt[: branch_blocks * TPB] + list(range(5000, 5000 + (8 - branch_blocks) * TPB))
    window_start, _ = drafter_window(len(branch), WINDOW, TPB)
    branch_point = branch_blocks * TPB
    assert branch_point <= window_start

    full = pairs(tail=False)
    full.prefill(prompt)
    _, reuse_full = full.prefill(branch)
    assert reuse_full == branch_point

    tail = pairs(tail=True)
    tail.prefill(prompt)
    tail.evict_draft_until(lambda: tail.probes(branch)[0] < branch_point)
    draft_match, target_match = tail.probes(branch)
    assert target_match == branch_point
    # The joint minimum the rule replaces would have cut the target to the draft's depth.
    assert min(draft_match, target_match) < branch_point

    req, reuse_tail = tail.prefill(branch, keep=True)
    assert reuse_tail == reuse_full
    assert tail.draft.kv_cache_map[req.py_request_id].absent_before == branch_point
    tail.free(req)


def test_window_complete_on_linear_growth_after_eviction(pairs):
    """A prompt that extends a stored one, after the draft pool lost the stored one entirely:
    the target reuses up to the window start, the draft starts there with no pages before it,
    holds every page of the window, and publishes only the pages it wrote."""
    pair = pairs(tail=True)
    stored = list(range(6 * TPB))
    grown = stored + list(range(7000, 7000 + 2 * TPB))
    pair.prefill(stored)
    pair.evict_draft_until(lambda: pair.probes(stored)[0] == 0)
    draft_match, target_match = pair.probes(grown)
    assert draft_match == 0 and target_match == len(stored)

    window_start, window_len = drafter_window(len(grown), WINDOW, TPB)
    req, reused = pair.prefill(grown, keep=True)
    assert reused == min(target_match, window_start)
    kv_cache = pair.draft.kv_cache_map[req.py_request_id]
    assert kv_cache.absent_before == reused
    pages = _draft_pages(pair, req)
    first, end = window_start // TPB, -(-len(grown) // TPB)
    assert all(p != BAD_PAGE_INDEX for p in pages[first:end]), pages
    assert all(p == BAD_PAGE_INDEX for p in pages[: reused // TPB]), pages
    pair.free(req)

    # The draft tree carries the grown prompt's window at its end: a prompt extending it again
    # is a draft hit at its full length.
    extended = grown + list(range(9000, 9000 + TPB))
    assert pair.probes(extended) == (len(grown), len(grown))


@pytest.mark.parametrize("draft_evicted", [False, True], ids=["draft_hit", "draft_lost"])
def test_branch_hit_past_the_window_start_keeps_the_window_whole(pairs, draft_evicted):
    """A branch hit at c > s (P - c < W): the window starts before the branch point.

    With the draft holding the stored prompt's pages at c, both pools reuse c and the draft's
    reused pages cover [s, c). With the draft having lost them, the target is held at s (the
    rule's floor) and the draft recomputes [s, P) from an absent prefix. Either way the draft
    holds every page of [s, P), so the context sends and reports the whole window.
    """
    pair = pairs(tail=True)
    stored = list(range(8 * TPB))
    branch_point = 7 * TPB
    branch = stored[:branch_point] + list(range(8000, 8000 + TPB))
    window_start, _ = drafter_window(len(branch), WINDOW, TPB)
    assert branch_point > window_start
    pair.prefill(stored)
    if draft_evicted:
        pair.evict_draft_until(lambda: pair.probes(branch)[0] < window_start)
    draft_match, target_match = pair.probes(branch)
    assert target_match == branch_point

    req, reused = pair.prefill(branch, keep=True)
    kv_cache = pair.draft.kv_cache_map[req.py_request_id]
    if draft_evicted:
        assert reused == window_start and kv_cache.absent_before == window_start
    else:
        assert draft_match == branch_point
        assert reused == branch_point and kv_cache.absent_before == 0
    pages = _draft_pages(pair, req)
    assert all(p != BAD_PAGE_INDEX for p in pages[window_start // TPB : len(branch) // TPB]), pages
    pair.free(req)

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
"""``TRTLLM_K3LL_DRAFTER_PROMPT_CTX``, the generation worker's admission: a disaggregated generation request that
received its drafter prompt window (``py_drafter_ctx_len`` = L, the latents at columns and positions 0 .. L-1) gets
its drafter slot with context length L instead of 0 (``DFlashSpecMetadata.populate_request_state`` + ``prepare``).

* Off: a generation request without a window is admitted at 0, as before.
* On, in one process: the context worker's store of a P-token prompt, its last L rows moved by -s
  (``DSparkMLAContextRenumber``, as the transceiver does on receive) and written into the generation slot's pages,
  then admitted. The drafter's block decode over that slot matches the one over an aggregated slot that stored the
  same window tokens at positions 0 .. L-1 -- the drafter state an aggregated engine has at the same window.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the drafter runs on CUDA")


def _load(name):
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare(worker, request_ids, requests):
    """One step's ``populate_request_state`` + ``prepare`` with every request in generation."""
    meta = SimpleNamespace(
        request_ids=request_ids,
        num_generations=len(request_ids),
        batch_indices_cuda=torch.zeros(8, dtype=torch.int, device="cuda"),
        _dflash_worker=worker,
    )
    from tensorrt_llm._device_copy import DeviceCopyBuffer

    meta._batch_indices_buffer = DeviceCopyBuffer(meta.batch_indices_cuda)
    DFlashSpecMetadata.populate_request_state(meta, requests)
    DFlashSpecMetadata.prepare(meta)
    torch.cuda.synchronize()


def _write_rows(worker, slot, rows):
    """Write ``rows`` ([layers, n, latent]) into ``slot``'s pages at columns 0 .. n-1 (the receive's layout)."""
    buf = worker._ctx_kv_buf  # [layers, pages, 1, 1, page, latent]
    page = buf.size(-2)
    cols = torch.arange(rows.size(1), device=buf.device)
    pages = worker._ctx_page_table[slot].long()[cols // page]
    buf[:, pages, 0, 0, cols % page] = rows


@needs_cuda
def test_without_a_window_admission_is_unchanged():
    ctx = _load("test_k3_drafter_prompt_ctx")
    sem = ctx._semantics()
    drafter = ctx._build(ctx.MLADSparkForCausalLM, sem._tiny_mla_weights())
    _, lens, slots, worker = ctx._store(
        drafter, [[(1, 0, 40)]], sem.MLA_HIDDEN * sem.NUM_CAPTURE, keep_worker=True
    )
    # Request 2 is a generation request with no received window: the attribute is absent, or 0.
    for requests in (
        [SimpleNamespace(py_request_id=2)],
        [SimpleNamespace(py_request_id=2, py_drafter_ctx_len=0)],
    ):
        _prepare(worker, [1, 2], requests)
        slot = worker._req_to_slot[2]
        assert worker._ctx_len_host[slot] == 0 and int(worker._ctx_len[slot]) == 0
        assert worker._ctx_len_host[slots[1]] == 40
        _prepare(worker, [1], [])  # request 2 leaves; its slot is freed at 0
        assert 2 not in worker._req_to_slot


@needs_cuda
@pytest.mark.parametrize("window", [100, 1000], ids=["W<P", "W>=P"])
def test_received_window_matches_the_aggregated_drafter_at_the_same_window(window):
    from tensorrt_llm._torch.disaggregation.transceiver import drafter_window

    ctx = _load("test_k3_drafter_prompt_ctx")
    sem = ctx._semantics()
    drafter = ctx._build(ctx.MLADSparkForCausalLM, sem._tiny_mla_weights())
    hidden_width = sem.MLA_HIDDEN * sem.NUM_CAPTURE
    prompt_len = 203
    g = torch.Generator(device="cuda").manual_seed(5)
    prompt = (torch.randn(prompt_len, hidden_width, generator=g, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    # The worker's arena page size bounds the window start, as the draft manager's tokens_per_block does.
    probe = ctx._store(drafter, [[(9, 0, 8)]], hidden_width, keep_worker=True)[3]
    start, length = drafter_window(prompt_len, window, probe._ctx_page_size)
    assert length == prompt_len - start and (start > 0) == (window < prompt_len)
    # request 1: the context worker's store of the whole prompt (absolute positions 0 .. P-1);
    # request 2: the aggregated reference, the window's tokens stored at 0 .. L-1.
    _, lens, slots, worker = ctx._store(
        drafter,
        [[(1, 0, prompt_len), (2, 0, length)]],
        hidden_width,
        hidden=[torch.cat([prompt, prompt[start:]])],
        keep_worker=True,
    )
    assert lens[slots[2]] == length
    received = ctx._slot_rows(worker, slots[1], prompt_len)[:, start:].contiguous()
    drafter.context_renumber().renumber_rows_(received, start)

    # Generation request 3 is admitted with its received window.
    _prepare(worker, [1, 2, 3], [SimpleNamespace(py_request_id=3, py_drafter_ctx_len=length)])
    gen_slot = worker._req_to_slot[3]
    assert worker._ctx_len_host[gen_slot] == length and int(worker._ctx_len[gen_slot]) == length
    _write_rows(worker, gen_slot, received)
    agg_rows = ctx._slot_rows(worker, slots[2], length)
    gen_rows = ctx._slot_rows(worker, gen_slot, length)
    kv_lora = drafter.context_renumber().rope_offset
    assert torch.equal(
        gen_rows[..., :kv_lora].view(torch.int16), agg_rows[..., :kv_lora].view(torch.int16)
    )
    assert ctx._pair_ulps(gen_rows[..., kv_lora:], agg_rows[..., kv_lora:]).max() <= 2.0

    # The drafter's block decode over each slot at context length L: queries at L .. L + block - 1.
    block = sem.MLA_BLOCK
    noise = (torch.randn(1, block, sem.MLA_HIDDEN, generator=g, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    buf = worker._ctx_kv_buf

    def decode(slot, ctx_len):
        out = drafter.dflash_forward(
            noise_embedding=noise.clone(),
            query_positions=torch.arange(ctx_len, ctx_len + block, device="cuda").unsqueeze(0),
            num_ctx_per_req=torch.tensor([ctx_len], device="cuda"),
            ctx_k_cache=buf[0],
            ctx_v_cache=None,
            ctx_cache_batch_idx=torch.tensor([slot], device="cuda"),
            ctx_kv_cache=[buf[layer] for layer in range(buf.size(0))],
            ctx_page_table=worker._ctx_page_table,
        )
        return out.float()

    agg = decode(slots[2], length)
    gen = decode(gen_slot, length)
    empty = decode(gen_slot, 0)  # today's disaggregated admission: no prompt context
    rel = ((gen - agg).norm() / agg.norm()).item()
    rel_empty = ((empty - agg).norm() / agg.norm()).item()
    print(
        f"window {window}: s={start} L={length}: gen vs agg rel {rel:.2e}; empty vs agg rel {rel_empty:.2e}"
    )
    assert rel < 1e-2
    assert rel_empty > 10 * max(rel, 1e-3), (
        "the prompt context made no difference: the test measures nothing"
    )

    # Generated tokens append at column L.
    worker._advance_ctx_len(
        torch.tensor([gen_slot], device="cuda"), torch.tensor([3], device="cuda")
    )
    torch.cuda.synchronize()
    assert int(worker._ctx_len[gen_slot]) == length + 3

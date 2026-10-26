# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_DRAFTER_PROMPT_CTX``, the disaggregated context worker's half: the capture-only MLA DSpark drafter
(``MLADSparkCaptureOnly``).

* It holds only what the context store reads (``fc`` / ``hidden_norm`` and the per-layer latent projection + norm)
  and loads only those tensors from a full checkpoint.
* Agg reference: ``DFlashWorker._store_prefill_context`` -- the worker's own store, chunked prefill included --
  writes bitwise the same latents into the context cache with it as with the full ``MLADSparkForCausalLM`` of the
  same weights.
* It is built only on a context worker (``TRTLLM_DISAGG_ROLE=context``) with the switch > 0; the switch at 0 (the
  default) builds the full drafter, as before.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import MLADSparkCaptureOnly, MLADSparkForCausalLM
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the drafter runs on CUDA")


def _semantics():
    """The tiny MLA drafter config and weights of test_kimi_k3_dspark_semantics.py (not a package)."""
    path = pathlib.Path(__file__).with_name("test_kimi_k3_dspark_semantics.py")
    spec = importlib.util.spec_from_file_location("_k3_dspark_semantics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(cls, weights):
    from tensorrt_llm._torch.model_config import ModelConfig

    sem = _semantics()
    model_config = ModelConfig(pretrained_config=sem._tiny_mla_config(), attn_backend="TRTLLM")
    drafter = cls(model_config, dflash_attention_backend="TRTLLM").to("cuda")
    drafter.load_weights(dict(weights))
    return drafter


@needs_cuda
def test_capture_only_drafter_holds_and_loads_only_the_capture():
    sem = _semantics()
    weights = sem._tiny_mla_weights()
    capture = _build(MLADSparkCaptureOnly, weights)
    full = _build(MLADSparkForCausalLM, weights)

    names = {name for name, _ in capture.named_parameters()}
    assert names, "the capture-only drafter lost its parameters"
    # lm_head is the target's (load_weights_from_target_model), never loaded from the drafter checkpoint
    kept = (
        "fc.",
        "hidden_norm.",
        ".self_attn.kv_a_proj_with_mqa.",
        ".self_attn.kv_a_layernorm.",
        "lm_head.",
    )
    assert all(any(k in f".{name}" for k in kept) for name in names), sorted(names)
    assert (
        sum(p.numel() for p in capture.parameters()) < sum(p.numel() for p in full.parameters()) / 2
    )
    assert capture.draft_model_full.model.embed_tokens is None
    for name in ("fc", "hidden_norm"):
        torch.testing.assert_close(
            getattr(capture, name).weight, getattr(full, name).weight, rtol=0, atol=0
        )
    for lc, lf in zip(capture.model.layers, full.model.layers):
        for name in MLADSparkCaptureOnly.KEPT_ATTENTION_MODULES:
            torch.testing.assert_close(
                getattr(lc.self_attn, name).weight,
                getattr(lf.self_attn, name).weight,
                rtol=0,
                atol=0,
            )
        assert lc.self_attn.kv_b_proj is None and lc.mlp is None
    with pytest.raises(RuntimeError, match="does not draft"):
        capture.dflash_forward()


def _store(drafter, chunks, hidden_width, seed=7):
    """Run ``DFlashWorker._store_prefill_context`` over ``chunks`` (one list of (request id, first position, length)
    per forward) on a fresh worker; return the context cache and the per-slot lengths."""
    sem = _semantics()
    worker = DFlashWorker(
        DFlashDecodingConfig(max_draft_len=sem.MLA_BLOCK - 1, attention_backend="TRTLLM"), Mapping()
    )
    worker.set_draft_model(drafter)
    worker._lazy_init_ctx_buffers(
        drafter, SimpleNamespace(max_num_requests=4), SimpleNamespace(max_seq_len=256)
    )
    g = torch.Generator(device="cuda").manual_seed(seed)
    for forward in chunks:
        lens = [n for _, _, n in forward]
        total = sum(lens)
        captured = (torch.randn(total, hidden_width, generator=g, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        positions = torch.cat(
            [torch.arange(p, p + n, device="cuda", dtype=torch.int32) for _, p, n in forward]
        )
        spec_metadata = SimpleNamespace(
            get_hidden_states=lambda n, c=captured: c[:n],
            host_position_ids=positions.cpu(),
            request_ids=[rid for rid, _, _ in forward],
        )
        attn_metadata = SimpleNamespace(
            num_ctx_tokens=total, num_contexts=len(forward), _seq_lens=torch.tensor(lens)
        )
        worker._store_prefill_context(drafter, spec_metadata, attn_metadata, positions, total)
    torch.cuda.synchronize()
    return worker._ctx_kv_buf.clone(), list(worker._ctx_len_host), dict(worker._req_to_slot)


@needs_cuda
def test_capture_only_store_is_bitwise_the_full_drafters():
    """The agg reference: the worker's own context store, a whole prompt and a chunked one, with the capture-only
    drafter and with the full drafter of the same weights, writes the same bytes."""
    sem = _semantics()
    weights = sem._tiny_mla_weights()
    hidden_width = sem.MLA_HIDDEN * sem.NUM_CAPTURE
    # forward 1: request 1 whole (40 tokens) + request 2's first chunk (16); forward 2: request 2's second chunk
    chunks = [[(1, 0, 40), (2, 0, 16)], [(2, 16, 12)]]
    full = _store(_build(MLADSparkForCausalLM, weights), chunks, hidden_width)
    capture = _store(_build(MLADSparkCaptureOnly, weights), chunks, hidden_width)
    assert full[2] == capture[2] and len(full[2]) == 2
    slots = full[2]
    assert full[1][slots[1]] == 40 and full[1][slots[2]] == 28
    assert full[1] == capture[1]
    assert full[0].abs().sum() > 0, "nothing was stored"
    assert torch.equal(full[0].view(torch.int16), capture[0].view(torch.int16))


def test_capture_only_needs_the_context_role_and_the_switch(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import k3ll_flags

    calls = []
    monkeypatch.setattr(
        k3ll_flags, "k3ll_flag_any_target", lambda name: calls.append(name) or 8192, raising=True
    )
    monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    assert not modeling_dspark.dspark_capture_only()
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    assert not modeling_dspark.dspark_capture_only()
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    assert modeling_dspark.dspark_capture_only()
    assert calls == ["TRTLLM_K3LL_DRAFTER_PROMPT_CTX"]
    monkeypatch.setattr(k3ll_flags, "k3ll_flag_any_target", lambda name: 0, raising=True)
    assert not modeling_dspark.dspark_capture_only()


def test_switch_default_off_and_documented():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    _, default, doc = flags.flags()["TRTLLM_K3LL_DRAFTER_PROMPT_CTX"]
    assert default == 0
    assert "0 (default): off" in doc and "TRTLLM_DISAGG_ROLE=context" in doc

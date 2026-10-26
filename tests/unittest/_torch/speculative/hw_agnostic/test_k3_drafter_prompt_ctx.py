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

The generation worker's half of the interface: the received window, stored by the context at absolute positions
s .. P-1, is moved to 0 .. L-1 once (``DSparkMLAContextRenumber``, applied in the draft pool by the transceiver's
``_renumber_pool_rows``). Reference: the drafter's own store of the same tokens at 0 .. L-1, within one bf16 ulp.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import (
    DSparkMLAContextRenumber,
    MLADSparkCaptureOnly,
    MLADSparkForCausalLM,
)
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


# Inferact/Kimi-K3-DSpark (hf-cf6b824) config.json rope_parameters: the long-shift measurement uses the real phases.
K3_DSPARK_ROPE = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 50000.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def _build(cls, weights, rope_parameters=None):
    from tensorrt_llm._torch.model_config import ModelConfig

    sem = _semantics()
    config = sem._tiny_mla_config()
    if rope_parameters is not None:
        config.rope_parameters = dict(rope_parameters)
        config.rope_theta = rope_parameters["rope_theta"]
    model_config = ModelConfig(pretrained_config=config, attn_backend="TRTLLM")
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
    kept = (
        "fc.",
        "hidden_norm.",
        ".self_attn.kv_a_proj_with_mqa.",
        ".self_attn.kv_a_layernorm.",
    )
    assert all(any(k in f".{name}" for k in kept) for name in names), sorted(names)
    assert (
        sum(p.numel() for p in capture.parameters()) < sum(p.numel() for p in full.parameters()) / 2
    )
    assert capture.draft_model_full.model.embed_tokens is None
    # No head, not even after the target's is offered: no [vocab, hidden] tensor (2.35 GB on K3) survives.
    capture.load_weights_from_target_model(SimpleNamespace(lm_head=full.lm_head, model=full.model))
    assert capture.lm_head is None and capture.draft_model_full.lm_head is None
    head_shape = full.lm_head.weight.shape
    assert all(t.shape != head_shape for t in [*capture.parameters(), *capture.buffers()])
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


def _store(drafter, chunks, hidden_width, seed=7, hidden=None, keep_worker=False):
    """Run ``DFlashWorker._store_prefill_context`` over ``chunks`` (one list of (request id, first position, length)
    per forward) on a fresh worker; return the context cache and the per-slot lengths. ``hidden`` gives each
    forward's captured hidden states instead of random ones; ``keep_worker`` appends the worker."""
    sem = _semantics()
    worker = DFlashWorker(
        DFlashDecodingConfig(max_draft_len=sem.MLA_BLOCK - 1, attention_backend="TRTLLM"), Mapping()
    )
    worker.set_draft_model(drafter)
    # The store's bytes are compared on the private arena. The draft-manager pool a capture-only drafter needs on a
    # context worker is test_capture_only_ctx_cache_must_be_the_reuse_addressable_pool's.
    drafter.CAPTURE_ONLY = False
    worker._lazy_init_ctx_buffers(
        drafter, SimpleNamespace(max_num_requests=4), SimpleNamespace(max_seq_len=256)
    )
    g = torch.Generator(device="cuda").manual_seed(seed)
    for forward_idx, forward in enumerate(chunks):
        lens = [n for _, _, n in forward]
        total = sum(lens)
        captured = (
            hidden[forward_idx]
            if hidden is not None
            else (torch.randn(total, hidden_width, generator=g, device="cuda") * 0.5).to(
                torch.bfloat16
            )
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
    out = (worker._ctx_kv_buf.clone(), list(worker._ctx_len_host), dict(worker._req_to_slot))
    return out + (worker,) if keep_worker else out


def _slot_rows(worker, slot, n):
    """The first ``n`` stored rows of ``slot`` on the private paged arena: ``[layers, n, latent]``."""
    buf = worker._ctx_kv_buf  # [layers, pages, 1, 1, page, latent]
    pages = worker._ctx_page_table[slot].long()
    rows = buf[:, pages].reshape(buf.size(0), -1, buf.size(-1))
    return rows[:, :n].clone()


def _bf16_ulp_distance(a, b):
    """Elementwise distance in bf16 ulps: the difference of the bit patterns read as ordered integers."""

    def ordered(x):
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _pair_ulps(got, ref):
    """Error of each rope pair component in bf16 ulps of the reference pair's modulus.

    The per-element ulp is the wrong yardstick for a rotation. Rotating an already-rounded bf16 pair moves that
    rounding error between the two components, so a component near zero carries an error of the pair's scale. That
    error can be many of its own ulps, even for an exact rotation. The pair's modulus is rotation invariant, so its
    ulp is the scale the error belongs to.
    """
    g = got.double().unflatten(-1, (-1, 2))
    r = ref.double().unflatten(-1, (-1, 2))
    modulus = r.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(modulus)) - 7)
    return ((g - r).abs() / ulp).flatten(-2)


def _exact_move(rope, shift, inv_freq):
    """``rope`` (bf16 adjacent pairs) rotated by ``-shift * inv_freq`` in float64, unrounded."""
    angle = -float(shift) * inv_freq.to(rope.device)
    phase = torch.polar(torch.ones_like(angle), angle)
    xc = torch.view_as_complex(rope.double().contiguous().unflatten(-1, (-1, 2)))
    return torch.view_as_real(xc * phase).flatten(-2)


def _summary(name, got, ref):
    pair = _pair_ulps(got, ref)
    elem = _bf16_ulp_distance(got, ref.to(got.dtype)) if ref.dtype == got.dtype else None
    line = (
        f"{name}: pair-ulp max {pair.max().item():.3f} mean {pair.mean().item():.4f} "
        f"p99.9 {np.percentile(pair.cpu().numpy(), 99.9):.3f}"
    )
    if elem is not None:
        line += f"; element-ulp max {int(elem.max())} frac>1 {(elem > 1).float().mean().item():.5f}"
    print(line)
    return pair


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


def test_serve_carries_the_role_to_the_model_ranks(monkeypatch):
    """With the switch on, the model ranks get the role through env_overrides, which every rank applies before its
    build. With it off, nothing is carried: other readers of the role on the model ranks see what they did before."""
    from tensorrt_llm.commands import serve

    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    monkeypatch.setenv("TRTLLM_K3LL_DRAFTER_PROMPT_CTX", "16384")
    args = {"env_overrides": None}
    serve._carry_disagg_role_to_model_ranks(args)
    assert args["env_overrides"] == {"TRTLLM_DISAGG_ROLE": "context"}
    args = {"env_overrides": {"A": "1", "TRTLLM_DISAGG_ROLE": "generation"}}
    serve._carry_disagg_role_to_model_ranks(args)
    assert args["env_overrides"] == {"A": "1", "TRTLLM_DISAGG_ROLE": "generation"}
    for off in ("0", ""):
        monkeypatch.setenv("TRTLLM_K3LL_DRAFTER_PROMPT_CTX", off)
        args = {"env_overrides": None}
        serve._carry_disagg_role_to_model_ranks(args)
        assert args == {"env_overrides": None}
    monkeypatch.delenv("TRTLLM_K3LL_DRAFTER_PROMPT_CTX")
    serve._carry_disagg_role_to_model_ranks(args)
    assert args == {"env_overrides": None}
    monkeypatch.setenv("TRTLLM_K3LL_DRAFTER_PROMPT_CTX", "16384")
    monkeypatch.delenv("TRTLLM_DISAGG_ROLE")
    args = {}
    serve._carry_disagg_role_to_model_ranks(args)
    assert args == {}


def test_capture_only_ctx_keeps_no_kda_draft_records(monkeypatch):
    """A context worker with the capture-only drafter never verifies: no draft records, so block reuse stays."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        implementation,
    )

    mode = SimpleNamespace(is_dflash=lambda: False, is_dspark=lambda: True)
    backbone = SimpleNamespace(
        model_config=SimpleNamespace(
            spec_config=SimpleNamespace(spec_dec_mode=mode, tokens_per_gen_step=8)
        ),
        layers=[],
    )
    records = implementation.KimiLinearModel.kda_draft_records.fget
    monkeypatch.setattr(implementation, "dspark_capture_only", lambda: False)
    assert records(backbone)
    monkeypatch.setattr(implementation, "dspark_capture_only", lambda: True)
    assert not records(backbone)


def test_capture_only_ctx_cache_must_be_the_reuse_addressable_pool():
    """The window is sent from the draft manager's pages: a capture-only drafter on a private arena, or on pages a
    reused prefix left unwritten, would send pages it never wrote. Both raise."""
    check = DFlashWorker._check_capture_only_ctx_cache
    reuse_on, reuse_off = (
        SimpleNamespace(enable_block_reuse=True),
        SimpleNamespace(enable_block_reuse=False),
    )
    with pytest.raises(RuntimeError, match="private arena"):
        check(SimpleNamespace(_ctx_pool_bound=False, _ctx_reuse_addressable=False), reuse_off)
    with pytest.raises(RuntimeError, match="not reuse-addressable"):
        check(SimpleNamespace(_ctx_pool_bound=True, _ctx_reuse_addressable=False), reuse_on)
    check(SimpleNamespace(_ctx_pool_bound=True, _ctx_reuse_addressable=False), reuse_off)
    check(SimpleNamespace(_ctx_pool_bound=True, _ctx_reuse_addressable=True), reuse_on)


def test_context_window_needs_the_capture_only_drafter():
    """A context-only request's window from a rank whose drafter is not capture-only means the role was lost."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    rank = SimpleNamespace(
        _draft_kv_cache_manager=object(),
        _draft_reuse_adapter=object(),
        _page_table=object(),
        _draft_capture_only=False,
    )
    with pytest.raises(RuntimeError, match="capture-only"):
        KvCacheTransceiverV2._describe_draft_window(
            rank, SimpleNamespace(is_context_only_request=True)
        )


def test_capture_only_context_warms_up_no_generation_shape():
    """A context worker with the capture-only drafter runs no generation step: the runner skips the generation
    warmup shapes, which with the speculative config would run the target's verify step."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner

    prop = runner.DecoderRunner._capture_only_context.fget
    assert prop(
        SimpleNamespace(model=SimpleNamespace(draft_model=SimpleNamespace(CAPTURE_ONLY=True)))
    )
    assert not prop(SimpleNamespace(model=SimpleNamespace(draft_model=SimpleNamespace())))
    assert not prop(SimpleNamespace(model=SimpleNamespace()))


def test_capture_only_context_allocates_no_kda_replay_state():
    """The KDA replay caches serve the target's verify step, which a capture-only context worker never runs."""
    from tensorrt_llm._torch.pyexecutor._util import _target_verifies_drafts

    spec = object()

    def engine(drafter):
        return SimpleNamespace(model=SimpleNamespace(draft_model=drafter))

    assert not _target_verifies_drafts(None, engine(None))
    assert _target_verifies_drafts(spec, engine(SimpleNamespace()))
    assert _target_verifies_drafts(spec, SimpleNamespace(model=SimpleNamespace()))
    assert not _target_verifies_drafts(spec, engine(SimpleNamespace(CAPTURE_ONLY=True)))


def _starved_adp_rank(capture_only):
    """An attention-DP rank that holds an active request but scheduled none (short of KV cache), as
    ``_pad_empty_attention_dp_batch`` finds it."""
    from unittest.mock import MagicMock

    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = PyExecutor.__new__(PyExecutor)
    dummy = SimpleNamespace(py_request_id=0, is_last_context_chunk=True)
    executor.__dict__.update(
        enable_attention_dp=True,
        active_requests=[SimpleNamespace(py_request_id=11)],
        expected_num_active_requests=1,
        max_num_active_requests=8,
        max_total_draft_tokens=7,
        _capture_only_context=capture_only,
        _adp_dummy_gen_role=True,
        _pending_adp_dummy_request=None,
        _should_skip_dummy_for_benchmark_disagg=lambda num_active: False,
        _count_schedulable_active_requests=lambda: 1,
        _has_adp_dummy_kv_capacity=MagicMock(return_value=True),
        kv_cache_manager=MagicMock(**{"add_dummy_requests.return_value": [dummy]}),
        draft_kv_cache_manager=MagicMock(name="draft_kv_cache_manager"),
        resource_manager=MagicMock(**{"get_resource_manager.return_value": None}),
    )
    return executor, dummy


@pytest.mark.parametrize("capture_only", [True, False])
def test_capture_only_context_pads_attention_dp_with_a_context_dummy(capture_only):
    """A starved attention-DP rank is padded with a dummy. On a capture-only context worker it is a one-token
    context request: a generation dummy carries 1 + max_draft_len tokens, which routes the KDA layers into the
    target's verify step, whose state that worker does not provision (16p3d c84 on, jhb 1297305: HTTP 400
    "'State' object has no attribute 'intermediate_conv_window'"). Elsewhere it stays a generation dummy."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests

    executor, dummy = _starved_adp_rank(capture_only)
    batch = ScheduledRequests()
    PyExecutor._pad_empty_attention_dp_batch(executor, batch)

    kwargs = executor.kv_cache_manager.add_dummy_requests.call_args.kwargs
    executor._has_adp_dummy_kv_capacity.assert_called_once_with(kwargs["token_nums"])
    if capture_only:
        assert kwargs["is_gen"] is False and kwargs["token_nums"] == [1]
        assert kwargs["draft_kv_cache_manager"] is executor.draft_kv_cache_manager
        assert batch.generation_requests == [] and batch.context_requests == [dummy]
    else:
        assert kwargs["is_gen"] is True and kwargs["token_nums"] is None
        assert "draft_kv_cache_manager" not in kwargs
        assert batch.generation_requests == [dummy] and batch.context_requests == []
    assert executor._pending_adp_dummy_request is dummy and dummy.is_attention_dp_dummy


def test_capture_only_context_dummy_short_of_draft_pages_retries():
    """On a capture-only context worker the padding dummy's draft mirror is allocated with it. When the draft
    pool cannot hold it the allocation returns nothing (as ``add_dummy_requests`` does on a failed draft resize)
    and the rank schedules an empty batch to retry, rather than failing the draft resize later in
    ``_prepare_draft_resources``."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests

    executor, _ = _starved_adp_rank(capture_only=True)
    full_draft_pool = executor.draft_kv_cache_manager

    def add_dummy_requests(**kwargs):
        return None if kwargs.get("draft_kv_cache_manager") is full_draft_pool else [object()]

    executor.kv_cache_manager.add_dummy_requests.side_effect = add_dummy_requests
    active_before = list(executor.active_requests)
    batch = ScheduledRequests()
    PyExecutor._pad_empty_attention_dp_batch(executor, batch)

    assert batch.batch_size == 0
    assert executor.active_requests == active_before
    assert executor._pending_adp_dummy_request is None


def test_capture_only_context_never_picks_a_generation_dummy_role():
    """The pre-schedule pad's role (`_adp_dummy_is_gen`) is generation by default and follows the requests seen;
    on a capture-only context worker it reads as context whatever was set."""
    executor, _ = _starved_adp_rank(capture_only=True)
    executor._adp_dummy_is_gen = True
    assert executor._adp_dummy_is_gen is False
    executor._capture_only_context = False
    assert executor._adp_dummy_is_gen is True


def test_kda_verify_without_verify_state_raises():
    """A verify step that reaches a KDA layer whose cache manager provisioned no verify state raises a named
    error rather than a missing attribute deep in the sequential path."""
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

    layer = SimpleNamespace(
        layer_idx=2,
        _has_kda_replay_caches=lambda cache: KimiKDALinearAttention._has_kda_replay_caches(
            None, cache
        ),
    )
    cache = SimpleNamespace(has_kda_replay_caches=False, conv=None, temporal=None)
    with pytest.raises(RuntimeError, match="capture-only context worker"):
        KimiKDALinearAttention.forward_verify(layer, torch.zeros(8, 4), 8, cache, None, None, None)


def test_switch_default_off_and_documented():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    _, default, doc = flags.flags()["TRTLLM_K3LL_DRAFTER_PROMPT_CTX"]
    assert default == 0
    assert "0 (default): off" in doc and "TRTLLM_DISAGG_ROLE=context" in doc


@needs_cuda
def test_renumbered_window_matches_a_store_at_window_positions():
    """The interface to the generation half. The context stores the prompt at 0 .. P-1 and sends rows s .. P-1.
    Moved by -s, they equal the drafter's own store of the same tokens at 0 .. L-1: the latent half bitwise, the
    rope half within 2 ulps of each pair's modulus. Both stores run in one forward, so they share one latent GEMM.

    The 2 is the double-rounding floor: the stored pair's rounding (<= 0.71 ulp after rotation) + the move's own
    rounding (0.5) + the reference's rounding (0.5). The move itself is held to its single rounding: within 1 ulp
    of the pair modulus of the exact float64 rotation of the same stored bits."""
    from tensorrt_llm._torch.disaggregation.transceiver import drafter_window

    sem = _semantics()
    drafter = _build(MLADSparkForCausalLM, sem._tiny_mla_weights())
    hidden_width = sem.MLA_HIDDEN * sem.NUM_CAPTURE
    prompt_len, window, tokens_per_block = 203, 100, 8
    start, length = drafter_window(prompt_len, window, tokens_per_block)
    assert (start, length) == (96, 107)
    g = torch.Generator(device="cuda").manual_seed(3)
    prompt = (torch.randn(prompt_len, hidden_width, generator=g, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    # request 1: the whole prompt, as the context stores it; request 2: the window's tokens alone (the reference)
    _, lens, slots, worker = _store(
        drafter,
        [[(1, 0, prompt_len), (2, 0, length)]],
        hidden_width,
        hidden=[torch.cat([prompt, prompt[start:]])],
        keep_worker=True,
    )
    assert lens[slots[1]] == prompt_len and lens[slots[2]] == length
    received = _slot_rows(worker, slots[1], prompt_len)[:, start:]
    reference = _slot_rows(worker, slots[2], length)
    renumber = drafter.context_renumber()
    kv_lora = renumber.rope_offset
    assert renumber.row_elems == received.size(-1) == sem.MLA_LATENT

    # Unmoved, the rope half is far off: the shift is what the test measures.
    assert _pair_ulps(received[..., kv_lora:], reference[..., kv_lora:]).max() > 8
    exact = _exact_move(received[..., kv_lora:], start, renumber.inv_freq)
    renumber.renumber_rows_(received, start)
    assert torch.equal(
        received[..., :kv_lora].view(torch.int16), reference[..., :kv_lora].view(torch.int16)
    )
    moved = received[..., kv_lora:]
    assert _summary(f"RENUMBER s={start} move vs exact", moved, exact).max() <= 1.0
    assert _summary(f"RENUMBER s={start} vs store", moved, reference[..., kv_lora:]).max() <= 2.0
    # A zero shift leaves the rows bitwise unchanged.
    again = reference.clone()
    renumber.renumber_rows_(again, 0)
    assert torch.equal(again.view(torch.int16), reference.view(torch.int16))


@needs_cuda
@pytest.mark.parametrize("shift", [8192, 32768, 150000, 300000, 600000])
def test_renumber_at_long_shift(shift):
    """Measured at real shifts with the real K3 DSpark rope parameters: the context's store at s + j, moved by -s,
    against the store at j. Two moves are measured: the float64 angle (DSparkMLAContextRenumber) and the float32
    table's own phase at s (the conj of the _mla_freqs_cis row). Both are also measured against the exact rotation of
    the same stored bits. The printed lines are the report.

    The move holds its single rounding at every shift (<= 1 pair ulp against the exact rotation). Against the store
    it is held to the double-rounding floor (<= 2 pair ulps) up to 150K, the longest bin where the drafter's
    acceptance with the window shift is measured. Above that, a residual that grows with s is reported, not
    asserted: the store's own float32 angle rounding at s + j (up to about (s + j) * inv_freq * 2^-24 rad). No move
    can undo it, and a store at absolute positions carries it too."""
    sem = _semantics()
    drafter = _build(MLADSparkForCausalLM, sem._tiny_mla_weights(), rope_parameters=K3_DSPARK_ROPE)
    n = 256
    drafter._runtime_position_ceiling = shift + n + 1
    g = torch.Generator(device="cuda").manual_seed(5)
    projected = (torch.randn(n, sem.MLA_HIDDEN, generator=g, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    at_shift, _ = drafter.precompute_context_kv(
        projected, torch.arange(shift, shift + n, device="cuda")
    )
    at_zero, _ = drafter.precompute_context_kv(projected, torch.arange(0, n, device="cuda"))
    renumber = drafter.context_renumber()
    kv_lora = renumber.rope_offset
    reference = at_zero[..., kv_lora:]
    assert torch.equal(
        at_shift[..., :kv_lora].view(torch.int16), at_zero[..., :kv_lora].view(torch.int16)
    )
    stored = at_shift[..., kv_lora:]
    exact = _exact_move(stored, shift, renumber.inv_freq)

    moved64 = at_shift.clone()
    renumber.renumber_rows_(moved64, shift)
    moved64 = moved64[..., kv_lora:]
    table_phase = torch.conj(drafter._mla_freqs_cis(torch.device("cuda"))[shift])
    xc = torch.view_as_complex(stored.float().contiguous().unflatten(-1, (-1, 2)))
    moved32 = torch.view_as_real(xc * table_phase).flatten(-2).to(at_shift.dtype)

    move64 = _summary(f"RENUMBER s={shift} fp64-angle move vs exact", moved64, exact)
    _summary(f"RENUMBER s={shift} fp32-table move vs exact", moved32, exact)
    e2e64 = _summary(f"RENUMBER s={shift} fp64-angle vs store", moved64, reference)
    e2e32 = _summary(f"RENUMBER s={shift} fp32-table vs store", moved32, reference)
    assert move64.max() <= 1.0
    assert e2e64.mean() <= e2e32.mean() + 1e-9
    if shift <= 150000:
        assert e2e64.max() <= 2.0


@needs_cuda
def test_renumber_pool_rows_moves_only_the_listed_pages():
    """The transceiver's in-pool move (``_renumber_pool_rows``) on a pool laid out as the V2 draft pool:
    slots of two layers' buffers each, every buffer whole latent rows. Listed pages move exactly as
    ``renumber_rows_`` moves them; every other byte stays."""
    from tensorrt_llm._torch.disaggregation.resource.page import BUFFER_ENTRY_DTYPE, PhysicalPool
    from tensorrt_llm._torch.disaggregation.transceiver import _renumber_pool_rows

    sem = _semantics()
    renumber = DSparkMLAContextRenumber(
        _build(MLADSparkForCausalLM, sem._tiny_mla_weights())._mla_rope_params,
        sem.MLA_KV_LORA,
        sem.MLA_ROPE,
    )
    tokens_per_block, layers, num_slots = 8, 2, 6
    row = renumber.row_elems
    g = torch.Generator(device="cuda").manual_seed(9)
    pool = torch.randn(num_slots, layers, tokens_per_block, row, generator=g, device="cuda").to(
        torch.bfloat16
    )
    before = pool.clone()
    layer_bytes = tokens_per_block * row * 2
    entries = np.zeros(layers, dtype=BUFFER_ENTRY_DTYPE)
    entries["local_layer_id"] = np.arange(layers)
    entries["offset"] = np.arange(layers) * layer_bytes
    entries["size"] = layer_bytes
    physical = PhysicalPool(pool.data_ptr(), layers * layer_bytes, num_slots)
    pages = np.array([4, 1, 3], dtype=np.int64)
    _renumber_pool_rows(physical, entries, pages, torch.bfloat16, renumber, 96)
    torch.cuda.synchronize()

    expected = before.clone()
    picked = expected[torch.from_numpy(pages).cuda()]
    renumber.renumber_rows_(picked, 96)
    expected[torch.from_numpy(pages).cuda()] = picked
    assert torch.equal(pool.view(torch.int16), expected.view(torch.int16))
    untouched = [i for i in range(num_slots) if i not in pages.tolist()]
    assert torch.equal(pool[untouched].view(torch.int16), before[untouched].view(torch.int16))

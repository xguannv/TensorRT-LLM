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
"""Engine names the private Kimi-K3 package probes behind the switches its Rubin deployment runs.

The package reaches main's engine through duck-typed probes (``hasattr``, ``getattr(..., None)``,
``inspect.signature``). A probe for a name the runtime object does not have returns its fallback,
so the switch behind it stops working without an error: ``TRTLLM_K3_LD_FUSED`` on main's FC12 leaf
was one such case. Each test pins the probes of one switch against the class the package gets at
runtime (the leaf ``create_moe`` builds, the KV cache manager the executor creates, ...), so a
rename on either side fails here rather than as a slower step in a perf run.

CPU only: classes and signatures, no kernels.
"""

import dataclasses
import importlib
import inspect
from types import SimpleNamespace

import pytest
from torch import nn

K3_RUBIN = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default"
)
TARGETS = ("tp1", "helix8", "helix16_moetp4ep4", "dep16")


def _k3_module(name):
    """A module of the Kimi K3 target package (``modeling``, ``drafter``)."""
    return importlib.import_module(f"{K3_RUBIN}.{name}")


def _bare(cls):
    """An instance of the ``nn.Module`` ``cls`` without running its constructor."""
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    return instance


def _is_property(cls, name):
    return isinstance(inspect.getattr_static(cls, name, None), property)


def _select_profile(monkeypatch, target):
    """Make ``target``'s feature profile the active one for fields read when called."""
    features = _k3_module("features")
    profile = features.PROFILES[target]
    monkeypatch.setattr(features, "_ACTIVE", profile)
    return profile


@pytest.mark.parametrize("target", TARGETS)
def test_runtime_hooks_reach_the_engine(monkeypatch, target):
    """``TLLM_SPEC_HEAD_FUSION`` (the decode step's head) and ``TRTLLM_K3_ULL_STEP_PREP`` (the
    packed step inputs): the engine arms them only from the ``RuntimeHooks`` it reads off the
    outermost model (``runtime_hooks_of``), which is the package's VL wrapper. The wrapper must
    hand on the text model's hooks, and the text model must declare exactly the hooks its target's
    profile turns on; everything else stays at the default."""
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, runtime_hooks_of

    spec_decode_head = _k3_module("kernels.spec_decode_head").spec_decode_head
    profile = _select_profile(monkeypatch, target)
    text = _bare(_k3_module("modeling").KimiK3DefaultForCausalLM)
    wrapper = _bare(_k3_module("modeling").ModelingV2KimiK3Nvfp4Sm107Default)
    wrapper.llm = text

    hooks = runtime_hooks_of(wrapper)
    assert hooks == runtime_hooks_of(text)
    assert (hooks.decode_prologue is spec_decode_head) == profile.embedding_gather
    assert hooks.pack_step_inputs == profile.pack_step_inputs
    assert hooks.pack_spec_rng_uploads == profile.pack_step_inputs
    assert (
        dataclasses.replace(
            hooks, decode_prologue=None, pack_step_inputs=False, pack_spec_rng_uploads=False
        )
        == DEFAULT_RUNTIME_HOOKS
    )


def test_runtime_hooks_are_on_where_the_recipes_set_them():
    """The recipes set ``TLLM_SPEC_HEAD_FUSION=1`` on helix8 and helix16 and
    ``TRTLLM_K3_ULL_STEP_PREP=1`` on helix16 only."""
    features = _k3_module("features")
    head = {t for t in TARGETS if features.PROFILES[t].embedding_gather}
    pack = {t for t in TARGETS if features.PROFILES[t].pack_step_inputs}
    assert head == {"helix8", "helix16_moetp4ep4"}
    assert pack == {"helix16_moetp4ep4"}


def test_spec_decode_head_reads_main_names():
    """``kernels.spec_decode_head`` (``decode_prologue`` on helix8 / helix16) reads main's preparer
    and metadata directly: the overlap switch, the previous step's offsets, the helix buffers and
    the per-token validity flag. ``preprocess_inputs`` must run the prologue before its own head
    and read the same names, so the two write the same step."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    source = inspect.getsource(DecoderInputPreparer.preprocess_inputs)
    for name in (
        "self.decode_prologue",
        "self._config.disable_overlap_scheduler",
        "self._buffers.previous_pos_id_offsets_cuda",
        "self._buffers.previous_kv_lens_offsets_cuda",
        "num_chunked_ctx_requests",
        "recompute_helix_spec_buffers",
        "_helix_spec_tokens_valid",
    ):
        assert name in source, name
    assert source.index("self.decode_prologue") < source.index(
        "self._config.disable_overlap_scheduler"
    )
    assert "get_runtime_tokens_per_gen_step" in inspect.signature(DecoderInputPreparer).parameters
    assert "self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step" in (
        inspect.getsource(DecoderInputPreparer.__init__)
    )
    for name in (
        "_helix_spec_tokens_valid",
        "helix_position_offsets",
        "helix_local_slots",
        "helix_kv_bounds",
        "on_update_kv_lens",
        "recompute_helix_spec_buffers",
    ):
        assert hasattr(TrtllmAttentionMetadata, name), name


def test_routed_experts_tune_the_routing_tile_with_mains_runners():
    """The routed experts (every target but dep16) run their own kernel chains; the AutoTuner picks
    the routing tile through main's two CuteDSL outer runners, called with the experts' chain as
    ``forward_impl``. The experts read the tile of a tactic (``tile_size_of``) and the tuning config
    off them, and FC12 offers only the tiles its fused kernel serves."""
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.fused_fc12 import CuteDslFc12FusedMoENvfp4Runner
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.runners import CuteDslFusedMoENvfp4Runner

    routed = _k3_module("moe")
    runners = routed._tile_runners()
    assert runners["CUTEDSL"] == (CuteDslFusedMoENvfp4Runner, "CuteDslFusedMoE::run_moe_nvfp4")
    assert runners["CUTEDSL_FC12"] == (
        CuteDslFc12FusedMoENvfp4Runner,
        "CuteDslFc12FusedMoE::run_moe_nvfp4",
    )
    params = inspect.signature(CuteDslFusedMoENvfp4Runner.__init__).parameters
    for name in (
        "forward_impl",
        "num_experts",
        "top_k",
        "num_local_experts",
        "local_expert_offset",
        "enable_finalize_fusion",
        "enable_alltoall",
        "workload_identity",
    ):
        assert name in params, name
    assert CuteDslFusedMoENvfp4Runner.tile_size_of(-1) == 128
    assert CuteDslFc12FusedMoENvfp4Runner._tile_sizes() == [128, 256]


def test_latent_down_fusion_feeds_the_fc12_experts():
    """``TRTLLM_K3_LD_FUSED`` (helix8, helix16): the latent down projection hands the routed
    experts its own NVFP4 output only where they run the FC12 kernels, whose input is quantized
    with ``fp4_quantize`` against ``fc31_input_scale``, as k3_latent_down_fp4 quantizes it."""
    features = _k3_module("features")
    assert {t for t in TARGETS if features.PROFILES[t].ld_fused} == {"helix8", "helix16_moetp4ep4"}
    source = inspect.getsource(_k3_module("moe_runtime").K3MoERuntime._fused_latent_input)
    assert '"CUTEDSL_FC12"' in source and "fc31_input_scale" in source


@pytest.mark.parametrize(
    "target, expected",
    [
        ("tp1", 0),
        ("helix8", 16),
        ("helix16_moetp4ep4", 16),
        ("dep16", 0),
    ],
)
def test_routed_experts_sm_reserve_follows_the_profile(monkeypatch, target, expected):
    """Where B's recipes set the FC12 switches: ``TRTLLM_K3_FC12_SM_RESERVE=16`` on helix8
    (beside the shared expert confined to ``TRTLLM_K3_SHARED_SM=16`` SMs); die-major weights
    (``TRTLLM_K3_FC12_DIE_LAYOUT``) are not available and only logged. On helix16
    B's small-batch experts kernel ran on 196 / 204 / 196 of the 212 SMs at 8 / 16 / 32 rows,
    beside the 16-CTA persistent shared expert of the column-parallel tail. Only the FC12 kernels
    take a reserve."""
    moe_runtime = _k3_module("moe_runtime")
    profile = _k3_module("features").PROFILES[target]
    monkeypatch.setattr(moe_runtime, "_K3_SHARED_SM", profile.shared_sm)
    monkeypatch.setattr(moe_runtime, "_K3_FC12_DIE_MAJOR", profile.fc12_die_major)
    fc12 = SimpleNamespace(kernels="CUTEDSL_FC12", sm_reserve=0)
    moe_runtime._k3_routed_backend_opt_ins(fc12, column_parallel_tail=profile.ull2_coll_cp_tail)
    assert fc12.sm_reserve == expected
    other = SimpleNamespace(kernels="CUTEDSL", sm_reserve=0)
    moe_runtime._k3_routed_backend_opt_ins(other, column_parallel_tail=profile.ull2_coll_cp_tail)
    assert other.sm_reserve == 0


class _RoutedCalled(Exception):
    """Raised by a stand-in for the routed experts to stop the MoE tail right after them."""


def _routed_experts_push(runtime, num_tokens):
    """The ``output_push`` the MoE tail of a ``num_tokens``-row forward calls the routed experts
    with (``None``: none)."""
    import torch

    pushes = []

    def routed_output(output_push=None):
        pushes.append(output_push)
        raise _RoutedCalled

    rows = torch.zeros(num_tokens, runtime.hidden_size)
    with pytest.raises(_RoutedCalled):
        runtime._rp_tail_forward(routed_output, rows, None, rows, None, True)
    assert len(pushes) == 1
    return pushes[0]


@pytest.mark.parametrize("target", TARGETS)
def test_one_kernel_tail_requests_the_output_push(target, monkeypatch):
    """``TRTLLM_K3_ULL_TAIL1`` with ``TRTLLM_K3_ULL2_COLL_CP_TAIL`` (helix16 generation): on the
    column-parallel tail of at most ``k3_moe_tail1.MAX_M`` rows the routed experts are called with
    ``output_push``, the latent workspace whose slots k3_moe_tail1 polls, where they push those
    rows (``K3SmallBatchExperts.applies``); every other tail calls them without one."""
    import torch

    routed = _k3_module("moe")
    assert "output_push" in inspect.signature(routed.K3RoutedExperts.forward).parameters

    LamportWorkspaceView = _k3_module("layers").LamportWorkspaceView
    moe_runtime = _k3_module("moe_runtime")
    tail1 = _k3_module("kernels.k3_moe_tail1")
    profile = _k3_module("features").PROFILES[target]
    monkeypatch.setattr(moe_runtime, "_K3_ULL2_COLL_CP_TAIL", profile.ull2_coll_cp_tail)
    world = 16
    latent = LamportWorkspaceView(
        world=world,
        rank=0,
        buffer=torch.empty(0),
        mc_base=0,
        peer_ptrs=torch.empty(0, dtype=torch.int64),
        flags=torch.zeros(9, dtype=torch.uint32),
        bytes_per_buffer=0,
    )
    runtime = _bare(moe_runtime.K3MoERuntime)
    runtime.hidden_size = tail1.HIDDEN
    runtime.moe_hidden_size = tail1.LATENT
    runtime.shared_expert_stream = None
    # The small-batch kernel pushes up to 32 rows.
    runtime.small_experts = SimpleNamespace(applies=lambda num_tokens: 1 <= num_tokens <= 32)
    # The tail's workspaces and k3_moe_tail1's operands as construction and loading build them;
    # the column-parallel tail serves every row count here.
    runtime._cp_ws = SimpleNamespace(world=world)
    runtime._cp_latent_ws = SimpleNamespace(world=world, rank=0)
    runtime._cp_latent_push = latent
    runtime._tail1_wexp = torch.ones(tail1.HIDDEN // 128, tail1.LATENT // 128, dtype=torch.uint8)
    runtime._cp_tail_applies = lambda num_tokens, target: True

    pushed = [m for m in range(1, 33) if _routed_experts_push(runtime, m) is latent]
    expected = list(range(1, tail1.MAX_M + 1)) if profile.ull2_coll_cp_tail else []
    assert pushed == expected

    # Experts that do not push (no small-batch kernel, a communication strategy) keep the tail.
    runtime.small_experts = SimpleNamespace(applies=lambda num_tokens: False)
    assert _routed_experts_push(runtime, 1) is None


@pytest.mark.parametrize(
    "change, max_tokens",
    [
        ({}, 32),
        ({"kernels": "CUTEDSL"}, 0),
        ({"intermediate_size_per_partition": 100}, 0),
    ],
)
def test_small_batch_experts_serve_the_fc12_calls_of_at_most_32_tokens(change, max_tokens):
    """``K3SmallBatchExperts`` (helix8, helix16 generation) takes the forwards of 1..32 tokens of
    routed experts running the FC12 kernels; the published routing word exists only on MoE-TP
    slices once the counters do."""
    import torch

    small_batch = _k3_module("moe")
    op = _k3_module("kernels.k3_small_batch_moe_op")
    kernel = _k3_module("kernels.k3_small_batch_moe")
    backend = SimpleNamespace(
        intermediate_size_per_partition=change.get("intermediate_size_per_partition", 768),
        hidden_size=3584,
        expert_size_per_partition=224,
        tp_size=4,
    )
    routed = SimpleNamespace(backend=backend, kernels=change.get("kernels", "CUTEDSL_FC12"))
    experts = small_batch.K3SmallBatchExperts(routed)
    assert experts.max_tokens == max_tokens
    assert [n for n in range(40) if experts.applies(n)] == list(range(1, max_tokens + 1))
    assert experts.published_routing_word(8) is None
    experts.counters = op.new_small_batch_moe_counters(torch.device("cpu"))
    word = experts.published_routing_word(8)
    if max_tokens:
        assert word.data_ptr() == experts.counters[kernel.CTR_ROUTE].data_ptr()
        assert experts.published_routing_word(33) is None
        backend.tp_size = 1
        assert experts.published_routing_word(8) is None
    else:
        assert word is None


def test_private_persistent_grids_take_the_shared_sm_reserve(monkeypatch):
    """``TRTLLM_DWDP_PREFETCH_GATHER`` with ``TRTLLM_K3_ATTN_RES_SM_RESERVE`` (tp1 context): the
    DWDP gather's resident SMs reach persistent grids through ``cute_dsl_utils.sm_reserve()``
    (``DwdpManager`` sets it before the model is built). The KDA prefill grid sizes itself by it
    and the persistent attention-residual op takes it as its ``sm_reserve`` argument."""
    import torch

    from tensorrt_llm._torch import cute_dsl_utils

    assert _k3_module("attn_res").sm_reserve is cute_dsl_utils.sm_reserve
    schema = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd.default._schema
    assert [arg.name for arg in schema.arguments][-1] == "sm_reserve"
    prefill_ops = pytest.importorskip(f"{K3_RUBIN}.ops.ssm._kda_prefill")
    monkeypatch.setattr(cute_dsl_utils, "_sm_reserve", 16)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=212),
    )
    assert prefill_ops._kda_persistent_grid_blocks(torch.device("cuda", 0)) == 196


def test_fp8_kv_layer_selector_on_the_cache_policy():
    """``TLLM_KIMI_K3_FP8_KV_LAYERS`` (both roles): the MLA layers take their FP8 KV-cache
    fallback only where ``kimi_k3_cache_policy`` provides the selector the cache manager reads;
    otherwise every layer keeps the model-wide NVFP4 dtype and a warning names the gap."""
    from tensorrt_llm._torch import kimi_k3_cache_policy

    selector = getattr(kimi_k3_cache_policy, "get_kimi_k3_fp8_kv_layer_ids", None)
    assert callable(selector)
    assert "nvfp4_enabled" in inspect.signature(selector).parameters


def test_fp4_mla_probe_on_trtllm_attention():
    """The FP4 MLA paths (packed decode, ``TRTLLM_K3_ULL4_MLA_PLAN_EARLY``) key off
    ``mqa.uses_fp4_mla_attention`` on main's ``TrtllmAttention``."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    assert _is_property(TrtllmAttention, "uses_fp4_mla_attention")


def test_dspark_unified_draft_cache_probes_on_the_k3_hybrid_manager():
    """DSpark's drafter context KV in the target manager (``TRTLLM_DISAGG_SHARED_DRAFT_KV``).
    The engine appends the drafter's layers to
    ``KimiK3HybridCacheManager`` only when the class opts in, and only when the loaded drafter
    says it reads the manager's pages (``reads_managed_ctx_pages`` /
    ``reads_helix_sharded_ctx_pages``, which the private drafter must override). The private
    worker finds the layers through ``draft_layer_ids`` and ``get_buffers(..., kv_layout=...)``.
    Main's ``DFlashSpecMetadata.prepare`` hands a disaggregated generation request's new slot to
    the worker's ``_seed_generation_slot``, and the private worker seeds the slot's context
    length only when it reports ``_ctx_unified_cache``. The KDA bookkeeping after accept runs
    only when the manager is a ``MambaHybridCacheManager``."""
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    assert issubclass(KimiK3HybridCacheManager, MambaHybridCacheManager)
    assert issubclass(KimiK3HybridCacheManager, KVCacheManagerV2)
    if not hasattr(KimiK3HybridCacheManager, "accepts_appended_draft_layers"):
        pytest.skip("main has no drafter layers in the target manager (unified draft KV)")
    assert KimiK3HybridCacheManager.accepts_appended_draft_layers is True
    assert _is_property(KVCacheManagerV2, "draft_layer_ids")
    assert "kv_layout" in inspect.signature(KVCacheManagerV2.get_buffers).parameters
    for name in ("get_layer_kv_factor", "get_layer_page_index_scale", "get_draft_kv_cache_view"):
        assert callable(getattr(KVCacheManagerV2, name, None)), name
    assert "_seed_generation_slot" in inspect.getsource(DFlashSpecMetadata.prepare)
    worker = _k3_module("drafter").K3DSparkWorker
    assert "_ctx_unified_cache" in inspect.getsource(worker._seed_generation_slot)
    drafter = _k3_module("drafter").K3MLADSparkForCausalLM
    assert drafter.reads_managed_ctx_pages is not DFlashForCausalLM.reads_managed_ctx_pages
    assert (
        drafter.reads_helix_sharded_ctx_pages is not DFlashForCausalLM.reads_helix_sharded_ctx_pages
    )


def test_kda_inforward_snapshot_plan_on_the_private_metadata():
    """In-forward KDA snapshots (context, block reuse): ``KimiK3HybridCacheManager`` plans them
    whenever main's KDA kernels are present, whatever the model, and
    ``finish_inforward_snapshots`` then commits the prefix from the live slot and copies the
    staging rows back. ``Mamba2Metadata.prepare`` hands the plan to
    ``_prepare_inforward_split``, which the private metadata turns into a ``kda_snapshot`` plan
    on the live slots with the plan's staging rows; the KDA layers read those rows through
    ``inforward_snapshot_stage_views``."""
    import pytest
    import torch

    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        InforwardSnapshotEntry,
        MambaHybridCacheManagerV2,
    )

    assert issubclass(KimiK3HybridCacheManager, MambaHybridCacheManagerV2)
    for name in ("inforward_snapshot_stage_views", "state_index", "finish_inforward_snapshots"):
        assert callable(getattr(MambaHybridCacheManagerV2, name, None)), name
    assert "_prepare_inforward_split(" in inspect.getsource(Mamba2Metadata.prepare)
    metadata_cls = _k3_module("kda").K3MambaMetadata
    assert list(inspect.signature(metadata_cls._prepare_inforward_split).parameters) == list(
        inspect.signature(Mamba2Metadata._prepare_inforward_split).parameters
    )

    metadata = metadata_cls.__new__(metadata_cls)
    metadata._state_indices_long = torch.zeros(4, dtype=torch.long)
    metadata.kda_snapshot = None
    manager = SimpleNamespace(conv_state_shape=(12, 3), state_index={11: 7, 13: 2}.__getitem__)
    attn_metadata = SimpleNamespace(seq_lens=torch.tensor([200, 130, 96]), request_ids=[11, 12, 13])
    plan = {
        11: InforwardSnapshotEntry(row=1, point=1152, num_before=128, num_after=72),
        13: InforwardSnapshotEntry(row=0, point=64, num_before=64, num_after=32),
    }
    metadata._prepare_inforward_split(attn_metadata, manager, plan, 3, 3)
    snapshot = metadata.kda_snapshot
    assert snapshot.num_snapshots == 2
    assert snapshot.slots.tolist() == [7, 2]
    assert snapshot.stage_rows.tolist() == [1, 0]
    assert snapshot.token_indices.tolist() == [128, 330 + 64]
    assert snapshot.chunks.tolist() == [2, -1, 1]
    with pytest.raises(RuntimeError, match="generation requests"):
        metadata._prepare_inforward_split(attn_metadata, manager, plan, 3, 4)
    misaligned = {13: InforwardSnapshotEntry(row=0, point=48, num_before=48, num_after=48)}
    with pytest.raises(RuntimeError, match="does not match the plan"):
        metadata._prepare_inforward_split(attn_metadata, manager, misaligned, 3, 3)


@pytest.mark.parametrize("takes_flag", [False, True])
def test_dspark_helix_drafter_probes(takes_flag, monkeypatch):
    """The DSpark drafter under Helix (generation): its Helix exchange keeps the NCCL reformats
    eager only where main's ``_helix_post_process`` takes ``compile_nccl_reformat``, and calls it
    without the keyword otherwise."""
    drafter = _k3_module("drafter")
    fn, _ = drafter._helix_post_process_fn()
    assert callable(fn)
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return "out"

    monkeypatch.setattr(drafter, "_helix_post_process_fn", lambda: (fake, takes_flag))
    mask = object()
    assert drafter._draft_helix_post_process(1, 2, 3, 4, 5, mask) == "out"
    expected = {"zero_kv_mask": mask}
    if takes_flag:
        expected["compile_nccl_reformat"] = False
    assert calls == [((1, 2, 3, 4, 5), expected)]


def test_dspark_latent_cache_helix_mapping():
    """The DSpark drafter under Helix with ``TRTLLM_DISAGG_SHARED_DRAFT_KV=1`` (generation): main
    unifies the draft KV with a CP > 1 target only when the drafter reports
    ``reads_helix_sharded_ctx_pages`` (``_dflash_unify_blocker``), which the private drafter does
    when its latent cache is page-sharded. The sharding comes from the target's helix CP mapping:
    tp = 1 and cp over the repurposed TP group, on the target's CP ranks, with the target's
    all-to-all transport for ``TRTLLM_K3_ULL2_DRAFT_A2A``."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.mapping import CpType, Mapping

    shard = _k3_module("drafter")._latent_cache_helix_mapping
    if hasattr(KvCacheCreator, "_unified_drafter_reads_helix_pages"):
        assert "_unified_drafter_reads_helix_pages()" in inspect.getsource(
            KvCacheCreator._dflash_unify_blocker
        )
        assert "reads_helix_sharded_ctx_pages" in inspect.getsource(
            KvCacheCreator._unified_drafter_reads_helix_pages
        )
    else:
        # Main without the helix unified draft KV refuses to unify a CP > 1 target; the drafter
        # then keeps its own manager and the sharding below serves its latent cache alone.
        assert "cp_size" in inspect.getsource(KvCacheCreator._dflash_unify_blocker)
    for rank in (0, 5):
        helix = Mapping(
            world_size=8,
            rank=rank,
            gpus_per_node=4,
            tp_size=1,
            cp_size=8,
            cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False},
        )
        repurposed = helix.repurpose_helix_cp_to_tp()
        inherited = shard(repurposed, helix, inherit_cp_config=True)
        assert (inherited.tp_size, inherited.cp_size, inherited.pp_size) == (1, 8, 1)
        assert inherited.has_cp_helix()
        assert (inherited.cp_rank, inherited.rank) == (helix.cp_rank, rank)
        assert inherited.cp_config["use_nccl_for_alltoall"] is False
        assert shard(repurposed, helix).cp_config == {"cp_type": CpType.HELIX}
        assert shard(repurposed, None) is None
    plain = Mapping(world_size=8, rank=0, gpus_per_node=4, tp_size=8)
    assert shard(plain, None) is None


def test_router_picks_the_routing_kernel():
    """The routing each forward hands the routed experts (``K3MoERuntime._router``): the CuTe
    routing tail where ``_route_tail_target`` set a target, with its ``moe_sort`` tables where they
    are of the experts asked for; else ``noaux_tc_route_sort`` where it serves; else
    ``noaux_tc_op`` (the experts then run ``moe_sort``)."""
    import torch

    moe_runtime = _k3_module("moe_runtime")
    routing_types = _k3_module("moe")
    assert _k3_module("kernels.k3_route_tail").TILE == 128
    ids, weights = torch.zeros(2, 16, dtype=torch.int32), torch.ones(2, 16)
    tables = routing_types.SortTables(
        128, tuple(torch.zeros(1, dtype=torch.int32) for _ in range(6))
    )
    calls = []

    def route_tail(logits, offset, num):
        calls.append(("tail", offset, num))
        return routing_types.SortedRouting(ids, weights, tables)

    def route_with_sort(logits, offset, num):
        calls.append(("sort", offset, num))
        return routing_types.SortedRouting(ids, weights, tables) if logits.shape[0] <= 256 else None

    def route(logits):
        calls.append(("topk",))
        return ids, weights

    runtime = _bare(moe_runtime.K3MoERuntime)
    runtime.gate = SimpleNamespace(
        route=route, route_tail=route_tail, route_with_sort=route_with_sort
    )
    runtime.routed_experts = SimpleNamespace(consumes_sort_tables=True)
    logits = torch.empty(2, 896)

    assert runtime._router((0, 896))(logits, 0, 896).sort_tables is tables
    assert runtime._router((0, 896))(logits, 448, 448).sort_tables is None
    assert runtime._router(None)(logits, 0, 896).sort_tables is tables
    assert runtime._router(None)(torch.empty(300, 896), 0, 896).sort_tables is None
    assert calls == [
        ("tail", 0, 896),
        ("tail", 0, 896),
        ("sort", 0, 896),
        ("sort", 0, 896),
        ("topk",),
    ]
    calls.clear()
    runtime.routed_experts = SimpleNamespace(consumes_sort_tables=False)
    assert runtime._router(None)(logits, 0, 896).sort_tables is None
    assert calls == [("topk",)]


def test_dspark_split_plan_planner():
    """The DSpark drafter's split-KV plan (generation) runs the planner kernel of
    ``kernels/spec/mla_split_plan.py`` at the drafter's piece cost (``_launch_mla_split_plan``) on the
    CuTe DSL MLA decode runner's geometry (``_mla_split_plan_geometry``): the kernel takes the
    piece cost as an argument, and the runner exposes what the geometry reads."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    drafter = _k3_module("drafter")
    plan = _k3_module("kernels.spec.mla_split_plan")
    params = inspect.signature(plan._mla_split_kv_plan_kernel.fn).parameters
    for name in ("PIECE_COST", "MAX_PIECES", "BLOCK", "KV_TILE", "MIN_TILES"):
        assert name in params, name
    for name in (
        "_SPLIT_PLAN_MAX_REQUESTS",
        "_SPLIT_PLAN_MAX_SPLIT",
        "_SPLIT_PLAN_KV_TILE",
        "_SPLIT_PLAN_MIN_TILES",
    ):
        assert isinstance(getattr(plan, name, None), int), name
    runner = CuteDSLNVMlaDecodeBlackwellRunner
    assert list(inspect.signature(runner.get_num_q_tiles).parameters) == ["num_heads", "seq_len_q"]
    assert callable(runner._get_max_active_blocks)
    assert runner._CLUSTER_SHAPE_MNK[0] >= 1
    assert isinstance(drafter._MLA_SPLIT_PLAN_PIECE_COST_TILES, int)
    assert drafter._MLA_SPLIT_PLAN_PIECE_COST_TILES > 0


def test_dspark_drafter_declares_its_slot_convention():
    """DSpark's ``shift_label`` slot convention (generation): the engine sizes the unified draft
    KV from ``draft_model.shift_label`` (``get_draft_block_width``, K slots instead of K + 1), which
    main's ``DFlashForCausalLM`` initializes off and the drafter's DSpark heads set."""
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    drafter = _k3_module("drafter")
    assert issubclass(drafter.K3MLADSparkForCausalLM, DFlashForCausalLM)
    heads = drafter._DSparkHeadMixin.__new__(drafter._DSparkHeadMixin)
    heads._init_dspark_heads(SimpleNamespace())
    assert heads.shift_label is True
    heads._init_dspark_heads(SimpleNamespace(dflash_config={"shift_label": False}))
    assert heads.shift_label is False


def test_dspark_worker_takes_the_shared_slot_calls():
    """Main's ``DFlashSpecMetadata.prepare()`` drives the drafter's context slots by calling
    methods on the worker it is bound to, and on K3 that is the private ``K3DSparkWorker``: every
    ``worker.<method>(...)`` call there binds to the private method of that name. A missing
    method or a changed signature raises ``TypeError`` on the first step after the context
    buffers exist, so no DSpark run starts."""
    import ast
    import textwrap

    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    tree = ast.parse(textwrap.dedent(inspect.getsource(DFlashSpecMetadata.prepare)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "worker"
    ]
    names = {call.func.attr for call in calls}
    assert {"_assign_slot", "_write_ctx_len"} <= names, names

    worker = _k3_module("drafter").K3DSparkWorker
    unbound = []
    for call in calls:
        method = getattr(worker, call.func.attr, None)
        if not callable(method):
            unbound.append((call.func.attr, "missing"))
            continue
        args = [object()] * (1 + len(call.args))
        kwargs = {kw.arg: object() for kw in call.keywords if kw.arg is not None}
        try:
            inspect.signature(method).bind(*args, **kwargs)
        except TypeError as error:
            unbound.append((call.func.attr, str(error)))
    assert not unbound, unbound


# The FP8 block-scale projections of K3's MLA layer; ``KimiK3ForCausalLM._finalize_weight_load``
# swaps each one for ``_Fp8BlockScaleWeightReadLinear``.
_K3_MLA_PROJECTIONS = ("kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj", "g_proj")
_K3_MLA_MODULES = ("mla",)
# Read through ``getattr(proj, name, None)``; the bias-free FP8 projection has none.
_OPTIONAL_PROJECTION_READS = frozenset({"bias"})


def test_fp8_weight_read_linear_serves_the_k3_mla_projection_reads():
    """K3's MLA chains read attributes and methods of their projections, and on K3 those are
    ``_Fp8BlockScaleWeightReadLinear``: every one read exists on the private module.
    ``K3MLAAttention.cache_derived_state`` calls ``prepare_output_slice`` at post-load on every
    Helix rank, so one missing there stops every Helix GEN worker before it serves."""
    import re

    import torch

    source = "\n".join(inspect.getsource(_k3_module(name)) for name in _K3_MLA_MODULES)
    names = "|".join(_K3_MLA_PROJECTIONS)
    reads = set(re.findall(rf"(?:self|layer)\.(?:{names})\.(\w+)\b", source))
    reads |= set(re.findall(r"\bproj\.(\w+)\b", source))
    reads |= set(
        re.findall(rf"getattr\((?:(?:self|layer)\.(?:{names})|kv_b_proj), \"(\w+)\"", source)
    )
    assert {
        "prepare_output_slice",
        "forward_slice",
        "forward_mxfp8",
        "forward_gated_mxfp8",
        "gated_mxfp8_applies",
        "takes_mid_mxfp8_input",
        "supports_prequantized_input",
    } <= reads, reads

    linear = _k3_module("fp8_weight_read")._Fp8BlockScaleWeightReadLinear
    projection = linear(torch.zeros(256, 128, dtype=torch.float8_e4m3fn), torch.ones(2, 1), 256)
    # Set by ``from_linear``, the constructor K3's swap uses.
    projection.quant_config, projection.tp_size, projection.tp_rank = None, 1, 0
    missing = sorted(
        attr
        for attr in reads
        if attr not in _OPTIONAL_PROJECTION_READS and not hasattr(projection, attr)
    )
    assert not missing, missing

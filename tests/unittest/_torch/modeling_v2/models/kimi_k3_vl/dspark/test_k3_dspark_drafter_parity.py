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
"""The family's MLA DSpark drafter (``dspark.drafter.K3DSparkDrafter``) against the SM107 target's
drafter (``kimi_k3_nvfp4__sm_107__default.drafter.K3MLADSparkForCausalLM``) on the same weights and
inputs, bit for bit, on a BF16 latent pool: two layers of the real drafter's shapes (64 heads, MLA
576, hidden 7168), a small vocab and MLP, four requests with 517 / 1203 / 64 / 333 context tokens,
block 7, page 32.

Each case sets the old drafter's feature profile fields and the new drafter's ``K3DSparkPath`` to the
same deployment path:

* ``tp1``: one GPU, no fusion; the block placed by the worker (``block_placement``) and built by the
  drafter.
* ``dep16``: one GPU under attention DP, ``spec_tail_fusion``; the worker's ``block_fixup`` and the
  drafter's own.
* ``tp4`` (4 GPUs): drafter TP4, ``spec_tail_fusion`` + ``tail_v2``; MNNVL fused all-reduce and the
  pushed o_proj / down sites.
* ``helix8`` (4 GPUs): as ``tp4`` with the target's helix cp=4 (latent pages sharded, NCCL helix merge).

Compared: the projected capture (``project_target_hidden``), the context latents (both layouts), the
block hidden states of every variant, the block rows stored into the pool, the Markov-biased logits of
the workers' bias and the eager chain, and a CUDA-graph replay of the new drafter.
"""

import dataclasses
import importlib
import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

_K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
OLD = f"{_K3}.kimi_k3_nvfp4__sm_107__default"
NEW = f"{_K3}.dspark"

HIDDEN, INTER, LAYERS, HEADS, VOCAB, RANK = 7168, 2048, 2, 64, 4096, 16
Q_LORA, KV_LORA, NOPE, ROPE, V_DIM = 1536, 512, 128, 64, 128
TARGET_LAYERS = (0, 1)
CTX_LENS = (517, 1203, 64, 333)
BLOCK, PAGE, MAX_SEQ = 7, 32, 16384

# case -> (world, attention DP, helix, old profile fields, path fields, variants)
CASES = {
    "tp1": (1, False, False, {}, {}, ("build", "placement")),
    "dep16": (
        1,
        True,
        False,
        {"spec_tail_fusion": True},
        {"spec_tail_fusion": True},
        ("build", "fixup"),
    ),
    "tp4": (
        4,
        False,
        False,
        {"spec_tail_fusion": True, "dspark_tail_v2": True},
        {"spec_tail_fusion": True, "tail_v2": True},
        ("build", "fixup"),
    ),
    "helix8": (
        4,
        False,
        True,
        {"spec_tail_fusion": True, "dspark_tail_v2": True},
        {"spec_tail_fusion": True, "tail_v2": True},
        ("build", "fixup"),
    ),
}


def drafter_config_dict():
    """The drafter checkpoint's config.json (Inferact/Kimi-K3-DSpark) at two layers, a 4096 vocab."""
    return dict(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        mla_use_nope=False,
        mla_use_output_gate=False,
        vocab_size=VOCAB,
        rms_norm_eps=1e-5,
        max_position_embeddings=1048576,
        rope_theta=50000.0,
        num_target_layers=len(TARGET_LAYERS),
        target_hidden_size=HIDDEN,
        target_layer_ids=list(TARGET_LAYERS),
        mask_token_id=VOCAB - 3,
        markov_rank=RANK,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        tie_word_embeddings=False,
        draft_vocab_size=VOCAB,
        torch_dtype="bfloat16",
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


def hashed(n, seed, device):
    x = torch.arange(n, device=device, dtype=torch.int64)
    x = (x * 2654435761 + seed * 40503 + 12345) & 0xFFFFFFFF
    for _ in range(2):
        x = x ^ (x >> 16)
        x = (x * 0x45D9F3B) & 0xFFFFFFFF
    return x ^ (x >> 16)


def normalish(shape, seed, scale, device="cuda", offset=0.0):
    """Sum of four hashed uniforms in [-1, 1): bell-shaped, std ~1.15, times ``scale``; BF16. The
    same values on every rank and in every run."""
    n = torch.Size(shape).numel()
    acc = torch.zeros(n, device=device, dtype=torch.float32)
    for i in range(4):
        acc += (hashed(n, seed * 4 + i, device) % 65536).to(torch.float32) / 32768.0 - 1.0
    return (acc * scale + offset).to(torch.bfloat16).view(shape)


def drafter_weights():
    """A checkpoint of the drafter (HF MLA names under ``model.``, the TorchSpec head names)."""
    seed = iter(range(1000))
    w = {}

    def put(name, *shape, scale=0.02, offset=0.0):
        w[name] = normalish(shape, next(seed), scale, offset=offset)

    put("fc.weight", HIDDEN, len(TARGET_LAYERS) * HIDDEN, scale=0.01)
    put("hidden_norm.weight", HIDDEN, scale=0.05, offset=1.0)
    put("model.embed_tokens.weight", VOCAB, HIDDEN, scale=0.5)
    put("model.norm.weight", HIDDEN, scale=0.05, offset=1.0)
    put("markov_head.markov_w1.weight", VOCAB, RANK, scale=0.5)
    put("markov_head.markov_w2.weight", VOCAB, RANK, scale=0.5)
    put("confidence_head.proj.weight", 1, HIDDEN + RANK)
    put("confidence_head.proj.bias", 1)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        put(p + "self_attn.q_a_proj.weight", Q_LORA, HIDDEN)
        put(p + "self_attn.q_a_layernorm.weight", Q_LORA, scale=0.05, offset=1.0)
        put(p + "self_attn.q_b_proj.weight", HEADS * (NOPE + ROPE), Q_LORA, scale=0.03)
        put(p + "self_attn.kv_a_proj_with_mqa.weight", KV_LORA + ROPE, HIDDEN)
        put(p + "self_attn.kv_a_layernorm.weight", KV_LORA, scale=0.05, offset=1.0)
        put(p + "self_attn.kv_b_proj.weight", HEADS * (NOPE + V_DIM), KV_LORA, scale=0.04)
        put(p + "self_attn.o_proj.weight", HIDDEN, HEADS * V_DIM, scale=0.01)
        put(p + "mlp.gate_proj.weight", INTER, HIDDEN)
        put(p + "mlp.up_proj.weight", INTER, HIDDEN)
        put(p + "mlp.down_proj.weight", HIDDEN, INTER, scale=0.02)
        put(p + "input_layernorm.weight", HIDDEN, scale=0.05, offset=1.0)
        put(p + "post_attention_layernorm.weight", HIDDEN, scale=0.05, offset=1.0)
    return w


def mappings(world, rank, adp, helix):
    from tensorrt_llm.mapping import CpType, Mapping

    if world == 1:
        draft = Mapping(world_size=1, rank=0, tp_size=1, enable_attention_dp=adp)
    else:
        draft = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
    target_helix = None
    if helix:
        # The target's helix CP mapping as the helix8 recipe sets it (the drafter's own merge is
        # NCCL whatever the target's flag).
        target_helix = Mapping(
            world_size=world,
            rank=rank,
            gpus_per_node=world,
            tp_size=1,
            cp_size=world,
            cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False},
        )
    return draft, target_helix


def build_pair(case, rank):
    """``(old drafter, new drafter, lm_head weight)`` of one case, loaded with the same checkpoint."""
    from tensorrt_llm._torch.configs import K3DsparkConfig
    from tensorrt_llm._torch.model_config import ModelConfig

    world, adp, helix, profile, path_fields, _ = CASES[case]
    features = importlib.import_module(f"{OLD}.features")
    features._ACTIVE = dataclasses.replace(features.K3Features(), **profile)
    old_mod = importlib.import_module(f"{OLD}.drafter")
    new_mod = importlib.import_module(f"{NEW}.drafter")
    draft_map, helix_map = mappings(world, rank, adp, helix)
    weights = drafter_weights()
    lm_w = normalish((VOCAB, HIDDEN), 7, 0.02)
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(lm_w, requires_grad=False)
    spec = SimpleNamespace(attention_backend="AUTO", draft_is_embedded_in_target=False)

    def config():
        cfg = K3DsparkConfig(**drafter_config_dict())
        cfg.torch_dtype = torch.bfloat16
        return ModelConfig(pretrained_config=cfg, mapping=draft_map, max_seq_len=MAX_SEQ)

    with torch.device("cuda"):
        old = old_mod.build_k3_dspark_draft(
            SimpleNamespace(spec_config=spec, _helix_mapping_with_cp=helix_map),
            config(),
            None,
            None,
        )
        new = new_mod.build_k3_dspark_drafter(
            config(), spec, path=new_mod.K3DSparkPath(**path_fields), helix_mapping=helix_map
        )
    for drafter in (old, new):
        drafter.load_weights(weights=dict(weights))
        drafter.load_weights_from_target_model(SimpleNamespace(lm_head=head))
    return old, new, lm_w


class Inputs:
    """One draft step's inputs and a latent pool holding the context latents, per drafter."""

    def __init__(self, drafter, dtype=torch.bfloat16):
        dev = torch.device("cuda")
        self.batch = len(CTX_LENS)
        self.ctx = torch.tensor(CTX_LENS, dtype=torch.int32, device=dev)
        n_ctx = sum(CTX_LENS)
        self.captured = normalish((n_ctx, len(TARGET_LAYERS) * HIDDEN), 1, 1.0)
        self.positions = torch.cat(
            [torch.arange(c, device=dev, dtype=torch.int32) for c in CTX_LENS]
        )
        helix = drafter._helix_mapping
        self.cp, self.cp_rank = (helix.cp_size, helix.cp_rank) if helix is not None else (1, 0)
        pages_global = [-(-(c + BLOCK) // PAGE) for c in CTX_LENS]
        local = [max(0, -(-(g - self.cp_rank) // self.cp)) for g in pages_global]
        table = torch.zeros(self.batch, max(local) + 1, dtype=torch.int32, device=dev)
        phys = list(range(1, 1 + sum(local)))[::-1]  # reversed: a non-trivial page table
        i = 0
        for b, n in enumerate(local):
            table[b, :n] = torch.tensor(phys[i : i + n], dtype=torch.int32, device=dev)
            i += n
        self.table = table
        self.num_pages = 1 + sum(local)
        self.dtype = dtype
        mask_id = int(drafter.mask_token_id)
        self.bonus = (hashed(self.batch, 21, dev) % (VOCAB - 3)).to(torch.int64)
        ids = torch.full((self.batch, BLOCK), mask_id, dtype=torch.int64, device=dev)
        ids[:, 0] = self.bonus
        self.ids = ids
        self.query_positions = self.ctx.view(-1, 1) + torch.arange(
            BLOCK, device=dev, dtype=torch.int32
        )

    def pools(self, latent):
        """Per-layer ``[pages, 1, 1, page, 576]`` pools: hashed filler, the context latents stored
        on this rank's pages (``to_draft_fp8`` of them in an FP8 pool)."""
        from importlib import import_module

        to_fp8 = import_module(f"{NEW}.primitives").to_draft_fp8
        dev = latent.device
        req = torch.repeat_interleave(torch.arange(self.batch, device=dev), self.ctx.long())
        pos = self.positions.long()
        page_g = pos // PAGE
        owned = page_g % self.cp == self.cp_rank
        phys = self.table.long()[req, page_g // self.cp]
        pools = []
        for layer in range(latent.shape[1]):
            pool = normalish((self.num_pages, 1, 1, PAGE, KV_LORA + ROPE), 100 + layer, 1.0)
            pool[phys[owned], 0, 0, (pos % PAGE)[owned]] = latent[owned, layer, 0]
            if self.dtype == torch.float8_e4m3fn:
                pool = to_fp8(pool)
            pools.append(pool)
        return pools


def forward(drafter, inp, pools, noise, variant, fixup, placement):
    kwargs = {}
    if variant == "placement":
        kwargs["block_placement"] = placement
    elif variant == "fixup":
        kwargs["block_fixup"] = fixup
    return drafter.dflash_forward(
        noise_embedding=noise,
        query_positions=inp.query_positions,
        num_ctx_per_req=inp.ctx,
        ctx_k_cache=torch.empty(0, device="cuda"),
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(inp.batch, device="cuda"),
        ctx_kv_cache=pools,
        ctx_page_table=inp.table,
        **kwargs,
    )


def run_drafter(drafter, inp, variants, worker_cls, lm_w, graph=False):
    """Everything a draft step computes from ``drafter``, by name."""
    out = {}
    projected = drafter.project_target_hidden(inp.captured)
    out["ctx_projected"] = projected
    latent, _ = drafter.precompute_context_kv(projected, inp.positions)
    out["ctx_latent"] = latent
    layer_major_fn = getattr(drafter, "precompute_context_kv_layer_major", None)
    layer_major = None if layer_major_fn is None else layer_major_fn(projected, inp.positions)
    if layer_major is not None:
        out["ctx_latent_layer_major"] = layer_major
    embed = drafter.draft_model_full.model.embed_tokens(inp.ids.reshape(-1))
    out["embed"] = embed
    noise = embed.view(inp.batch, BLOCK, HIDDEN)
    shard_cls = importlib.import_module(f"{NEW}.primitives").HelixDraftShard
    fixup_fn = importlib.import_module(f"{NEW}.drafter")._build_mla_block_fixup
    shard = None if inp.cp == 1 else shard_cls(inp.cp, inp.cp_rank, PAGE)
    fixup = fixup_fn(inp.ctx, inp.table.long(), BLOCK, PAGE, shard)
    placement = importlib.import_module(
        f"{OLD}.kernels.spec.spec_tail_kernels"
    ).DraftBlockPlacement(fixup.pages, fixup.offsets, fixup.seq_lens_i32, fixup.kv_bounds_i32)
    blk_pages, blk_offsets = fixup.pages.reshape(-1), fixup.offsets.reshape(-1)
    stub = SimpleNamespace()

    def outputs(tag, hidden_out, pools):
        out[f"hidden_{tag}"] = hidden_out
        out[f"pool_blk_{tag}"] = torch.stack(
            [p[:, 0, 0][blk_pages, blk_offsets].view(torch.uint8) for p in pools]
        )
        logits = F.linear(hidden_out.view(inp.batch, BLOCK, HIDDEN), lm_w).float()
        biased = worker_cls._apply_dspark_markov_bias(stub, drafter, logits, inp.bonus, None)
        out[f"biased_{tag}"] = biased
        out[f"tokens_eager_{tag}"] = drafter.apply_markov_chain_logits(
            logits.clone(), inp.bonus
        ).argmax(-1)

    for variant in variants:
        # Two eager calls: the first builds the drafter's lazy state (fused weights, plans).
        forward(drafter, inp, inp.pools(latent), noise.clone(), variant, fixup, placement)
        pools = inp.pools(latent)
        outputs(
            variant, forward(drafter, inp, pools, noise.clone(), variant, fixup, placement), pools
        )
    if graph:
        if inp.cp > 1 or drafter.model_config.mapping.world_size > 1:
            MPI.COMM_WORLD.Barrier()
        pools = inp.pools(latent)
        static_noise = noise.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_g = forward(drafter, inp, pools, static_noise, variants[-1], fixup, placement)
        pools_ref = inp.pools(latent)
        for p, ref in zip(pools, pools_ref):
            p.copy_(ref)
        g.replay()
        torch.cuda.synchronize()
        outputs("graph", out_g, pools)
        del g
    torch.cuda.synchronize()
    return out


def differing(a, b):
    """Names whose values are not bitwise equal (NaN payloads included)."""
    bad = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            bad.append(f"{name}: only in {'old' if name in a else 'new'}")
            continue
        x, y = a[name].detach(), b[name].detach()
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append(f"{name}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}")
            continue
        xb = x.contiguous().view(torch.uint8)
        yb = y.contiguous().view(torch.uint8)
        if not torch.equal(xb, yb):
            n = int((xb != yb).sum())
            bad.append(f"{name}: {n} of {xb.numel()} bytes differ")
    return bad


def run_case(case, rank=0):
    """Old against new for ``case`` on this rank; returns the list of differing outputs."""
    world, _, _, _, _, variants = CASES[case]
    if world > 1:
        os.environ["TLLM_TEST_MNNVL"] = "1"
        os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
        torch.cuda.set_device(rank)
    old_worker = importlib.import_module(f"{OLD}.drafter").K3DSparkWorker
    new_worker = importlib.import_module(f"{NEW}.worker").K3DSparkWorker
    with torch.inference_mode():
        old, new, lm_w = build_pair(case, rank)
        inp = Inputs(new)
        ref = run_drafter(old, inp, variants, old_worker, lm_w)
        got = run_drafter(new, inp, variants, new_worker, lm_w, graph=True)
    graph = {k[: -len("graph")] + variants[-1]: v for k, v in got.items() if k.endswith("graph")}
    got = {k: v for k, v in got.items() if not k.endswith("graph")}
    bad = differing(ref, got)
    bad += [f"graph replay {m}" for m in differing({k: got[k] for k in graph}, graph)]
    return bad


def _run_rank(case):
    rank = MPI.COMM_WORLD.Get_rank()
    try:
        return run_case(case, rank)
    except Exception:
        traceback.print_exc()
        raise


_sm100f = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the cute-dsl MLA block decode runs on SM100, SM103 and SM107",
)


@_sm100f
@pytest.mark.parametrize("case", ["tp1", "dep16"])
def test_drafter_matches_the_target_drafter(case):
    assert run_case(case) == []


@_sm100f
@pytest.mark.skipif(
    torch.cuda.device_count() < 4 or os.environ.get("TLLM_TEST_SKIP_MNNVL") == "1",
    reason="needs four GPUs with MNNVL",
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
@pytest.mark.parametrize("case", ["tp4", "helix8"])
def test_tp_drafter_matches_the_target_drafter(mpi_pool_executor, case):
    results = list(mpi_pool_executor.map(_run_rank, *zip(*[(case,)] * 4)))
    assert results == [[]] * 4, results

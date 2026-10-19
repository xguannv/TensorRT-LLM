# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiLinearForCausalLM — Kimi K3 text model, PyTorch backend.

Runtime integration of the Kimi K3 hybrid architecture for the standard
TRT-LLM PyTorch-backend flow (``LLM(model=<ckpt>) -> generate``):

* 93 decoder layers: 69 KDA (Kimi Delta Attention, linear attention) layers
  and 24 MLA (absorbed-MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 uses a dense SiTU MLP; layers 1..92 use the 896-expert latent MoE
  (top-16, sigmoid + e_score_correction_bias routing, MXFP4 routed experts,
  2 shared experts, latent down/norm/up projections).
* The attention-residual ("attn_res") scheme from the HF reference
  ``modeling_kimi.py`` is applied per token: snapshot mixing before
  ``input_layernorm`` / ``post_attention_layernorm`` and at the model output,
  with a new snapshot appended whenever ``layer_idx % attn_res_block_size == 0``.

Caching
-------
KDA states live on the mamba side of a ``MixedMambaHybridCacheManager``
(wired in ``pyexecutor/_util.py``): per layer, a short-conv slot of
``[3 * num_heads * head_dim, W]`` bf16 (the full FLA ``ShortConvolution``
cache window, sections ``[q | k | v]``) and a delta-rule recurrent slot of
``[num_heads, head_dim, head_dim]`` fp32 (``[H, V, K]``, the
``state_v_first`` FLA layout). MLA layers use the paged-KV side with
``num_kv_heads=1`` and ``head_dim = kv_lora_rank + qk_rope_head_dim`` (576),
SELFKONLY, exactly like DeepSeek MLA.

MLA prefill routing
-------------------
The in-tree ``KimiK3MLAAttention`` routes prefill through the normal
unabsorbed MLA context FMHA. It consumes the executor's original mixed-batch
metadata, letting the shared MLA implementation dispatch context and cached
generation work in one forward call.

Parallelism
-----------
The routed-expert bank supports a configurable MoE TP x EP split
(``moe_tp_size * moe_ep_size == mapping.tp_size``); the default is EP-only
(``moe_ep_size == mapping.tp_size``), the historical K3 layout. Under EP each
MoE layer holds a contiguous ``num_experts / moe_ep_size`` slice of the MXFP4
expert bank (whole experts); under MoE TP each rank holds ALL experts, with
w1/w3 column-sharded and w2 row-sharded along the intermediate dim
(``intermediate / moe_tp_size`` per rank; group-32 MXFP4 packed bytes and
scales sliced consistently by the stock TRTLLM-Gen quant-method loaders).
The split is EP-only unless the user sets ``moe_tensor_parallel_size`` /
``moe_expert_parallel_size`` explicitly. Routing is computed replicated; the
routed partial sums — EP partials of whole experts, or TP partials over the
intermediate shards — are all-reduced in the latent space (before
``routed_expert_norm`` / ``routed_expert_up_proj``, which are nonlinear/linear
layers applied to the full sum). When attention DP is off, the shared experts
use standard MLP TP over the model TP group: gate/up are column-sharded and down
is row-sharded. The shared down projection reduces on the auxiliary stream
while the routed expert chain runs on the main stream. After the streams join,
the routed latent partial is reduced before its norm/up projection. With
``TRTLLM_K3_MERGED_MOE_AR=1`` the shared expert leaves its partial unreduced and
one all-reduce over ``[latent partial | shared partial]`` serves both.
Fused-communication routed backends already return a complete routed result and
need no outer reduction. Under attention DP the shared experts stay replicated.
``lm_head`` uses the stock ``LMHead`` (vocab-sharded + gather), so logits are
identical on all ranks.

Speculative decoding: SA (suffix automaton, one-engine, draft-weight-free);
the KDA/MLA runtimes implement multi-token verification with deferred
state promotion.

Chunked prefill is supported: continuation chunks feed the previous KDA
conv/recurrent state back into the FLA kernels (``use_initial_states``)
and the MLA prefill path natively attends over the cached latent prefix
(``kv_len = cached + q_len``). KV-cache block reuse is supported as an
opt-in via ``kv_cache_config.enable_block_reuse=true``, which routes to
the unified-pool ``CppMambaHybridCacheManager`` (per-block KDA state
snapshots every ``mamba_state_cache_interval`` tokens, FORCE_CHUNK
context chunking).

Not supported: pipeline parallelism, draft-head spec-dec modes
(MTP/Eagle — no draft-head checkpoint exists). SA speculative decoding
is validated only without block reuse (Mixed cache manager).
"""

from __future__ import annotations

import copy
import functools
import gc
import json
import math
import os
import threading
import weakref
from contextlib import ExitStack
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

import torch
from safetensors import safe_open
from torch import nn

from ..._utils import get_sm_version, is_sm_100f
from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantAlgo, QuantConfig
from ..attention_backend import AttentionMetadata
from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from ..distributed.nvls_allreduce import NVLSAllReduce
from ..distributed.ops import LamportResult
from ..kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV, get_kimi_k3_fp8_kv_layer_ids
from ..model_config import ModelConfig
from ..modules.fused_moe import ConfigurableMoE, create_moe
from ..modules.fused_moe.interface import MoESchedulerKind
from ..modules.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from ..modules.gated_mlp import GatedMLP
from ..modules.kimi_k3_expert_placement import (
    expert_placement_for_layer,
    expert_replicas_for_layer,
    get_routing_recorder,
    permute_router_to_slots,
)
from ..modules.kimi_kda import KimiKDALinearAttention
from ..modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from ..modules.linear import Linear as TrtllmLinear
from ..modules.linear import TensorParallelMode
from ..modules.multi_stream_utils import do_multi_stream, maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.situ import SituAndMul
from ..speculative.spec_head_kernels import embedding_gather, spec_head_fusion_enabled
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model, run_concurrently

_K3_FUSE_MOE_RESIDUAL = os.environ.get("TRTLLM_K3_FUSE_MOE_RESIDUAL", "0") == "1"

# MoE block with a TP-sharded shared expert and an outer routed reduction: one all-reduce over
# [routed latent partial | shared-expert partial] replaces the latent all-reduce (+RMSNorm) and
# the shared expert's own all-reduce (see ``KimiK3MoERuntime._merged_reduce``).
_K3_MERGED_MOE_AR = os.environ.get("TRTLLM_K3_MERGED_MOE_AR", "0") == "1"
# Largest token count that takes the merged all-reduce (0: no limit). At a few hundred tokens
# the reduction is bandwidth-bound, so merging saves little, and it loses the overlap of the
# shared expert's own all-reduce with the routed experts when that chain is the longer one.
_K3_MERGED_MOE_AR_MAX_TOKENS = int(os.environ.get("TRTLLM_K3_MERGED_MOE_AR_MAX_TOKENS", "128"))

# Zero residual operand of the merged MoE all-reduce, one per device, shared by every layer.
# It only grows outside CUDA graph capture, and a replaced buffer is kept alive because
# graphs captured earlier still read it.
_K3_ZERO_RESIDUALS: Dict[torch.device, List[torch.Tensor]] = {}


def _k3_zero_residual(numel: int, device: torch.device) -> Optional[torch.Tensor]:
    """``numel`` BF16 zeros, or ``None`` if that needs a new buffer during graph capture."""
    buffers = _K3_ZERO_RESIDUALS.setdefault(device, [])
    if not buffers or buffers[-1].numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            return None
        size = max(numel, 2 * buffers[-1].numel()) if buffers else numel
        buffers.append(torch.zeros(size, dtype=torch.bfloat16, device=device))
    return buffers[-1][:numel]


# SM100/SM103 FP8 weight-read projections with few rows (per-shape limit, see
# k3_fp8w_lowm_gemm.can_run) run the CuTe low-M GEMM instead of DeepGEMM swap-AB.
_K3_FP8W_LOWM_GEMM = os.environ.get("TRTLLM_K3_FP8W_LOWM_GEMM", "0") == "1"
# With the low-M GEMM on, also serve projections marked ``fp8w_lowm_partial_n_tile`` (the MLA
# fused q_a/kv_a projection, 2112 rows) although N is not a multiple of 128.
_K3_FP8W_LOWM_FUSED_A = os.environ.get("TRTLLM_K3_FP8W_LOWM_FUSED_A", "0") == "1"

# SMs the DeepGEMM swap-AB GEMM of the SM100/SM103 FP8 weight-read projections leaves free
# (default 0). DeepGEMM compiles its static persistent grid for a fixed SM count, one CTA per SM;
# while another kernel holds SMs, the CTAs that land behind it run as a second wave. A CTX worker
# with the DWDP SM gather (one kernel per MoE layer, resident for most of a prefill iteration)
# sets this to TRTLLM_DWDP_GATHER_BLOCKS: the grid then fits beside the gather, at the cost of
# fewer SMs while no gather runs. Projections with fewer than TRTLLM_K3_DEEPGEMM_SM_RESERVE_MIN_K
# input features keep the full grid: with so few K blocks per tile the GEMM is bound by its
# output stores, which a smaller grid slows down (MLA kv_b, K = 512: +24% without the gather).
_K3_DEEPGEMM_SM_RESERVE = int(os.environ.get("TRTLLM_K3_DEEPGEMM_SM_RESERVE", "0"))
if _K3_DEEPGEMM_SM_RESERVE < 0:
    raise ValueError(f"TRTLLM_K3_DEEPGEMM_SM_RESERVE must be >= 0, got {_K3_DEEPGEMM_SM_RESERVE}")
_K3_DEEPGEMM_SM_RESERVE_MIN_K = int(os.environ.get("TRTLLM_K3_DEEPGEMM_SM_RESERVE_MIN_K", "1024"))

# Helix MLA generation applies the V up-projection of all heads before the CP
# all-to-all (see ``MLA._helix_project_v_partials``), which needs every head's
# BF16 V weight on each rank.
_K3_HELIX_V_UP_BEFORE_A2A = os.environ.get("TRTLLM_HELIX_V_UP_BEFORE_A2A", "0") == "1"

# Small decode batches route with one fused CuTe launch: router GEMV + noaux_tc top-16
# ("topk"), plus the CuteDSL MoE's moe_sort metadata of this rank ("sort").
_K3_FUSED_ROUTER = os.environ.get("TRTLLM_K3_FUSED_ROUTER", "0")
if _K3_FUSED_ROUTER not in ("0", "topk", "sort"):
    raise ValueError(f"TRTLLM_K3_FUSED_ROUTER must be 0, topk or sort, got {_K3_FUSED_ROUTER!r}")

# Decode-batch routing tail of the CuteDSL MoE (see k3_route_tail): "1" replaces noaux_tc_op ->
# moe_sort with one launch on a cluster of at most eight CTAs (same logits, same routing);
# "gemv" also computes the logits of M <= 32 tokens with a 40-CTA GEMV that runs beside the
# latent down-projection, which then quantizes the routed input right away.
_K3_ROUTE_TAIL = os.environ.get("TRTLLM_K3_ROUTE_TAIL", "0")
if _K3_ROUTE_TAIL not in ("0", "1", "gemv"):
    raise ValueError(f"TRTLLM_K3_ROUTE_TAIL must be 0, 1 or gemv, got {_K3_ROUTE_TAIL!r}")
if _K3_ROUTE_TAIL != "0" and _K3_FUSED_ROUTER != "0":
    raise ValueError("TRTLLM_K3_ROUTE_TAIL and TRTLLM_K3_FUSED_ROUTER are mutually exclusive")
# Largest token count the routing tail serves: up to 256 tokens one cluster, above that (at most
# 2048 tokens, see k3_route_tail.wide_sort_fits) a grid of CTAs whose last CTA does the sort.
_K3_ROUTE_TAIL_MAX_M = int(os.environ.get("TRTLLM_K3_ROUTE_TAIL_MAX_M", "2048"))
# With TRTLLM_K3_ROUTE_TAIL=1: "umma" computes the router logits of up to
# TRTLLM_K3_ROUTER_GEMM_MAX_M tokens as split-K partials of the 35-CTA tcgen05 router GEMM
# (cute_dsl_kernels/blackwell/k3_router_umma.py) instead of the cuBLAS GEMM (about 112 CTAs, which
# slow the latent down-projection running beside it) + split-K reduction; the routing tail sums
# the partials. Same stream order as the cuBLAS GEMM (the shared expert forks after it); the
# logits differ from cuBLAS at FP32 rounding level. The default limit of 64 tokens keeps 128-token
# steps (b16) on cuBLAS: there the 35-CTA GEMM and the tail on its partials end later than the
# cuBLAS chain and the ranks' shared-expert chains drift apart (gen-only b16: +5.5 ms per step).
_K3_ROUTER_GEMM = os.environ.get("TRTLLM_K3_ROUTER_GEMM", "cublas")
if _K3_ROUTER_GEMM not in ("cublas", "umma"):
    raise ValueError(f"TRTLLM_K3_ROUTER_GEMM must be cublas or umma, got {_K3_ROUTER_GEMM!r}")
_K3_ROUTER_GEMM_MAX_M = int(os.environ.get("TRTLLM_K3_ROUTER_GEMM_MAX_M", "64"))
# With TRTLLM_K3_ROUTE_TAIL=gemv: "1" launches only the router GEMV (k3_route_tail.ROUTER_GEMV)
# before the shared expert forks and runs the routing tail on its split-K partials when the routed
# experts ask for the routing, so the shared expert starts right after the GEMV.
_K3_ROUTE_GEMV_DEFER_TAIL = os.environ.get("TRTLLM_K3_ROUTE_GEMV_DEFER_TAIL", "0") == "1"

# Fused latent projections (cute_dsl_kernels/blackwell/k3_latent_fp8_gemm.py), for token counts
# in [TRTLLM_K3_LD_FUSED_MIN_M, TRTLLM_K3_LD_FUSED_MAX_M]:
#   TRTLLM_K3_LD_FUSED: latent_down + the routed experts' NVFP4 input quantization in one chain
#     (trtllm::k3_latent_down_fp4), bit-exact with fp8_swap_ab_gemm followed by fp4_quantize.
#   TRTLLM_K3_LU_FUSED: latent_up + the two BF16 adds of the MoE residual update
#     (trtllm::k3_latent_up_residual), bit-exact with fp8_swap_ab_gemm and the two adds.
# "1" fuses only where the unfused projection would run fp8_swap_ab_gemm; "2" also replaces the
# low-M kernel (TRTLLM_K3_FP8W_LOWM_GEMM), whose 1x32 activation scales then become the 1x128
# scales of the larger-M path.
_K3_LD_FUSED = os.environ.get("TRTLLM_K3_LD_FUSED", "0")
_K3_LU_FUSED = os.environ.get("TRTLLM_K3_LU_FUSED", "0")
for _name, _value in (("TRTLLM_K3_LD_FUSED", _K3_LD_FUSED), ("TRTLLM_K3_LU_FUSED", _K3_LU_FUSED)):
    if _value not in ("0", "1", "2"):
        raise ValueError(f"{_name} must be 0, 1 or 2, got {_value!r}")
_K3_LD_FUSED_MIN_M = int(os.environ.get("TRTLLM_K3_LD_FUSED_MIN_M", "1"))
_K3_LD_FUSED_MAX_M = int(os.environ.get("TRTLLM_K3_LD_FUSED_MAX_M", "4096"))
# Up to this M the low-M kernel's cluster split-K beats the fused chains (which use one CTA per
# 128-row x 128..256-column tile) even in mode "2".
_K3_LOWM_OVER_FUSED_MAX_M = 128

# Mid-M FP8 GEMM (cute_dsl_kernels/blackwell/k3_fp8_mid_gemm.py: persistent 2-CTA swap-AB MXFP8,
# no split-K, bit-exact with fp8_swap_ab_gemm) for token counts in [TRTLLM_K3_FP8_MID_MIN_M,
# TRTLLM_K3_FP8_MID_MAX_M] where its tuned table (k3_fp8_mid_gemm.tile_for) measured it faster
# than the production path. TRTLLM_K3_FP8_MID_GEMM is a comma list of projection families:
#   lu: latent_up + the MoE residual update (replaces k3_latent_up_residual / the unfused path)
#   o: attention o_proj   qb: MLA q_b_proj   qa: MLA fused q_a/kv_a   g: MLA output gate
#   qkvg: KDA fused q/k/v/g projection
_K3_FP8_MID_FAMILIES = frozenset(
    f for f in os.environ.get("TRTLLM_K3_FP8_MID_GEMM", "").split(",") if f
)
_K3_FP8_MID_KNOWN = frozenset(("lu", "o", "qb", "qa", "g", "qkvg"))
if not _K3_FP8_MID_FAMILIES <= _K3_FP8_MID_KNOWN:
    raise ValueError(
        f"TRTLLM_K3_FP8_MID_GEMM families must be in {sorted(_K3_FP8_MID_KNOWN)}, "
        f"got {sorted(_K3_FP8_MID_FAMILIES)}"
    )
_K3_FP8_MID_MIN_M = int(os.environ.get("TRTLLM_K3_FP8_MID_MIN_M", "1"))
# With the "lu" family on: the routed all-reduce + RMSNorm also quantizes the latent to FP8
# (MNNVL RMS_NORM_QUANT_FP8_BLOCKSCALE: 1x128 UE8M0 block scales in DeepGEMM's packed layout,
# bit-identical to quantizing the BF16 norm output separately), and the mid-M GEMM reads it
# directly, so the separate latent_up activation quantization disappears.
_K3_LU_AR_QUANT = os.environ.get("TRTLLM_K3_LU_AR_QUANT", "0") == "1"
_K3_FP8_MID_MAX_M = int(os.environ.get("TRTLLM_K3_FP8_MID_MAX_M", "4096"))
# Attribute name of an FP8 weight-read projection -> its TRTLLM_K3_FP8_MID_GEMM family.
_K3_FP8_MID_FAMILY_OF_ATTR = {
    "o_proj": "o",
    "q_b_proj": "qb",
    "kv_a_proj_with_mqa": "qa",
    "g_proj": "g",
    "qkvg_proj": "qkvg",
    "routed_expert_up_proj": "lu",
}

# CUDA stream priority of the MoE input stream (latent_down chain, routed-output memset).
# Negative values rank above the main and shared-expert streams, so the latent chain that
# gates the routed GEMM1 gets free SMs first.
_K3_LATENT_STREAM_PRIORITY = int(os.environ.get("TRTLLM_K3_LATENT_STREAM_PRIORITY", "0"))
# TRTLLM_K3_ROUTER_FIRST=1: for up to TRTLLM_K3_ROUTER_FIRST_MAX_M tokens, the router GEMM is issued
# (captured) on the main stream before latent_down is issued on the latent stream. Both still
# depend only on the MoE input (the latent fork event is recorded before the router), so the graph
# has the same edges; only the node creation order changes, which decides which of the two the
# graph launches first once the MoE input's producer releases its dependents (PDL). At decode
# batch sizes latent_down's low-M GEMM is 112 CTAs x ~225 KB of shared memory, and with up to 16
# tokens the router GEMM (dsv3 router MMA kernel) is 112 CTAs whose registers fill most of an SM,
# so no SM holds one of each: launched second, the router gets only the SMs latent_down leaves
# and runs in several waves, and the routing (which gates the routed FC1 together with
# latent_down) ends late. Launched first, the router takes its SMs and latent_down starts on the
# rest. nsys node tracing turns latent_down's cross-stream programmatic edge into a full edge (it
# then waits for the MoE input's producer to complete), so such traces show the router first in
# either order.
_K3_ROUTER_FIRST = os.environ.get("TRTLLM_K3_ROUTER_FIRST", "0") == "1"
_K3_ROUTER_FIRST_MAX_M = int(os.environ.get("TRTLLM_K3_ROUTER_FIRST_MAX_M", "64"))
# Shared expert ahead of the routed FC1 for token counts in [TRTLLM_K3_SHARED_FIRST_MIN_M,
# TRTLLM_K3_SHARED_FIRST_MAX_M]. The persistent routed grids (FC1, FC2) hold every SM while they
# run, and FC1 also fills each SM's register file, so shared-expert kernels that are still
# pending when FC1 starts wait for it (and the down GEMM for FC2 too). With this on, the shared
# gate/up + down GEMMs run first: they fork with the latent chain ("1") or after the router GEMM
# ("router"), FC1 waits for them, and the shared all-reduce waits for FC1, so it runs beside FC2
# (whose CTAs leave room for it). Same kernels and inputs, so bit-exact; needs the latent-chain
# stream and the CuteDSL backend.
_K3_SHARED_FIRST = os.environ.get("TRTLLM_K3_SHARED_FIRST", "0")
if _K3_SHARED_FIRST not in ("0", "1", "router"):
    raise ValueError(f"TRTLLM_K3_SHARED_FIRST must be 0, 1 or router, got {_K3_SHARED_FIRST!r}")
_K3_SHARED_FIRST_MIN_M = int(os.environ.get("TRTLLM_K3_SHARED_FIRST_MIN_M", "257"))
_K3_SHARED_FIRST_MAX_M = int(os.environ.get("TRTLLM_K3_SHARED_FIRST_MAX_M", "2048"))
# The shared expert's all-reduce runs beside the persistent routed GEMMs, which leave no room for
# the per-SM programs of the pipelined NVLS kernel (TRTLLM_NVLS_AR_V2=1), so it keeps the one-shot
# NVLS kernel unless TRTLLM_K3_SHARED_AR_NVLS_V2=1.
_K3_SHARED_AR_NVLS_V2 = os.environ.get("TRTLLM_K3_SHARED_AR_NVLS_V2", "0") == "1"
# NVLS workspace slot of the shared expert's all-reduce. "main" shares the o_proj all-reduce's
# workspace (and, with TRTLLM_NVLS_AR_ZERO_COPY=1, its input buffer): every shared-expert
# all-reduce, from its down projection on, lies between the layer's o_proj all-reduce (whose
# output the shared expert consumes) and the next layer's o_proj (which runs after the MoE
# join), so the two never overlap. Any other name gives it a workspace of its own.
_K3_SHARED_AR_NVLS_SLOT = os.environ.get("TRTLLM_K3_SHARED_AR_NVLS_SLOT", "main")
# CUDA stream priority of the stream shared by the attention-side aux work and the MoE shared
# expert. Negative values let the shared-expert chain take SMs ahead of the persistent routed
# GEMMs on the main stream, so its all-reduce does not land after them.
_K3_SHARED_STREAM_PRIORITY = int(os.environ.get("TRTLLM_K3_SHARED_STREAM_PRIORITY", "0"))
# TRTLLM_K3_SHARED_TAIL_PRIO=<p> (e.g. -1; unset: off): for TRTLLM_K3_SHARED_TAIL_MIN_M ..
# TRTLLM_K3_SHARED_TAIL_MAX_M tokens, where the routed latent all-reduce runs on its own workspace
# (TRTLLM_K3_LATENT_AR_OWN_WS), the shared expert's activation, down projection and all-reduce run
# on a second stream of CUDA priority <p>, forked from the shared-expert stream after gate_up, and
# the main stream joins that stream instead. At decode batch sizes the routed FC1 / FC2 kernels
# are persistent grids whose pending CTAs (the PDL-launched FC2 waits for SMs while FC1 runs)
# would otherwise be served before the shared expert's small kernels launched after them; with
# the higher priority those kernels take the SMs that FC1 / FC2 leave (see
# TRTLLM_MOE_LOWM_FC1_CTAS / _FC2_CTAS) and the first ones FC1's CTAs free. Same kernels and
# inputs: bit-exact.
_K3_SHARED_TAIL_PRIO = os.environ.get("TRTLLM_K3_SHARED_TAIL_PRIO", "")
_K3_SHARED_TAIL_MIN_M = int(os.environ.get("TRTLLM_K3_SHARED_TAIL_MIN_M", "1"))
_K3_SHARED_TAIL_MAX_M = int(os.environ.get("TRTLLM_K3_SHARED_TAIL_MAX_M", "64"))
# The shared-expert tail streams, by (device index, priority); every MoE layer uses the same one.
_K3_SHARED_TAIL_STREAMS: Dict[Tuple[int, int], torch.cuda.Stream] = {}


def _k3_shared_tail_stream() -> Optional[torch.cuda.Stream]:
    """The ``TRTLLM_K3_SHARED_TAIL_PRIO`` stream of the current device, or ``None`` if off."""
    if not _K3_SHARED_TAIL_PRIO or not torch.cuda.is_available():
        return None
    key = (torch.cuda.current_device(), int(_K3_SHARED_TAIL_PRIO))
    stream = _K3_SHARED_TAIL_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(priority=key[1])
        _K3_SHARED_TAIL_STREAMS[key] = stream
    return stream


# Small-batch MoE front end (b1-b4 decode), each bit-identical to the path it replaces:
#   TRTLLM_K3_LD_FP4=1: latent_down with the routed experts' NVFP4 input quantization in its epilogue
#     (k3_fp8w_lowm_gemm_nvfp4, byte-identical to the low-M GEMM followed by fp4_quantize) for up to
#     TRTLLM_K3_LD_FP4_MAX_M tokens where the low-M kernel serves latent_down and the CuteDSL NVFP4
#     experts would quantize it. Pair it with TRTLLM_K3_LATENT_AR_OWN_WS: the earlier routed FC1
#     otherwise pushes the shared-expert chain (and the latent all-reduce behind it) later.
#   TRTLLM_K3_ROUTER_EXACT=1: router logits of M <= 16 tokens from k3_router_exact (bit-identical
#     to dsv3_router_gemm_op; its weight is copied into shared memory before griddepcontrol.wait).
#     =kda: only in the MoE blocks of KDA layers. The faster router also starts the route tail
#     earlier, inside the burst of latent_down and the shared gate_up; after an MLA attention block
#     the route tail then stretches by ~3 us (b2 gen-only, GB300) and the routing chain gets longer
#     than with dsv3_router_gemm_op, while after a KDA attention block the front gets shorter.
_K3_LD_FP4 = os.environ.get("TRTLLM_K3_LD_FP4", "0") == "1"
_K3_LD_FP4_MAX_M = int(os.environ.get("TRTLLM_K3_LD_FP4_MAX_M", "32"))
_K3_ROUTER_EXACT_MODE = os.environ.get("TRTLLM_K3_ROUTER_EXACT", "0")
if _K3_ROUTER_EXACT_MODE not in ("0", "1", "kda"):
    raise ValueError(f"TRTLLM_K3_ROUTER_EXACT must be 0, 1 or kda, got {_K3_ROUTER_EXACT_MODE!r}")
# TRTLLM_K3_MOE_L2_PREFETCH: comma list of what the latent-down stream prefetches into L2 once the
# routed FC2 is done (the latent all-reduce, latent_up and the next attention block leave HBM mostly
# idle; k3_l2_prefetch, hints only): "lu" this layer's latent_up weight, "front" the next MoE layer's
# router, latent_down and shared-expert weights. Up to TRTLLM_K3_MOE_L2_PREFETCH_MAX_M tokens. The
# prefetch stays on the latent-down stream, which the next MoE layer joins after its latent_down.
_K3_MOE_L2_PREFETCH = frozenset(
    v for v in os.environ.get("TRTLLM_K3_MOE_L2_PREFETCH", "").split(",") if v and v != "0"
)
if not _K3_MOE_L2_PREFETCH <= {"lu", "front"}:
    raise ValueError(f"TRTLLM_K3_MOE_L2_PREFETCH must list lu / front, got {_K3_MOE_L2_PREFETCH}")
_K3_MOE_L2_PREFETCH_MAX_M = int(os.environ.get("TRTLLM_K3_MOE_L2_PREFETCH_MAX_M", "64"))
# TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA=1: where the next decoder layer is an MLA layer, the MoE block
# also prefetches that layer's q_b projection weight and scale (28 MB) into L2 on the latent-down
# stream once its routed FC2 is done, under the same conditions as TRTLLM_K3_MOE_L2_PREFETCH (up to
# TRTLLM_K3_MOE_L2_PREFETCH_MAX_M tokens, a later MoE layer joins the stream). At decode batch
# sizes most q_b CTAs become resident only when the q_a GEMM's CTAs leave and then stream their
# weight slice from HBM; from L2 that part is shorter. Hints only: bit-exact.
_K3_MOE_L2_PREFETCH_NEXT_MLA = os.environ.get("TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA", "0") == "1"
# MoE runtimes of one model by layer index (keyed by the model's text config), for the prefetch of the
# next MoE layer's weights.
_K3_MOE_LAYERS: Dict[int, "weakref.WeakValueDictionary[int, KimiK3MoERuntime]"] = {}
# MLA runtimes of one model by layer index (keyed like _K3_MOE_LAYERS), for the prefetch of the
# next MLA layer's q_b weight (TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA).
_K3_MLA_LAYERS: Dict[int, "weakref.WeakValueDictionary[int, KimiMLARuntime]"] = {}
# TRTLLM_K3_EXPERT_L2_PREFETCH=<B> (1..32; 0 = off): routed-expert weights from L2 at decode batch
# sizes, for up to TRTLLM_K3_EXPERT_L2_PREFETCH_MAX_M tokens with the CuTe routing tail. Every MoE
# layer records its routing on this rank (the tokens of each touched local expert) into a persistent
# routing record once its routed FC2 is done (one small k3_expert_l2_prefetch launch on a side
# stream). In the next decode step the layer's attention block calls the layer's
# issue_expert_l2_prefetch once its weight-heavy projections are issued (KDA: after QKVG and
# [f_a | b]; MLA: after q_b and the q absorption), and a k3_expert_l2_prefetch launch on the same
# side stream prefetches into L2 the gate_up / down weights and scales of the B experts with the
# most tokens in that record, while the recurrence / MLA decode and the all-reduces leave HBM mostly
# idle. The tokens of a request mostly pick the same experts again, and the prefetched lines stay in
# L2 until the layer's routed GEMMs read them (see k3_expert_l2_prefetch). Hints only: bit-exact,
# and a stale record (the first step, a step of another batch size) only changes what is prefetched.
# TRTLLM_K3_EXPERT_L2_PREFETCH_CTAS / _CHUNK_KB: prefetch launch shape (32 one-warp CTAs, 32 KB pieces).
_K3_EXPERT_L2_PREFETCH = int(os.environ.get("TRTLLM_K3_EXPERT_L2_PREFETCH", "0"))
if not 0 <= _K3_EXPERT_L2_PREFETCH <= 32:
    raise ValueError(
        f"TRTLLM_K3_EXPERT_L2_PREFETCH must be in [0, 32], got {_K3_EXPERT_L2_PREFETCH}"
    )
_K3_EXPERT_L2_PREFETCH_MAX_M = int(os.environ.get("TRTLLM_K3_EXPERT_L2_PREFETCH_MAX_M", "32"))
_K3_EXPERT_L2_PREFETCH_CTAS = int(os.environ.get("TRTLLM_K3_EXPERT_L2_PREFETCH_CTAS", "32"))
_K3_EXPERT_L2_PREFETCH_CHUNK = (
    int(os.environ.get("TRTLLM_K3_EXPERT_L2_PREFETCH_CHUNK_KB", "32")) << 10
)


class _ExpertL2PrefetchState:
    """``TRTLLM_K3_EXPERT_L2_PREFETCH`` state shared by the MoE layers of one model: the side
    stream of the prefetch launches, and whether it holds work the main stream has not joined."""

    def __init__(self):
        self.stream: Optional[torch.cuda.Stream] = None
        self.pending = False


# Expert L2 prefetch state of each model (keyed like _K3_MOE_LAYERS).
_K3_EXPERT_L2_STATE: Dict[int, _ExpertL2PrefetchState] = {}
# TRTLLM_K3_LATENT_AR_OWN_WS=1: for up to TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M tokens, the routed latent
# all-reduce + RMSNorm (the same MNNVL Lamport kernels) runs on a workspace of its own instead of the
# one it shares with the shared expert's all-reduce. It then no longer has to follow the shared-expert
# chain: the main stream issues it and latent_up right after the routed FC2 and joins the shared
# stream after latent_up. Applies where latent_up does not add the shared output itself (no fused
# latent_up + residual variant); the output is bit-identical.
_K3_LATENT_AR_OWN_WS = os.environ.get("TRTLLM_K3_LATENT_AR_OWN_WS", "0") == "1"
_K3_LATENT_AR_OWN_WS_MAX_M = int(os.environ.get("TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M", "32"))
# TRTLLM_K3_LATENT_AR_CUTE (with TRTLLM_K3_LATENT_AR_OWN_WS): the one-shot sizes of that all-reduce
# run k3_latent_ar_norm, bit-identical to the C++ one-shot kernel but triggering its programmatic
# dependents (latent_up) right after its griddepcontrol.wait instead of after the cross-rank
# reduction ("1"), or after the reduction like the C++ kernel ("late").
_K3_LATENT_AR_CUTE = os.environ.get("TRTLLM_K3_LATENT_AR_CUTE", "0")
if _K3_LATENT_AR_CUTE not in ("0", "1", "late"):
    raise ValueError(f"TRTLLM_K3_LATENT_AR_CUTE must be 0, 1 or late, got {_K3_LATENT_AR_CUTE!r}")
# TRTLLM_K3_LU_MXFP8=1 (with TRTLLM_K3_LATENT_AR_CUTE): k3_latent_ar_norm also stores the MXFP8 form
# of the normalized latent and latent_up runs the low-M FP8 GEMM on it (forward_mxfp8) instead of
# quantizing the BF16 rows itself: bitwise the same projection, where latent_up runs the low-M kernel.
_K3_LU_MXFP8 = os.environ.get("TRTLLM_K3_LU_MXFP8", "0") == "1"
# TRTLLM_K3_LATENT_AR_ROW=1 (with TRTLLM_K3_LATENT_AR_CUTE): the one-shot sizes of the latent
# all-reduce run k3_latent_ar_norm's row kernel (one 512-thread CTA per row instead of a cluster of
# four; the rank's own slot is summed from registers instead of being polled back from the
# multicast loopback; slots that arrived are not reloaded): the same result, bit for bit.
_K3_LATENT_AR_ROW = os.environ.get("TRTLLM_K3_LATENT_AR_ROW", "0") == "1"
# TRTLLM_K3_LATENT_AR_SIDE_CLEAR=1 (with TRTLLM_K3_LATENT_AR_CUTE): once latent_up is issued, the
# latent-down stream re-arms the Lamport buffer the latent all-reduce just used
# (k3_latent_ar_norm.run_k3_lamport_clear) and marks the workspace state clean, so the next layer's
# all-reduce skips the clear (the M x ranks x 7 KB of sentinel stores) on its critical path. Only
# where a later MoE layer exists: its latent_down follows on that stream and is joined into the main
# stream before its routed GEMMs. The data is unchanged.
_K3_LATENT_AR_SIDE_CLEAR = os.environ.get("TRTLLM_K3_LATENT_AR_SIDE_CLEAR", "0") == "1"
# TRTLLM_K3_LU_MX_STAGES=<a>:<b> (with TRTLLM_K3_LU_MXFP8): latent_up's MXFP8 GEMM runs with <a> weight
# stages and <b> activation stages (k3_fp8w_lowm_mxfp8.K3LowMMxTactic; same split-K and K-tile order,
# so bitwise the default tactic's result) instead of the default (deepest weight ring next to an
# activation ring holding every K tile, whose scales are released all at once).
_K3_LU_MX_STAGES = os.environ.get("TRTLLM_K3_LU_MX_STAGES", "")
# The routed latent all-reduce's own MNNVL workspaces, by all-reduce mapping.
_K3_LATENT_AR_WORKSPACES: Dict[Mapping, Dict[str, Any]] = {}


def _k3_latent_ar_workspace(
    mapping: Mapping, dtype: torch.dtype, hidden_size: int, max_tokens: int
) -> Dict[str, Any]:
    """MNNVL Lamport workspace of the routed latent all-reduce alone (three buffers and the state
    flags, laid out as ``get_or_scale_allreduce_mnnvl_workspace`` lays out the shared one), sized
    for up to ``max_tokens`` rows of ``hidden_size``. Collective over the all-reduce group: every
    rank builds it at the same point of model construction."""
    workspace = _K3_LATENT_AR_WORKSPACES.get(mapping)
    if workspace is not None:
        if workspace["max_tokens"] < max_tokens:
            raise ValueError(
                f"Kimi K3 latent all-reduce workspace serves {workspace['max_tokens']} tokens, "
                f"{max_tokens} requested"
            )
        return workspace
    from ...bindings.internal.runtime import McastGPUBuffer
    from ..distributed.ops import MNNVLAllReduce, get_or_scale_allreduce_mnnvl_workspace

    # The shared workspace's communicator (split by the all-reduce group) is the same group.
    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)["mpi_comm"]
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, hidden_size, mapping.tp_size, dtype)
        for m in range(1, max_tokens + 1)
    )
    buffer_bytes = -(-need // (1 << 20)) << 20
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    handle = McastGPUBuffer(
        3 * buffer_bytes,
        mapping.tp_size,
        mapping.tp_rank,
        mapping.local_rank,
        use_fabric_handle,
        comm.py2f(),
    )
    # Lamport-clean (-0.0) before any rank writes into it.
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    # [current index, dirty index, bytes per buffer, dirty stages, bytes to clear x4, arrivals]
    flags = torch.tensor(
        [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=buffer.device
    )
    workspace = {
        "handle": handle,
        "buffer": buffer.view(dtype).view(3, -1),
        # The multicast view of the same buffers (stores reach every rank's copy).
        "mc": handle.get_mc_buffer((3 * buffer_bytes // 4,), torch.float32, 0),
        "flags": flags,
        # Counter of the side-stream Lamport clear (TRTLLM_K3_LATENT_AR_SIDE_CLEAR).
        "clear_ctr": torch.zeros(1, dtype=torch.int32, device=buffer.device),
        "max_tokens": max_tokens,
        # Bytes of one of the three Lamport buffers.
        "buffer_bytes": buffer_bytes,
        "rank": mapping.tp_rank,
        "world_size": mapping.tp_size,
    }
    _K3_LATENT_AR_WORKSPACES[mapping] = workspace
    return workspace


# A/B escape hatch: restore nn.Linear for the K3 latent MoE projections
# instead of the min-latency fused GEMM op (read once at import).
_K3_DISABLE_MIN_LATENCY_LATENT_PROJ = (
    os.environ.get("TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", "0") == "1"
)

# Routed-expert MoE TP/EP split. An explicit moe_tensor_parallel_size /
# moe_expert_parallel_size pair from the user config is honored; otherwise the
# default stays EP-only (moe_ep == tp_size).
#
# Measured tradeoff for the TP override (16xGB200, TEP16, ISL 8192 / OSL
# 1024, concurrency 1, one allocation with the EP-only baseline bracketing
# the candidates): moe_tp 1 = 53.41 tok/s/user (TPOT 17.93 ms), moe_tp 4 =
# 61.59 (+15.3%, TPOT 15.54), moe_tp 8 = 61.06 (+14.3%). EP-only is
# balls-into-bins at decode -- top_k 16 over 16 EP ranks leaves the busiest
# rank ~3 experts and ~6 ranks with none, and the routed AllReduce waits for
# the busiest (per-layer routed gemm1 spans 4.1-27.4 us across the 93 layers
# purely from that draw). Splitting the intermediate makes every rank's share
# identical. 4 is the sweet spot: 8 leaves too little work per rank for the
# TRTLLM-Gen 128-wide intermediate tile, and 16 is worse still (3072/16 = 192
# is not 128-aligned, so the intermediate pads to 4096 and each rank stores
# and reads 33% more expert weight than it computes on). The win is a
# small-batch effect and fades as the batch grows (concurrency 4 measured
# +4.4%, roughly the run-to-run spread), so it stays opt-in.
if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...llmapi.llm_args import DecodingBaseConfig

# Identity-RoPE table positions for the MLA backends. K3 is NoPE (the table
# holds cos=1/sin=0), but the chunked-context path indexes the table by
# absolute position, so it must cover max_position_embeddings (~512MB per
# backend for the 1M-position checkpoint); a smaller table is read out of
# bounds. KIMI_K3_MLA_MAX_POSITIONS overrides the size for short-context
# deployments.
_KIMI_K3_MLA_MAX_POSITIONS_ENV = "KIMI_K3_MLA_MAX_POSITIONS"
_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES = (
    ".self_attn.mixer.k_b_proj_trans",
    ".self_attn.mixer.v_b_proj",
    ".self_attn.mixer.k_b_proj_trans_scale",
    ".self_attn.mixer.v_b_proj_scale",
    ".self_attn.mixer.k_b_proj_trans_dequant",
    ".self_attn.mixer.v_b_proj_dequant",
)

# BF16 shared/latent MLP weights are optionally quantized once after loading.
# Checkpoint-native attention FP8 is independent of this lossy conversion.
_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV = "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP"
# Also convert the TP-sharded shared expert (gate_up and the row-parallel down,
# whose reduction the MoE runtime then applies) to the FP8 weight read.
_KIMI_K3_FP8_WEIGHT_READ_SHARED_TP_ENV = "KIMI_K3_FP8_WEIGHT_READ_SHARED_TP"
# Also convert the dense (non-MoE) layers' BF16 MLP (layer 0) to the FP8 weight read (lossy like
# the other conversions): see _convert_dense_mlps_to_fp8_weight_read.
_KIMI_K3_FP8_WEIGHT_READ_DENSE_MLP_ENV = "KIMI_K3_FP8_WEIGHT_READ_DENSE_MLP"


class KimiK3MoEGate(nn.Module):
    """Kimi K3 gate weights and routing method for ``ConfigurableMoE``."""

    def __init__(
        self,
        config: Any,
        *,
        logits_gemm_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size

        assert self.moe_router_activation_func in ("sigmoid", "softmax"), (
            "K3 MoE gate supports sigmoid or softmax scoring only"
        )

        # The checkpoint stores the gate weight in bf16. Storing it in bf16
        # permits the single bf16xbf16 router GEMM while retaining fp32 output.
        weight_dtype = logits_gemm_dtype or torch.float32
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=weight_dtype, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        if self.weight.dtype == torch.bfloat16 and hidden_2d.dtype == torch.bfloat16:
            return torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_2d.contiguous(),
                self.weight.t(),
                bias=None,
                out_dtype=torch.float32,
            )
        return torch.nn.functional.linear(
            hidden_2d.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        """Return the shared DeepSeek-V3 router used by ``ConfigurableMoE``."""
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi K3 ConfigurableMoE routing requires sigmoid scores.")
        if not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 ConfigurableMoE routing requires top-k weight renormalization."
            )
        return _KimiK3RoutingMethod(
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            is_fused=True,
        )


class _KimiK3RoutingMethod(DeepSeekV3MoeRoutingMethod):
    """DeepSeek-V3 routing that can hand out top-k results computed ahead of time.

    ``KimiK3MoERuntime`` stores the fused router's ``(topk_ids, topk_weights)`` in
    ``precomputed`` right before calling the routed experts; the next ``apply``
    returns and clears them instead of routing ``logits`` again. Alternatively it
    stores a ``deferred`` callable that the next ``apply`` runs (once) on ``logits``.
    ``observer``, when set, is called with every top-k id tensor ``apply`` returns.
    ``slot_map`` (``TRTLLM_K3_EXPERT_REPLICAS``), when set, maps the routed (home) expert ids of a
    result without attached ``moe_sort`` outputs to the routed MoE's home-copy slots.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precomputed: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.deferred: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self.observer: Optional[Callable[[torch.Tensor], None]] = None
        self.slot_map: Optional[torch.Tensor] = None

    def apply(
        self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.precomputed is not None:
            result, self.precomputed = self.precomputed, None
        elif self.deferred is not None:
            route, self.deferred = self.deferred, None
            result = route(logits)
        else:
            result = super().apply(logits, input_ids)
        if self.observer is not None and not torch.compiler.is_compiling():
            self.observer(result[0])
        if self.slot_map is not None and getattr(result[0], "_k3_presorted", None) is None:
            ids = result[0]
            result = (
                torch.index_select(self.slot_map, 0, ids.reshape(-1)).view(ids.shape),
                result[1],
            )
        return result


class KimiK3RMSNorm(nn.Module):
    """RMSNorm matching the Kimi checkpoint implementation's rounding."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_float = hidden_states.to(torch.float32)
        variance = hidden_states_float.pow(2).mean(-1, keepdim=True)
        hidden_states_float = hidden_states_float * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states_float.to(input_dtype)


# ---------------------------------------------------------------------------
# Config helpers.
# ---------------------------------------------------------------------------


def _get_text_config(pretrained_config: "PretrainedConfig"):
    """Return the Kimi text config, unwrapping a composite kimi_k3 config."""
    if getattr(pretrained_config, "model_type", None) == "kimi_k3" or (
        not hasattr(pretrained_config, "linear_attn_config")
        and hasattr(pretrained_config, "text_config")
    ):
        return pretrained_config.text_config
    return pretrained_config


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


def _router_exact_for_layer(cfg, layer_idx: int) -> bool:
    """Whether ``TRTLLM_K3_ROUTER_EXACT`` serves the router of this layer's MoE block."""
    if _K3_ROUTER_EXACT_MODE == "kda":
        return _is_kda_layer(cfg, layer_idx)
    return _K3_ROUTER_EXACT_MODE == "1"


# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------


KIMI_K3_AUX_ATTN_RES_STREAM_ENV = "KIMI_K3_AUX_ATTN_RES_STREAM"
"""Which residual-stream value the DFlash/DSpark hidden-state tap captures.

``1`` (default) captures the pre-norm attn_res mixture -- the value the next
consumer actually reads. ``0`` captures the raw running prefix sum instead.

Both conventions exist in the wild and a drafter distilled against one scores
lower on the other with nothing raised, so this is a property of the DRAFTER
checkpoint, not a performance knob. SGLang (and therefore RadixArk/Kimi-K3-DSpark)
uses the mixture: ``kimi_k3.py _dspark_capture_stream`` -> ``attn_residual.py
aggregate_stream``. vLLM implements both and defaults to the prefix
(``VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0``, ``models/kimi_k3/nvidia/model.py
_capture_aux_hidden_stream``), which is what a TorchSpec-distilled drafter may
have been trained against. Measured cost of getting it wrong on K3 + RadixArk:
AR 71.4% -> 66.9%.

Per-checkpoint measurements on K3, GSM8K AL, n=200, TEP8:

===================  ==================  ============  ======
drafter              stream (default 1)  prefix (0)    delta
===================  ==================  ============  ======
RadixArk (GQA)       71.8%               --            --
Inferact (MLA)       65.7%               66.6%         +0.9pt
===================  ==================  ============  ======

So the default is right for RadixArk. For Inferact the prefix convention
matches its vLLM/TorchSpec lineage and measures better, but +0.9pt at n=200 is
inside this harness's noise band (it treats RadixArk's own 71-73% spread as
noise), so this is a direction, not a settled requirement -- unlike the 4.5pt
RadixArk case above, which was unambiguous. Both Inferact acc_len values (5.60
and 5.66) sit on its model card's 5.64, so neither convention is grossly wrong
for it.

Deriving this from checkpoint metadata is not possible today: neither published
drafter's config records which capture convention it was distilled against."""

_AUX_ATTN_RES_STREAM_ENABLED = os.environ.get(KIMI_K3_AUX_ATTN_RES_STREAM_ENV, "1") == "1"

KIMI_K3_FUSED_ATTN_RES_ENV = "KIMI_K3_FUSED_ATTN_RES"
"""Set to ``0`` to run attention-residual as the exact fp32 reference.

Two states, not three: either the fused ops are used -- selection and the
trailing RMSNorm together, and the residual add too where the call site has one
-- or nothing is fused and ``_apply_attn_res`` computes the reference in fp32.

There used to be a middle state, ``KIMI_K3_FUSED_ATTN_RES_NORM=0``, which kept
``trtllm::attn_res_fwd`` and ran the RMSNorm as a separate PyTorch module. It
existed to A/B the norm fusion when that fusion was new; that question is
answered and the fusion is the default. It is removed rather than kept, because
by now it was actively misleading: the persistent op instantiates
``HAS_OUTPUT_NORM=true`` only, so under the ``split`` and ``persistent``
topologies setting it to 0 did not give "persistent without the fused norm", it
silently fell all the way back to ``per_token``.

When this is ``0``, ``KIMI_K3_ATTN_RES_TOPOLOGY`` and
``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` have no effect: no fused op is reached, so
there is no topology to pick and no ceiling to compare against."""

_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"

KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV = "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS"
"""Largest per-rank token count the fused epilogue is taken for.

Under attention-DP this is a concurrency ceiling divided by the DP degree: at
DEP16, 32 tokens per rank is concurrency 512, and the production max_batch_size
is 32 anyway. The default is the top of the measured range rather than a limit
of the kernel, which will launch for any token count -- past 32 there is simply
no measurement, and the split-K variant is already spilling into a second wave
there (256 CTAs against 212 SMs), so the trend is not safe to extrapolate.

The default follows KIMI_K3_ATTN_RES_TOPOLOGY rather than being a constant: 1
when the port is off, which is the pre-existing decode-only gate, and 32 when
it is on. Setting this explicitly overrides both.

Exposed as an environment variable so an A/B can hold the tree, the .so and the
image fixed and vary only this. Read at import, so CUDA-graph capture and replay
cannot disagree about it.
"""


def _read_fused_attn_res_max_tokens() -> int:
    """Resolved after the topology, because its default follows it.

    With the port off -- the default -- the ceiling is 1, which is exactly the
    pre-existing gate: the fused epilogue is taken at the single-token decode
    shape and nowhere else. Enabling the port raises it to 32, the top of the
    measured range.

    Tying the two together is the point. A ceiling of 32 under a topology of
    per_token would leave the shapes in 1 < M <= 32 quietly using the fused
    add+RMSNorm ops while the feature reads as disabled, so "off" would not mean
    "unchanged" -- and those shapes are the ones the end-to-end evidence covers
    least: the two 8k1k ABBAs that measured exactly this change sit inside a
    12-26% within-arm spread and neither show a gain nor rule out a regression.
    """
    default = "1" if _ATTN_RES_TOPOLOGY == "per_token" else "32"
    raw = os.environ.get(KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV, default)
    try:
        value = int(raw)
    except ValueError:
        # Failing loudly matters more than usual here: a mistyped A/B arm that
        # silently fell back to the default would measure the candidate twice
        # and report no difference.
        raise ValueError(
            f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be a positive integer, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be >= 1, got {value}")
    return value


KIMI_K3_ATTN_RES_TOPOLOGY_ENV = "KIMI_K3_ATTN_RES_TOPOLOGY"
"""Which fused attention-residual kernel topology to use.

Two implementations of the same fusion now exist, and they are good at opposite
ends of the token count:

  ``per_token``   one CTA (N<=4) or one 8-CTA cluster (N>=5) per token. Wins
                  when tokens are scarce enough that the machine is not full.
  ``persistent``  one CTA per SM looping over tokens, so the score projection is
                  loaded once per CTA rather than once per token.

Measured on both architectures (`notes/experiments/kimi_k3/0902_vllm_persistent_fusion`),
`persistent` wins everywhere except small token counts at N>=5:

  Rubin  SM107, 212 SM   per_token wins at (N=5,T<=8) and (N=9,T<=8), 4 of 21
  GB300  SM103, 152 SM   per_token wins at (N=9,T<=8), 2 of 21

At T=8192 `persistent` beats the unfused three-kernel path by 17-43% while
`per_token` loses by 46-468%, so the prefill case is not close.

A third value, ``split``, routes by token count instead of picking one kernel
for the whole run: at or below ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` tokens the
per-token kernels, above it the persistent one. That is the policy the two
measurements above imply, and it is the only mode in which both fused paths are
reachable in a single process.

``split`` is deliberately one threshold and not a tuned table. The threshold is
the production ``max_batch_size``, which makes it the prefill/decode boundary
rather than a fitted constant: at most ``max_batch_size`` tokens can be in
flight one-per-request, so anything above it is several tokens of one sequence.
That is a fact about the deployment, not about a particular SM count, so it does
not need re-measuring per machine.

Resisting the alternative on purpose: the kernel sweep does show a crossover
that moves with N, and following it leads to a per-(N, token-count) lookup
table. SGLang has one -- ``_TMA_BEST_CONFIG`` in
``python/sglang/kernels/ops/kimi_k3/attn_res.py``, eight entries plus a
``num_tokens < 128`` special case -- and it went through a two-kernel selector
before deleting it. A table has to be re-measured whenever the SM count, the
candidate-count distribution, or the tile constants move, and nothing tells you
when it has gone stale. One number tied to a deployment parameter does not.

The default is ``per_token``, i.e. off: the ported kernel is not used at any
shape, and the behaviour is what shipped before this branch. Enabling it is
``KIMI_K3_ATTN_RES_TOPOLOGY=1``, which resolves to ``split`` -- the persistent
kernel above ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` tokens, the per-token
kernels at or below. In deployment terms that is persistent for prefill and
per-token for decode.

Off by default because the port is not uniformly safe, and the reason is not in
this kernel. It is measured as no-regression on NVFP4 and as a 2.98x
context-phase regression on W4A8_MXFP4_MXFP8, and the difference is downstream:
the ported kernel's output agrees with the old one to row_rel_max 3.087e-03,
which flips 8.7% of top-16 expert selections with no bias at all (one EP16
rank's share moves from 6.36% to 6.36% against a balanced 6.25%), while the
TRTLLM-gen MoE on that checkpoint turns the same perturbation into a 86.7x rank
skew. Swapping only the MoE implementation removes the regression entirely
(MEGAMOE_DEEPGEMM: ctx_step 1224.4 ms against TRTLLM-gen's 4557.9 ms and the
per_token baseline's 1528.2 ms). A default that is free on one checkpoint and
3x slower on another is not a default, however well documented.

Enabling it selects ``split`` rather than ``persistent`` because that is the
strictly smaller change. Below the threshold it is byte-for-byte the previous
behaviour, so the only shapes that move are the ones the persistent kernel
exists for. And the end-to-end evidence is an OSL=1 workload, which is
prefill-dominated -- precisely the regime where ``split`` and ``persistent``
agree. ``persistent`` would additionally change decode, which that measurement
does not cover.

The raw numbers behind that. Four-phase ABBA, one binary, the arms one
environment variable apart, 8-node DEP16 disaggregated, ISL 8192 OSL 1:

    checkpoint                   per_token   persistent
    moonshotai_kimi-k3           1528.2 ms    4557.9 ms   ctx_step, 2.98x
    nvidia_kimi-k3-nvfp4        19831.1 ms   19799.2 ms   TTFT,    0.998x

The cost is not in this kernel: 544 ms against the old kernel's 480 ms over
identical 4650 instances. It is in the MoE, and it is an imbalance rather than
a slowdown -- inside the persistent arm gemm1 costs 48412 ms on rank 0 against
559 ms on rank 1 (2024 and 1840 instances), an 86.7x skew where per_token has
1.63x (6255 ms against 3838 ms), while rank 1 idles 78 s in the all-to-all.

The regression is in prefill, so ``split`` reaches it exactly as ``persistent``
does: neither is safe on that checkpoint *while the MoE backend is TRTLLM-gen*.
``moe_config.backend=MEGAMOE_DEEPGEMM`` is, and beats the pre-port baseline.
NVFP4 is unaffected either way.

What the NVFP4 ABBA does and does not support: TTFT -0.161% and throughput
+0.201% for ``persistent``, with the four values not interleaving, but |delta|
at 0.96x the within-arm spread. That is enough to say "no regression" and not
enough to claim a speedup. The decode side (8k1k) is unmeasured on NVFP4 --
which is an argument for ``split`` rather than against it, since ``split``
leaves decode on the previously shipped path.

The row-error outlier previously recorded here as unexplained -- ~5e-1 against
~4e-3, at (N=7) and (N=8) with an addend -- was the shared-workspace race fixed
in the kernel: ws_stats barriered the writes before the reads and nothing
ordered those reads before the next chunk's writes. It is gone, and the sweep
is now bit-identical across repeats rather than merely close.

Read at import, so CUDA-graph capture and replay cannot disagree about it.
"""

_ATTN_RES_TOPOLOGIES = ("per_token", "persistent", "split")

# Turning the port on should not require knowing which of the three topologies
# is the right one. "1" is the policy this branch measured -- persistent at
# prefill, per-token at decode -- so enabling the feature and choosing the
# policy are the same action, through one variable rather than two.
_ATTN_RES_TOPOLOGY_ON = ("1", "on", "true")


def _read_attn_res_topology() -> str:
    """Default off. The persistent kernel is opt-in on this branch.

    ``per_token`` is what shipped before the port, so the default changes no
    behaviour for anyone who does not ask for it. That is deliberate rather
    than timid: the port is measured as no-regression on NVFP4 (four-phase
    ABBA, TTFT -0.161% against a within-arm spread of 0.168%), but it is
    measured as a 2.98x context-phase regression on W4A8_MXFP4_MXFP8, and the
    cause of that lives downstream in the TRTLLM-gen MoE rather than in this
    kernel -- see the module docstring above. A default that is safe on one
    checkpoint and 3x slower on another is not a default.

    Enabling it is ``=1``, which resolves to ``split``. The two named
    topologies stay available for measurement: ``persistent`` uses the ported
    kernel at every shape, ``per_token`` at none.
    """
    raw = os.environ.get(KIMI_K3_ATTN_RES_TOPOLOGY_ENV, "per_token")
    if raw.lower() in _ATTN_RES_TOPOLOGY_ON:
        return "split"
    if raw not in _ATTN_RES_TOPOLOGIES:
        # Loudly, for the same reason as the token ceiling: a mistyped A/B arm
        # that silently fell back to the default would measure one side twice.
        raise ValueError(
            f"{KIMI_K3_ATTN_RES_TOPOLOGY_ENV} must be one of "
            f"{_ATTN_RES_TOPOLOGIES} or one of {_ATTN_RES_TOPOLOGY_ON} "
            f"(which means 'split'), got {raw!r}"
        )
    return raw


_ATTN_RES_TOPOLOGY = _read_attn_res_topology()
_FUSED_ATTN_RES_MAX_TOKENS = _read_fused_attn_res_max_tokens()


def _persistent_attn_res_applicable(M: int, H: int, N: int) -> bool:
    """Shape gate for the persistent kernel.

    Note what is absent: a token ceiling. The persistent grid is sized by the SM
    count, not by the token count, so prefill is the case it exists for.
    """
    del M  # deliberately unused; see above
    return H == 7168 and 2 <= N <= 9


def _use_persistent_attn_res(M: int, H: int, N: int) -> bool:
    """Pick between the two fused kernels for this call site.

    ``split`` routes by token count, which is the only thing this layer can see
    and, as it happens, the thing that should decide. "Prefill" and "decode" are
    not available here and would be the wrong question anyway: chunked prefill
    produces intermediate token counts, a single forward can mix context and
    generation requests, and whether the two phases even run in separate
    processes is a deployment choice. A token-count threshold covers all three
    cases with one rule.

    One comparison against one constant, on purpose. The constant is the
    existing ``_FUSED_ATTN_RES_MAX_TOKENS``, i.e. the production
    ``max_batch_size``: at most that many tokens can be in flight one per
    request, so above it the batch is several tokens of one sequence. Today the
    gate means "above this, give up and run unfused"; under ``split`` it means
    "above this, switch topology".

    Note the two gates here are different kinds of thing.
    ``_persistent_attn_res_applicable`` is a capability gate -- the kernel is
    only instantiated for those shapes, so it tracks the code and never needs
    re-measuring. The token count is the only performance decision, and it stays
    a single number rather than a per-(N, tokens) table; see the module constant
    for why that trade is made deliberately.

    Shapes the persistent kernel does not implement (N outside [2, 9], H other
    than 7168) fall through to the caller's existing gate and land on the
    unfused path, exactly as they do today.
    """
    if not _persistent_attn_res_applicable(M, H, N):
        return False
    if _ATTN_RES_TOPOLOGY == "persistent":
        return True
    return _ATTN_RES_TOPOLOGY == "split" and M > _FUSED_ATTN_RES_MAX_TOKENS


# Token limit of trtllm::attn_res_fwd (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: KimiK3RMSNorm
) -> Optional[torch.Tensor]:
    """Fused attn_res via the in-tree ``trtllm::attn_res_fwd`` op.

    Returns ``None`` when the call falls outside the fused kernel's
    contract (dtype/shape/arch) so the caller can use the exact fp32 reference
    instead. ``block_residual`` is kept in the kernel-native ``[K, M, H]``
    layout. Candidate order matches the reference: snapshots first, the
    running prefix sum last.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if K + 1 > 12 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    try:
        attn_res_op = torch.ops.trtllm.attn_res_fwd
    except (AttributeError, RuntimeError):
        return None
    proj_w = proj.weight.reshape(-1).to(torch.bfloat16).contiguous()
    norm_w = norm.weight.to(torch.bfloat16).contiguous()

    def run(prefix, block):
        t = prefix.shape[0]
        output, _rsigma, _probs, _logits = attn_res_op(
            prefix.reshape(t, 1, H).contiguous(),
            block.reshape(K, t, 1, H).contiguous(),
            proj_w,
            norm_w,
            float(norm.eps),
        )
        return output.reshape(t, H)

    if M <= _ATTN_RES_OP_MAX_TOKENS:
        return run(prefix_sum, block_residual)
    # The op takes at most _ATTN_RES_OP_MAX_TOKENS tokens per call and is
    # per-token, so longer prefill chunks run in slices; the fp32 reference
    # below would otherwise cost seconds per iteration at 24k+ tokens.
    out = torch.empty_like(prefix_sum)
    for lo in range(0, M, _ATTN_RES_OP_MAX_TOKENS):
        hi = min(M, lo + _ATTN_RES_OP_MAX_TOKENS)
        out[lo:hi] = run(prefix_sum[lo:hi], block_residual[:, lo:hi])
    return out


def _rms_norm_eps(norm: nn.Module) -> float:
    if hasattr(norm, "eps"):
        return float(norm.eps)
    return float(norm.variance_epsilon)


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether the fused path was actually reached, once per shape.

    ``_FUSED_ATTN_RES_ENABLED`` only says the feature is switched on. It does
    not say the shape gate below let the call through, and a rejected call
    looks exactly like a disabled one in the logs. Under attention-DP the
    per-rank token count decides it, so one job can fuse at low concurrency and
    fall back at high concurrency -- without this line, a benchmark that shows
    no change is indistinguishable from one that never ran the kernel.
    """
    logger.info_once(
        f"Kimi K3 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


# TRTLLM_K3_ATTN_RES_PDL=1: up to TRTLLM_K3_ATTN_RES_PDL_MAX_M tokens, the fused attn_res(+add)
# +RMSNorm runs as trtllm::k3_attn_res_(add_)rmsnorm (cute_dsl_kernels/blackwell/k3_attn_res.py):
# a CTA cluster per token that loads the weights and snapshot rows and reduces their statistics
# before griddepcontrol.wait, i.e. under the kernel in front of it. It takes precedence over the
# persistent topology at those token counts. The updated prefix sum is bit-identical; the
# normalized output can differ from the C++ per-token kernels by one BF16 rounding step in a small
# fraction of the elements (TRTLLM_K3_ATTN_RES_PDL_EXACT=1 selects a slower, bit-identical variant).
# TRTLLM_K3_ATTN_RES_PDL=2 also leaves the MoE residual update residual + bf16(routed + shared)
# of the BF16-add path (kimi_k3_moe_residual) to the next layer's pre-attention mix, which
# computes it bit-identically (trtllm::k3_attn_res_moe_add_rmsnorm) behind the latent-up GEMM.
_K3_ATTN_RES_PDL_MODE = os.environ.get("TRTLLM_K3_ATTN_RES_PDL", "0")
if _K3_ATTN_RES_PDL_MODE not in ("0", "1", "2"):
    raise ValueError(f"TRTLLM_K3_ATTN_RES_PDL must be 0, 1 or 2, got {_K3_ATTN_RES_PDL_MODE!r}")
_K3_ATTN_RES_PDL = _K3_ATTN_RES_PDL_MODE != "0"
_K3_ATTN_RES_PDL_MAX_M = int(os.environ.get("TRTLLM_K3_ATTN_RES_PDL_MAX_M", "64"))
# TRTLLM_K3_ATTN_RES_MXFP8=1: the pre-attention mix that folds in the previous MoE residual
# (trtllm::k3_attn_res_moe_add_rmsnorm_mx) also stores the MXFP8 form of its output, and the
# attention's first projection (KDA fused QKVG, MLA fused q_a / kv_a) runs the low-M GEMM on it
# (k3_fp8w_lowm_mxfp8) instead of quantizing the BF16 rows itself: bitwise the same projection,
# generation-only batches where the projection runs the low-M kernel.
_K3_ATTN_RES_MXFP8 = os.environ.get("TRTLLM_K3_ATTN_RES_MXFP8", "0") == "1"
# TRTLLM_K3_AR_PUSH_AMID=1 (with TRTLLM_K3_AR_PUSH=1): where the mix after attention (A_mid) adds
# the attention output to the running prefix sum, the attention block finishes its pushed o_proj
# all-reduce in place (two-shot transport, no gather) and A_mid reads the result straight from the
# MNNVL Lamport workspace (trtllm::k3_attn_res_add_rmsnorm_lamport): the same bits, one kernel
# boundary and one copy of the rows fewer.
_K3_AR_PUSH_AMID = os.environ.get("TRTLLM_K3_AR_PUSH_AMID", "0") == "1"
# TRTLLM_K3_DENSE_MLP_FULL_TP=1: the dense layer-0 MLP is split over the whole TP group, across
# nodes (its all-reduce runs on the multi-node NVLink fabric like the attention's), instead of the
# largest subgroup within one node: each rank reads 1/tp_size of the BF16 gate_up / down weights
# instead of 1/gpus_per_node. The all-reduce sums tp_size partials instead of gpus_per_node, so
# the rounding differs from the default split (not bitwise).
_K3_DENSE_MLP_FULL_TP = os.environ.get("TRTLLM_K3_DENSE_MLP_FULL_TP", "0") == "1"


class _DeferredMoEResidual(NamedTuple):
    """A MoE layer's ``residual + bf16(routed + shared)`` that the next layer fuses or materializes."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor

    def materialize(self) -> torch.Tensor:
        from ..modules.kimi_k3_moe_residual import _fused_moe_residual_add

        return _fused_moe_residual_add(self.routed, self.shared, self.residual)


def _defer_moe_residual(num_tokens: int) -> bool:
    """Whether a MoE layer leaves its BF16 residual update to the next layer."""
    return _K3_ATTN_RES_PDL_MODE == "2" and num_tokens <= _K3_ATTN_RES_PDL_MAX_M


def _materialize_prefix_sum(hidden_states):
    if isinstance(hidden_states, _DeferredMoEResidual):
        return hidden_states.materialize()
    return hidden_states


def _k3_attn_res_pdl(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
):
    """``trtllm::k3_attn_res_(add_)rmsnorm`` when enabled and applicable, else ``None``."""
    if not _K3_ATTN_RES_PDL or prefix_sum.shape[0] > _K3_ATTN_RES_PDL_MAX_M:
        return None
    from ..cute_dsl_kernels.blackwell import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    addends = () if addend is None else (addend,)
    if not k3_attn_res.can_use_k3_attn_res(prefix_sum, addends, block_residual, *weights):
        return None
    eps = float(norm.eps)
    out_eps = _rms_norm_eps(output_norm)
    if addend is None:
        return torch.ops.trtllm.k3_attn_res_rmsnorm(
            prefix_sum, block_residual, *weights, eps, out_eps
        )
    return torch.ops.trtllm.k3_attn_res_add_rmsnorm(
        prefix_sum, addend, block_residual, *weights, eps, out_eps
    )


def _k3_attn_res_lamport_applies(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> bool:
    """Whether A_mid on ``prefix_sum`` can read its addend from a Lamport workspace
    (``TRTLLM_K3_AR_PUSH_AMID``): the vectorized PDL attn_res kernel serves the ADD_PREFIX mix."""
    if (
        not _K3_AR_PUSH_AMID
        or not _K3_ATTN_RES_PDL
        or torch.compiler.is_compiling()
        or prefix_sum.shape[0] > _K3_ATTN_RES_PDL_MAX_M
    ):
        return False
    from ..cute_dsl_kernels.blackwell import k3_attn_res

    if k3_attn_res._EXACT:
        return False
    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    return k3_attn_res.can_use_k3_attn_res(prefix_sum, (prefix_sum,), block_residual, *weights)


def _k3_attn_res_pdl_lamport(
    prefix_sum: torch.Tensor,
    addend: LamportResult,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum + all-reduce result, normed mix)`` with the result read from the Lamport
    workspace (``trtllm::k3_attn_res_add_rmsnorm_lamport``), or ``None`` if that does not apply."""
    if tuple(addend.shape) != tuple(prefix_sum.shape) or not _k3_attn_res_lamport_applies(
        prefix_sum, block_residual, proj, norm, output_norm
    ):
        return None
    M, H = prefix_sum.shape
    _note_attn_res_fusion(
        "add+attn_res+norm/pdl/lamport", True, M, H, int(block_residual.shape[0]) + 1
    )
    return torch.ops.trtllm.k3_attn_res_add_rmsnorm_lamport(
        prefix_sum,
        addend.workspace.uc_local,
        addend.workspace.flags.data_ptr(),
        block_residual,
        proj.weight.reshape(-1),
        norm.weight,
        output_norm.weight,
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )


def _k3_attn_res_pdl_moe_add(
    deferred: _DeferredMoEResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    mxfp8: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """``(prefix_sum, normed mix)`` of a deferred MoE residual update, or ``None``; with
    ``mxfp8``, ``(prefix_sum, normed mix, xq, xs)`` (the MXFP8 form of the mix)."""
    residual, routed, shared = deferred
    if residual.shape[0] > _K3_ATTN_RES_PDL_MAX_M:
        return None
    from ..cute_dsl_kernels.blackwell import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (routed, shared), block_residual, *weights):
        return None
    M, H = residual.shape
    _note_attn_res_fusion("moe_add+attn_res+norm/pdl", True, M, H, int(block_residual.shape[0]) + 1)
    op = (
        torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm_mx
        if mxfp8
        else torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm
    )
    return op(
        residual,
        routed,
        shared,
        block_residual,
        *weights,
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[torch.Tensor]:
    """Fuse attention-residual mixing with its immediately following norm."""
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # Measured on Rubin (SM107) over M in {1, 2, 4, 8, 16, 32} x N in {2, 5, 9,
    # 12}: the fused kernel beat the unfused add + attn_res_fwd + RMSNorm path by
    # 33-53% in every cell, because at these sizes all three kernels are bound by
    # their fixed per-launch cost rather than by bandwidth or occupancy, and
    # collapsing three into one removes two of those costs regardless of M. See
    # notes/run_records/kimi_k3/0901_attn_res_multi_token.md.
    #
    # The ceiling is the largest token count measured, which is also the
    # production max_batch_size. Prefill stays on the unfused path on a
    # measured negative result, recorded on GB300 at T=8192:
    #
    #   Folding the trailing RMSNorm into the PERSISTENT attn_res grid was
    #   41-108% slower than attn_res_fwd + FlashInfer RMSNorm. The added
    #   RMSNorm cost 150-159 us, 4.1-4.35x the standalone kernel, because the
    #   persistent grid runs one CTA per SM and loops over tokens, so a
    #   CTA-wide reduction inside that loop serializes across tokens.
    #   FlashInfer instead exposes an independent CTA per token.
    #
    # (branch xguan/kimi_k3_attn_res_fusion @ 7293523d99, run record
    # kimi_k3_attn_res_fusion_kernel_opt/20260730_185026 on ociaga.)
    #
    # That result is about the persistent topology, and this gate selects the
    # single-CTA / split-K kernels instead -- which put one CTA (or one
    # cluster) on each token, i.e. the same shape the note credits FlashInfer
    # with. Whether those hold up at prefill is a separate question that has
    # not been measured; the ceiling stays at the top of the measured range
    # until it is.
    pdl_out = _k3_attn_res_pdl(prefix_sum, None, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        try:
            persistent_op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
        except (AttributeError, RuntimeError):
            return None
        _, output = persistent_op(
            prefix_sum.reshape(M, 1, H).contiguous(),
            None,
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            _rms_norm_eps(output_norm),
        )
        _note_attn_res_fusion("attn_res+norm/persistent", True, M, H, N)
        return output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("attn_res+norm", False, M, H, N)
        return None
    try:
        attn_res_rmsnorm_op = torch.ops.trtllm.attn_res_rmsnorm_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output = attn_res_rmsnorm_op(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fuse ``prefix_sum + addend``, attention-residual, and trailing norm.

    The production residual add produces a BF16 tensor that remains live
    across the following MLP. The kernel therefore returns that materialized,
    BF16-rounded prefix sum alongside the normalized attention-residual
    output, while avoiding a separate add launch and a re-read of the
    intermediate by attention-residual selection.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or addend.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not addend.is_cuda
        or not block_residual.is_cuda
        or prefix_sum.shape != addend.shape
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # Same measured window as _apply_attn_res_rmsnorm_fused above.
    pdl_out = _k3_attn_res_pdl(prefix_sum, addend, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("add+attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        try:
            persistent_op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
        except (AttributeError, RuntimeError):
            return None
        updated_prefix_sum, output = persistent_op(
            prefix_sum.reshape(M, 1, H).contiguous(),
            addend.reshape(M, 1, H).contiguous(),
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            _rms_norm_eps(output_norm),
        )
        _note_attn_res_fusion("add+attn_res+norm/persistent", True, M, H, N)
        return updated_prefix_sum.reshape(M, H), output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("add+attn_res+norm", False, M, H, N)
        return None
    try:
        attn_res_add_rmsnorm_op = torch.ops.trtllm.attn_res_add_rmsnorm_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    addend_kernel = addend.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    updated_prefix_sum, output = attn_res_add_rmsnorm_op(
        layer_kernel,
        addend_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: KimiK3RMSNorm
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Unless ``KIMI_K3_FUSED_ATTN_RES=0``, inputs fitting the fused kernel's
    contract dispatch directly to the in-tree ``trtllm::attn_res_fwd`` op.
    Only the fallback boundary restores the HF ``[M, K, H]`` layout.
    """
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
        if fused is not None:
            return fused
        # The fp32 reference below is orders of magnitude slower at prefill
        # sizes; say so once per shape rather than fall back silently.
        _note_attn_res_fusion(
            "attn_res",
            False,
            prefix_sum.shape[0],
            prefix_sum.shape[1],
            int(block_residual.shape[0]) + 1,
        )
    block_residual_hf = block_residual.transpose(0, 1)
    v = torch.cat((block_residual_hf, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _apply_attn_res_and_rmsnorm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_rmsnorm_fused(prefix_sum, block_residual, proj, norm, output_norm)
        if fused is not None:
            return fused
    return output_norm(_apply_attn_res(prefix_sum, block_residual, proj, norm))


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm.

    ``addend`` may be a ``LamportResult`` (``TRTLLM_K3_AR_PUSH_AMID``): the attention block's
    all-reduce result left in the MNNVL workspace, read in place where the PDL attn_res kernel
    serves the mix and gathered into a tensor otherwise."""
    if isinstance(addend, LamportResult):
        fused = _k3_attn_res_pdl_lamport(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
        if fused is not None:
            return fused
        addend = addend.materialize()
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_add_rmsnorm_fused(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
        if fused is not None:
            return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm
    )


# ---------------------------------------------------------------------------
# Dense / shared-expert MLP: fused [gate | up] layout (``GatedMLP``).
#
# The HF checkpoint stores separate ``gate_proj`` / ``up_proj`` tensors;
# ``load_weights`` row-concatenates them into ``gate_up_proj`` (see
# ``_gate_up_ckpt_keys``), replacing two GEMMs + torch.cat with one GEMM.
# ---------------------------------------------------------------------------


_GATE_UP_FUSED_SUFFIX = ".gate_up_proj.weight"


def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:
    """Checkpoint ``(gate_proj, up_proj)`` keys whose row-concat loads the
    fused ``gate_up_proj`` parameter named by ``fused_key``."""
    return (
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".gate_proj.weight"),
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".up_proj.weight"),
    )


# ---------------------------------------------------------------------------
# FP8 block-scale weight read for attention and replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Same predicate ``FP8BlockScalesLinearMethod`` uses for the generic
    FP8-block-scale ``Linear``. Recomputed rather than cached: the SM version
    cannot change within a run, so weight preparation and forward always agree.
    """
    from ..cute_dsl_utils import IS_CUTLASS_DSL_INTERNAL_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_INTERNAL_AVAILABLE


def _k3_lowm_gemm_can_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    allow_partial_n_tile: bool = False,
) -> bool:
    """Whether ``trtllm::k3_fp8w_lowm_gemm`` serves this projection (registers the op)."""
    from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm.can_run(
        x, weight, weight_scale, allow_partial_n_tile=allow_partial_n_tile
    )


@functools.lru_cache(maxsize=None)
def _k3_deep_gemm_num_sms(device_index: int) -> int:
    """DeepGEMM SM count of the FP8 weight-read GEMM, or 0 to keep DeepGEMM's own."""
    if _K3_DEEPGEMM_SM_RESERVE == 0:
        return 0
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    if _K3_DEEPGEMM_SM_RESERVE >= sms:
        raise ValueError(
            f"TRTLLM_K3_DEEPGEMM_SM_RESERVE={_K3_DEEPGEMM_SM_RESERVE} leaves no SM of {sms}"
        )
    return sms - _K3_DEEPGEMM_SM_RESERVE


def _k3_deep_gemm_fp8_swap_ab(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """``fp8_swap_ab_gemm`` with ``TRTLLM_K3_DEEPGEMM_SM_RESERVE`` SMs left free.

    DeepGEMM's SM count is process state, read when a GEMM is configured, JIT-compiled (the
    kernel is specialized on it) and launched; it is set around this call and restored after,
    so other DeepGEMM users keep the full grid. The choice depends only on K, so each
    projection always runs (and its autotuner warmup compiles) one grid size.
    """
    num_sms = 0
    if (
        _K3_DEEPGEMM_SM_RESERVE
        and x.shape[-1] >= _K3_DEEPGEMM_SM_RESERVE_MIN_K
        and not torch.compiler.is_compiling()
    ):
        num_sms = _k3_deep_gemm_num_sms(x.device.index)
    if num_sms:
        from ... import deep_gemm

        saved = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
    try:
        return torch.ops.trtllm.fp8_swap_ab_gemm(
            x, weight, weight_scale, output_dtype=output_dtype, disable_ue8m0_cast=True
        )
    finally:
        if num_sms:
            deep_gemm.set_num_sms(saved)


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. Both prepare UE8M0 scales
    for DeepGEMM and quantize BF16 activations inside the GEMM.

    On Rubin, the activation is quantized directly into the GEMM's scale layout
    by the specialized CuTe quantizer. SM100/103 use DeepGEMM's internal
    activation quantization. The CuTe wrapper caches its scalar alpha so Rubin
    forward does not launch a fill kernel during CUDA graph capture.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        self._weights_transformed = True
        self._fused_projection = None
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Set by a caller for the next forward only (``forward``, ``forward_fp8_packed`` or
        # ``forward_prequantized`` consumes and clears it): a contiguous BF16
        # ``[M, out_features]`` tensor that the low-M and mid-M GEMMs write their result into.
        # Other paths return their own tensor, so the caller compares the result with the
        # buffer it handed in.
        self.next_output: Optional[torch.Tensor] = None
        # Set by a caller for the next forward only (every forward consumes and clears it): the
        # ``LamportPushTarget`` (``AllReduce.lamport_push_target``) of the all-reduce that
        # follows this projection (TRTLLM_K3_AR_PUSH). Where the low-M GEMM without split-K
        # serves ``forward`` / ``forward_gated``, it stores its output into that all-reduce's
        # Lamport buffers instead, marks the target pushed and returns the all-reduce's
        # output tensor, which ``AllReduce.finish_lamport_push`` fills.
        self.next_ar_push = None
        # Copied from the source ``Linear`` (the MLA fused q_a/kv_a projection sets it): the
        # low-M GEMM may serve this weight although N is not a multiple of 128.
        self.fp8w_lowm_partial_n_tile = False
        # TRTLLM_K3_FP8_MID_GEMM family of this projection (set when it is swapped in) and,
        # once prepared, its weight scale expanded to per-row scale atoms for the mid-M GEMM.
        self.k3_mid_family: Optional[str] = None
        self.mid_weight_sf: Optional[torch.Tensor] = None

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether the KDA fused producer emits this GEMM's scale layout."""
        return _rubin_mxfp8_is_available()

    @staticmethod
    def _pack_scales(
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a K128 block scale to the active GEMM's required layout."""
        from ...quantization.utils.fp8_utils import resmooth_to_fp8_e8m0

        weight_fp8, k128_scale = resmooth_to_fp8_e8m0(
            weight_fp8.contiguous(), weight_scale.contiguous().float()
        )
        mn, k = weight_fp8.shape[0], weight_fp8.shape[1]

        if _rubin_mxfp8_is_available():
            from ...quantization.utils.fp8_utils import (
                transform_k128_scales_to_cutedsl_mxfp8_layout,
            )

            weight_scale = transform_k128_scales_to_cutedsl_mxfp8_layout(k128_scale, mn=mn, k=k)
        else:
            from ...quantization.utils.fp8_utils import transform_sf_into_required_layout

            weight_scale = transform_sf_into_required_layout(
                k128_scale,
                mn=mn,
                k=k,
                recipe=(1, 128, 128),
                is_sfa=False,
            )
        return weight_fp8, weight_scale

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 ``[out, in]`` weight -> FP8 weight and backend-ready scale.

        Both dims must be multiples of 128. Because the 128x128 block scale is
        computed per block, concatenating several such weights along ``out``
        and quantizing the result is per-block identical to quantizing each
        separately (no block crosses a 128-aligned boundary), so a fused
        weight's row slices equal the individually quantized weights.
        """
        # Lazy import: only pulled in on the FP8 path.
        from ...deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        return _Fp8BlockScaleWeightReadLinear._pack_scales(weight_fp8, weight_scale)

    @staticmethod
    def prepare_checkpoint_scale(
        weight_fp8: torch.Tensor, weight_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Checkpoint FP8 + FP32 K128 scale -> FP8 and backend-ready scale.

        This is ``quantize_weight`` without its first step. A checkpoint that
        already stores FP8_PB_WO weights (``nvidia/Kimi-K3-NVFP4``) supplies
        exactly what ``per_block_cast_to_fp8`` would have produced, so the only
        work left is the scale bridge -- see ``_pack_scales``.

        Taking this route instead of dequantizing to BF16 and re-quantizing
        avoids a lossy round trip: the re-quantization recomputes each block's
        scale from the dequantized values, which need not reproduce the
        scale the checkpoint was written with.
        """
        return _Fp8BlockScaleWeightReadLinear._pack_scales(weight_fp8, weight_scale)

    @classmethod
    def from_linear(cls, linear: nn.Linear | TrtllmLinear) -> "_Fp8BlockScaleWeightReadLinear":
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = linear.quant_config
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = False
            result.fp8w_lowm_partial_n_tile = getattr(linear, "fp8w_lowm_partial_n_tile", False)
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            self.weight, self.weight_scale = self.prepare_checkpoint_scale(
                self.weight, self.weight_scale
            )
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()

    def prepare_mid_gemm(self) -> bool:
        """Expand the weight scale for the mid-M GEMM if this projection's family is enabled."""
        if self.k3_mid_family not in _K3_FP8_MID_FAMILIES or _rubin_mxfp8_is_available():
            return False
        if self._fused_projection is not None:
            # A row view of the KDA fused q/k/v/g weight; the fused module serves forward.
            return False
        self.transform_weights()
        from ..cute_dsl_kernels.blackwell.k3_fp8_mid_gemm import expand_weight_scale

        self.mid_weight_sf = expand_weight_scale(
            self.weight_scale, self.out_features, self.in_features
        )
        return True

    def mid_gemm_applies(self, x_flat: torch.Tensor) -> bool:
        """Whether ``trtllm::k3_fp8_mid_gemm`` serves this forward."""
        if self.mid_weight_sf is None or torch.compiler.is_compiling():
            return False
        if not _K3_FP8_MID_MIN_M <= x_flat.shape[0] <= _K3_FP8_MID_MAX_M:
            return False
        from ..cute_dsl_kernels.blackwell import k3_fp8_mid_gemm

        m, n = x_flat.shape[0], self.out_features
        return k3_fp8_mid_gemm.tile_for(
            m, n, self.in_features
        ) is not None and k3_fp8_mid_gemm.can_run(x_flat, self.weight, self.mid_weight_sf)

    def _mid_gemm_applies_rows(self, num_rows: int, packed: bool = False) -> bool:
        if self.mid_weight_sf is None or torch.compiler.is_compiling():
            return False
        if not _K3_FP8_MID_MIN_M <= num_rows <= _K3_FP8_MID_MAX_M:
            return False
        from ..cute_dsl_kernels.blackwell import k3_fp8_mid_gemm

        tile = k3_fp8_mid_gemm.tile_for(
            num_rows, self.out_features, self.in_features, packed=packed
        )
        return tile is not None

    def takes_fp8_packed_input(self, num_rows: int) -> bool:
        """Whether :meth:`forward_fp8_packed` serves ``num_rows`` rows with the GEMM ``forward``
        would run on their BF16 values (the mid-M GEMM, or DeepGEMM where ``forward`` quantizes
        for ``fp8_swap_ab_gemm``), so a producer may emit the 1x128 FP8 operand instead."""
        if _rubin_mxfp8_is_available() or torch.compiler.is_compiling():
            return False
        if self._mid_gemm_applies_rows(num_rows, packed=True):
            return True
        if _K3_FP8W_LOWM_GEMM:
            from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

            if num_rows <= k3_fp8w_lowm_gemm.max_m_for(self.out_features, self.in_features):
                return False
        if _K3_DEEPGEMM_SM_RESERVE and self.in_features >= _K3_DEEPGEMM_SM_RESERVE_MIN_K:
            return False
        return True

    def _take_next_output(self, num_rows: int) -> Optional[torch.Tensor]:
        """Clear :attr:`next_output`; return it if it is a contiguous, 16-byte aligned BF16
        ``[num_rows, out_features]`` tensor."""
        out, self.next_output = self.next_output, None
        if (
            out is None
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (num_rows, self.out_features)
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            return None
        return out

    def _take_next_ar_push(self, x_flat: torch.Tensor, gated: bool = False):
        """Clear :attr:`next_ar_push`; return it if the low-M GEMM can push this forward's
        ``[M, out_features]`` output into it (low-M kernel without split-K)."""
        push, self.next_ar_push = self.next_ar_push, None
        if (
            push is None
            or x_flat.dtype != torch.bfloat16
            or tuple(push.output.shape) != (x_flat.shape[0], self.out_features)
        ):
            return None
        from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.ar_push_tactic(
            x_flat.shape[0], self.out_features, self.in_features, gated=gated
        )
        return push if tactic is not None else None

    def _run_ar_push(self, x_flat: torch.Tensor, push, gate=None) -> torch.Tensor:
        """The low-M GEMM with the all-reduce push epilogue; returns the all-reduce output."""
        from ..cute_dsl_kernels.blackwell.k3_fp8w_lowm_gemm import run_k3_fp8w_lowm_gemm_ar_push

        run_k3_fp8w_lowm_gemm_ar_push(
            x_flat, self.weight, self.weight_scale, push.workspace, push.one_shot, gate=gate
        )
        push.pushed = True
        return push.output

    def forward_fp8_packed(self, act: torch.Tensor, act_scale: torch.Tensor) -> torch.Tensor:
        """Project an activation quantized upstream: E4M3 ``[M, K]`` plus DeepGEMM-packed UE8M0
        1x128 scales (int32 ``[M, ceil(K / 512)]``, strides ``(1, ld)``), the bytes
        ``fp8_quantize_1x128_packed_ue8m0`` makes of the BF16 activation. Same output as
        :meth:`forward` on that activation where :meth:`takes_fp8_packed_input` holds."""
        self.next_ar_push = None
        out_buffer = self._take_next_output(act.shape[0])
        if self._mid_gemm_applies_rows(act.shape[0], packed=True):
            from ..cute_dsl_kernels.blackwell import k3_fp8_mid_gemm

            if k3_fp8_mid_gemm.can_run_packed(act, act_scale, self.weight, self.mid_weight_sf):
                if out_buffer is not None:
                    return k3_fp8_mid_gemm.k3_fp8_mid_gemm_packed(
                        act, act_scale, self.weight, self.mid_weight_sf, out=out_buffer
                    )
                return torch.ops.trtllm.k3_fp8_mid_gemm_packed(
                    act, act_scale, self.weight, self.mid_weight_sf
                )
        return torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
            act,
            act_scale,
            self.weight,
            self.weight_scale,
            output_dtype=torch.bfloat16,
            disable_ue8m0_cast=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if lora_params:
            raise NotImplementedError("Kimi K3 FP8 weight read does not support LoRA.")
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        out_buffer = self._take_next_output(x_flat.shape[0])
        ar_push = self._take_next_ar_push(x_flat)

        if self.mid_weight_sf is not None and self.mid_gemm_applies(x_flat):
            if out_buffer is not None:
                from ..cute_dsl_kernels.blackwell.k3_fp8_mid_gemm import k3_fp8_mid_gemm

                out = k3_fp8_mid_gemm(x_flat, self.weight, self.mid_weight_sf, out=out_buffer)
            else:
                out = torch.ops.trtllm.k3_fp8_mid_gemm(x_flat, self.weight, self.mid_weight_sf)
        elif _rubin_mxfp8_is_available():
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                act_fp8,
                self.weight,
                act_scale,
                self.weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        elif (
            _K3_FP8W_LOWM_GEMM
            and x.dtype == torch.bfloat16
            and _k3_lowm_gemm_can_run(
                x_flat,
                self.weight,
                self.weight_scale,
                allow_partial_n_tile=_K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile,
            )
        ):
            if ar_push is not None:
                out = self._run_ar_push(x_flat, ar_push)
            elif out_buffer is not None:
                from ..cute_dsl_kernels.blackwell.k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm_out

                out = k3_fp8w_lowm_gemm_out(x_flat, self.weight, self.weight_scale, out_buffer)
            else:
                out = torch.ops.trtllm.k3_fp8w_lowm_gemm(x_flat, self.weight, self.weight_scale)
        else:
            out = _k3_deep_gemm_fp8_swap_ab(x_flat, self.weight, self.weight_scale, x.dtype)
        return out.reshape(out_shape)

    def _runs_low_m_kernel(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows would run the low-M kernel (the
        input-independent part of its dispatch: not the mid-M or Rubin paths, the low-M GEMM on,
        ``num_rows`` within the low-M kernel's range)."""
        if (
            not _K3_FP8W_LOWM_GEMM
            or _rubin_mxfp8_is_available()
            or torch.compiler.is_compiling()
            or self._mid_gemm_applies_rows(num_rows)
        ):
            return False
        if self.weight.shape[0] % 128 and not (
            _K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile
        ):
            return False
        from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

        return 1 <= num_rows <= k3_fp8w_lowm_gemm.low_m_max_for(*self.weight.shape)

    def takes_mxfp8_input(self, num_rows: int) -> bool:
        """Whether a producer should hand :meth:`forward_mxfp8` the MXFP8 form of ``num_rows``
        rows: where :meth:`forward` would run the low-M kernel, whose in-kernel quantization the
        MXFP8 GEMM replaces."""
        return self._runs_low_m_kernel(num_rows)

    def forward_mxfp8(self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """:meth:`forward` on the BF16 rows ``x``, bit for bit, given their MXFP8 form ``(xq,
        xs)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, e.g. from
        ``trtllm::k3_attn_res_*_mx``): the low-M GEMM on the prequantized operand where
        :meth:`forward` would run the low-M kernel, else :meth:`forward` on ``x``."""
        from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8

        x_flat = x.reshape(-1, x.shape[-1])
        allow_partial = _K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile
        if not (
            x.dtype == torch.bfloat16
            and self._runs_low_m_kernel(x_flat.shape[0])
            and xq.shape[0] == x_flat.shape[0]
            and _k3_lowm_gemm_can_run(
                x_flat, self.weight, self.weight_scale, allow_partial_n_tile=allow_partial
            )
            and k3_fp8w_lowm_mxfp8.can_run_mxfp8(
                xq, xs, self.weight, self.weight_scale, allow_partial_n_tile=allow_partial
            )
        ):
            return self.forward(x)
        out = self._take_next_output(x_flat.shape[0])
        # The MXFP8 GEMM has no all-reduce push epilogue: the caller runs the all-reduce.
        self.next_ar_push = None
        if out is None:
            out = torch.empty(
                (x_flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        k3_fp8w_lowm_mxfp8.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, self.weight, self.weight_scale, out)
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def gated_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated` serves ``x`` and ``gate`` with one gated low-M GEMM.

        That is where :meth:`forward` would run the low-M GEMM on ``x * gate`` (same kernel
        family and split), so the gated launch is bitwise equal to the multiply plus
        :meth:`forward`.
        """
        if (
            not _K3_FP8W_LOWM_GEMM
            or _rubin_mxfp8_is_available()
            or torch.compiler.is_compiling()
            or x.dtype != torch.bfloat16
            or x.dim() != 2
            or not x.is_contiguous()
        ):
            return False
        if self.mid_weight_sf is not None and self._mid_gemm_applies_rows(x.shape[0]):
            return False
        from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.can_run_gated(
            x,
            gate,
            self.weight,
            self.weight_scale,
            allow_partial_n_tile=_K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile,
        )

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(x).sigmoid()``, bit for bit; one low-M GEMM launch with a sigmoid
        epilogue where :meth:`forward` would run the low-M GEMM."""
        x_flat = x.reshape(-1, x.shape[-1])
        if (
            _K3_FP8W_LOWM_GEMM
            and not _rubin_mxfp8_is_available()
            and not torch.compiler.is_compiling()
            and x.dtype == torch.bfloat16
            and not (self.mid_weight_sf is not None and self.mid_gemm_applies(x_flat))
            and _k3_lowm_gemm_can_run(
                x_flat,
                self.weight,
                self.weight_scale,
                allow_partial_n_tile=_K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile,
            )
        ):
            from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

            if x_flat.shape[0] <= k3_fp8w_lowm_gemm.low_m_max_for(
                self.out_features, self.in_features
            ):
                self.next_output = None
                self.next_ar_push = None
                out = k3_fp8w_lowm_gemm.k3_fp8w_lowm_gemm_sigmoid(
                    x_flat, self.weight, self.weight_scale
                )
                return out.reshape(x.shape[:-1] + (self.out_features,))
        return self.forward(x).sigmoid()

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``forward(x * gate)`` in one gated low-M GEMM launch (where :meth:`gated_applies`).

        ``x`` and ``gate`` are BF16 ``[M, in_features]``; the gate may be a row-strided view.
        """
        from ..cute_dsl_kernels.blackwell.k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm_gated

        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            return self._run_ar_push(x, ar_push, gate=gate)
        return k3_fp8w_lowm_gemm_gated(
            x, gate, self.weight, self.weight_scale, out=self._take_next_output(x.shape[0])
        )

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project an activation that the fused KDA producer already quantized."""
        self.next_output = None
        self.next_ar_push = None
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
            activation.reshape(-1, activation.shape[-1]),
            self.weight,
            activation_scale,
            self.weight_scale,
            output_dtype=torch.bfloat16,
            fine_grained_m=True,
        )
        return out.reshape(activation.shape[:-1] + (self.out_features,))


def _swap_linear_to_fp8_weight_read(
    parent: nn.Module,
    attr: str,
    linear_types: Tuple[type, ...] = (nn.Linear,),
) -> int:
    """Replace ``parent.<attr>`` with an FP8 weight-read module if it is a
    plain linear of one of ``linear_types``; return the number of modules
    converted (0 or 1), so callers can accumulate a conversion count.

    Transfer checkpoint FP8 storage or release the original BF16 storage
    after conversion. The loader's transient parameter map must not retain
    an unused BF16 copy until loading completes.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    converted = _Fp8BlockScaleWeightReadLinear.from_linear(child)
    converted.k3_mid_family = _K3_FP8_MID_FAMILY_OF_ATTR.get(attr)
    setattr(parent, attr, converted)
    child.weight.data = child.weight.data.new_empty(0)
    return 1


class _Fp8RowParallelLinear(nn.Module):
    """FP8 weight-read GEMM of a row-parallel projection followed by the all-reduce that the
    replaced ``Linear`` (``reduce_output``) ran on its output."""

    def __init__(self, fp8: "_Fp8BlockScaleWeightReadLinear", all_reduce: Optional[nn.Module]):
        super().__init__()
        self.fp8 = fp8
        self.all_reduce = all_reduce
        self.in_features = fp8.in_features
        self.out_features = fp8.out_features

    def forward(
        self,
        x: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self.fp8(x)
        if self.all_reduce is not None and (
            all_reduce_params is None or all_reduce_params.enable_allreduce
        ):
            out = self.all_reduce(out, all_reduce_params=all_reduce_params)
        return out


def _convert_dense_mlps_to_fp8_weight_read(model: nn.Module) -> int:
    """``KIMI_K3_FP8_WEIGHT_READ_DENSE_MLP``: swap the BF16 ``GatedMLP`` of the dense (non-MoE)
    layers to FP8 block-scale weight reads (``_Fp8BlockScaleWeightReadLinear``, the low-M GEMM at
    decode batch sizes). The per-rank intermediate size need not be a multiple of the 128-wide
    scale blocks: each half of ``gate_up`` then gains zero rows and ``down`` zero columns up to
    the next multiple, so the padded intermediate columns are ``act(0) * 0 = 0`` and add nothing.
    The row-parallel ``down`` keeps its all-reduce (``_Fp8RowParallelLinear``). Returns the number
    of projections converted."""
    count = 0
    for layer in model.layers:
        if getattr(layer, "is_moe", True):
            continue
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, GatedMLP):
            continue
        gate_up, down = mlp.gate_up_proj, mlp.down_proj
        if not isinstance(gate_up, (nn.Linear, TrtllmLinear)) or not isinstance(
            down, (nn.Linear, TrtllmLinear)
        ):
            continue
        if getattr(gate_up, "bias", None) is not None or getattr(down, "bias", None) is not None:
            continue
        w_gu, w_dn = gate_up.weight.data, down.weight.data
        if w_gu.dtype != torch.bfloat16 or w_dn.dtype != torch.bfloat16:
            continue
        inter = w_dn.shape[1]
        if w_gu.shape[0] != 2 * inter or w_gu.shape[1] % 128 or w_dn.shape[0] % 128:
            continue
        pad = -inter % 128
        if pad:
            zeros = w_gu.new_zeros((pad, w_gu.shape[1]))
            w_gu = torch.cat((w_gu[:inter], zeros, w_gu[inter:], zeros))
            w_dn = torch.cat((w_dn, w_dn.new_zeros((w_dn.shape[0], pad))), dim=1)
        quantize = _Fp8BlockScaleWeightReadLinear.quantize_weight
        mlp.gate_up_proj = _Fp8BlockScaleWeightReadLinear(*quantize(w_gu), w_gu.shape[0])
        all_reduce = down.all_reduce if getattr(down, "reduce_output", False) else None
        mlp.down_proj = _Fp8RowParallelLinear(
            _Fp8BlockScaleWeightReadLinear(*quantize(w_dn), w_dn.shape[0]), all_reduce
        )
        gate_up.weight.data = gate_up.weight.data.new_empty(0)
        down.weight.data = down.weight.data.new_empty(0)
        count += 2
    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count


def _convert_moe_mlps_to_fp8_weight_read(
    model: nn.Module, include_fused_gate_up: bool = True, include_tp_shards: bool = False
) -> int:
    """Swap the replicated MoE-layer MLP projections to an FP8 weight read.

    Targets the shared-expert MLP (gate/up/down) and the latent up/down
    projection on every MoE layer — the bias-free BF16 projections that
    attention data-parallelism re-reads in full each decode step. Attention
    follows checkpoint quantization separately. Routed experts and the dense
    layer-0 MLP are not converted. With include_tp_shards the TP-sharded
    shared expert is converted as well; its row-parallel down projection then
    no longer reduces its output, and the MoE runtime applies that reduction
    with the original module (shared_down_all_reduce). Returns the number
    of projections converted.
    """
    count = 0

    for layer in model.layers:
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            # GatedMLP fuses gate and up into gate_up_proj; keep the split
            # names too so either MLP layout converts. The fused gate_up read
            # only pays off when attention DP re-reads it per rank per step;
            # under TP the bf16 GEMM overlaps on the aux stream and the FP8
            # quantize+GEMM would serialize onto the critical path.
            shared_attrs = (
                ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
                if include_fused_gate_up
                else ("gate_proj", "up_proj", "down_proj")
            )
            for attr in shared_attrs:
                child = getattr(shared, attr, None)
                if isinstance(child, TrtllmLinear) and child.tp_size != 1:
                    if not include_tp_shards:
                        continue
                    if child.reduce_output:
                        # The FP8 read is a plain GEMM; the MoE runtime applies
                        # the row-parallel reduction with the original module.
                        moe.shared_down_all_reduce = child.all_reduce
                count += _swap_linear_to_fp8_weight_read(
                    shared, attr, linear_types=(nn.Linear, TrtllmLinear)
                )
        for attr in ("routed_expert_down_proj", "routed_expert_up_proj"):
            count += _swap_linear_to_fp8_weight_read(moe, attr)

    # Return the freed BF16 blocks to the driver so the raw (non-caching-
    # allocator) allocations made during executor creation succeed on the
    # tight DEP16 memory headroom.
    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count


# ---------------------------------------------------------------------------
# Latent MoE block using the unified ConfigurableMoE stack.
# ---------------------------------------------------------------------------

# Routed-expert key spellings that ModelOpt emits for Kimi K3. The NVFP4
# checkpoint (``nvidia/Kimi-K3-NVFP4``) lists every prefix x module-name
# combination in ``quantized_layers``, so a lookup over this product finds it
# without needing the MiniMax-M3-style prefix normalization in ``ModelConfig``.
_K3_ROUTED_EXPERT_KEY_PREFIXES = ("language_model.model.", "model.", "")
_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")

# Routed-expert quantization used when the checkpoint declares nothing per
# layer. The original ``moonshotai/Kimi-K3`` ships a compressed-tensors
# ``mxfp4-pack-quantized`` config with no ModelOpt per-layer entries, and that
# checkpoint is what this default has always served.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


# ---------------------------------------------------------------------------
# Routed-expert checkpoint layouts.
#
# K3 streams routed experts one at a time (see ``load_weights``), so the tensor
# names, the per-expert loader and the finalization it needs are all decided by
# the checkpoint's routed-expert quantization rather than by the MoE backend.
# Three call sites need that decision — the expected-key plan, the loader, and
# the file grouping — so it lives in one spec instead of three conditionals.
# ---------------------------------------------------------------------------


# Loader signature: ``(backend, base, ckpt_expert_idx, slot_expert_idx, local_slot_id,
# get_tensor)``. ``ckpt_expert_idx`` names the checkpoint tensors; ``slot_expert_idx`` is the
# routed-expert id the slot serves (the same id unless TRTLLM_K3_EXPERT_PLACEMENT relabels it).
def _load_packed_mxfp4_expert(
    backend, base, ckpt_expert_idx, slot_expert_idx, local_slot_id, get_tensor
) -> None:
    prefix = f"{base}.{ckpt_expert_idx}"
    backend.quant_method.load_packed_mxfp4_expert(
        backend,
        global_expert_id=slot_expert_idx,
        local_slot_id=local_slot_id,
        w1_weight=get_tensor(f"{prefix}.w1.weight_packed"),
        w1_weight_scale=get_tensor(f"{prefix}.w1.weight_scale"),
        w2_weight=get_tensor(f"{prefix}.w2.weight_packed"),
        w2_weight_scale=get_tensor(f"{prefix}.w2.weight_scale"),
        w3_weight=get_tensor(f"{prefix}.w3.weight_packed"),
        w3_weight_scale=get_tensor(f"{prefix}.w3.weight_scale"),
    )


def _load_nvfp4_expert(
    backend, base, ckpt_expert_idx, slot_expert_idx, local_slot_id, get_tensor
) -> None:
    backend.quant_method.load_streaming_nvfp4_expert(
        backend,
        global_expert_id=slot_expert_idx,
        local_slot_id=local_slot_id,
        **{
            f"{w}_{kind}": get_tensor(f"{base}.{ckpt_expert_idx}.{w}.{kind}")
            for w in ("w1", "w2", "w3")
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        },
    )


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # NVFP4 defers cat/pad/interleave and the alpha computation to
    # ``process_weights_after_loading``; the MXFP4 loaders write through.
    needs_layer_finalize: bool


_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.W4A8_MXFP4_MXFP8: _K3ExpertCkptSpec(
        kinds=("weight_packed", "weight_scale"),
        loader=_load_packed_mxfp4_expert,
        loaded_slots_attr="_packed_mxfp4_loaded_slots",
        needs_layer_finalize=False,
    ),
    QuantAlgo.NVFP4: _K3ExpertCkptSpec(
        kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
        loader=_load_nvfp4_expert,
        loaded_slots_attr="_streamed_expert_slots",
        needs_layer_finalize=True,
    ),
}


def _k3_expert_ckpt_spec(quant_algo: Optional[QuantAlgo]) -> _K3ExpertCkptSpec:
    spec = _K3_EXPERT_CKPT_SPECS.get(quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in _K3_EXPERT_CKPT_SPECS)}."
        )
    return spec


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block backed by ConfigurableMoE."""

    # TRTLLM_K3_LATENT_AR_OWN_WS: the routed latent all-reduce's own MNNVL workspace (``None``: it
    # shares the one of the shared expert's all-reduce).
    _latent_ar_ws: Optional[Dict[str, Any]] = None
    # TRTLLM_K3_LATENT_AR_SIDE_CLEAR: recorded once latent_up is issued, and the stream that then
    # re-arms the latent all-reduce's Lamport buffer (the latent-down stream; ``None``: no side clear).
    moe_lu_event: Optional[torch.cuda.Event] = None
    latent_ar_clear_stream: Optional[torch.cuda.Stream] = None
    # TRTLLM_K3_EXPERT_L2_PREFETCH: the model's shared state (``None``: off), this layer's routing
    # record (allocated outside graph capture), the routing-tail outputs of the current forward,
    # whether a routing of this layer was ever recorded, and this layer's expert weight ranges with
    # the weight addresses they were built for.
    _expl2_state: Optional[_ExpertL2PrefetchState] = None
    _expl2_record: Optional[torch.Tensor] = None
    _expl2_route: Optional[Tuple[torch.Tensor, ...]] = None
    _expl2_recorded = False
    _expl2_ranges: Tuple[Tuple[int, ...], List[Tuple[int, int, int]]] = ((), [])
    # TRTLLM_K3_EXPERT_REPLICAS: this layer's redundant expert copies (``None``: off) and the
    # routing tail's copy-choice table of this rank.
    expert_replicas = None
    _replica_table: Optional[torch.Tensor] = None

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._router_exact = _router_exact_for_layer(cfg, layer_idx)
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3 runtime expects latent_moe_use_norm=True")

        _cfg_beta = getattr(cfg, "activation_situ_beta", None)
        # Keep 0.0 (invalid, rejected downstream) instead of swallowing it
        # with ``or 1.0``. Shared-expert SituAndMul still keeps
        # ``linear_beta is None`` as "up half stays linear".
        situ_beta = float(_cfg_beta) if _cfg_beta is not None else 1.0
        situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
        situ_linear_beta_moe = float(situ_linear_beta) if situ_linear_beta is not None else 1.0
        dtype = torch.bfloat16

        # Routing scores stay fp32; with attention-DP off the gate GEMM runs
        # bf16xbf16 with fp32 accumulate/output (checkpoint stores the gate
        # weight in bf16; saves a per-layer input cast + fp32 splitK pair on
        # the bs1 decode path). Under attention-DP the legacy upcast-to-fp32
        # GEMM is kept: the bf16-input min-latency GEMM's different reduction
        # order flips borderline top-16 picks (GSM8K 96.7 -> 96.1/96.4,
        # 3-run bisect on 62b20dd868), and the bs1-latency win is irrelevant
        # at DEP batch sizes. KIMI_K3_ROUTER_BF16=1/0 forces either path.
        _router_bf16_env = os.environ.get("KIMI_K3_ROUTER_BF16")
        _router_bf16 = (
            _router_bf16_env == "1"
            if _router_bf16_env is not None
            else not model_config.mapping.enable_attention_dp
        )
        self.gate = KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16 if _router_bf16 else None)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config.quant_algo)
        routed_moe_kwargs = dict(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            dtype=dtype,
            # Kimi owns the latent reduction so it can order that collective
            # after the shared expert's auxiliary-stream reduction.
            reduce_results=False,
            model_config=routed_moe_model_config,
            override_quant_config=routed_quant_config,
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            # Let CommunicationFactory select the best available strategy.
            communication_method=None,
        )
        # trtllm-gen has fused SiTu FC1 cubins for two input formats and no
        # standalone SiTu activation kernel, so anything else dies here rather
        # than in a cubin lookup. Keep this set in sync with
        # TRTLLMGenFusedMoE._SITU_SUPPORTED_QUANT_ALGOS.
        #   NVFP4               group-16 Bmm_E2m1_E2m1E2m1_..._siTuGlu_*
        #   W4A8_MXFP4_MXFP8    group-32 Bmm_MxE4m3_MxE2m1MxE4m3_..._siTuGlu_*
        # Checked against the resolved backend, not the K3 architecture
        # branch, because the generic FP8_BLOCK_SCALES fallback in
        # ``resolve_moe_backend`` can also land on TRTLLM.
        if (
            routed_moe_model_config.moe_backend == "TRTLLM"
            and routed_quant_config.quant_algo
            not in {
                QuantAlgo.NVFP4,
                QuantAlgo.W4A8_MXFP4_MXFP8,
            }
        ):
            raise ValueError(
                f"Kimi K3 routed experts are quantized as "
                f"{routed_quant_config.quant_algo}, which the TRTLLM "
                "(trtllm-gen) MoE backend cannot serve: fused SiTu cubins "
                "exist only for NVFP4 and W4A8_MXFP4_MXFP8. Set "
                "moe_config.backend to one of the other backends in "
                "_routed_moe_model_config's supported_backends "
                "(CUTLASS / CUTEDSL / CUTEDSL_FC12 / MEGAMOE_CUTEDSL for "
                "NVFP4, MEGAMOE_DEEPGEMM for W4A8_MXFP4_MXFP8)."
            )

        # SiTU is named once, in the canonical form. ``create_moe_backend``
        # translates it into whichever signature the resolved backend wants --
        # per-expert tensors for CUTLASS, a backend-local enum for TRTLLM-Gen,
        # scalars for CuteDSL and MegaMoE. Keeping those dialects out of the
        # model is the point: an unsupported backend is rejected there with a
        # message naming the backend, not here.
        routed_moe_kwargs.update(
            activation="situ",
            situ_beta=float(situ_beta),
            situ_linear_beta=situ_linear_beta_moe,
        )
        # TRTLLM_K3_EXPERT_REPLICAS: every EP rank holds its home slots, then copy slots (the
        # routed MoE's slots are rank-major; routing still returns home slot ids).
        self.expert_replicas = expert_replicas_for_layer(
            layer_idx, self.num_experts, routed_moe_model_config.mapping.moe_ep_size
        )
        if self.expert_replicas is not None:
            from ..cute_dsl_kernels.blackwell import k3_route_tail

            replicas = self.expert_replicas
            if (
                replicas.num_home != k3_route_tail.REP_NUM_HOME
                or replicas.num_local > k3_route_tail.REP_LOCAL_CAP
                or len(replicas.entries) > k3_route_tail.REP_MAX_ENTRIES
            ):
                raise ValueError(
                    f"TRTLLM_K3_EXPERT_REPLICAS serves EP4 ({k3_route_tail.REP_NUM_HOME} home "
                    f"slots per rank), at most {k3_route_tail.REP_LOCAL_CAP} local slots and "
                    f"{k3_route_tail.REP_MAX_ENTRIES} two-copy experts per layer"
                )
            routed_moe_kwargs["num_experts"] = replicas.num_slots
        self.routed_experts = create_moe(**routed_moe_kwargs)
        if not isinstance(self.routed_experts, ConfigurableMoE):
            raise RuntimeError(
                "Kimi K3 requires ConfigurableMoE; ENABLE_CONFIGURABLE_MOE must not be disabled."
            )
        if routed_moe_model_config.moe_backend == "MEGAMOE_DEEPGEMM":
            from ..modules.fused_moe.mega_moe import MegaMoEDeepGemm

            if not isinstance(self.routed_experts.backend, MegaMoEDeepGemm):
                raise RuntimeError(
                    "Kimi K3 explicitly requested MEGAMOE_DEEPGEMM, but the "
                    f"MoE factory selected {type(self.routed_experts.backend).__name__}."
                )
        if routed_moe_model_config.moe_backend == "CUTEDSL":
            from ..modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE

            # Same guard, same reason as the MegaMoE ones below: create_moe
            # silently falls back when a backend declines the config, and a
            # request that quietly became CUTLASS would be measured as if it
            # were CuteDSL.
            #
            # Exact type, not isinstance: CuteDslB12xFusedMoE subclasses
            # CuteDslFusedMoE but is a different (SM120/121 FlashInfer-decode)
            # backend that has no SiTu epilogue, so isinstance would wave it
            # through.
            if type(self.routed_experts.backend) is not CuteDslFusedMoE:
                raise RuntimeError(
                    "Kimi K3 explicitly requested CUTEDSL, but the MoE factory "
                    f"selected {type(self.routed_experts.backend).__name__}."
                )
        if routed_moe_model_config.moe_backend == "CUTEDSL_FC12":
            from ..modules.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE

            # Same reason as CUTEDSL: CUTEDSL_FC12 is not in
            # NO_FALLBACK_BACKENDS, so a declined config becomes CUTLASS and
            # would be measured as the fused kernel. Exact type: the class
            # subclasses CuteDslFusedMoE.
            if type(self.routed_experts.backend) is not CuteDslFc12FusedMoE:
                raise RuntimeError(
                    "Kimi K3 explicitly requested CUTEDSL_FC12, but the MoE "
                    "factory selected "
                    f"{type(self.routed_experts.backend).__name__}."
                )
        if routed_moe_model_config.moe_backend == "MEGAMOE_CUTEDSL":
            from ..modules.fused_moe.mega_moe import MegaMoECuteDsl

            # Same guard as MEGAMOE_DEEPGEMM above, and for the same reason:
            # create_moe silently falls back when a backend declines the
            # config, and for MegaMoE the decline is easy to trigger (it is
            # EP-only and has its own token/top-k limits), so an explicit
            # request that quietly became CUTLASS would be measured as if it
            # were MegaMoE.
            if not isinstance(self.routed_experts.backend, MegaMoECuteDsl):
                raise RuntimeError(
                    "Kimi K3 explicitly requested MEGAMOE_CUTEDSL, but the "
                    f"MoE factory selected {type(self.routed_experts.backend).__name__}."
                )
        if self.routed_experts.layer_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "dynamic EPLB or replicated expert slots."
            )
        local_expert_ids = list(self.routed_experts.backend.initial_local_expert_ids)
        if local_expert_ids != list(
            range(local_expert_ids[0], local_expert_ids[0] + len(local_expert_ids))
        ):
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming currently requires a "
                "contiguous static expert partition."
            )
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank
        # TRTLLM_K3_EXPERT_PLACEMENT: routed-expert id (slot) s holds checkpoint expert
        # expert_of_slot[s]; ``load_weights`` loads the slots accordingly and permutes the router
        # into slot order. ``ckpt_expert_ids`` names the checkpoint expert of each local slot.
        self.expert_of_slot = expert_placement_for_layer(
            layer_idx, self.num_experts, routed_moe_model_config.mapping.moe_ep_size
        )
        self.ckpt_expert_ids = (
            self.local_expert_ids
            if self.expert_of_slot is None
            else tuple(self.expert_of_slot[slot] for slot in self.local_expert_ids)
        )
        if self.expert_replicas is not None:
            # Local slots: the rank's home slots, then its copies (each holds a home slot's
            # checkpoint expert).
            homes = self.expert_replicas.home_slots_of_local(self.replica_rank)
            self.ckpt_expert_ids = tuple(
                homes if self.expert_of_slot is None else (self.expert_of_slot[x] for x in homes)
            )
        routing_recorder = get_routing_recorder(
            rank=model_config.mapping.rank,
            num_layers=cfg.num_hidden_layers,
            num_experts=self.num_experts,
            first_layer=cfg.first_k_dense_replace,
        )
        if routing_recorder is not None and isinstance(
            self.routed_experts.routing_method, _KimiK3RoutingMethod
        ):
            self.routed_experts.routing_method.observer = functools.partial(
                routing_recorder.record, layer_idx
            )

        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        attention_dp = model_config.mapping.enable_attention_dp
        shared_model_config = copy.copy(model_config)
        shared_model_config.quant_config = QuantConfig()
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and must not reduce across ranks.
        #
        # Replication is a limitation, not a design goal. Sharding it under
        # attention DP would need the DP group's tokens all-gathered before the
        # GEMMs and scattered back after -- ConfigurableMoE owns its own
        # reduction, so column-shard partial sums have no existing collective to
        # ride on, and joining them to the MoE combine needs a partial-carry
        # hook in the wrapper (follow-up).
        #
        # The cost is concentrated on the decode path, which is weight-bandwidth
        # bound: at DEP16 every rank re-reads the full shared-expert
        # gate_up+down every step. Order of magnitude with the K3 dims (92 MoE
        # layers, hidden 7168, shared intermediate 6144, bf16) that is ~24 GB
        # per rank per step, against a few MB of all-gathered activations at
        # c256. The FP8 block-scale read of these projections (on by default,
        # see KIMI_K3_FP8_WEIGHT_READ_MOE_MLP) halves that read today without
        # needing the collective.
        #
        # Otherwise GatedMLP owns the row-parallel reduction, which runs on
        # the auxiliary stream used for the shared-expert branch.
        # Fused-communication routed backends already return the cross-rank
        # combined result and need no outer routed all-reduce.
        use_shared_tp = not attention_dp and model_config.mapping.tp_size > 1
        # CuteDSL is the validated scope, not a correctness dependency:
        # DeepGEMM rejects TP without attention DP.
        self._defer_shared_allreduce = (
            use_shared_tp
            and routed_moe_model_config.moe_backend == "MEGAMOE_CUTEDSL"
            and self.routed_experts.backend.scheduler_kind == MoESchedulerKind.FUSED_COMM
        )
        self._reduce_routed_output = (
            use_shared_tp
            and self.routed_experts.backend.scheduler_kind != MoESchedulerKind.FUSED_COMM
        )
        if self._reduce_routed_output and self.routed_experts.all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the "
                "ConfigurableMoE all-reduce even when reduce_results=False."
            )
        # TRTLLM_K3_MERGED_MOE_AR: the shared expert hands back its row-parallel partial and the
        # routed reduction sums it together with the latent partial (``_merged_reduce``). The
        # shared partial is reduced as rows of the latent width, so the hidden size must be a
        # multiple of it.
        self._merge_moe_allreduce = (
            _K3_MERGED_MOE_AR
            and self._reduce_routed_output
            and cfg.hidden_size % self.moe_hidden_size == 0
        )
        self.shared_experts = GatedMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=shared_intermediate,
            bias=False,
            activation=SituAndMul(
                beta=situ_beta,
                linear_beta=situ_linear_beta,
                use_fused_activation=True,
            ),
            dtype=dtype,
            config=shared_model_config,
            overridden_tp_size=1 if attention_dp else None,
            reduce_output=use_shared_tp,
            layer_idx=layer_idx,
            is_shared_expert=True,
        )
        if isinstance(self.shared_experts.down_proj, TrtllmLinear):
            # Its all-reduce follows it on the same stream, after the layer's o_proj
            # all-reduce (see _K3_SHARED_AR_NVLS_SLOT).
            self.shared_experts.down_proj.nvls_zero_copy = True
        shared_ar = getattr(self.shared_experts.down_proj, "all_reduce", None)
        shared_nvls = getattr(shared_ar, "nvls_allreduce", None)
        if shared_nvls is not None:
            if _K3_SHARED_AR_NVLS_SLOT != shared_nvls.slot:
                shared_nvls = NVLSAllReduce(
                    shared_nvls.mapping, slot=_K3_SHARED_AR_NVLS_SLOT, version=shared_nvls.version
                )
                shared_ar.nvls_allreduce = shared_nvls
            if not _K3_SHARED_AR_NVLS_V2:
                shared_nvls.version = 1
        # Set when the shared expert's TP-sharded down projection is converted
        # to the FP8 weight read, which does not reduce its output itself.
        self.shared_down_all_reduce = None
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        # TRTLLM_K3_SHARED_TAIL_PRIO: the shared expert's activation / down / all-reduce stream and
        # the event (after gate_up) it forks from.
        self.shared_tail_stream = _k3_shared_tail_stream()
        self.moe_shared_act_event = (
            torch.cuda.Event() if self.shared_tail_stream is not None else None
        )
        self.moe_routed_event = torch.cuda.Event() if self._defer_shared_allreduce else None
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        self.latent_down_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if routed_moe_model_config.moe_backend in ("CUTEDSL", "CUTEDSL_FC12")
            else None
        )
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        # TRTLLM_K3_SHARED_FIRST: shared GEMMs (and latent chain) done; routed FC1 done.
        self.moe_shared_gemm_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_fc1_event = torch.cuda.Event() if has_latent_stream else None
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # Stock fused RMSNorm (flashinfer kernel; the no-flashinfer
        # fallback is the same fp32-variance eager math as KimiK3RMSNorm).
        self.routed_expert_norm = RMSNorm(
            hidden_size=self.moe_hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        # TRTLLM_K3_MOE_L2_PREFETCH: the MoE layers of this model by index, and the event after the
        # routed FC2 that starts this layer's prefetch on the latent-down stream.
        self._moe_group = _K3_MOE_LAYERS.setdefault(id(cfg), weakref.WeakValueDictionary())
        self._moe_group[layer_idx] = self
        self._layers_key = id(cfg)
        self._l2_ranges_cache: Dict[str, List[Tuple[int, int]]] = {}
        self.moe_fc2_event = torch.cuda.Event() if has_latent_stream else None
        if _K3_LATENT_AR_SIDE_CLEAR and has_latent_stream:
            self.moe_lu_event = torch.cuda.Event()
            self.latent_ar_clear_stream = self.latent_down_stream
        if _K3_EXPERT_L2_PREFETCH and self.expert_replicas is None:
            self._expl2_state = _K3_EXPERT_L2_STATE.setdefault(id(cfg), _ExpertL2PrefetchState())
            self._expl2_event = torch.cuda.Event()
            self._expl2_attn_event = torch.cuda.Event()
        if _K3_LATENT_AR_OWN_WS and self._reduce_routed_output:
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                self._latent_ar_ws = _k3_latent_ar_workspace(
                    mnnvl.mapping, mnnvl.dtype, self.moe_hidden_size, _K3_LATENT_AR_OWN_WS_MAX_M
                )

    def _next_moe_layer(self) -> Optional["KimiK3MoERuntime"]:
        """The MoE runtime of the next MoE layer of this model, if any."""
        later = [idx for idx in self._moe_group.keys() if idx > self.layer_idx]
        return self._moe_group.get(min(later)) if later else None

    def l2_prefetch_ranges(self, what: str) -> List[Tuple[int, int]]:
        """``(address, bytes)`` of the weights a decode MoE block streams from HBM: ``"lu"`` the
        latent_up weight and scale; ``"front"`` the router, latent_down and shared-expert weights
        and scales (the front end of the block)."""
        ranges = self._l2_ranges_cache.get(what)
        if ranges is None:
            from ..cute_dsl_kernels.blackwell.k3_l2_prefetch import MAX_RANGES, tensor_range

            if what == "lu":
                modules = [self.routed_expert_up_proj]
                tensors = []
            else:
                modules = [
                    self.routed_expert_down_proj,
                    self.shared_experts.gate_up_proj,
                    self.shared_experts.down_proj,
                ]
                tensors = [self.gate.weight]
            for module in modules:
                tensors += [getattr(module, "weight", None), getattr(module, "weight_scale", None)]
            ranges = []
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda and tensor.numel():
                    try:
                        ranges.append(tensor_range(tensor))
                    except ValueError:
                        continue
            ranges = ranges[:MAX_RANGES]
            self._l2_ranges_cache[what] = ranges
        return ranges

    def _issue_l2_prefetch(self, num_tokens: int) -> None:
        """``TRTLLM_K3_MOE_L2_PREFETCH`` / ``TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA``: once the routed
        FC2 is done, prefetch into L2 on the latent-down stream. Only when a later MoE layer exists:
        its latent_down follows on this stream and is joined into the main stream, so the prefetch
        needs no join of its own (a join in front of a programmatic dependent would cost it its
        early launch)."""
        if (
            not (_K3_MOE_L2_PREFETCH or _K3_MOE_L2_PREFETCH_NEXT_MLA)
            or num_tokens > _K3_MOE_L2_PREFETCH_MAX_M
            or self.latent_down_stream is None
            or self.moe_fc2_event is None
            or not do_multi_stream()
            or torch.compiler.is_compiling()
        ):
            return
        nxt = self._next_moe_layer()
        if nxt is None or nxt.latent_down_stream is not self.latent_down_stream:
            return
        mla_ranges = []
        if _K3_MOE_L2_PREFETCH_NEXT_MLA:
            next_mla = _K3_MLA_LAYERS.get(getattr(self, "_layers_key", None), {}).get(
                self.layer_idx + 1
            )
            if next_mla is not None:
                mla_ranges = next_mla.q_b_l2_prefetch_ranges()
        if not _K3_MOE_L2_PREFETCH and not mla_ranges:
            return
        from ..cute_dsl_kernels.blackwell.k3_l2_prefetch import run_k3_l2_prefetch

        self.moe_fc2_event.record()
        with torch.cuda.stream(self.latent_down_stream):
            self.moe_fc2_event.wait()
            if "lu" in _K3_MOE_L2_PREFETCH:
                run_k3_l2_prefetch(self.l2_prefetch_ranges("lu"))
            if "front" in _K3_MOE_L2_PREFETCH:
                run_k3_l2_prefetch(nxt.l2_prefetch_ranges("front"))
            if mla_ranges:
                run_k3_l2_prefetch(mla_ranges)

    def _issue_latent_ar_clear(self, num_tokens: int) -> None:
        """``TRTLLM_K3_LATENT_AR_SIDE_CLEAR``: once latent_up is issued, re-arm the Lamport buffer of
        the latent all-reduce that just ran on the latent-down stream (``run_k3_lamport_clear``),
        which also marks the workspace state clean, so the next latent all-reduce on the workspace
        skips its own clear. Only when a later MoE layer exists: its latent_down follows on this
        stream and is joined into the main stream before its routed GEMMs (and with them that
        layer's all-reduce), so the clear needs no join of its own; the next call on the workspace
        writes into the cleared buffer only after its push of the call after it (three buffers)."""
        workspace = self._latent_ar_ws
        stream = self.latent_ar_clear_stream
        if (
            self.moe_lu_event is None
            or stream is None
            or workspace is None
            or num_tokens > workspace["max_tokens"]
            or not do_multi_stream()
            or torch.compiler.is_compiling()
        ):
            return
        nxt = self._next_moe_layer()
        if nxt is None or nxt.latent_down_stream is not stream:
            return
        from ..cute_dsl_kernels.blackwell.k3_latent_ar_norm import run_k3_lamport_clear

        self.moe_lu_event.record()
        with torch.cuda.stream(stream):
            self.moe_lu_event.wait()
            run_k3_lamport_clear(
                workspace["buffer"].data_ptr(), workspace["flags"], workspace["clear_ctr"]
            )

    def _latent_up_mxfp8(
        self, routed_latent: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """latent_up on the MXFP8 form ``(xq, xs)`` of ``routed_latent`` (``TRTLLM_K3_LU_MXFP8``),
        with the stages of ``TRTLLM_K3_LU_MX_STAGES`` where that tactic is valid for the shape."""
        up = self.routed_expert_up_proj
        if _K3_LU_MX_STAGES:
            from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8

            x = routed_latent.reshape(-1, routed_latent.shape[-1])
            m = x.shape[0]
            if k3_fp8w_lowm_mxfp8.can_run_mxfp8(xq, xs, up.weight, up.weight_scale):
                base = k3_fp8w_lowm_mxfp8.default_mx_tactic(m, up.out_features, up.in_features)
                a_stages, b_stages = (int(v) for v in _K3_LU_MX_STAGES.split(":"))
                tactic = k3_fp8w_lowm_mxfp8.K3LowMMxTactic(
                    mma_n=base.mma_n, split_k=base.split_k, a_stages=a_stages, b_stages=b_stages
                )
                try:
                    k3_fp8w_lowm_mxfp8.validate_tactic(up.out_features, up.in_features, tactic)
                except ValueError:
                    tactic = None
                if tactic is not None:
                    out = torch.empty(
                        (m, up.out_features), dtype=torch.bfloat16, device=routed_latent.device
                    )
                    k3_fp8w_lowm_mxfp8.run_k3_fp8w_lowm_gemm_mxfp8(
                        xq, xs, up.weight, up.weight_scale, out, tactic=tactic
                    )
                    return out.reshape(routed_latent.shape[:-1] + (up.out_features,))
        return up.forward_mxfp8(routed_latent, xq, xs)

    def _expert_l2_weight_ranges(self) -> List[Tuple[int, int, int]]:
        """``(base, stride, bytes)`` of every local expert's slice of this layer's routed gate_up
        and down weights and their block scales (``TRTLLM_K3_EXPERT_L2_PREFETCH``), or ``[]`` when
        the backend does not hold them as dense ``[local experts, ...]`` tensors."""
        backend = self.routed_experts.backend
        scales = getattr(backend, "quant_scales", None)
        tensors = [
            getattr(backend, "w3_w1_weight", None),
            getattr(scales, "fc1_weight_block", None),
            getattr(backend, "w2_weight", None),
            getattr(scales, "fc2_weight_block", None),
        ]
        if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
            return []
        key = tuple(t.data_ptr() for t in tensors)
        if self._expl2_ranges[0] != key:
            from ..cute_dsl_kernels.blackwell.k3_expert_l2_prefetch import expert_ranges

            try:
                ranges = expert_ranges(tensors, self.experts_per_rank)
            except ValueError as err:
                logger.warning_once(
                    f"TRTLLM_K3_EXPERT_L2_PREFETCH: layer {self.layer_idx} weights not prefetched "
                    f"({err})",
                    key="k3_expert_l2_prefetch_ranges",
                )
                ranges = []
            self._expl2_ranges = (key, ranges)
        return self._expl2_ranges[1]

    def _expert_l2_ready(self, num_tokens: int, device: torch.device) -> bool:
        """Whether ``TRTLLM_K3_EXPERT_L2_PREFETCH`` serves a forward of ``num_tokens`` tokens.

        The routing records (one ``int32 [RECORD_INTS]`` per MoE layer) and the side stream are
        created by the first eager forward that gets here, i.e. a CUDA graph warmup run: the records
        then live outside every graph memory pool, and all captured graphs share them."""
        state = self._expl2_state
        if (
            state is None
            or num_tokens > _K3_EXPERT_L2_PREFETCH_MAX_M
            or not do_multi_stream()
            or torch.compiler.is_compiling()
        ):
            return False
        from ..cute_dsl_kernels.blackwell import k3_expert_l2_prefetch as xpf

        if self.experts_per_rank > xpf.MAX_LOCAL:
            return False
        if self._expl2_record is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            for layer in list(self._moe_group.values()):
                if layer._expl2_record is None:
                    layer._expl2_record = xpf.new_record(device)
            if state.stream is None:
                state.stream = torch.cuda.Stream(device=device)
        return True

    def issue_expert_l2_prefetch(self, num_tokens: int) -> None:
        """``TRTLLM_K3_EXPERT_L2_PREFETCH``: called by this decoder layer's attention block once
        its weight-heavy projections are issued (KDA: after QKVG and [f_a | b]; MLA: after q_b and
        the q absorption). One ``k3_expert_l2_prefetch`` launch on the model's expert-prefetch
        stream prefetches into L2 the weights of the budget experts with the most tokens in this
        layer's routing record, i.e. its routing in the previous decode step, while the
        recurrence / MLA decode and the all-reduces leave HBM mostly idle.

        The launch waits for an event recorded on the main stream at the call point and is
        consumed only by the side stream, so no main-stream kernel gains an incoming edge. The
        record it reads was written by this layer's last :meth:`_record_expert_l2_routing` launch
        (same stream, the previous graph replay or eager step); on the first steps after start-up
        or after a batch-size switch it holds the warmup's or another batch size's routing of this
        layer, a valid but less accurate hint, and before any routing was recorded it is empty
        (nothing prefetched)."""
        if not self._expl2_recorded or not self._expert_l2_ready(
            num_tokens, self._expl2_record.device
        ):
            return
        ranges = self._expert_l2_weight_ranges()
        if not ranges:
            return
        from ..cute_dsl_kernels.blackwell.k3_expert_l2_prefetch import run_k3_expert_l2_prefetch

        state = self._expl2_state
        self._expl2_attn_event.record()
        with torch.cuda.stream(state.stream):
            self._expl2_attn_event.wait()
            run_k3_expert_l2_prefetch(
                num_local=self.experts_per_rank,
                record_in=self._expl2_record,
                ranges=ranges,
                budget=_K3_EXPERT_L2_PREFETCH,
                ctas=_K3_EXPERT_L2_PREFETCH_CTAS,
                chunk=_K3_EXPERT_L2_PREFETCH_CHUNK,
            )
        state.pending = True

    def _record_expert_l2_routing(self, num_tokens: int) -> None:
        """``TRTLLM_K3_EXPERT_L2_PREFETCH``: once this layer's routed FC2 is done, one
        single-CTA ``k3_expert_l2_prefetch`` launch on the model's expert-prefetch stream records
        this layer's routing (the routing-tail outputs of this forward, see ``_attach_route``)
        into its routing record, for the prefetch of this layer in the next decode step.

        The launch waits for an event recorded on the main stream right after the routed FC2 (a
        full dependency: FC2 and, through the PDL chain, the routing tail are complete) and is
        consumed only by the side stream, so no main-stream kernel (latent all-reduce, latent_up)
        gains an incoming edge. The routing-tail outputs it reads stay allocated until the side
        stream is done with them (``record_stream``: a few KB per layer, held until graph capture
        ends). The side stream is joined once per forward, by the last MoE layer where the main
        stream joins the shared-expert stream (:meth:`_join_expert_l2_prefetch`)."""
        outs = self._expl2_route
        self._expl2_route = None
        if outs is None or not self._expert_l2_ready(num_tokens, outs[0].device):
            return
        from ..cute_dsl_kernels.blackwell.k3_expert_l2_prefetch import run_k3_expert_l2_prefetch

        state = self._expl2_state
        sort_outputs = (outs[2], outs[3], outs[7])
        self._expl2_event.record()
        with torch.cuda.stream(state.stream):
            self._expl2_event.wait()
            for t in sort_outputs:
                t.record_stream(state.stream)
            run_k3_expert_l2_prefetch(
                num_local=self.experts_per_rank,
                record_out=self._expl2_record,
                sort_outputs=sort_outputs,
                ctas=1,
            )
        state.pending = True
        self._expl2_recorded = True

    def _join_expert_l2_prefetch(self) -> None:
        """``TRTLLM_K3_EXPERT_L2_PREFETCH``: in the last MoE layer, join the expert-prefetch stream
        into the current (main) stream. Called right where the main stream joins the shared-expert
        stream, so the kernel after it already waits for another stream; the last launch on the
        prefetch stream (this layer's routing record) started after this layer's routed FC2 and is
        short."""
        state = self._expl2_state
        if state is None or not state.pending or self._next_moe_layer() is not None:
            return
        torch.cuda.current_stream().wait_stream(state.stream)
        state.pending = False

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits; ``TRTLLM_K3_ROUTER_EXACT`` serves ``M <= 16`` BF16 tokens with
        ``k3_router_exact`` (bit-identical to the ``dsv3_router_gemm_op`` it replaces) in the
        layers it selects."""
        if self._router_exact and not torch.compiler.is_compiling():
            from ..cute_dsl_kernels.blackwell import k3_router_exact

            x = hidden_states.reshape(-1, self.gate.gating_dim)
            if k3_router_exact.can_run(x, self.gate.weight):
                return torch.ops.trtllm.k3_router_logits(x, self.gate.weight)
        return self.gate.compute_logits(hidden_states)

    def _lowm_fp4_latent_input(self, hidden_states: torch.Tensor) -> Optional[Fp4QuantizedTensor]:
        """``TRTLLM_K3_LD_FP4``: the routed experts' NVFP4 input straight from latent_down's
        low-M GEMM epilogue, or ``None``. Byte-identical to the low-M GEMM followed by the
        experts' ``fp4_quantize`` (CuteDSL NVFP4, no dispatch)."""
        if not _K3_LD_FP4 or not _K3_FP8W_LOWM_GEMM or torch.compiler.is_compiling():
            return None
        down = self.routed_expert_down_proj
        if not isinstance(down, _Fp8BlockScaleWeightReadLinear) or _rubin_mxfp8_is_available():
            return None
        moe = self.routed_experts
        backend = moe.backend
        global_sf = getattr(backend, "fc31_input_scale", None)
        if (
            moe.comm is not None
            or moe.apply_router_weight_on_input
            or not getattr(backend, "has_nvfp4", False)
            or getattr(backend, "scaling_vector_size", None) != 16
            or not hasattr(backend, "quantize_input_async")
            or global_sf is None
        ):
            return None
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if (
            x.dtype != torch.bfloat16
            or x.shape[0] > _K3_LD_FP4_MAX_M
            or (down.mid_weight_sf is not None and down.mid_gemm_applies(x))
        ):
            return None
        from ..cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

        if not k3_fp8w_lowm_gemm.can_run_nvfp4(x, down.weight, down.weight_scale, global_sf):
            return None
        codes, scales = torch.ops.trtllm.k3_fp8w_lowm_gemm_nvfp4(
            x, down.weight, down.weight_scale, global_sf
        )
        return Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)

    def _shared_first_applies(self, num_tokens: int) -> bool:
        """Whether ``TRTLLM_K3_SHARED_FIRST`` orders this forward's shared expert before FC1."""
        return (
            _K3_SHARED_FIRST != "0"
            and _K3_SHARED_FIRST_MIN_M <= num_tokens <= _K3_SHARED_FIRST_MAX_M
            and self.shared_expert_stream is not None
            and self.moe_shared_gemm_event is not None
            and not self._defer_shared_allreduce
            and not self._merge_moe_allreduce
            and not getattr(self.shared_experts.down_proj, "use_fused_gemm_allreduce", False)
            and hasattr(self.routed_experts.backend, "post_fc1_event")
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        )

    def _shared_gemms_first(
        self, hidden_states: torch.Tensor, fork_event: torch.cuda.Event
    ) -> torch.Tensor:
        """``TRTLLM_K3_SHARED_FIRST``: issue the shared expert's GEMMs (without its all-reduce) on
        the shared stream after ``fork_event``; ``moe_shared_gemm_event`` then marks them and the
        latent chain done, which is what the routed FC1 waits for. Returns the partial."""
        with torch.cuda.stream(self.shared_expert_stream):
            fork_event.wait()
            partial = self.shared_experts(
                hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
            self.moe_latent_ready_event.wait()
            self.moe_shared_gemm_event.record()
        return partial

    def _shared_needs_all_reduce(self) -> bool:
        """Whether the shared expert's output is a row-parallel partial to all-reduce."""
        return self.shared_down_all_reduce is not None or bool(
            getattr(self.shared_experts.down_proj, "reduce_output", False)
        )

    def _shared_all_reduce(self) -> nn.Module:
        """The shared expert's row-parallel reduction (kept aside if its down projection is FP8)."""
        if self.shared_down_all_reduce is not None:
            return self.shared_down_all_reduce
        return self.shared_experts.down_proj.all_reduce

    def _merged_reduce_buffers(
        self, hidden_states: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(buffer, zero_residual)`` of the merged MoE all-reduce, or ``None`` to keep two.

        ``buffer`` is flat: the routed latent partial ``[M, moe_hidden_size]`` followed by the
        shared-expert partial ``[M, hidden_size]``, both contiguous.
        """
        num_tokens = hidden_states.shape[0]
        if (
            not self._merge_moe_allreduce
            or num_tokens == 0
            or 0 < _K3_MERGED_MOE_AR_MAX_TOKENS < num_tokens
            or hidden_states.dtype != torch.bfloat16
            or torch.compiler.is_compiling()
        ):
            return None
        numel = num_tokens * (self.moe_hidden_size + self.hidden_size)
        zero_residual = _k3_zero_residual(numel, hidden_states.device)
        if zero_residual is None:
            return None
        return torch.empty_like(zero_residual), zero_residual

    def _merged_reduce(
        self, buffer: torch.Tensor, zero_residual: torch.Tensor, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``[routed latent partial | shared partial]`` in one collective.

        The buffer is viewed as rows of the latent width (a shared row spans
        ``hidden_size / moe_hidden_size`` of them) and reduced with the residual-RMSNorm fusion
        against a zero residual: the residual output is the plain sum of every row, and the norm
        output of the first ``num_tokens`` rows is ``routed_expert_norm`` of the reduced latent.
        Returns ``(normed latent [M, moe_hidden_size], reduced shared [M, hidden_size])``.
        """
        latent = self.moe_hidden_size
        rows = buffer.numel() // latent
        normed, reduced = self.routed_experts.all_reduce(
            buffer.view(rows, latent),
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=zero_residual.view(rows, latent),
                norm_weight=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
            ),
        )
        return normed[:num_tokens], reduced[num_tokens:].view(num_tokens, self.hidden_size)

    def _fused_route(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Route with the fused CuTe router when enabled and applicable.

        Hands the top-k to the routed experts' routing method and, in ``sort``
        mode, attaches this rank's ``moe_sort`` outputs to the top-k ids (read by
        the CuteDSL backend). Returns the FP32 router logits, or ``None`` when
        the caller must route the regular way.
        """
        if (
            _K3_FUSED_ROUTER == "0"
            or torch.compiler.is_compiling()
            or self.expert_replicas is not None
        ):
            return None
        routing = self.routed_experts.routing_method
        backend = self.routed_experts.backend
        if not isinstance(routing, _KimiK3RoutingMethod) or self.routed_experts.use_dp:
            return None
        from ..cute_dsl_kernels.blackwell import k3_fused_router

        bias = self.gate.e_score_correction_bias
        if not k3_fused_router.can_use_k3_fused_router(hidden_states, self.gate.weight, bias):
            return None
        if torch.cuda.is_current_stream_capturing() and not k3_fused_router.workspace_ready(
            hidden_states.device
        ):
            return None
        scale = float(self.gate.routed_scaling_factor)
        slot_start = getattr(backend, "slot_start", None)
        num_local = getattr(backend, "expert_size_per_partition", None)
        if (
            _K3_FUSED_ROUTER == "sort"
            and slot_start is not None
            and num_local is not None
            and getattr(backend, "num_slots", None) == self.gate.num_experts
        ):
            (topk_ids, topk_weights, *sort_outputs, logits) = torch.ops.trtllm.k3_fused_router_sort(
                hidden_states,
                self.gate.weight,
                bias,
                scale,
                slot_start,
                num_local,
                128,
                True,
            )
            topk_ids._k3_presorted = ((slot_start, num_local, 128), tuple(sort_outputs))
        else:
            topk_ids, topk_weights, logits = torch.ops.trtllm.k3_fused_router_topk(
                hidden_states, self.gate.weight, bias, scale, True
            )
        routing.precomputed = (topk_ids, topk_weights)
        return logits

    def _route_tail_target(self, num_tokens: int) -> Optional[Tuple[int, int]]:
        """``(local_expert_offset, local_num_experts)`` if the CuTe routing tail serves this
        forward (``TRTLLM_K3_ROUTE_TAIL``), else ``None``."""
        if _K3_ROUTE_TAIL == "0" or torch.compiler.is_compiling():
            return None
        routing = self.routed_experts.routing_method
        backend = self.routed_experts.backend
        if not isinstance(routing, _KimiK3RoutingMethod) or self.routed_experts.use_dp:
            return None
        slot_start = getattr(backend, "slot_start", None)
        num_local = getattr(backend, "expert_size_per_partition", None)
        num_slots = (
            self.gate.num_experts
            if self.expert_replicas is None
            else self.expert_replicas.num_slots
        )
        if (
            slot_start is None
            or num_local is None
            or getattr(backend, "num_slots", None) != num_slots
        ):
            return None
        from ..cute_dsl_kernels.blackwell import k3_route_tail

        if not 1 <= num_tokens <= min(_K3_ROUTE_TAIL_MAX_M, k3_route_tail.WIDE_MAX_M):
            return None
        if self.expert_replicas is not None:
            # Only the cluster tail chooses copies; above it the home copies serve (slot_map).
            if num_tokens > k3_route_tail.MAX_M or not self._ensure_replica_tables():
                return None
            return slot_start, num_local
        if num_tokens > k3_route_tail.MAX_M:
            if not k3_route_tail.wide_sort_fits(num_tokens, num_local):
                return None
            # The multi-CTA kernel's workspace must exist before graph capture.
            device = self.gate.e_score_correction_bias.device
            if not torch.cuda.is_current_stream_capturing():
                k3_route_tail.wide_workspace(device)
            elif not k3_route_tail.wide_workspace_ready(device):
                return None
        return slot_start, num_local

    def _can_gemv_route(self, hidden_states: torch.Tensor) -> bool:
        """Whether ``TRTLLM_K3_ROUTE_TAIL=gemv`` computes this forward's router logits."""
        from ..cute_dsl_kernels.blackwell import k3_route_tail

        backend = self.routed_experts.backend
        return (
            _K3_ROUTE_TAIL == "gemv"
            and self.expert_replicas is None
            and hidden_states.shape[0] <= k3_route_tail.GEMV_ROUTE_MAX_M
            and getattr(backend, "has_nvfp4", False)
            and getattr(backend, "fc31_input_scale", None) is not None
            and k3_route_tail.can_use_k3_route(hidden_states, self.gate.weight)
        )

    def _attach_route(
        self, outs: Tuple[torch.Tensor, ...], target: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(topk_ids, topk_weights)`` of a routing-tail result; the ids carry the
        ``moe_sort`` outputs for the CuteDSL backend."""
        topk_ids = outs[0]
        topk_ids._k3_presorted = ((target[0], target[1], 128), tuple(outs[2:8]))
        if self._expl2_state is not None:
            # Recorded by _record_expert_l2_routing after the routed FC2.
            self._expl2_route = outs
        return topk_ids, outs[1]

    def _tail_route(
        self, logits: torch.Tensor, target: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expert_replicas is not None:
            outs = torch.ops.trtllm.k3_route_tail_rep(
                logits,
                self.gate.e_score_correction_bias,
                float(self.gate.routed_scaling_factor),
                target[0],
                target[1],
                self.replica_rank * self.expert_replicas.num_home,
                self._replica_table,
                len(self.expert_replicas.entries),
                128,
            )
            return self._attach_route(outs, target)
        outs = torch.ops.trtllm.k3_route_tail(
            logits,
            self.gate.e_score_correction_bias,
            float(self.gate.routed_scaling_factor),
            target[0],
            target[1],
            128,
        )
        return self._attach_route(outs, target)

    @property
    def replica_rank(self) -> int:
        """EP rank of this layer's routed experts (``TRTLLM_K3_EXPERT_REPLICAS``)."""
        return self.expert_lo // self.expert_replicas.num_local

    def _ensure_replica_tables(self) -> bool:
        """Create the copy-choice table of the routing tail and the home-slot map of the other
        routing paths (``TRTLLM_K3_EXPERT_REPLICAS``) on the router's device; False while a CUDA
        graph is being captured before they exist."""
        if self._replica_table is not None:
            return True
        device = self.gate.e_score_correction_bias.device
        if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return False
        from ..cute_dsl_kernels.blackwell import k3_route_tail

        replicas = self.expert_replicas
        table = k3_route_tail.replica_table(
            replicas.entries, self.replica_rank, replicas.num_home
        )
        self._replica_table = torch.tensor(table, dtype=torch.int32, device=device)
        self.routed_experts.routing_method.slot_map = torch.tensor(
            replicas.routed_slot_of_home(), dtype=torch.int32, device=device
        )
        return True

    def _can_umma_route(self, hidden_states: torch.Tensor) -> bool:
        """Whether ``TRTLLM_K3_ROUTER_GEMM=umma`` computes this forward's router logits."""
        if _K3_ROUTER_GEMM != "umma" or not 1 <= hidden_states.shape[0] <= _K3_ROUTER_GEMM_MAX_M:
            return False
        from ..cute_dsl_kernels.blackwell import k3_router_umma

        return k3_router_umma.can_run(hidden_states, self.gate.weight)

    def _umma_router_partials(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 split-K partial router logits ``[S, M, 896]`` of the UMMA router GEMM."""
        from ..cute_dsl_kernels.blackwell import k3_router_umma

        partials = torch.empty(
            (k3_router_umma.DEFAULT_SPLITS, hidden_states.shape[0], self.gate.num_experts),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        k3_router_umma.run_k3_router_umma(hidden_states, self.gate.weight, partials)
        return partials

    def _gemv_router_partials(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Split-K partial router logits ``[S, M, 896]`` of ``k3_route``'s router GEMM (the
        ``mma.sync`` GEMV or, with ``TRTLLM_K3_ROUTER_GEMV=umma``, the UMMA GEMM)."""
        from ..cute_dsl_kernels.blackwell import k3_route_tail

        if k3_route_tail.ROUTER_GEMV == "umma":
            return self._umma_router_partials(hidden_states)
        partials = torch.empty(
            (k3_route_tail.DEFAULT_GEMV_SPLITS, hidden_states.shape[0], self.gate.num_experts),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        k3_route_tail.run_k3_router_gemv(hidden_states, self.gate.weight, partials)
        return partials

    def _tail_route_partials(
        self, partials: torch.Tensor, logits: torch.Tensor, target: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Routing tail on a router GEMM's split-K partials (``logits`` is their first split, the
        placeholder the routed experts were handed)."""
        return self._tail_route(partials, target)

    def _gemv_route(self, hidden_states: torch.Tensor, target: Tuple[int, int]) -> torch.Tensor:
        """Route with the router GEMV + tail; hands the result to the routing method and
        returns the FP32 logits."""
        outs = torch.ops.trtllm.k3_route(
            hidden_states,
            self.gate.weight,
            self.gate.e_score_correction_bias,
            float(self.gate.routed_scaling_factor),
            target[0],
            target[1],
            128,
        )
        self.routed_experts.routing_method.precomputed = self._attach_route(outs, target)
        return outs[8]

    def _quantize_routed_input(self, x: torch.Tensor) -> Fp4QuantizedTensor:
        """NVFP4-quantize the routed input like the CuteDSL backend's ``quantize_input``."""
        backend = self.routed_experts.backend
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
            x, backend.fc31_input_scale, backend.scaling_vector_size, False, False
        )
        return Fp4QuantizedTensor(x_fp4, x_sf, is_sf_swizzled=False)

    @staticmethod
    def _fused_projection_applies(mode: str, x: torch.Tensor, projection: nn.Module) -> bool:
        """Whether a fused latent projection (mode "1"/"2") replaces ``projection(x)``."""
        if mode == "0" or torch.compiler.is_compiling():
            return False
        if not isinstance(projection, _Fp8BlockScaleWeightReadLinear):
            return False
        if _rubin_mxfp8_is_available() or x.dtype != torch.bfloat16:
            return False
        if not _K3_LD_FUSED_MIN_M <= x.shape[0] <= _K3_LD_FUSED_MAX_M:
            return False
        if (
            (mode == "1" or x.shape[0] <= _K3_LOWM_OVER_FUSED_MAX_M)
            and _K3_FP8W_LOWM_GEMM
            and _k3_lowm_gemm_can_run(x, projection.weight, projection.weight_scale)
        ):
            return False
        return True

    def _fused_latent_input(self, hidden_states: torch.Tensor) -> Optional[Fp4QuantizedTensor]:
        """NVFP4 routed-expert input straight from the hidden states, or ``None``.

        Runs the latent_down projection with the experts' input quantization fused in
        (``trtllm::k3_latent_down_fp4``) when enabled and the routed experts would quantize
        the projection with ``fp4_quantize`` themselves (CuteDSL NVFP4, no dispatch).
        """
        down = self.routed_expert_down_proj
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not self._fused_projection_applies(_K3_LD_FUSED, x, down):
            return None
        moe = self.routed_experts
        backend = moe.backend
        if (
            moe.comm is not None
            or moe.apply_router_weight_on_input
            or not getattr(backend, "has_nvfp4", False)
            or getattr(backend, "scaling_vector_size", None) != 16
            or not hasattr(backend, "quantize_input_async")
        ):
            return None
        max_tokens = getattr(moe, "moe_max_num_tokens", None)
        if max_tokens is not None and x.shape[0] > max_tokens:
            return None
        from ..cute_dsl_kernels.blackwell import k3_latent_fp8_gemm

        global_sf = getattr(backend, "fc31_input_scale", None)
        if not k3_latent_fp8_gemm.can_run(x, down.weight, down.weight_scale, global_sf):
            return None
        codes, scales = torch.ops.trtllm.k3_latent_down_fp4(
            x, down.weight, down.weight_scale, global_sf
        )
        return Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)

    def _fused_latent_up_residual(
        self, latent: torch.Tensor, shared_out: torch.Tensor, residual: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """``residual + (latent_up(latent) + shared_out)`` in one fused chain, or ``None``."""
        up = self.routed_expert_up_proj
        x = latent.reshape(-1, latent.shape[-1])
        if (
            isinstance(up, _Fp8BlockScaleWeightReadLinear)
            and up.mid_weight_sf is not None
            and x.dtype == torch.bfloat16
            and up.mid_gemm_applies(x)
        ):
            from ..cute_dsl_kernels.blackwell import k3_fp8_mid_gemm

            m = x.shape[0]
            shared_2d = shared_out.reshape(m, -1)
            residual_2d = residual.reshape(m, -1)
            if k3_fp8_mid_gemm.can_run(x, up.weight, up.mid_weight_sf, shared_2d, residual_2d):
                out = torch.ops.trtllm.k3_fp8_mid_residual(
                    x, up.weight, up.mid_weight_sf, shared_2d, residual_2d
                )
                return out.view(residual.shape)
        if not self._fused_projection_applies(_K3_LU_FUSED, x, up):
            return None
        from ..cute_dsl_kernels.blackwell import k3_latent_fp8_gemm

        m = x.shape[0]
        shared_2d = shared_out.reshape(m, -1)
        residual_2d = residual.reshape(m, -1)
        if not k3_latent_fp8_gemm.can_run_residual(
            x, up.weight, up.weight_scale, shared_2d, residual_2d
        ):
            return None
        out = torch.ops.trtllm.k3_latent_up_residual(
            x, up.weight, up.weight_scale, shared_2d, residual_2d
        )
        return out.view(residual.shape)

    def _latent_up_reads_shared(self, latent: torch.Tensor) -> bool:
        """Whether a latent_up variant that adds the shared-expert output itself (the quantized
        all-reduce + mid-M GEMM, the mid-M GEMM, the fused latent_up + residual) may serve the
        ``[M, moe_hidden_size]`` BF16 routed latent ``latent`` (or a tensor shaped like it)."""
        up = self.routed_expert_up_proj
        if isinstance(up, _Fp8BlockScaleWeightReadLinear) and up.mid_weight_sf is not None:
            m = latent.shape[0]
            if _K3_LU_AR_QUANT and _K3_FP8_MID_MIN_M <= m <= _K3_FP8_MID_MAX_M:
                from ..cute_dsl_kernels.blackwell import k3_fp8_mid_gemm

                if (
                    k3_fp8_mid_gemm.tile_for(m, up.out_features, up.in_features, packed=True)
                    is not None
                ):
                    return True
            if up.mid_gemm_applies(latent):
                return True
        return self._fused_projection_applies(_K3_LU_FUSED, latent, up)

    def _own_workspace_latent_tail(
        self,
        routed_out: torch.Tensor,
        shared_out: torch.Tensor,
        residual: Optional[torch.Tensor],
        defer_residual: bool,
    ):
        """``TRTLLM_K3_LATENT_AR_OWN_WS``: the routed latent all-reduce + RMSNorm on its own MNNVL
        workspace and latent_up, issued on the main stream before it joins the shared-expert
        stream, whose output only the residual update reads. Returns what ``forward`` returns."""
        workspace = self._latent_ar_ws
        gamma = self.routed_expert_norm.weight
        eps = float(self.routed_expert_norm.variance_epsilon)
        latent = routed_out.reshape(-1, routed_out.shape[-1]).contiguous()
        up = self.routed_expert_up_proj
        routed_latent = None
        latent_mxfp8 = None
        if _K3_LATENT_AR_CUTE != "0":
            from ..cute_dsl_kernels.blackwell import k3_latent_ar_norm
            from ..distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

            world_size = workspace["world_size"]
            one_shot = (
                latent.numel() * world_size * latent.element_size()
                <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
            )
            row = _K3_LATENT_AR_ROW and k3_latent_ar_norm.can_run_row(
                latent, gamma, world_size, workspace["buffer_bytes"]
            )
            if one_shot and (row or k3_latent_ar_norm.can_run(latent, gamma, world_size)):
                routed_latent = torch.empty_like(latent)
                m, k = latent.shape
                if (
                    _K3_LU_MXFP8
                    and hasattr(up, "forward_mxfp8")
                    and up.takes_mxfp8_input(m)
                    and k % 32 == 0
                ):
                    latent_mxfp8 = (
                        torch.empty((m, k), dtype=torch.float8_e4m3fn, device=latent.device),
                        torch.empty((m, k // 32), dtype=torch.uint8, device=latent.device),
                    )
                launch = (
                    k3_latent_ar_norm.run_k3_latent_ar_norm_row
                    if row
                    else k3_latent_ar_norm.run_k3_latent_ar_norm
                )
                launch(
                    latent,
                    gamma,
                    eps,
                    routed_latent,
                    uc_addr=workspace["buffer"].data_ptr(),
                    mc_addr=workspace["mc"].data_ptr(),
                    flags=workspace["flags"],
                    rank=workspace["rank"],
                    world_size=world_size,
                    early_trigger=_K3_LATENT_AR_CUTE == "1",
                    mxfp8_out=latent_mxfp8,
                )
        if routed_latent is None:
            (routed_latent,) = torch.ops.trtllm.mnnvl_fusion_allreduce(
                latent,
                gamma,
                None,
                eps,
                workspace["buffer"],
                workspace["flags"],
                True,
                None,
                int(AllReduceFusionOp.RMS_NORM),
            )
        routed_latent = routed_latent.view(routed_out.shape)
        if latent_mxfp8 is not None:
            routed_out = self._latent_up_mxfp8(routed_latent, *latent_mxfp8)
        else:
            routed_out = self._routed_projection(routed_latent, up)
        self._issue_latent_ar_clear(latent.shape[0])
        self._join_expert_l2_prefetch()
        self.moe_shared_event.wait()
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            from ..modules.kimi_k3_moe_residual import _fused_moe_residual_add

            return _fused_moe_residual_add(routed_out, shared_out, residual)
        return routed_out + shared_out

    def _latent_up_from_quantized_all_reduce(
        self,
        routed_out: torch.Tensor,
        shared_out: torch.Tensor,
        residual: Optional[torch.Tensor],
        moe_all_reduce: Optional[nn.Module],
    ) -> Optional[torch.Tensor]:
        """``TRTLLM_K3_LU_AR_QUANT``: routed all-reduce + RMSNorm + FP8 block quantization in one
        MNNVL launch, then the mid-M latent_up GEMM with the residual update; ``None`` if it does
        not apply (the caller then runs the BF16 all-reduce + RMSNorm path)."""
        if (
            not _K3_LU_AR_QUANT
            or residual is None
            or moe_all_reduce is None
            or getattr(moe_all_reduce, "mnnvl_allreduce", None) is None
            or moe_all_reduce.mapping.tp_size == 1
            or torch.compiler.is_compiling()
        ):
            return None
        up = self.routed_expert_up_proj
        if not isinstance(up, _Fp8BlockScaleWeightReadLinear) or up.mid_weight_sf is None:
            return None
        x = routed_out.reshape(-1, routed_out.shape[-1])
        m = x.shape[0]
        if x.dtype != torch.bfloat16 or not _K3_FP8_MID_MIN_M <= m <= _K3_FP8_MID_MAX_M:
            return None
        from ..cute_dsl_kernels.blackwell import k3_fp8_mid_gemm

        if k3_fp8_mid_gemm.tile_for(m, up.out_features, up.in_features, packed=True) is None:
            return None
        shared_2d = shared_out.reshape(m, -1)
        residual_2d = residual.reshape(m, -1)
        out = moe_all_reduce(
            routed_out,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RMS_NORM_QUANT_FP8_BLOCKSCALE,
                norm_weight=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
            ),
        )
        if not isinstance(out, tuple) or len(out) != 2:
            raise RuntimeError(
                "Kimi K3: the MNNVL all-reduce did not return FP8 block-scale output"
            )
        latent_fp8, latent_scale = out
        latent_fp8 = latent_fp8.reshape(m, -1)
        if not k3_fp8_mid_gemm.can_run_packed(
            latent_fp8, latent_scale, up.weight, up.mid_weight_sf, shared_2d, residual_2d
        ):
            raise RuntimeError("Kimi K3: unexpected FP8 block-scale all-reduce output layout")
        fused = torch.ops.trtllm.k3_fp8_mid_residual_packed(
            latent_fp8, latent_scale, up.weight, up.mid_weight_sf, shared_2d, residual_2d
        )
        return fused.view(residual.shape)

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    # The TP CuteDSL gate scopes validation; fusion does not depend on it.
    def _can_fuse_router_down(self, hidden_states: torch.Tensor) -> bool:
        if (
            self.expert_replicas is not None
            or torch.compiler.is_compiling()
            or _K3_DISABLE_MIN_LATENCY_LATENT_PROJ
            or not hidden_states.is_cuda
            or self.routed_experts.use_dp
            or self.routed_experts.parallel_size <= 1
            or self.routed_experts.moe_backend != "MEGAMOE_CUTEDSL"
            or self.routed_experts.backend.scheduler_kind != MoESchedulerKind.FUSED_COMM
            or get_sm_version() != 107
        ):
            return False
        down = self.routed_expert_down_proj
        if not isinstance(down, nn.Linear) or down.bias is not None:
            return False
        if (
            hidden_states.ndim != 2
            or not 1 <= hidden_states.shape[0] <= 16
            or hidden_states.shape[1] != 7168
            or self.gate.weight.shape != (896, 7168)
            or down.weight.shape != (3584, 7168)
        ):
            return False
        # Inspect the loaded weights: FP8 conversion can replace the down module.
        return all(
            tensor.dtype == torch.bfloat16
            and tensor.is_cuda
            and tensor.device == hidden_states.device
            and tensor.stride() == (7168, 1)
            and tensor.data_ptr() % 16 == 0
            for tensor in (hidden_states, self.gate.weight, down.weight)
        )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        2. Default: EP-only (``moe_tp=1, moe_ep=tp_size``), the historical
           K3 layout.
        """
        tp_size = mapping.tp_size
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``, taken from the checkpoint.

        ``nvidia/Kimi-K3-NVFP4`` declares the routed experts per layer as
        ``NVFP4`` with ``group_size=16``; the original ``moonshotai/Kimi-K3``
        declares nothing per layer and keeps the historical
        ``W4A8_MXFP4_MXFP8`` default. Reading the checkpoint instead of
        hardcoding is what lets one code path serve both.
        """
        per_layer = getattr(model_config, "quant_config_dict", None)
        if per_layer:
            for prefix in _K3_ROUTED_EXPERT_KEY_PREFIXES:
                for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES:
                    cfg = per_layer.get(f"{prefix}layers.{layer_idx}.{suffix}")
                    if cfg is not None and cfg.quant_algo is not None:
                        # Logged once per layer: the routed-expert format decides
                        # which MoE backends can serve this checkpoint at all.
                        logger.debug(
                            "Kimi K3 layer %d routed experts: %s (group_size=%s) "
                            "from the checkpoint",
                            layer_idx,
                            cfg.quant_algo,
                            cfg.group_size,
                        )
                        return cfg
        logger.debug(
            "Kimi K3 layer %d routed experts: no per-layer quant config in the "
            "checkpoint, defaulting to %s",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO)

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
        # Every backend listed here must have a matching SiTU translation in
        # ``create_moe_backend`` -- the model passes SiTU explicitly in the
        # canonical form; no backend infers it from the checkpoint config, and
        # no backend dialect appears in this file.
        #
        # CUTLASS is admitted together with SM107 in ``fused_moe_cutlass.py``'s
        # NVFP4 ``sm_constraint`` -- ``resolve_moe_cls`` runs ``can_implement``
        # even for an explicitly requested backend, so listing it here without
        # that constraint would only trade this message for a less obvious
        # "no MoE implementation can serve this layer".
        #
        # W4A8_MXFP4_MXFP8 still excludes SM107, so MXFP4 + CUTLASS remains
        # unavailable on Rubin. That path has an L1 smoke on the k3-latest tree
        # (job 450060) but no GSM8K, so it is left closed until measured.
        supported_backends = {
            "TRTLLM",
            "MEGAMOE_DEEPGEMM",
            "MEGAMOE_CUTEDSL",
            "CUTEDSL",
            "CUTLASS",
            "CUTEDSL_FC12",
        }
        if model_config.moe_backend not in supported_backends:
            raise ValueError(
                "Kimi K3 SiTU routed experts support "
                f"{', '.join(sorted(supported_backends))}; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        if getattr(mapping, "_dwdp_size", 0) > 1:
            # DWDP does not hook the load path at all: setup_dwdp() scans the
            # model *after* every weight is loaded, reads w3_w1_weight /
            # w2_weight and their block scales straight off the backend
            # (dwdp/setup.py::collect_moe_params), copies them into fabric
            # memory, frees the originals, and rebinds param.data to a
            # composite VA view.  K3's packed-checkpoint streaming produces
            # exactly those tensors, so the two are compatible -- but only on
            # the backends that declare supports_dwdp, which is the CuteDSL
            # family (fused_moe_cute_dsl.py) and NVFP4 only.
            #
            # MEGAMOE_CUTEDSL is excluded deliberately rather than by
            # oversight: its process_weights_after_loading packs into the mega
            # buffers and shrinks the raw source params back to placeholders
            # (see maybe_finalize_layer), which has never been checked against
            # DWDP's copy-then-free of those same originals.
            if model_config.moe_backend not in ("CUTEDSL", "CUTEDSL_FC12"):
                raise NotImplementedError(
                    "Kimi K3 with DWDP requires moe_config.backend CUTEDSL or "
                    f"CUTEDSL_FC12; got {model_config.moe_backend!r}."
                )

        moe_tp, moe_ep = KimiK3MoERuntime._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        if moe_tp > 1 and mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 MoE tensor parallelism requires "
                "enable_attention_dp=false (the attention-DP dispatch/combine "
                "path is validated for EP-only splits)."
            )
        logger.info_once(
            f"Kimi K3 routed MoE parallelism: moe_tp={moe_tp}, "
            f"moe_ep={moe_ep} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = moe_tp
        mapping_dict["moe_ep_size"] = moe_ep
        routed_mapping = Mapping.from_dict(mapping_dict)

        routed_model_config = copy.copy(model_config)
        routed_model_config._frozen = False
        routed_model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        routed_model_config.mapping = routed_mapping
        routed_model_config.moe_backend = model_config.moe_backend
        # MegaMoE uses this value as global DP SymmBuffer capacity, then
        # divides it by EP size for the per-rank allocation. Other backends
        # keep the user-configured value as their MoE chunking bound.
        # Preserve an explicitly larger capacity.
        if routed_model_config.moe_backend == "MEGAMOE_DEEPGEMM":
            default_moe_max_num_tokens = routed_model_config.max_num_tokens * routed_mapping.dp_size
            configured_moe_max_num_tokens = int(routed_model_config.moe_max_num_tokens or 0)
            if configured_moe_max_num_tokens < default_moe_max_num_tokens:
                logger.info_once(
                    "Kimi K3 MegaMoE raises moe_max_num_tokens from "
                    f"{configured_moe_max_num_tokens} to {default_moe_max_num_tokens} "
                    "because the global DP SymmBuffer requires capacity for "
                    "max_num_tokens * dp_size.",
                    key=(
                        "kimi_k3_megamoe_capacity_override_"
                        f"{configured_moe_max_num_tokens}_{default_moe_max_num_tokens}"
                    ),
                )
            routed_model_config.moe_max_num_tokens = max(
                configured_moe_max_num_tokens,
                default_moe_max_num_tokens,
            )
        routed_model_config._frozen = True
        return routed_model_config

    def _shared_tail_applies(self, num_tokens: int) -> bool:
        """Whether ``TRTLLM_K3_SHARED_TAIL_PRIO`` splits this forward's shared expert (the
        unfused gate_up -> activation -> down GatedMLP path only)."""
        mlp = self.shared_experts
        return (
            self.shared_tail_stream is not None
            and _K3_SHARED_TAIL_MIN_M <= num_tokens <= _K3_SHARED_TAIL_MAX_M
            and isinstance(mlp, GatedMLP)
            and not mlp._can_fuse_gate_up_swiglu_fp4out()
            and not mlp._can_fuse_gate_up_swiglu()
            and not mlp._can_fuse_swiglu_fp8_quant()
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        )

    def _issue_router(
        self,
        hidden_states: torch.Tensor,
        gemv_route: bool,
        route_target: Optional[Tuple[int, int]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Issue the router GEMM on the current stream; returns ``(router_logits,
        router_partials)`` (the split-K partials of the router GEMMs whose routing tail sums
        them, else ``None``; ``router_logits`` is ``None`` with the fused router)."""
        # With the fused router, routing runs inside the routed branch, so the shared experts
        # fork before it instead of waiting for the router.
        if gemv_route and _K3_ROUTE_GEMV_DEFER_TAIL:
            router_partials = self._gemv_router_partials(hidden_states)
            return router_partials[0], router_partials
        if gemv_route:
            return self._gemv_route(hidden_states, route_target), None
        if route_target is not None and self._can_umma_route(hidden_states):
            router_partials = self._umma_router_partials(hidden_states)
            return router_partials[0], router_partials
        if _K3_FUSED_ROUTER != "0":
            return None, None
        return self._router_logits(hidden_states), None

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        *,
        residual: Optional[torch.Tensor] = None,
        defer_residual: bool = False,
    ) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16.

        With ``residual``, returns ``residual + moe_output``; with ``defer_residual`` as well, a
        ``_DeferredMoEResidual`` instead where that update would be the separate BF16 add.
        """
        identity = hidden_states
        projected_input = None
        early_routed_in = None
        shared_first = False
        shared_partial = None
        if self.expert_replicas is not None and not self._ensure_replica_tables():
            raise RuntimeError(
                "TRTLLM_K3_EXPERT_REPLICAS: the copy tables are created after loading the "
                "weights or in the first eager forward, before CUDA graph capture"
            )
        route_target = self._route_tail_target(hidden_states.shape[0])
        gemv_route = False
        router_partials = None
        if self._can_fuse_router_down(hidden_states):
            router_logits, projected_input = torch.ops.trtllm.dsv3_router_latent_gemm_op(
                hidden_states, self.gate.weight, self.routed_expert_down_proj.weight
            )
        else:
            router_issued = False
            if (
                self.latent_down_stream is not None
                and do_multi_stream()
                and not torch.compiler.is_compiling()
            ):
                # Forked ahead of the router GEMM: the latent chain needs only
                # hidden_states.
                gemv_route = route_target is not None and self._can_gemv_route(hidden_states)
                self.moe_latent_fork_event.record()
                if _K3_ROUTER_FIRST and hidden_states.shape[0] <= _K3_ROUTER_FIRST_MAX_M:
                    # Router before latent_down in creation order (see _K3_ROUTER_FIRST).
                    router_logits, router_partials = self._issue_router(
                        hidden_states, gemv_route, route_target
                    )
                    router_issued = True
                with torch.cuda.stream(self.latent_down_stream):
                    self.moe_latent_fork_event.wait()
                    early_routed_in = self._lowm_fp4_latent_input(hidden_states)
                    if early_routed_in is None:
                        early_routed_in = self._fused_latent_input(hidden_states)
                    if early_routed_in is None:
                        early_routed_in = self._routed_projection(
                            hidden_states, self.routed_expert_down_proj
                        )
                        if gemv_route:
                            # Quantized here, the routed input does not wait for the routing.
                            early_routed_in = self._quantize_routed_input(early_routed_in)
                    self.moe_latent_ready_event.record()
                # No record_stream (under graph capture it holds every layer's
                # blocks until capture ends): both tensors are read on main only
                # after the join, and later work on this stream forks from main.
                shared_first = self._shared_first_applies(hidden_states.shape[0])
                if shared_first and _K3_SHARED_FIRST == "1":
                    shared_partial = self._shared_gemms_first(identity, self.moe_latent_fork_event)
            if not router_issued:
                router_logits, router_partials = self._issue_router(
                    hidden_states, gemv_route, route_target
                )
            if shared_first and _K3_SHARED_FIRST == "router":
                self.moe_main_event.record()
                shared_partial = self._shared_gemms_first(identity, self.moe_main_event)
        moe_all_reduce = self.routed_experts.all_reduce if self._reduce_routed_output else None
        overlap_shared_allreduce = (
            self._defer_shared_allreduce
            and do_multi_stream()
            and self.shared_expert_stream is not None
            and not torch.compiler.is_compiling()
        )
        num_tokens = hidden_states.shape[0]
        merged = self._merged_reduce_buffers(hidden_states)
        routed_slot = shared_slot = None
        if merged is not None:
            split = num_tokens * self.moe_hidden_size
            routed_slot = merged[0][:split].view(num_tokens, self.moe_hidden_size)
            shared_slot = merged[0][split:].view(num_tokens, self.hidden_size)

        def _routed_output():
            logits = router_logits
            if logits is None:
                logits = self._fused_route(hidden_states)
                if logits is None:
                    logits = self.gate.compute_logits(hidden_states)
            moe_kwargs = {}
            if projected_input is not None:
                routed_in = projected_input
            elif early_routed_in is not None:
                routed_in = early_routed_in
                moe_kwargs = {
                    "input_ready_event": (
                        self.moe_latent_ready_event
                        if shared_partial is None
                        else self.moe_shared_gemm_event
                    )
                }
            else:
                routed_in = self._fused_latent_input(hidden_states)
                if routed_in is None:
                    routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            if isinstance(routed_in, Fp4QuantizedTensor):
                moe_kwargs["output_dtype"] = hidden_states.dtype
            if merged is not None:
                moe_kwargs["moe_output"] = routed_slot
            if route_target is not None and (router_partials is not None or not gemv_route):
                # Routed by the CuTe tail when the backend asks for the routing (after it
                # forks the input quantization); with the UMMA router GEMM the tail sums its
                # split-K partials.
                if router_partials is not None:
                    self.routed_experts.routing_method.deferred = functools.partial(
                        self._tail_route_partials, router_partials, target=route_target
                    )
                else:
                    self.routed_experts.routing_method.deferred = functools.partial(
                        self._tail_route, target=route_target
                    )
            y = self.routed_experts(
                routed_in,
                logits,
                all_rank_num_tokens=all_rank_num_tokens,
                **moe_kwargs,
            )
            if early_routed_in is not None:
                self._issue_l2_prefetch(num_tokens)
            self._record_expert_l2_routing(num_tokens)
            if route_target is not None:
                self.routed_experts.routing_method.deferred = None
            if overlap_shared_allreduce:
                self.moe_routed_event.record()
            if merged is not None:
                if y.data_ptr() != routed_slot.data_ptr():
                    routed_slot.copy_(y)
                return routed_slot
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output(tail_stream: Optional[torch.cuda.Stream] = None):
            """The shared expert's (reduced) output. With ``tail_stream``, gate_up runs on the
            current stream and everything after it on ``tail_stream``, which then records
            ``moe_shared_event``."""
            down_proj = self.shared_experts.down_proj
            ar_push = None
            if isinstance(down_proj, _Fp8BlockScaleWeightReadLinear):
                if merged is not None:
                    down_proj.next_output = shared_slot
                elif self.shared_down_all_reduce is not None and not self._defer_shared_allreduce:
                    # The FP8 down projection writes its partial straight into the NVLS
                    # all-reduce's input buffer when there is one (TRTLLM_NVLS_AR_ZERO_COPY=1),
                    # or into the MNNVL all-reduce's Lamport buffers (TRTLLM_K3_AR_PUSH).
                    down_proj.next_output = self.shared_down_all_reduce.nvls_input_buffer(
                        (identity.shape[0], down_proj.out_features)
                    )
                    ar_push = self.shared_down_all_reduce.lamport_push_target(
                        identity.shape[0], down_proj.out_features
                    )
                    down_proj.next_ar_push = ar_push
            final_params = (
                AllReduceParams(enable_allreduce=False)
                if self._defer_shared_allreduce or merged is not None
                else None
            )
            if tail_stream is None:
                return _shared_reduce(
                    self.shared_experts(identity, final_all_reduce_params=final_params),
                    ar_push,
                )
            # GatedMLP.forward's unfused path, split after gate_up (see _shared_tail_applies).
            mlp = self.shared_experts
            gate_up = mlp.gate_up_proj(identity)
            self.moe_shared_act_event.record()
            with torch.cuda.stream(tail_stream):
                self.moe_shared_act_event.wait()
                partial = mlp.down_proj(
                    mlp._apply_activation(gate_up),
                    all_reduce_params=final_params,
                    layer_idx=mlp.layer_idx,
                )
                shared_out = _shared_reduce(partial, ar_push)
                self.moe_shared_event.record()
            return shared_out

        def _shared_reduce(shared_out: torch.Tensor, ar_push) -> torch.Tensor:
            """Reduce the shared expert's down-projection output (or hand it to the merged
            reduction)."""
            down_proj = self.shared_experts.down_proj
            if merged is not None:
                # The partial joins the routed reduction (``_merged_reduce``).
                if shared_out.data_ptr() != shared_slot.data_ptr():
                    shared_slot.copy_(shared_out)
                return shared_slot
            if ar_push is not None:
                down_proj.next_ar_push = None
            if overlap_shared_allreduce:
                # Wait for all routed chunks, but overlap the norm/up tail.
                self.moe_routed_event.wait()
                shared_out = self._shared_all_reduce()(shared_out)
            elif ar_push is not None and ar_push.pushed:
                shared_out = self.shared_down_all_reduce.finish_lamport_push(ar_push).view(
                    shared_out.shape
                )
            elif self.shared_down_all_reduce is not None and not self._defer_shared_allreduce:
                shared_out = self.shared_down_all_reduce(shared_out)
            return shared_out

        # The final join includes the shared all-reduce (or, merged, the shared partial),
        # keeping it ordered before any outer routed reduction or the next layer's collectives.
        if shared_partial is not None:
            backend = self.routed_experts.backend
            backend.post_fc1_event = self.moe_fc1_event
            routed_out = _routed_output()
            fc1_recorded = backend.post_fc1_event is None
            backend.post_fc1_event = None
            with torch.cuda.stream(self.shared_expert_stream):
                if fc1_recorded:
                    self.moe_fc1_event.wait()
                shared_out = shared_partial
                if self._shared_needs_all_reduce():
                    shared_out = self._shared_all_reduce()(shared_partial)
                self.moe_shared_event.record()
            self._join_expert_l2_prefetch()
            self.moe_shared_event.wait()
        elif (
            self._latent_ar_ws is not None
            and merged is None
            and moe_all_reduce is not None
            and not self._defer_shared_allreduce
            and num_tokens <= self._latent_ar_ws["max_tokens"]
            and self.shared_expert_stream is not None
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        ):
            # The fork / join of maybe_execute_in_parallel, with the join left open until it is
            # known whether the latent all-reduce runs on its own workspace.
            self.moe_main_event.record()
            routed_out = _routed_output()
            tail_stream = self.shared_tail_stream if self._shared_tail_applies(num_tokens) else None
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_out = _shared_output(tail_stream)
                if tail_stream is None:
                    self.moe_shared_event.record()
            latent = routed_out.reshape(-1, routed_out.shape[-1])
            if latent.dtype == torch.bfloat16 and not self._latent_up_reads_shared(latent):
                return self._own_workspace_latent_tail(
                    routed_out, shared_out, residual, defer_residual
                )
            self._join_expert_l2_prefetch()
            self.moe_shared_event.wait()
        else:
            routed_out, shared_out = maybe_execute_in_parallel(
                _routed_output,
                _shared_output,
                self.moe_main_event,
                self.moe_shared_event,
                self.shared_expert_stream,
                disable_on_compile=True,
            )
            self._join_expert_l2_prefetch()
        if self._defer_shared_allreduce and not overlap_shared_allreduce:
            shared_out = self._shared_all_reduce()(shared_out)
        if merged is not None or self._reduce_routed_output:
            if merged is not None:
                routed_latent, shared_out = self._merged_reduce(*merged, num_tokens)
            else:
                fused = self._latent_up_from_quantized_all_reduce(
                    routed_out, shared_out, residual, moe_all_reduce
                )
                if fused is not None:
                    return fused
                # RMSNorm folded into the routed all-reduce (kARRMSNorm): the
                # MNNVL kernel normalizes straight out of the lamport buffer, so
                # the reduced row never round-trips through HBM for a norm kernel.
                routed_latent = moe_all_reduce(
                    routed_out,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RMS_NORM,
                        norm_weight=self.routed_expert_norm.weight,
                        eps=float(self.routed_expert_norm.variance_epsilon),
                    ),
                )
            if residual is not None:
                fused = self._fused_latent_up_residual(routed_latent, shared_out, residual)
                if fused is not None:
                    return fused
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            from ..modules.kimi_k3_moe_residual import _fused_moe_residual_add

            return _fused_moe_residual_add(routed_out, shared_out, residual)
        return routed_out + shared_out


def _helix_cp_v_b_shard(
    v_weight: torch.Tensor, *, num_heads_tp_cp: int, cp_rank: int
) -> torch.Tensor:
    """Select the CP rank's head chunk from a TP-local KV-B tensor."""
    start = cp_rank * num_heads_tp_cp
    return v_weight[start : start + num_heads_tp_cp]


def resolve_attention_quant_config(
    config: ModelConfig | None, layer_idx: int, projection: str
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.self_attn.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
        raise ValueError(
            f"Kimi K3 attention projection {names[0]} has unsupported checkpoint "
            f"quantization {selected.quant_algo}"
        )
    if selected.quant_algo == QuantAlgo.FP8_BLOCK_SCALES and selected.group_size not in (None, 128):
        raise ValueError(f"Kimi K3 attention requires 128x128 FP8 blocks for {names[0]}")
    result = copy.copy(selected)
    result.kv_cache_quant_algo = global_config.kv_cache_quant_algo
    return result


# ---------------------------------------------------------------------------
# KDA runtime (pool-backed prefill / decode via the FLA kernels).
# ---------------------------------------------------------------------------


class KimiMLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
    ) -> None:
        super().__init__()

        from ..modules.kimi_k3_mla import KimiK3MLAAttention

        max_positions = int(
            os.environ.get(
                _KIMI_K3_MLA_MAX_POSITIONS_ENV,
                cfg.max_position_embeddings,
            )
        )
        self.layer_idx = layer_idx
        # KimiK3MLAAttention owns MLA projection/head sharding. Keep only the
        # final output reduction in this wrapper so the output gate remains
        # between attention and the row-parallel o_proj.
        # Helix: mapping_with_cp (the CP original) activates the base MLA's
        # helix machinery; this wrapper's allreduce over the repurposed
        # mapping sums the base o_proj's tp*cp partials.
        mapping = model_config.mapping
        reduce_output = not mapping.enable_attention_dp and mapping.tp_size > 1
        self._o_allreduce = (
            AllReduce(
                mapping=mapping,
                strategy=model_config.allreduce_strategy,
                dtype=torch.bfloat16,
            )
            if reduce_output
            else None
        )
        # Set by the decoder layer for one forward (TRTLLM_K3_AR_PUSH_AMID): a pushed o_proj
        # all-reduce may be finished in place and returned as a LamportResult.
        self.ar_result_in_place = False
        # TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA: found by the previous layer's MoE block.
        _K3_MLA_LAYERS.setdefault(id(cfg), weakref.WeakValueDictionary())[layer_idx] = self
        self._q_b_l2_ranges: Tuple[Tuple[int, ...], List[Tuple[int, int]]] = ((), [])
        attention_config = copy.copy(model_config)
        attention_config._frozen = False
        attention_config.quant_config_dict = {
            name: resolve_attention_quant_config(model_config, layer_idx, name)
            for name in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "g_proj",
                "o_proj",
            )
        }
        kv_quant_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        fp8_kv_layer_ids = get_kimi_k3_fp8_kv_layer_ids(
            cfg, nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4
        )
        if layer_idx in fp8_kv_layer_ids:
            if kv_quant_algo != QuantAlgo.NVFP4:
                raise ValueError(
                    f"{KIMI_K3_FP8_KV_LAYERS_ENV} requires the model-wide "
                    "KV cache dtype to be NVFP4."
                )
            kv_quant_algo = QuantAlgo.FP8
            logger.info_once(
                f"Kimi K3 MLA layers using FP8 KV-cache fallback: {sorted(fp8_kv_layer_ids)}.",
                key="kimi_k3_fp8_kv_fallback_layers",
            )
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
        for projection_config in attention_config.quant_config_dict.values():
            projection_config.kv_cache_quant_algo = kv_quant_algo
        attention_config._frozen = model_config._frozen
        self.mixer = KimiK3MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            use_output_gate=cfg.mla_use_output_gate,
            max_position_embeddings=max_positions,
            model_config=attention_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
        )

    def q_b_l2_prefetch_ranges(self) -> List[Tuple[int, int]]:
        """``(address, bytes)`` of the q_b projection's weight and scale
        (TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA); recomputed when the tensors move."""
        q_b_proj = getattr(self.mixer, "q_b_proj", None)
        tensors = [
            t
            for t in (getattr(q_b_proj, "weight", None), getattr(q_b_proj, "weight_scale", None))
            if isinstance(t, torch.Tensor) and t.is_cuda and t.numel()
        ]
        key = tuple(t.data_ptr() for t in tensors)
        if self._q_b_l2_ranges[0] != key:
            from ..cute_dsl_kernels.blackwell.k3_l2_prefetch import tensor_range

            ranges = []
            for tensor in tensors:
                try:
                    ranges.append(tensor_range(tensor))
                except ValueError:
                    continue
            self._q_b_l2_ranges = (key, ranges)
        return self._q_b_l2_ranges[1]

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        o_proj = self.mixer.o_proj
        zero_copy = self._o_allreduce is not None and hasattr(o_proj, "next_output")
        if zero_copy:
            # o_proj writes its partial straight into the NVLS all-reduce's input
            # buffer when there is one (TRTLLM_NVLS_AR_ZERO_COPY=1).
            o_proj.next_output = self._o_allreduce.nvls_input_buffer(
                (hidden_states.shape[0], o_proj.out_features)
            )
        # TRTLLM_K3_AR_PUSH: o_proj stores its partial straight into the all-reduce's Lamport
        # buffers where the low-M GEMM serves it.
        ar_push = None
        if zero_copy and hasattr(o_proj, "next_ar_push"):
            ar_push = self._o_allreduce.lamport_push_target(
                hidden_states.shape[0], o_proj.out_features, in_place=self.ar_result_in_place
            )
            o_proj.next_ar_push = ar_push
        out = self.mixer(hidden_states, attn_metadata)
        if zero_copy:
            o_proj.next_output = None
        if ar_push is not None:
            o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            # TRTLLM_K3_AR_PUSH_AMID: the result may stay in the workspace for A_mid.
            out_shape = out.shape
            out = self._o_allreduce.finish_lamport_push(ar_push, in_place=self.ar_result_in_place)
            if not isinstance(out, LamportResult):
                out = out.view(out_shape)
        elif self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        dtype = torch.bfloat16

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            attention_config = copy.copy(model_config)
            attention_config._frozen = False
            attention_config.quant_config_dict = {
                name: resolve_attention_quant_config(model_config, layer_idx, name)
                for name in projection_names
            }
            attention_config._frozen = model_config._frozen
            self.linear_attn = KimiKDALinearAttention(
                cfg,
                layer_idx,
                mapping=model_config.mapping,
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
            )
        else:
            self.self_attn = KimiMLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                # CP original stashed by _setup_helix_mappings; None outside helix.
                mapping_with_cp=getattr(model_config, "_helix_mapping_with_cp", None),
            )

        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )
        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
            if _K3_EXPERT_L2_PREFETCH:
                # The attention block starts the prefetch of this layer's likeliest experts.
                attn = self.linear_attn if self.is_kda else self.self_attn.mixer
                attn.expert_l2_prefetch_hook = self.block_sparse_moe.issue_expert_l2_prefetch
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            attention_dp = model_config.mapping.enable_attention_dp
            if attention_dp:
                self.mlp_tp_size = 1
            else:
                self.mlp_tp_size = math.gcd(cfg.intermediate_size, model_config.mapping.tp_size)
                if (
                    self.mlp_tp_size > model_config.mapping.gpus_per_node
                    and not _K3_DENSE_MLP_FULL_TP
                ):
                    self.mlp_tp_size = math.gcd(
                        self.mlp_tp_size, model_config.mapping.gpus_per_node
                    )
            mlp_model_config = copy.copy(model_config)
            mlp_model_config.quant_config = QuantConfig()
            # K3's dense layer is BF16, so a unit block size gives the same
            # subgroup selection as DeepSeek-V3. Attention DP replicates the
            # MLP because ranks own different tokens; otherwise the subgroup
            # is block-aligned and stays within one node, unless
            # TRTLLM_K3_DENSE_MLP_FULL_TP spans the whole TP group.
            self.mlp = GatedMLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                bias=False,
                activation=SituAndMul(
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                    use_fused_activation=True,
                ),
                dtype=dtype,
                config=mlp_model_config,
                overridden_tp_size=self.mlp_tp_size,
                reduce_output=self.mlp_tp_size > 1,
                layer_idx=layer_idx,
            )

        # Stock fused RMSNorm for the plain (whole-tensor) norms; numerics
        # are drop-in for KimiK3RMSNorm (fp32 variance, weight applied
        # after downcast, use_gemma=False).
        self.input_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3). The res norms stay
        # KimiK3RMSNorm: they are consumed field-wise (.weight/.eps) by
        # _apply_attn_res and the fused attn_res op, never called as
        # modules.
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.mlp_res_norm = KimiK3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)

    def _attn_takes_mxfp8(self, num_rows: int, attn_metadata: AttentionMetadata) -> bool:
        """Whether the pre-attention mix should also emit its MXFP8 form for the attention mixer
        (``TRTLLM_K3_ATTN_RES_MXFP8``; generation-only batches): the KDA fused QKVG projection
        (``qkvg_takes_mxfp8``) or the MLA fused q_a / kv_a projection (``q_a_takes_mxfp8``)."""
        if not _K3_ATTN_RES_MXFP8 or attn_metadata.num_contexts != 0:
            return False
        if self.is_kda:
            return self.linear_attn.qkvg_takes_mxfp8(num_rows)
        q_a_takes_mxfp8 = getattr(self.self_attn.mixer, "q_a_takes_mxfp8", None)
        return q_a_takes_mxfp8 is not None and q_a_takes_mxfp8(num_rows)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows. The prefix sum (input and output) can be a
        ``_DeferredMoEResidual`` (``TRTLLM_K3_ATTN_RES_PDL=2``).

        ``capture`` is ``(spec_metadata, layer_id)``: the attention-side mixture
        computed below IS the aggregated stream value for ``layer_id``, so the
        DSpark tap reuses it instead of recomputing it.
        """
        valid_block_residual = block_residual[:num_snapshots]
        normed = None
        attn_input_mxfp8 = None
        if isinstance(hidden_states, _DeferredMoEResidual):
            # The previous layer's MoE residual update, folded into this mix when possible.
            if capture is None and num_snapshots > 0:
                fused = _k3_attn_res_pdl_moe_add(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=self._attn_takes_mxfp8(hidden_states.residual.shape[0], attn_metadata),
                )
                if fused is not None:
                    hidden_states, normed, *attn_input_mxfp8 = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

        if capture is not None and not _AUX_ATTN_RES_STREAM_ENABLED:
            # prefix_only wants the incoming running prefix, which is already in
            # hand -- no pre-norm intermediate is needed, so the tap is a plain
            # read and this layer keeps the fused attn_res+RMSNorm path like the
            # other 88. ``prefix_sum`` already has the previous layer's MLP
            # output folded in, i.e. the quantity vLLM builds as
            # ``prefix_sum + pending_mlp_out`` on its deferred-add path.
            capture[0].maybe_capture_hidden_states(capture[1], prefix_sum, None)
            capture = None

        if capture is not None:
            # The mixture tap needs the PRE-norm value (SGLang's
            # aggregate_stream is a "softmax mixture, no output norm"), which
            # the fused attn-res+RMSNorm kernel below does not expose. Split
            # the two steps on captured layers only — 5 of 93 — and leave the
            # fused path for every other layer.
            mixed = (
                _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
                if num_snapshots > 0
                else hidden_states
            )
            capture[0].maybe_capture_hidden_states(capture[1], mixed, None)
            hidden_states = self.input_layernorm(mixed)
        elif normed is not None:
            hidden_states = normed
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        # TRTLLM_K3_AR_PUSH_AMID: when A_mid below adds the attention output to the prefix sum,
        # the attention block may hand back its all-reduce result in place (a LamportResult).
        # A_mid must read it before the next all-reduce on the workspace re-arms that buffer:
        # the shared expert, whose down-projection all-reduce is the next one, forks after A_mid,
        # and its pushed finish kernel can only become resident once its producer has passed its
        # own griddepcontrol.wait (every grid before, A_mid included, has then completed).
        attn = self.linear_attn if self.is_kda else self.self_attn
        attn.ar_result_in_place = prefix_sum is not None and _k3_attn_res_lamport_applies(
            prefix_sum,
            valid_block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.post_attention_layernorm,
        )
        if self.is_kda:
            if attn_input_mxfp8:
                self.linear_attn.next_input_mxfp8 = tuple(attn_input_mxfp8)
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            if attn_input_mxfp8:
                self.self_attn.mixer.next_input_mxfp8 = tuple(attn_input_mxfp8)
            hidden_states = self.self_attn(hidden_states, attn_metadata)
        attn.ar_result_in_place = False

        if prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if (
            self.is_moe
            and (_K3_FUSE_MOE_RESIDUAL or _K3_LU_FUSED != "0")
            and not torch.compiler.is_compiling()
        ):
            prefix_sum = self.block_sparse_moe(
                hidden_states,
                getattr(attn_metadata, "all_rank_num_tokens", None),
                residual=prefix_sum,
                defer_residual=_defer_moe_residual(prefix_sum.shape[0]),
            )
            return prefix_sum, num_snapshots
        if self.is_moe:
            hidden_states = self.block_sparse_moe(
                hidden_states, getattr(attn_metadata, "all_rank_num_tokens", None)
            )
        else:
            hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiLinearModel(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream(priority=_K3_SHARED_STREAM_PRIORITY)]
        aux_stream_list += [torch.cuda.Stream() for _ in range(2)]
        aux_stream_list.append(torch.cuda.Stream(priority=_K3_LATENT_STREAM_PRIORITY))
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # KimiK3RMSNorm (not RMSNorm): consumed field-wise (.weight/.eps)
        # by _apply_attn_res and the fused attn_res op.
        self.output_attn_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

        logger.info_once(
            f"Kimi K3 attention-residual kernels: fused={_FUSED_ATTN_RES_ENABLED}, "
            f"topology={_ATTN_RES_TOPOLOGY}, max_tokens={_FUSED_ATTN_RES_MAX_TOKENS}",
            key="kimi_k3_attn_res_fusion",
        )
        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
            key="kimi_k3_aux_capture_mode",
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            weight = self.embed_tokens.weight
            if spec_head_fusion_enabled() and input_ids.dim() == 1 and weight.is_contiguous():
                # Same rows as nn.Embedding, copied with 16-byte accesses.
                inputs_embeds = embedding_gather(input_ids, weight)
            else:
                inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        num_tokens = attn_metadata.num_tokens
        assert hidden_states.shape[0] == num_tokens, (
            f"Kimi K3 does not support padded batches "
            f"(got {hidden_states.shape[0]} rows, metadata says {num_tokens} "
            "tokens); disable CUDA graphs and the overlap scheduler."
        )

        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            # DFlash/DSpark hidden-state capture. The drafter is distilled on
            # the aggregated stream value -- the pre-norm softmax mixture its
            # next consumer sees -- not on the raw prefix sum a layer returns.
            # Layer i+1 computes exactly that tensor as its own attention-side
            # mixture, so the tap rides along instead of recomputing it.
            # Capturing the prefix sum instead costs 4.5pt of draft acceptance
            # on K3 + RadixArk DSpark (AR 66.9% -> 71.4%). Ground truth:
            # SGLang kimi_k3.py:2589 _dspark_capture_stream, attn_residual.py:285
            # aggregate_stream.
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            hidden_states, num_snapshots = layer(
                hidden_states, block_residual, num_snapshots, attn_metadata, capture=capture
            )
        hidden_states = _materialize_prefix_sum(hidden_states)

        # The last layer has no successor to fold the tap into, so its stream
        # value uses the model's own output-side score weights.
        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0 and _AUX_ATTN_RES_STREAM_ENABLED
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        return _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )


# ---------------------------------------------------------------------------
# Causal LM wrapper + weight loading.
# ---------------------------------------------------------------------------


_FP8_BLOCK_SCALE_SUFFIX = "_scale"
_FP8_BLOCK = 128


def _fp8_block_scale_key(weight_key: str) -> str:
    return weight_key + _FP8_BLOCK_SCALE_SUFFIX


def _value_shape(value) -> Tuple[int, ...]:
    """Return a tensor or lazy safetensors slice shape without materializing it."""
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    get_shape = getattr(value, "get_shape", None)
    if get_shape is None:
        raise ValueError(f"unsupported weight type: {type(value)}")
    return tuple(get_shape())


def _materialize(value, indices=None) -> torch.Tensor:
    """Materialize all or part of a tensor or lazy safetensors slice."""
    if indices is not None:
        return value[indices]
    if isinstance(value, torch.Tensor):
        return value
    # ``[:]`` is invalid for the scalar NVFP4 weight_scale_2/input_scale
    # entries. This matches ``_ReopenSafeTensorSlice._tensor``.
    if len(_value_shape(value)) == 0:
        return value[()]
    return value[:]


def _checkpoint_fp8_pair(
    ckpt_key: str,
    src,
    weights,
    *,
    weight_slice=None,
    scale_slice=None,
):
    """``(fp8 weight, fp32 128x128 block scale)`` if this tensor is FP8_PB_WO.

    ``None`` for an ordinary (BF16) checkpoint tensor. Raises when an FP8
    tensor has no companion scale rather than letting the caller fall through
    to a cast that would silently drop it.
    """
    src = _materialize(src, weight_slice)
    if src.dtype != torch.float8_e4m3fn:
        return None
    scale_key = _fp8_block_scale_key(ckpt_key)
    if scale_key not in weights and scale_key + "_inv" in weights:
        scale_key += "_inv"
    if scale_key not in weights:
        raise KeyError(
            f"Kimi K3: {ckpt_key} is FP8 E4M3 but has no {scale_key} or {scale_key}_inv; refusing "
            "to load it as if it were unquantized."
        )
    scale = _materialize(weights[scale_key]).float()
    # The checkpoint stores the block scale 4-D as
    # [ceil(N/128), 1, ceil(K/128), 1]. Normalize here, once, so every
    # consumer sees the plain 2-D [n_blocks_m, n_blocks_k] expected by both
    # the dequantization below and the CuTe scale preparation.
    if scale.dim() == 4:
        scale = scale.reshape(scale.shape[0], scale.shape[2])
    if scale_slice is not None:
        scale = scale[scale_slice]
    return src, scale


def _dequantize_fp8_block_scaled(ckpt_key: str, src: torch.Tensor, weights) -> torch.Tensor:
    """Apply checkpoint scales for projections explicitly configured as BF16.

    Checkpoint-native FP8 attention bypasses this loader. This handles an
    excluded projection or another unquantized trunk parameter whose source
    tensor is quantized; casting the codes alone would discard its scales.
    """
    pair = _checkpoint_fp8_pair(ckpt_key, src, weights)
    if pair is None:
        return src
    src, scale = pair
    out = src.to(torch.float32)
    if scale.numel() == 1:
        return (out * scale.reshape(())).to(torch.bfloat16)
    expanded = scale.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
    if expanded.shape[0] < out.shape[0] or expanded.shape[1] < out.shape[1]:
        raise ValueError(
            f"Kimi K3: {_fp8_block_scale_key(ckpt_key)} covers "
            f"{tuple(expanded.shape)} but {ckpt_key} is {tuple(out.shape)}."
        )
    return (out * expanded[: out.shape[0], : out.shape[1]]).to(torch.bfloat16)


@register_auto_model("KimiLinearForCausalLM")
class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[KimiLinearModel, Any]):
    """Kimi K3 text core (KDA + MLA + MoE).

    Serves text-only ``kimi_linear`` checkpoints directly, and is reused as the
    text backbone by the multimodal ``KimiK3ForConditionalGeneration`` wrapper
    (``modeling_kimi_k3_vl``). The composite ``KimiK3ForConditionalGeneration``
    architecture is registered by that wrapper, not here."""

    mamba_metadata_cls = KimiK3MambaMetadata

    def __init__(self, model_config: ModelConfig):
        mlp_fp8 = os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV, "0")
        if mlp_fp8 not in ("0", "1"):
            raise ValueError(
                f"{_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV} must be 0 or 1; got {mlp_fp8!r}"
            )
        self._fp8_weight_read_moe_mlp = mlp_fp8 == "1"
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)

        # Helix: swap in the repurposed mapping; restored after super().__init__.
        self._setup_helix_mappings(model_config, cfg, spec_config)
        # Supported spec-dec modes:
        # - SA (suffix automaton): one-engine in-forward drafting, no draft
        #   weights; the KDA/MLA verify paths below implement multi-token
        #   verification for it.
        # - DFlash: external-drafter parallel drafting; the drafter is a
        #   separate dense checkpoint (K2.7-Code-DFlash schema) consumed by
        #   the generic DFlashForCausalLM wrapper, and the target only has
        #   to expose per-layer hidden states via maybe_capture_hidden_states
        #   (see KimiLinearModel.forward).
        # - DSpark: the same external-drafter flow with the Markov and
        #   confidence heads enabled (RadixArk/Kimi-K3-DSpark and friends).
        #   The target side is identical -- the capture in
        #   KimiLinearModel.forward is unconditional -- so this gate is the
        #   only place the mode has to be admitted.
        # Modes needing draft heads (MTP/Eagle) are blocked until a
        # draft-head checkpoint exists.
        assert (
            spec_config is None
            or spec_config.spec_dec_mode.is_sa()
            or spec_config.spec_dec_mode.is_dflash()
            or spec_config.spec_dec_mode.is_dspark()
        ), "Kimi K3 supports speculative decoding only with SA, DFlash or DSpark"
        super().__init__(
            KimiLinearModel(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

        # Restore the CP original: executor-side helix bookkeeping keys off
        # has_cp_helix() at runtime.
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True

    def _setup_helix_mappings(
        self,
        model_config: ModelConfig,
        cfg: "PretrainedConfig",
        spec_config: Optional["DecodingBaseConfig"],
    ) -> None:
        """Validate helix preconditions and stage the dual-mapping swap.

        DeepseekV3 pattern: the MLA layers keep the CP original; everything
        else is built against the repurposed mapping (CP ranks become TP
        ranks). Sets ``mapping_with_cp`` (restored after construction) and
        ``_repurposed_tp_mapping`` (load_weights shard selection); both stay
        None outside helix.
        """
        self.mapping_with_cp = None
        self._repurposed_tp_mapping = None
        if not model_config.mapping.has_cp_helix():
            return
        if model_config.mapping.enable_attention_dp:
            raise ValueError(
                "Kimi K3 helix phase 1 requires enable_attention_dp="
                "False: the helix ADP token-scatter conflicts with the "
                "per-request locality of KDA recurrent state."
            )
        if spec_config is not None:
            # Helix supports only the standalone DSpark drafter (verified on
            # the V2 superblock ledger); reject everything else loudly rather
            # than let an unsupported spec mode run silently wrong.
            if not spec_config.spec_dec_mode.is_dspark():
                raise ValueError(
                    "Kimi K3 helix supports speculative decoding only with "
                    f"DSpark (standalone drafter); got "
                    f"{spec_config.decoding_type!r}."
                )
            # The SpeculationGate acceptance-rate trip permanently disables
            # speculation mid-flight while enable_spec_decode stays True;
            # in-flight helix requests then fall into the plain generation
            # loop whose position math (total_input_len_cp +
            # py_decoding_iter - 1) is stale once any draft token was
            # accepted -> silently wrong RoPE positions and KV slots. Reject
            # the trip wires until that loop is helix-group aware.
            if (
                spec_config.acceptance_rate_window_size is not None
                or spec_config.acceptance_rate_threshold is not None
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "acceptance-rate gate (acceptance_rate_window_size / "
                    "acceptance_rate_threshold): dynamically disabling "
                    "speculation mid-flight leaves helix requests on a "
                    "single-token position formula."
                )
            # DSparkDecodingConfig resolves this from the drafter checkpoint's
            # weight index; the embedded flavour also carries a
            # speculative_model path, so the flag is the only reliable probe.
            if spec_config.draft_is_embedded_in_target:
                raise ValueError(
                    "Kimi K3 helix supports only the standalone DSpark "
                    "drafter; the embedded (in-target) flavour shares the "
                    "target KV bookkeeping in ways the helix ledger does "
                    "not model."
                )
        cp = model_config.mapping.cp_size
        repurposed_tp = model_config.mapping.tp_size * cp
        if cfg.num_attention_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) "
                f"to divide the MLA head count ({cfg.num_attention_heads})."
            )
        kda_heads = cfg.linear_attn_config["num_heads"]
        if kda_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) to "
                f"divide the KDA head count ({kda_heads})."
            )
        # MoE splits apply to the repurposed tp*cp group (helix
        # moe_world_size = tp*cp); default EP-only. The Mapping constructor
        # skips its product check when both sizes are 1, so validate here.
        moe_ep = repurposed_tp
        if model_config.mapping.moe_tp_ep_user_specified:
            moe_tp = model_config.mapping.moe_tp_size
            moe_ep = model_config.mapping.moe_ep_size
            if moe_tp * moe_ep != repurposed_tp:
                raise ValueError(
                    f"Kimi K3 helix: moe_tensor_parallel_size ({moe_tp}) x "
                    f"moe_expert_parallel_size ({moe_ep}) must equal "
                    f"tp_size*cp_size ({repurposed_tp}): MoE runs on the "
                    "repurposed tp*cp group."
                )
        if cfg.num_experts and cfg.num_experts % moe_ep != 0:
            raise ValueError(
                f"Kimi K3 helix requires the MoE EP size ({moe_ep}) to "
                f"divide the routed expert count ({cfg.num_experts}): each "
                "EP rank of the repurposed tp*cp group holds whole experts."
            )
        self.mapping_with_cp = copy.deepcopy(model_config.mapping)
        repurposed = model_config.mapping.repurpose_helix_cp_to_tp()
        # repurpose passes resolved moe sizes, which the Mapping constructor
        # mistakes for user-specified values; restore the flag.
        repurposed.moe_tp_ep_user_specified = self.mapping_with_cp.moe_tp_ep_user_specified
        # load_weights shard selection must use this tp_rank; the restored
        # CP original's tp_rank is 0 on every rank.
        self._repurposed_tp_mapping = repurposed
        model_config._frozen = False
        model_config.mapping = repurposed
        # Side-channel for the MLA layers (avoids threading a kwarg through
        # every intermediate signature).
        model_config._helix_mapping_with_cp = self.mapping_with_cp
        model_config._frozen = True

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - enable_block_reuse defaults off: reuse is supported as an
        #   explicit opt-in (routes to CppMambaHybridCacheManager with
        #   per-block KDA state snapshots); the default stays on the
        #   Mixed manager, which SA speculative decoding requires.
        # - tokens_per_block=64 remains the validated K3 page size. When the
        #   effective MLA backend is trtllm-gen, H=96 is declined by the
        #   FlashInfer wrapper and served by the legacy TRTLLM-Gen fallback.
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            }
        }

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi K3 disaggregated serving requires the Python transceiver.

        Only the Python NIXL transceiver (KvCacheTransceiverV2) can move
        the KDA recurrent state; the C++ transceiver has no KDA support.
        Adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL. An explicit non-Python runtime is
        rejected by ``get_kv_cache_manager_cls`` rather than silently
        routed to a path that cannot transfer the recurrent state.
        """
        return "PYTHON"

    # ------------------------------------------------------------------
    # Weight loading (streams the 1.5TB checkpoint; only the rank-local
    # expert slice of each MoE layer is kept: whole experts under MoE EP,
    # the intra-expert intermediate shard of ALL experts under MoE TP —
    # in the TP case every expert tensor is read and sliced, so expect a
    # correspondingly longer load).
    # ------------------------------------------------------------------

    def _attention_fp8_linears(self) -> list[tuple[str, TrtllmLinear, nn.Module]]:
        """Collect checkpoint-backed FP8 attention projections for weight loading.

        The loader uses this list to enumerate required weight/scale keys, exclude
        these parameters from ordinary trunk loading, and load FP8 codes and scales
        together. Derived QKVG fusion is excluded because it is built after loading.

        Returns:
            Tuples of runtime module path, linear, and owning attention module.
        """
        projections = []
        for index, layer in enumerate(self.model.layers):
            if layer.is_kda:
                attention = layer.linear_attn
                scope = f"model.layers.{index}.linear_attn"
            else:
                attention = layer.self_attn.mixer
                scope = f"model.layers.{index}.self_attn.mixer"
            for attr, module in attention.named_children():
                if attr == "qkvg_proj":
                    # Decode fusion is derived from separately loaded checkpoint projections.
                    continue
                if isinstance(module, TrtllmLinear) and module.has_fp8_block_scales:
                    projections.append((f"{scope}.{attr}", module, attention))
        return projections

    def _load_attention_fp8(self, weights: Dict[str, torch.Tensor], prefix: str) -> int:
        """Load checkpoint FP8 codes/scales without a BF16 projection intermediate."""
        projections = self._attention_fp8_linears()
        for name, linear, attention in projections:
            # Both attention implementations use self_attn keys in the checkpoint.
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            attr = name.rsplit(".", 1)[1]
            if attr == "kv_a_proj_with_mqa" and attention.fuse_qkv_a_proj:
                # The runtime fuses Q-A/KV-A, but the checkpoint stores them separately.
                # Construction checks that the fusion boundary aligns with FP8 blocks.
                q_key = key.replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                pairs = []
                for part_key, rows in (
                    (q_key, attention.q_lora_rank),
                    (key, attention.kv_lora_rank + attention.qk_rope_head_dim),
                ):
                    part = _materialize(weights[part_key])
                    pair = _checkpoint_fp8_pair(part_key, part, weights)
                    if pair is None:
                        raise ValueError(
                            f"{part_key}: quant config declares FP8 but tensor is not E4M3"
                        )
                    codes, scales = pair
                    expected = (rows, attention.hidden_size)
                    if tuple(codes.shape) != expected:
                        raise ValueError(
                            f"{part_key}: checkpoint shape {tuple(codes.shape)} != {expected}"
                        )
                    if tuple(scales.shape) != tuple(math.ceil(d / _FP8_BLOCK) for d in expected):
                        raise ValueError(f"{part_key}: scale shape does not cover {expected}")
                    pairs.append(pair)
                linear.load_weights(
                    [
                        {
                            "weight": torch.cat([pair[0] for pair in pairs], dim=0),
                            "weight_scale": torch.cat([pair[1] for pair in pairs], dim=0),
                        }
                    ]
                )
                continue
            source = weights[key]
            shape = (
                tuple(source.shape)
                if isinstance(source, torch.Tensor)
                else tuple(source.get_shape())
            )
            local_shape = (linear.out_features, linear.in_features)
            scale_slice = None
            if isinstance(attention, KimiKDALinearAttention):
                # KDA linears already have rank-local dimensions. Slice the lazy source
                # before materializing it, and apply the same shard to its block scales.
                split_dim = 1 if attr == "o_proj" else 0
                tp_size = attention._kda_tp_size
                tp_rank = attention._kda_tp_rank
                expected_shape = list(local_shape)
                expected_shape[split_dim] *= tp_size
                if shape != tuple(expected_shape):
                    raise ValueError(f"{key}: checkpoint shape {shape} != {tuple(expected_shape)}")
                if tp_size > 1:
                    width = local_shape[split_dim]
                    start, end = tp_rank * width, (tp_rank + 1) * width
                    if start % _FP8_BLOCK or end % _FP8_BLOCK:
                        raise ValueError(f"{key}: TP slice [{start}:{end}] is not block-aligned")
                    indices = [slice(None), slice(None)]
                    indices[split_dim] = slice(start, end)
                    source = source[tuple(indices)]
                    scale_indices = [slice(None), slice(None)]
                    scale_indices[split_dim] = slice(
                        start // _FP8_BLOCK, math.ceil(end / _FP8_BLOCK)
                    )
                    scale_slice = tuple(scale_indices)
            pair = _checkpoint_fp8_pair(key, _materialize(source), weights)
            if pair is None:
                raise ValueError(
                    f"{key}: quant config declares FP8 but checkpoint tensor is not E4M3"
                )
            weight, scale = pair
            full_scale_shape = tuple(math.ceil(dim / _FP8_BLOCK) for dim in shape)
            if tuple(scale.shape) != full_scale_shape:
                raise ValueError(f"{key}: scale shape {tuple(scale.shape)} != {full_scale_shape}")
            if scale_slice is not None:
                scale = scale[scale_slice]
            if isinstance(attention, KimiKDALinearAttention):
                if tuple(scale.shape) != tuple(math.ceil(dim / _FP8_BLOCK) for dim in local_shape):
                    raise ValueError(f"{key}: scale shard does not cover local shape {local_shape}")
                linear.load_weights([{"weight": weight, "weight_scale": scale}])
            elif attr == "kv_b_proj":
                # KV-B also supplies decode absorption weights; reorder codes and scales
                # together and retain raw FP8 copies for the absorption kernels.
                self._load_mla_fp8_kv_b(attention, weight, scale)
            else:
                # Other MLA projections delegate TP sharding to the common Linear loader.
                expected_shape = list(local_shape)
                if linear.tp_mode is not None:
                    split_dim = 0 if linear.tp_mode == TensorParallelMode.COLUMN else 1
                    expected_shape[split_dim] *= linear.tp_size
                if shape != tuple(expected_shape):
                    raise ValueError(f"{key}: checkpoint shape {shape} != {tuple(expected_shape)}")
                linear.load_weights([{"weight": weight, "weight_scale": scale}])
        return len(projections)

    @staticmethod
    def _load_mla_fp8_kv_b(attention: nn.Module, weight: torch.Tensor, scale: torch.Tensor) -> None:
        """Reorder checkpoint KV-B heads and their scale grid together."""
        linear = attention.kv_b_proj
        h, n, v, k = (
            attention.num_heads_tp,
            attention.qk_nope_head_dim,
            attention.v_head_dim,
            attention.kv_lora_rank,
        )
        if any(dim % _FP8_BLOCK for dim in (n, v, k)):
            raise ValueError(
                "Kimi MLA FP8 absorption requires block-aligned head and latent dimensions"
            )
        expected_shape = (h * linear.tp_size * (n + v), k)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"Kimi MLA KV-B shape {tuple(weight.shape)} != {expected_shape}")
        device = linear.weight.device
        local = linear.load_shard(weight, device=device).reshape(h, n + v, k)
        local_scale = linear.load_shard(scale, scale_span=_FP8_BLOCK, device=device).reshape(
            h, (n + v) // _FP8_BLOCK, k // _FP8_BLOCK
        )
        key_weight, value_weight = local.split((n, v), dim=1)
        key_scale, value_scale = local_scale.split((n // _FP8_BLOCK, v // _FP8_BLOCK), dim=1)
        linear.weight.data.copy_(
            torch.cat((key_weight.reshape(h * n, k), value_weight.reshape(h * v, k)))
        )
        linear.weight_scale.data.copy_(
            torch.cat((key_scale.flatten(0, 1), value_scale.flatten(0, 1)))
        )
        attention.k_b_proj_trans.data.copy_(key_weight.transpose(1, 2))
        attention.k_b_proj_trans_scale.data.copy_(key_scale.transpose(1, 2))
        if _K3_HELIX_V_UP_BEFORE_A2A and attention.mapping.has_cp_helix():
            expanded = value_scale.repeat_interleave(_FP8_BLOCK, -2).repeat_interleave(
                _FP8_BLOCK, -1
            )
            attention.helix_v_b_proj_full = (value_weight.float() * expanded).to(torch.bfloat16)
        value_weight = _helix_cp_v_b_shard(
            value_weight,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        value_scale = _helix_cp_v_b_shard(
            value_scale,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        # Absorption must keep raw checkpoint codes independent of Linear's
        # backend-specific weight transformations.
        attention.v_b_proj = nn.Parameter(value_weight.clone().contiguous(), requires_grad=False)
        attention.v_b_proj_scale.data.copy_(value_scale)
        for target, codes, scales in (
            (
                attention.k_b_proj_trans_dequant,
                key_weight.transpose(1, 2),
                key_scale.transpose(1, 2),
            ),
            (attention.v_b_proj_dequant, value_weight, value_scale),
        ):
            if target is not None:
                expanded = scales.repeat_interleave(_FP8_BLOCK, -2).repeat_interleave(
                    _FP8_BLOCK, -1
                )
                target.data.copy_((codes.float() * expanded).to(target.dtype))
        linear.process_weights_after_loading()

    def _trunk_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Named parameters of the trunk only. Spec-dec draft modules
        (e.g. the DFlash drafter attached by SpecDecOneEngineForCausalLM)
        live in a separate checkpoint loaded by
        ModelLoader.load_draft_weights, not in the target checkpoint. MLA
        K/V absorb Parameters are derived by the KV-B loader and are likewise
        excluded from checkpoint jobs."""
        fp8_prefixes = tuple(name + "." for name, _, _ in self._attention_fp8_linears())
        return {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith("draft_model.")
            and not name.startswith(fp8_prefixes)
            and ".linear_attn.qkvg_proj." not in name
            and not name.endswith(_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES)
        }

    def checkpoint_name_plan(self, prefix: str):
        """Return ``(name_map, expected_keys, expert_jobs)``.

        ``name_map`` maps every model parameter name to its checkpoint key
        (for fused ``gate_up_proj`` parameters the mapped key is virtual;
        the two real per-half keys come from ``_gate_up_ckpt_keys``);
        ``expected_keys`` additionally covers the rank-local per-expert MXFP4
        tensors; ``expert_jobs`` lists ``(layer_idx, moe_module, key_base)``
        for backend-owned expert slots. Exposed separately so the weight-name
        mapping can be dry-run without touching any tensor data.
        """
        params = self._trunk_parameters()
        expected_keys = set()
        name_map: Dict[str, str] = {}
        for name in params:
            # ConfigurableMoE's backend owns already-packed runtime weights,
            # generated zero biases, and SiTU constants. They do not have
            # one-to-one checkpoint parameter names.
            if ".routed_experts.backend." in name:
                continue
            if name == "lm_head.weight":
                ckpt_key = prefix + "lm_head.weight"
            elif ".linear_attn." in name:
                ckpt_key = prefix + name.replace(".linear_attn.", ".self_attn.")
            else:
                # MLA backends remain wrapped by ``self_attn.mixer`` while
                # checkpoint names have no ``mixer`` scope.
                ckpt_key = prefix + name.replace(".self_attn.mixer.", ".self_attn.")
            name_map[name] = ckpt_key
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Fused [gate | up] MLP layout (dense mlp / shared_experts):
                # the checkpoint stores two separate tensors.
                expected_keys.update(_gate_up_ckpt_keys(ckpt_key))
            else:
                expected_keys.add(ckpt_key)

        # Backend-owned expert slots (per-expert checkpoint tensors; the
        # rank-local id range — an EP slice of whole experts, or ALL experts
        # when the routed MoE is TP-sharded (moe_ep=1 -> ids 0..num_experts)).
        expert_jobs = []
        for layer_idx, layer in enumerate(self.model.layers):
            if not getattr(layer, "is_moe", False):
                continue
            moe = layer.block_sparse_moe
            base = f"{prefix}model.layers.{layer_idx}.block_sparse_moe.experts"
            kinds = moe.expert_ckpt_spec.kinds
            for expert_idx in moe.ckpt_expert_ids:
                for w in ("w1", "w2", "w3"):
                    for kind in kinds:
                        expected_keys.add(f"{base}.{expert_idx}.{w}.{kind}")
            expert_jobs.append((layer_idx, moe, base))
        for name, _, _ in self._attention_fp8_linears():
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            expected_keys.update((key, _fp8_block_scale_key(key)))
        for index, layer in enumerate(self.model.layers):
            if getattr(layer, "is_kda", True):
                continue
            attention = layer.self_attn.mixer
            if getattr(attention, "fuse_qkv_a_proj", False):
                key = f"{prefix}model.layers.{index}.self_attn.q_a_proj.weight"
                expected_keys.add(key)
                if attention.kv_a_proj_with_mqa.has_fp8_block_scales:
                    expected_keys.add(_fp8_block_scale_key(key))
        return name_map, expected_keys, expert_jobs

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
        params = self._trunk_parameters()
        name_map, expected_keys, expert_jobs = self.checkpoint_name_plan(prefix)

        self._validate_checkpoint_keys(weights, expected_keys, prefix)
        num_params = self._load_trunk_params(weights, params, name_map)
        num_params += self._load_attention_fp8(weights, prefix)
        self._load_expert_slices(weights, expert_jobs)
        self._finalize_weight_load(num_params, len(expert_jobs))
        device = next(self.parameters()).device
        if device.type == "cuda":
            # Lazy source mappings are load-scoped; finish nonblocking H2D
            # work before the caller can release the weights container.
            torch.cuda.synchronize(device)
        self._permute_routers_to_expert_placement()

    def _permute_routers_to_expert_placement(self) -> None:
        """Reorder the router of every layer with a static expert placement into slot order.

        Runs once, after the checkpoint values are in place; see ``TRTLLM_K3_EXPERT_PLACEMENT``.
        """
        num_permuted = 0
        for layer in self.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if not isinstance(moe, KimiK3MoERuntime) or moe.expert_of_slot is None:
                continue
            permute_router_to_slots(
                moe.gate.weight.data, moe.gate.e_score_correction_bias.data, moe.expert_of_slot
            )
            num_permuted += 1
        for layer in self.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if isinstance(moe, KimiK3MoERuntime) and moe.expert_replicas is not None:
                moe._ensure_replica_tables()
        if num_permuted:
            logger.info(
                f"Kimi K3: routers of {num_permuted} MoE layers permuted to the expert placement"
            )

    def _validate_checkpoint_keys(
        self, weights: Dict[str, torch.Tensor], expected_keys: Set[str], prefix: str
    ) -> None:
        """Key-set validation (both directions): every expected key must be
        present; unmatched checkpoint keys (beyond the expected leftovers)
        only warn."""
        ckpt_keys = set(weights.keys())
        relevant_ckpt_keys = {
            k
            for k in ckpt_keys
            if not (k.startswith("vision_tower.") or k.startswith("mm_projector."))
        }
        missing = sorted(
            key
            for key in expected_keys - ckpt_keys
            if not (key.endswith(".weight_scale") and key + "_inv" in ckpt_keys)
        )
        if missing:
            raise KeyError(
                f"Kimi K3 load_weights: {len(missing)} expected checkpoint "
                f"keys are missing, e.g. {missing[:10]}"
            )
        unexpected = relevant_ckpt_keys - expected_keys
        # Non-local experts and (in layer-truncated debug mode) extra layers
        # are expected leftovers.
        surprising = sorted(
            k
            for k in unexpected
            if ".block_sparse_moe.experts." not in k and not k.startswith(f"{prefix}model.layers.")
        )
        if surprising:
            logger.warning(
                f"Kimi K3 load_weights: {len(surprising)} unmatched "
                f"checkpoint keys, e.g. {surprising[:10]}"
            )

    def _load_trunk_params(
        self,
        weights: Dict[str, torch.Tensor],
        params: Dict[str, torch.nn.Parameter],
        name_map: Dict[str, str],
    ) -> int:
        """Load every non-expert trunk parameter concurrently (with the
        per-parameter TP-shard / pad / fuse conversions) and return the
        number of parameters loaded."""
        # The checkpoint stores every MLA KV-B head as interleaved [K | V]
        # rows. Runtime keeps one DeepSeek-style [all K | all V] parameter
        # instead, so context can project directly into the FMHA layout and
        # absorbed decode can take zero-copy K/V views.
        mla_mixers = [
            layer.self_attn.mixer
            for layer in self.model.layers
            if not getattr(layer, "is_kda", True)
        ]
        mla_kv_b_mixers = {id(mixer.kv_b_proj.weight): mixer for mixer in mla_mixers}
        mla_fused_a_mixers = {
            id(mixer.kv_a_proj_with_mqa.weight): mixer
            for mixer in mla_mixers
            if getattr(mixer, "fuse_qkv_a_proj", False)
        }
        mla_head_shard_linears = {}
        for mixer in mla_mixers:
            mla_head_shard_linears[id(mixer.q_b_proj.weight)] = mixer.q_b_proj
            mla_head_shard_linears[id(mixer.o_proj.weight)] = mixer.o_proj
            g_proj = getattr(mixer, "g_proj", None)
            if g_proj is not None:
                mla_head_shard_linears[id(g_proj.weight)] = g_proj

        device = next(self.parameters()).device

        # MLP TP shard index. A dense MLP whose intermediate size does not
        # divide model TP uses a smaller repeated TP subgroup, so its local
        # shard rank is model tp_rank modulo the parameter's shard count.
        # Under helix the modules were sharded against the repurposed
        # mapping; the restored CP original's tp_rank is 0 on every rank.
        model_tp_rank = (
            self._repurposed_tp_mapping.tp_rank
            if self._repurposed_tp_mapping is not None
            else self.model_config.mapping.tp_rank
        )
        # KDA head-shard (attention-DP off): rank r loads head rows/cols
        # [r*local : (r+1)*local] of every head-major KDA tensor.
        kda_tp_size, kda_tp_rank = 1, 0
        for layer in self.model.layers:
            if getattr(layer, "is_kda", False):
                kda_tp_size = layer.linear_attn._kda_tp_size
                kda_tp_rank = layer.linear_attn._kda_tp_rank
                break

        def load_param(name: str, param: torch.nn.Parameter):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Row-concat the checkpoint's separate gate_proj / up_proj
                # tensors into the fused [gate | up] parameter.
                gate_key, up_key = _gate_up_ckpt_keys(name_map[name])
                # This branch materializes its own sources, so it needs the
                # same FP8 handling as the single-tensor path below.
                gate = _dequantize_fp8_block_scaled(
                    gate_key, _materialize(weights[gate_key]), weights
                )
                up = _dequantize_fp8_block_scaled(up_key, _materialize(weights[up_key]), weights)
                inter = param.shape[0] // 2
                if gate.shape[0] != inter and gate.shape[0] % inter == 0:
                    # TP-sharded fused MLP (shared experts on the direct
                    # MoE path, dense MLP with attention-DP off): take this
                    # subgroup rank's matching row block from each half so
                    # the SiTU gate/up pairs stay aligned.
                    shard_count = gate.shape[0] // inter
                    lo = (model_tp_rank % shard_count) * inter
                    gate = gate[lo : lo + inter]
                    up = up[lo : lo + inter]
                if gate.shape != (inter, param.shape[1]) or up.shape != gate.shape:
                    raise ValueError(
                        f"{name}: checkpoint gate/up shapes "
                        f"{tuple(gate.shape)} / {tuple(up.shape)} do not "
                        f"concat to param shape {tuple(param.shape)}"
                    )
                param.data[:inter].copy_(gate.to(param.dtype))
                param.data[inter:].copy_(up.to(param.dtype))
                return
            src = _materialize(weights[name_map[name]])
            src = _dequantize_fp8_block_scaled(name_map[name], src, weights)
            fused_a = mla_fused_a_mixers.get(id(param))
            if fused_a is not None:
                q_key = name_map[name].replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                q_weight = _dequantize_fp8_block_scaled(
                    q_key, _materialize(weights[q_key]), weights
                )
                if q_weight.shape != (fused_a.q_lora_rank, fused_a.hidden_size) or src.shape != (
                    fused_a.kv_lora_rank + fused_a.qk_rope_head_dim,
                    fused_a.hidden_size,
                ):
                    raise ValueError(
                        f"{name}: checkpoint Q-A/KV-A shapes do not match MLA dimensions"
                    )
                src = torch.cat((q_weight, src), dim=0)
            if name == "lm_head.weight":
                # LMHead is vocab-sharded (TP column) + gathered; its
                # load_weights shards the full checkpoint tensor.
                self.lm_head.load_weights(weights=[{"weight": src}])
                return
            mla_mixer = mla_kv_b_mixers.get(id(param))
            if mla_mixer is not None:
                h = mla_mixer.num_heads_tp
                n = mla_mixer.qk_nope_head_dim
                v = mla_mixer.v_head_dim
                kv = mla_mixer.kv_lora_rank
                local = mla_mixer.kv_b_proj.load_shard(src, device=param.device).view(h, n + v, kv)
                k_weight, v_weight = local.split([n, v], dim=1)
                param.data.copy_(
                    torch.cat(
                        [
                            k_weight.reshape(h * n, kv),
                            v_weight.reshape(h * v, kv),
                        ],
                        dim=0,
                    ).to(param.dtype)
                )
                mla_mixer.k_b_proj_trans.data.copy_(k_weight.transpose(1, 2))
                # Share the loaded BF16 V rows with context GEMM instead of keeping a
                # separate copy. Wrap the view to retain v_b_proj's parameter registration.
                v_weight = _helix_cp_v_b_shard(
                    param[h * n :].view(h, v, kv),
                    num_heads_tp_cp=mla_mixer.num_heads_tp_cp,
                    cp_rank=mla_mixer.mapping.cp_rank,
                )
                mla_mixer.v_b_proj = nn.Parameter(v_weight, requires_grad=False)
                if _K3_HELIX_V_UP_BEFORE_A2A and mla_mixer.mapping.has_cp_helix():
                    mla_mixer.helix_v_b_proj_full = param[h * n :].view(h, v, kv)
                return
            if name.endswith(".A_log") and src.numel() != param.numel():
                # The checkpoint pads A_log from [num_heads] to [head_dim]
                # (e.g. [96] -> [128]); the tail must be zeros. Under KDA
                # head-shard TP the param holds this rank's head range
                # instead of the full [num_heads].
                assert src.numel() > param.numel(), (name, src.shape)
                if kda_tp_size > 1:
                    lo = kda_tp_rank * param.numel()
                    src = src[lo : lo + param.numel()]
                else:
                    tail = src[param.numel() :]
                    if tail.abs().max().item() != 0.0:
                        raise ValueError(
                            f"{name}: expected zero padding beyond "
                            f"{param.numel()} entries, got nonzero tail"
                        )
                    src = src[: param.numel()]
            if src.shape != param.shape:
                # Delegate MLA q_b/g/o slicing to the same Linear modules that
                # own their COLUMN/ROW sharding policy. KV-B is handled above.
                mla_sharded_linear = mla_head_shard_linears.get(id(param))
                if mla_sharded_linear is not None:
                    shard = mla_sharded_linear.load_shard(src, device=param.device)
                    if shard.shape != param.shape:
                        raise ValueError(
                            f"{name}: MLA shard shape {tuple(shard.shape)} does "
                            f"not match param shape {tuple(param.shape)}"
                        )
                    param.data.copy_(shard.to(param.dtype))
                    return
                # KDA head-shard (attention-DP off): every mismatching KDA
                # tensor is head-major with the checkpoint exactly
                # kda_tp_size times larger on one axis — q/k/v/g/f_b
                # projections, b_proj, dt_bias, and the depthwise conv
                # weights on dim 0 (rows), o_proj on dim 1 (columns).
                # MLA head-sharded projections were handled by parameter
                # identity above, so shape ratios identify the KDA slices.
                if kda_tp_size > 1 and ".linear_attn." in name:
                    if (
                        src.shape[0] == param.shape[0] * kda_tp_size
                        and src.shape[1:] == param.shape[1:]
                    ):
                        s = param.shape[0]
                        lo = kda_tp_rank * s
                        param.data.copy_(src[lo : lo + s].to(param.dtype))
                        return
                    if (
                        src.dim() == 2
                        and src.shape[0] == param.shape[0]
                        and src.shape[1] == param.shape[1] * kda_tp_size
                    ):
                        s = param.shape[1]
                        lo = kda_tp_rank * s
                        param.data.copy_(src[:, lo : lo + s].to(param.dtype))
                        return
                # Shared-expert TP (direct MoE path): the module holds a
                # 1/tp shard of the FFN dim — column shard for gate/up
                # (output rows), row shard for down (input columns).
                if ".shared_experts." in name or ".mlp." in name:
                    # Shared experts (direct MoE path) and the dense L0
                    # MLP (attention-DP off): the fused gate_up_proj is
                    # sliced in its dedicated branch above; here the
                    # unfused halves (if ever configured) and down_proj.
                    if (
                        name.endswith((".gate_proj.weight", ".up_proj.weight"))
                        and src.shape[0] % param.shape[0] == 0
                        and src.shape[1:] == param.shape[1:]
                    ):
                        shard_count = src.shape[0] // param.shape[0]
                        lo = (model_tp_rank % shard_count) * param.shape[0]
                        param.data.copy_(src[lo : lo + param.shape[0]].to(param.dtype))
                        return
                    if (
                        name.endswith(".down_proj.weight")
                        and src.shape[1] % param.shape[1] == 0
                        and src.shape[0] == param.shape[0]
                    ):
                        shard_count = src.shape[1] // param.shape[1]
                        lo = (model_tp_rank % shard_count) * param.shape[1]
                        param.data.copy_(src[:, lo : lo + param.shape[1]].to(param.dtype))
                        return
                raise ValueError(
                    f"{name}: checkpoint shape "
                    f"{tuple(src.shape)} != param shape "
                    f"{tuple(param.shape)}"
                )
            param.data.copy_(src.to(param.dtype))

        param_jobs = [(name, params[name]) for name in name_map]
        run_concurrently(load_param, param_jobs, num_workers=8)

        logger.info(
            f"Kimi K3: loaded {len(mla_mixers)} MLA KV-B projections in grouped runtime layout"
        )
        return len(param_jobs)

    def _load_expert_slices(
        self,
        weights: Dict[str, torch.Tensor],
        expert_jobs: List[Tuple[int, KimiK3MoERuntime, str]],
    ) -> None:
        """Load FP8-resident attention projections and rank-local expert slices."""
        device = next(self.parameters()).device

        # Layouts whose per-expert loader only stages its input must have the
        # staging containers created before any thread runs, and must be
        # finalized once the layer's last slot lands: nothing else on this path
        # calls process_weights_after_loading. Which thread completes a layer is
        # a race, so the completion test and the "already finalized" bookkeeping
        # are one critical section.
        finalize_lock = threading.Lock()
        finalized_backends = set()
        prepared_backends = set()

        def ensure_prepared(moe: KimiK3MoERuntime):
            """Prepare a layer's streaming state on its FIRST expert, not up front.

            Preparing every layer before the load starts is what OOM-ed the
            MegaMoE CuteDSL backend: it keeps its raw NVFP4 source params as
            0-element placeholders and rematerializes them at full shape here,
            so preparing all 92 layers held 92 layers of raw weights at once
            instead of the handful actually being filled. Cutlass did not care
            because its parameters are allocated either way.

            Bounded lazily instead. The paired shrink already happens per layer
            in process_weights_after_loading, so the live set is whatever is
            genuinely in flight -- measured at 1 layer per shard file for this
            checkpoint (its rank-local experts are 1:1 with files), so ~4 with
            4 loader threads. Correctness does not depend on that layout
            though; a checkpoint that split a layer across files would only
            raise the peak, not break this.

            The membership add happens AFTER preparing, so the lock-free fast
            path can only ever be stale in the safe direction.
            """
            spec = moe.expert_ckpt_spec
            if not spec.needs_layer_finalize:
                return
            backend = moe.routed_experts.backend
            if id(backend) in prepared_backends:
                return
            with finalize_lock:
                if id(backend) in prepared_backends:
                    return
                backend.quant_method.prepare_streaming_expert_load(backend)
                prepared_backends.add(id(backend))

        def maybe_finalize_layer(moe: KimiK3MoERuntime):
            spec = moe.expert_ckpt_spec
            if not spec.needs_layer_finalize:
                return
            backend = moe.routed_experts.backend
            with finalize_lock:
                loaded = len(getattr(backend, spec.loaded_slots_attr, ()))
                if loaded != backend.expert_size_per_partition:
                    return
                if id(backend) in finalized_backends:
                    return
                finalized_backends.add(id(backend))
            # Computes the alphas, interleaves the w2 scales, and (MegaMoE)
            # packs into the mega buffers and shrinks the raw source params
            # back to placeholders. Paired with ensure_prepared above, that
            # pairing is what bounds the per-layer footprint. The per-expert
            # drain inside load_streaming_nvfp4_expert separately bounds the
            # Cutlass w3_w1 staging, which is per expert rather than per layer.
            backend.process_weights_after_loading()

        def load_expert(
            moe: KimiK3MoERuntime, base: str, local_slot_id: int, expert_idx: int, get_tensor
        ):
            """Load checkpoint expert ``expert_idx`` into local slot ``local_slot_id``."""
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            # Every route into an expert goes through here (file-grouped,
            # split-file and the shared-dict fallback alike), so this is the
            # one place preparation has to be hooked.
            ensure_prepared(moe)
            moe.expert_ckpt_spec.loader(
                backend,
                base,
                expert_idx,
                moe.local_expert_ids[local_slot_id],
                local_slot_id,
                get_tensor,
            )
            maybe_finalize_layer(moe)

        def load_experts_from_weights(layer_idx: int, moe: KimiK3MoERuntime, base: str):
            del layer_idx
            for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
                load_expert(
                    moe,
                    base,
                    local_slot_id,
                    expert_idx,
                    lambda key: _materialize(weights[key]),
                )

        # ---- backend expert slots: file-grouped streaming ----
        # The shared lazy ``weights`` dict keeps every shard mmapped for the
        # whole load, so pages it touches cannot be dropped until the load
        # ends (fadvise skips mapped pages). The expert slices are ~90 GB of
        # DISTINCT pages per rank — with 4 ranks/node that overruns the job
        # cgroup and OOM-kills the step (observed repeatedly on GB300
        # trays). Instead, group the rank-local expert tensors by shard file
        # and stream each file through a short-lived handle:
        # open -> copy -> close (unmap) -> fadvise(DONTNEED).
        # The lazy loader records the directory it opened; prefer it over
        # ``_name_or_path``, which transformers no longer populates (it is
        # empty on transformers 5.x, which silently sent the whole load down
        # the fallback below and OOM-killed the step).
        ckpt_dir = getattr(weights, "checkpoint_dir", None) or getattr(
            self.model_config.pretrained_config, "_name_or_path", None
        )
        index_path = os.path.join(ckpt_dir or "", "model.safetensors.index.json")
        checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")

        def checkpoint_key(key: str) -> str:
            return f"{checkpoint_prefix}{key}"

        if expert_jobs and ckpt_dir and os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            per_file: Dict[str, list] = {}
            split_file_jobs = []
            for layer_idx, moe, base in expert_jobs:
                del layer_idx
                for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
                    keys = [
                        f"{base}.{expert_idx}.{w}.{kind}"
                        for w in ("w1", "w2", "w3")
                        for kind in moe.expert_ckpt_spec.kinds
                    ]
                    files = {weight_map[checkpoint_key(key)] for key in keys}
                    job = (moe, base, local_slot_id, expert_idx)
                    if len(files) == 1:
                        per_file.setdefault(files.pop(), []).append(job)
                    else:
                        split_file_jobs.append((job, files))

            def drop_file_pages(file_name: str):
                path = os.path.join(ckpt_dir, file_name)
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass

            def load_expert_file(file_name: str, jobs: list):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                path = os.path.join(ckpt_dir, file_name)
                with safe_open(path, framework="pt", device="cpu") as fh:
                    for moe, base, local_slot_id, expert_idx in jobs:
                        load_expert(
                            moe,
                            base,
                            local_slot_id,
                            expert_idx,
                            lambda key: fh.get_tensor(checkpoint_key(key)),
                        )
                # Handle closed -> pages unmapped -> the drop takes effect.
                drop_file_pages(file_name)

            def load_split_file_expert(job, files):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                with ExitStack() as stack:
                    handles = {
                        file_name: stack.enter_context(
                            safe_open(
                                os.path.join(ckpt_dir, file_name), framework="pt", device="cpu"
                            )
                        )
                        for file_name in files
                    }

                    def get_tensor(key):
                        source_key = checkpoint_key(key)
                        return handles[weight_map[source_key]].get_tensor(source_key)

                    load_expert(*job, get_tensor)
                for file_name in files:
                    drop_file_pages(file_name)

            run_concurrently(load_expert_file, sorted(per_file.items()), num_workers=4)
            run_concurrently(load_split_file_expert, split_file_jobs, num_workers=4)
        else:
            # Falling back is a silent loss of the whole point of the block
            # above: the shared lazy dict keeps every shard mapped, which is
            # the OOM this streaming path exists to avoid. Say so.
            if expert_jobs:
                logger.warning(
                    f"Kimi K3: no safetensors index at '{index_path}', so routed "
                    "experts are loaded from the shared lazy weight dict instead "
                    "of being streamed per shard file. Every shard stays mapped "
                    "for the whole load, which OOM-kills the step at DEP8 scale."
                )
            run_concurrently(load_experts_from_weights, expert_jobs, num_workers=4)

        for _, moe, _ in expert_jobs:
            spec = moe.expert_ckpt_spec
            backend = moe.routed_experts.backend
            loaded_slots = getattr(backend, spec.loaded_slots_attr, set())
            expected_slots = set(range(backend.expert_size_per_partition))
            if loaded_slots != expected_slots:
                missing_slots = sorted(expected_slots - loaded_slots)
                raise RuntimeError(
                    "Kimi K3 streaming expert loading did not fill all backend "
                    f"slots; missing {missing_slots[:10]}."
                )
            if (
                spec.needs_layer_finalize
                and expected_slots
                and id(backend) not in finalized_backends
            ):
                # Unreachable via load_expert (the last slot finalizes), so
                # reaching it means the two bookkeeping paths disagree. Guarded
                # on expected_slots because a layer that owns no local slots is
                # never prepared and so is legitimately never finalized.
                raise RuntimeError(
                    "Kimi K3 streaming expert loading filled every slot but "
                    "never finalized the layer."
                )
            backend._weights_transformed = False

    def _finalize_weight_load(self, num_params: int, num_moe_layers: int) -> None:
        """Finalize checkpoint-loaded attention and optional BF16 MLP conversion."""
        for layer in self.model.layers:
            if getattr(layer, "is_kda", False):
                attention = layer.linear_attn
                attention.finalize_decode_weights()
                attention._build_mtp_conv_weights()
            else:
                attention = layer.self_attn.mixer
            for name, linear in attention.named_children():
                if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
                    _swap_linear_to_fp8_weight_read(attention, name, (TrtllmLinear,))
            fused = getattr(attention, "qkvg_proj", None)
            if isinstance(fused, _Fp8BlockScaleWeightReadLinear):
                offset = 0
                for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                    projection = getattr(attention, name)
                    projection._fused_projection = (weakref.ref(fused), offset)
                    offset += projection.out_features
        logger.info(
            f"Kimi K3: loaded {num_params} parameters and expert slices of {num_moe_layers} layers"
        )
        if os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_DENSE_MLP_ENV, "0") == "1" and is_sm_100f():
            converted = _convert_dense_mlps_to_fp8_weight_read(self.model)
            logger.info(f"Kimi K3: quantized {converted} BF16 dense MLP projections to FP8")
        if self._fp8_weight_read_moe_mlp and is_sm_100f():
            shared_tp = os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_SHARED_TP_ENV, "0") == "1"
            converted = _convert_moe_mlps_to_fp8_weight_read(
                self.model,
                include_fused_gate_up=self.model_config.mapping.enable_attention_dp or shared_tp,
                include_tp_shards=shared_tp,
            )
            logger.info(f"Kimi K3: quantized {converted} BF16 shared/latent MLP projections to FP8")
        if _K3_FP8_MID_FAMILIES:
            prepared = sum(
                module.prepare_mid_gemm()
                for module in self.model.modules()
                if isinstance(module, _Fp8BlockScaleWeightReadLinear)
            )
            logger.info(
                f"Kimi K3: mid-M FP8 GEMM for {sorted(_K3_FP8_MID_FAMILIES)} on {prepared} "
                f"projections, M in [{_K3_FP8_MID_MIN_M}, {_K3_FP8_MID_MAX_M}]"
            )

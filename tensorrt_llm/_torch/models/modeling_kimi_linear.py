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
the routed latent partial is reduced before its norm/up projection.
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
import gc
import json
import math
import os
import threading
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
from ..distributed import AllReduce, AllReduceParams
from ..kimi_k3_cache_policy import KIMI_K3_BF16_KV_LAYERS_ENV, get_kimi_k3_bf16_kv_layer_ids
from ..model_config import ModelConfig
from ..modules.fused_moe import ConfigurableMoE, create_moe
from ..modules.fused_moe.interface import MoESchedulerKind
from ..modules.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from ..modules.gated_mlp import GatedMLP
from ..modules.kimi_kda import KimiKDALinearAttention
from ..modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from ..modules.linear import Linear as TrtllmLinear
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.situ import SituAndMul
from ..utils import AuxStreamType
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model, run_concurrently

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
)

# Online FP8 block-scale cast for the replicated MoE-layer MLPs — default ON,
# set KIMI_K3_FP8_WEIGHT_READ_MOE_MLP=0 to keep them BF16.
#
# Targets the shared-expert MLP (gate/up/down) and the latent up/down
# projection of every MoE layer. Under attention data-parallelism each rank
# re-reads these dense replicated weights in full on every decode step, so
# decode is bound by that HBM read and the FP8 (e4m3, 128x128 block-scale)
# copy roughly halves those bytes. It is also a DEP8 *prerequisite*, not a
# tuning knob: without it job 484926 died in forward with 15 MiB free.
#
# Unlike the attention projections, this one is genuinely a cast: these
# tensors are BF16 in nvidia/Kimi-K3-NVFP4 (every FP8 tensor in that
# checkpoint lives under ``self_attn``), so the FP8 copy is quantized online
# here and is LOSSY relative to the checkpoint. The off-switch exists so that
# loss can be attributed if a precision question ever points here; GSM8K
# parity with it on is what licenses the default.
#
# Requires datacenter Blackwell. SM100/103 use DeepGEMM, while SM107 uses the
# native CuTe DSL MXFP8 kernel when available. The routed experts and dense
# layer-0 MLP are never converted. The fused shared-expert ``gate_up_proj`` is
# included only under attention DP — under TP the BF16 GEMM overlaps on the aux
# stream and FP8 quantize+GEMM would serialize onto the critical path.
_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV = "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP"


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
        return DeepSeekV3MoeRoutingMethod(
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            is_fused=True,
        )


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
    if K + 1 > 12 or M > 16384 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    try:
        attn_res_op = torch.ops.trtllm.attn_res_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output, _rsigma, _probs, _logits = attn_res_op(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
    )
    return output.reshape(M, H)


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
    """Add an attention output to the running residual, then select and norm."""
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
# FP8 block-scale weight read for the replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


def _fp8_weight_read_is_available() -> bool:
    """Whether this rank can keep K3's eligible weights resident in FP8."""
    return is_sm_100f()


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Same predicate ``FP8BlockScalesLinearMethod`` uses for the generic
    FP8-block-scale ``Linear``. Recomputed rather than cached: the SM version
    cannot change within a run, so weight preparation and forward always agree.
    """
    from ..cute_dsl_utils import IS_CUTLASS_DSL_INTERNAL_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_INTERNAL_AVAILABLE


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free linear replacement that reads its weight at FP8.

    The BF16 weight ``[out, in]`` is quantized once (at load) to
    ``float8_e4m3fn`` with 128x128 block scales, then served through an FP8
    block-scale GEMM. SM100/103 use DeepGEMM; on Rubin the CuTe DSL kernel uses
    native 2xFP8 MMA and a K32 R128c4 UE8M0 scale layout. Halving the weight
    bytes cuts the dominant HBM read that bounds K3's memory-bound decode step.
    Both ``out`` and ``in`` are multiples of 128 for every projection this is
    applied to, so the block scales cover the weight exactly.

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
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

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
        # If the checkpoint stored this projection as FP8_PB_WO, the loader kept
        # the original pair on the parameter. Use it: re-deriving the block
        # scale from the dequantized BF16 does NOT reproduce the checkpoint's
        # bytes (16-19% differ -- see _K3_CKPT_FP8_ATTR). Only the MoE-layer
        # MLPs, which are BF16 on disk, legitimately fall through to
        # quantize_weight.
        ckpt_pair = getattr(linear.weight, _K3_CKPT_FP8_ATTR, None)
        if ckpt_pair is not None:
            return cls.from_checkpoint_fp8(ckpt_pair[0], ckpt_pair[1], linear.out_features)
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    @classmethod
    def empty_placeholder(
        cls, out_features: int, in_features: int
    ) -> "_Fp8BlockScaleWeightReadLinear":
        """A module that occupies (almost) nothing until the loader fills it.

        This is the contract construction-time FP8 dispatch has to satisfy:
        build one of these instead of an ``nn.Linear`` and the BF16 weight is
        never allocated at all. That matters because the DEP8 failure was an
        OOM during module CONSTRUCTION, before a single weight had been read,
        so any scheme that allocates BF16 and converts afterwards cannot fix
        it however cheap the steady state ends up.

        Same shape of trick MegaMoE uses for its raw NVFP4 params -- and with
        the same hazard, learned there: anything that indexes these before the
        loader has filled them sees an empty tensor. ``load_checkpoint_pair``
        is the only way to populate them, and ``forward`` refuses to run while
        they are still empty rather than producing garbage.
        """
        # Shaped [0, in_features] rather than a flat empty: __init__ reads
        # in_features off the weight, and numel() == 0 still marks it unfilled.
        return cls(
            torch.empty(0, in_features, dtype=torch.float8_e4m3fn),
            torch.empty(0, dtype=torch.uint8),
            out_features,
        )

    @property
    def is_placeholder(self) -> bool:
        return self.weight.numel() == 0

    def load_checkpoint_pair(self, pairs) -> None:
        """Fill a placeholder from one or more checkpoint FP8_PB_WO pairs.

        Several pairs fuse along ``out`` (KDA's q/k/v/g); one is the plain
        case. Rejects a second fill so a double-load is loud rather than
        silently keeping whichever arrived last.
        """
        pairs = list(pairs)
        filled = (
            self.fuse_checkpoint_fp8(pairs)
            if len(pairs) > 1
            else self.from_checkpoint_fp8(pairs[0][0], pairs[0][1], self.out_features)
        )
        self.fill_prepared(filled.weight, filled.weight_scale)

    def fill_prepared(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        """Fill a placeholder from an ALREADY-prepared FP8 weight + scale.

        The single fill path, so the filled-twice guard cannot be bypassed.
        ``load_checkpoint_pair`` delegates here after converting checkpoint
        pairs; the KDA fused path uses it directly to install a *view* of the
        fused weight (its own storage would defeat the point of fusing).

        ``weight_scale`` must already be in the active GEMM's layout from
        ``prepare_checkpoint_scale`` or ``quantize_weight``.
        """
        if not self.is_placeholder:
            raise RuntimeError(
                "Kimi K3 FP8 placeholder was filled twice; the second load would silently win."
            )
        if weight_fp8.shape[0] != self.out_features:
            raise ValueError(
                f"Kimi K3 FP8 placeholder expects out_features="
                f"{self.out_features}, checkpoint gives {weight_fp8.shape[0]}."
            )
        if weight_fp8.dtype is not torch.float8_e4m3fn:
            raise ValueError(
                f"Kimi K3 FP8 placeholder expects a float8_e4m3fn weight, got {weight_fp8.dtype}."
            )
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        self.in_features = weight_fp8.shape[1]

    @classmethod
    def fuse_checkpoint_fp8(cls, pairs) -> "_Fp8BlockScaleWeightReadLinear":
        """Fuse per-projection checkpoint FP8_PB_WO pairs along ``out``.

        ``pairs`` is an ordered sequence of ``(weight_fp8, fp32 block scale)``
        sharing one input dim -- e.g. KDA's q/k/v/g.

        This exists so a fused projection can be built WITHOUT ever
        materializing the BF16 concatenation the current path goes through,
        which is what makes constructing attention as FP8 possible at all.
        It is sound for the reason ``quantize_weight`` documents: with every
        ``out`` a multiple of 128 no 128x128 block crosses a boundary, so the
        fused quantization is per-block identical to the individual ones --
        hence the fused FP8 rows ARE the individual FP8 rows, and likewise
        for the block-scale rows.

        The layout transform is applied ONCE to the assembled scale rather
        than per part: both backend layouts pack against the full ``mn``, so
        transforming the pieces and stacking afterwards would not produce the
        same bytes.
        """
        weights = [w for w, _ in pairs]
        scales = [s for _, s in pairs]
        if any(w.shape[0] % _FP8_BLOCK for w in weights):
            raise ValueError(
                "Kimi K3 fused FP8 read requires every out dim to be a "
                f"multiple of {_FP8_BLOCK}; got "
                f"{[tuple(w.shape) for w in weights]}."
            )
        fused_weight = torch.cat(weights, dim=0)
        fused_scale = torch.cat(scales, dim=0)
        prepared, scale = cls.prepare_checkpoint_scale(fused_weight, fused_scale)
        return cls(prepared, scale, fused_weight.shape[0])

    @classmethod
    def from_checkpoint_fp8(
        cls, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, out_features: int
    ) -> "_Fp8BlockScaleWeightReadLinear":
        """Build directly from a checkpoint's FP8_PB_WO pair, no BF16 detour."""
        prepared, scale = cls.prepare_checkpoint_scale(weight_fp8, weight_scale)
        return cls(prepared, scale, out_features)

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
        if self.is_placeholder:
            # Fail before the empty operands reach the GEMM: this whole path
            # has a habit of turning wrong weights into plausible-looking
            # numbers rather than errors.
            raise RuntimeError(
                "Kimi K3 FP8 placeholder was never filled by the loader; "
                "load_checkpoint_pair must run before forward."
            )
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        if _rubin_mxfp8_is_available():
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                act_fp8,
                self.weight,
                act_scale,
                self.weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        else:
            out = torch.ops.trtllm.fp8_swap_ab_gemm(
                x_flat,
                self.weight,
                self.weight_scale,
                output_dtype=x.dtype,
                disable_ue8m0_cast=True,
            )
        return out.reshape(out_shape)

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project an activation that the fused KDA producer already quantized."""
        if self.is_placeholder:
            raise RuntimeError(
                "Kimi K3 FP8 placeholder was never filled by the loader; "
                "load_checkpoint_pair must run before forward."
            )
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

    Frees the original BF16 weight storage immediately: the loader holds a
    transient name->Parameter map that keeps it alive until load returns, so
    without this the FP8 copy is purely additive and fragments the pool the
    FP8 GEMM autotuner and KV-cache init need.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    setattr(parent, attr, _Fp8BlockScaleWeightReadLinear.from_linear(child))
    child.weight.data = child.weight.data.new_empty(0)
    return 1


def _convert_moe_mlps_to_fp8_weight_read(
    model: nn.Module, include_fused_gate_up: bool = True
) -> int:
    """Swap the replicated MoE-layer MLP projections to an FP8 weight read.

    Targets the shared-expert MLP (gate/up/down) and the latent up/down
    projection on every MoE layer — the bias-free BF16 projections that
    attention data-parallelism re-reads in full each decode step. The routed
    experts and the dense layer-0 MLP are left in BF16. Attention is handled
    elsewhere: KDA is built FP8-resident at construction time, and MLA goes
    through ``_convert_mla_projections_to_fp8_weight_read``.
    Returns the number of projections converted.
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
                    continue
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


def _k3_kda_fp8_resident_jobs(model: nn.Module, prefix: str):
    """Yield ``(layer_idx, mixer, attr, ckpt_key)`` per FP8-resident projection.

    🔴 Single source of truth, and that is the point. These projections hold
    BUFFERS, not Parameters, so they never appear in ``named_parameters()`` and
    are therefore absent from ``name_map`` and ``expected_keys``. Worse, the
    unexpected-key check cannot catch the omission either: ``surprising``
    already filters out everything under ``{prefix}model.layers.``. A projection
    that was never loaded would be **silent** in both directions.

    So ``checkpoint_name_plan`` extends ``expected_keys`` from this generator
    and ``load_weights`` drives the fills from it. One walk, both uses: a
    projection cannot be validated-but-unloaded or loaded-but-unvalidated.
    """
    for layer_idx, layer in enumerate(model.layers):
        if not getattr(layer, "is_kda", False):
            continue
        mixer = getattr(layer, "linear_attn", None)
        if mixer is None or not getattr(mixer, "_fp8_resident_projections", False):
            continue
        for attr in _K3_KDA_FP8_RESIDENT_PROJECTIONS:
            mod = getattr(mixer, attr, None)
            # g_proj is absent in the low-rank-gate config.
            if isinstance(mod, _Fp8BlockScaleWeightReadLinear):
                key = f"{prefix}model.layers.{layer_idx}.self_attn.{attr}.weight"
                yield layer_idx, mixer, attr, key


def _fill_kda_fp8_resident_projections(jobs, weights) -> int:
    """Fill construction-time FP8 placeholders from the checkpoint's FP8 pairs.

    Under KDA TP, q/k/v/g weights are row-sharded and o is column-sharded.
    Their 128x128 block scales follow the same split in scale coordinates,
    matching MLA's ``Linear.load_shard(..., scale_span=128)`` policy.

    Produces one fused ``qkvg_proj`` that is the sole storage, with q/k/v/g
    rebuilt as views of their slice, **without ever materializing BF16**:
    ``fuse_checkpoint_fp8`` concatenates the checkpoint FP8 directly and
    transforms the assembled scale once.

    Returns the number of projections filled.
    """
    by_layer = {}
    for layer_idx, mixer, attr, key in jobs:
        by_layer.setdefault(layer_idx, (mixer, {}))[1][attr] = key

    filled = 0
    for layer_idx, (mixer, attrs) in sorted(by_layer.items()):
        # The projections that read the same normed hidden, in the fused
        # order. o_proj reads the decode-kernel output instead and is filled
        # on its own.
        group = [a for a in ("q_proj", "k_proj", "v_proj", "g_proj") if a in attrs]
        if group:
            pairs = [_k3_kda_checkpoint_fp8_pair(mixer, a, attrs[a], weights) for a in group]
            fused = _Fp8BlockScaleWeightReadLinear.fuse_checkpoint_fp8(pairs)
            mixer.qkvg_proj = fused
            mixer.qkvg_split_sizes = [getattr(mixer, a).out_features for a in group]

            offset = 0
            for a, (w_fp8, w_scale) in zip(group, pairs):
                mod = getattr(mixer, a)
                n = mod.out_features
                # Own block scale, but a VIEW of the fused weight: the fused
                # tensor stays the only storage. Scale must be prepared, not
                # raw -- see fill_prepared.
                _, own_scale = _Fp8BlockScaleWeightReadLinear.prepare_checkpoint_scale(
                    w_fp8, w_scale
                )
                mod.fill_prepared(fused.weight[offset : offset + n], own_scale)
                offset += n
                filled += 1

        if "o_proj" in attrs:
            w_fp8, w_scale = _k3_kda_checkpoint_fp8_pair(mixer, "o_proj", attrs["o_proj"], weights)
            getattr(mixer, "o_proj").load_checkpoint_pair([(w_fp8, w_scale)])
            filled += 1

    return filled


def _convert_mla_projections_to_fp8_weight_read(model: nn.Module) -> int:
    """Swap the MLA q_a/q_b/o and output-gate projections to an FP8 weight read.

    Targets the large bias-free BF16 projections of every MLA (full-attention)
    layer that are read only through their ``forward`` — ``q_a_proj``,
    ``q_b_proj``, ``o_proj`` and, when the output gate is enabled, ``g_proj``,
    each ``[out, in]`` with both dims a multiple of 128 — replicated attention
    weights re-read in full by every rank each decode step under attention
    data-parallelism. Two MLA projections are left in BF16 on purpose:
    ``kv_a_proj_with_mqa`` outputs ``kv_lora_rank + qk_rope_head_dim`` (576, not
    a multiple of 128), and ``kv_b_proj`` supplies ``k_b_proj_trans`` and
    ``v_b_proj`` directly (the absorbed generation path never calls its
    ``forward``), with no FP8 dequant path. Returns the number of projections
    converted.

    Kept when the rest of the load-time FP8 weight read was retired, because
    the reason that retirement rested on does not apply here. KDA went away
    because the FP8-resident path *replaces* it -- q/k/v/g/o are built FP8 at
    construction time and never allocate BF16 at all. MLA has no such
    replacement: ``_k3_attn_stored_fp8_block_scaled`` only ever makes KDA
    resident, so deleting this would simply leave these 96 projections in BF16
    and cost ~2.4 GiB/rank of KV cache at DEP8 (measured: 7.57 -> 5.16 GiB).

    Nor is the BF16 round trip here lossy, which is the other thing that made
    the KDA path worth removing. ``from_linear`` re-derives the block scale
    from the dequantized weight, and
    ``test_bf16_roundtrip_is_identity_so_both_fp8_routes_agree`` asserts
    BITWISE that this lands on the checkpoint's own FP8 bytes:
    ``per_block_cast_to_fp8`` recovers the same block maximum and therefore the
    same codes. That test exists because the opposite was once assumed.
    """
    count = 0

    for layer in model.layers:
        # MLA layers are the non-KDA layers (each layer is exactly one of the
        # two); their projections live on the KimiK3MLAAttention mixer.
        if getattr(layer, "is_kda", False):
            continue
        mixer = getattr(getattr(layer, "self_attn", None), "mixer", None)
        if mixer is None:
            continue
        # g_proj exists only when the MLA output gate is enabled; a missing
        # attr is a safe no-op.
        for attr in ("q_a_proj", "q_b_proj", "o_proj", "g_proj"):
            count += _swap_linear_to_fp8_weight_read(
                mixer, attr, linear_types=(nn.Linear, TrtllmLinear)
            )

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

# The KDA projections eligible to stay FP8 on device. Deliberately only the
# five whose ``out`` and ``in`` are both multiples of 128, so every 128x128
# block scale is whole: b_proj outputs num_heads, and f_a/f_b, g_a/g_b, the
# short convolutions and dt are small and feed the accuracy-sensitive
# recurrent decay. Verified against nvidia/Kimi-K3-NVFP4, which declares
# exactly these (plus b_proj/f_a_proj/f_b_proj, which we leave BF16) as
# FP8_PB_WO under all three key spellings.
_K3_KDA_FP8_RESIDENT_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")


def _k3_attn_stored_fp8_block_scaled(model_config: "ModelConfig", layer_idx: int) -> bool:
    """Does the checkpoint store this layer's KDA projections as FP8_PB_WO?

    FP8 residency follows the checkpoint's stored format. ``moonshotai/Kimi-K3``
    stores attention as plain BF16 and declares no per-layer attention entry,
    so it answers False and keeps the BF16 path.

    Requires ALL five projections to be declared. A partial declaration is
    rejected rather than silently dequantized: the runtime cannot preserve a
    partially FP8 layer without allocating BF16 for the undeclared projections.

    ⚠️ Matches ``FP8_BLOCK_SCALES``, **not** the ``FP8_PB_WO`` spelling the
    checkpoint uses. ``canonicalize_quant_algo`` (Blocker #0) rewrites the
    alias on the way in -- same on-disk layout as the DeepSeek block recipe --
    so ``quant_config_dict`` only ever holds the canonical name. There is no
    ``QuantAlgo.FP8_PB_WO`` to compare against.
    """
    per_layer = getattr(model_config, "quant_config_dict", None)
    if not per_layer:
        return False
    for prefix in _K3_ROUTED_EXPERT_KEY_PREFIXES:
        base = f"{prefix}layers.{layer_idx}.self_attn."
        algos = [
            getattr(per_layer.get(base + p), "quant_algo", None)
            for p in _K3_KDA_FP8_RESIDENT_PROJECTIONS
        ]
        n_declared = sum(a == QuantAlgo.FP8_BLOCK_SCALES for a in algos)
        if n_declared == len(algos):
            return True
        if n_declared:
            raise ValueError(
                f"Kimi K3 layer {layer_idx}: only {n_declared} of {len(algos)} "
                f"KDA projections are declared FP8 block-scaled under prefix "
                f"{prefix!r}; partial FP8 residency is unsupported."
            )
    return False


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


def _load_packed_mxfp4_expert(backend, base, expert_idx, local_slot_id, get_tensor) -> None:
    backend.quant_method.load_packed_mxfp4_expert(
        backend,
        global_expert_id=expert_idx,
        local_slot_id=local_slot_id,
        w1_weight=get_tensor(f"{base}.{expert_idx}.w1.weight_packed"),
        w1_weight_scale=get_tensor(f"{base}.{expert_idx}.w1.weight_scale"),
        w2_weight=get_tensor(f"{base}.{expert_idx}.w2.weight_packed"),
        w2_weight_scale=get_tensor(f"{base}.{expert_idx}.w2.weight_scale"),
        w3_weight=get_tensor(f"{base}.{expert_idx}.w3.weight_packed"),
        w3_weight_scale=get_tensor(f"{base}.{expert_idx}.w3.weight_scale"),
    )


def _load_nvfp4_expert(backend, base, expert_idx, local_slot_id, get_tensor) -> None:
    backend.quant_method.load_streaming_nvfp4_expert(
        backend,
        global_expert_id=expert_idx,
        local_slot_id=local_slot_id,
        **{
            f"{w}_{kind}": get_tensor(f"{base}.{expert_idx}.{w}.{kind}")
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
        self._reduce_routed_output = (
            use_shared_tp
            and self.routed_experts.backend.scheduler_kind != MoESchedulerKind.FUSED_COMM
        )
        if self._reduce_routed_output and self.routed_experts.all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the "
                "ConfigurableMoE all-reduce even when reduce_results=False."
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
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
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

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
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

    def forward(self, hidden_states: torch.Tensor, all_rank_num_tokens=None) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16."""
        identity = hidden_states
        router_logits = self.gate.compute_logits(hidden_states)
        moe_all_reduce = self.routed_experts.all_reduce if self._reduce_routed_output else None

        def _routed_output():
            # Latent down/up projections via the min-latency fused GEMM op:
            # at <=16 tokens (decode graphs) it runs a single pipelined
            # bf16 kernel per projection instead of cuBLAS's split-K GEMV +
            # splitKreduce pair (~17+3.6us -> ~8us for 7168->3584 at M=1);
            # for larger token counts the op falls back to cuBLAS internally.
            # TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ=1 restores nn.Linear
            # (A/B escape hatch). When the FP8 weight-read conversion has
            # replaced the projection module, call it directly: its weight is
            # an e4m3 buffer the bf16 dsv3 op must not read, and its forward
            # is already a quantize + native 2xFP8 GEMM path.
            routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            y = self.routed_experts(
                routed_in,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed dispatch/expert/combine chain. Multi-stream engages
        # only under CUDA graphs; otherwise both branches run in order on the
        # default stream. The shared GatedMLP includes its output all-reduce on
        # the auxiliary stream. The join below must precede the routed
        # all-reduce: concurrent collectives on different streams can corrupt
        # SYMM_MEM all-reduce state.
        routed_out, shared_out = maybe_execute_in_parallel(
            _routed_output,
            lambda: self.shared_experts(identity),
            self.moe_main_event,
            self.moe_shared_event,
            self.shared_expert_stream,
            disable_on_compile=True,
        )
        if self._reduce_routed_output:
            routed_latent = moe_all_reduce(routed_out)
            routed_latent = self.routed_expert_norm(routed_latent)
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        return routed_out + shared_out


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
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
        )

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        out = self.mixer(hidden_states, attn_metadata)
        if self._o_allreduce is not None:
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
            # Preserve the checkpoint's KDA projection datatype. FP8_PB_WO
            # weights use FP8 placeholders so no BF16 storage or
            # dequantize/re-quantize round trip exists.
            fp8_resident = _k3_attn_stored_fp8_block_scaled(model_config, layer_idx)
            if fp8_resident and not _fp8_weight_read_is_available():
                raise NotImplementedError(
                    "Kimi K3 FP8-resident KDA projections require datacenter "
                    "Blackwell; refusing to dequantize checkpoint FP8 weights "
                    "to BF16."
                )
            proj_factory = None
            if fp8_resident:

                def proj_factory(in_features: int, out_features: int) -> nn.Module:
                    return _Fp8BlockScaleWeightReadLinear.empty_placeholder(
                        out_features, in_features
                    )

            self.linear_attn = KimiKDALinearAttention(
                cfg,
                layer_idx,
                mapping=model_config.mapping,
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                fp8_resident_projections=fp8_resident,
                proj_factory=proj_factory,
            )
        else:
            # Forward only the KV-cache quantization to the MLA attention
            # backends. The attention projection
            # weights themselves stay BF16 — the model-level weight-quant
            # algo must not leak into the attention backend's weight paths.
            kv_quant_algo = (
                model_config.quant_config.kv_cache_quant_algo
                if model_config.quant_config is not None
                else None
            )
            bf16_kv_layer_ids = get_kimi_k3_bf16_kv_layer_ids(cfg)
            if layer_idx in bf16_kv_layer_ids:
                if kv_quant_algo != QuantAlgo.NVFP4:
                    raise ValueError(
                        f"{KIMI_K3_BF16_KV_LAYERS_ENV} requires the model-wide "
                        "KV cache dtype to be NVFP4."
                    )
                kv_quant_algo = None
                logger.info_once(
                    f"Kimi K3 MLA layers using BF16 KV-cache fallback: {sorted(bf16_kv_layer_ids)}.",
                    key="kimi_k3_bf16_kv_fallback_layers",
                )
            mla_model_config = copy.copy(model_config)
            mla_model_config.quant_config = (
                QuantConfig(kv_cache_quant_algo=kv_quant_algo)
                if kv_quant_algo is not None
                else QuantConfig()
            )
            self.self_attn = KimiMLARuntime(
                cfg,
                layer_idx,
                model_config=mla_model_config,
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
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            attention_dp = model_config.mapping.enable_attention_dp
            if attention_dp:
                self.mlp_tp_size = 1
            else:
                self.mlp_tp_size = math.gcd(cfg.intermediate_size, model_config.mapping.tp_size)
                if self.mlp_tp_size > model_config.mapping.gpus_per_node:
                    self.mlp_tp_size = math.gcd(
                        self.mlp_tp_size, model_config.mapping.gpus_per_node
                    )
            mlp_model_config = copy.copy(model_config)
            mlp_model_config.quant_config = QuantConfig()
            # K3's dense layer is BF16, so a unit block size gives the same
            # subgroup selection as DeepSeek-V3. Attention DP replicates the
            # MLP because ranks own different tokens; otherwise the subgroup
            # is block-aligned and stays within one node.
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
        number of valid bank rows.

        ``capture`` is ``(spec_metadata, layer_id)``: the attention-side mixture
        computed below IS the aggregated stream value for ``layer_id``, so the
        DSpark tap reuses it instead of recomputing it.
        """
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]

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

        if self.is_kda:
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            hidden_states = self.self_attn(hidden_states, attn_metadata)

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
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
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
# Set on a Parameter by the loader when the checkpoint stored that projection as
# FP8_PB_WO, so the MLA conversion can reuse the checkpoint's own bytes instead
# of re-deriving them from the dequantized BF16.
#
# 🔴 This is NOT an optimization, it is a correctness requirement. Measured on
# nvidia/Kimi-K3-NVFP4 (examples/kimi_k3/check_fp8_roundtrip_identity.py, job 514653): for
# all 10 MLA projections sampled, dequantize->requantize differs from the
# checkpoint's stored FP8 in 16-19% of bytes. The checkpoint was quantized by
# ModelOpt, which does not agree bit-for-bit with `per_block_cast_to_fp8`.
#
# ⚠️ test_bf16_roundtrip_is_identity_so_both_fp8_routes_agree does NOT license
# dropping this. It builds its "checkpoint" pair *with* per_block_cast_to_fp8,
# so it only proves that function is idempotent -- not that the checkpoint
# agrees with it. Deleting this stash on the strength of that test silently
# changed the MLA weights once already.
_K3_CKPT_FP8_ATTR = "_k3_ckpt_fp8_pair"


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
    if scale_key not in weights:
        raise KeyError(
            f"Kimi K3: {ckpt_key} is FP8 E4M3 but has no {scale_key}; refusing "
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


def _k3_kda_checkpoint_fp8_pair(mixer, attr: str, ckpt_key: str, weights):
    """Load this rank's KDA FP8 weight and matching block-scale shard."""
    module = getattr(mixer, attr)
    tp_size = mixer._kda_tp_size
    tp_rank = mixer._kda_tp_rank
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"Kimi K3 KDA TP rank {tp_rank} is invalid for TP size {tp_size}.")

    # Q/K/V/G produce head shards (column parallel); O consumes them (row
    # parallel). Its block-scale grid uses the same split axis divided by 128.
    split_dim = 1 if attr == "o_proj" else 0
    local_shape = (module.out_features, module.in_features)
    if any(dim % _FP8_BLOCK for dim in local_shape):
        raise ValueError(
            f"Kimi K3 {attr} rank-local FP8 shape {local_shape} is not "
            f"{_FP8_BLOCK}x{_FP8_BLOCK} block-aligned."
        )

    full_shape = list(local_shape)
    full_shape[split_dim] *= tp_size
    stored_shape = _value_shape(weights[ckpt_key])
    if stored_shape != tuple(full_shape):
        raise ValueError(
            f"Kimi K3 {ckpt_key} shape {stored_shape} does not match the "
            f"expected TP{tp_size} shape {tuple(full_shape)} for local "
            f"{attr} shape {local_shape}."
        )

    weight_slice = scale_slice = None
    if tp_size > 1:
        shard_width = local_shape[split_dim]
        start = tp_rank * shard_width
        end = start + shard_width
        if start % _FP8_BLOCK or end % _FP8_BLOCK:
            raise ValueError(
                f"Kimi K3 {attr} TP shard [{start}:{end}] is not aligned to "
                f"the FP8 block size {_FP8_BLOCK}."
            )
        weight_indices = [slice(dim) for dim in stored_shape]
        weight_indices[split_dim] = slice(start, end)
        weight_slice = tuple(weight_indices)
        scale_indices = [slice(None), slice(None)]
        scale_indices[split_dim] = slice(start // _FP8_BLOCK, end // _FP8_BLOCK)
        scale_slice = tuple(scale_indices)

    pair = _checkpoint_fp8_pair(
        ckpt_key,
        weights[ckpt_key],
        weights,
        weight_slice=weight_slice,
        scale_slice=scale_slice,
    )
    if pair is None:
        raise ValueError(
            f"Kimi K3: {ckpt_key} was built FP8-resident but the checkpoint "
            f"does not store it as FP8_PB_WO."
        )
    weight, scale = pair
    expected_scale_shape = tuple(dim // _FP8_BLOCK for dim in local_shape)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Kimi K3 {_fp8_block_scale_key(ckpt_key)} shard shape "
            f"{tuple(scale.shape)} does not match {attr} FP8 block grid "
            f"{expected_scale_shape}."
        )
    return weight, scale


def _dequantize_fp8_block_scaled(ckpt_key: str, src: torch.Tensor, weights) -> torch.Tensor:
    """Undo FP8_PB_WO block quantization, returning a BF16-compatible tensor.

    ``nvidia/Kimi-K3-NVFP4`` stores the attention projections as FP8 E4M3
    weights plus a per-128x128-block FP32 scale, where ``moonshotai/Kimi-K3``
    stores plain BF16. **The two are the same shape**, so nothing downstream
    can tell them apart: the shape check passes and ``src.to(param.dtype)``
    happily reinterprets quantized values as if they were the real ones, with
    the scale left behind as an unreferenced checkpoint key. The model then
    loads clean and generates nonsense.

    Still needed when checkpoint metadata selects FP8-resident KDA, because
    that path only takes over the KDA projections. MLA's
    ``q_a``/``q_b``/``o`` load as BF16 parameters through this function -- the
    load-time conversion that re-quantizes them runs afterwards and consumes
    ``linear.weight``, so they have to be materialized first -- and
    ``kv_b_proj`` is consumed directly by ``_kv_b_absorb_split`` (never through
    its ``forward``), so it has no FP8 path at all and stays BF16 for good.
    Dequantizing here keeps both on the same BF16 tensors the MXFP4 checkpoint
    would have supplied.
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

    def _trunk_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Named parameters of the trunk only. Spec-dec draft modules
        (e.g. the DFlash drafter attached by SpecDecOneEngineForCausalLM)
        live in a separate checkpoint loaded by
        ModelLoader.load_draft_weights, not in the target checkpoint. MLA
        K/V absorb Parameters are derived by the KV-B loader and are likewise
        excluded from checkpoint jobs."""
        return {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith("draft_model.")
            and not name.endswith(_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES)
        }

    def checkpoint_name_plan(self, prefix: str):
        """Return ``(name_map, expected_keys, expert_jobs, fp8_attn_jobs)``.

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
            for expert_idx in moe.local_expert_ids:
                for w in ("w1", "w2", "w3"):
                    for kind in kinds:
                        expected_keys.add(f"{base}.{expert_idx}.{w}.{kind}")
            expert_jobs.append((layer_idx, moe, base))

        # FP8-resident attention projections. These hold buffers, so they are
        # absent from ``params`` above; registering their checkpoint keys here
        # is what keeps the strict missing-key check covering them. Built from
        # the same generator ``load_weights`` fills from -- see its docstring
        # for why anything else would fail silently.
        fp8_attn_jobs = list(_k3_kda_fp8_resident_jobs(self.model, prefix))
        for _layer_idx, _mixer, _attr, ckpt_key in fp8_attn_jobs:
            expected_keys.add(ckpt_key)
        return name_map, expected_keys, expert_jobs, fp8_attn_jobs

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
        params = self._trunk_parameters()
        name_map, expected_keys, expert_jobs, fp8_attn_jobs = self.checkpoint_name_plan(prefix)

        self._validate_checkpoint_keys(weights, expected_keys, prefix)
        num_params = self._load_trunk_params(weights, params, name_map)
        self._load_expert_slices(weights, expert_jobs, fp8_attn_jobs)
        self._finalize_weight_load(num_params, len(expert_jobs))
        device = next(self.parameters()).device
        if device.type == "cuda":
            # Lazy source mappings are load-scoped; finish nonblocking H2D
            # work before the caller can release the weights container.
            torch.cuda.synchronize(device)

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
        missing = sorted(expected_keys - ckpt_keys)
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
        # Read here rather than at the conversion site because load_param runs
        # first. See _K3_CKPT_FP8_ATTR for why the pairs are load-bearing.
        stash_ckpt_fp8 = _fp8_weight_read_is_available()

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
            # Keep the checkpoint's FP8 pair for the MLA conversion at the end
            # of this method. Gated like that conversion, so the pairs are not
            # retained for a run that will never use them.
            ckpt_fp8_pair = (
                _checkpoint_fp8_pair(name_map[name], src, weights) if stash_ckpt_fp8 else None
            )
            src = _dequantize_fp8_block_scaled(name_map[name], src, weights)
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
                # Helix: v_b_proj holds this rank's 1/cp post-all-to-all
                # head chunk; kv_b/k_b_proj_trans keep every tp-local head.
                h_cp = mla_mixer.num_heads_tp_cp
                if h_cp != h:
                    lo = mla_mixer.mapping.cp_rank * h_cp
                    v_weight = v_weight[lo : lo + h_cp]
                mla_mixer.v_b_proj.data.copy_(v_weight)
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
            # Only on this path: the branches above shard or pad, and the
            # conversion consumes linear.weight in its module-local shape, so a
            # full-size pair would not match. Under enable_attention_dp the
            # attention projections are replicated, which is exactly this path.
            if ckpt_fp8_pair is not None and ckpt_fp8_pair[0].shape == param.shape:
                setattr(param, _K3_CKPT_FP8_ATTR, ckpt_fp8_pair)

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
        fp8_attn_jobs: list[tuple],
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
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            # Every route into an expert goes through here (file-grouped,
            # split-file and the shared-dict fallback alike), so this is the
            # one place preparation has to be hooked.
            ensure_prepared(moe)
            moe.expert_ckpt_spec.loader(backend, base, expert_idx, local_slot_id, get_tensor)
            maybe_finalize_layer(moe)

        def load_experts_from_weights(layer_idx: int, moe: KimiK3MoERuntime, base: str):
            del layer_idx
            for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
                load_expert(
                    moe,
                    base,
                    local_slot_id,
                    expert_idx,
                    lambda key: _materialize(weights[key]),
                )

        # ---- FP8-resident attention projections ----
        # Not part of param_jobs: these are buffers, so the parameter-driven
        # loader above cannot see them. Filled serially and before the expert
        # streaming, while their (small) source tensors are still cheap to
        # reach; their checkpoint keys were validated with the rest.
        if fp8_attn_jobs:
            n_resident = _fill_kda_fp8_resident_projections(fp8_attn_jobs, weights)
            logger.info(
                f"Kimi K3: filled {n_resident} FP8-resident KDA projections "
                f"from the checkpoint (no BF16 storage was allocated for them)"
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
                for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
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
        """Finalize KDA decode state and FP8 storage after all weights are loaded."""

        # Build the KDA fused projection views and decode kernel constants.
        # This must run after every KDA parameter is loaded and sharded, and
        # after the FP8-resident fills above — those supply the fused FP8
        # ``qkvg_proj``, so the FP8 finalizer only has to fuse the two small
        # BF16 projections ([f_a | b]) and precompute the kernel constants.
        num_kda_fused = 0
        num_kda_fp8_fused = 0
        for layer in self.model.layers:
            if not getattr(layer, "is_kda", False):
                continue
            attn = layer.linear_attn
            if getattr(attn, "_fp8_resident_projections", False):
                attn.finalize_decode_weights_fp8()
                num_kda_fp8_fused += int(attn._bfa_proj_weight is not None)
            else:
                attn.finalize_decode_weights()
                num_kda_fused += int(
                    attn._qkvg_proj_weight is not None and attn._bfa_proj_weight is not None
                )
            # Verify constants must exist before any CUDA graph capture, for both
            # resident-FP8 and BF16 KDA layers.
            attn._build_mtp_conv_weights()
        logger.info(
            f"Kimi K3: loaded {num_params} parameters and the expert "
            f"slices of {num_moe_layers} MoE layers; fused prefill/decode/verify "
            f"projections on {num_kda_fused} BF16 and {num_kda_fp8_fused} FP8 KDA layers"
        )

        # Online FP8 block-scale cast for the replicated MoE-layer MLPs —
        # default on, off with KIMI_K3_FP8_WEIGHT_READ_MOE_MLP=0 (see the env
        # definition near the top of this file). Requires datacenter Blackwell.
        if os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV, "1") == "0":
            logger.info(
                f"Kimi K3: {_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV}=0 — keeping the "
                f"MoE-layer MLP projections in BF16"
            )
        elif not _fp8_weight_read_is_available():
            logger.info(
                "Kimi K3: the MoE-layer MLP FP8 block-scale read requires "
                "datacenter Blackwell; keeping them BF16"
            )
        else:
            n_fp8 = _convert_moe_mlps_to_fp8_weight_read(
                self.model,
                # Only worth it when attention DP makes every rank re-read the
                # fused weight per step; under TP the BF16 GEMM overlaps on the
                # aux stream (see the conversion helper's comment).
                include_fused_gate_up=self.model_config.mapping.enable_attention_dp,
            )
            logger.info(
                f"Kimi K3: reading {n_fp8} MoE-layer MLP projections "
                f"(shared-expert + latent) at FP8 block-scale"
            )
        if _fp8_weight_read_is_available():
            # The MLA q_a/q_b/o and output-gate projections are the remaining
            # replicated attention weight read. Unlike KDA these have no
            # FP8-resident path, so without this conversion they stay BF16 --
            # worth ~2.4 GiB/rank of KV cache at DEP8. No env: the switch was
            # the reviewable part, the conversion is not.
            n_mla = _convert_mla_projections_to_fp8_weight_read(self.model)
            logger.info(f"Kimi K3: reading {n_mla} MLA q_a/q_b/o/g projections at FP8 block-scale")

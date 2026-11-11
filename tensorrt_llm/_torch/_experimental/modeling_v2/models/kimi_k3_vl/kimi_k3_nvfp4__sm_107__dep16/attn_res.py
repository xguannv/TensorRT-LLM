# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), dep16: attention-residual mixing.

Before each sublayer norm a token mixes a bank of earlier sublayer outputs (the HF
reference's ``_apply_attn_res``). The functions here pick the kernel for each mix: the K3 PDL
kernels up to ``_PDL_MAX_M`` tokens (which also fold in a deferred MoE residual update), the
per-token and persistent fused kernels with the RMSNorm fused in, and the FP32 reference outside
the kernels' limits. ``_DeferredMoEResidual`` carries a MoE layer's residual update to the next
mix.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_fwd import (
    attn_res_fwd,
)
from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm.logger import logger

from ..ops.moe.k3_moe_residual_add import k3_moe_residual_add as k3_moe_residual_add_op
from ..ops.norm.k3_attn_res_norm import k3_attn_res_norm
from .layers import RMSNormWeight, rms_norm

# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------

# Fused attention-residual kernels (every Kimi K3 recipe runs the "split" topology):
# * up to _FUSED_ATTN_RES_MAX_TOKENS tokens the per-token C++ kernels (one CTA for N <= 4, one 8-CTA
#   cluster per token for N >= 5): trtllm::attn_res_rmsnorm_fwd / attn_res_add_rmsnorm_fwd;
# * above it the persistent kernel (one CTA per SM looping over tokens),
#   trtllm::attn_res_add_rmsnorm_persistent_fwd, where it implements the shape (H == 7168 and
#   2 <= N <= 9);
# * elsewhere trtllm::attn_res_fwd and the RMSNorm, or the FP32 reference past that op's contract.
# The K3 PDL kernels take precedence up to ``_PDL_MAX_M`` tokens.
_FUSED_ATTN_RES_MAX_TOKENS = 32


def _use_persistent_attn_res(M: int, H: int, N: int) -> bool:
    """Whether the persistent kernel serves this mix: H == 7168, 2 <= N <= 9 and more than
    ``_FUSED_ATTN_RES_MAX_TOKENS`` tokens (its grid is sized by the SM count, not the token
    count)."""
    return H == 7168 and 2 <= N <= 9 and M > _FUSED_ATTN_RES_MAX_TOKENS


# Token limit of trtllm::attn_res_fwd (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNormWeight
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
    attn_res_op = attn_res_fwd
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


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether a fused kernel served the mix, once per shape (a shape gate that rejects
    the call is otherwise silent). Emitted at debug level: it is once per distinct shape, not once
    per process, so it is a diagnostic rather than a summary.
    """
    logger.debug_once(
        f"Kimi K3 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


# The K3 PDL kernels: up to ``_PDL_MAX_M`` tokens, the fused attn_res(+add)
# +RMSNorm runs as the k3_attn_res_norm entry (kernels/k3_attn_res.py):
# a CTA cluster per token that loads the weights and snapshot rows and reduces their statistics
# before griddepcontrol.wait, i.e. under the kernel in front of it. It takes precedence over the
# persistent topology at those token counts. The updated prefix sum is bit-identical; the
# normalized output can differ from the C++ per-token kernels by one BF16 rounding step in a small
# fraction of the elements. The dependent grid is not released early (``early_trigger`` off).
# It also leaves the MoE residual update residual + bf16(routed + shared) of the BF16-add path
# (kimi_k3_moe_residual) to the next layer's pre-attention mix, which computes it bit-identically
# (k3_attn_res_moe_add_rmsnorm) behind the latent-up GEMM.
_PDL_MAX_M = 64


class _DeferredMoEResidual(NamedTuple):
    """A MoE layer's ``residual + bf16(routed + shared)`` that the next layer fuses or materializes."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor

    def materialize(self) -> torch.Tensor:
        return k3_moe_residual_add_op(self.routed, self.shared, self.residual)


def _defer_moe_residual(num_tokens: int) -> bool:
    """Whether a MoE layer leaves its BF16 residual update to the next layer."""
    return num_tokens <= _PDL_MAX_M


def _materialize_prefix_sum(hidden_states):
    if isinstance(hidden_states, _DeferredMoEResidual):
        return hidden_states.materialize()
    return hidden_states


def _k3_attn_res_pdl(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
):
    """``k3_attn_res_norm`` (with ``addend`` as the one addend) when applicable, else ``None``."""
    if prefix_sum.shape[0] > _PDL_MAX_M:
        return None
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    addends = () if addend is None else (addend,)
    if not k3_attn_res.can_use_k3_attn_res(prefix_sum, addends, block_residual, *weights):
        return None
    eps = float(norm.eps)
    out_eps = float(output_norm.eps)
    return k3_attn_res_norm(
        prefix_sum,
        addends,
        block_residual,
        *weights,
        eps,
        out_eps,
        early_trigger=False,
    )


def _k3_attn_res_pdl_moe_add(
    deferred: _DeferredMoEResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum, normed mix)`` of a deferred MoE residual update, or ``None``."""
    residual, routed, shared = deferred
    if residual.shape[0] > _PDL_MAX_M:
        return None
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (routed, shared), block_residual, *weights):
        return None
    M, H = residual.shape
    _note_attn_res_fusion("moe_add+attn_res+norm/pdl", True, M, H, int(block_residual.shape[0]) + 1)
    return k3_attn_res_norm(
        residual,
        (routed, shared),
        block_residual,
        *weights,
        float(norm.eps),
        float(output_norm.eps),
        early_trigger=False,
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
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
    # The fused path is taken for M <= _FUSED_ATTN_RES_MAX_TOKENS, H == 7168 and
    # N <= 12, which is the measured window; larger token counts have not been
    # measured and fall back to the unfused add + attn_res_fwd + RMSNorm path.
    pdl_out = _k3_attn_res_pdl(prefix_sum, None, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        _, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix_sum.reshape(M, 1, H).contiguous(),
            None,
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            float(output_norm.eps),
            sm_reserve(),
        )
        _note_attn_res_fusion("attn_res+norm/persistent", True, M, H, N)
        return output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output = torch.ops.trtllm.attn_res_rmsnorm_fwd(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        float(output_norm.eps),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fuse ``prefix_sum + addend``, attention-residual, and trailing norm.

    The production residual add produces a BF16 tensor that remains live across
    the following MLP. The kernel therefore returns that materialized,
    BF16-rounded prefix sum alongside the normalized attention-residual output,
    while avoiding a separate add launch and a re-read of the intermediate by
    attention-residual selection.
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
        updated_prefix_sum, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix_sum.reshape(M, 1, H).contiguous(),
            addend.reshape(M, 1, H).contiguous(),
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            float(output_norm.eps),
            sm_reserve(),
        )
        _note_attn_res_fusion("add+attn_res+norm/persistent", True, M, H, N)
        return updated_prefix_sum.reshape(M, H), output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("add+attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    addend_kernel = addend.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    updated_prefix_sum, output = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        layer_kernel,
        addend_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        float(output_norm.eps),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNormWeight
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Inputs fitting the fused kernel's contract dispatch directly to the in-tree
    ``trtllm::attn_res_fwd`` op. Only the fallback boundary restores the HF
    ``[M, K, H]`` layout.
    """
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
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm."""
    fused = _apply_attn_res_rmsnorm_fused(prefix_sum, block_residual, proj, norm, output_norm)
    if fused is not None:
        return fused
    mixed = _apply_attn_res(prefix_sum, block_residual, proj, norm)
    return rms_norm(mixed, output_norm.weight, output_norm.eps)


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm."""
    fused = _apply_attn_res_add_rmsnorm_fused(
        prefix_sum, addend, block_residual, proj, norm, output_norm
    )
    if fused is not None:
        return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm
    )

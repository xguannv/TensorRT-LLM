# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe DSL MLA decodes with programmatic dependent launch.

``trtllm::k3_mla_decode_fp8_pdl`` (the split read's, ``TRTLLM_K3LL_MLA_DECODE_PDL``) is
``trtllm::cute_dsl_mla_decode_fp8_blackwell`` and ``trtllm::k3_mla_decode_fp16_pdl`` (the MLA DSpark drafter's bf16
block decode, ``TRTLLM_K3LL_DRAFTER_PDL``) is ``trtllm::cute_dsl_mla_decode_fp16_blackwell`` (each: same runner, same
tuner entry and so the same tactic, same arguments) with the runner's ``use_pdl``: the split kernel and the split-KV
reduction launch with programmatic dependent launch, so each one's prologue overlaps the tail of the kernel before it
and each waits for that kernel (``griddepcontrol.wait``) before touching global memory. The arithmetic is the same
kernel's.
"""

from typing import Optional

import torch


def runner(
    num_heads: int,
    seq_len_q: int,
    page_size: int,
    max_batch_size: int,
    stats: bool,
    use_pdl: bool,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    """The decode runner for queries of ``dtype`` (FP8, FP16 or BF16)."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    in_dtype = {
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[dtype]
    return CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=in_dtype,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=max_batch_size,
        emit_softmax_stats=stats,
        use_pdl=use_pdl,
    )


def tactic(run, inputs, op: str = "trtllm::cute_dsl_mla_decode_fp8_blackwell"):
    """The tactic the stock ``op`` picks for these inputs (the same tuner entry)."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    _, best = AutoTuner.get().choose_one(op, [run], run.get_tuning_config(), inputs)
    if not (isinstance(best, tuple) and len(best) == 4):
        best = run.default_tactic(int(inputs[0].shape[-1]))
    return best


@torch.library.custom_op(
    "trtllm::k3_mla_decode_fp8_pdl",
    mutates_args=("o", "workspace", "softmax_stats"),
    device_types="cuda",
)
def mla_decode_fp8_pdl(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    c_latent: torch.Tensor,
    c_rope: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqs: torch.Tensor,
    o: torch.Tensor,
    workspace: torch.Tensor,
    num_heads: int,
    seq_len_q: int,
    page_size: int,
    softmax_scale: float,
    output_scale: float,
    max_batch_size: int,
    softmax_stats: Optional[torch.Tensor],
    kv_bounds: Optional[torch.Tensor],
) -> None:
    """``trtllm::cute_dsl_mla_decode_fp8_blackwell`` (same arguments) with programmatic dependent launch."""
    run = runner(num_heads, seq_len_q, page_size, max_batch_size, softmax_stats is not None, True)
    inputs = [q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, o, workspace, softmax_stats,
              kv_bounds]  # fmt: skip
    run(inputs, tactic=tactic(run, inputs), softmax_scale=softmax_scale, output_scale=output_scale)


@mla_decode_fp8_pdl.register_fake
def _(
    q_latent,
    q_rope,
    c_latent,
    c_rope,
    page_table,
    cache_seqs,
    o,
    workspace,
    num_heads,
    seq_len_q,
    page_size,
    softmax_scale,
    output_scale,
    max_batch_size,
    softmax_stats,
    kv_bounds,
) -> None:
    return None


@torch.library.custom_op(
    "trtllm::k3_mla_decode_fp16_pdl",
    mutates_args=("o", "workspace", "softmax_stats"),
    device_types="cuda",
)
def mla_decode_fp16_pdl(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    c_latent: torch.Tensor,
    c_rope: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqs: torch.Tensor,
    o: torch.Tensor,
    workspace: torch.Tensor,
    num_heads: int,
    seq_len_q: int,
    page_size: int,
    softmax_scale: float,
    output_scale: float,
    max_batch_size: int,
    softmax_stats: Optional[torch.Tensor],
    kv_bounds: Optional[torch.Tensor],
) -> None:
    """``trtllm::cute_dsl_mla_decode_fp16_blackwell`` (same arguments; FP16 or BF16, q, KV and output of one dtype)
    with programmatic dependent launch."""
    dtype = q_latent.dtype
    if dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != dtype for t in (q_rope, c_latent, c_rope, o)
    ):
        raise ValueError(
            "trtllm::k3_mla_decode_fp16_pdl requires FP16 or BF16 q, KV and output of one dtype, got "
            f"{[t.dtype for t in (q_latent, q_rope, c_latent, c_rope, o)]}"
        )
    run = runner(
        num_heads, seq_len_q, page_size, max_batch_size, softmax_stats is not None, True, dtype
    )
    inputs = [q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, o, workspace, softmax_stats,
              kv_bounds]  # fmt: skip
    best = tactic(run, inputs, "trtllm::cute_dsl_mla_decode_fp16_blackwell")
    run(inputs, tactic=best, softmax_scale=softmax_scale, output_scale=output_scale)


@mla_decode_fp16_pdl.register_fake
def _(
    q_latent,
    q_rope,
    c_latent,
    c_rope,
    page_table,
    cache_seqs,
    o,
    workspace,
    num_heads,
    seq_len_q,
    page_size,
    softmax_scale,
    output_scale,
    max_batch_size,
    softmax_stats,
    kv_bounds,
) -> None:
    return None

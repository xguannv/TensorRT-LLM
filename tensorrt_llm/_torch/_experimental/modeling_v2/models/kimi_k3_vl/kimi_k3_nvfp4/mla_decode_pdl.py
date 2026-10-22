# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The split read's CuTe DSL FP8 MLA decode with programmatic dependent launch (``TRTLLM_K3LL_MLA_DECODE_PDL``).

``trtllm::k3_mla_decode_fp8_pdl`` is ``trtllm::cute_dsl_mla_decode_fp8_blackwell`` (same runner, same tuner entry and
so the same tactic, same arguments) with the runner's ``use_pdl``: the split kernel and the split-KV reduction launch
with programmatic dependent launch, so each one's prologue overlaps the tail of the kernel before it and each waits
for that kernel (``griddepcontrol.wait``) before touching global memory. The arithmetic is the same kernel's.
"""

from typing import Optional

import torch


def runner(
    num_heads: int, seq_len_q: int, page_size: int, max_batch_size: int, stats: bool, use_pdl: bool
):
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=max_batch_size,
        emit_softmax_stats=stats,
        use_pdl=use_pdl,
    )


def tactic(run, inputs):
    """The tactic ``trtllm::cute_dsl_mla_decode_fp8_blackwell`` picks for these inputs (the same tuner entry)."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    _, best = AutoTuner.get().choose_one(
        "trtllm::cute_dsl_mla_decode_fp8_blackwell", [run], run.get_tuning_config(), inputs
    )
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

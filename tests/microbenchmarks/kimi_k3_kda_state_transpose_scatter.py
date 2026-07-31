# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for Kimi K3 KDA final-state pool writes."""

import argparse
import time
from collections.abc import Callable

import torch

from tensorrt_llm._torch.modules.kimi_kda._kda_state import kda_state_transpose_scatter


def _time_cuda(operation: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=96)
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_seqs > args.num_slots:
        raise ValueError("--num-seqs cannot exceed --num-slots")

    torch.manual_seed(0)
    source = torch.randn(
        args.num_seqs,
        args.num_heads,
        args.head_k_dim,
        args.head_v_dim,
        dtype=torch.float32,
        device="cuda",
    )
    pool = torch.empty(
        args.num_slots,
        args.num_heads,
        args.head_v_dim,
        args.head_k_dim,
        dtype=torch.float32,
        device="cuda",
    )
    slot_indices = torch.arange(
        args.num_slots - 1,
        args.num_slots - 1 - args.num_seqs,
        -1,
        dtype=torch.int64,
        device="cuda",
    )

    def baseline() -> None:
        pool.index_copy_(
            0,
            slot_indices,
            source.transpose(-1, -2).contiguous(),
        )

    def fused() -> None:
        kda_state_transpose_scatter(source, pool, slot_indices)

    baseline()
    expected = pool.index_select(0, slot_indices).clone()
    pool.fill_(float("nan"))
    fused()
    torch.testing.assert_close(
        pool.index_select(0, slot_indices),
        expected,
        rtol=0,
        atol=0,
    )

    baseline_ms = _time_cuda(baseline, args.warmup, args.iterations)
    fused_ms = _time_cuda(fused, args.warmup, args.iterations)
    payload_mib = source.numel() * source.element_size() / (1024**2)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"shape: N={args.num_seqs} H={args.num_heads} K={args.head_k_dim} V={args.head_v_dim}")
    print(f"state payload: {payload_mib:.2f} MiB")
    print(f"transpose.contiguous + index_copy_: {baseline_ms * 1e3:.2f} us")
    print(f"transpose_scatter:                 {fused_ms * 1e3:.2f} us")
    print(f"speedup:                           {baseline_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    started_at = time.perf_counter()
    main()
    print(f"wall time: {time.perf_counter() - started_at:.2f} s")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark Kimi K3 attention-residual + trailing RMSNorm fusion.

The production split path is:

    trtllm::attn_res_fwd -> FlashInfer RMSNorm

The fused path is:

    trtllm::attn_res_rmsnorm_fwd

CUDA-graph replay timings remove Python and dispatcher overhead.  ``--profile``
instead emits eager calls inside NVTX ranges so Nsys can attribute every
kernel to a shape and mode.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm

HIDDEN_SIZE = 7168
RMS_EPS = 1e-6


@dataclass
class CaseInputs:
    layer_residual: torch.Tensor
    block_residual: torch.Tensor
    res_weight: torch.Tensor
    score_rms_weight: torch.Tensor
    output_rms_weight: torch.Tensor


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        token_count, candidate_count = item.lower().split("x", maxsplit=1)
        shape = (int(token_count), int(candidate_count))
        if shape[0] < 1 or not 1 <= shape[1] <= 12:
            raise argparse.ArgumentTypeError(
                f"invalid shape {item!r}: expected T>=1 and 1<=N<=12")
        shapes.append(shape)
    return shapes


def _default_iterations(num_tokens: int) -> int:
    if num_tokens <= 1:
        return 2000
    if num_tokens <= 16:
        return 1000
    if num_tokens <= 64:
        return 500
    if num_tokens <= 1024:
        return 100
    return 20


def _make_inputs(num_tokens: int, num_candidates: int) -> CaseInputs:
    device = torch.device("cuda")
    shape = (num_tokens, 1, HIDDEN_SIZE)
    layer_residual = torch.empty(shape, dtype=torch.bfloat16, device=device)
    block_residual = torch.empty(
        (num_candidates - 1, *shape),
        dtype=torch.bfloat16,
        device=device,
    )
    layer_residual.uniform_(-0.05, 0.05)
    block_residual.uniform_(-0.05, 0.05)
    res_weight = torch.empty(
        HIDDEN_SIZE, dtype=torch.bfloat16, device=device).uniform_(-0.02, 0.02)
    score_rms_weight = torch.empty(
        HIDDEN_SIZE, dtype=torch.bfloat16, device=device).uniform_(0.98, 1.02)
    output_rms_weight = torch.empty(
        HIDDEN_SIZE, dtype=torch.bfloat16, device=device).uniform_(0.98, 1.02)
    return CaseInputs(
        layer_residual=layer_residual,
        block_residual=block_residual,
        res_weight=res_weight,
        score_rms_weight=score_rms_weight,
        output_rms_weight=output_rms_weight,
    )


def _attn_res(inputs: CaseInputs) -> torch.Tensor:
    output, _rsigma, _probs, _logits = torch.ops.trtllm.attn_res_fwd(
        inputs.layer_residual,
        inputs.block_residual,
        inputs.res_weight,
        inputs.score_rms_weight,
        RMS_EPS,
    )
    return output


def _rms_norm(
    hidden_states: torch.Tensor,
    output_rms_weight: torch.Tensor,
) -> torch.Tensor:
    return flashinfer_rmsnorm(hidden_states, output_rms_weight, RMS_EPS)


def _split(inputs: CaseInputs) -> torch.Tensor:
    return _rms_norm(_attn_res(inputs), inputs.output_rms_weight)


def _fused(inputs: CaseInputs) -> torch.Tensor:
    return torch.ops.trtllm.attn_res_rmsnorm_fwd(
        inputs.layer_residual,
        inputs.block_residual,
        inputs.res_weight,
        inputs.score_rms_weight,
        inputs.output_rms_weight,
        RMS_EPS,
        RMS_EPS,
    )


def _capture(fn: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _time_graph(
    fn: Callable[[], torch.Tensor],
    iterations: int,
    samples: int,
) -> tuple[float, float, float]:
    graph, output = _capture(fn)
    del output
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations)
    return (
        statistics.median(timings),
        min(timings),
        max(timings),
    )


def _similarity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float,
        expected_float,
        dim=0,
    ).item()
    relative_l2 = (
        (actual_float - expected_float).norm()
        / (expected_float.norm() + 1e-12)
    ).item()
    return cosine, relative_l2


def _benchmark_case(
    num_tokens: int,
    num_candidates: int,
    iterations_override: int | None,
    samples: int,
) -> dict[str, float | int | str]:
    inputs = _make_inputs(num_tokens, num_candidates)
    split_output = _split(inputs)
    fused_output = _fused(inputs)
    torch.cuda.synchronize()
    cosine, relative_l2 = _similarity(fused_output, split_output)
    if cosine <= 0.9999 or relative_l2 >= 5e-3:
        raise AssertionError(
            f"T={num_tokens}, N={num_candidates}: cosine={cosine}, "
            f"relative_l2={relative_l2}")

    iterations = iterations_override or _default_iterations(num_tokens)
    attn_us, attn_min_us, attn_max_us = _time_graph(
        lambda: _attn_res(inputs), iterations, samples)
    rms_us, rms_min_us, rms_max_us = _time_graph(
        lambda: _rms_norm(split_output, inputs.output_rms_weight),
        iterations,
        samples,
    )
    split_us, split_min_us, split_max_us = _time_graph(
        lambda: _split(inputs), iterations, samples)
    fused_us, fused_min_us, fused_max_us = _time_graph(
        lambda: _fused(inputs), iterations, samples)

    return {
        "num_tokens": num_tokens,
        "num_candidates": num_candidates,
        "iterations": iterations,
        "samples": samples,
        "cosine": cosine,
        "relative_l2": relative_l2,
        "attn_us": attn_us,
        "attn_min_us": attn_min_us,
        "attn_max_us": attn_max_us,
        "rms_us": rms_us,
        "rms_min_us": rms_min_us,
        "rms_max_us": rms_max_us,
        "split_us": split_us,
        "split_min_us": split_min_us,
        "split_max_us": split_max_us,
        "fused_us": fused_us,
        "fused_min_us": fused_min_us,
        "fused_max_us": fused_max_us,
        "fused_vs_split_pct": (fused_us / split_us - 1.0) * 100.0,
        "saved_us": split_us - fused_us,
    }


def _profile_case(
    num_tokens: int,
    num_candidates: int,
    iterations_override: int | None,
) -> None:
    inputs = _make_inputs(num_tokens, num_candidates)
    split_output = _attn_res(inputs)
    modes: Sequence[tuple[str, Callable[[], torch.Tensor]]] = (
        ("attn", lambda: _attn_res(inputs)),
        (
            "rms",
            lambda: _rms_norm(split_output, inputs.output_rms_weight),
        ),
        ("split", lambda: _split(inputs)),
        ("fused", lambda: _fused(inputs)),
    )
    iterations = iterations_override or min(_default_iterations(num_tokens), 100)
    for _name, fn in modes:
        for _ in range(10):
            fn()
    torch.cuda.synchronize()

    for name, fn in modes:
        range_name = f"attn_res|T={num_tokens}|N={num_candidates}|mode={name}"
        torch.cuda.nvtx.range_push(range_name)
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(
            json.dumps(
                {
                    "profile_range": range_name,
                    "iterations": iterations,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=_parse_shapes,
        default=_parse_shapes(
            "1x2,1x3,1x4,1x5,1x6,1x7,1x8,1x9,"
            "16x2,16x3,16x4,16x5,16x6,16x7,16x8,16x9,"
            "1024x2,1024x5,1024x9,8192x2,8192x5,8192x9"),
        help="Comma-separated T-by-N pairs, where N is candidate count.",
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit eager kernels in NVTX ranges for Nsys instead of timing.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError(f"SM100/SM103 is required, got {capability}")
    torch.manual_seed(0)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": capability,
                "profile": args.profile,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for num_tokens, num_candidates in args.shapes:
        if args.profile:
            _profile_case(num_tokens, num_candidates, args.iterations)
        else:
            result = _benchmark_case(
                num_tokens,
                num_candidates,
                args.iterations,
                args.samples,
            )
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

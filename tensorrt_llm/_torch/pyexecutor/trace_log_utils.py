# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated trace/log utilities for pyexecutor.

Leaf module — no other pyexecutor file is imported here, so any consumer
(``_util``, ``model_engine``, ``model_loader``, ``resource_manager``)
can import freely without creating circular dependencies.
"""

import os

import torch

from tensorrt_llm.logger import logger

from ..mem_probe import SnapshotContext, log_snapshot


def log_mem_snapshot(tag: str) -> None:
    """Preserve the existing warmup API while using the unified snapshot."""

    log_snapshot(tag, context=SnapshotContext(rank=logger.rank))


def log_tensor_size(tag: str, tensor: torch.Tensor, **extra) -> None:
    """Log a single tensor's footprint (shape / dtype / bytes) at a tag.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).

    Bytes = ``numel * element_size``. Any keyword arguments are appended
    as ``key=value`` for caller-specific context (e.g. routing config).
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    size_bytes = tensor.numel() * tensor.element_size()
    extras = "".join(f" {k}={v}" for k, v in extra.items())
    logger.info(
        f"[mem-profile/{tag}] "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"size={size_bytes / 1024 / 1024:.2f}MiB{extras}"
    )

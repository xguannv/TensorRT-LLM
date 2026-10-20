# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path census of the Kimi K3 decode path (switch ``TRTLLM_K3LL_CENSUS``).

Counts, per model forward that Python executes (an eager step or a CUDA graph capture; graph replays repeat the
captured path and run no Python):

- ``step:<kind>:<mode>``: the step's classification, ``kind`` in ``generic`` (``decode_step`` returned None: a step
  with context requests or too many rows), ``decode`` (a pure decode step of R requests x T tokens) or ``small``
  (a non-decode step of at most ``DECODE_MAX_TOKENS`` rows); ``mode`` in ``eager`` / ``capture``;
- ``fallback:<Class>:<mode>``: a module's ``forward_generic`` called inside a step ``decode_step`` classified as
  decode or small, i.e. a module that fell back to the generic path where the decode kernels were meant to run.

A line ``Kimi K3 census: ...`` is logged on rank 0 whenever a new key appears or a count reaches a power of two,
which bounds the log at O(log steps) lines per key. The last such line holds the totals.
"""

import functools
from collections import Counter
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from . import flags as _flags

ENABLED = bool(_flags.TRTLLM_K3LL_CENSUS)

_counts: Counter = Counter()
_decode_depth = 0
_installed = set()


def _mode() -> str:
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    return "capture" if capturing else "eager"


def _bump(key: str) -> None:
    _counts[key] += 1
    n = _counts[key]
    if n & (n - 1) == 0:
        rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if rank0:
            logger.info(
                "Kimi K3 census: " + " ".join(f"{k}={v}" for k, v in sorted(_counts.items()))
            )


def step(kind: Optional[str]) -> None:
    """Record one model forward's classification (``None`` = generic)."""
    if ENABLED:
        _bump(f"step:{kind or 'generic'}:{_mode()}")


class decode_scope:
    """Marks the body of a decode-classified forward, inside which a ``forward_generic`` call is a fallback."""

    def __enter__(self):
        global _decode_depth
        _decode_depth += 1

    def __exit__(self, *exc):
        global _decode_depth
        _decode_depth -= 1
        return False


def install(*classes) -> None:
    """Wrap each class's ``forward_generic`` to count calls made inside a decode-classified forward."""
    if not ENABLED:
        return
    for cls in classes:
        if cls in _installed or not hasattr(cls, "forward_generic"):
            continue
        original = cls.forward_generic

        @functools.wraps(original)
        def counted(self, *args, __original=original, __name=cls.__name__, **kwargs):
            if _decode_depth > 0:
                _bump(f"fallback:{__name}:{_mode()}")
            return __original(self, *args, **kwargs)

        cls.forward_generic = counted
        _installed.add(cls)


def install_model(model_cls, decode_step) -> None:
    """Wrap ``model_cls.forward`` (``forward(attn_metadata, input_ids, position_ids, inputs_embeds, ...)``) to record
    the step's classification by ``decode_step`` and open a ``decode_scope`` around a decode-classified forward."""
    if not ENABLED or model_cls in _installed:
        return
    original = model_cls.forward

    @functools.wraps(original)
    def counted(
        self, attn_metadata, input_ids=None, position_ids=None, inputs_embeds=None, *args, **kwargs
    ):
        rows = input_ids if inputs_embeds is None else inputs_embeds
        shape = decode_step(attn_metadata, rows.shape[0]) if rows is not None else None
        step(None if shape is None else ("decode" if shape.decode else "small"))
        if shape is None:
            return original(
                self, attn_metadata, input_ids, position_ids, inputs_embeds, *args, **kwargs
            )
        with decode_scope():
            return original(
                self, attn_metadata, input_ids, position_ids, inputs_embeds, *args, **kwargs
            )

    model_cls.forward = counted
    _installed.add(model_cls)


def counts() -> dict:
    return dict(_counts)

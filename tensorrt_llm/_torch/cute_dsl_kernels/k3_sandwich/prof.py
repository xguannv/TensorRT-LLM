# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic profile of ``k3_sandwich_tail`` (prototype; ``TRTLLM_K3LL_SANDWICH_PROF=1``, read at every call).

The kernel writes (globaltimer, clock64) pairs per CTA at its marks (``k3_sandwich_kernel.PROF_MARKS``) into one
slice of a device buffer per call position (calls mod ``PROF_LAYERS`` per forward; one buffer per token count). A
daemon thread copies every buffer to ``$TRTLLM_K3LL_SANDWICH_PROF_DIR/sandwich_prof_rank<R>.pt`` every 2 s on its
own stream. The runner captures its graphs in global capture mode from another thread, so the dump thread runs in
relaxed capture mode and copies only once no slice has been handed out for 15 s (no warm-up or capture running); each
token count records the host time of its last slice hand-out, so a reader keeps graph replays only.
"""

from __future__ import annotations

import os
import threading
import time

import torch

PROF_LAYERS = 92
_prof = dict(bufs={}, calls=0, stamps={}, thread=None, last_call=0.0, dummy=None)


def enabled() -> bool:
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
            flags,
        )
    except ImportError:  # pragma: no cover
        return False
    return bool(getattr(flags, "TRTLLM_K3LL_SANDWICH_PROF", 0))


def _dump_loop(device) -> None:
    import cuda.bindings.driver as cuda_driver

    from tensorrt_llm._utils import prefer_pinned

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")))
    path = os.path.join(
        os.environ.get("TRTLLM_K3LL_SANDWICH_PROF_DIR", "/tmp"), f"sandwich_prof_rank{rank}.pt"
    )
    torch.cuda.set_device(device)
    cuda_driver.cuThreadExchangeStreamCaptureMode(
        cuda_driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
    )
    stream = torch.cuda.Stream(device=device)
    hosts = {}
    while True:
        time.sleep(2.0)
        if time.time() - _prof["last_call"] < 15.0:
            continue
        for key, buf in list(_prof["bufs"].items()):
            if key not in hosts:
                hosts[key] = torch.empty(buf.shape, dtype=torch.int64, pin_memory=prefer_pinned())
            with torch.cuda.stream(stream):
                hosts[key].copy_(buf, non_blocking=True)
        stream.synchronize()
        torch.save(
            dict(
                bufs={("sandtail", k): v.clone() for k, v in hosts.items()},
                calls={"sandtail": _prof["calls"]},
                stamps={("sandtail", k): v for k, v in _prof["stamps"].items()},
                host_ns=time.time_ns(),
            ),
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)


def arg(device, num_tokens: int, numel: int) -> torch.Tensor:
    """The profile slice of this call (``numel`` int64 words), or a dummy when the profile is off."""
    _prof["last_call"] = (
        time.time()
    )  # a Python forward (warm-up / capture) is running: the dump thread waits
    if not enabled():
        if _prof["dummy"] is None or _prof["dummy"].device != device:
            _prof["dummy"] = torch.zeros(2, dtype=torch.int64, device=device)
        return _prof["dummy"]
    buf = _prof["bufs"].get(num_tokens)
    if buf is None:
        buf = _prof["bufs"][num_tokens] = torch.zeros(
            PROF_LAYERS, numel, dtype=torch.int64, device=device
        )
    if _prof["thread"] is None:
        _prof["thread"] = threading.Thread(target=_dump_loop, args=(device,), daemon=True)
        _prof["thread"].start()
    i = _prof["calls"]
    _prof["calls"] = i + 1
    _prof["stamps"][num_tokens] = time.time_ns()  # marks older than this are warm-up / pre-capture
    return buf[i % PROF_LAYERS]

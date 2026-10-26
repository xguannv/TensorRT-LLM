# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-situ MoE front phase timeline (diagnostic measurement build, never for trunk).

``TRTLLM_K3LL_FRONT_TL_DIR=<dir>`` at process start: every decode front call passes ``k3_moe_front``
a timeline slice of one device buffer, keyed by (MoE layer, token build 8 / 16 / 32), so each
captured call of a CUDA graph writes its own slice on every replay. The kernel writes %globaltimer
at its phase boundaries per CTA (``k3_moe_front.TL_SLOTS`` words per CTA; slot meanings there). A
daemon thread snapshots the whole buffer on its own stream (relaxed capture mode) every
``TRTLLM_K3LL_FRONT_TL_PERIOD`` seconds (default 1.0) once no Python front call has run for 15 s
(warm-up and capture done; graph replays only), keeps the last ``TRTLLM_K3LL_FRONT_TL_SNAPS``
(default 3: ~50 MB each, and after the replay the snapshots repeat) and writes them to
``<dir>/fronttl_rank<R>.pt`` (with the slot index and every key's geometry); an atexit hook writes a
last copy. A snapshot can tear across a step: a reader checks each slice's stamps for one call
(monotone slots, span < 100 us).
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from typing import Optional

import torch

_ENV = "TRTLLM_K3LL_FRONT_TL_DIR"
MAX_LAYERS = 128
BUILDS = (8, 16, 32)
_st = dict(
    buf=None,
    words=0,
    device=None,
    thread=None,
    last_call=0.0,
    keys={},
    snaps=[],
    lock=threading.Lock(),
)


def enabled() -> bool:
    return bool(os.environ.get(_ENV))


def view(layer_idx: int, rows: int, device: torch.device) -> Optional[torch.Tensor]:
    """This (layer, build)'s zeroed timeline slice, or None when off."""
    if not enabled() or layer_idx < 0 or layer_idx >= MAX_LAYERS:
        return None
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op

    build = front_op.m_build(rows)
    with _st["lock"]:
        if _st["buf"] is None:
            _st["words"] = max(front_op.timeline_words(b) for b in BUILDS)
            _st["buf"] = torch.zeros(
                MAX_LAYERS * len(BUILDS) * _st["words"], dtype=torch.int64, device=device
            )
            _st["device"] = device
            _st["thread"] = threading.Thread(target=_loop, daemon=True)
            _st["thread"].start()
            atexit.register(_save)
        slot = layer_idx * len(BUILDS) + BUILDS.index(build)
        _st["keys"][slot] = (layer_idx, build, rows)
        _st["last_call"] = time.time()
        w = _st["words"]
        return _st["buf"][slot * w : (slot + 1) * w]


def _path() -> str:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")))
    return os.path.join(os.environ[_ENV], f"fronttl_rank{rank}.pt")


def _snap(stream) -> None:
    with torch.cuda.stream(stream):
        cpu = _st["buf"].to("cpu", non_blocking=True)
    stream.synchronize()
    _st["snaps"].append((time.time_ns(), cpu))
    del _st["snaps"][: -int(os.environ.get("TRTLLM_K3LL_FRONT_TL_SNAPS", "3"))]


def _save() -> None:
    if _st["buf"] is None or not _st["snaps"]:
        return
    path = _path()
    torch.save(
        dict(snaps=_st["snaps"], words=_st["words"], builds=BUILDS, keys=dict(_st["keys"])),
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


def _loop() -> None:
    import cuda.bindings.driver as cuda_driver

    torch.cuda.set_device(_st["device"])
    cuda_driver.cuThreadExchangeStreamCaptureMode(
        cuda_driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
    )
    stream = torch.cuda.Stream(device=_st["device"])
    period = float(os.environ.get("TRTLLM_K3LL_FRONT_TL_PERIOD", "1.0"))
    while True:
        time.sleep(period)
        if time.time() - _st["last_call"] < 15.0:
            continue
        _snap(stream)
        if len(_st["snaps"]) % 10 == 0:
            _save()

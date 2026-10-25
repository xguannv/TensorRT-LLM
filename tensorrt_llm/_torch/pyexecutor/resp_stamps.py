# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wall-clock stamps of a request's first and final response at each hop of the serving path.

``TRTLLM_K3LL_RESP_STAMPS=<directory>`` (unset, empty or ``0``: off, and every call below returns at once) makes each
process append one JSON line per request and hop to ``<directory>/<hop>_<host>_<pid>.jsonl``::

    {"hop": "E", "key": 1768..., "first_ns": ..., "final_ns": ..., "n": 7}

``first_ns`` / ``final_ns`` are ``time.time_ns()`` when the hop handled the request's first and final response, ``n``
how many it handled. Hops:

* ``E``: the executor enqueues a response (``PyExecutor._enqueue_responses``, the rank that enqueues); key = the
  executor request id (the disaggregated request id where the request carries one), ``cid`` = the client id;
* ``R``: the OpenAI server receives a response output of a completion request (after postprocessing), key = the
  disaggregated request id, ``cid`` = the client-side request id;
* ``G``: the OpenAI server yields a streamed completion chunk (its last one is ``data: [DONE]``), same key;
* ``S``: the disaggregated server yields a streamed chunk to its client; key = the disaggregated request id, and
  ``xrid`` = the client's ``X-Request-ID`` header.

Within one request a hop's ``final_ns - first_ns`` is a difference on one host's clock, so the time a hop adds between
the first and the final response (the next hop's span minus this hop's) needs no clock alignment across hosts.
Nothing else changes: the stamps only read clocks and write files.
"""

import json
import os
import socket
import threading
import time

ENV = "TRTLLM_K3LL_RESP_STAMPS"
_DIR = os.environ.get(ENV, "").strip()
ENABLED = _DIR not in ("", "0")

_lock = threading.Lock()
_files = {}
_open = {}


def _file(hop: str):
    f = _files.get(hop)
    if f is None:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"{hop}_{socket.gethostname()}_{os.getpid()}.jsonl")
        f = _files[hop] = open(path, "a", buffering=1)
    return f


def mark(hop: str, key, final: bool, **extra) -> None:
    """Record that ``hop`` handled a response of request ``key`` now; ``final``: its last one. ``extra``: other ids
    of the request (kept from the call that last passed them)."""
    if not ENABLED or key is None:
        return
    now = time.time_ns()
    with _lock:
        rec = _open.get((hop, key))
        if rec is None:
            rec = _open[(hop, key)] = {"hop": hop, "key": key, "first_ns": now, "n": 0}
        rec["n"] += 1
        rec.update((k, v) for k, v in extra.items() if v is not None)
        if final:
            rec["final_ns"] = now
            del _open[(hop, key)]
            _file(hop).write(json.dumps(rec) + "\n")

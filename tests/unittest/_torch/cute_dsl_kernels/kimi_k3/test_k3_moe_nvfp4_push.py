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
"""The push build of trtllm::k3_moe_nvfp4 (K3MoeNvfp4Layer.push): each rank's routed partial over NVFP4 experts stored
into a latent exchange for trtllm::k3_latent_reduce instead of returned. One process per GPU over the TP group of this
run (4 on one GB300 tray). Every rank holds the same random NVFP4 experts (test_k3_moe_nvfp4's, the TP16 route-A rank
layout) and routes its own tokens, so the ranks' partials differ.

Per token count (1, 3, 8) and exchange call-count start (0 and 2^31 - 2, so the int32 count wraps), against the plain
call on the same inputs, whose partial must be nonzero (test_k3_moe_push's checks):
  exact    the partial pushed into the group's exchange (one slot per rank): the reduce equals MNNVLAllReduce's one-shot
           of the plain partials bit for bit, on every rank and run to run (three push + reduce pairs); afterwards the
           exchange is empty, its call count advanced and the arrival word 0;
  exact16  the same into a 16-slot exchange, every rank's partial in 4 slots (TP16's receive side on one tray).

A kernel (the latent's quantization) runs between any two launches of the layer, as in the model.

Run one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_moe_nvfp4_push.py
"""

import importlib.util
import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = (
        mod  # test_k3_moe_push registers itself with cloudpickle by its sys.modules entry
    )
    spec.loader.exec_module(mod)
    return mod


P = _load(
    "test_k3_moe_push"
)  # the MPI context, the exchange, the one-shot reference, the result rows
N = _load("test_k3_moe_nvfp4")  # the NVFP4 experts, tokens and latent quantization

TOKENS = (1, 3, 8)


def check_push(ctx):
    rows = []
    _, _, scales = N._experts()
    layer = N._new_layer(False)
    copies16 = 16 // ctx.world
    for start in P.COUNT_STARTS:
        ex, ex16 = P._Exchange(ctx, ctx.world, start), P._Exchange(ctx, 16, start)
        for m in TOKENS:
            x, ids, w = N._tokens("random", m, seed=300 + 17 * ctx.rank)
            x_fp4, x_sf = N._quant_x(x, scales)
            y = layer(x_fp4, x_sf, ids, w, N.OFFSET).clone()
            ref = P._allreduce(ctx, y)
            got = []
            for _ in range(3):
                x_fp4, x_sf = N._quant_x(x, scales)
                layer.push(x_fp4, x_sf, ids, w, N.OFFSET, ex, ctx.rank)
                got.append(ex.reduce(m).clone())
            state = ex.state_ok(ctx)
            partials = [t.cuda() for t in ctx.comm.allgather(y.cpu())]
            for c in range(copies16):
                x_fp4, x_sf = N._quant_x(x, scales)
                layer.push(x_fp4, x_sf, ids, w, N.OFFSET, ex16, ctx.rank * copies16 + c)
            got16 = ex16.reduce(m).clone()
            state16 = ex16.state_ok(ctx)
            rows.append(P._row(ctx, "K3MoeNvfp4Layer.push", f"count{start}", m, y, ref, got, state, partials,
                               copies16, got16, state16))  # fmt: skip
    return rows


def main() -> int:
    try:
        ctx = P._context()
        from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op_nvfp4  # noqa: F401

        with torch.inference_mode():
            rows = check_push(ctx)
    except Exception:
        traceback.print_exc()
        raise
    ok = all(r["ok"] for r in rows)
    if ctx.rank == 0:
        P._report(rows)
        print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

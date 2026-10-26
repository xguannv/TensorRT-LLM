# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE-path switches read at forward time, flipped after warm-up and first run inside a CUDA-graph capture, at W
ranks (default 4) with the TP16 per-rank shapes.

    K3LL_WARM_UP_STATE=on CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_lowm_warm_up_capture_op_matrix.py

A switch read at forward time can change after load (the gates flip switches at runtime and recapture), so every
build any of its values selects must have been compiled by ``K3DecodeMoeNvfp4Layer.warm_up``, whatever the switch
said then: a build met first under capture raises in its op (it may not compile there). This proves it the only way
that cannot be fooled by an earlier eager call, in a fresh process: build and warm up the layer with the switches at
one state, flip them all, and capture the layer's decode forward at every row count before any eager call at the new
state. One process per state (``K3LL_WARM_UP_STATE``):

* ``on``: warm-up with ``SWITCHES`` all on (``TRTLLM_K3LL_MOE_LOWM_FC12`` included, so ``K3MoeFc12`` is built);
  capture with all off.
* ``off``: warm-up with ``SWITCHES`` off except ``TRTLLM_K3LL_MOE_LOWM_FC12`` (its state is built at load only when
  on; a 0 -> 1 flip without it never engages, ``fc12_takes``); capture with all on.

Each capture is then replayed and checked bit for bit against an eager call at the flipped state (one-hot routing,
so the routed partials do not depend on the reds' order), and the census shows which switched paths the capture
took: at the ``off`` state's capture every switch's dispatch key counts, at the ``on`` state's none does.

Built from ``_kimi_k3_lowm_switches_op_matrix``'s stand-ins (the same layer, routing and payloads); only the layer
with the production load-time switches (front sort, 16- and 32-token fronts) is built.
"""

import os
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kimi_k3_lowm_switches_op_matrix as base  # noqa: E402
import _lockstep as ls  # noqa: E402

DEADLINE_S = 1500
STATE_ENV = "K3LL_WARM_UP_STATE"
SWITCHES = (
    "TRTLLM_K3LL_MOE_SELF_SORT",
    "TRTLLM_K3LL_MOE_SELF_SORT_FC12",
    "TRTLLM_K3LL_MOE_LOWM_DYN_FAST",
    "TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK",
    "TRTLLM_K3LL_MOE_LOWM_FC12",
)
#: the census key each switch's path counts when it engages (``decode_moe_lowm.py``)
DISPATCH = {
    "TRTLLM_K3LL_MOE_SELF_SORT": "dispatch:moe_self_sort:capture",
    "TRTLLM_K3LL_MOE_SELF_SORT_FC12": "dispatch:moe_self_sort_fc12:capture",
    "TRTLLM_K3LL_MOE_LOWM_DYN_FAST": "dispatch:moe_lowm_dyn_fast:capture",
    "TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK": "dispatch:moe_fc2_ceil_chunk:capture",
    "TRTLLM_K3LL_MOE_LOWM_FC12": "dispatch:moe_lowm_fc12:capture",
}
#: 1-16 rows: the single expert kernel (its own sort at most 16 rows); 17-32: FC1 + FC2 or the 32-token single kernel
ROWS = (1, 8, 9, 16, 17, 24, 32)

CENSUS = None
STATE = None
LAYER = None
MOE = None


def _values(state: str):
    on = state == "on"
    values = {name: int(on) for name in SWITCHES}
    values["TRTLLM_K3LL_MOE_LOWM_FC12"] = 1
    return values


def _flipped(state: str):
    on = state == "on"
    return {name: int(not on) for name in SWITCHES}


def _set(values):
    for name, value in values.items():
        setattr(base.FLAGS, name, int(value))


def _tensors(out):
    if isinstance(out, torch.Tensor):
        return [out]
    return [v for v in out if isinstance(v, torch.Tensor)]


def check_build_at_state():
    """The layer built and warmed up as ``_build_decode_moe_lowm`` does, with ``SWITCHES`` at this process's state."""
    global LAYER, MOE
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy

    if base.R.world not in (4, 8, 16):
        raise AssertionError(
            f"the MoE front and the MNNVL ops run 4, 8 or 16 ranks, not {base.R.world}"
        )
    _set(_values(STATE))
    all_reduce = AllReduce(
        mapping=base.R.mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
    )
    backend = base._backend()
    moe = base._moe_layer(21, base.LATENT, backend, all_reduce)
    base.DC.use_decode_one_shot(nn.ModuleList([moe, all_reduce]))
    comm = base.DC.K3DecodeComm.create(base.R.mapping, None)
    base.DM.fold_latent_norm(moe)
    LAYER = base._build_lowm(moe, comm, front_sort=1, front16=1, front32=1)
    MOE = moe
    assert (LAYER.state.fc12 is not None) == bool(base.FLAGS.TRTLLM_K3LL_MOE_LOWM_FC12)


def check_capture_after_flip():
    """Every row count's decode forward first run inside a capture with ``SWITCHES`` flipped, then replayed and
    compared with an eager call at the flipped state; the capture's census shows the flipped paths. Before the flip,
    each row count runs once eagerly at the load state, as served steps would (it also sets up what any capture
    needs, e.g. the cuBLAS handle), compiling nothing the flipped state selects."""
    with torch.inference_mode():
        for rows in ROWS:
            LAYER.forward(MOE, base._moe_input(4000 + rows, rows), None, True)
    torch.cuda.synchronize()
    _set(_flipped(STATE))
    graphs = []
    before = dict(CENSUS.counts())
    with torch.inference_mode():
        for rows in ROWS:
            x = base._moe_input(4100 + rows, rows)
            graph, out = base._capture(lambda x=x: LAYER.forward(MOE, x, None, True))
            graphs.append((rows, x, graph, _tensors(out)))
        after = dict(CENSUS.counts())
        took = {name: after.get(key, 0) - before.get(key, 0) for name, key in DISPATCH.items()}
        if STATE == "off":
            missing = [name for name, n in took.items() if n == 0]
            assert not missing, f"the flipped-on capture did not take {missing} (census {took})"
        else:
            taken = [name for name, n in took.items() if n]
            assert not taken, f"the flipped-off capture still took {taken} (census {took})"
        for rows, x, graph, got in graphs:
            for rep in range(base.REPLAYS):
                x.copy_(base._moe_input(4200 + 10 * rows + rep, rows))
                base._replay(graph)
                want = _tensors(LAYER.forward(MOE, x, None, True))
                assert len(got) == len(want), (rows, len(got), len(want))
                for i, (g, w) in enumerate(zip(got, want)):
                    base._expect_same(
                        f"state {STATE} -> flipped, {rows} rows, replay {rep}, output {i}", g, w
                    )
            del graph
    if base.R.rank == 0:
        print(
            f"[rank 0] warm-up at {STATE}, flipped and first run under capture at {ROWS} rows: replays = eager "
            f"(census {took})",
            flush=True,
        )


def _run_one_rank(args) -> int:
    global CENSUS, STATE
    import importlib

    STATE = os.environ.get(STATE_ENV, "")
    assert STATE in ("on", "off"), f"set {STATE_ENV}=on or off"
    base.R = ls.Rank(args)
    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    base.IMPL = importlib.import_module(f"{package}.implementation")
    base.DC = importlib.import_module(f"{package}.decode_comm")
    base.DM = importlib.import_module(f"{package}.decode_moe")
    base.DML = importlib.import_module(f"{package}.decode_moe_lowm")
    base.DG = importlib.import_module(f"{package}.decode_gemv")
    base.FLAGS = importlib.import_module(f"{package}.flags")
    base.LOWM_OP = importlib.import_module(
        "tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.nvfp4_lowm_op"
    )
    CENSUS = importlib.import_module(f"{package}.census")
    CENSUS.ENABLED = True
    code = ls.run_checks(base.R, [check_build_at_state])
    if code:
        return code
    return ls.run_checks(base.R, [check_capture_after_flip])


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

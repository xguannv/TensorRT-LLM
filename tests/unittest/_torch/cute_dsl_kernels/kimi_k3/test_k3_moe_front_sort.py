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
"""trtllm::k3_moe_front with ``sort_out`` (the NVFP4 latent): the routing's moe_sort over the local experts, written
by the front's last route CTA, one process per GPU over the TP group of this run, at every M in 1..8, over one
K3MoeHeadWorkspace, for this rank's 224 experts (TP4 x EP4) and for all 896.

Per call: the routing, latent and shared activation the bits of the call without the sort; the tile count, each
tile's local expert and mn limit and its rows (expanded indices) those of trtllm::moe_sort on the front's routing
with 128-row tiles (the rows as a set, and in token order: the order the front writes); the fp32 weights the bf16
weights widened; the counter word zero again.

Run under pytest (a pool of 4 MPI workers) or directly, one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_moe_front_sort.py
"""

import pickle
import sys
import traceback

import pytest
import torch

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the test is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _trtllm():
    """``torch.ops.trtllm``, named only in a nested function (cloudpickle cannot pickle a function whose own code
    names ``torch.ops``)."""

    def namespace():
        return torch.ops.trtllm

    return namespace()


WORLD = 4
HIDDEN, LATENT, EXPERTS, TOP_K = 7168, 3584, 896, 16
SHARED_INTER = 6144
GATE_CAP, LINEAR_CAP = 4.0, 25.0
RSF = 2.827
E_LOCAL = 224
TILE = 128
M_ALL = list(range(1, 17))  # 9-16: the 16-token build, on its own workspace


def _supported() -> bool:
    if MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not _supported(), reason=f"needs {WORLD} sm_100 / sm_103 GPUs with MNNVL and mpi4py"
    ),
]


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _context():
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op as moe_op
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    wl, we, inter = LATENT // world, EXPERTS // world, SHARED_INTER // world
    gen = torch.Generator(device="cuda").manual_seed(13)  # the same on every rank
    bias = (torch.randn(EXPERTS, device="cuda", generator=gen) * 0.05).float()
    x8 = torch.randn(8, HIDDEN, device="cuda", generator=gen).bfloat16()
    x8 = torch.cat([x8, torch.randn(8, HIDDEN, device="cuda", generator=gen).bfloat16()])  # 16 rows
    wgen = torch.Generator(device="cuda").manual_seed(131 + rank)
    head = (torch.randn(wl + we, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    head[wl:] *= 8.0  # router rows: logits of a few units
    gate_up = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    ws = moe_op.K3MoeHeadWorkspace.create(mapping, fabric_handle=True)
    ws16 = moe_op.K3MoeHeadWorkspace.create(mapping, fabric_handle=True, m_max=16)
    return dict(ws16=ws16,
        comm=comm, rank=rank, world=world, inter=inter, bias=bias, x8=x8, front=front_op.front_weight(head, gate_up),
        ws=ws, gsf=torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda"),
        ctr=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )  # fmt: skip


def _front(ctx, x, sort_out=None, offset=0, num_local=0):
    ws = ctx["ws16"] if x.shape[0] > 8 else ctx["ws"]
    return _trtllm().k3_moe_front(
        x, ctx["front"], ctx["bias"], RSF, ctx["inter"], GATE_CAP, LINEAR_CAP, ws.uc, ws.mc, ws.flags, ws.rank,
        ctx["world"], nvfp4_gsf=ctx["gsf"], sort_out=sort_out, sort_ctr=ctx["ctr"] if sort_out is not None else None,
        sort_offset=offset, sort_local=num_local,
    )  # fmt: skip


def _ordered_reference(ids, offset, num_local):
    """The tiles in expert order, each tile's rows the selecting tokens' expanded indices in token order."""
    flat = ids.view(-1).tolist()
    by_expert = {}
    for i, e in enumerate(flat):
        if offset <= e < offset + num_local:
            by_expert.setdefault(e - offset, []).append(i)
    return [(e, by_expert[e]) for e in sorted(by_expert)]


def check_sort(ctx):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op

    results = []
    for offset, num_local in (((ctx["rank"] % 4) * E_LOCAL, E_LOCAL), (0, EXPERTS)):
        for m in M_ALL:
            x = ctx["x8"][:m].contiguous()
            plain = _front(ctx, x)
            words = front_op.sort_layout(m, num_local)[3]
            sort_out = torch.full((words,), -7, dtype=torch.int32, device="cuda")
            out = _front(ctx, x, sort_out, offset, num_local)
            torch.cuda.synchronize()
            group, mn_limit, perm, n_tiles, scales = front_op.sort_views(sort_out, m, num_local)
            ids, weights = out[0], out[1]
            r_group, r_limit, _, r_perm, _, r_tiles = _trtllm().moe_sort(
                token_selected_experts=ids, token_final_scales=weights.float(), num_experts=EXPERTS, top_k=TOP_K,
                local_expert_offset=offset, local_num_experts=num_local, tile_tokens_dim=TILE,
            )  # fmt: skip
            nt, r_nt = int(n_tiles.item()), int(r_tiles.item())
            rows_as_set = rows_ordered = nt == r_nt
            ordered = _ordered_reference(ids, offset, num_local)
            rows_ordered = rows_ordered and nt == len(ordered)
            if rows_as_set:
                for t in range(nt):
                    lim, r_lim = int(mn_limit[t]) - t * TILE, int(r_limit[t]) - t * TILE
                    got = perm[t * TILE : t * TILE + lim].tolist()
                    ref = r_perm[t * TILE : t * TILE + r_lim].tolist()
                    rows_as_set = rows_as_set and lim == r_lim and sorted(got) == sorted(ref)
                    if rows_ordered:
                        rows_ordered = (int(group[t]), got) == ordered[t]
            row = dict(
                op="k3_moe_front_sort", case=f"tp{ctx['world']}_local{num_local}", M=m, tiles=nt, moe_sort_tiles=r_nt,
                front_bits=all(_same(a, b) for a, b in zip(out, plain)),
                experts_equal=nt == r_nt and torch.equal(group[:nt], r_group[:nt]),
                limits_equal=nt == r_nt and torch.equal(mn_limit[:nt], r_limit[:nt]),
                rows_as_set=rows_as_set, rows_token_order=rows_ordered,
                weights_fp32=_same(scales, weights.float()), ctr_zero=int(ctx["ctr"].item()) == 0,
            )  # fmt: skip
            good = all(row[k] for k in ("front_bits", "experts_equal", "limits_equal", "rows_as_set",
                                        "rows_token_order", "weights_fp32", "ctr_zero"))  # fmt: skip
            row["rank"], row["good"] = ctx["rank"], bool(good)
            row["ok"] = all(ctx["comm"].allgather(bool(good)))
            results.append(row)
    return results


def _run_checks(_names=None):
    try:
        ctx = _context()
        with torch.inference_mode():
            return check_sort(ctx)
    except Exception:
        traceback.print_exc()
        raise


def _report(per_rank):
    rows = list(per_rank[0]) + [row for rows in per_rank[1:] for row in rows if not row["good"]]
    for row in rows:
        fields = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("op", "case", "M"))
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_k3_moe_front_sort(mpi_pool_executor):
    per_rank = list(mpi_pool_executor.map(_run_checks, [None] * WORLD))
    _report(per_rank)
    assert all(row["ok"] for rows in per_rank for row in rows)


def main() -> int:
    rows = _run_checks()
    per_rank = MPI.COMM_WORLD.gather(rows, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        _report(per_rank)
        print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

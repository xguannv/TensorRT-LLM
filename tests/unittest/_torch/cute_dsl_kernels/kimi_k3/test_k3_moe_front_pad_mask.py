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
"""trtllm::k3_moe_front's sort with ``pad_mask`` (``TRTLLM_K3LL_MOE_PAD_MASK``): the tokens of the requests the mask
marks as CUDA-graph padding are left out of the sort, one process per GPU over a TP4 group, on the 8-, 16- and 32-token
builds (384 shared columns per rank, Kimi K3 TP16's width, which all three builds take at TP4).

Per case (M rows, T tokens per request, the first R real requests, the rest padding; padding last, as the CUDA graph
runner appends it):

* the front's routing, latent and shared activation the bits of the call without the sort or the mask;
* the sorted routing that of trtllm::moe_sort on the real tokens alone (tile count, each tile's local expert and mn
  limit, its rows as a set and in token order), so no tile and no row comes from a padding token;
* trtllm::k3_moe_nvfp4_lowm on the masked sort, per routing slot with one non-zero weight per token (so no bf16 sum
  order enters): the real rows bit for bit the kernel on the real rows alone (nvfp4_lowm_op.run), the padding rows
  +0.0; with all the weights the real rows within the bf16 accumulation bound of the exact sum and the padding rows
  +0.0 (no expert touched them, the kernel zeroed them);
* the counter words zero again.

Run under pytest (a pool of 4 MPI workers) or directly, one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_moe_front_pad_mask.py
"""

import pickle
import sys
import traceback
from types import SimpleNamespace

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
SHARED = 384  # shared columns per rank
GATE_CAP, LINEAR_CAP = 4.0, 25.0
RSF = 2.827
I_TP, E_LOCAL = 768, 224  # one rank of the routed experts' TP4 x EP4
SF16 = 16
TILE = 128
NVFP4_GSF = 448.0 * 6.0 / 8.0
# (M rows, T tokens per request, real requests): MTP7 graphs of 1 / 2 / 4 requests (8 / 16 / 32 rows) with every
# padding a real step can have, the no-MTP graph of 4 requests, every request real, and every request padding.
CASES = [(8, 8, 1), (16, 8, 2), (16, 8, 1), (32, 8, 4), (32, 8, 3), (32, 8, 2), (32, 8, 1), (32, 8, 0),
         (4, 1, 3), (8, 1, 5), (24, 8, 2)]  # fmt: skip


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


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _quantize(t):
    gsf = t.abs().max().float() / (448 * 6)
    q, sf = _trtllm().fp4_quantize(t, 1 / gsf, SF16, False)
    return q, sf, gsf


def _nvfp4_experts(num_local, hidden, inter_size, seed):
    """Random NVFP4 experts in the CuteDSL backend's buffer shapes (as test_k3_moe_lowm.py builds them)."""
    torch.manual_seed(seed)
    w1 = torch.randn(num_local, 2 * inter_size, hidden, device="cuda", dtype=torch.bfloat16)
    w1, w1_sf, w1_gsf = _quantize(w1)
    w1_sf = w1_sf.view(num_local, 2 * inter_size, hidden // SF16)
    w2 = torch.randn(num_local, hidden, inter_size, device="cuda", dtype=torch.bfloat16)
    w2, w2_sf, w2_gsf = _quantize(w2)
    w2_sf = w2_sf.view(num_local, hidden, inter_size // SF16)
    fc2_input_scale = torch.tensor([448 * 6 / 4.0], dtype=torch.float32, device="cuda")
    spread = torch.linspace(0.5, 1.5, num_local, device="cuda")
    return SimpleNamespace(
        w1=w1.view(torch.uint8), w1_sf=w1_sf.view(torch.uint8), w2=w2.view(torch.uint8), w2_sf=w2_sf.view(torch.uint8),
        w1_gsf=w1_gsf, alpha2=(spread * (w2_gsf / fc2_input_scale[0])).float(), spread=spread,
        fc2_input_scale=fc2_input_scale,
    )  # fmt: skip


def _context():
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op, nvfp4_lowm_op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op as moe_op
    from tensorrt_llm.mapping import Mapping

    device = torch.device("cuda", torch.cuda.current_device())
    for m_max in (8, 16, 32):
        assert front_op.weight_supported(world, SHARED, HIDDEN, device, m_max=m_max), m_max
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    wl, we = LATENT // world, EXPERTS // world
    gen = torch.Generator(device="cuda").manual_seed(29)  # the same on every rank
    bias = (torch.randn(EXPERTS, device="cuda", generator=gen) * 0.05).float()
    x32 = torch.randn(32, HIDDEN, device="cuda", generator=gen).bfloat16()
    wgen = torch.Generator(device="cuda").manual_seed(291 + rank)
    head = (torch.randn(wl + we, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    head[wl:] *= 8.0  # router rows: logits of a few units
    gate_up = (torch.randn(2 * SHARED, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    gsf = torch.tensor([NVFP4_GSF], dtype=torch.float32, device="cuda")
    ex = _nvfp4_experts(E_LOCAL, LATENT, I_TP, 5000 + rank)
    alpha1 = (ex.spread * (ex.w1_gsf / gsf[0])).float()
    return SimpleNamespace(
        comm=comm, rank=rank, world=world, bias=bias, x32=x32, front=front_op.front_weight(head, gate_up), gsf=gsf,
        ws={m: moe_op.K3MoeHeadWorkspace.create(mapping, fabric_handle=True, m_max=m) for m in (8, 16, 32)},
        ctr=torch.zeros(1, dtype=torch.int32, device="cuda"), ex=ex,
        weights=(ex.w1, ex.w1_sf, alpha1, ex.w2, ex.w2_sf, ex.alpha2), offset=(rank % 4) * E_LOCAL,
        counters=nvfp4_lowm_op.new_counters(device),
    )  # fmt: skip


def _front(ctx, x, sort_out=None, offset=0, num_local=0, mask=None, tokens=1):
    m = x.shape[0]
    ws = ctx.ws[8 if m <= 8 else 16 if m <= 16 else 32]
    return _trtllm().k3_moe_front(
        x, ctx.front, ctx.bias, RSF, SHARED, GATE_CAP, LINEAR_CAP, ws.uc, ws.mc, ws.flags, ws.rank, ctx.world,
        nvfp4_gsf=ctx.gsf, sort_out=sort_out, sort_ctr=ctx.ctr if sort_out is not None else None, sort_offset=offset,
        sort_local=num_local, pad_mask=mask, pad_tokens=tokens,
    )  # fmt: skip


def _ordered_reference(ids, offset, num_local):
    """The tiles in expert order, each tile's rows the selecting tokens' expanded indices in token order."""
    by_expert = {}
    for i, e in enumerate(ids.reshape(-1).tolist()):
        if offset <= e < offset + num_local:
            by_expert.setdefault(e - offset, []).append(i)
    return [(e, by_expert[e]) for e in sorted(by_expert)]


def _sort_check(ids, weights, views, n, offset, num_local):
    """The masked sort ``views`` against moe_sort of the first ``n`` tokens alone."""
    group, mn_limit, perm, n_tiles, scales = views
    nt = int(n_tiles.item())
    ordered = _ordered_reference(ids[:n], offset, num_local)
    if n == 0:
        return dict(tiles=nt, sort_as_real_only=nt == 0, rows_token_order=nt == 0)
    r_group, r_limit, _, r_perm, _, r_tiles = _trtllm().moe_sort(
        token_selected_experts=ids[:n].contiguous(), token_final_scales=weights[:n].float(), num_experts=EXPERTS,
        top_k=TOP_K, local_expert_offset=offset, local_num_experts=num_local, tile_tokens_dim=TILE,
    )  # fmt: skip
    r_nt = int(r_tiles.item())
    good = (
        nt == r_nt
        and torch.equal(group[:nt], r_group[:nt])
        and torch.equal(mn_limit[:nt], r_limit[:nt])
    )
    in_order = nt == len(ordered)
    for t in range(nt if good else 0):
        lim = int(mn_limit[t]) - t * TILE
        got = perm[t * TILE : t * TILE + lim].tolist()
        ref = r_perm[t * TILE : t * TILE + lim].tolist()
        good = good and sorted(got) == sorted(ref) and all(0 <= r < n * TOP_K for r in got)
        in_order = in_order and (int(group[t]), got) == ordered[t]
    return dict(tiles=nt, sort_as_real_only=bool(good), rows_token_order=bool(in_order))


def _expert_check(ctx, q, s, ids, weights, views, n):
    """The routed experts on the masked sort against the experts on the real rows alone."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op

    m = q.shape[0]
    scales = views[4]
    full = weights.float()
    run_args = (ctx.ex.fc2_input_scale, ctx.counters, GATE_CAP, LINEAR_CAP)
    slots_bits, pad_zero, partials = True, True, []
    for slot in range(TOP_K):
        one_hot = torch.zeros_like(full)
        one_hot[:, slot] = full[:, slot]
        scales.copy_(one_hot)
        y = nvfp4_lowm_op.run_sorted(q, s, views, ctx.weights, *run_args)
        if n > 0:
            ref = nvfp4_lowm_op.run(
                q[:n], s[:n], ids[:n].contiguous(), one_hot[:n].contiguous(), ctx.weights, ctx.offset, EXPERTS,
                *run_args,
            )  # fmt: skip
            torch.cuda.synchronize()
            slots_bits = slots_bits and torch.equal(y[:n].view(torch.int16), ref.view(torch.int16))
            partials.append(ref.double())
        torch.cuda.synchronize()
        pad_zero = pad_zero and bool((y[n:].view(torch.int16) == 0).all())
    scales.copy_(full)
    y = nvfp4_lowm_op.run_sorted(q, s, views, ctx.weights, *run_args)
    torch.cuda.synchronize()
    sum_ok = True
    if n > 0:
        stack = torch.stack(partials)
        exact, bound = stack.sum(0), TOP_K * 2.0**-8 * stack.abs().sum(0) + 1e-30
        sum_ok = bool(((y[:n].double() - exact).abs() <= bound).all())
    pad_zero = pad_zero and bool((y[n:].view(torch.int16) == 0).all())
    return dict(
        slots_bits=bool(slots_bits), sum_in_bound=sum_ok, pad_rows_zero=pad_zero,
        counters_zero=int(ctx.counters.abs().sum().item()) == 0, rows=m,
    )  # fmt: skip


def check_pad_mask(ctx):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op

    results = []
    for m, tokens, real in CASES:
        requests, n = m // tokens, real * tokens
        mask = torch.zeros(32, dtype=torch.bool, device="cuda")
        mask[real:requests] = True
        x = ctx.x32[:m].contiguous()
        plain = _front(ctx, x)
        for offset, num_local in ((ctx.offset, E_LOCAL), (0, EXPERTS)):
            words = front_op.sort_layout(m, num_local)[3]
            sort_out = torch.full((words,), -7, dtype=torch.int32, device="cuda")
            out = _front(ctx, x, sort_out, offset, num_local, mask, tokens)
            unmasked = torch.full((words,), -7, dtype=torch.int32, device="cuda")
            _front(ctx, x, unmasked, offset, num_local)
            torch.cuda.synchronize()
            ids, weights, q, s, _ = out
            views = front_op.sort_views(sort_out, m, num_local)
            row = dict(
                op="k3_moe_front_pad_mask", case=f"local{num_local}_T{tokens}_real{real}", M=m,
                front_bits=all(_same(a, b) for a, b in zip(out, plain)),
                tiles_unmasked=int(front_op.sort_views(unmasked, m, num_local)[3].item()),
                **_sort_check(ids, weights, views, n, offset, num_local),
                weights_fp32=_same(views[4], weights.float()), ctr_zero=int(ctx.ctr.item()) == 0,
            )  # fmt: skip
            checks = [
                "front_bits",
                "sort_as_real_only",
                "rows_token_order",
                "weights_fp32",
                "ctr_zero",
            ]
            if num_local == E_LOCAL:
                row.update(_expert_check(ctx, q, s, ids, weights, views, n))
                checks += ["slots_bits", "sum_in_bound", "pad_rows_zero", "counters_zero"]
            good = all(row[k] for k in checks)
            row["rank"], row["good"] = ctx.rank, bool(good)
            row["ok"] = all(ctx.comm.allgather(bool(good)))
            results.append(row)
    return results


def _run_checks(_names=None):
    try:
        ctx = _context()
        with torch.inference_mode():
            return check_pad_mask(ctx)
    except Exception:
        traceback.print_exc()
        raise


def _report(per_rank):
    rows = list(per_rank[0]) + [row for rows in per_rank[1:] for row in rows if not row["good"]]
    for row in rows:
        fields = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("op", "case", "M"))
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_k3_moe_front_pad_mask(mpi_pool_executor):
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

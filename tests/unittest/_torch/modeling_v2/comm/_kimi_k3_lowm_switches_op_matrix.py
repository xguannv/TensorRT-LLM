# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE-path switches of the Kimi K3 NVFP4 decode path (``kimi_k3_nvfp4/flags.py``), each arm against the other
and the off arm against the chain it replaced, at W ranks (default 4) with the TP16 per-rank shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_lowm_switches_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _kimi_k3_lowm_switches_op_matrix.py --launcher srun \
        --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (the arms share the decode path's collective
state). The collected entry point is ``test_modeling_v2_kimi_k3_lowm_switches_op_matrix.py``.

What runs. Every rank builds MoE layers as ``_build_decode_moe_lowm`` builds them (``K3DecodeMoeNvfp4.create``,
``fold_latent_norm``, ``K3DecodeMoeNvfp4Layer.create``, ``warm_up``) from stand-in layers holding what the paths read:
the target's ``KimiK3MoEGate``, nn.Linear latent projections, the stock RMSNorm and shared GatedMLP (384 shared
columns per rank, TP16's), a stock MNNVL ``AllReduce``, and this rank's routed experts in the CuteDSL backend's NVFP4
buffers (224 experts of a 768-wide intermediate slice at global ids ``[224 (r % 4), 224 (r % 4) + 224)``, a TP4 x
EP4 rank of the target; random, per-expert global scales). The generic path runs the stand-in through
``KimiK3MoERuntime.forward`` / ``forward_generic`` themselves, its routed experts through ``CuteDslFusedMoE``'s own
NVFP4 methods (``quantize_input``, ``run_moe_nvfp4``) after the scheduler's no-communication steps (the gate's
routing method, int32 ids, fp32 weights). The decoder layer's residual updates run
``KimiLinearDecoderLayer.forward`` itself on a stand-in layer (attention and MoE stand-ins that return fixed
partials). A switch read at forward time (``TRTLLM_K3LL_WIDE_AR_ATTN_RES``, ``TRTLLM_K3LL_WIDE_LATENT_TAIL``,
``TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS``, ``TRTLLM_K3LL_ROUTE_QUANT_FP32``) is set on the flags module per arm; a
switch read when the target is built (``TRTLLM_K3LL_MOE_FRONT_SORT``, ``TRTLLM_K3LL_MOE_FRONT16``: the state
``K3DecodeMoeNvfp4.create`` builds; ``TRTLLM_K3LL_MOE_LOWM``: whether the layer gets a ``decode_moe`` at all) is one
built layer per arm. The decode GEMV sites are not built
(``decode_gemvs`` None): every GEMV runs on ``_gemv``'s stock GEMM.

Payloads. The MoE inputs and the head and shared gate_up rows are small multiples of powers of two, so every head and
shared gate_up sum is exact in fp32 in any order (the payloads ``moe/k3_moe_front``'s matrix certifies with): the
front's split-K GEMV, cuBLAS and the router GEMM give the same logits and latent bits. The routing is one-hot: 15
experts carry a routing bias of 8 and a router row whose logit is -256 for every input (column 0 of every input is
1/4), so each token's top 16 hold those 15 at a sigmoid of exactly 0 and one real expert with all the weight. Each
check asserts that every token has exactly one non-zero routing weight.

Why one-hot. The lowm expert kernel (``K3MoeLowMDynKernel``, ``cute_dsl_kernels/blackwell/k3_moe_lowm_dyn.py``) adds
each expert's weighted FC2 rows into its bf16 output with ``red.global.add`` in the order the hardware retires them,
so with several non-zero weights per token its output is not bit-reproducible even for identical inputs (nor is the
CuteDSL backend's fused finalize, an atomic scatter-add). With one non-zero weight per token every other addend is
zero and the output is deterministic, and the lowm kernel's is bit for bit the backend's kernels'
(``test_k3_moe_lowm.py::test_moe_nvfp4_lowm``). The bitwise comparisons below cover only deterministic parts: the
front's outputs (routing, sort views, NVFP4 latent and scales, shared activation), a fixed routed partial fed into the
tail, and the experts under one-hot routing.

Checks (each eagerly and from a captured CUDA graph replayed with rewritten inputs, each replay bit for bit the
eager call on the same inputs):
  * FRONT_SORT (1, 8, 9, 12, 16 rows; 9-16 on the 16-token front): ``_front_routed`` with the front's sort and
    without it. Off arm against ``_ref_front_sort_chain`` (the front, ``moe_sort``, ``k3_moe_nvfp4_lowm``): the
    front's outputs, the tile count, each tile's local expert and mn limit, the fp32 weights and the routed partial
    bit for bit, each tile's rows as a set (``moe_sort`` orders a tile's rows as its threads claim slots, which
    varies run to run from 8 rows up: two runs of the pre-change chain on one input are held to the same).
    On arm against the off arm: the front's outputs, the tile count, each tile's local expert and mn limit, the
    fp32 weights and the routed partial bit for bit; each tile's rows (expanded indices) equal as a set (the front
    writes them in token order; the kernel's FC1 row of a (token, expert) pair does not depend on its place).
  * FRONT16 (9, 12, 16 rows): ``_wide`` with and without the 16-token front, with TRTLLM_K3LL_WIDE_LATENT_TAIL off.
    Off arm against ``_ref_wide_head_chain`` + ``_ref_latent_tail_chain``: routing, NVFP4 latent and scales, routed
    partial, shared activation, tail input and output bit for bit. On arm against the off arm: routing, latent,
    scales and routed partial bit for bit (``k3_moe_front``: its routing and quantization are ``k3_route_quant``'s
    device code on the gathered head, here the same exact sums); the shared activation within ``SHARED_TOL`` of its
    largest magnitude (the front's SiTU uses fast tanh / sigmoid approximations, ``k3_moe_front.md``); the tail input's
    latent columns bit for bit and the output within ``_gemv_bound``.
  * WIDE_AR_ATTN_RES (16, 32, 64 rows; a layer that snapshots and one that does not; TRTLLM_K3LL_OPROJ_PUSH off, so
    the post-attention update is the all-reduce of the attention's partial): ``KimiLinearDecoderLayer.forward``
    on a wide step, its pre-attention update of the previous layer's MoE partial and its post-attention update. Off
    arm against ``_ref_wide_residual_chain`` (the stock MNNVL all-reduce, then the stock fused add + attn_res +
    RMSNorm helpers) bit for bit, and its ``updated`` against the exact sums. On arm (``mnnvl_allreduce_attn_res``
    two-shot) against the off arm: every ``updated`` bit for bit (the addends are small multiples of 1/16, so the sum
    is exact in any order; the two-shot op returns -0.0 as +0.0, so zeros are compared by value); every ``normed``
    within ``NORMED_TOL`` of its largest magnitude (the op's epilogue mixes the snapshots and sums the squares in
    another order than the stock ``attn_res_add_rmsnorm_fwd`` / ``attn_res_rmsnorm_fwd``; on GB300 a few elements
    per 16 x 7168 differ, up to about 6 bf16 ulp of their own magnitude where the mix nearly cancels).
  * WIDE_ATTN_RES_MAX_TOKENS (32, 48, 64 rows; both layer kinds; WIDE_AR_ATTN_RES and OPROJ_PUSH off, so both
    updates run the epilogue helpers): the same forward with the switch at 32 (off, the default) and 64 (on). Off
    arm against ``_ref_wide_residual_chain`` bit for bit and, above 32 rows, against ``_ref_unfused_residual_chain``
    (the add, ``attn_res_fwd`` and the RMSNorm module) bit for bit. On arm against the off arm: every ``updated`` bit
    for bit (exact sums); ``normed`` bit for bit up to 32 rows (the same fused kernel), above 32 within
    ``ATTN_RES_FUSED_TOL`` of its largest magnitude (the fused kernel normalizes the fp32 mixture, the chain the
    mixture rounded to bf16: not bitwise).
  * ROUTE_QUANT_FP32 (17, 32 rows, and 64 where the expert kernel takes it): ``_wide_head`` with and without fp32
    routing weights from ``k3_route_quant``. Off arm against ``_ref_wide_head_chain`` bit for bit (routing, latent,
    scales, routed partial, shared activation). On arm against the off arm: ids, latent and scales bit for bit, its
    fp32 weights bit for bit the off arm's bf16 weights widened, ``moe_sort``'s tiles, limits and weights equal (each
    tile's rows as a set), the routed partial and the shared activation bit for bit.
  * WIDE_LATENT_TAIL (16, 32, 64 rows, a fixed routed partial fed into ``_wide``): a tail-only stand-in with 224 latent
    columns per rank (the latent scaled with W as ``_mnnvl_allreduce_rmsnorm_slice_op_matrix.py`` scales it: the
    fused op takes at most 256 columns per rank, so at W 4 the 3584-wide latent keeps the stock tail). Off arm
    against ``_ref_latent_tail_chain`` bit for bit, its normalized slice within one bf16 ulp of the exact sum's.
    On arm against the off arm: the tail input's latent columns within one bf16 ulp (the op sums the squares in
    another order, ``mnnvl_allreduce_rmsnorm_slice``), the padding and the shared activation bit for bit, the GEMV
    output within ``_gemv_bound``.
  * MOE_LOWM (8, 16, 32 rows; 16 and 32 hand the row-parallel partial on, reduced here by the consumer's wide
    all-reduce): ``KimiK3MoERuntime.forward`` with and without a ``decode_moe``. Off arm against ``forward_generic``
    called directly, bit for bit. On arm against the off arm (``MOE_TOL`` of the largest magnitude: the shared
    activation's SiTU approximations, the folded latent norm, the routing weight in bf16, the tails' other
    roundings); the routing's experts equal, the weights within one bf16 ulp, the NVFP4 latent and scales bit for bit;
    and the expert kernel on the on arm's routing and latent bit for bit ``CuteDslFusedMoE.run_moe_nvfp4`` on the same
    inputs.

Outputs the ranks share (routing, latent, reduced outputs, ``updated`` / ``normed``) are also compared bitwise across
the ranks; the shared activation, the routed partial and the tail are this rank's own.
"""

import contextlib
import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the Kimi K3 lowm MoE switches require CUDA devices"

DEADLINE_S = 2700
H = 7168
LATENT = 3584
NUM_EXPERTS = 896
TOP_K = 16
E_LOCAL = 224  # a TP4 x EP4 rank's routed experts
I_TP = 768  # their intermediate slice
SHARED_PER_RANK = 384  # TP16's shared activation columns per rank
# WIDE_LATENT_TAIL's latent columns per rank: the tail stand-in's latent is TAIL_SLICE * W.
TAIL_SLICE = 224
SITU_CAPS = (4.0, 25.0)
RSF = 2.827
SF16 = 16
FP4_MAX = 448.0 * 6.0
# The latent's NVFP4 global scale: |latent| up to about 8 maps to the largest code.
LATENT_GSF = FP4_MAX / 8.0
FC2_GSF = FP4_MAX / 4.0
LATENT_EPS = 1e-5
RES_EPS = 1e-5
TILE = 128
# One-hot routing: these experts' router rows give a logit of DUMMY_LOGIT_WEIGHT * 1/4 = -256 (sigmoid exactly 0) and
# their bias puts them in every token's top 16.
DUMMY_EXPERTS = tuple(7 + 59 * j for j in range(TOP_K - 1))
DUMMY_LOGIT_WEIGHT = -1024.0
DUMMY_BIAS = 8.0
X_COLUMN0 = 0.25
ATTN_RES_BLOCK = 4
SNAPSHOTS = 3

FRONT_SORT_ROWS = (1, 8, 9, 12, 16)
FRONT16_ROWS = (9, 12, 16)
WIDE_ROWS = (16, 32, 64)
ATTN_RES_CAP_ROWS = (32, 48, 64)
ROUTE_FP32_ROWS = (17, 32, 64)
MOE_ROWS = (8, 16, 32)
GRAPH_FRONT_SORT_ROWS = (8, 16)
GRAPH_FRONT16_ROWS = (9, 16)
REPLAYS = 2

ULP_REL = 2.0**-7  # one bf16 ulp, relative
SHARED_TOL = 2e-2  # the front's shared activation: max |err| / max |ref| (k3_moe_front.md)
NORMED_TOL = 2e-2  # normed without a prefix sum: max |err| / max |ref| (the attn_res matrix's TOL)
MOE_TOL = 3e-2  # MOE_LOWM, the on arm's reduced output: max |err| / max |ref|
# WIDE_ATTN_RES_MAX_TOKENS above 32 rows, fused against the add -> attn_res_fwd -> RMSNorm chain: max |err| / max |ref|
ATTN_RES_FUSED_TOL = 2e-2
STOCK_WIDE_ONE_SHOT_MAX_BYTES = 1 << 20  # the stock MNNVL all-reduce's one-shot ceiling

R = None
IMPL = None  # kimi_k3_nvfp4/implementation.py
DC = None  # decode_comm
DM = None  # decode_moe
DML = None  # decode_moe_lowm
FLAGS = None  # flags
LOWM_OP = None  # cute_dsl_kernels/k3_fused_moe/nvfp4_lowm_op.py
BUILT = None
STATS = {}


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0, offset=0.0):
    return (offset + scale * torch.randn(tuple(shape), generator=g, device="cuda")).bfloat16()


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(t.shape[0], -1)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape[0] == b.shape[0] and torch.equal(_bytes(a), _bytes(b))


def _expect_same(where: str, got: torch.Tensor, want: torch.Tensor, replicated: bool = False):
    """``got`` bit for bit ``want``; with ``replicated``, also the same bits on every rank."""
    same = _same(got, want)
    agree = R.same_on_ranks(got) if replicated else True
    if not same:
        differ = (
            int((_bytes(got) != _bytes(want)).sum())
            if _bytes(got).shape == _bytes(want).shape
            else -1
        )
        raise AssertionError(
            f"{where}: not bit for bit ({differ} bytes differ, shapes {got.shape} {want.shape})"
        )
    assert agree, f"{where}: the ranks differ"


def _expect_close(
    where: str, got: torch.Tensor, want: torch.Tensor, tol: float, replicated: bool = False
):
    """max |got - want| / max |want| within ``tol``; with ``replicated``, ``got`` the same bits on every rank."""
    err = ls.rel_err(got, want)
    agree = R.same_on_ranks(got) if replicated else True
    STATS[where.split(":")[0]] = max(STATS.get(where.split(":")[0], 0.0), err)
    assert torch.isfinite(got.float()).all(), f"{where}: not finite"
    assert err <= tol, f"{where}: rel err {err:.3e} > {tol}"
    assert agree, f"{where}: the ranks differ"
    return err


def _within_ulp(where: str, got: torch.Tensor, want: torch.Tensor):
    """Every element of ``got`` within one bf16 ulp of ``want`` (|diff| <= |want| 2^-7)."""
    diff = (got.float() - want.float()).abs()
    allowed = want.float().abs() * ULP_REL + 1e-30
    assert bool((diff <= allowed).all()), (
        f"{where}: off by more than one bf16 ulp (max {(diff / allowed).max().item():.3f} ulp)"
    )


def _gemv_bound(where, y_got, y_want, x_got, x_want, weight):
    """``y = x @ weight.T`` on the same GEMM for two inputs that differ: ``|y_got - y_want|`` within the inputs'
    difference carried through ``|weight|``, plus each output's bf16 rounding (2^-8 of both magnitudes) and the fp32
    accumulations' rounding (2^-12 of ``|x| @ |weight|.T``, generous for K <= 1408)."""
    w = weight.float().abs()
    carried = (x_got.float() - x_want.float()).abs() @ w.t()
    scale = x_want.float().abs() @ w.t()
    bound = (
        carried + 2.0**-8 * (y_got.float().abs() + y_want.float().abs()) + 2.0**-12 * scale + 1e-30
    )
    err = (y_got.float() - y_want.float()).abs()
    assert bool((err <= bound).all()), f"{where}: max err / bound {(err / bound).max().item():.3f}"


def _canonical_zero(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t == 0, torch.zeros_like(t), t)


def _assert_one_hot(where: str, weights: torch.Tensor):
    nonzero = (weights.float() != 0).sum(dim=-1)
    assert bool((nonzero == 1).all()), (
        f"{where}: the stand-in's routing is not one-hot (non-zero weights per token {nonzero.tolist()}); the "
        "bitwise expert comparisons need exactly one"
    )


@contextlib.contextmanager
def _switch(name: str, value: int):
    """Set switch ``name`` on the flags module (read at forward time as ``_flags.<name>``) for the block."""
    old = getattr(FLAGS, name)
    setattr(FLAGS, name, int(value))
    try:
        yield
    finally:
        setattr(FLAGS, name, old)


_TAPPED = ("_front", "_routed", "_front_routed", "_wide_head")


@contextlib.contextmanager
def _taps(layer):
    """Keep the last result of each of the layer's steps in ``_TAPPED`` (and ``_routed``'s inputs) while the block
    runs: instance attributes wrapping its methods, removed on exit. Under capture the kept tensors are the graph's
    own, which each replay rewrites."""
    seen = {}
    for name in _TAPPED:

        def tapped(*args, _method=getattr(layer, name), _name=name, **kwargs):
            out = _method(*args, **kwargs)
            seen[_name] = (*args[-4:], out) if _name == "_routed" else out
            return out

        setattr(layer, name, tapped)
    try:
        yield seen
    finally:
        for name in _TAPPED:
            layer.__dict__.pop(name, None)


@contextlib.contextmanager
def _gemv_tap():
    """Keep ``decode_moe_lowm._gemv``'s input and output per site while the block runs."""
    seen = {}
    original = DML._gemv

    def tapped(gemvs, site, x, weight, out_fp32=False):
        y = original(gemvs, site, x, weight, out_fp32=out_fp32)
        seen[site] = (x, y)
        return y

    DML._gemv = tapped
    try:
        yield seen
    finally:
        DML._gemv = original


@contextlib.contextmanager
def _sort_tap():
    """Keep the sorted routing ``nvfp4_lowm_op.run_sorted`` hands the expert kernel (``run`` calls it after
    ``moe_sort``; the front's sort calls it with ``sort_views``)."""
    seen = {}
    original = LOWM_OP.run_sorted

    def tapped(x_fp4, x_sf, sorted_routing, *args, **kwargs):
        seen["sort"] = sorted_routing
        return original(x_fp4, x_sf, sorted_routing, *args, **kwargs)

    LOWM_OP.run_sorted = tapped
    try:
        yield seen
    finally:
        LOWM_OP.run_sorted = original


def _capture(fn):
    """``fn()`` captured once in a CUDA graph on a side stream (every kernel it runs compiled by an eager call
    before); returns the graph and ``fn``'s result, the graph's tensors."""
    R.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn()
    R.barrier()
    return graph, out


def _replay(graph):
    R.barrier()
    graph.replay()
    torch.cuda.synchronize()


def _wide_step(rows: int):
    """A pure decode step of ``rows`` tokens, 8 per request (one request below 8)."""
    return IMPL.DecodeStep(rows, max(1, rows // 8), min(rows, 8))


def _moe_input(seed: int, rows: int) -> torch.Tensor:
    """A MoE input, the same on every rank: multiples of 1/8 in [-1/4, 1/4], column 0 at 1/4 (the one-hot routing's
    dummy logits)."""
    x = ls.exact_bf16(_gen(9000 + seed), (rows, H), -2, 3, 1 / 8)
    x[:, 0] = X_COLUMN0
    return x


def _stock_wide_all_reduce(all_reduce, x: torch.Tensor) -> torch.Tensor:
    """A stock ``AllReduce`` of a wide step: its MNNVL all-reduce at the stock one-shot ceiling, else the module."""
    from tensorrt_llm._torch.distributed import AllReduceParams

    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if mnnvl is not None:
        out = mnnvl(
            x.contiguous(), AllReduceParams(), one_shot_max_bytes=STOCK_WIDE_ONE_SHOT_MAX_BYTES
        )
        if out is not None:
            return out
    return all_reduce(x)


# ----------------------------------------------------------------------------------------------------------------------
# Stand-ins
# ----------------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _backend_class():
    """``CuteDslFusedMoE``'s NVFP4 forward (its own methods) over buffers set as attributes."""
    from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE

    class CuteDslBackendStandIn:
        quantize_input = CuteDslFusedMoE.quantize_input
        run_moe_nvfp4 = CuteDslFusedMoE.run_moe_nvfp4
        run_moe_nvfp4_impl = CuteDslFusedMoE.run_moe_nvfp4_impl
        _build_local_weight_view = CuteDslFusedMoE._build_local_weight_view

        def __init__(self, **fields):
            self.__dict__.update(fields)

        def _has_moe_output_memset_aux_stream(self) -> bool:
            # No memset stream: the output is zeroed on the calling stream.
            return False

    return CuteDslBackendStandIn


def _backend():
    """This rank's routed experts in the CuteDSL backend's NVFP4 buffers, as ``test_k3_moe_lowm.py`` builds them:
    W1 ``[224, 1536, 3584]`` and W2 ``[224, 3584, 768]`` E2M1 pairs with E4M3 scales in the 128x4 blocked layout,
    per-expert global scales, the latent's input global scale ``LATENT_GSF`` and the FC1 output's ``FC2_GSF``."""
    from tensorrt_llm._torch.utils import ActivationType

    g = _gen(3000 + R.rank)

    def quantize(t):
        gsf = t.abs().max().float() / FP4_MAX
        q, sf = torch.ops.trtllm.fp4_quantize(t, 1 / gsf, SF16, False)
        return q.view(torch.uint8), sf.view(torch.uint8), gsf

    w1 = torch.randn(E_LOCAL, 2 * I_TP, LATENT, generator=g, device="cuda", dtype=torch.bfloat16)
    w1, w1_sf, w1_gsf = quantize(w1)
    w2 = torch.randn(E_LOCAL, LATENT, I_TP, generator=g, device="cuda", dtype=torch.bfloat16)
    w2, w2_sf, w2_gsf = quantize(w2)
    spread = torch.linspace(0.5, 1.5, E_LOCAL, device="cuda")
    fc31_input_scale = torch.tensor([LATENT_GSF], dtype=torch.float32, device="cuda")
    fc2_input_scale = torch.tensor([FC2_GSF], dtype=torch.float32, device="cuda")
    quant_scales = SimpleNamespace(
        fc1_weight_block=w1_sf.view(E_LOCAL, 2 * I_TP, LATENT // SF16),
        fc1_global=(spread * (w1_gsf / LATENT_GSF)).float(),
        fc2_weight_block=w2_sf.view(E_LOCAL, LATENT, I_TP // SF16),
        fc2_global=(spread * (w2_gsf / FC2_GSF)).float(),
    )
    torch.cuda.synchronize()
    return _backend_class()(
        has_nvfp4=True,
        w3_w1_weight=w1,
        w2_weight=w2,
        quant_scales=quant_scales,
        fc31_input_scale=fc31_input_scale,
        fc2_input_scale=fc2_input_scale,
        scaling_vector_size=SF16,
        hidden_size=LATENT,
        num_slots=NUM_EXPERTS,
        expert_size_per_partition=E_LOCAL,
        slot_start=(R.rank % 4) * E_LOCAL,
        activation_type=ActivationType.SiTu,
        act_alpha=SITU_CAPS[0],
        act_beta=SITU_CAPS[1],
        act_clamp=float("inf"),
        use_fused_finalize=True,
        _locality_domain_runtime=None,
        mapping=SimpleNamespace(moe_ep_size=4),
    )


class _GenericRoutedExperts:
    """The routed experts of the generic path: ``ConfigurableMoE``'s no-communication flow on the CuteDSL backend
    (``moe_scheduler``: the routing method, int32 ids and fp32 weights, ``quantize_input``, the NVFP4 run), and the
    routed experts' all-reduce module. Keeps the last call's ``(ids, weights, x_fp4, x_sf, out)`` as ``seen``."""

    def __init__(self, backend, gate, all_reduce):
        self.backend, self.gate, self.all_reduce = backend, gate, all_reduce
        self.seen = None

    def __call__(self, x, router_logits, all_rank_num_tokens=None):
        ids, scales = self.gate.routing_method.apply(router_logits)
        ids = ids.to(torch.int32)
        assert scales.dtype == torch.float32, scales.dtype
        x_fp4, x_sf = self.backend.quantize_input(x, post_quant_comm=False)
        out = self.backend.run_moe_nvfp4(
            x=x_fp4,
            token_selected_experts=ids,
            token_final_scales=scales,
            x_sf=x_sf,
            weight_view=self.backend._build_local_weight_view(),
        )
        self.seen = (ids, scales, x_fp4, x_sf, out)
        return out


@functools.lru_cache(maxsize=None)
def _moe_class():
    """A module with ``KimiK3MoERuntime``'s own ``forward`` / ``forward_generic`` over the attributes they read."""
    runtime = IMPL.KimiK3MoERuntime

    class MoeStandIn(nn.Module):
        forward = runtime.forward
        forward_generic = runtime.forward_generic
        _routed_projection = runtime.__dict__["_routed_projection"]

    return MoeStandIn


def _moe_layer(seed: int, latent: int, backend, all_reduce):
    """A MoE layer with what both paths read, from the modules KimiK3MoERuntime builds; ``latent`` columns. The gate,
    the latent projections and the norm are the same on every rank (seed ``seed``); the shared expert is this rank's
    slice. Router and latent-down rows and shared gate_up rows are exact payloads (module docstring); the routing is
    one-hot (``DUMMY_EXPERTS``)."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm._torch.modules.situ import SituAndMul
    from tensorrt_llm.functional import AllReduceStrategy

    cfg = SimpleNamespace(
        num_experts_per_token=TOP_K,
        num_experts=NUM_EXPERTS,
        routed_scaling_factor=RSF,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        hidden_size=H,
    )
    with torch.device("cuda"):
        gate = IMPL.KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16)
        down = nn.Linear(H, latent, bias=False, dtype=torch.bfloat16)
        up = nn.Linear(latent, H, bias=False, dtype=torch.bfloat16)
        norm = RMSNorm(hidden_size=latent, eps=LATENT_EPS, dtype=torch.bfloat16)
        shared = GatedMLP(
            hidden_size=H,
            intermediate_size=SHARED_PER_RANK * R.world,
            bias=False,
            activation=SituAndMul(
                beta=SITU_CAPS[0], linear_beta=SITU_CAPS[1], use_fused_activation=True
            ),
            dtype=torch.bfloat16,
            config=ModelConfig(mapping=R.mapping, allreduce_strategy=AllReduceStrategy.MNNVL),
            reduce_output=True,
            layer_idx=seed,
            is_shared_expert=True,
        )
    g = _gen(5000 + seed)  # the same on every rank
    gr = _gen(5100 + 16 * seed + R.rank)  # this rank's shared expert slice
    dummies = torch.tensor(DUMMY_EXPERTS, device="cuda")
    with torch.no_grad():
        router = ls.exact_bf16(g, gate.weight.shape, -1, 2, 1 / 8)
        router[:, 0] = 0
        router[dummies] = 0
        router[dummies, 0] = DUMMY_LOGIT_WEIGHT
        gate.weight.copy_(router)
        bias = 0.05 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
        bias[dummies] = DUMMY_BIAS + 0.01 * torch.arange(len(DUMMY_EXPERTS), device="cuda")
        gate.e_score_correction_bias.copy_(bias)
        down.weight.copy_(ls.exact_bf16(g, down.weight.shape, -2, 3, 1 / 16))
        up.weight.copy_(_normal(g, up.weight.shape, 0.02))
        norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
        shared.gate_up_proj.weight.copy_(
            ls.exact_bf16(gr, shared.gate_up_proj.weight.shape, -2, 3, 1 / 16)
        )
        shared.down_proj.weight.copy_(_normal(gr, shared.down_proj.weight.shape, 0.02))
    moe = _moe_class()()
    moe.num_experts, moe.top_k, moe.moe_hidden_size, moe.hidden_size = NUM_EXPERTS, TOP_K, latent, H
    moe.gate = gate
    moe.routed_expert_down_proj = down
    moe.routed_expert_up_proj = up
    moe.routed_expert_norm = norm
    moe.shared_experts = shared
    moe.routed_experts = _GenericRoutedExperts(backend, gate, all_reduce)
    moe._situ_betas = SITU_CAPS
    moe._reduce_routed_output = True
    moe.moe_main_event = torch.cuda.Event()
    moe.moe_shared_event = torch.cuda.Event()
    moe.shared_expert_stream = torch.cuda.Stream()
    moe.decode_moe = None
    moe.decode_gemvs = None
    return moe


def _ref_weights(moe):
    """The decode weights of ``moe`` rebuilt from its modules (as ``K3DecodeMoeNvfp4Layer.create`` lays them out):
    this rank's head rows, the front weight, the tail weight ``[latent up columns | zeros | shared down]``, the
    routing bias, the global scales and the experts' buffers."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe_front import front_weight

    width = moe.moe_hidden_size // R.world
    lo, per = R.rank * width, NUM_EXPERTS // R.world
    pad = -width % DM.TAIL_K_TILE
    up = moe.routed_expert_up_proj.weight
    backend = moe.routed_experts.backend
    scales = backend.quant_scales
    head = torch.cat(
        [
            moe.routed_expert_down_proj.weight[lo : lo + width],
            moe.gate.weight[R.rank * per : (R.rank + 1) * per],
        ]
    ).contiguous()
    tail = torch.cat(
        [up[:, lo : lo + width], up.new_zeros(H, pad), moe.shared_experts.down_proj.weight], dim=1
    ).contiguous()
    return SimpleNamespace(
        head=head,
        front=front_weight(head, moe.shared_experts.gate_up_proj.weight),
        tail=tail,
        lo=lo,
        width=width,
        pad=pad,
        bias=moe.gate.e_score_correction_bias.detach(),
        gsf=backend.fc31_input_scale,
        fc2_gsf=backend.fc2_input_scale,
        offset=backend.slot_start,
        experts=(
            backend.w3_w1_weight,
            scales.fc1_weight_block,
            scales.fc1_global,
            backend.w2_weight,
            scales.fc2_weight_block,
            scales.fc2_global,
        ),
    )


def _build_lowm(moe, comm, front_sort: int, front16: int):
    """``moe``'s decode path as ``_build_decode_moe_lowm`` builds it with these values of TRTLLM_K3LL_MOE_FRONT_SORT
    and TRTLLM_K3LL_MOE_FRONT16 (read when the target is built): its own state on the TP group's MNNVL workspace,
    the layer, and one call of every kernel (``warm_up``). ``fold_latent_norm`` has run on ``moe``."""
    front16 = bool(front16) and DML.front16_supported(moe, R.world)
    state = DML.K3DecodeMoeNvfp4.create(
        R.mapping,
        moe.routed_experts.backend.w3_w1_weight.device,
        comm.mnnvl,
        front_sort=bool(front_sort),
        front16=front16,
    )
    layer = DML.K3DecodeMoeNvfp4Layer.create(moe, state, R.rank, R.world)
    layer.warm_up(moe)
    return layer


# ----------------------------------------------------------------------------------------------------------------------
# The chains the switches replaced, from the stock ops
# ----------------------------------------------------------------------------------------------------------------------
def _ref_experts(x_fp4, x_sf, ids, weights, ref, counters):
    """This rank's routed partial as before the front's sort: ``moe_sort`` of the routing over the local experts
    (128-row tiles), then ``trtllm::k3_moe_nvfp4_lowm``. Returns ``(sorted routing, partial)``."""
    rows = ids.shape[0]
    scales = weights.float()
    t2g, t2lim, _, p2e, _, nnet = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=ref.offset,
        local_num_experts=E_LOCAL,
        tile_tokens_dim=TILE,
    )
    out = torch.empty(rows, LATENT, dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.k3_moe_nvfp4_lowm(
        x_fp4, x_sf, *ref.experts, out, t2g, t2lim, p2e, nnet, ref.fc2_gsf, scales, counters, TOP_K,
        *SITU_CAPS,
    )  # fmt: skip
    return (t2g, t2lim, p2e, nnet, scales), out


def _ref_front_sort_chain(state, ref, moe, x):
    """``_front_routed`` before TRTLLM_K3LL_MOE_FRONT_SORT: the front without ``sort_out`` on the state's head
    workspace (the 16-token build's above 8 rows), then ``_ref_experts``."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe_front import k3_moe_front

    head = state.head16 if x.shape[0] > DML.MAX_TOKENS else state.head
    front = k3_moe_front(
        x, ref.front, ref.bias, RSF, SHARED_PER_RANK, *SITU_CAPS, head, nvfp4_gsf=ref.gsf
    )
    sort, routed = _ref_experts(front[2], front[3], front[0], front[1], ref, state.counters)
    return SimpleNamespace(front=front, sort=sort, routed=routed, act=front[4])


def _ref_wide_head_chain(state, ref, moe, x):
    """A 9-16 token step's head before TRTLLM_K3LL_MOE_FRONT16: the head GEMV (fp32 out, the router GEMM op), the
    MNNVL all-gather of the slices, ``k3_route_quant`` with the NVFP4 latent, ``_ref_experts``, and the shared
    gate_up GEMM + the GatedMLP's activation."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
        mnnvl_allgather_split,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_route_quant import (
        k3_route_quant,
    )

    head = torch.ops.trtllm.dsv3_router_gemm_op(
        x.contiguous(), ref.head.t(), bias=None, out_dtype=torch.float32
    )
    latent, logits = mnnvl_allgather_split(head, ref.width, state.mnnvl)
    ids, weights, x_fp4, x_sf = k3_route_quant(
        logits, ref.bias, latent.contiguous(), RSF, nvfp4_gsf=ref.gsf
    )
    _, routed = _ref_experts(x_fp4, x_sf, ids, weights, ref, state.counters)
    shared = moe.shared_experts
    act = shared._apply_activation(F.linear(x, shared.gate_up_proj.weight))
    return SimpleNamespace(routing=(ids, weights, x_fp4, x_sf), routed=routed, act=act)


def _ref_latent_tail_chain(moe, ref, routed, act):
    """A wide step's tail before TRTLLM_K3LL_WIDE_LATENT_TAIL: the routed experts' all-reduce at the wide step's
    stock one-shot ceiling, the latent RMSNorm module (its weight folded into the tail weight), this rank's latent
    columns, the zero padding, the shared activation, and the tail GEMM (``F.linear``, the GEMM ``_gemv`` runs without
    decode GEMV state). Returns ``(tail input, tail output)``."""
    normed = moe.routed_expert_norm(_stock_wide_all_reduce(moe.routed_experts.all_reduce, routed))
    parts = [normed[:, ref.lo : ref.lo + ref.width]]
    if ref.pad:
        parts.append(normed.new_zeros(normed.shape[0], ref.pad))
    parts.append(act)
    tail_in = torch.cat(parts, dim=1)
    return tail_in, F.linear(tail_in, ref.tail)


def _ref_wide_residual_chain(layer, call):
    """A wide step's two residual updates before TRTLLM_K3LL_WIDE_AR_ATTN_RES: the stock MNNVL all-reduce of the
    previous layer's MoE partial, the fused add + attn_res + RMSNorm helper; the snapshot; the stock all-reduce of the
    attention partial, the fused (add +) attn_res + RMSNorm helper. Returns the step's four outputs."""
    max_tokens = IMPL._attn_res_max_tokens(call.step)
    valid = call.bank[:SNAPSHOTS]
    moe_out = _stock_wide_all_reduce(layer._o_allreduce(), call.moe_partial)
    updated1, normed1 = IMPL._apply_attn_res_add_and_rmsnorm(
        call.prefix, moe_out, valid, layer.self_attention_res_proj, layer.self_attention_res_norm,
        layer.input_layernorm, max_tokens,
    )  # fmt: skip
    prefix = updated1
    if call.snapshot_layer:
        call.bank[SNAPSHOTS].copy_(updated1)
        valid, prefix = call.bank[: SNAPSHOTS + 1], None
    attn_out = _stock_wide_all_reduce(layer._o_allreduce(), call.attn_partial)
    if prefix is None:
        updated2 = attn_out
        normed2 = IMPL._apply_attn_res_and_rmsnorm(
            updated2, valid, layer.mlp_res_proj, layer.mlp_res_norm,
            layer.post_attention_layernorm, max_tokens,
        )  # fmt: skip
    else:
        updated2, normed2 = IMPL._apply_attn_res_add_and_rmsnorm(
            prefix, attn_out, valid, layer.mlp_res_proj, layer.mlp_res_norm,
            layer.post_attention_layernorm, max_tokens,
        )  # fmt: skip
    return SimpleNamespace(normed1=normed1, updated1=updated1, normed2=normed2, updated2=updated2)


# ----------------------------------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------------------------------
def check_build():
    """The decode paths built as post_load_weights builds them, outside inference mode with autograd on: the TP
    group's ``K3DecodeComm``, the stand-in layers (the lowm path's with its latent norm folded, the generic path's
    not, as TRTLLM_K3LL_MOE_LOWM=0 leaves it), one lowm state and layer per arm of the load-time switches, and the
    decoder stand-in's residual modules."""
    global BUILT
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.functional import AllReduceStrategy

    if R.world not in (4, 8, 16):
        raise AssertionError(f"the MoE front and the MNNVL ops run 4, 8 or 16 ranks, not {R.world}")
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    all_reduce = AllReduce(
        mapping=R.mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
    )
    backend = _backend()
    moe_on = _moe_layer(21, LATENT, backend, all_reduce)
    moe_off = _moe_layer(21, LATENT, backend, all_reduce)
    moe_tail = _moe_layer(22, TAIL_SLICE * R.world, backend, all_reduce)
    DC.use_decode_one_shot(nn.ModuleList([moe_on, moe_off, moe_tail, all_reduce]))
    comm = DC.K3DecodeComm.create(R.mapping, None)
    for moe in (moe_on, moe_tail):
        DM.fold_latent_norm(moe)
    # Production defaults: the front's sort and the 16-token front.
    on = _build_lowm(moe_on, comm, front_sort=1, front16=1)
    plain16 = _build_lowm(moe_on, comm, front_sort=0, front16=1)
    wide = _build_lowm(moe_on, comm, front_sort=1, front16=0)
    # The tail stand-in's head is replaced by a fixed partial: its layer is built, never warmed (no front of its width).
    tail = DML.K3DecodeMoeNvfp4Layer.create(moe_tail, wide.state, R.rank, R.world)
    moe_on.decode_moe = on
    g = _gen(7000)
    with torch.device("cuda"):
        res = SimpleNamespace(
            sa_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            mlp_proj=nn.Linear(H, 1, bias=False, dtype=torch.bfloat16),
            sa_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            mlp_norm=IMPL.KimiK3RMSNorm(H, eps=RES_EPS, dtype=torch.bfloat16),
            in_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
            post_norm=RMSNorm(hidden_size=H, eps=RES_EPS, dtype=torch.bfloat16),
        )
    with torch.no_grad():
        for proj in (res.sa_proj, res.mlp_proj):
            proj.weight.copy_(_normal(g, proj.weight.shape, 0.05))
        for norm in (res.sa_norm, res.mlp_norm, res.in_norm, res.post_norm):
            norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
        refs = SimpleNamespace(on=_ref_weights(moe_on), tail=_ref_weights(moe_tail))
    torch.cuda.synchronize()
    BUILT = SimpleNamespace(
        comm=comm, all_reduce=all_reduce, moe_on=moe_on, moe_off=moe_off, moe_tail=moe_tail,
        on=on, plain16=plain16, wide=wide, tail=tail, res=res, refs=refs,
    )  # fmt: skip
    if R.rank == 0:
        print(
            f"[rank 0] built at world {R.world}: 16-token front {on.state.head16 is not None}, "
            f"tail stand-in latent {moe_tail.moe_hidden_size}",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TRTLLM_K3LL_MOE_FRONT_SORT
# ----------------------------------------------------------------------------------------------------------------------
def _front_sort_arm(layer, moe, x):
    with _taps(layer) as seen, _sort_tap() as sort:
        routed, act = layer._front_routed(moe, x)
    return SimpleNamespace(front=seen["_front"], sort=sort["sort"], routed=routed, act=act)


def _sort_table(sorted_routing):
    """The valid part of a sorted routing for 128-row tiles: the tile count, each tile's local expert and mn limit,
    each tile's rows (expanded indices) as stored, and the fp32 weights."""
    t2g, t2lim, p2e, nnet, scales = sorted_routing
    tiles = int(nnet.reshape(-1)[0].item())
    groups, limits = t2g[:tiles].tolist(), t2lim[:tiles].tolist()
    rows = [p2e[t * TILE : limits[t]].tolist() for t in range(tiles)]
    return tiles, groups, limits, rows, scales


_FRONT_NAMES = ("ids", "weights", "latent", "scales", "shared")


def _same_rows(where, what, a, b):
    """Each tile's rows of two sorted routings, as sets: ``moe_sort`` places a tile's rows in the order its threads
    claim slots, which varies run to run on the same input (measured on GB300 from 8 rows up); a row's FC1 does not
    depend on its place in the tile."""
    assert [sorted(r) for r in a] == [sorted(r) for r in b], (
        f"{where}: {what}: a tile's rows differ"
    )


def _verify_front_sort(where, on, off, ref, ref_again):
    _assert_one_hot(where, off.front[1])
    for name, a, b in zip(_FRONT_NAMES, off.front, ref.front):
        _expect_same(
            f"{where}: off arm vs pre-change front {name}", a, b, replicated=name != "shared"
        )
    off_t, ref_t, on_t = _sort_table(off.sort), _sort_table(ref.sort), _sort_table(on.sort)
    again_t = _sort_table(ref_again.sort)
    assert again_t[:3] == ref_t[:3], (
        f"{where}: moe_sort's tiles differ between two runs on the same input"
    )
    _same_rows(where, "moe_sort run twice", again_t[3], ref_t[3])
    _expect_same(f"{where}: pre-change routed partial run twice", ref_again.routed, ref.routed)
    assert off_t[:3] == ref_t[:3], (
        f"{where}: off arm's tiles {off_t[:3]} differ from moe_sort's {ref_t[:3]}"
    )
    _same_rows(where, "off arm vs moe_sort", off_t[3], ref_t[3])
    _expect_same(f"{where}: off arm vs pre-change sort weights", off_t[4], ref_t[4])
    _expect_same(f"{where}: off arm vs pre-change routed partial", off.routed, ref.routed)
    for name, a, b in zip(_FRONT_NAMES, on.front, off.front):
        _expect_same(f"{where}: on vs off front {name}", a, b)
    assert on_t[:3] == off_t[:3], (
        f"{where}: the front's tiles {on_t[:3]} differ from moe_sort's {off_t[:3]}"
    )
    assert [sorted(r) for r in on_t[3]] == [sorted(r) for r in off_t[3]], (
        f"{where}: a tile's rows differ"
    )
    _expect_same(f"{where}: on vs off sort weights", on_t[4], off_t[4])
    _expect_same(f"{where}: on vs off routed partial", on.routed, off.routed)
    _expect_same(f"{where}: on vs off shared activation", on.act, off.act)


def _same_arm(where, a, b, fields):
    for field in fields:
        x, y = getattr(a, field), getattr(b, field)
        for i, (p, q) in enumerate(zip(x, y) if isinstance(x, tuple) else ((x, y),)):
            _expect_same(f"{where}: {field}[{i}]", p, q)


def check_front_sort():
    """``_front_routed`` with and without the front's sort, at 1, 8 and 9-16 rows (module docstring)."""
    on, off, moe, ref = BUILT.on, BUILT.plain16, BUILT.moe_on, BUILT.refs.on
    assert on.state.sort_ctr is not None and off.state.sort_ctr is None
    for rows in FRONT_SORT_ROWS:
        if rows > DML.MAX_TOKENS and on.state.head16 is None:
            continue
        x = _moe_input(100 + rows, rows)
        a_on, a_off = _front_sort_arm(on, moe, x), _front_sort_arm(off, moe, x)
        _verify_front_sort(
            f"FRONT_SORT {rows} rows", a_on, a_off, _ref_front_sort_chain(off.state, ref, moe, x),
            _ref_front_sort_chain(off.state, ref, moe, x),
        )  # fmt: skip
    fields = ("front", "routed", "act")
    for rows in GRAPH_FRONT_SORT_ROWS:
        if rows > DML.MAX_TOKENS and on.state.head16 is None:
            continue
        x = _moe_input(200 + rows, rows)
        graph, (g_on, g_off) = _capture(
            lambda: (_front_sort_arm(on, moe, x), _front_sort_arm(off, moe, x))
        )
        for rep in range(REPLAYS):
            x.copy_(_moe_input(300 + 10 * rows + rep, rows))
            _replay(graph)
            where = f"FRONT_SORT graph {rows} rows replay {rep}"
            e_on, e_off = _front_sort_arm(on, moe, x), _front_sort_arm(off, moe, x)
            _same_arm(f"{where} on vs eager", g_on, e_on, fields)
            _same_arm(f"{where} off vs eager", g_off, e_off, fields)
            assert _sort_table(g_on.sort)[:4] == _sort_table(e_on.sort)[:4], (
                f"{where}: on arm's sort"
            )
            _verify_front_sort(
                where, g_on, g_off, _ref_front_sort_chain(off.state, ref, moe, x),
                _ref_front_sort_chain(off.state, ref, moe, x),
            )  # fmt: skip
        del graph
    if R.rank == 0:
        print(
            "[rank 0] FRONT_SORT: off arm = pre-change chain, on arm = off arm (eager and replays)",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TRTLLM_K3LL_MOE_FRONT16
# ----------------------------------------------------------------------------------------------------------------------
def _front16_arm(layer, moe, x):
    """``_wide`` on ``x`` (9-16 rows): the head's routing and latent, routed partial and shared activation, the tail
    GEMV's input and output."""
    with _taps(layer) as seen, _gemv_tap() as gemv:
        out = layer._wide(moe, x, None)
    if "_front" in seen:
        routing, (routed, act) = seen["_front"][:4], seen["_front_routed"]
    else:
        routing, routed, act = seen["_routed"][:4], seen["_routed"][4], seen["_wide_head"][1]
    return SimpleNamespace(
        routing=routing, routed=routed, act=act, tail_in=gemv["moe_tail"][0], out=out
    )


def _verify_front16(where, on, off, ref, weights):
    names = ("ids", "weights", "latent", "scales")
    _assert_one_hot(where, off.routing[1])
    for name, a, b in zip(names, off.routing, ref.routing):
        _expect_same(f"{where}: off arm vs pre-change {name}", a, b, replicated=True)
    for name in ("routed", "act", "tail_in", "out"):
        _expect_same(
            f"{where}: off arm vs pre-change {name}", getattr(off, name), getattr(ref, name)
        )
    for name, a, b in zip(names, on.routing, off.routing):
        _expect_same(f"{where}: on vs off {name}", a, b)
    _expect_same(f"{where}: on vs off routed partial", on.routed, off.routed)
    _expect_close(f"FRONT16 shared: {where}", on.act, off.act, SHARED_TOL)
    latent_cols = weights.width + weights.pad
    _expect_same(
        f"{where}: on vs off tail latent", on.tail_in[:, :latent_cols], off.tail_in[:, :latent_cols]
    )
    _gemv_bound(
        f"{where}: on vs off tail output", on.out, off.out, on.tail_in, off.tail_in, weights.tail
    )


def _front16_ref(moe, x):
    ref = BUILT.refs.on
    head = _ref_wide_head_chain(BUILT.wide.state, ref, moe, x)
    head.tail_in, head.out = _ref_latent_tail_chain(moe, ref, head.routed, head.act)
    return head


def check_front16():
    """``_wide`` at 9, 12 and 16 rows with and without the 16-token front (module docstring)."""
    on, off, moe = BUILT.on, BUILT.wide, BUILT.moe_on
    assert on.state.head16 is not None, (
        f"the 16-token front does not take this layer at world {R.world} (front16_supported); FRONT16 is not covered"
    )
    assert off.state.head16 is None
    with _switch("TRTLLM_K3LL_WIDE_LATENT_TAIL", 0):
        for rows in FRONT16_ROWS:
            x = _moe_input(400 + rows, rows)
            a_on, a_off = _front16_arm(on, moe, x), _front16_arm(off, moe, x)
            _verify_front16(
                f"FRONT16 {rows} rows", a_on, a_off, _front16_ref(moe, x), BUILT.refs.on
            )
        fields = ("routing", "routed", "act", "tail_in", "out")
        for rows in GRAPH_FRONT16_ROWS:
            x = _moe_input(500 + rows, rows)
            graph, (g_on, g_off) = _capture(
                lambda: (_front16_arm(on, moe, x), _front16_arm(off, moe, x))
            )
            for rep in range(REPLAYS):
                x.copy_(_moe_input(600 + 10 * rows + rep, rows))
                _replay(graph)
                where = f"FRONT16 graph {rows} rows replay {rep}"
                _same_arm(f"{where} on vs eager", g_on, _front16_arm(on, moe, x), fields)
                _same_arm(f"{where} off vs eager", g_off, _front16_arm(off, moe, x), fields)
                _verify_front16(where, g_on, g_off, _front16_ref(moe, x), BUILT.refs.on)
            del graph
    if R.rank == 0:
        print(
            f"[rank 0] FRONT16: off arm = pre-change chain; on arm's routing / latent / partial = off arm's, "
            f"shared activation max rel err {STATS.get('FRONT16 shared', 0.0):.2e}",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TRTLLM_K3LL_WIDE_AR_ATTN_RES
# ----------------------------------------------------------------------------------------------------------------------
class _Recorder:
    """A sub-module stand-in that keeps its last input and returns a fixed tensor."""

    def __init__(self, out):
        self.out, self.seen = out, None

    def __call__(self, hidden_states, *args, **kwargs):
        self.seen = hidden_states
        return self.out


class _ResidualCall:
    """One wide step's residual updates: every rank's MoE partial (the previous layer's, unreduced) and attention
    partial (multiples of 1/16 in [-1/4, 1/4], so their sums are exact), the running prefix sum and the snapshot bank
    (``SNAPSHOTS`` valid rows); this rank's buffers, rewritten in place for a replay."""

    def __init__(self, seed, rows, snapshot_layer):
        self.rows, self.snapshot_layer, self.step = rows, snapshot_layer, _wide_step(rows)
        empty = dict(dtype=torch.bfloat16, device="cuda")
        self.moe_partial = torch.empty(rows, H, **empty)
        self.attn_partial = torch.empty(rows, H, **empty)
        self.prefix = torch.empty(rows, H, **empty)
        self.bank = torch.empty(SNAPSHOTS + 1, rows, H, **empty)
        self.moe_next = torch.zeros(rows, H, **empty)
        self.draw(seed)

    def draw(self, seed):
        g = _gen(seed)
        self.moe_all = [ls.exact_bf16(g, (self.rows, H), -4, 5, 1 / 16) for _ in range(R.world)]
        self.attn_all = [ls.exact_bf16(g, (self.rows, H), -4, 5, 1 / 16) for _ in range(R.world)]
        self.prefix.copy_(ls.exact_bf16(g, (self.rows, H), -32, 33, 1 / 16))
        self.bank.copy_(torch.randn(self.bank.shape, generator=g, device="cuda").bfloat16())
        self.moe_partial.copy_(self.moe_all[R.rank])
        self.attn_partial.copy_(self.attn_all[R.rank])

    def exact_updated(self):
        """The two updates' exact values: prefix + the MoE sum, then (+) the attention sum."""
        moe = sum(t.float() for t in self.moe_all)
        attn = sum(t.float() for t in self.attn_all)
        first = self.prefix.float() + moe
        return first.bfloat16(), (attn if self.snapshot_layer else first + attn).bfloat16()


def _decoder(call, layer_idx):
    """``KimiLinearDecoderLayer``'s attributes its wide-step forward reads: the TP group's ``K3DecodeComm``, the
    residual modules, a stock all-reduce module for both partials, a KDA attention and a MoE stand-in."""
    res = BUILT.res
    return SimpleNamespace(
        layer_idx=layer_idx, attn_res_block_size=ATTN_RES_BLOCK, decode_comm=BUILT.comm,
        self_attention_res_proj=res.sa_proj, self_attention_res_norm=res.sa_norm, input_layernorm=res.in_norm,
        mlp_res_proj=res.mlp_proj, mlp_res_norm=res.mlp_norm, post_attention_layernorm=res.post_norm,
        is_kda=True, linear_attn=_Recorder(call.attn_partial), sandwich_oproj=False,
        is_moe=True, block_sparse_moe=_Recorder(call.moe_next), _o_allreduce=lambda: BUILT.all_reduce,
    )  # fmt: skip


def _residual_arm(layer, call, on):
    # The stand-in has no o_proj: the post-attention update is the all-reduce of the attention's partial.
    with _switch("TRTLLM_K3LL_WIDE_AR_ATTN_RES", on), _switch("TRTLLM_K3LL_OPROJ_PUSH", 0):
        prefix, snapshots, _ = IMPL.KimiLinearDecoderLayer.forward(
            layer, call.prefix, call.bank, SNAPSHOTS, None, call.step,
            pending_moe_partial=call.moe_partial, defer_moe_tail=True,
        )  # fmt: skip
    assert snapshots == SNAPSHOTS + int(call.snapshot_layer), snapshots
    return SimpleNamespace(
        normed1=layer.linear_attn.seen,
        updated1=call.bank[SNAPSHOTS] if call.snapshot_layer else None,
        normed2=layer.block_sparse_moe.seen,
        updated2=prefix,
    )


def _cloned(ns):
    return SimpleNamespace(**{k: None if v is None else v.clone() for k, v in vars(ns).items()})


def _verify_residual(where, call, on, off, ref):
    for name in ("normed1", "updated1", "normed2", "updated2"):
        a = getattr(off, name)
        if a is not None:
            _expect_same(
                f"{where}: off arm vs pre-change {name}", a, getattr(ref, name), replicated=True
            )
    want1, want2 = call.exact_updated()
    if off.updated1 is not None:
        _expect_same(f"{where}: off arm's first update vs the exact sum", off.updated1, want1)
    _expect_same(f"{where}: off arm's second update vs the exact sum", off.updated2, want2)
    for name in ("updated1", "updated2"):
        a, b = getattr(on, name), getattr(off, name)
        if b is not None:
            _expect_same(
                f"{where}: on vs off {name}",
                _canonical_zero(a),
                _canonical_zero(b),
                replicated=True,
            )
    _expect_close(
        f"WIDE_AR_ATTN_RES normed1: {where}", on.normed1, off.normed1, NORMED_TOL, replicated=True
    )
    if call.snapshot_layer:
        _expect_close(
            f"WIDE_AR_ATTN_RES normed2: {where}",
            on.normed2,
            off.normed2,
            NORMED_TOL,
            replicated=True,
        )
    else:
        _expect_close(
            f"WIDE_AR_ATTN_RES normed2: {where}",
            on.normed2,
            off.normed2,
            NORMED_TOL,
            replicated=True,
        )


def check_wide_ar_attn_res():
    """``KimiLinearDecoderLayer.forward``'s two residual updates on wide steps of 16 and 32 tokens, with and without
    the two-shot ``mnnvl_allreduce_attn_res`` (module docstring)."""
    for rows in WIDE_ROWS:
        for snapshot_layer in (False, True):
            layer_idx = ATTN_RES_BLOCK if snapshot_layer else 1
            call = _ResidualCall(8000 + rows + int(snapshot_layer), rows, snapshot_layer)
            layer = _decoder(call, layer_idx)
            for on in (1, 0):
                with _switch("TRTLLM_K3LL_WIDE_AR_ATTN_RES", on):
                    takes = BUILT.comm.takes_wide(call.moe_partial, call.step)
                assert takes == bool(on), (
                    f"takes_wide {takes} with TRTLLM_K3LL_WIDE_AR_ATTN_RES={on}"
                )
            where = f"WIDE_AR_ATTN_RES {rows} rows snapshot {snapshot_layer}"
            a_on = _cloned(_residual_arm(layer, call, 1))
            a_off = _cloned(_residual_arm(layer, call, 0))
            _verify_residual(
                where, call, a_on, a_off, _cloned(_ref_wide_residual_chain(layer, call))
            )
            graphs = {on: _capture(lambda on=on: _residual_arm(layer, call, on)) for on in (1, 0)}
            for rep in range(REPLAYS):
                call.draw(8100 + 10 * rows + 2 * rep + int(snapshot_layer))
                replayed = {}
                for on, (graph, out) in graphs.items():
                    _replay(graph)
                    replayed[on] = _cloned(out)
                gw = f"{where} graph replay {rep}"
                fields = ("normed1", "normed2", "updated2") + (
                    ("updated1",) if snapshot_layer else ()
                )
                for on in (1, 0):
                    eager = _cloned(_residual_arm(layer, call, on))
                    for name in fields:
                        _expect_same(f"{gw}: arm {on} {name} vs eager", getattr(replayed[on], name),
                                     getattr(eager, name))  # fmt: skip
                ref = _cloned(_ref_wide_residual_chain(layer, call))
                _verify_residual(gw, call, replayed[1], replayed[0], ref)
            del graphs
    if R.rank == 0:
        print(
            "[rank 0] WIDE_AR_ATTN_RES: off arm = pre-change chain = exact sums; on arm's updated = off arm's, "
            f"normed max rel err {STATS.get('WIDE_AR_ATTN_RES normed1', 0.0):.2e} / "
            f"{STATS.get('WIDE_AR_ATTN_RES normed2', 0.0):.2e}",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS
# ----------------------------------------------------------------------------------------------------------------------
ATTN_RES_CAP_OFF, ATTN_RES_CAP_ON = 32, 64


def _ref_unfused_residual_chain(layer, call):
    """A wide step's two residual updates above the fused kernels' token limit, as before the switch: the stock MNNVL
    all-reduce of each partial, the residual add, ``attn_res_fwd`` (``_apply_attn_res``) and the RMSNorm module.
    Returns the step's four outputs."""
    valid = call.bank[:SNAPSHOTS]
    moe_out = _stock_wide_all_reduce(layer._o_allreduce(), call.moe_partial)
    updated1 = call.prefix + moe_out
    normed1 = layer.input_layernorm(
        IMPL._apply_attn_res(
            updated1, valid, layer.self_attention_res_proj, layer.self_attention_res_norm
        )
    )
    prefix = updated1
    if call.snapshot_layer:
        call.bank[SNAPSHOTS].copy_(updated1)
        valid, prefix = call.bank[: SNAPSHOTS + 1], None
    attn_out = _stock_wide_all_reduce(layer._o_allreduce(), call.attn_partial)
    updated2 = attn_out if prefix is None else prefix + attn_out
    normed2 = layer.post_attention_layernorm(
        IMPL._apply_attn_res(updated2, valid, layer.mlp_res_proj, layer.mlp_res_norm)
    )
    return SimpleNamespace(normed1=normed1, updated1=updated1, normed2=normed2, updated2=updated2)


def _cap_arm(layer, call, cap):
    """The wide step's forward with WIDE_AR_ATTN_RES off (both updates on the epilogue helpers) and the helpers'
    token limit ``cap``."""
    with _switch("TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS", cap):
        return _residual_arm(layer, call, 0)


def _verify_cap(where, call, on, off):
    rows = call.rows
    with _switch("TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS", ATTN_RES_CAP_OFF):
        ref = _cloned(_ref_wide_residual_chain(call.layer, call))
    for name in ("normed1", "updated1", "normed2", "updated2"):
        a = getattr(off, name)
        if a is not None:
            _expect_same(
                f"{where}: off arm vs pre-change {name}", a, getattr(ref, name), replicated=True
            )
    if rows > ATTN_RES_CAP_OFF:
        unfused = _cloned(_ref_unfused_residual_chain(call.layer, call))
        for name in ("normed1", "updated1", "normed2", "updated2"):
            a = getattr(off, name)
            if a is not None:
                _expect_same(
                    f"{where}: off arm vs add -> attn_res_fwd -> RMSNorm {name}",
                    a,
                    getattr(unfused, name),
                    replicated=True,
                )
    want1, want2 = call.exact_updated()
    if off.updated1 is not None:
        _expect_same(f"{where}: off arm's first update vs the exact sum", off.updated1, want1)
    _expect_same(f"{where}: off arm's second update vs the exact sum", off.updated2, want2)
    for name in ("updated1", "updated2"):
        if getattr(off, name) is not None:
            _expect_same(
                f"{where}: on vs off {name}", getattr(on, name), getattr(off, name), replicated=True
            )
    for name in ("normed1", "normed2"):
        a, b = getattr(on, name), getattr(off, name)
        if rows <= ATTN_RES_CAP_OFF:
            _expect_same(f"{where}: on vs off {name}", a, b, replicated=True)
        else:
            _expect_close(
                f"WIDE_ATTN_RES_MAX_TOKENS {name}: {where}",
                a,
                b,
                ATTN_RES_FUSED_TOL,
                replicated=True,
            )


def check_wide_attn_res_max_tokens():
    """``KimiLinearDecoderLayer.forward``'s two residual updates on wide steps of 32, 48 and 64 tokens with the fused
    kernels' token limit at 32 and at 64 (module docstring)."""
    for rows in ATTN_RES_CAP_ROWS:
        for snapshot_layer in (False, True):
            layer_idx = ATTN_RES_BLOCK if snapshot_layer else 1
            call = _ResidualCall(8500 + rows + int(snapshot_layer), rows, snapshot_layer)
            layer = _decoder(call, layer_idx)
            call.layer = layer
            for cap in (ATTN_RES_CAP_ON, ATTN_RES_CAP_OFF):
                with _switch("TRTLLM_K3LL_WIDE_ATTN_RES_MAX_TOKENS", cap):
                    assert IMPL._attn_res_max_tokens(call.step) == cap
            where = f"WIDE_ATTN_RES_MAX_TOKENS {rows} rows snapshot {snapshot_layer}"
            a_on = _cloned(_cap_arm(layer, call, ATTN_RES_CAP_ON))
            a_off = _cloned(_cap_arm(layer, call, ATTN_RES_CAP_OFF))
            _verify_cap(where, call, a_on, a_off)
            caps = (ATTN_RES_CAP_ON, ATTN_RES_CAP_OFF)
            graphs = {cap: _capture(lambda cap=cap: _cap_arm(layer, call, cap)) for cap in caps}
            for rep in range(REPLAYS):
                call.draw(8600 + 10 * rows + 2 * rep + int(snapshot_layer))
                replayed = {}
                for cap, (graph, out) in graphs.items():
                    _replay(graph)
                    replayed[cap] = _cloned(out)
                gw = f"{where} graph replay {rep}"
                fields = ("normed1", "normed2", "updated2") + (
                    ("updated1",) if snapshot_layer else ()
                )
                for cap in caps:
                    eager = _cloned(_cap_arm(layer, call, cap))
                    for name in fields:
                        _expect_same(f"{gw}: cap {cap} {name} vs eager", getattr(replayed[cap], name),
                                     getattr(eager, name))  # fmt: skip
                _verify_cap(gw, call, replayed[ATTN_RES_CAP_ON], replayed[ATTN_RES_CAP_OFF])
            del graphs
    if R.rank == 0:
        print(
            "[rank 0] WIDE_ATTN_RES_MAX_TOKENS: off arm = pre-change chain = exact sums; on arm's updated = off arm's, "
            "normed bit for bit up to 32 rows, above it max rel err "
            f"{STATS.get('WIDE_ATTN_RES_MAX_TOKENS normed1', 0.0):.2e} / "
            f"{STATS.get('WIDE_ATTN_RES_MAX_TOKENS normed2', 0.0):.2e}",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TRTLLM_K3LL_ROUTE_QUANT_FP32
# ----------------------------------------------------------------------------------------------------------------------
def _route_fp32_arm(layer, moe, x, on):
    """``_wide_head`` on ``x`` with TRTLLM_K3LL_ROUTE_QUANT_FP32 = ``on``: the routing and latent the expert step
    reads, the sorted routing it hands the expert kernel, the routed partial and the shared activation."""
    with _switch("TRTLLM_K3LL_ROUTE_QUANT_FP32", on), _taps(layer) as seen, _sort_tap() as sort:
        routed, act = layer._wide_head(moe, x, None)
    return SimpleNamespace(routing=seen["_routed"][:4], sort=sort["sort"], routed=routed, act=act)


def _verify_route_fp32(where, on, off, ref):
    names = ("ids", "weights", "latent", "scales")
    _assert_one_hot(where, off.routing[1])
    for name, a, b in zip(names, off.routing, ref.routing):
        _expect_same(f"{where}: off arm vs pre-change {name}", a, b, replicated=True)
    _expect_same(f"{where}: off arm vs pre-change routed partial", off.routed, ref.routed)
    _expect_same(f"{where}: off arm vs pre-change shared activation", off.act, ref.act)
    assert off.routing[1].dtype == torch.bfloat16, off.routing[1].dtype
    assert on.routing[1].dtype == torch.float32, on.routing[1].dtype
    for name, a, b in zip(names, on.routing, off.routing):
        want = b.float() if name == "weights" else b
        _expect_same(f"{where}: on vs off {name}", a, want, replicated=True)
    on_t, off_t = _sort_table(on.sort), _sort_table(off.sort)
    assert on_t[:3] == off_t[:3], f"{where}: the tiles {on_t[:3]} differ from {off_t[:3]}"
    _same_rows(where, "on vs off", on_t[3], off_t[3])
    _expect_same(f"{where}: on vs off sort weights", on_t[4], off_t[4])
    _expect_same(f"{where}: on vs off routed partial", on.routed, off.routed)
    _expect_same(f"{where}: on vs off shared activation", on.act, off.act)


def check_route_quant_fp32():
    """``_wide_head`` with and without fp32 routing weights from ``k3_route_quant`` (module docstring)."""
    layer, moe, ref = BUILT.on, BUILT.moe_on, BUILT.refs.on
    rows_list = [rows for rows in ROUTE_FP32_ROWS if rows <= LOWM_OP.MAX_TOKENS]
    for rows in rows_list:
        x = _moe_input(1100 + rows, rows)
        a_on = _route_fp32_arm(layer, moe, x, 1)
        a_off = _route_fp32_arm(layer, moe, x, 0)
        _verify_route_fp32(
            f"ROUTE_QUANT_FP32 {rows} rows", a_on, a_off,
            _ref_wide_head_chain(layer.state, ref, moe, x),
        )  # fmt: skip
    fields = ("routing", "routed", "act")
    for rows in rows_list:
        x = _moe_input(1200 + rows, rows)
        graph, (g_on, g_off) = _capture(
            lambda: (_route_fp32_arm(layer, moe, x, 1), _route_fp32_arm(layer, moe, x, 0))
        )
        for rep in range(REPLAYS):
            x.copy_(_moe_input(1300 + 10 * rows + rep, rows))
            _replay(graph)
            where = f"ROUTE_QUANT_FP32 graph {rows} rows replay {rep}"
            _same_arm(f"{where} on vs eager", g_on, _route_fp32_arm(layer, moe, x, 1), fields)
            _same_arm(f"{where} off vs eager", g_off, _route_fp32_arm(layer, moe, x, 0), fields)
            _verify_route_fp32(where, g_on, g_off, _ref_wide_head_chain(layer.state, ref, moe, x))
        del graph
    if R.rank == 0:
        print(
            f"[rank 0] ROUTE_QUANT_FP32 ({rows_list} rows): off arm = pre-change chain; on arm's routing, sort and "
            "routed partial bit for bit the off arm's (weights widened)",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TRTLLM_K3LL_WIDE_LATENT_TAIL
# ----------------------------------------------------------------------------------------------------------------------
class _TailCall:
    """A wide step's fixed routed partial on every rank (multiples of 1/16 in [-1/4, 1/4]: the sum is exact) and this
    rank's shared activation; this rank's buffers, rewritten in place for a replay."""

    def __init__(self, seed, rows, latent):
        self.rows, self.latent = rows, latent
        self.routed = torch.empty(rows, latent, dtype=torch.bfloat16, device="cuda")
        self.act = torch.empty(rows, SHARED_PER_RANK, dtype=torch.bfloat16, device="cuda")
        # The MoE input: the head is fixed, so only its row count is read.
        self.x = torch.zeros(rows, H, dtype=torch.bfloat16, device="cuda")
        self.draw(seed)

    def draw(self, seed):
        g = _gen(seed)
        self.routed_all = [
            ls.exact_bf16(g, (self.rows, self.latent), -4, 5, 1 / 16) for _ in range(R.world)
        ]
        self.routed.copy_(self.routed_all[R.rank])
        self.act.copy_(_normal(_gen(seed * 16 + R.rank), self.act.shape, 0.5))

    def normalized_slice(self, ref, eps):
        """This rank's columns of the exact sum's RMSNorm (no weight), bf16."""
        r = sum(t.float() for t in self.routed_all).bfloat16().float()
        normed = (r * torch.rsqrt(r.square().mean(dim=-1, keepdim=True) + eps)).bfloat16()
        return normed[:, ref.lo : ref.lo + ref.width]


@contextlib.contextmanager
def _fixed_head(layer, call):
    """The layer's head steps (the 16-token front and the wide head) return ``call``'s routed partial and shared
    activation."""

    def head(*args, **kwargs):
        return call.routed, call.act

    layer._front_routed = layer._wide_head = head
    try:
        yield
    finally:
        layer.__dict__.pop("_front_routed", None)
        layer.__dict__.pop("_wide_head", None)


def _tail_arm(layer, moe, call, on):
    with _switch("TRTLLM_K3LL_WIDE_LATENT_TAIL", on), _fixed_head(layer, call), _gemv_tap() as gemv:
        out = layer._wide(moe, call.x, None)
    return SimpleNamespace(tail_in=gemv["moe_tail"][0], out=out)


def _verify_tail(where, call, on, off, ref, moe):
    weights = BUILT.refs.tail
    width, pad = weights.width, weights.pad
    _expect_same(f"{where}: off arm vs pre-change tail input", off.tail_in, ref[0])
    _expect_same(f"{where}: off arm vs pre-change tail output", off.out, ref[1])
    eps = float(moe.routed_expert_norm.variance_epsilon)
    exact = call.normalized_slice(weights, eps)
    _within_ulp(f"{where}: off arm's slice vs the exact sum's", off.tail_in[:, :width], exact)
    _within_ulp(f"{where}: on vs off slice", on.tail_in[:, :width], off.tail_in[:, :width])
    zeros = off.tail_in.new_zeros(call.rows, pad)
    _expect_same(f"{where}: on arm's padding", on.tail_in[:, width : width + pad], zeros)
    _expect_same(f"{where}: on arm's shared activation", on.tail_in[:, width + pad :], call.act)
    _gemv_bound(
        f"{where}: on vs off tail output", on.out, off.out, on.tail_in, off.tail_in, weights.tail
    )
    STATS["tail slice ulps"] = STATS.get("tail slice ulps", 0) + int(
        (on.tail_in[:, :width] != off.tail_in[:, :width]).sum()
    )


def check_wide_latent_tail():
    """``_wide``'s tail on a fixed routed partial at 16 and 32 tokens, with and without
    ``mnnvl_allreduce_rmsnorm_slice`` (module docstring)."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce_rmsnorm_slice import (
        fits,
    )

    layer, moe, weights = BUILT.tail, BUILT.moe_tail, BUILT.refs.tail
    for rows in WIDE_ROWS:
        assert fits(rows, moe.moe_hidden_size, weights.pad, SHARED_PER_RANK, BUILT.comm.mnnvl), (
            f"mnnvl_allreduce_rmsnorm_slice does not take {rows} rows of the tail stand-in: the on arm would not run"
        )
        call = _TailCall(9100 + rows, rows, moe.moe_hidden_size)
        where = f"WIDE_LATENT_TAIL {rows} rows"
        a_on, a_off = (
            _cloned(_tail_arm(layer, moe, call, 1)),
            _cloned(_tail_arm(layer, moe, call, 0)),
        )
        _verify_tail(
            where,
            call,
            a_on,
            a_off,
            _ref_latent_tail_chain(moe, weights, call.routed, call.act),
            moe,
        )
        graphs = {on: _capture(lambda on=on: _tail_arm(layer, moe, call, on)) for on in (1, 0)}
        for rep in range(REPLAYS):
            call.draw(9200 + 10 * rows + rep)
            replayed = {}
            for on, (graph, out) in graphs.items():
                _replay(graph)
                replayed[on] = _cloned(out)
            gw = f"{where} graph replay {rep}"
            for on in (1, 0):
                eager = _tail_arm(layer, moe, call, on)
                _expect_same(
                    f"{gw}: arm {on} tail input vs eager", replayed[on].tail_in, eager.tail_in
                )
                _expect_same(f"{gw}: arm {on} tail output vs eager", replayed[on].out, eager.out)
            ref = _ref_latent_tail_chain(moe, weights, call.routed, call.act)
            _verify_tail(gw, call, replayed[1], replayed[0], ref, moe)
        del graphs
    if R.rank == 0:
        print(
            f"[rank 0] WIDE_LATENT_TAIL: off arm = pre-change chain; on arm within one ulp "
            f"({STATS.get('tail slice ulps', 0)} slice elements differ on rank 0)",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TRTLLM_K3LL_MOE_LOWM
# ----------------------------------------------------------------------------------------------------------------------
def _moe_off(x, step):
    """TRTLLM_K3LL_MOE_LOWM=0: the layer without a decode path; ``forward`` runs ``forward_generic``."""
    moe = BUILT.moe_off
    assert moe.decode_moe is None
    out = moe.forward(x, step=step)
    return SimpleNamespace(out=out, generic=moe.routed_experts.seen)


def _moe_on(x, step):
    """TRTLLM_K3LL_MOE_LOWM=1: ``forward`` on the lowm path; a wide step's row-parallel partial reduced by the
    consumer's wide all-reduce."""
    moe = BUILT.moe_on
    with _taps(moe.decode_moe) as seen:
        out = moe.forward(x, partial_tail=step.wide, step=step)
    if step.wide:
        out = DC.wide_all_reduce(BUILT.all_reduce, out)
    routing = seen["_front"][:4] if "_front" in seen else seen["_routed"][:4]
    return SimpleNamespace(out=out, routing=routing)


def _verify_moe(where, on, off, direct):
    _expect_same(f"{where}: off arm (forward) vs forward_generic", off.out, direct, replicated=True)
    ids, weights, x_fp4, x_sf = on.routing
    g_ids, g_weights, g_fp4, g_sf, _ = off.generic
    _assert_one_hot(where, weights)
    _assert_one_hot(where, g_weights)
    order, g_order = ids.long().argsort(dim=-1), g_ids.long().argsort(dim=-1)
    assert torch.equal(ids.long().gather(1, order), g_ids.long().gather(1, g_order)), (
        f"{where}: routed experts"
    )
    w, gw = weights.float().gather(1, order), g_weights.gather(1, g_order)
    # KimiK3MoeRoutingMethod: a bf16 weight can differ by one ulp from the fp32 routing's.
    assert bool(((w - gw).abs() <= gw.abs() * ULP_REL).all()), (
        f"{where}: routing weights over one bf16 ulp apart"
    )
    _expect_same(f"{where}: on vs off NVFP4 latent", x_fp4, g_fp4, replicated=True)
    _expect_same(f"{where}: on vs off latent scales", x_sf, g_sf, replicated=True)
    _expect_close(f"MOE_LOWM: {where}", on.out, off.out, MOE_TOL, replicated=True)
    # The expert step alone, on the same inputs: the lowm kernel against the CuteDSL backend's kernels.
    layer, backend = BUILT.moe_on.decode_moe, BUILT.moe_off.routed_experts.backend
    lowm = layer._routed(ids, weights, x_fp4, x_sf)
    generic = backend.run_moe_nvfp4(
        x=x_fp4,
        token_selected_experts=ids,
        token_final_scales=weights.float(),
        x_sf=x_sf,
        weight_view=backend._build_local_weight_view(),
    )
    _expect_same(f"{where}: lowm experts vs CuteDSL backend experts", lowm, generic)


def check_moe_lowm():
    """``KimiK3MoERuntime.forward`` at 8, 16 and 32 tokens with and without the lowm decode path (module
    docstring)."""
    with _switch("TRTLLM_K3LL_WIDE_LATENT_TAIL", 1):
        for rows in MOE_ROWS:
            step = _wide_step(rows)
            x = _moe_input(700 + rows, rows)
            off = _moe_off(x, step)
            off.generic = tuple(t.clone() for t in off.generic)
            direct = IMPL.KimiK3MoERuntime.forward_generic(BUILT.moe_off, x)
            _verify_moe(f"MOE_LOWM {rows} rows", _moe_on(x, step), off, direct)
        for rows in MOE_ROWS:
            step = _wide_step(rows)
            x = _moe_input(800 + rows, rows)
            graph, (g_off, g_on) = _capture(lambda: (_moe_off(x, step), _moe_on(x, step)))
            for rep in range(REPLAYS):
                x.copy_(_moe_input(900 + 10 * rows + rep, rows))
                _replay(graph)
                where = f"MOE_LOWM graph {rows} rows replay {rep}"
                direct = IMPL.KimiK3MoERuntime.forward_generic(BUILT.moe_off, x)
                _expect_same(f"{where}: on vs eager", g_on.out, _moe_on(x, step).out)
                _verify_moe(where, g_on, g_off, direct)
            del graph
    if R.rank == 0:
        print(
            f"[rank 0] MOE_LOWM: off arm = forward_generic; on arm max rel err {STATS.get('MOE_LOWM', 0.0):.2e}; "
            "experts bit for bit the CuteDSL backend's",
            flush=True,
        )


CHECKS = [
    check_front_sort,
    check_front16,
    check_wide_ar_attn_res,
    check_wide_attn_res_max_tokens,
    check_route_quant_fp32,
    check_wide_latent_tail,
    check_moe_lowm,
]


def _run_one_rank(args) -> int:
    global R, IMPL, DC, DM, DML, FLAGS, LOWM_OP
    R = ls.Rank(args)
    import importlib

    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
    IMPL = importlib.import_module(f"{package}.implementation")
    DC = importlib.import_module(f"{package}.decode_comm")
    DM = importlib.import_module(f"{package}.decode_moe")
    DML = importlib.import_module(f"{package}.decode_moe_lowm")
    FLAGS = importlib.import_module(f"{package}.flags")
    LOWM_OP = importlib.import_module(
        "tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.nvfp4_lowm_op"
    )
    # First, before anything exists as an inference tensor: the decode paths as post_load_weights builds them.
    code = ls.run_checks(R, [check_build])
    if code:
        return code
    with torch.inference_mode():
        code = ls.run_checks(R, CHECKS)
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")

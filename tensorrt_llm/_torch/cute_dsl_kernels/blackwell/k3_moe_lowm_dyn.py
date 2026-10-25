# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-batch NVFP4 routed MoE (FC1 + SiTU + FC2 + finalize) in one persistent grid with dynamic work
distribution (SM100/SM103).

Same results as the two kernels of :mod:`k3_moe_lowm`: FC1 output rows and scales bit-identical, every
FC2 partial bit-identical and added with ``red.global.add.noftz.v4.bf16x2`` in hardware order.

* Work units: an FC1 unit is one 128-row block of one touched expert's W1 over the whole K
  (``k1 / 256`` stages); an FC2 unit is ``fc2_chunk`` consecutive 128-row blocks of one expert's W2
  (``k2 / 256`` stages each). A stage is 128 x 256 NVFP4 weights plus their scale block (18 KB) in both
  GEMMs, so one TMA ring serves both. The work list is every FC1 unit (tile-major) followed by every
  FC2 unit (tile-major).
* Dynamic distribution: the weight TMA warp of every CTA takes the next unit of the list from a global
  counter (one ``atom.add`` per unit, issued while the current unit's stages stream) and publishes it to
  the other warps through a shared-memory queue. A CTA therefore runs FC1 units and then FC2 units,
  never the other way round, and the SMs that stream faster or start earlier take more units: there is
  no wave quantization and no grid boundary between FC1 and FC2.
* FC2's token operand (the FC1 output rows and scales of the unit's tile) goes through the weight
  ring: the weight TMA warp reserves one ring slot (two at 32 tokens) in front of the unit's weight
  stages, and the token warp fills it as soon as the unit is published, so the load is issued as far
  ahead of the MMA as the ring reaches.
* Output: the kernel zeroes the BF16 output itself (no memset before it). The token warp of the CTA
  that runs FC1 unit ``u < _ZERO_SLICES`` zeroes slice ``u`` of the output and then adds 1 to every
  tile counter (GPU-scope release after a warp barrier); with no touched expert, CTA 0 zeroes it all.
* Dependencies: after the stores of an FC1 unit the epilogue adds 1 to ``tile_ready[tile]`` (GPU-scope
  release). Before loading the FC1 output rows of a tile, the token warp waits until the counter
  reaches ``n1 / 128 + _ZERO_SLICES`` (every FC1 unit of the tile stored, the whole output zeroed;
  acquire, then a generic-to-async proxy fence before the TMA). FC1 units have no dependencies, the
  zeroing slices belong to the first FC1 units, and every CTA finishes its FC1 units before its FC2
  units, so the grid cannot deadlock, also when not all CTAs are resident at once.
* Counter reset (CUDA graph replays reuse the buffer): every FC2 block adds 1 more to its tile's counter
  after the wait; the add that brings it to ``n1 / 128 + _ZERO_SLICES + n2 / 128`` stores zero. Every
  CTA adds 1 to an exit counter (acquire-release) after its last fetch; the CTA that brings it to the
  grid size stores zero to the work and exit counters.
* Warp roles: epilogue warpgroups (warpgroup 0 also converts the token scales and runs every FC1
  epilogue: SiTU, NVFP4 quantization; the FC2 blocks alternate between the warpgroups: BF16 scaling,
  16-byte reds), the MMA warp, the weight TMA warp, the token warp (the step's tokens once, then the
  token operand of every FC2 unit).
* Startup: the weight TMA warp issues its first fetch and loads the experts of the first 32 tiles
  together with the tile count, so the first weight load waits for one memory latency.
* Self-sort (``self_sort_local``, at most 16 tokens): after the wait every thread of the CTA sorts the
  step's top-k ids over the local experts into shared memory (``_self_sort``: per-token bitmaps, their
  union and prefix counts, then each (token, k)'s tile and row), and the kernel reads the tile
  metadata from there instead of from the previous grid: the routing kernel need not sort it.
* PDL: every warp waits on the previous grid (tile metadata, tokens and token scales come from it)
  before it reads any of them or the counters; each CTA triggers its dependents after all of its
  warps are done (later than the safe default of right after the wait).
"""

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from .k3_moe_lowm import (
    _EVICT_FIRST,
    _KBLK,
    _MMA_M,
    _SF_KBLK_BYTES,
    _SF_VEC,
    _SMEM_CAPACITY,
    SUPPORTED_N_TOK,
    _asm,
    _atom_add_relaxed,
    _bulk_g2s,
    _globaltimer,
    _iv,
    _ld_acquire_u32,
    _ldg_u32,
    _lds_v4_f32,
    _red_add_v8_bf16_from_smem,
    _s2t_copy_and_partition,
    _signal_add_release,
    _st_relaxed_u32,
    _stg_u8,
    _stg_u64,
    _stg_v2_b32,
    _sts_f32,
    _sts_u16,
    _sts_u32,
)
from .utils import fmin, griddepcontrol_launch_dependents, griddepcontrol_wait

_SACC_PAD = 4
#: Named barrier of epilogue warpgroup ``wg``: ``_EPI_BARRIER_BASE + wg``.
_EPI_BARRIER_BASE = 1
#: int32 words of the counter buffer before the per-tile counters: [0] work, [1] exited CTAs.
CTR_TILE_BASE = 16
#: Bytes of the weight part of one ring stage (128 x 256 NVFP4).
_SLOT_BYTES = _MMA_M * _KBLK // 2
#: Slices of the output zeroed by the kernel (the CTAs that run FC1 units 0 .. _ZERO_SLICES - 1).
_ZERO_SLICES = 4


def counter_words(max_tiles: int) -> int:
    """int32 words of the (zero-initialized) counter buffer for up to ``max_tiles`` tiles."""
    return CTR_TILE_BASE + max_tiles


@dsl_user_op
def _atom_add_acq_rel(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atom_max_u64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.max.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_min_u64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.min.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_v4_zero(addr, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "st.global.v4.b32 [$0], {0, 0, 0, 0};",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_release(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _popc(value, *, loc=None, ip=None):
    """``popc.b32``."""
    return Int32(
        _asm(T.i32(), [_iv(value, loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", loc=loc, ip=ip)
    )


def _align(v: int, a: int) -> int:
    return (v + a - 1) // a * a


class K3MoeLowMDynKernel:
    """FC1 + SiTU + NVFP4 quantization + FC2 + finalize for at most 32 tokens in one dynamically
    scheduled persistent grid.

    :param n_tok: Token tile (MMA N) of both GEMMs: 8, 16 or 32.
    :param n1: Rows of one expert's W1 (2 x intermediate, gate/up interleaved every 64 rows).
    :param k1: Hidden size (W1 K, W2 rows).
    :param topk: Experts per token.
    :param situ_beta: SiTU gate-side constant.
    :param situ_linear_beta: SiTU up-side constant.
    :param num_acc: Accumulator stages in TMEM (shared by FC1 units and FC2 blocks).
    :param num_epi_wg: Epilogue warpgroups (1 or 2; one if two leave the ring too small).
    :param fc2_chunk: 128-row W2 blocks of one tile per FC2 unit (at most; reduced until it divides
        ``k1 / 128`` and the unit fits in the ring).
    :param queue_depth: Entries of the shared-memory unit queue.
    :param max_ab_stage: Cap on the weight ring depth (the rest of shared memory sizes it).
    :param use_pdl: Launch with programmatic dependent launch and wait on the previous grid.
    :param self_sort_local: 0: the tile metadata (``moe_sort``'s four outputs) comes from the
        previous grid. ``L > 0`` (at most 16 tokens): the kernel sorts the step's routing itself, in
        every CTA, from the top-k ids of the tokens over the ``L`` local experts at a runtime offset
        (the same tiles, rows and order as ``k3_moe_front``'s sort: one tile per touched local
        expert in expert order, its rows the selecting tokens' expanded indices in token order),
        into shared memory, and reads the metadata from there.
    """

    def __init__(
        self,
        n_tok: int,
        n1: int,
        k1: int,
        topk: int,
        situ_beta: float,
        situ_linear_beta: float,
        num_acc: int = 4,
        num_epi_wg: int = 2,
        fc2_chunk: int = 1,
        queue_depth: int = 16,
        max_ab_stage: int = 16,
        use_pdl: bool = True,
        trace: bool = False,
        self_sort_local: int = 0,
    ):
        assert n_tok in SUPPORTED_N_TOK, f"n_tok {n_tok} not in {SUPPORTED_N_TOK}"
        assert n1 % (2 * _MMA_M) == 0 and k1 % _KBLK == 0 and (n1 // 2) % _KBLK == 0
        assert k1 % _MMA_M == 0
        assert num_epi_wg in (1, 2) and num_acc >= num_epi_wg
        assert situ_beta > 0 and situ_linear_beta > 0
        self.n_tok = n_tok
        self.n1, self.k1 = n1, k1
        self.n2, self.k2 = k1, n1 // 2
        self.topk = topk
        self.nkb1, self.nblk1 = k1 // _KBLK, n1 // _MMA_M
        self.nkb2, self.nblk2 = self.k2 // _KBLK, self.n2 // _MMA_M
        self.situ_beta = float(situ_beta)
        self.situ_linear_beta = float(situ_linear_beta)
        self.num_acc = num_acc
        self.queue_depth = queue_depth
        self.use_pdl = use_pdl
        self.trace = trace
        # Self-sort scratch (int32 words): the step's ids, per-token bitmaps of the local experts, their
        # union and prefix counts, the tile count, each tile's expert and row limit, and the rows of
        # each tile at stride n_tok (the permuted-index view spans 128 more words).
        self.self_sort = self_sort_local > 0
        assert not self.self_sort or n_tok <= 16, "self-sort: at most 16 tokens"
        self.ss_words = max(1, -(-self_sort_local // 32))
        self.ss_tiles = max(1, min(n_tok * topk, self_sort_local))
        self.ss_ids = n_tok * topk if self.self_sort else 4
        self.ss_bits = n_tok * self.ss_words if self.self_sort else 4
        self.ss_perm = self.ss_tiles * n_tok + _MMA_M if self.self_sort else 4
        ss_words_total = (
            self.ss_ids + self.ss_bits + 2 * _align(self.ss_words, 4) + 4 + 2 * _align(self.ss_tiles, 4)
            + self.ss_perm
        )  # fmt: skip
        self.ss_bytes = 4 * ss_words_total + 8 * 16 if self.self_sort else 0
        self.mma_tiler = (_MMA_M, n_tok, _KBLK)
        self.acc_cols = n_tok * num_acc
        self.sfa_cols = 16
        self.sfb_cols = 16
        cols = self.acc_cols + self.sfa_cols + self.sfb_cols
        self.tmem_cols = 32
        while self.tmem_cols < cols:
            self.tmem_cols *= 2
        self.a_stage_bytes = _SLOT_BYTES + _SF_KBLK_BYTES
        self.b_stage_bytes = n_tok * _KBLK // 2
        assert _SLOT_BYTES % self.b_stage_bytes == 0
        # token-operand stages (n_tok x 256) per ring slot
        self.b_per_slot = _SLOT_BYTES // self.b_stage_bytes
        self.x_bytes = self.nkb1 * self.b_stage_bytes
        # FC2 token operand in the ring: its rows in one slot, its scales after them in the same slot
        # if they fit, else at the start of the next slot
        self.b2_data_bytes = self.nkb2 * self.b_stage_bytes
        self.b2_sf_bytes = self.nkb2 * _SF_KBLK_BYTES
        self.b2_sf_off = _align(self.b2_data_bytes, _SF_KBLK_BYTES)
        assert self.b2_data_bytes <= _SLOT_BYTES and self.b2_sf_bytes <= _SLOT_BYTES
        self.b2_slots = 1 if self.b2_sf_off + self.b2_sf_bytes <= _SLOT_BYTES else 2
        self.sacc_row_floats = _MMA_M + _SACC_PAD
        self.out_row_bytes = _MMA_M * 2

        # Region right after the ring's weight slots: FC1's token operand [0, x_bytes) and token
        # scales [sfb_off, sfb_off + nkb1 * 2 KB), then the FC2 output staging buffers of each
        # epilogue warpgroup. Two buffers per warpgroup (alternating) and up to num_epi_wg
        # warpgroups, fewer if they leave the ring too small for an FC2 unit with one slot to spare.
        self.sfb_off = _align(self.x_bytes, 1024)
        p1_end = self.sfb_off + self.nkb1 * _SF_KBLK_BYTES
        need = self.b2_slots + self.nkb2 + 1
        for wgs, sout_bufs in ((2, 2), (2, 1), (1, 2), (1, 1)):
            if wgs > num_epi_wg:
                continue
            self.num_epi_wg = wgs
            self.sout_bufs = sout_bufs
            self.sout_bytes = sout_bufs * n_tok * self.out_row_bytes
            self.sout_off = [_align(p1_end, 128) + wg * self.sout_bytes for wg in range(wgs)]
            self.region_bytes = _align(self.sout_off[-1] + self.sout_bytes, 1024)
            fixed = (
                self.region_bytes
                + n_tok * self.sacc_row_floats * 4
                + 2 * wgs * n_tok * 4
                + 2 * queue_depth * 4
                + 2 * 1024  # barriers, TMEM holder and alignment slack
                + self.ss_bytes
            )
            self.num_ab_stage = min(max_ab_stage, (_SMEM_CAPACITY - fixed) // self.a_stage_bytes)
            if self.num_ab_stage >= need:
                break
        assert self.num_ab_stage >= need, "ring too small for FC2"
        assert num_acc >= self.num_epi_wg
        self.epi_warps = 4 * self.num_epi_wg
        self.mma_warp = self.epi_warps
        self.tma_warp = self.epi_warps + 1
        self.tok_warp = self.epi_warps + 2
        self.threads = 32 * (self.epi_warps + 3)
        # FC2 unit: the largest number of blocks <= fc2_chunk that divides n2 / 128 and whose slots
        # (token operand + weights) fit in the ring at once with one to spare
        chunk = max(1, fc2_chunk)
        while self.nblk2 % chunk or self.b2_slots + chunk * self.nkb2 + 1 > self.num_ab_stage:
            chunk -= 1
        self.fc2_chunk = chunk
        self.nchunk2 = self.nblk2 // chunk

    def smem_bytes(self) -> int:
        """Shared memory of one CTA (after ``__call__`` has built the storage)."""
        return self.shared_storage.size_in_bytes() if hasattr(self, "shared_storage") else -1

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w1: cute.Tensor,
        w1_sf: cute.Tensor,
        h: cute.Tensor,
        w2: cute.Tensor,
        w2_sf: cute.Tensor,
        x_sf_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        h_sf_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        alpha1: cute.Tensor,
        alpha2: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        global_sf: cute.Tensor,
        token_final_scales: cute.Tensor,
        ctr_ptr: cute.Pointer,
        topk_ids: cute.Tensor,
        local_offset: Int32,
        num_local: Int32,
        orig_m: Int32,
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
        trace_ptr: Optional[cute.Pointer] = None,
    ):
        self.ab_dtype = w1.element_type
        self.sf_dtype = w1_sf.element_type
        major = utils.LayoutEnum.from_tensor(w1).mma_major_mode()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            major,
            major,
            self.sf_dtype,
            _SF_VEC,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        S = self.num_ab_stage
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.ab_dtype, S)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, _SF_VEC, S)
        # Token-operand views over [ring weight slots | region]: FC2's stages live in ring slots,
        # FC1's (the step's tokens) in the region that follows the slots.
        self.b_view_stages = (S * _SLOT_BYTES + self.x_bytes) // self.b_stage_bytes
        self.sf_view_stages = (S * _SLOT_BYTES + self.sfb_off) // _SF_KBLK_BYTES + self.nkb1
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.b_view_stages
        )
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, _SF_VEC, self.sf_view_stages
        )
        w1_sf = cute.make_tensor(
            w1_sf.iterator, blockscaled_utils.tile_atom_to_shape_SF(w1.shape, _SF_VEC)
        )
        w2_sf = cute.make_tensor(
            w2_sf.iterator, blockscaled_utils.tile_atom_to_shape_SF(w2.shape, _SF_VEC)
        )
        cl = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_a1, tA1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w1, a_one, self.mma_tiler, tiled_mma, cl.shape
        )
        tma_sfa1, tSFA1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w1_sf, sfa_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int16
        )
        tma_a2, tA2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w2, a_one, self.mma_tiler, tiled_mma, cl.shape
        )
        tma_sfa2, tSFA2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w2_sf, sfa_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int16
        )
        tma_b1, tB1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, x, b_one, self.mma_tiler, tiled_mma, cl.shape
        )
        tma_b2, tB2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, h, b_one, self.mma_tiler, tiled_mma, cl.shape
        )

        NACC, NWG, Q = self.num_acc, self.num_epi_wg, self.queue_depth

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, S]
            a_empty: cute.struct.MemRange[cutlass.Int64, S]
            acc_full: cute.struct.MemRange[cutlass.Int64, NACC]
            acc_empty: cute.struct.MemRange[cutlass.Int64, NACC]
            uq_full: cute.struct.MemRange[cutlass.Int64, Q]
            uq_empty: cute.struct.MemRange[cutlass.Int64, Q]
            tok_full: cutlass.Int64
            sfb_full: cutlass.Int64
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sUnits: cute.struct.Align[cute.struct.MemRange[Int32, Q], 16]
            sSlots: cute.struct.Align[cute.struct.MemRange[Int32, Q], 16]
            # the region must follow the weight slots directly (token-operand views span both)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(a_layout.outer)], 1024
            ]
            sRegion: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.region_bytes], 1024]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_layout)], 1024
            ]
            sAcc: cute.struct.Align[
                cute.struct.MemRange[Float32, self.n_tok * self.sacc_row_floats], 16
            ]
            sTok: cute.struct.Align[cute.struct.MemRange[Int32, NWG * self.n_tok], 16]
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, NWG * self.n_tok], 16]
            sSsIds: cute.struct.Align[cute.struct.MemRange[Int32, self.ss_ids], 16]
            sSsBits: cute.struct.Align[cute.struct.MemRange[Int32, self.ss_bits], 16]
            sSsUnion: cute.struct.Align[cute.struct.MemRange[Int32, _align(self.ss_words, 4)], 16]
            sSsPrefix: cute.struct.Align[cute.struct.MemRange[Int32, _align(self.ss_words, 4)], 16]
            sSsNt: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]
            sSsExpert: cute.struct.Align[cute.struct.MemRange[Int32, _align(self.ss_tiles, 4)], 16]
            sSsLimit: cute.struct.Align[cute.struct.MemRange[Int32, _align(self.ss_tiles, 4)], 16]
            sSsPerm: cute.struct.Align[cute.struct.MemRange[Int32, self.ss_perm], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_a1,
            tA1,
            tma_sfa1,
            tSFA1,
            tma_a2,
            tA2,
            tma_sfa2,
            tSFA2,
            tma_b1,
            tB1,
            tma_b2,
            tB2,
            x_sf_ptr,
            h_ptr,
            h_sf_ptr,
            out_ptr,
            alpha1,
            alpha2,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            global_sf,
            token_final_scales,
            ctr_ptr,
            topk_ids,
            local_offset,
            num_local,
            orig_m,
            trace_ptr,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            cl,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    def _view(self, st, layout, dtype):
        """Tensor of ``layout`` over [ring weight slots | region] (starting at the first slot)."""
        base = st.sA.data_ptr()
        if isinstance(layout, cute.ComposedLayout):
            return cute.make_tensor(cute.recast_ptr(base, layout.inner, dtype=dtype), layout.outer)
        return cute.make_tensor(cute.recast_ptr(base, dtype=dtype), layout)

    def _region_addr(self, st, off: int):
        return (st.sRegion.data_ptr() + off).toint()

    @cute.jit
    def _stamp(self, trace_ptr, bid, ev: int):
        _stg_u64(trace_ptr.toint() + Int64(bid * 16 + ev) * 8, _globaltimer())

    @cute.jit
    def _wg_barrier(self, wg: Int32):
        for i in cutlass.range_constexpr(self.num_epi_wg):
            if wg == i:
                cute.arch.barrier(barrier_id=_EPI_BARRIER_BASE + i, number_of_threads=128)

    # ------------------------------------------------------------------ unit queue
    @cute.jit
    def _q_write(self, st, j: Int32, u: Int32, slot: Int32):
        """Publish the CTA's ``j``-th unit (-1: no more units) and its token-operand ring slot."""
        Q = self.queue_depth
        q = j % Q
        if j >= Q:
            cute.arch.mbarrier_wait(st.uq_empty.data_ptr() + q, ((j // Q) & 1) ^ 1)
        units = st.sUnits.get_tensor(cute.make_layout(Q))
        slots = st.sSlots.get_tensor(cute.make_layout(Q))
        with cute.arch.elect_one():
            units[q] = u
            slots[q] = slot
            cute.arch.mbarrier_arrive(st.uq_full.data_ptr() + q)

    @cute.jit
    def _q_read(self, st, j: Int32):
        """The CTA's ``j``-th unit and its token-operand ring slot (whole warp)."""
        Q = self.queue_depth
        q = j % Q
        cute.arch.mbarrier_wait(st.uq_full.data_ptr() + q, (j // Q) & 1)
        units = st.sUnits.get_tensor(cute.make_layout(Q))
        slots = st.sSlots.get_tensor(cute.make_layout(Q))
        u = units[q]
        slot = slots[q]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(st.uq_empty.data_ptr() + q)
        return u, slot

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_sfa1: cute.CopyAtom,
        mSFA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_sfa2: cute.CopyAtom,
        mSFA2: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        tma_b2: cute.CopyAtom,
        mB2: cute.Tensor,
        x_sf_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        h_sf_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        alpha1: cute.Tensor,
        alpha2: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        global_sf: cute.Tensor,
        token_final_scales: cute.Tensor,
        ctr_ptr: cute.Pointer,
        topk_ids: cute.Tensor,
        local_offset: Int32,
        num_local: Int32,
        orig_m: Int32,
        trace_ptr: Optional[cute.Pointer],
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        cl: cute.Layout,
    ):
        S = self.num_ab_stage
        NACC, Q = self.num_acc, self.queue_depth
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas_, _, _ = cute.arch.grid_dim()
        nctas = Int32(nctas_)

        st = utils.SmemAllocator().allocate(self.shared_storage)
        if warp_idx == self.tma_warp:
            for atom in (tma_a1, tma_sfa1, tma_a2, tma_sfa2, tma_b1, tma_b2):
                cpasync.prefetch_descriptor(atom)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(st.a_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.a_empty.data_ptr() + i, 1)
                for i in cutlass.range_constexpr(NACC):
                    cute.arch.mbarrier_init(st.acc_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.acc_empty.data_ptr() + i, 4)
                for i in cutlass.range_constexpr(Q):
                    cute.arch.mbarrier_init(st.uq_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.uq_empty.data_ptr() + i, self.epi_warps + 2)
                cute.arch.mbarrier_init(st.tok_full, 1)
                cute.arch.mbarrier_init(st.sfb_full, 4)
                cute.arch.mbarrier_init(st.tmem_ready, 1)
                cute.arch.mbarrier_init(st.tmem_free, self.epi_warps)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 0)
        if cutlass.const_expr(self.use_pdl):
            # Tile metadata, tokens, token scales and the zeroed output come from the previous grid;
            # the counters are reset by the previous launch of this kernel, which completed before it.
            griddepcontrol_wait()
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 1)
        if cutlass.const_expr(self.self_sort):
            (
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
            ) = self._self_sort(st, topk_ids, local_offset, num_local, orig_m, tidx)

        # the CTA's first unit and the experts of the first 32 tiles: both loads are in flight while
        # the tile count loads
        first = Int32(0)
        t2e_pre = Int32(0)
        if warp_idx == self.tma_warp:
            lane = cute.arch.lane_idx()
            if lane == 0:
                first = _atom_add_relaxed(ctr_ptr.toint(), 1)
            t2e_pre = tile_idx_to_expert_idx[
                cutlass.min(lane, cute.size(tile_idx_to_expert_idx) - 1)
            ]
        nt = Int32(num_non_exiting_tiles[0])
        w1 = nt * self.nblk1
        total = w1 + nt * self.nchunk2
        thr_mma = tiled_mma.get_slice(0)

        if warp_idx == self.tma_warp:
            self._weight_producer(
                st,
                thr_mma,
                cl,
                a_layout,
                sfa_layout,
                tma_a1,
                mA1,
                tma_sfa1,
                mSFA1,
                tma_a2,
                mA2,
                tma_sfa2,
                mSFA2,
                tile_idx_to_expert_idx,
                ctr_ptr,
                first,
                t2e_pre,
                w1,
                total,
                nctas,
                trace_ptr,
                bid,
            )
        elif warp_idx == self.tok_warp:
            self._token_producer(
                st,
                thr_mma,
                cl,
                b_layout,
                tma_b1,
                mB1,
                tma_b2,
                mB2,
                h_sf_ptr,
                out_ptr,
                ctr_ptr,
                orig_m,
                nt,
                w1,
                trace_ptr,
                bid,
            )
        elif warp_idx == self.mma_warp:
            self._mma(st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, w1, trace_ptr, bid)
        else:
            wg = tidx // 128
            et = tidx % 128
            cute.arch.mbarrier_wait(st.tmem_ready, 0)
            acc_ptr = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
            )
            if wg == 0:
                if w1 > 0:
                    self._token_scales_to_smem(st, x_sf_ptr, orig_m, et)
            self._epilogue(
                st,
                tiled_mma,
                acc_ptr,
                h_ptr,
                h_sf_ptr,
                out_ptr,
                alpha1,
                alpha2,
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                global_sf,
                token_final_scales,
                ctr_ptr,
                w1,
                et,
                wg,
                trace_ptr,
                bid,
            )
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(st.tmem_free)

        # dependents launch once every warp of every CTA is done
        cute.arch.barrier()
        if cutlass.const_expr(self.trace):
            if tidx == 0:
                self._stamp(trace_ptr, bid, 12)
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------ self-sort
    @cute.jit
    def _self_sort(self, st, topk_ids, local_offset: Int32, num_local: Int32, orig_m: Int32, tidx):
        """The step's tile metadata in shared memory, by every thread of the CTA (``k3_moe_front``'s
        ``_sort_local``): ``(expert of tile, tile * 128 + rows of tile, permuted -> expanded index,
        tile count)`` as tensors indexed like ``moe_sort``'s outputs (the permuted view maps
        ``tile * 128 + r`` to word ``tile * n_tok + r``)."""
        n_tok, K, W, nthr = self.n_tok, self.topk, self.ss_words, self.threads
        ids = st.sSsIds.get_tensor(cute.make_layout(self.ss_ids))
        bits = st.sSsBits.get_tensor(cute.make_layout(self.ss_bits))
        union = st.sSsUnion.get_tensor(cute.make_layout(_align(W, 4)))
        prefix = st.sSsPrefix.get_tensor(cute.make_layout(_align(W, 4)))
        nt = st.sSsNt.get_tensor(cute.make_layout(1))
        expert = st.sSsExpert.get_tensor(cute.make_layout(self.ss_tiles))
        limit = st.sSsLimit.get_tensor(cute.make_layout(self.ss_tiles))
        perm = st.sSsPerm.get_tensor(cute.make_layout((_MMA_M, self.ss_tiles), stride=(1, n_tok)))
        perm_flat = st.sSsPerm.get_tensor(cute.make_layout(self.ss_perm))
        n = orig_m * Int32(K)
        for i in cutlass.range_constexpr(-(-(n_tok * K) // nthr)):
            j = tidx + Int32(i * nthr)
            if j < n:
                ids[j] = topk_ids[j]
        cute.arch.barrier()
        for i in cutlass.range_constexpr(-(-(n_tok * W) // nthr)):
            j = tidx + Int32(i * nthr)
            if j < orig_m * Int32(W):
                t = j // Int32(W)
                w = j - t * Int32(W)
                word = Int32(0)
                for k in cutlass.range_constexpr(K):
                    lx = ids[t * Int32(K) + k] - local_offset
                    hit = (lx >= Int32(0)) & (lx < num_local) & ((lx >> 5) == w)
                    word = word | cutlass.select_(hit, Int32(1) << (lx & 31), Int32(0))
                bits[j] = word
        cute.arch.barrier()
        if tidx < Int32(W):
            u = Int32(0)
            for t in cutlass.range_constexpr(n_tok):
                if Int32(t) < orig_m:
                    u = u | bits[t * W + tidx]
            union[tidx] = u
        cute.arch.barrier()
        if tidx < Int32(W + 1):
            before = Int32(0)
            for w in cutlass.range_constexpr(W):
                if Int32(w) < tidx:
                    before = before + _popc(union[w])
            if tidx < Int32(W):
                prefix[tidx] = before
            else:
                nt[0] = before
        cute.arch.barrier()
        for i in cutlass.range_constexpr(-(-(n_tok * K) // nthr)):
            j = tidx + Int32(i * nthr)
            if j < n:
                t = j // Int32(K)
                lx = ids[j] - local_offset
                if (lx >= Int32(0)) & (lx < num_local):
                    w = lx >> 5
                    b = lx & 31
                    tile = prefix[w] + _popc(union[w] & ((Int32(1) << b) - Int32(1)))
                    row = Int32(0)
                    count = Int32(0)
                    for q in cutlass.range_constexpr(n_tok):
                        if Int32(q) < orig_m:
                            bit = (bits[q * W + w] >> b) & Int32(1)
                            count = count + bit
                            if Int32(q) < t:
                                row = row + bit
                    perm_flat[tile * Int32(n_tok) + row] = j
                    if row == Int32(0):
                        expert[tile] = lx
                        limit[tile] = tile * Int32(_MMA_M) + count
        cute.arch.barrier()
        return expert, limit, perm, nt

    # ------------------------------------------------------------------ producers
    @cute.jit
    def _reserve_slot(self, a_empty, g):
        """Wait until ring stage ``g`` may be overwritten."""
        S = self.num_ab_stage
        stage = g % S
        if g >= S:
            cute.arch.mbarrier_wait(a_empty + stage, ((g // S) & 1) ^ 1)
        return stage

    @cute.jit
    def _load_stage(
        self, tma_a, tma_sfa, tAsA, tAgA, tAsSFA, tAgSFA, blk, kb, expert, a_full, a_empty, g
    ):
        stage = self._reserve_slot(a_empty, g)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, self.a_stage_bytes)
        cute.copy(
            tma_a,
            tAgA[(None, blk, kb, expert)],
            tAsA[(None, stage)],
            tma_bar_ptr=a_full + stage,
            cache_policy=Int64(_EVICT_FIRST),
        )
        cute.copy(
            tma_sfa,
            tAgSFA[(None, blk, kb, expert)],
            tAsSFA[(None, stage)],
            tma_bar_ptr=a_full + stage,
            cache_policy=Int64(_EVICT_FIRST),
        )
        return g + 1

    @cute.jit
    def _expert_of(self, tile_idx_to_expert_idx, t2e_pre: Int32, tile: Int32):
        """Expert of ``tile`` (warp-uniform): from the prefetched lanes for the first 32 tiles."""
        expert = Int32(0)
        if tile < 32:
            expert = cute.arch.shuffle_sync(t2e_pre, tile)
        else:
            expert = tile_idx_to_expert_idx[tile]
        return expert

    @cute.jit
    def _weight_producer(
        self,
        st,
        thr_mma,
        cl,
        a_layout,
        sfa_layout,
        tma_a1,
        mA1,
        tma_sfa1,
        mSFA1,
        tma_a2,
        mA2,
        tma_sfa2,
        mSFA2,
        tile_idx_to_expert_idx,
        ctr_ptr,
        first: Int32,
        t2e_pre: Int32,
        w1: Int32,
        total: Int32,
        nctas: Int32,
        trace_ptr,
        bid,
    ):
        """Fetches the CTA's units from the work counter (one ahead; ``first``: lane 0's first
        fetch; ``t2e_pre``: lane i holds the expert of tile i), reserves the ring slot(s) of an FC2
        unit's token operand, publishes the unit and streams its weight stages through the ring;
        the CTA that exits last resets the work counter."""
        a_full, a_empty = st.a_full.data_ptr(), st.a_empty.data_ptr()
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sSFA = st.sSFA.get_tensor(sfa_layout)
        a_cta = cute.make_layout(cute.slice_(cl, (0, 0, None, 0)).shape)
        tiler = cute.slice_(self.mma_tiler, (None, 0, None))
        parts = []
        for mA, mSFA, tma_a, tma_sfa in (
            (mA1, mSFA1, tma_a1, tma_sfa1),
            (mA2, mSFA2, tma_a2, tma_sfa2),
        ):
            tCgA = thr_mma.partition_A(cute.local_tile(mA, tiler, (None, None, None)))
            tCgSFA = thr_mma.partition_A(cute.local_tile(mSFA, tiler, (None, None, None)))
            tAsA, tAgA = cpasync.tma_partition(
                tma_a, 0, a_cta, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
            )
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_sfa, 0, a_cta, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3)
            )
            parts.append((tAsA, tAgA, cute.filter_zeros(tAsSFA), cute.filter_zeros(tAgSFA)))
        tAsA1, tAgA1, tAsSFA1, tAgSFA1 = parts[0]
        tAsA2, tAgA2, tAsSFA2, tAgSFA2 = parts[1]
        lane = cute.arch.lane_idx()
        ctr = ctr_ptr.toint()
        cur = cute.arch.shuffle_sync(first, 0)
        g = Int32(0)
        j = Int32(0)
        n1u = Int32(0)
        while cur < total:
            nxt = Int32(0)
            if lane == 0:
                nxt = _atom_add_relaxed(ctr, 1)
            if cur < w1:
                self._q_write(st, j, cur, Int32(-1))
                tile = cur // self.nblk1
                blk = cur - tile * self.nblk1
                expert = self._expert_of(tile_idx_to_expert_idx, t2e_pre, tile)
                if cutlass.const_expr(self.trace):
                    if j == 0:
                        with cute.arch.elect_one():
                            self._stamp(trace_ptr, bid, 2)
                for kb in cutlass.range(self.nkb1, unroll=1):
                    g = self._load_stage(
                        tma_a1,
                        tma_sfa1,
                        tAsA1,
                        tAgA1,
                        tAsSFA1,
                        tAgSFA1,
                        blk,
                        kb,
                        expert,
                        a_full,
                        a_empty,
                        g,
                    )
                n1u += 1
            else:
                # the token warp fills the reserved slot(s) with the unit's token operand
                slot = self._reserve_slot(a_empty, g)
                g += 1
                if cutlass.const_expr(self.b2_slots > 1):
                    self._reserve_slot(a_empty, g)
                    g += 1
                self._q_write(st, j, cur, slot)
                v = cur - w1
                tile = v // self.nchunk2
                c = v - tile * self.nchunk2
                expert = self._expert_of(tile_idx_to_expert_idx, t2e_pre, tile)
                for sb in cutlass.range(self.fc2_chunk, unroll=1):
                    for kb in cutlass.range(self.nkb2, unroll=1):
                        g = self._load_stage(
                            tma_a2,
                            tma_sfa2,
                            tAsA2,
                            tAgA2,
                            tAsSFA2,
                            tAgSFA2,
                            c * self.fc2_chunk + sb,
                            kb,
                            expert,
                            a_full,
                            a_empty,
                            g,
                        )
            cur = cute.arch.shuffle_sync(nxt, 0)
            j += 1
        self._q_write(st, j, Int32(-1), Int32(-1))
        if cutlass.const_expr(self.trace):
            with cute.arch.elect_one():
                self._stamp(trace_ptr, bid, 3)
                _stg_u64(trace_ptr.toint() + Int64(bid * 16 + 10) * 8, Int64(n1u * 1000 + j - n1u))
        if lane == 0:
            # every fetch of this CTA happens before its release; the last CTA's acquire sees all
            if _atom_add_acq_rel(ctr + 4, 1) == nctas - 1:
                _st_relaxed_u32(ctr, 0)
                _st_relaxed_u32(ctr + 4, 0)

    @cute.jit
    def _token_producer(
        self,
        st,
        thr_mma,
        cl,
        b_layout,
        tma_b1,
        mB1,
        tma_b2,
        mB2,
        h_sf_ptr,
        out_ptr,
        ctr_ptr,
        orig_m: Int32,
        nt: Int32,
        w1: Int32,
        trace_ptr,
        bid,
    ):
        """The step's tokens (FC1's token operand) once; for FC1 unit ``u < _ZERO_SLICES`` slice
        ``u`` of the output is zeroed and every tile counter released once more; then for every
        FC2 unit the FC1 output rows and scales of its tile into the unit's ring slot(s), once FC1
        has stored all of the tile's rows and the output is zeroed."""
        S = self.num_ab_stage
        b_cta = cute.make_layout(cute.slice_(cl, (0, None, 0, 0)).shape)
        tiler = cute.slice_(self.mma_tiler, (0, None, None))
        sB = self._view(st, b_layout, self.ab_dtype)
        tCgB1 = thr_mma.partition_B(cute.local_tile(mB1, tiler, (None, None, None)))
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_b1, 0, b_cta, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB1, 0, 3)
        )
        tCgB2 = thr_mma.partition_B(cute.local_tile(mB2, tiler, (None, None, None)))
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_b2, 0, b_cta, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB2, 0, 3)
        )
        x_stage0 = S * self.b_per_slot
        if w1 > 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(st.tok_full, self.x_bytes)
            for kb in cutlass.range(self.nkb1, unroll=1):
                cute.copy(
                    tma_b1,
                    tBgB1[(None, 0, kb, 0)],
                    tBsB1[(None, x_stage0 + kb)],
                    tma_bar_ptr=st.tok_full,
                )
        a_full = st.a_full.data_ptr()
        slot_base = st.sA.data_ptr().toint()
        h_sf_base = h_sf_ptr.toint()
        ctr = ctr_ptr.toint()
        lane = cute.arch.lane_idx()
        w_poll = Int64(0)
        out_bytes = orig_m * (self.n2 * 2)
        slice_bytes = (out_bytes + _ZERO_SLICES * 16 - 1) // (_ZERO_SLICES * 16) * 16
        out_base = out_ptr.toint()
        if nt == 0:
            # no local expert: no FC2 adds, CTA 0 zeroes the whole output
            if bid == 0:
                for off in cutlass.range(lane * 16, out_bytes, 32 * 16, unroll=1):
                    _stg_v4_zero(out_base + Int64(off))
        e = Int32(0)
        u, slot = self._q_read(st, e)
        while u >= 0:
            if u < _ZERO_SLICES:
                # zero slice u, then release every tile counter once more: FC2 adds of a tile wait
                # for all slices (the warp barrier orders every lane's stores before the releases)
                lo = u * slice_bytes
                hi = cutlass.min(lo + slice_bytes, out_bytes)
                for off in cutlass.range(lo + lane * 16, hi, 32 * 16, unroll=1):
                    _stg_v4_zero(out_base + Int64(off))
                cute.arch.sync_warp()
                for t in cutlass.range(lane, nt, 32, unroll=1):
                    _red_add_release(ctr + Int64(CTR_TILE_BASE + t) * 4, 1)
            if u >= w1:
                v = u - w1
                tile = v // self.nchunk2
                ready_addr = ctr + Int64(CTR_TILE_BASE + tile) * 4
                if cutlass.const_expr(self.trace):
                    t_poll = _globaltimer()
                    nct, _, _ = cute.arch.grid_dim()
                    tt = trace_ptr.toint() + (Int64(nct) * 16 + Int64(tile) * 4) * 8
                    with cute.arch.elect_one():
                        _atom_min_u64(tt + 8, t_poll)
                ready = _ld_acquire_u32(ready_addr)
                while ready < self.nblk1 + _ZERO_SLICES:
                    ready = _ld_acquire_u32(ready_addr)
                cute.arch.fence_proxy("async.global")
                if cutlass.const_expr(self.trace):
                    t_done = _globaltimer()
                    w_poll += t_done - t_poll
                    with cute.arch.elect_one():
                        _atom_max_u64(tt + 16, t_done - t_poll)
                h_sf_src = h_sf_base + Int64(tile) * (_MMA_M * (self.k2 // _SF_VEC))
                if cutlass.const_expr(self.b2_slots == 1):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            a_full + slot, self.b2_data_bytes + self.b2_sf_bytes
                        )
                        _bulk_g2s(
                            slot_base + slot * _SLOT_BYTES + self.b2_sf_off,
                            h_sf_src,
                            self.b2_sf_bytes,
                            (a_full + slot).toint(),
                        )
                else:
                    slot2 = (slot + 1) % S
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(a_full + slot, self.b2_data_bytes)
                        cute.arch.mbarrier_arrive_and_expect_tx(a_full + slot2, self.b2_sf_bytes)
                        _bulk_g2s(
                            slot_base + slot2 * _SLOT_BYTES,
                            h_sf_src,
                            self.b2_sf_bytes,
                            (a_full + slot2).toint(),
                        )
                for kb in cutlass.range(self.nkb2, unroll=1):
                    cute.copy(
                        tma_b2,
                        tBgB2[(None, tile * (_MMA_M // self.n_tok), kb, 0)],
                        tBsB2[(None, slot * self.b_per_slot + kb)],
                        tma_bar_ptr=a_full + slot,
                    )
                # counter reset: every FC2 block of the tile adds 1 after the wait; the last zeroes it
                if lane == 0:
                    old = _atom_add_relaxed(ready_addr, self.fc2_chunk)
                    if old + self.fc2_chunk == self.nblk1 + _ZERO_SLICES + self.nblk2:
                        _st_relaxed_u32(ready_addr, 0)
            e += 1
            u, slot = self._q_read(st, e)
        if cutlass.const_expr(self.trace):
            with cute.arch.elect_one():
                _stg_u64(trace_ptr.toint() + Int64(bid * 16 + 8) * 8, w_poll)

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def _mma(
        self, st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, w1: Int32, trace_ptr, bid
    ):
        S = self.num_ab_stage
        NACC = self.num_acc
        a_full, a_empty = st.a_full.data_ptr(), st.a_empty.data_ptr()
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        cute.arch.alloc_tmem(self.tmem_cols, st.tmem_holding_buf)
        cute.arch.relinquish_tmem_alloc_permit()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(st.tmem_ready)
        acc_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tAcc_base = cute.make_tensor(
            acc_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, NACC)).layout
        )
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(acc_ptr + self.acc_cols, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, _SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            ),
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(acc_ptr + self.acc_cols + self.sfa_cols, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, _SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
            ),
        )
        sSFA = st.sSFA.get_tensor(sfa_layout)
        s2t_sfa, tCsSFA, tCtSFA_s2t = _s2t_copy_and_partition(sSFA, tCtSFA, self.sf_dtype)
        s2t_sfb, tCsSFB, tCtSFB_s2t = _s2t_copy_and_partition(
            self._view(st, sfb_layout, self.sf_dtype), tCtSFB, self.sf_dtype
        )
        tCrA = tiled_mma.make_fragment_A(st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner))
        tCrB = tiled_mma.make_fragment_B(self._view(st, b_layout, self.ab_dtype))
        nkp = cute.size(tCrA, mode=[2])
        sf_per_slot = _SLOT_BYTES // _SF_KBLK_BYTES
        x_b0 = S * self.b_per_slot
        x_sf0 = (S * _SLOT_BYTES + self.sfb_off) // _SF_KBLK_BYTES

        g = Int32(0)
        j = Int32(0)
        first1 = Int32(1)
        in_fc2 = Int32(0)
        w_a = Int64(0)
        w_b2 = Int64(0)
        w_acc = Int64(0)
        t_w = Int64(0)
        e = Int32(0)
        u, slot = self._q_read(st, e)
        while u >= 0:
            nsub = Int32(1)
            nst = Int32(self.nkb1)
            b0 = Int32(x_b0)
            sf0 = Int32(x_sf0)
            if u < w1:
                if first1 > 0:
                    first1 = Int32(0)
                    cute.arch.mbarrier_wait(st.tok_full, 0)
                    cute.arch.mbarrier_wait(st.sfb_full, 0)
            else:
                if cutlass.const_expr(self.trace):
                    if in_fc2 == 0:
                        in_fc2 = Int32(1)
                        with cute.arch.elect_one():
                            self._stamp(trace_ptr, bid, 14)
                    t_w = _globaltimer()
                # the unit's token operand: ring slot(s) in front of its weight stages
                for _ in cutlass.range_constexpr(self.b2_slots):
                    cute.arch.mbarrier_wait(a_full + g % S, (g // S) & 1)
                    g += 1
                if cutlass.const_expr(self.trace):
                    w_b2 += _globaltimer() - t_w
                nsub = Int32(self.fc2_chunk)
                nst = Int32(self.nkb2)
                b0 = slot * self.b_per_slot
                if cutlass.const_expr(self.b2_slots == 1):
                    sf0 = (slot * _SLOT_BYTES + self.b2_sf_off) // _SF_KBLK_BYTES
                else:
                    sf0 = ((slot + 1) % S) * sf_per_slot
            for sb in cutlass.range(nsub, unroll=1):
                acc_stage = j % NACC
                if j >= NACC:
                    if cutlass.const_expr(self.trace):
                        t_w = _globaltimer()
                    cute.arch.mbarrier_wait(acc_empty + acc_stage, ((j // NACC) & 1) ^ 1)
                    if cutlass.const_expr(self.trace):
                        w_acc += _globaltimer() - t_w
                tAcc = tAcc_base[(None, None, None, acc_stage)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range(nst, unroll=1):
                    stage = g % S
                    if cutlass.const_expr(self.trace):
                        t_w = _globaltimer()
                    cute.arch.mbarrier_wait(a_full + stage, (g // S) & 1)
                    if cutlass.const_expr(self.trace):
                        w_a += _globaltimer() - t_w
                    cute.copy(s2t_sfa, tCsSFA[(None, None, None, None, stage)], tCtSFA_s2t)
                    cute.copy(s2t_sfb, tCsSFB[(None, None, None, None, sf0 + kb)], tCtSFB_s2t)
                    for kp in cutlass.range_constexpr(nkp):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                        cute.gemm(
                            tiled_mma,
                            tAcc[(None, 0, 0)],
                            tCrA[(None, 0, kp, stage)],
                            tCrB[(None, 0, kp, b0 + kb)],
                            tAcc[(None, 0, 0)],
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(a_empty + stage)
                    g += 1
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + acc_stage)
                j += 1
            if u >= w1:
                # the token-operand slot(s) are free once the unit's MMAs are done
                with cute.arch.elect_one():
                    tcgen05.commit(a_empty + slot)
                    if cutlass.const_expr(self.b2_slots > 1):
                        tcgen05.commit(a_empty + (slot + 1) % S)
            e += 1
            u, slot = self._q_read(st, e)
        if cutlass.const_expr(self.trace):
            with cute.arch.elect_one():
                self._stamp(trace_ptr, bid, 7)
                base = trace_ptr.toint() + Int64(bid * 16) * 8
                _stg_u64(base + 4 * 8, w_a)
                _stg_u64(base + 5 * 8, w_b2)
                _stg_u64(base + 15 * 8, w_acc)
        cute.arch.mbarrier_wait(st.tmem_free, 0)
        cute.arch.dealloc_tmem(acc_ptr, self.tmem_cols)

    # ------------------------------------------------------------------ epilogues
    @cute.jit
    def _token_scales_to_smem(self, st, x_sf_ptr, orig_m: Int32, tidx: Int32):
        """Token scales, linear [orig_m, k1 / 16], into the MMA's 128x4-blocked SMEM layout (rows
        past orig_m zero), every load in flight before the first store."""
        n_tok = self.n_tok
        nwords = self.nkb1 * 4 * n_tok
        per_thread = (nwords + 127) // 128
        sf_row_bytes = self.k1 // _SF_VEC
        x_sf_base = x_sf_ptr.toint()
        sfb_base = self._region_addr(st, self.sfb_off)
        vals = cute.make_rmem_tensor((per_thread,), Int32)
        for t in cutlass.range_constexpr(per_thread):
            i = tidx + t * 128
            r = i % n_tok
            rest = i // n_tok
            atom = rest % 4
            kb = cutlass.min(rest // 4, self.nkb1 - 1)
            rr = cutlass.min(r, orig_m - 1)
            vals[t] = _ldg_u32(x_sf_base + Int64(rr * sf_row_bytes + kb * 16 + atom * 4))
        for t in cutlass.range_constexpr(per_thread):
            i = tidx + t * 128
            r = i % n_tok
            rest = i // n_tok
            atom = rest % 4
            kb = rest // 4
            if i < nwords:
                v = vals[t]
                if r >= orig_m:
                    v = Int32(0)
                _sts_u32(
                    sfb_base + kb * _SF_KBLK_BYTES + atom * 512 + (r % 32) * 16 + (r // 32) * 4,
                    v,
                )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(st.sfb_full)

    @cute.jit
    def _situ(self, up, gate, alpha_val, out):
        """The packed-f32x2 SiTU sequence of the regular gather kernel's epilogue."""
        beta = self.situ_beta
        linear_beta = self.situ_linear_beta
        LOG2_E = Float32(1.4426950408889634)
        neg_log2e_pair = (-LOG2_E, -LOG2_E)
        one_pair = (Float32(1.0), Float32(1.0))
        inv_2beta = Float32(2.0 / beta)
        two_beta = Float32(2.0 * beta)
        neg_beta = Float32(-beta)
        inv_2lbeta = Float32(2.0 / linear_beta)
        two_lbeta = Float32(2.0 * linear_beta)
        neg_lbeta = Float32(-linear_beta)

        def _sigmoid(p0, p1):
            neg = cute.arch.mul_packed_f32x2((p0, p1), neg_log2e_pair)
            e = (cute.math.exp2(neg[0], fastmath=True), cute.math.exp2(neg[1], fastmath=True))
            d = cute.arch.add_packed_f32x2(e, one_pair)
            return (cute.arch.rcp_approx(d[0]), cute.arch.rcp_approx(d[1]))

        alpha_pair = (Float32(alpha_val), Float32(alpha_val))
        for i in cutlass.range_constexpr(0, 16, 2):
            gg = cute.arch.mul_packed_f32x2((gate[i], gate[i + 1]), alpha_pair)
            uu = cute.arch.mul_packed_f32x2((up[i], up[i + 1]), alpha_pair)
            sigmoid_g = _sigmoid(gg[0], gg[1])
            gs = _sigmoid(*cute.arch.mul_packed_f32x2(gg, (inv_2beta, inv_2beta)))
            tanh_g = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(gs, (two_beta, two_beta)), (neg_beta, neg_beta)
            )
            us = _sigmoid(*cute.arch.mul_packed_f32x2(uu, (inv_2lbeta, inv_2lbeta)))
            tanh_u = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(us, (two_lbeta, two_lbeta)), (neg_lbeta, neg_lbeta)
            )
            situ_gate = cute.arch.mul_packed_f32x2(tanh_g, sigmoid_g)
            out_pair = cute.arch.mul_packed_f32x2(situ_gate, tanh_u)
            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]

    @cute.jit
    def _epilogue(
        self,
        st,
        tiled_mma,
        acc_ptr,
        h_ptr,
        h_sf_ptr,
        out_ptr,
        alpha1,
        alpha2,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        global_sf,
        token_final_scales,
        ctr_ptr,
        w1: Int32,
        et: Int32,
        wg: Int32,
        trace_ptr,
        bid,
    ):
        """Every epilogue warp walks the CTA's units. FC1 units: warpgroup 0 drains the accumulator
        into a token-major staging buffer, applies SiTU and quantizes 16 outputs of one row per thread
        (``K3MoeFc1LowMKernel``'s sequence), stores them and releases the tile counter. FC2 blocks
        (alternating warpgroups, the first one on warpgroup 1): BF16 partials of every valid row into
        the warpgroup's staging buffer (two, alternating), then 16-byte reds into ``out``."""
        NACC, NWG = self.num_acc, self.num_epi_wg
        n_tok = self.n_tok
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tAcc_base = cute.make_tensor(
            acc_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, NACC)).layout
        )
        copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(n_tok)), Float32)
        tAcc_epi = cute.flat_divide(tAcc_base[((None, None), 0, 0, None)], (_MMA_M, n_tok))
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom, tAcc_epi[(None, None, 0, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(et)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((_MMA_M, n_tok)))
        rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()

        # FC1 epilogue state
        sacc_base = st.sAcc.data_ptr().toint()
        col_bytes = self.sacc_row_floats * 4
        h_base = h_ptr.toint()
        h_sf_base = h_sf_ptr.toint()
        h_row_bytes = self.k2 // 2
        tile_sf_bytes = _MMA_M * (self.k2 // _SF_VEC)
        r = et >> 2
        q = et & 3
        norm_const = global_sf[0]
        rcp_limit = Float32(1.0 / 6.0)
        fp32_max = Float32(3.40282346638528859812e38)
        up = cute.make_rmem_tensor((16,), Float32)
        gate = cute.make_rmem_tensor((16,), Float32)
        tCompute = cute.make_rmem_tensor((16,), Float32)
        rSF = cute.make_rmem_tensor((2,), self.sf_dtype)
        rQ = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
        rPv = cute.make_rmem_tensor((2,), Float32)
        ctr = ctr_ptr.toint()

        # FC2 epilogue state
        rBf = cute.make_rmem_tensor((2,), cutlass.BFloat16)
        out_base = out_ptr.toint()
        out_row_bytes = self.n2 * 2
        stage_bytes = n_tok * self.out_row_bytes
        sout_base = self._region_addr(st, self.sout_off[0])
        if cutlass.const_expr(NWG > 1):
            if wg == 1:
                sout_base = self._region_addr(st, self.sout_off[1])
        stok = st.sTok.get_tensor(cute.make_layout((n_tok, NWG), stride=(1, n_tok)))
        sscale = st.sScale.get_tensor(cute.make_layout((n_tok, NWG), stride=(1, n_tok)))
        chunks_per_row = _MMA_M // 8

        prev_tile = Int32(-1)
        n_valid2 = Int32(0)
        nb = Int32(0)
        j = Int32(0)
        f2 = Int32(0)
        e = Int32(0)
        u, _slot = self._q_read(st, e)
        while u >= 0:
            if u < w1:
                if wg == 0:
                    tile = u // self.nblk1
                    blk = u - tile * self.nblk1
                    n_valid = cutlass.min(
                        cutlass.max(tile_idx_to_mn_limit[tile] - tile * _MMA_M, Int32(0)),
                        Int32(n_tok),
                    )
                    alpha_val = alpha1[tile_idx_to_expert_idx[tile]]
                    tok = Int32(0)
                    if r < n_valid:
                        tok = permuted_idx_to_expanded_idx[tile * _MMA_M + r] // self.topk
                    acc_stage = j % NACC
                    cute.arch.mbarrier_wait(acc_full + acc_stage, (j // NACC) & 1)
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, 0, acc_stage)], rAcc)
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(acc_empty + acc_stage)
                    for i in cutlass.range_constexpr(cute.size(rAcc)):
                        row, col = tTR_cAcc[i]
                        _sts_f32(sacc_base + col * col_bytes + row * 4, rAcc[i])
                    self._wg_barrier(Int32(0))
                    if r < n_valid:
                        src = sacc_base + tok * col_bytes + q * 64
                        for vv in cutlass.range_constexpr(4):
                            u0, u1, u2, u3 = _lds_v4_f32(src + vv * 16)
                            g0, g1, g2, g3 = _lds_v4_f32(src + 256 + vv * 16)
                            up[4 * vv] = u0
                            up[4 * vv + 1] = u1
                            up[4 * vv + 2] = u2
                            up[4 * vv + 3] = u3
                            gate[4 * vv] = g0
                            gate[4 * vv + 1] = g1
                            gate[4 * vv + 2] = g2
                            gate[4 * vv + 3] = g3
                        self._situ(up, gate, alpha_val, tCompute)
                        frg = cute.logical_divide(tCompute, cute.make_layout(_SF_VEC))
                        acc_frg = frg.load()
                        abs_ir = math.absf(acc_frg.ir_value())
                        abs_frg = type(acc_frg)(abs_ir, acc_frg.shape, acc_frg.dtype)
                        amax = abs_frg[None, 0].reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
                        pv = cute.arch.mul_packed_f32x2(
                            (amax, Float32(0.0)), (rcp_limit, rcp_limit)
                        )
                        pv = cute.arch.mul_packed_f32x2((pv[0], pv[1]), (norm_const, norm_const))
                        rPv[0] = pv[0]
                        rPv[1] = pv[1]
                        rSF.store(rPv.load().to(self.sf_dtype))
                        sf_f32 = rSF.load().to(Float32)
                        acc_scale = cute.arch.mul_packed_f32x2(
                            (cute.arch.rcp_approx(sf_f32[0]), cute.arch.rcp_approx(sf_f32[1])),
                            (norm_const, norm_const),
                        )
                        scale0 = fmin(acc_scale[0], fp32_max, nan=True)
                        for ei in cutlass.range_constexpr(0, 16, 2):
                            tCompute[ei], tCompute[ei + 1] = cute.arch.mul_packed_f32x2(
                                (tCompute[ei], tCompute[ei + 1]), (scale0, scale0)
                            )
                        rQ.store(tCompute.load().to(cutlass.Float4E2M1FN))
                        rQw = cute.recast_tensor(rQ, Int32)
                        rSFb = cute.recast_tensor(rSF, cutlass.Uint8)
                        grow = tile * _MMA_M + r
                        _stg_v2_b32(
                            h_base + Int64(grow) * h_row_bytes + blk * 32 + q * 8, rQw[0], rQw[1]
                        )
                        _stg_u8(
                            h_sf_base
                            + Int64(tile) * tile_sf_bytes
                            + blk * 512
                            + (r % 32) * 16
                            + (r // 32) * 4
                            + q,
                            Int32(rSFb[0]),
                        )
                    self._wg_barrier(Int32(0))
                    # the barrier orders every thread's stores of this unit before the release
                    if et == 0:
                        _signal_add_release(ctr + Int64(CTR_TILE_BASE + tile) * 4)
                        if cutlass.const_expr(self.trace):
                            if j == 0:
                                self._stamp(trace_ptr, bid, 6)
                            self._stamp(trace_ptr, bid, 13)
                            nct, _, _ = cute.arch.grid_dim()
                            _atom_max_u64(
                                trace_ptr.toint() + (Int64(nct) * 16 + Int64(tile) * 4) * 8,
                                _globaltimer(),
                            )
                j += 1
            else:
                v = u - w1
                tile = v // self.nchunk2
                c = v - tile * self.nchunk2
                for sb in cutlass.range(self.fc2_chunk, unroll=1):
                    owner = (f2 + 1) % NWG
                    if owner == wg:
                        blk = c * self.fc2_chunk + sb
                        if tile != prev_tile:
                            prev_tile = tile
                            n_valid2 = cutlass.min(
                                cutlass.max(tile_idx_to_mn_limit[tile] - tile * _MMA_M, Int32(0)),
                                Int32(n_tok),
                            )
                            scale = Float32(0.0)
                            token = Int32(0)
                            if et < n_valid2:
                                expanded = permuted_idx_to_expanded_idx[tile * _MMA_M + et]
                                token = expanded // self.topk
                                slot_k = expanded - token * self.topk
                                scale = alpha2[tile_idx_to_expert_idx[tile]] * Float32(
                                    token_final_scales[(token, slot_k)]
                                )
                            # the previous block's reds still read the previous tile's metadata
                            self._wg_barrier(wg)
                            if et < n_valid2:
                                stok[et, wg] = token
                                sscale[et, wg] = scale
                            self._wg_barrier(wg)
                        acc_stage = j % NACC
                        cute.arch.mbarrier_wait(acc_full + acc_stage, (j // NACC) & 1)
                        cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, 0, acc_stage)], rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(acc_empty + acc_stage)
                        sbuf = sout_base + (nb % self.sout_bufs) * stage_bytes
                        if cutlass.const_expr(self.sout_bufs == 1):
                            # the previous block's reds read this buffer
                            self._wg_barrier(wg)
                        for i in cutlass.range_constexpr(cute.size(rAcc)):
                            row, col = tTR_cAcc[i]
                            if col < n_valid2:
                                rBf[0] = (rAcc[i] * sscale[col, wg]).to(cutlass.BFloat16)
                                rBfw = cute.recast_tensor(rBf, cutlass.Int16)
                                _sts_u16(sbuf + col * self.out_row_bytes + row * 2, Int32(rBfw[0]))
                        self._wg_barrier(wg)
                        qq = et
                        while qq < n_valid2 * chunks_per_row:
                            rr = qq // chunks_per_row
                            cc = (qq - rr * chunks_per_row) * 8
                            _red_add_v8_bf16_from_smem(
                                out_base
                                + Int64(stok[rr, wg]) * out_row_bytes
                                + (blk * _MMA_M + cc) * 2,
                                sbuf + rr * self.out_row_bytes + cc * 2,
                            )
                            qq += 128
                        nb += 1
                    f2 += 1
                    j += 1
            e += 1
            u, _slot = self._q_read(st, e)
        if cutlass.const_expr(self.trace):
            if et == 0:
                if wg == 0:
                    self._stamp(trace_ptr, bid, 9)
                else:
                    self._stamp(trace_ptr, bid, 11)

    # ------------------------------------------------------------------ wrapper
    @cute.jit
    def wrapper(
        self,
        x_ptr: cute.Pointer,
        w1_ptr: cute.Pointer,
        x_sf_ptr: cute.Pointer,
        w1_sf_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        h_sf_ptr: cute.Pointer,
        alpha1_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2_sf_ptr: cute.Pointer,
        alpha2_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        ctr_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        orig_m: Int32,
        m: Int32,
        num_local: Int32,
        local_offset: Int32,
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
        trace_ptr: Optional[cute.Pointer] = None,
    ):
        """``x`` [orig_m, k1] tokens and ``x_sf`` their linear scales; ``h`` / ``h_sf`` the FC1
        output buffers in the permuted layout ([m, k2] rows, 128x4-blocked scales); ``out`` the
        zeroed BF16 [orig_m, k1] output; ``ctr`` int32 [counter_words(m // 128)] zeros.
        ``topk_ids`` int32 [orig_m, topk] and ``local_offset``: read only with ``self_sort_local``
        (then the four tile-metadata pointers are not read)."""
        n1, k1, n2, k2 = self.n1, self.k1, self.n2, self.k2
        L = num_local
        num_tiles = m // _MMA_M

        def kmajor(ptr, rows, cols, batch):
            return cute.make_tensor(
                ptr, layout=cute.make_ordered_layout((rows, cols, batch), order=(1, 0, 2))
            )

        def blocked_sf(ptr, rows, cols, batch):
            return cute.make_tensor(
                ptr,
                layout=cute.make_ordered_layout(
                    (32, 4, rows // 128, 4, cols // (_SF_VEC * 4), batch), order=(2, 1, 4, 0, 3, 5)
                ),
            )

        self(
            kmajor(x_ptr, orig_m, k1, 1),
            kmajor(w1_ptr, n1, k1, L),
            blocked_sf(w1_sf_ptr, n1, k1, L),
            kmajor(h_ptr, m, k2, 1),
            kmajor(w2_ptr, n2, k2, L),
            blocked_sf(w2_sf_ptr, n2, k2, L),
            x_sf_ptr,
            h_ptr,
            h_sf_ptr,
            out_ptr,
            cute.make_tensor(alpha1_ptr, layout=cute.make_layout((L,))),
            cute.make_tensor(alpha2_ptr, layout=cute.make_layout((L,))),
            cute.make_tensor(tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))),
            cute.make_tensor(tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))),
            cute.make_tensor(permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))),
            cute.make_tensor(num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))),
            cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,))),
            cute.make_tensor(
                token_final_scales_ptr,
                layout=cute.make_ordered_layout((orig_m, self.topk), order=(1, 0)),
            ),
            ctr_ptr,
            cute.make_tensor(topk_ids_ptr, layout=cute.make_layout((orig_m * self.topk,))),
            local_offset,
            num_local,
            orig_m,
            num_ctas,
            stream,
            trace_ptr,
        )


__all__: Tuple[str, ...] = ("K3MoeLowMDynKernel", "CTR_TILE_BASE", "counter_words")

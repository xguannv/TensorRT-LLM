# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cutlass
import cutlass.cute as cute


class MLAStaticTileSchedulerParams:

    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisor = None,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        split_kv_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler parameters prepared for MLA static tile scheduler.

        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param problem_shape_b: The shape of the problem
        :type problem_shape_b: cute.Int32
        :param problem_shape_s: The shape of the problem in sequence length Q dimension
        :type problem_shape_s: cute.Int32
        :param cluster_shape_mnk: The shape of the cluster
        :type cluster_shape_mnk: cute.Shape
        :param split_kv: The scalar factor for split KV
        """
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.split_kv_fdd = split_kv_fdd
        if cutlass.const_expr(problem_shape_b_fdd is None):
            self.problem_shape_b_fdd = cute.fast_divmod_create_divisor(
                problem_shape_b, loc=loc, ip=ip)
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor(
                problem_shape_s, loc=loc, ip=ip)
        if cutlass.const_expr(split_kv_fdd is None):
            self.split_kv_fdd = cute.fast_divmod_create_divisor(split_kv,
                                                                loc=loc,
                                                                ip=ip)
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.problem_shape_s)
        values += cutlass.extract_mlir_values(self.split_kv)
        values += cutlass.extract_mlir_values(self.problem_shape_b_fdd)
        values += cutlass.extract_mlir_values(self.problem_shape_s_fdd)
        values += cutlass.extract_mlir_values(self.split_kv_fdd)
        return values

    def __new_from_mlir_values__(self, values):
        # Slice per field by that field's own extraction width, in
        # __extract_mlir_values__ order. FastDivmodDivisor occupies 2 SSA values
        # on nvidia-cutlass-dsl-internal, so fixed indices mis-slice (cutlass #3243).
        fields = (
            self.problem_shape_b,
            self.problem_shape_s,
            self.split_kv,
            self.problem_shape_b_fdd,
            self.problem_shape_s_fdd,
            self.split_kv_fdd,
        )
        rebuilt = []
        offset = 0
        for field in fields:
            width = len(cutlass.extract_mlir_values(field))
            rebuilt.append(
                cutlass.new_from_mlir_values(
                    field, tuple(values[offset:offset + width])))
            offset += width
        (problem_shape_b, problem_shape_s, split_kv, problem_shape_b_fdd,
         problem_shape_s_fdd, split_kv_fdd) = rebuilt
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            split_kv,
            problem_shape_b_fdd=problem_shape_b_fdd,
            problem_shape_s_fdd=problem_shape_s_fdd,
            split_kv_fdd=split_kv_fdd,
            loc=self.loc,
        )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    problem_shape_s: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    return MLAStaticTileSchedulerParams(is_persistent, problem_shape_b,
                                        problem_shape_s, cluster_shape_mnk,
                                        split_kv)


class WorkTileInfo:

    def __init__(self, blk_coord: cute.Coord, is_valid: bool):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        return values

    def __new_from_mlir_values__(self, values):
        new_tile_idx = cutlass.new_from_mlir_values(self.blk_coord, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(
            self.is_valid, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


class MLAStaticTileScheduler:

    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        is_valid: bool = True,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler for MLA split kv kernel.
        Based on `is_persistent`, it provides 2 modes for use:
        - Persistent mode: Launch fixed blocks and reschedule the data blocks.
        - Non-persistent mode: Launch dynamic blocks and exit when the current work is done.

        :param params: The static tile scheduler parameters
        :type params: MLAStaticTileSchedulerParams
        :param current_work_linear_idx: The linear index of the current work
        :type current_work_linear_idx: cutlass.Int32
        :param blk_coord: The coordinate of the current work
        :type blk_coord: cute.Coord
        :param grid_shape: The shape of the grid
        :type grid_shape: cute.Shape
        :param is_valid: Whether the current work is valid
        :type is_valid: bool
        """
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        if params.is_persistent:
            self.persistent_blk_layout = cute.make_layout(
                (
                    params.cluster_shape_mnk[0],
                    params.problem_shape_s,
                    params.problem_shape_b,
                    params.split_kv,
                ),
                loc=loc,
                ip=ip,
            )
            self.num_blocks = cute.size(self.persistent_blk_layout,
                                        loc=loc,
                                        ip=ip)
            # Used for persistent scheduling
            self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        else:
            self.is_valid = is_valid
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: MLAStaticTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        # called by host
        grid_shape = (
            params.cluster_shape_mnk[0],
            params.problem_shape_b * params.problem_shape_s,
            params.split_kv,
        )
        if params.is_persistent:
            return (
                cutlass.min(
                    max_active_clusters * cute.size(params.cluster_shape_mnk),
                    cute.size(grid_shape, loc=loc, ip=ip),
                ),
                1,
                1,
            )
        else:
            return grid_shape

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = (self.current_work_linear_idx < self.num_blocks
                    if self.params.is_persistent else self.is_valid)

        if self.params.is_persistent:
            current_work_cluster_batch, cluster_idx = (
                self.current_work_linear_idx //
                self.params.cluster_shape_mnk[0],
                self.current_work_linear_idx % self.params.cluster_shape_mnk[0],
            )
            current_work_s_batch, s_idx = divmod(
                current_work_cluster_batch, self.params.problem_shape_s_fdd)
            current_work_b_batch, b_idx = divmod(
                current_work_s_batch, self.params.problem_shape_b_fdd)
            _, split_kv_idx = divmod(current_work_b_batch,
                                     self.params.split_kv_fdd)

            blk_coord = (cluster_idx, s_idx, b_idx, split_kv_idx)
        else:
            # Non-persistent grid: blockIdx.y spans ``problem_shape_b *
            # problem_shape_s`` with ``s`` as the fast-varying dim (matching the
            # persistent ``persistent_blk_layout`` ordering (cluster, s, b,
            # split)). Decode with ``problem_shape_s`` as the divisor so the
            # (b, s) mapping is consistent with the persistent path. Using
            # ``problem_shape_b`` here transposes b<->s and corrupts results for
            # ``seq_len_q > 1`` (MTP / spec-decode) in non-persistent mode.
            # Ported from flashinfer PR #3309.
            b_idx, s_idx = divmod(self.blk_coord[1],
                                  self.params.problem_shape_s_fdd)
            blk_coord = (self.blk_coord[0], s_idx, b_idx, self.blk_coord[2])

        return WorkTileInfo(blk_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self.params.is_persistent:
            self.current_work_linear_idx += advance_count * self.num_persistent_sm
        else:
            self.is_valid = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        # Width-driven slicing, as in MLAStaticTileSchedulerParams: params holds
        # three FastDivmodDivisors, which are not 1 SSA value each on every
        # cutlass-dsl build, so the old fixed 6/1/3/rest split mis-slices.
        rebuilt = []
        offset = 0
        for field in (self.params, self.current_work_linear_idx, self.blk_coord,
                      self.grid_shape):
            width = len(cutlass.extract_mlir_values(field))
            rebuilt.append(
                cutlass.new_from_mlir_values(
                    field, list(values[offset:offset + width])))
            offset += width
        assert offset == len(values), (
            f"MLAStaticTileScheduler consumed {offset} of {len(values)} values")
        return MLAStaticTileScheduler(*rebuilt)


def create_mla_static_tile_scheduler(
    params: MLAStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> MLAStaticTileScheduler:
    return MLAStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


class MLABalancedWorkTileInfo:
    """One work segment of ``MLABalancedTileScheduler``.

    ``blk_coord`` is ``(cta_in_cluster, chunk, request, partial_slot)``; the
    segment covers K tiles ``[k_index, k_index + k_count)`` of that
    (request, chunk) M tile. ``is_partial`` (Int32 0/1) is set when the segment
    does not cover the whole M tile, i.e. its result is a split-KV partial.
    """

    def __init__(self, blk_coord, is_valid, k_index, k_count, is_partial):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)
        self.k_index = k_index
        self.k_count = k_count
        self.is_partial = is_partial

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        values += cutlass.extract_mlir_values(self.k_index)
        values += cutlass.extract_mlir_values(self.k_count)
        values += cutlass.extract_mlir_values(self.is_partial)
        return values

    def __new_from_mlir_values__(self, values):
        n = len(values) - 4
        return MLABalancedWorkTileInfo(
            cutlass.new_from_mlir_values(self.blk_coord, values[:n]),
            cutlass.new_from_mlir_values(self.is_valid, [values[n]]),
            cutlass.new_from_mlir_values(self.k_index, [values[n + 1]]),
            cutlass.new_from_mlir_values(self.k_count, [values[n + 2]]),
            cutlass.new_from_mlir_values(self.is_partial, [values[n + 3]]),
        )

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


class MLABalancedTileScheduler:
    """Persistent 2-CTA clusters that split the KV work of a decode batch evenly.

    Work unit: one K tile of one (request, chunk) M tile. Every chunk of a
    request reads the same KV, so the work of one chunk is the sequence of all
    requests' K tiles, ``V = sum_b ceil(K_b / k_tile)`` units long, and the
    batch holds ``U = C * V`` units for ``C`` chunks. Each of the ``G``
    clusters gets ``W = max(ceil(U / G), min_units)`` units:

    * ``n = G // C`` lanes per chunk: lane ``l`` of chunk ``c`` (cluster
      ``l * C + c``) runs positions ``[l * W, (l + 1) * W)`` of chunk ``c``'s
      sequence. Lane ``l`` of every chunk works on the same requests at the
      same time, so the C M tiles of a request share its KV through L2.
    * ``s = G - n * C`` spare clusters split the leftover positions
      ``[n * W, V)`` of all chunks, concatenated chunk by chunk.

    A cluster's positions are cut at M-tile (request) boundaries into
    segments. A segment that covers only part of its M tile is a split-KV
    partial: it goes to a workspace slot (lanes own 2 slots, spares
    ``slots_per_spare``) and a reduction combines the partials of the tile.

    The schedule is written once per CTA to shared memory by
    ``build_mla_balanced_schedule``: ``table[b]`` is the number of units of
    requests ``< b`` per chunk (``table[num_requests] = V``) and ``params``
    holds ``(num_requests, n * W, V - n * W, range begin, range end)`` of
    this cluster. Every warp role reads the parameters and its lane's slice
    of the table into registers once (``create_mla_balanced_tile_scheduler``):
    with at most 32 requests lane ``l`` holds ``table[l]`` and a segment
    lookup is a warp ballot; with more, lane ``l`` holds ``table[32 l]`` and
    a lookup adds one shared-memory load per lane. The position ``q``, the
    partial count and the current segment's (K-tile count, partial flag)
    are carried in registers, so advancing to the next segment needs no
    lookup.
    """

    NUM_PARAMS = 5

    def __init__(
        self,
        table: cute.Tensor,
        params: cute.Tensor,
        num_clusters: int,
        num_chunks: int,
        slots_per_spare: int,
        q: cutlass.Int32,
        n_partial: cutlass.Int32,
        regs: tuple,
        cur_k_count: cutlass.Int32,
        cur_is_partial: cutlass.Int32,
    ):
        self.table = table
        self.params = params
        # 32-request groups of the table (its size is max_requests + 1)
        self.groups = (cute.size(table.layout) - 1 + 31) // 32
        self.num_clusters = num_clusters
        self.num_chunks = num_chunks
        self.num_lanes = (num_clusters // num_chunks) * num_chunks
        self.slots_per_spare = slots_per_spare
        self.q = q
        self.n_partial = n_partial
        # loop-invariant registers: (lane's table entry, num_requests, n * W,
        # V - n * W, range end, V)
        self.regs = regs
        self.cur_k_count = cur_k_count
        self.cur_is_partial = cur_is_partial

    @staticmethod
    def num_slots(num_clusters: int, num_chunks: int,
                  slots_per_spare: int) -> int:
        """Partial slots of all clusters (2 per lane, ``slots_per_spare`` per spare)."""
        lanes = (num_clusters // num_chunks) * num_chunks
        return 2 * lanes + (num_clusters - lanes) * slots_per_spare

    @staticmethod
    def default_slots_per_spare(num_chunks: int) -> int:
        """Upper bound of a spare cluster's partial segments: its range start
        and end plus one leftover start per chunk."""
        return num_chunks + 2

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.q)
        values += cutlass.extract_mlir_values(self.n_partial)
        values += cutlass.extract_mlir_values(self.cur_k_count)
        values += cutlass.extract_mlir_values(self.cur_is_partial)
        return values

    def __new_from_mlir_values__(self, values):
        return MLABalancedTileScheduler(
            self.table,
            self.params,
            self.num_clusters,
            self.num_chunks,
            self.slots_per_spare,
            cutlass.new_from_mlir_values(self.q, [values[0]]),
            cutlass.new_from_mlir_values(self.n_partial, [values[1]]),
            self.regs,
            cutlass.new_from_mlir_values(self.cur_k_count, [values[2]]),
            cutlass.new_from_mlir_values(self.cur_is_partial, [values[3]]),
        )

    @staticmethod
    def _cluster_idx():
        return cute.arch.block_idx()[0] // 2

    @cute.jit
    def _segment(self):
        """(request, chunk, k_index, k_count, is_partial) of the segment at q.

        Every lane gets the same values (warp ballots and shuffles over the
        table entries the lanes hold, results marked warp-uniform).
        """
        top, num_requests, n_w, l_w, range_end, v_total = self.regs
        cluster_idx = self._cluster_idx()
        c = cluster_idx % self.num_chunks
        p = self.q
        p_end = range_end
        if cutlass.const_expr(self.num_lanes < self.num_clusters):
            if cluster_idx >= self.num_lanes:
                # spare: q runs over the chunks' leftovers, Lw positions each
                lw = cutlass.max(l_w, 1)
                c = self.q // lw
                p = n_w + (self.q - c * lw)
                p_end = n_w + cutlass.min(l_w, range_end - c * lw)
        # largest b < num_requests with table[b] <= p (table[0] = 0 <= p);
        # requests without KV have no units
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.groups == 1):
            idx = cute.arch.popc(
                cute.arch.vote_ballot_sync(lane < num_requests
                                           and top <= p)) - 1
            idx = cute.arch.make_warp_uniform(idx)
            lo = idx
            tile_begin = cute.arch.shuffle_sync(top, idx)
            tile_end = cute.arch.shuffle_sync(top, cutlass.min(idx + 1, 31))
        else:
            g = cute.arch.popc(
                cute.arch.vote_ballot_sync(lane * 32 < num_requests
                                           and top <= p)) - 1
            g = cute.arch.make_warp_uniform(g)
            b = g * 32 + lane
            v = self.table[cutlass.min(b, num_requests - 1)]
            idx = cute.arch.popc(
                cute.arch.vote_ballot_sync(b < num_requests and v <= p)) - 1
            idx = cute.arch.make_warp_uniform(idx)
            lo = g * 32 + idx
            tile_begin = cute.arch.shuffle_sync(v, idx)
            tile_end = cute.arch.shuffle_sync(v, cutlass.min(idx + 1, 31))
            if idx == 31:
                tile_end = self.table[cutlass.min(lo + 1, num_requests)]
        if lo + 1 >= num_requests:
            tile_end = v_total
        tile_begin = cute.arch.make_warp_uniform(tile_begin)
        tile_end = cute.arch.make_warp_uniform(tile_end)
        k_count = cutlass.min(tile_end, p_end) - p
        is_partial = cutlass.Int32(0)
        if p > tile_begin or p + k_count < tile_end:
            is_partial = cutlass.Int32(1)
        return lo, c, p - tile_begin, k_count, is_partial

    @cute.jit
    def owned_slots(self):
        """(first slot, slot count) of this cluster's partial slots."""
        cluster_idx = self._cluster_idx()
        base = cluster_idx * 2
        count = cutlass.Int32(2)
        if cutlass.const_expr(self.num_lanes < self.num_clusters):
            if cluster_idx >= self.num_lanes:
                base = (self.num_lanes * 2 +
                        (cluster_idx - self.num_lanes) * self.slots_per_spare)
                count = cutlass.Int32(self.slots_per_spare)
        return base, count

    def _slot(self):
        base, _ = self.owned_slots()
        return base + self.n_partial

    def get_current_work(self, *, loc=None, ip=None) -> MLABalancedWorkTileInfo:
        """Work tile of the segment at q; also remembers its K-tile count and
        partial flag for ``advance_to_next_work``."""
        b, c, k_index, k_count, is_partial = self._segment()
        self.cur_k_count = k_count
        self.cur_is_partial = is_partial
        return MLABalancedWorkTileInfo(
            (cute.arch.block_idx()[0] % 2, c, b, self._slot()),
            self.q < self.regs[4],
            k_index,
            k_count,
            is_partial,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        """Step past the segment of the last ``get_current_work``."""
        self.q += self.cur_k_count
        self.n_partial += self.cur_is_partial


@cute.jit
def build_mla_balanced_schedule(
    cache_seqs: cute.Tensor,
    table: cute.Tensor,
    params: cute.Tensor,
    k_tile: cutlass.Constexpr,
    max_requests: cutlass.Constexpr,
    num_clusters: cutlass.Constexpr,
    num_chunks: cutlass.Constexpr,
    min_units: cutlass.Constexpr,
    aligned_segment_cost: cutlass.Constexpr = -1,
):
    """One warp writes the shared-memory schedule of its CTA's cluster.

    ``table[b]`` is the number of K tiles (``ceil(K / k_tile)``) of requests
    ``< b``. Lane l loads requests l, l + 32, ... up front (indices clamped to
    the batch, so the loads are unconditional), then each group of 32
    requests is prefix-summed with warp shuffles. ``min_units``
    lower-bounds the units per cluster so a small batch with short KV is not
    cut into segments shorter than their fixed cost.

    ``aligned_segment_cost >= 0`` (at most 32 requests): the lanes of a chunk
    may instead be aligned to requests. With the shortest lane length ``T``
    that fits, request ``b`` gets ``n_b = ceil(T_b / T)`` lanes and lane ``j``
    of them runs K tiles ``[j T_b / n_b, (j + 1) T_b / n_b)`` of it, a single
    segment; the remaining lanes and the spare clusters idle. The aligned
    schedule is used when its longest lane is at most ``W +
    aligned_segment_cost`` (the K tiles a segment boundary costs).
    """
    lane = cute.arch.lane_idx()
    num_requests = cache_seqs.shape[0]
    groups = max_requests // 32
    kv = cute.make_rmem_tensor(cute.make_layout(groups), cutlass.Int32)
    for i in cutlass.range_constexpr(groups):
        kv[i] = cache_seqs[cutlass.min(i * 32 + lane, num_requests - 1)]
    carry = cutlass.Int32(0)
    for i in cutlass.range_constexpr(groups):
        if i * 32 < num_requests:
            b = i * 32 + lane
            tiles = cutlass.Int32(0)
            if b < num_requests:
                tiles = (cutlass.max(kv[i], 0) + (k_tile - 1)) // k_tile
            incl = tiles
            for d in cutlass.range_constexpr(5):
                up = cute.arch.shuffle_sync_up(incl, offset=1 << d)
                if lane >= (1 << d):
                    incl = incl + up
            if b < num_requests:
                table[b + 1] = carry + incl
            carry = carry + cute.arch.shuffle_sync(incl, 31)
    G = num_clusters
    C = num_chunks
    n = G // C
    s = G - n * C
    cluster_idx = cute.arch.block_idx()[0] // 2
    V = carry
    W = cutlass.max((V * C + (G - 1)) // G, min_units)
    aligned = False
    qa_al = cutlass.Int32(0)
    qb_al = cutlass.Int32(0)
    if cutlass.const_expr(aligned_segment_cost >= 0 and groups == 1
                          and n <= 32):
        aligned, qa_al, qb_al = _aligned_lane_range(kv[0], num_requests, V, W,
                                                    k_tile, n, C, lane,
                                                    cluster_idx,
                                                    aligned_segment_cost)
    if lane == 0:
        table[0] = 0
        nW = cutlass.min(W * n, V)
        Lw = V - nW
        qa = cutlass.min(cluster_idx // C * W, V)
        qb = cutlass.min((cluster_idx // C + 1) * W, V)
        if cutlass.const_expr(s > 0):
            if cluster_idx >= n * C:
                spare = cluster_idx - n * C
                s_tot = Lw * C
                Ws = (s_tot + (s - 1)) // s
                qa = cutlass.min(spare * Ws, s_tot)
                qb = cutlass.min((spare + 1) * Ws, s_tot)
        if aligned:
            nW = V
            Lw = cutlass.Int32(0)
            qa = qa_al
            qb = qb_al
        params[0] = num_requests
        params[1] = nW
        params[2] = Lw
        params[3] = qa
        params[4] = qb


@cute.jit
def _warp_scan(value: cutlass.Int32, lane: cutlass.Int32) -> cutlass.Int32:
    """Inclusive warp prefix sum."""
    for d in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(value, offset=1 << d)
        if lane >= (1 << d):
            value = value + up
    return value


@cute.jit
def _aligned_lane_range(kv, num_requests, V, W, k_tile: cutlass.Constexpr,
                        n: cutlass.Constexpr, C: cutlass.Constexpr, lane,
                        cluster_idx, segment_cost: cutlass.Constexpr):
    """Request-aligned lanes of ``build_mla_balanced_schedule`` (lane b holds
    request b's KV length ``kv``): (use it, this cluster's range begin, end).

    Not used for fewer than 2 requests with KV or more than ``n``. The lane
    length is the smallest ``T`` with ``sum_b ceil(T_b / T) <= n``
    (``ceil(V / n)`` when that fits, else a binary search); request b gets
    ``ceil(T_b / T)`` lanes and lanes past them idle."""
    tiles = cutlass.Int32(0)
    if lane < num_requests:
        tiles = (cutlass.max(kv, 0) + (k_tile - 1)) // k_tile
    excl = _warp_scan(tiles, lane) - tiles
    # Some request gets at most n // r lanes (r requests with KV): a lane is
    # at least ceil(T_min / (n // r)) long, which rules out most batches
    # before the search.
    r = cute.arch.popc(cute.arch.vote_ballot_sync(tiles > 0))
    t_min = cute.arch.warp_redux_sync(tiles if tiles > 0 else 1 << 30, "min")
    lower = (t_min + n // cutlass.max(r, 1) - 1) // cutlass.max(
        n // cutlass.max(r, 1), 1)
    possible = r >= 2 and r <= n and lower <= W + segment_cost
    lane_len = cutlass.max((V + (n - 1)) // n, 1)
    need = cute.arch.warp_redux_sync((tiles + lane_len - 1) // lane_len, "add")
    if possible and need > n:
        lo = lane_len
        hi = cutlass.max(cute.arch.warp_redux_sync(tiles, "max"), 1)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cute.arch.warp_redux_sync((tiles + mid - 1) // mid, "add") <= n:
                hi = mid
            else:
                lo = mid
        lane_len = hi
    nb = (tiles + lane_len - 1) // lane_len
    longest = cute.arch.warp_redux_sync(
        (tiles + cutlass.max(nb, 1) - 1) // cutlass.max(nb, 1), "max")
    lanes_before = _warp_scan(nb, lane) - nb
    lane_of_chunk = cluster_idx // C
    owner_mask = cute.arch.vote_ballot_sync(
        nb > 0 and lanes_before <= lane_of_chunk
        and lane_of_chunk < lanes_before + nb)
    owner = cutlass.Int32(0)
    if owner_mask != 0:
        owner = cute.arch.popc(owner_mask ^ (owner_mask - 1)) - 1
    t = cute.arch.shuffle_sync(tiles, owner)
    k = cutlass.max(cute.arch.shuffle_sync(nb, owner), 1)
    j = lane_of_chunk - cute.arch.shuffle_sync(lanes_before, owner)
    base = cute.arch.shuffle_sync(excl, owner)
    qa = base + j * t // k
    qb = base + (j + 1) * t // k
    if owner_mask == 0 or cluster_idx >= n * C:
        qa = cutlass.Int32(0)
        qb = cutlass.Int32(0)
    use = possible and longest <= W + segment_cost
    return use, qa, qb


def create_mla_balanced_tile_scheduler(
    table: cute.Tensor,
    params: cute.Tensor,
    num_clusters: int,
    num_chunks: int,
    slots_per_spare: int,
) -> MLABalancedTileScheduler:
    """Scheduler over the shared-memory schedule written by
    ``build_mla_balanced_schedule``; reads the parameters and this lane's
    table entry (``table[l]``, or ``table[32 l]`` past 32 requests) into
    registers."""
    lane = cute.arch.lane_idx()
    num_requests = cute.arch.make_warp_uniform(params[0])
    if cutlass.const_expr(cute.size(table.layout) - 1 <= 32):
        top = table[cutlass.min(lane, num_requests)]
    else:
        top = table[cutlass.min(lane * 32, num_requests)]
    regs = (
        top,
        num_requests,
        cute.arch.make_warp_uniform(params[1]),
        cute.arch.make_warp_uniform(params[2]),
        cute.arch.make_warp_uniform(params[4]),
        cute.arch.make_warp_uniform(table[num_requests]),
    )
    return MLABalancedTileScheduler(table, params, num_clusters, num_chunks,
                                    slots_per_spare,
                                    cute.arch.make_warp_uniform(params[3]),
                                    cutlass.Int32(0), regs, cutlass.Int32(0),
                                    cutlass.Int32(0))


LOG2_E = 1.4426950408889634074
# avoid register indexing on array.
MAX_SPLITS = 256


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

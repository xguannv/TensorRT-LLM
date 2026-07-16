/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cutlass/gemm/kernel/static_tile_scheduler.hpp"
#include <cutlass/pipeline/sm100_pipeline.hpp>

namespace trtllm::dev {

using namespace cutlass;
using cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params;

///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
class PersistentTileSchedulerSm90
  : public cutlass::gemm::kernel::detail::StaticPersistentTileScheduler<
      PersistentTileSchedulerSm90> {

  using BaseScheduler =
    cutlass::gemm::kernel::detail::StaticPersistentTileScheduler<PersistentTileSchedulerSm90>;

public:
  using BaseScheduler::StaticPersistentTileScheduler;
  using Params = PersistentTileSchedulerSm90Params;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  using Arguments = BaseScheduler::Arguments;

  static constexpr bool IsDynamicPersistent = false;

  using Pipeline = cutlass::PipelineEmpty;
  using PipelineStorage = typename Pipeline::SharedStorage;
  using ThrottlePipeline = cutlass::PipelineEmpty;
  using ThrottlePipelineStorage = typename ThrottlePipeline::SharedStorage;

  struct CLCResponse {};

  class SharedStorage {
  public:
    CUTLASS_DEVICE PipelineStorage pipeline() { return PipelineStorage{}; }
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() { return ThrottlePipelineStorage{}; }
    CUTLASS_DEVICE CLCResponse* data() { return nullptr; }
  };

  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  static CUTLASS_DEVICE cute::tuple<int32_t, int32_t> get_work_idx_m_and_n(
    uint64_t blk_per_grid_dim,
    FastDivmodU64Pow2 const& divmod_cluster_shape_major,
    FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
    FastDivmodU64 const& divmod_cluster_blk_major,
    int32_t log_swizzle_size,
    RasterOrder raster_order) {
    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();
    return get_work_idx_m_and_n(blk_per_grid_dim,
                                divmod_cluster_shape_major,
                                divmod_cluster_shape_minor,
                                divmod_cluster_blk_major,
                                log_swizzle_size,
                                raster_order,
                                cta_m_in_cluster,
                                cta_n_in_cluster);
  }

  static CUTLASS_DEVICE cute::tuple<int32_t, int32_t> get_work_idx_m_and_n(
    uint64_t blk_per_grid_dim,
    FastDivmodU64Pow2 const& divmod_cluster_shape_major,
    FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
    FastDivmodU64 const& divmod_cluster_blk_major,
    int32_t log_swizzle_size,
    RasterOrder raster_order,
    uint64_t cta_m_in_cluster,
    uint64_t cta_n_in_cluster) {

    // {$nv-release-never begin}
    //
    // This could be changed to get_hier_coord with FastDivmodU64s
    //
    // The underlying approach is:
    // We have our blocks which we rasterize in a given direction
    //    e.g., (M, N) : (1, M)
    // We then tile this with our CGA
    //    e.g., for a 2x2 CGA the shape is ((M / 2, 2), (N / 2, 2))
    //
    // Finally we interleave blocks
    //    e.g., for a 1x1 CGA we may have (M, (2, N / 2)) : (N, (1, M * 2))
    //          This is M major interleaved
    //
    // To get from our linear index (blk_per_grid_dim) to our coordinates
    // we do the following:
    //
    // Calculate the cluster_id within the grid:
    //    cluster_id = linear index / CGA size major
    //    cluster_major_offset = linear index % CGA size major
    // This gives us the cluster we are operating in and the offset within it
    // We also know based on the way we launch the grid that our cluster minor
    // offset is blockIdx.x or blockIdx.y since we only launch CGA_M or CGA_N
    // CTAs in that dimension of the grid.
    //
    // Next we can calculate the cluster minor index and cluster major index
    // First we calculate our swizzle offset, which is
    //   offset_into_swizzle = cluster_id % (2 ^ log_swizzle_size)
    //   extra = cluster_id / 2^log_swizzle_size
    // Then we get our minor index divided by the swizzle and our major index
    //   cluster_idx_minor_div_swizzle = extra /
    //       (Problem Size in Major Dimension / CGA size in Major Dimension)
    //   cluster_idx_major = extra %
    //       (Problem Size in Major Dimension / CGA size in Major Dimension)
    // Then we calculate the minor index from the swizzle
    //   cluster_idx_minor = cluster_idx_minor_div_swizzle * 2^log_swizzle_size + offset
    // We can then get the index by multiplying the indexes by the cluster shape and adding
    // our offsets
    //
    // {$nv-release-never end}

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    } else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(
      cluster_idx_minor * divmod_cluster_shape_minor.divisor + cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(
      cluster_idx_major * divmod_cluster_shape_major.divisor + cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    } else {
      return {major_work_idx, minor_work_idx};
    }
  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static size_t get_workspace_size(Arguments const&,
                                   ProblemShape,
                                   KernelHardwareInfo const&,
                                   uint32_t,
                                   const uint32_t = 1,
                                   uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status initialize_workspace(Arguments const&,
                                              void*,
                                              cudaStream_t,
                                              ProblemShape,
                                              KernelHardwareInfo const&,
                                              uint32_t,
                                              const uint32_t = 1,
                                              uint32_t = 1,
                                              CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }
};

} // namespace trtllm::dev

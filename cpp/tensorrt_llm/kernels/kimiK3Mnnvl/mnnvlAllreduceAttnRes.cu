/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/kimiK3Mnnvl/mnnvlAllreduceAttnRes.h"
#include <algorithm>
#include <cfloat>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/quantization.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3Mnnvl
{

using mnnvl::AllReduceFusionParams;
using tensorrt_llm::common::isNegZero;
using tensorrt_llm::common::isLamportDirty;
using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::cuda_cast;
using tensorrt_llm::common::getMultiProcessorCount;
using tensorrt_llm::common::getDTypeSize;
using tensorrt_llm::common::loadPackedVolatile;

namespace detail
{
template <typename PackedType, typename T>
union PackedVec
{
    PackedType packed;
    T elements[sizeof(PackedType) / sizeof(T)];

    __device__ PackedVec& operator+=(PackedVec& other)
    {
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            elements[i] += other.elements[i];
        }
        return *this;
    }

    __device__ PackedVec operator+(PackedVec& other)
    {
        PackedVec result;
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            result.elements[i] = elements[i] + other.elements[i];
        }
        return result;
    }
};

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr)
{
    return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr)
{
    return *reinterpret_cast<PackedType const*>(ptr);
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;

template <typename T>
struct MnnvlAllReduceKernelParams
{
    T* outputPtr;
    T* residualOutPtr;
    T const* shardPtr;
    T const* residualInPtr;
    T const* gammaPtr;
    T** inputPtrs;
    T* bufferInputPtr;
    T* mcastPtr;
    void* quantOutPtr;
    void* scaleOutPtr;
    float const* scaleFactorPtr;
    int numTokens;
    int tokenDim;
    int nRanks;
    int rank;
    float epsilon;
    uint32_t* bufferFlags;
    bool waitForResults;
    QuantizationSFLayout layout;
};

template <typename PackedType, typename T>
inline __device__ void sanitizeLamportPayload(PackedVec<PackedType, T>& value)
{
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
    {
        if (isNegZero(value.elements[i]))
        {
            value.elements[i] = cuda_cast<T, float>(0.F);
        }
    }
}

template <int Rank, uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ bool pollOneshotRemoteRank(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    if constexpr (Rank == LocalRank)
    {
        return true;
    }
    else
    {
        auto loaded = loadPackedVolatile<PackedType>(
            &stagePtrLocal[token * tokenDim * WorldSize + Rank * tokenDim + packedIdx * kELTS_PER_THREAD]);
        remoteValues[Rank].packed = loaded.packed;
        return !isLamportDirty(loaded);
    }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD, int... Ranks>
inline __device__ bool pollOneshotRemoteRanks(PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token,
    int tokenDim, int packedIdx, std::integer_sequence<int, Ranks...>)
{
    bool valid = true;
    ((valid &= pollOneshotRemoteRank<Ranks, WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
          remoteValues, stagePtrLocal, token, tokenDim, packedIdx)),
        ...);
    return valid;
}

template <typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulatePacked(float (&accum)[kELTS_PER_THREAD], PackedVec<PackedType, T> const& value)
{
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] += cuda_cast<float, T>(value.elements[i]);
    }
}

template <uint8_t WorldSize, int kRankChunk, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulateLamportRanksChunked(
    float (&accum)[kELTS_PER_THREAD], T const* input, int token, int tokenDim, int packedIdx)
{
    static_assert(kRankChunk > 0);
    static_assert(WorldSize % kRankChunk == 0);

#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] = 0.F;
    }

#pragma unroll 1
    for (int rankBase = 0; rankBase < WorldSize; rankBase += kRankChunk)
    {
        float chunkAccum[kELTS_PER_THREAD];
        while (1)
        {
            bool valid = true;
#pragma unroll
            for (int i = 0; i < kELTS_PER_THREAD; i++)
            {
                chunkAccum[i] = 0.F;
            }
#pragma unroll
            for (int rr = 0; rr < kRankChunk; rr++)
            {
                int const r = rankBase + rr;
                auto loaded = loadPackedVolatile<PackedType>(
                    &input[token * tokenDim * WorldSize + r * tokenDim + packedIdx * kELTS_PER_THREAD]);
                PackedVec<PackedType, T> value;
                value.packed = loaded.packed;
                valid &= !isLamportDirty(loaded);
                accumulatePacked<T, PackedType, kELTS_PER_THREAD>(chunkAccum, value);
            }
            if (valid)
            {
                break;
            }
        }
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            accum[i] += chunkAccum[i];
        }
    }
}

template <int Rank, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulateOneshotRank(float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue)
{
    if constexpr (Rank == LocalRank)
    {
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, localValue);
    }
    else
    {
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, remoteValues[Rank]);
    }
}

template <uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD, int... Ranks>
inline __device__ void accumulateOneshotRanks(float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue,
    std::integer_sequence<int, Ranks...>)
{
    (accumulateOneshotRank<Ranks, LocalRank, T, PackedType, kELTS_PER_THREAD>(accum, remoteValues, localValue), ...);
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void waitOneshotRemoteRanks(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    static_assert(LocalRank < WorldSize);
    while (1)
    {
        bool const valid = pollOneshotRemoteRanks<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
            remoteValues, stagePtrLocal, token, tokenDim, packedIdx, std::make_integer_sequence<int, WorldSize>{});
        if (valid)
        {
            break;
        }
    }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministic(
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue)
{
    static_assert(LocalRank < WorldSize);
    float accum[kELTS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] = 0.F;
    }
    accumulateOneshotRanks<LocalRank, T, PackedType, kELTS_PER_THREAD>(
        accum, remoteValues, localValue, std::make_integer_sequence<int, WorldSize>{});

    PackedVec<PackedType, T> packedAccum;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
    }
    return packedAccum;
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministicFastPath(
    PackedVec<PackedType, T> const& localValue, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    PackedVec<PackedType, T> remoteValues[WorldSize];
    waitOneshotRemoteRanks<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
        remoteValues, stagePtrLocal, token, tokenDim, packedIdx);
    return reduceOneshotDeterministic<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(remoteValues, localValue);
}

// Fully deterministic: every rank uses the exact same reduction order. For WorldSize <= 8, specialize the local
// slot so the fast path reuses `val` from registers without a dynamic `remoteValues[rank]` store. Larger world sizes
// use the compact fallback because the benefit is thin but specializing every rank significantly increases compile
// time.
template <uint8_t WorldSize, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotLamport(
    PackedVec<PackedType, T> const& val, T* stagePtrLocal, int token, int tokenDim, int packedIdx, int rank)
{
    PackedVec<PackedType, T> packedAccum;
    if constexpr (WorldSize <= 8)
    {
        packedAccum = val;
#define RUN_ONESHOT_LOCAL_RANK(LOCAL_RANK)                                                                             \
    case LOCAL_RANK:                                                                                                   \
        if constexpr (WorldSize > LOCAL_RANK)                                                                          \
        {                                                                                                              \
            packedAccum = reduceOneshotDeterministicFastPath<WorldSize, LOCAL_RANK, T, PackedType, kELTS_PER_THREAD>(  \
                val, stagePtrLocal, token, tokenDim, packedIdx);                                                       \
        }                                                                                                              \
        break

        switch (rank)
        {
            RUN_ONESHOT_LOCAL_RANK(0);
            RUN_ONESHOT_LOCAL_RANK(1);
            RUN_ONESHOT_LOCAL_RANK(2);
            RUN_ONESHOT_LOCAL_RANK(3);
            RUN_ONESHOT_LOCAL_RANK(4);
            RUN_ONESHOT_LOCAL_RANK(5);
            RUN_ONESHOT_LOCAL_RANK(6);
            RUN_ONESHOT_LOCAL_RANK(7);
        }
#undef RUN_ONESHOT_LOCAL_RANK
    }
    else
    {
        // Chunk Lamport polling so only a bounded rank set is live at once, avoiding register spills for large
        // world sizes.
        constexpr int kRankChunk = 8;
        float accum[kELTS_PER_THREAD];
        accumulateLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS_PER_THREAD>(
            accum, stagePtrLocal, token, tokenDim, packedIdx);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
        }
    }
    return packedAccum;
}

constexpr int kAttnResThreads = 128;
constexpr int kAttnResEltsPerThread = 8;
constexpr int kAttnResEltsPerCta = kAttnResThreads * kAttnResEltsPerThread;
constexpr int kAttnResMaxClusterSize = 8;
constexpr int kAttnResMaxCandidates = 12;

// The attention-residual epilogue of one token's reduced row (oneshotAllreduceAttnResOp's statement), shared by the
// one-shot and two-shot kernels. A cluster of tokenDim / 1024 CTAs owns the token and each thread eight contiguous
// elements of it. The per-token statistics are summed in a fixed order: within a thread, over lanes, over the four
// warps, then over the cluster's CTAs in rank order through distributed shared memory. Every CTA of the cluster must
// have executed barrier.cluster.arrive before the call; the call waits on it before writing to its peers.
template <int N, bool AddPrefix>
__device__ __forceinline__ void attnResEpilogue(PackedVec<float4, __nv_bfloat16> const& reduced,
    PackedVec<float4, __nv_bfloat16> const& prefix,
    PackedVec<float4, __nv_bfloat16> const (&snapshot)[N > 1 ? N - 1 : 1],
    PackedVec<float4, __nv_bfloat16> const& resWeight, PackedVec<float4, __nv_bfloat16> const& rmsWeight,
    PackedVec<float4, __nv_bfloat16> const& outputRmsWeight, AttnResEpilogueParams const& epilogue, int tokenDim,
    PackedVec<float4, __nv_bfloat16>& updated, PackedVec<float4, __nv_bfloat16>& output)
{
    using T = __nv_bfloat16;
    using PackedType = float4;
    using Packed = PackedVec<PackedType, T>;
    constexpr int kELTS = kAttnResEltsPerThread;
    constexpr int kWarps = kAttnResThreads / kWARP_SIZE;
    constexpr int kStats = 2 * N; // Sum of squares and residual-projection dot product per candidate.
    constexpr float kLog2E = 1.4426950408889634F;
    static_assert(sizeof(PackedType) / sizeof(T) == kELTS);
    static_assert(N >= 1 && N <= kAttnResMaxCandidates && N <= 32, "one lane per candidate");

    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const clusterRank = static_cast<int>(cluster.block_rank());
    int const clusterSize = static_cast<int>(cluster.num_blocks());
    int const lane = threadIdx.x & kLANE_ID_MASK;
    int const warp = threadIdx.x >> kLOG2_WARP_SIZE;

    __shared__ float warpStats[kWarps][kStats];
    __shared__ float clusterStats[kAttnResMaxClusterSize][kStats];
    __shared__ float clusterOutputSq[kAttnResMaxClusterSize];
    __shared__ float candidateWeights[N];

    // ======================= Residual add: bf16(prefix + bf16(sum)) =============================
#pragma unroll
    for (int i = 0; i < kELTS; i++)
    {
        if constexpr (AddPrefix)
        {
            updated.elements[i]
                = __float2bfloat16_rn(__bfloat162float(prefix.elements[i]) + __bfloat162float(reduced.elements[i]));
        }
        else
        {
            updated.elements[i] = reduced.elements[i];
        }
    }

    // ======================= Attention-residual scores =============================
    // Candidates are the snapshots followed by the updated prefix sum.
    float q[kELTS];
#pragma unroll
    for (int i = 0; i < kELTS; i++)
    {
        q[i] = __bfloat162float(resWeight.elements[i]) * __bfloat162float(rmsWeight.elements[i]);
    }
    float stats[kStats];
#pragma unroll
    for (int n = 0; n < N; n++)
    {
        Packed const& candidate = n < N - 1 ? snapshot[n] : updated;
        float sumSq = 0.F;
        float dot = 0.F;
#pragma unroll
        for (int i = 0; i < kELTS; i++)
        {
            float const v = __bfloat162float(candidate.elements[i]);
            sumSq = fmaf(v, v, sumSq);
            dot = fmaf(v, q[i], dot);
        }
        stats[2 * n] = sumSq;
        stats[2 * n + 1] = dot;
    }
#pragma unroll
    for (int offset = kWARP_SIZE / 2; offset > 0; offset >>= 1)
    {
#pragma unroll
        for (int s = 0; s < kStats; s++)
        {
            stats[s] += __shfl_down_sync(0xffffffffU, stats[s], offset);
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int s = 0; s < kStats; s++)
        {
            warpStats[warp][s] = stats[s];
        }
    }
    __syncthreads();

    // Each CTA's partial goes into slot clusterRank of every CTA, so all of them sum the same values in the same
    // order.
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    for (int i = threadIdx.x; i < clusterSize * kStats; i += blockDim.x)
    {
        int const peer = i / kStats;
        int const s = i % kStats;
        float partial = 0.F;
#pragma unroll
        for (int w = 0; w < kWarps; w++)
        {
            partial += warpStats[w][s];
        }
        *cluster.map_shared_rank(&clusterStats[clusterRank][s], peer) = partial;
    }
    cluster.sync();

    // Warp 0 turns the cluster totals into the softmax weights; lane n owns candidate n.
    if (warp == 0)
    {
        float logit = -FLT_MAX;
        if (lane < N)
        {
            float sumSq = 0.F;
            float dot = 0.F;
            for (int r = 0; r < clusterSize; r++)
            {
                sumSq += clusterStats[r][2 * lane];
                dot += clusterStats[r][2 * lane + 1];
            }
            logit = dot * rsqrtf(sumSq / tokenDim + epilogue.rmsEps);
        }
        float maxLogit = logit;
#pragma unroll
        for (int offset = kWARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            maxLogit = fmaxf(maxLogit, __shfl_xor_sync(0xffffffffU, maxLogit, offset));
        }
        float const weight = lane < N ? exp2f((logit - maxLogit) * kLog2E) : 0.F;
        float denominator = weight;
#pragma unroll
        for (int offset = kWARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            denominator += __shfl_xor_sync(0xffffffffU, denominator, offset);
        }
        if (lane < N)
        {
            candidateWeights[lane] = weight * (1.F / denominator);
        }
    }
    __syncthreads();

    // ======================= Selection and its RMSNorm =============================
    float weights[N];
#pragma unroll
    for (int n = 0; n < N; n++)
    {
        weights[n] = candidateWeights[n];
    }
    Packed mixed;
    float outputSq = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS; i++)
    {
        float value = 0.F;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            Packed const& candidate = n < N - 1 ? snapshot[n] : updated;
            value = fmaf(weights[n], __bfloat162float(candidate.elements[i]), value);
        }
        mixed.elements[i] = __float2bfloat16_rn(value);
        float const rounded = __bfloat162float(mixed.elements[i]);
        outputSq = fmaf(rounded, rounded, outputSq);
    }
#pragma unroll
    for (int offset = kWARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        outputSq += __shfl_down_sync(0xffffffffU, outputSq, offset);
    }
    // Every thread has read warpStats before the cluster barrier above, so it can be reused.
    if (lane == 0)
    {
        warpStats[warp][0] = outputSq;
    }
    __syncthreads();
    if (threadIdx.x < clusterSize)
    {
        float partial = 0.F;
#pragma unroll
        for (int w = 0; w < kWarps; w++)
        {
            partial += warpStats[w][0];
        }
        *cluster.map_shared_rank(&clusterOutputSq[clusterRank], threadIdx.x) = partial;
    }
    cluster.sync();
    float totalSq = 0.F;
    for (int r = 0; r < clusterSize; r++)
    {
        totalSq += clusterOutputSq[r];
    }
    float const outputRsigma = rsqrtf(totalSq / tokenDim + epilogue.outputRmsEps);
#pragma unroll
    for (int i = 0; i < kELTS; i++)
    {
        // KimiK3RMSNorm: normalize in fp32, round to bf16, then apply the bf16 weight.
        T const normalized = __float2bfloat16_rn(__bfloat162float(mixed.elements[i]) * outputRsigma);
        output.elements[i]
            = __float2bfloat16_rn(__bfloat162float(normalized) * __bfloat162float(outputRmsWeight.elements[i]));
    }
}
} // namespace detail

using detail::PackedVec;
using detail::loadPacked;
using detail::MnnvlAllReduceKernelParams;
using detail::sanitizeLamportPayload;
using detail::reduceOneshotLamport;

// See oneshotAllreduceAttnResOp. A cluster of tokenDim / 1024 CTAs owns one token and each thread eight contiguous
// elements of it; the epilogue is detail::attnResEpilogue.
template <uint8_t WorldSize, int N, bool AddPrefix>
__global__ void __launch_bounds__(detail::kAttnResThreads)
    oneshotAllreduceAttnResKernel(MnnvlAllReduceKernelParams<__nv_bfloat16> params, AttnResEpilogueParams epilogue)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    using T = __nv_bfloat16;
    using PackedType = float4;
    using Packed = PackedVec<PackedType, T>;
    constexpr int kELTS = detail::kAttnResEltsPerThread;

    namespace cg = cooperative_groups;
    int const packedIdx = static_cast<int>(cg::this_cluster().thread_rank());
    int const token = blockIdx.x;
    int const threadOffset = token * params.tokenDim + packedIdx * kELTS;

    // Peers write into this CTA's shared memory only after the matching wait, which also guarantees
    // that every CTA of the cluster has started.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");

    cudaGridDependencySynchronize();
    // Every consumer of the outputs waits for this whole grid, so triggering here only lets the next
    // kernel launch and stream its weights while the ranks exchange their contributions.
    cudaTriggerProgrammaticLaunchCompletion();

    LamportFlags<PackedType> flag(params.bufferFlags, 1);
    T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
    T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));

    // ==================== Broadcast tokens to each rank =============================
    Packed val;
    val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
    sanitizeLamportPayload(val);
    reinterpret_cast<PackedType*>(
        &stagePtrMcast[token * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
        = val.packed;
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

    // The epilogue operands do not depend on the peers; load them while the peers' data is in flight.
    auto const* blockResidual = static_cast<T const*>(epilogue.blockResidual);
    Packed snapshot[N > 1 ? N - 1 : 1];
#pragma unroll
    for (int n = 0; n < N - 1; n++)
    {
        snapshot[n].packed = loadPacked<PackedType>(
            &blockResidual[static_cast<size_t>(n) * params.numTokens * params.tokenDim + threadOffset]);
    }
    [[maybe_unused]] Packed prefix;
    if constexpr (AddPrefix)
    {
        prefix.packed = loadPacked<PackedType>(&params.residualInPtr[threadOffset]);
    }
    Packed resWeight;
    Packed rmsWeight;
    Packed outputRmsWeight;
    resWeight.packed = loadPacked<PackedType>(&static_cast<T const*>(epilogue.resWeight)[packedIdx * kELTS]);
    rmsWeight.packed = loadPacked<PackedType>(&static_cast<T const*>(epilogue.rmsWeight)[packedIdx * kELTS]);
    outputRmsWeight.packed
        = loadPacked<PackedType>(&static_cast<T const*>(epilogue.outputRmsWeight)[packedIdx * kELTS]);

    // ======================= Reduction =============================
    Packed const reduced = reduceOneshotLamport<WorldSize, T, PackedType, kELTS>(
        val, stagePtrLocal, token, params.tokenDim, packedIdx, params.rank);

    Packed updated;
    Packed output;
    detail::attnResEpilogue<N, AddPrefix>(
        reduced, prefix, snapshot, resWeight, rmsWeight, outputRmsWeight, epilogue, params.tokenDim, updated, output);
    reinterpret_cast<PackedType*>(&params.residualOutPtr[threadOffset])[0] = updated.packed;
    reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = output.packed;
    flag.waitAndUpdate({static_cast<uint32_t>(params.numTokens * params.tokenDim * WorldSize * sizeof(T)), 0, 0, 0});
#endif
}

// See twoshotAllreduceAttnResOp. A cluster of tokenDim / 1024 CTAs per token, each thread eight contiguous elements
// of it, as in the one-shot kernel. Scatter: every cluster pushes its token's row to the token's owner rank
// (token % WorldSize). The owner sums the WorldSize rows in the one-shot kernel's order, runs detail::attnResEpilogue
// on the full row and multicasts the two result rows (normed, updated) to every rank. Every cluster then waits for
// its token's result rows and writes them out.
template <uint8_t WorldSize, int N, bool AddPrefix>
__global__ void __launch_bounds__(detail::kAttnResThreads)
    twoshotAllreduceAttnResKernel(MnnvlAllReduceKernelParams<__nv_bfloat16> params, AttnResEpilogueParams epilogue)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    using T = __nv_bfloat16;
    using PackedType = float4;
    using Packed = PackedVec<PackedType, T>;
    constexpr int kELTS = detail::kAttnResEltsPerThread;
    constexpr int kScatter = 0;
    constexpr int kBroadcast = 1;
    constexpr int kStages = 2;
    // The one-shot kernel's reduction order, so both kernels return the same bits.
    constexpr int kRankChunk = WorldSize <= 8 ? WorldSize : 8;

    namespace cg = cooperative_groups;
    int const packedIdx = static_cast<int>(cg::this_cluster().thread_rank());
    int const token = blockIdx.x;
    int const tokenDim = params.tokenDim;
    int const threadOffset = token * tokenDim + packedIdx * kELTS;
    int const owner = token % WorldSize;
    int const localToken = token / WorldSize;
    bool const owned = owner == params.rank;

    // Peers write into this CTA's shared memory only after the matching wait, which also guarantees
    // that every CTA of the cluster has started.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");

    cudaGridDependencySynchronize();
    // Every consumer of the outputs waits for this whole grid, so triggering here only lets the next
    // kernel launch and stream its weights while the ranks exchange their contributions.
    cudaTriggerProgrammaticLaunchCompletion();

    LamportFlags<PackedType> flag(params.bufferFlags, kStages);
    T* scatterLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], kScatter));
    T* scatterOwner = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[owner], kScatter));
    T* broadcastMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, kBroadcast));
    T* broadcastLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], kBroadcast));

    // ==================== Scatter: this rank's row to the token's owner =============================
    Packed val;
    val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
    sanitizeLamportPayload(val);
    reinterpret_cast<PackedType*>(&scatterOwner[localToken * tokenDim * WorldSize + params.rank * tokenDim])[packedIdx]
        = val.packed;
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], kScatter);

    // ==================== Owner: reduce, epilogue, broadcast =============================
    if (owned)
    {
        // The epilogue operands do not depend on the peers; load them while the peers' rows are in flight.
        auto const* blockResidual = static_cast<T const*>(epilogue.blockResidual);
        Packed snapshot[N > 1 ? N - 1 : 1];
#pragma unroll
        for (int n = 0; n < N - 1; n++)
        {
            snapshot[n].packed = loadPacked<PackedType>(
                &blockResidual[static_cast<size_t>(n) * params.numTokens * tokenDim + threadOffset]);
        }
        [[maybe_unused]] Packed prefix;
        if constexpr (AddPrefix)
        {
            prefix.packed = loadPacked<PackedType>(&params.residualInPtr[threadOffset]);
        }
        Packed resWeight;
        Packed rmsWeight;
        Packed outputRmsWeight;
        resWeight.packed = loadPacked<PackedType>(&static_cast<T const*>(epilogue.resWeight)[packedIdx * kELTS]);
        rmsWeight.packed = loadPacked<PackedType>(&static_cast<T const*>(epilogue.rmsWeight)[packedIdx * kELTS]);
        outputRmsWeight.packed
            = loadPacked<PackedType>(&static_cast<T const*>(epilogue.outputRmsWeight)[packedIdx * kELTS]);

        float accum[kELTS];
        detail::accumulateLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS>(
            accum, scatterLocal, localToken, tokenDim, packedIdx);
        Packed reduced;
#pragma unroll
        for (int i = 0; i < kELTS; i++)
        {
            reduced.elements[i] = cuda_cast<T, float>(accum[i]);
        }

        Packed updated;
        Packed output;
        detail::attnResEpilogue<N, AddPrefix>(
            reduced, prefix, snapshot, resWeight, rmsWeight, outputRmsWeight, epilogue, tokenDim, updated, output);
        sanitizeLamportPayload(output);
        sanitizeLamportPayload(updated);
        T* results = &broadcastMcast[static_cast<size_t>(token) * 2 * tokenDim];
        reinterpret_cast<PackedType*>(results)[packedIdx] = output.packed;
        reinterpret_cast<PackedType*>(results + tokenDim)[packedIdx] = updated.packed;
    }
    else
    {
        asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    }
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], kBroadcast);
    flag.ctaArrive();

    // ==================== Every rank: the token's result rows =============================
    T const* results = &broadcastLocal[static_cast<size_t>(token) * 2 * tokenDim + packedIdx * kELTS];
    auto output = loadPackedVolatile<PackedType>(results);
    while (isLamportDirty(output))
    {
        output = loadPackedVolatile<PackedType>(results);
    }
    auto updated = loadPackedVolatile<PackedType>(results + tokenDim);
    while (isLamportDirty(updated))
    {
        updated = loadPackedVolatile<PackedType>(results + tokenDim);
    }
    reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = output.packed;
    reinterpret_cast<PackedType*>(&params.residualOutPtr[threadOffset])[0] = updated.packed;
    flag.waitAndUpdate(
        {static_cast<uint32_t>((params.numTokens + WorldSize - 1) / WorldSize * WorldSize * tokenDim * sizeof(T)),
            static_cast<uint32_t>(2 * params.numTokens * tokenDim * sizeof(T)), 0, 0});
#endif
}

namespace
{

template <uint8_t WorldSize, int N>
void launchAllreduceAttnRes(cudaLaunchConfig_t const& config,
    MnnvlAllReduceKernelParams<__nv_bfloat16> const& kernelParams, AttnResEpilogueParams const& epilogue,
    bool addPrefix, bool twoShot)
{
    auto const kernel = twoShot ? (addPrefix ? &twoshotAllreduceAttnResKernel<WorldSize, N, true>
                                             : &twoshotAllreduceAttnResKernel<WorldSize, N, false>)
                                : (addPrefix ? &oneshotAllreduceAttnResKernel<WorldSize, N, true>
                                             : &oneshotAllreduceAttnResKernel<WorldSize, N, false>);
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, kernelParams, epilogue));
}

template <uint8_t WorldSize, int... CandidateIdx>
void dispatchAllreduceAttnRes(cudaLaunchConfig_t const& config,
    MnnvlAllReduceKernelParams<__nv_bfloat16> const& kernelParams, AttnResEpilogueParams const& epilogue,
    bool addPrefix, bool twoShot, std::integer_sequence<int, CandidateIdx...>)
{
    bool const launched
        = ((epilogue.numCandidates == CandidateIdx + 1 ? (
                launchAllreduceAttnRes<WorldSize, CandidateIdx + 1>(config, kernelParams, epilogue, addPrefix, twoShot),
                true)
                                                       : false)
            || ...);
    TLLM_CHECK_WITH_INFO(launched, "[MNNVL AllReduceAttnRes] unsupported number of candidates %d (1-%d).",
        epilogue.numCandidates, detail::kAttnResMaxCandidates);
}

void allreduceAttnResOp(AllReduceFusionParams const& params, AttnResEpilogueParams const& epilogue, bool twoShot)
{
    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSMVersion >= 90, "[MNNVL AllReduceAttnRes] requires SM 90 or newer.");
    TLLM_CHECK_WITH_INFO(
        params.dType == tensorrt_llm::DataType::kBF16, "[MNNVL AllReduceAttnRes] supports BF16 tensors only.");
    int const clusterSize = params.tokenDim / detail::kAttnResEltsPerCta;
    TLLM_CHECK_WITH_INFO(params.tokenDim % detail::kAttnResEltsPerCta == 0 && clusterSize >= 1
            && clusterSize <= detail::kAttnResMaxClusterSize,
        "[MNNVL AllReduceAttnRes] hidden dimension %d must be a multiple of %d and at most %d.", params.tokenDim,
        detail::kAttnResEltsPerCta, detail::kAttnResEltsPerCta * detail::kAttnResMaxClusterSize);

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 1;
    attrs[1].val.clusterDim.y = clusterSize;
    attrs[1].val.clusterDim.z = 1;
    cudaLaunchConfig_t config{
        .gridDim = dim3(params.numTokens, clusterSize, 1),
        .blockDim = detail::kAttnResThreads,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = attrs,
        .numAttrs = 2U,
    };

    using T = __nv_bfloat16;
    MnnvlAllReduceKernelParams<T> kernelParams{reinterpret_cast<T*>(params.output),
        reinterpret_cast<T*>(params.residualOut), reinterpret_cast<T const*>(params.input),
        reinterpret_cast<T const*>(params.residualIn), nullptr, reinterpret_cast<T**>(params.bufferPtrsDev),
        reinterpret_cast<T*>(params.bufferPtrLocal), reinterpret_cast<T*>(params.multicastPtr), nullptr, nullptr,
        nullptr, params.numTokens, params.tokenDim, params.nRanks, params.rank, 0.F, params.bufferFlags, false,
        params.layout};
    bool const addPrefix = params.residualIn != nullptr;
    auto constexpr kCandidates = std::make_integer_sequence<int, detail::kAttnResMaxCandidates>{};

    switch (params.nRanks)
    {
    case 2: dispatchAllreduceAttnRes<2>(config, kernelParams, epilogue, addPrefix, twoShot, kCandidates); break;
    case 4: dispatchAllreduceAttnRes<4>(config, kernelParams, epilogue, addPrefix, twoShot, kCandidates); break;
    case 8: dispatchAllreduceAttnRes<8>(config, kernelParams, epilogue, addPrefix, twoShot, kCandidates); break;
    case 16: dispatchAllreduceAttnRes<16>(config, kernelParams, epilogue, addPrefix, twoShot, kCandidates); break;
    default:
        TLLM_CHECK_WITH_INFO(
            false, "[MNNVL AllReduceAttnRes] unsupported world size %d (2, 4, 8 or 16).", params.nRanks);
    }
}

} // namespace

void oneshotAllreduceAttnResOp(AllReduceFusionParams const& params, AttnResEpilogueParams const& epilogue)
{
    allreduceAttnResOp(params, epilogue, /*twoShot=*/false);
}

void twoshotAllreduceAttnResOp(AllReduceFusionParams const& params, AttnResEpilogueParams const& epilogue)
{
    allreduceAttnResOp(params, epilogue, /*twoShot=*/true);
}

namespace detail
{
constexpr int kRmsSliceEltsPerPacket = 8;
constexpr int kRmsSliceMaxThreads = 1024;
} // namespace detail

// See allreduceRmsNormSliceOp. One CTA per token. Scatter: thread p sends packet p (eight columns) of this rank's row
// to the rank owning those columns. Warp 0 then reduces this rank's slice, one packet per lane, multicasts the slice's
// sum of squares and, once every rank's sum has arrived, scales and writes the slice; the other warps write the
// padding and the tail meanwhile.
template <uint8_t WorldSize>
__global__ void __launch_bounds__(detail::kRmsSliceMaxThreads)
    allreduceRmsNormSliceKernel(AllreduceRmsNormSliceParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    using T = __nv_bfloat16;
    using PackedType = float4;
    using Packed = PackedVec<PackedType, T>;
    constexpr int kELTS = detail::kRmsSliceEltsPerPacket;
    constexpr int kScatter = 0;
    constexpr int kSums = 1;
    constexpr int kStages = 2;
    // twoshotAllreduceKernel's reduction order, so the reduced slice has that op's bits.
    constexpr int kRankChunk = WorldSize < 16 ? WorldSize : 16;
    static_assert(WorldSize % 4 == 0, "the sums are polled four ranks per 16-byte load");

    int const token = blockIdx.x;
    int const hidden = params.hidden;
    int const sliceCols = hidden / WorldSize;
    int const slicePackets = sliceCols / kELTS;
    int const rowPackets = hidden / kELTS;
    int const outCols = sliceCols + params.padCols + params.tailCols;
    void* const* ucPtrs = params.bufferPtrsDev;
    T const* input = static_cast<T const*>(params.input);
    T* outRow = static_cast<T*>(params.output) + static_cast<size_t>(token) * outCols;

    cudaGridDependencySynchronize();
    // Every consumer of the output waits for this whole grid, so triggering here only lets the next kernel launch and
    // stream its weights while the ranks exchange.
    cudaTriggerProgrammaticLaunchCompletion();

    LamportFlags<PackedType> flag(params.bufferFlags, kStages);
    T const* scatterLocal = reinterpret_cast<T const*>(flag.getCurLamportBuf(ucPtrs[params.rank], kScatter));
    float* sumsMcast = reinterpret_cast<float*>(flag.getCurLamportBuf(params.multicastPtr, kSums));
    float const* sumsLocal = reinterpret_cast<float const*>(flag.getCurLamportBuf(ucPtrs[params.rank], kSums));

    // ==================== Scatter: columns [d * w, (d + 1) * w) to rank d, at [token][rank]
    // =============================
    for (int p = static_cast<int>(threadIdx.x); p < rowPackets; p += static_cast<int>(blockDim.x))
    {
        int const dest = p / slicePackets;
        Packed val;
        val.packed = loadPacked<PackedType>(&input[static_cast<size_t>(token) * hidden + p * kELTS]);
        sanitizeLamportPayload(val);
        T* scatterDest = reinterpret_cast<T*>(flag.getCurLamportBuf(ucPtrs[dest], kScatter));
        reinterpret_cast<PackedType*>(
            &scatterDest[(static_cast<size_t>(token) * WorldSize + params.rank) * sliceCols])[p - dest * slicePackets]
            = val.packed;
    }
    flag.clearDirtyLamportBuf(ucPtrs[params.rank], kScatter);

    // ==================== Warp 0: reduce this rank's slice, multicast its sum of squares
    // ================================
    bool const sliceLane = threadIdx.x < detail::kWARP_SIZE;
    float values[kELTS] = {};
    if (sliceLane)
    {
        float sumSq = 0.F;
        if (static_cast<int>(threadIdx.x) < slicePackets)
        {
            float accum[kELTS];
            detail::accumulateLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS>(
                accum, scatterLocal, token, sliceCols, static_cast<int>(threadIdx.x));
#pragma unroll
            for (int i = 0; i < kELTS; i++)
            {
                values[i] = cuda_cast<float, T>(cuda_cast<T, float>(accum[i]));
                sumSq += values[i] * values[i];
            }
        }
        // A butterfly: every lane ends with the same bits. A sum of squares is never -0.0, the Lamport sentinel.
#pragma unroll
        for (int offset = detail::kWARP_SIZE / 2; offset > 0; offset /= 2)
        {
            sumSq += __shfl_xor_sync(0xffffffffU, sumSq, offset);
        }
        if (threadIdx.x == 0)
        {
            sumsMcast[static_cast<size_t>(token) * WorldSize + params.rank] = sumSq;
        }
    }
    else
    {
        // ==================== The other warps: padding and tail =====================================================
        T const* tail = static_cast<T const*>(params.tail);
        int const padPackets = params.padCols / kELTS;
        int const extraPackets = padPackets + params.tailCols / kELTS;
        int const warpThreads = static_cast<int>(detail::kWARP_SIZE);
        for (int p = static_cast<int>(threadIdx.x) - warpThreads; p < extraPackets;
             p += static_cast<int>(blockDim.x) - warpThreads)
        {
            PackedType v = make_float4(0.F, 0.F, 0.F, 0.F);
            if (p >= padPackets)
            {
                v = loadPacked<PackedType>(
                    &tail[static_cast<size_t>(token) * params.tailCols + (p - padPackets) * kELTS]);
            }
            reinterpret_cast<PackedType*>(&outRow[sliceCols])[p] = v;
        }
    }
    flag.clearDirtyLamportBuf(ucPtrs[params.rank], kSums);
    flag.ctaArrive();

    // ==================== Warp 0: every rank's sum, in rank order; the scaled slice
    // ===================================
    if (sliceLane)
    {
        float const* sums = &sumsLocal[static_cast<size_t>(token) * WorldSize];
        float total = 0.F;
        bool ready = false;
        while (!ready)
        {
            ready = true;
            total = 0.F;
#pragma unroll
            for (int g = 0; g < WorldSize / 4; g++)
            {
                auto const loaded = loadPackedVolatile<PackedType>(&sums[g * 4]);
                ready &= !isLamportDirty(loaded);
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    total += __uint_as_float(loaded.words[j]);
                }
            }
        }
        float const rcpRms = rsqrtf(total / static_cast<float>(hidden) + params.eps);
        if (static_cast<int>(threadIdx.x) < slicePackets)
        {
            Packed out;
#pragma unroll
            for (int i = 0; i < kELTS; i++)
            {
                out.elements[i] = cuda_cast<T, float>(values[i] * rcpRms);
            }
            reinterpret_cast<PackedType*>(outRow)[threadIdx.x] = out.packed;
        }
    }
    flag.waitAndUpdate({static_cast<uint32_t>(params.numTokens * hidden * sizeof(T)),
        static_cast<uint32_t>(params.numTokens * WorldSize * sizeof(float)), 0, 0});
#endif
}

void allreduceRmsNormSliceOp(AllreduceRmsNormSliceParams const& params)
{
    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSMVersion >= 90, "[MNNVL AllReduceRmsNormSlice] requires SM 90 or newer.");
    constexpr int kELTS = detail::kRmsSliceEltsPerPacket;
    int const nRanks = params.nRanks;
    TLLM_CHECK_WITH_INFO(nRanks == 4 || nRanks == 8 || nRanks == 16,
        "[MNNVL AllReduceRmsNormSlice] unsupported world size %d (4, 8 or 16).", nRanks);
    int const sliceCols = params.hidden / nRanks;
    TLLM_CHECK_WITH_INFO(params.hidden % nRanks == 0 && sliceCols % kELTS == 0
            && sliceCols / kELTS <= static_cast<int>(detail::kWARP_SIZE),
        "[MNNVL AllReduceRmsNormSlice] hidden %d over %d ranks must give a multiple of %d columns, at most %d.",
        params.hidden, nRanks, kELTS, kELTS * static_cast<int>(detail::kWARP_SIZE));
    TLLM_CHECK_WITH_INFO(
        params.padCols % kELTS == 0 && params.tailCols % kELTS == 0 && params.padCols >= 0 && params.tailCols >= 0,
        "[MNNVL AllReduceRmsNormSlice] padCols %d and tailCols %d must be non-negative multiples of %d.",
        params.padCols, params.tailCols, kELTS);
    TLLM_CHECK_WITH_INFO(params.numTokens > 0, "[MNNVL AllReduceRmsNormSlice] needs at least one token.");

    int const rowPackets = params.hidden / kELTS;
    int const warp = static_cast<int>(detail::kWARP_SIZE);
    // Warp 0 reduces the slice; at least one more warp writes the padding and the tail.
    int const threads
        = std::min(std::max((rowPackets + warp - 1) / warp * warp, 2 * warp), detail::kRmsSliceMaxThreads);
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cudaLaunchConfig_t config{
        .gridDim = dim3(params.numTokens, 1, 1),
        .blockDim = dim3(threads, 1, 1),
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = attrs,
        .numAttrs = 1U,
    };
    switch (nRanks)
    {
    case 4: TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &allreduceRmsNormSliceKernel<4>, params)); break;
    case 8: TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &allreduceRmsNormSliceKernel<8>, params)); break;
    case 16: TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &allreduceRmsNormSliceKernel<16>, params)); break;
    default: break;
    }
}
} // namespace kernels::kimiK3Mnnvl

TRTLLM_NAMESPACE_END

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
#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3Mnnvl
{

/**
 * \brief Kimi K3 attention-residual epilogue of oneshotAllreduceAttnResOp (BF16 tensors).
 */
struct AttnResEpilogueParams
{
    void const* blockResidual;   //!< [numCandidates - 1, numTokens, tokenDim] snapshots; unused when numCandidates == 1
    void const* resWeight;       //!< [tokenDim] attention-residual projection
    void const* rmsWeight;       //!< [tokenDim] weight of the RMSNorm inside the attention-residual score
    void const* outputRmsWeight; //!< [tokenDim] weight of the RMSNorm applied to the selected residual
    float rmsEps;
    float outputRmsEps;
    int numCandidates; //!< Snapshots plus the running prefix sum, 1..12
    //! Optional MXFP8 copy of the output (both or neither): E4M3 codes [numTokens, tokenDim] and one UE8M0 exponent
    //! per 1x32 group [numTokens, tokenDim / 32], the activation quantization of the CuTe DSL low-M FP8 GEMM
    //! (k3_fp8w_lowm_gemm.py) bit for bit, so that k3_fp8w_lowm_mxfp8 consumes it in place of the BF16 output.
    void* mxQuant = nullptr;
    void* mxScale = nullptr;
};

/**
 * \brief One-shot all-reduce with Kimi K3's residual add, attention-residual selection and RMSNorm as the epilogue.
 *
 * Per token, with r = bf16(allreduce(input)):
 *   residualOut = bf16(residualIn + r), or r when residualIn is null
 *   output      = RMSNorm(attn_res(blockResidual[0 .. numCandidates-2], residualOut), outputRmsWeight)
 * with the rounding of the unfused sequence all-reduce -> attn_res_add_rmsnorm_fwd. The reduction order is fixed, so
 * the result is deterministic and identical on every rank. With epilogue.mxQuant / mxScale every rank also stores the
 * MXFP8 form of output (AttnResEpilogueParams).
 *
 * Requirements: BF16; tokenDim a multiple of 1024 and at most 8192; the one-shot footprint
 * numTokens * tokenDim * nRanks elements fits in one Lamport buffer; nRanks in {2, 4, 8, 16}.
 */
void oneshotAllreduceAttnResOp(mnnvl::AllReduceFusionParams const& params, AttnResEpilogueParams const& epilogue);

/**
 * \brief oneshotAllreduceAttnResOp's statement, with a two-shot exchange: every rank sends each token's row to the
 * token's owner rank (token % nRanks); the owner sums the rows in the one-shot op's order, runs the epilogue on the
 * full row and multicasts both outputs back. Returns the one-shot op's bits, except that a -0.0 output is returned as
 * +0.0 (the Lamport sentinel). Per rank it receives about 3 * numTokens * tokenDim elements instead of the one-shot's
 * numTokens * tokenDim * nRanks, so it is the cheaper exchange for more tokens.
 *
 * Requirements: those of oneshotAllreduceAttnResOp, except the footprint: each of the two stages, half a Lamport
 * buffer, holds ceil(numTokens / nRanks) * nRanks * tokenDim elements (scatter) and 2 * numTokens * tokenDim
 * elements (results).
 */
void twoshotAllreduceAttnResOp(mnnvl::AllReduceFusionParams const& params, AttnResEpilogueParams const& epilogue);

/**
 * \brief Parameters of allreduceRmsNormSliceOp (BF16 tensors).
 */
struct AllreduceRmsNormSliceParams
{
    void const* input; //!< [numTokens, hidden], this rank's addend
    void const* tail;  //!< [numTokens, tailCols], appended to every output row; unused when tailCols == 0
    void* output;      //!< [numTokens, hidden / nRanks + padCols + tailCols]
    int numTokens;
    int hidden;
    int padCols;
    int tailCols;
    float eps;

    int nRanks;
    int rank;
    void** bufferPtrsDev;
    void* multicastPtr;
    uint32_t* bufferFlags;
    cudaStream_t stream;
};

/**
 * \brief This rank's column slice of RMSNorm(allreduce(input)) (a norm without a weight), laid out as the input of a
 * row-parallel GEMM: with w = hidden / nRanks, s = rank * w and r = bf16(allreduce(input)), per token
 *   output[:, 0:w]           = bf16(r[s:s+w] * rsqrt(mean(r * r) + eps))
 *   output[:, w:w+padCols]   = 0
 *   output[:, w+padCols:]    = tail
 *
 * Two exchanges on the MNNVL workspace (one turn of its Lamport rotation, two stages): a reduce-scatter by columns, in
 * which every rank sums only its own w columns of every token, in the two-shot all-reduce's rank order (so r has that
 * op's bits); then every rank multicasts each token's sum of squares over its w columns, and every rank adds the
 * nRanks partial sums in rank order, so the scale is identical on every rank. The mean of squares is summed in a
 * different order from a whole-row RMSNorm, so the slice may differ from RMSNorm(r) in the last bit.
 *
 * Requirements: SM 90 or newer; nRanks in {4, 8, 16}; w a multiple of 8 and at most 256; padCols and tailCols
 * multiples of 8; each stage, half a Lamport buffer, holds numTokens * hidden bf16 elements (scatter) and
 * numTokens * nRanks floats (sums).
 */
void allreduceRmsNormSliceOp(AllreduceRmsNormSliceParams const& params);

} // namespace kernels::kimiK3Mnnvl

TRTLLM_NAMESPACE_END

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

#include <cute/tensor.hpp>
#include "CutlassUtils.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cuda_fp4.h> // __nvfp4*
#pragma GCC diagnostic pop

typedef __nv_fp8x4_storage_t __nv_fp4x8_storage_t;
typedef unsigned long __nv_fp8x8_storage_t;

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmUnpackFp4Mode8x16Numx2(uint32_t& out0,
                                                  uint32_t& out1,
                                                  void const* srcMem) {
  asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];"
               : "=r"(out0), "=r"(out1)
               : "l"(srcMem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmUnpackFp4ModeTranspose16x16Numx1(uint32_t& out0,
                                                            uint32_t& out1,
                                                            void const* srcMem) {
  asm volatile("ldmatrix.sync.aligned.shared::cta.m16n16.x1.trans.b8x16.b4x16_p64 {%0, %1}, [%2];"
               : "=r"(out0), "=r"(out1)
               : "l"(srcMem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// {$nv-internal-release begin}
inline __device__ void _qmul4E4m3E2m1E4m3B0(__nv_fp8x4_storage_t& res,
                                            __nv_fp8x4_storage_t a,
                                            __nv_fp8x4_storage_t b) {
#ifdef TLLM_BENCHMARK_E2M1_KV_CACHE
  // The intended SASS is QMUL4.E4M3x4.E2M1x4.N6420.E4M3x4.Bx. Since the compiler does not support
  // it yet, we resort to post-compilation SASS editing. We encode .iswzB into .atype. The <iswzB ->
  // Btype> mapping is described below: B0 -> .e3m2x4; B1 -> .e2m3x4; B2 -> .e5m2x4; B3 -> .ue8m0x4.
  // After compilation, we only have to replace QMUL4.E4M3x4.E2M1x4.N6420.<Btype>.B3210 with
  // QMUL4.E4M3x4.E2M1x4.N6420.E4M3x4.<iswzB>, according to the mapping.
  asm volatile("{\n"
               "mul.e4m3x4.e2m1x4.e3m2x4.satfinite %0, %1, %2;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#else
  asm volatile("{\n"
               ".reg .b32 tmp;\n"
               "prmt.b32 tmp, %2, 0, 0x0000;\n"
               "mul.e4m3x4.e2m1x4.e4m3x4.satfinite %0, %1, tmp;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void _qmul4E4m3E2m1E4m3B1(__nv_fp8x4_storage_t& res,
                                            __nv_fp8x4_storage_t a,
                                            __nv_fp8x4_storage_t b) {
#ifdef TLLM_BENCHMARK_E2M1_KV_CACHE
  asm volatile("{\n"
               "mul.e4m3x4.e2m1x4.e2m3x4.satfinite %0, %1, %2;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#else
  asm volatile("{\n"
               ".reg .b32 tmp;\n"
               "prmt.b32 tmp, %2, 0, 0x1111;\n"
               "mul.e4m3x4.e2m1x4.e4m3x4.satfinite %0, %1, tmp;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void _qmul4E4m3E2m1E4m3B2(__nv_fp8x4_storage_t& res,
                                            __nv_fp8x4_storage_t a,
                                            __nv_fp8x4_storage_t b) {
#ifdef TLLM_BENCHMARK_E2M1_KV_CACHE
  asm volatile("{\n"
               "mul.e4m3x4.e2m1x4.e5m2x4.satfinite %0, %1, %2;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#else
  asm volatile("{\n"
               ".reg .b32 tmp;\n"
               "prmt.b32 tmp, %2, 0, 0x2222;\n"
               "mul.e4m3x4.e2m1x4.e4m3x4.satfinite %0, %1, tmp;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void _qmul4E4m3E2m1E4m3B3(__nv_fp8x4_storage_t& res,
                                            __nv_fp8x4_storage_t a,
                                            __nv_fp8x4_storage_t b) {
#ifdef TLLM_BENCHMARK_E2M1_KV_CACHE
  asm volatile("{\n"
               "mul.e4m3x4.e2m1x4.ue8m0x4.satfinite %0, %1, %2;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#else
  asm volatile("{\n"
               ".reg .b32 tmp;\n"
               "prmt.b32 tmp, %2, 0, 0x3333;\n"
               "mul.e4m3x4.e2m1x4.e4m3x4.satfinite %0, %1, tmp;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void qmul4E4m3E2m1E4m3Bx(__nv_fp8x4_storage_t& res,
                                           __nv_fp8x4_storage_t a,
                                           __nv_fp8x4_storage_t b,
                                           int byteIdxB) {
  if (byteIdxB == 0) {
    _qmul4E4m3E2m1E4m3B0(res, a, b);
  } else if (byteIdxB == 1) {
    _qmul4E4m3E2m1E4m3B1(res, a, b);
  } else if (byteIdxB == 2) {
    _qmul4E4m3E2m1E4m3B2(res, a, b);
  } else if (byteIdxB == 3) {
    _qmul4E4m3E2m1E4m3B3(res, a, b);
  } else {
    // Should never reach here.
    assert(false);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void qmul4E4m3E2m1E4m3(__nv_fp8x4_storage_t& res,
                                         __nv_fp8x4_storage_t a,
                                         __nv_fp8x4_storage_t b) {
  asm volatile("{\n"
               "mul.e4m3x4.e2m1x4.e4m3x4.satfinite %0, %1, %2;\n"
               "}\n"
               : "=r"(res)
               : "r"(a), "r"(b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fused replacement for two qmul4E4m3E2m1E4m3 calls, for architectures without qmul4.
//   src0, src1: 4 unpacked E2M1 values each (in nibbles 0, 2, 4, 6).
//   sf:         4 packed E4M3 SFs, shared by both src0 and src1.
//   dst0, dst1: 4 E4M3 output values each.
inline __device__ void convertUnpackedE2m1ToE4m3x2(uint32_t& dst0,
                                                   uint32_t& dst1,
                                                   uint32_t src0,
                                                   uint32_t src1,
                                                   uint32_t sf) {
  // Interleave: src0 in even nibbles, src1 in odd nibbles.
  uint32_t packed = src1 * 16 + src0;

  // Duplicate each SF byte for the interleaved pair so both elements at position i
  // are scaled by sf_i: [sf0,sf0,sf1,sf1] and [sf2,sf2,sf3,sf3], then convert to fp16x2.
  asm volatile("{\n"
               ".reg .b32 sfDup01, sfDup23;\n"
               ".reg .b16 sf01Lo, sf01Hi, sf23Lo, sf23Hi;\n"
               ".reg .b32 sfFp16_0, sfFp16_1, sfFp16_2, sfFp16_3;\n"
               ".reg .b8 b0, b1, b2, b3;\n"
               ".reg .b32 h0, h1, h2, h3;\n"
               ".reg .b16 e0, e1, e2, e3;\n"
               ".reg .b32 resLo, resHi;\n"
               "prmt.b32 sfDup01, %2, %2, 0x1100;\n"
               "prmt.b32 sfDup23, %2, %2, 0x3322;\n"
               "mov.b32 {sf01Lo, sf01Hi}, sfDup01;\n"
               "mov.b32 {sf23Lo, sf23Hi}, sfDup23;\n"
               "cvt.rn.f16x2.e4m3x2 sfFp16_0, sf01Lo;\n"
               "cvt.rn.f16x2.e4m3x2 sfFp16_1, sf01Hi;\n"
               "cvt.rn.f16x2.e4m3x2 sfFp16_2, sf23Lo;\n"
               "cvt.rn.f16x2.e4m3x2 sfFp16_3, sf23Hi;\n"
               "mov.b32 {b0, b1, b2, b3}, %3;\n"
               "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
               "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
               "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
               "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
               "mul.rn.f16x2 h0, h0, sfFp16_0;\n"
               "mul.rn.f16x2 h1, h1, sfFp16_1;\n"
               "mul.rn.f16x2 h2, h2, sfFp16_2;\n"
               "mul.rn.f16x2 h3, h3, sfFp16_3;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
               "mov.b32 resLo, {e0, e1};\n"
               "mov.b32 resHi, {e2, e3};\n"
               "prmt.b32 %0, resLo, resHi, 0x6420;\n"
               "prmt.b32 %1, resLo, resHi, 0x7531;\n"
               "}\n"
               : "=r"(dst0), "=r"(dst1)
               : "r"(sf), "r"(packed));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fused replacement for two qmul4E4m3E2m1E4m3Bx calls, for architectures without qmul4.
//   src0, src1: 4 unpacked E2M1 values each (in nibbles 0, 2, 4, 6).
//   sf0, sf1:   4 packed E4M3 SFs each; byte at byteIdx is selected for both.
//   dst0, dst1: 4 E4M3 output values each.
inline __device__ void convertUnpackedE2m1ToE4m3x2Bx(uint32_t& dst0,
                                                     uint32_t& dst1,
                                                     uint32_t src0,
                                                     uint32_t src1,
                                                     uint32_t sf0,
                                                     uint32_t sf1,
                                                     int byteIdx) {
  // Interleave: src0 in even nibbles, src1 in odd nibbles.
  uint32_t packed = src1 * 16 + src0;

  // Extract byte byteIdx from sf0 (LSBs) and sf1 (MSBs) via prmt.
  uint32_t prmtSel = 0x0011u * uint32_t(byteIdx) + 0x0040u;
  uint32_t packedSf;
  asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(packedSf) : "r"(sf0), "r"(sf1), "r"(prmtSel));

  // Convert the packed e4m3x2 SF to fp16x2 for scaling.
  uint16_t sfE4m3x2 = uint16_t(packedSf);
  uint32_t sfFp16x2;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfFp16x2) : "h"(sfE4m3x2));

  // Convert 8 interleaved fp4 values through fp16, scale, and convert to fp8.
  uint32_t resLo, resHi;
  asm volatile("{\n"
               ".reg .b8 b0, b1, b2, b3;\n"
               ".reg .b32 h0, h1, h2, h3;\n"
               ".reg .b16 e0, e1, e2, e3;\n"
               "mov.b32 {b0, b1, b2, b3}, %2;\n"
               "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
               "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
               "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
               "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
               "mul.rn.f16x2 h0, h0, %3;\n"
               "mul.rn.f16x2 h1, h1, %3;\n"
               "mul.rn.f16x2 h2, h2, %3;\n"
               "mul.rn.f16x2 h3, h3, %3;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
               "mov.b32 %0, {e0, e1};\n"
               "mov.b32 %1, {e2, e3};\n"
               "}\n"
               : "=r"(resLo), "=r"(resHi)
               : "r"(packed), "r"(sfFp16x2));

  // Deinterleave: even bytes (src0 results) to dst0, odd bytes (src1 results) to dst1.
  asm volatile("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(dst0) : "r"(resLo), "r"(resHi));
  asm volatile("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(dst1) : "r"(resLo), "r"(resHi));
}
// {$nv-internal-release end}
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cute::uint128_t e2m1ToFp16(uint32_t src) {
  uint32_t outFp16[4];
  asm volatile("{\n"
               ".reg .b8 byte0, byte1, byte2, byte3;\n"
               "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
               "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
               "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
               "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
               "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
               "}\n"
               : "=r"(outFp16[0]), "=r"(outFp16[1]), "=r"(outFp16[2]), "=r"(outFp16[3])
               : "r"(src));
  return *reinterpret_cast<cute::uint128_t*>(&outFp16[0]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// {$nv-internal-release begin}
inline __device__ uint32_t qmul4E4m3E2m1E4m3Lower4Elts(uint32_t src, cutlass::float_e4m3_t sf) {
  uint32_t res;
  // {$nv-internal-release begin}
  // NOTE(tizheng): This following PTX should issue QMUL4.E4M3x4.E2M1x4.N3210.E4M3x4.B0.
  // However, due to the PTX optimization and exposure problem, it does not generate
  // single QMUL4 instruction as expected. Compiler team is working on this:
  // - https://nvbugspro.nvidia.com/bug/5221546
  // - https://nvbugspro.nvidia.com/bug/5229203

  // For now, we create two paths: 1) functionality is not correct but delivers the expected
  // performance; 2) functionality is correct but performance is low. We use 1) for benchmark and 2)
  // for validation.
  // {$nv-internal-release end}
#ifdef TLLM_BENCHMARK_E2M1_KV_CACHE
  // {$nv-internal-release begin}
  // Similar to qmul4E4m3E2m1E4m3Bx, we encode .iswzA into .atype.
  // The intended SASS is QMUL4.E4M3x4.E2M1x4.N3210.E4M3x4.B0. We encode .N3210 into atype as
  // .e0m3x4. The .B0 is encoded the same way as qmul4E4m3E2m1E4m3Bx.
  // {$nv-internal-release end}
  asm volatile("{\n"
               "mul.e4m3x4.e0m3x4.e3m2x4.satfinite %0, %1, %2;\n"
               "}\n"
               : "=r"(res)
               : "r"(src), "r"(static_cast<uint32_t>(sf.storage)));
#else
  // Code for validation.
  asm volatile("{\n"
               ".reg .b32 tmp;\n"
               "prmt.b32 tmp, %2, 0, 0;\n"
               ".reg .b8 byte0, byte1, byte2, byte3;\n"
               "mov.b32 {byte0, byte1, byte2, byte3}, %1;\n"
               ".reg .b32 ra;\n"
               "mov.b32 ra, {byte0, byte1, _, _};\n"
               "mul.e4m3x4.e2m1x4.e4m3x4.satfinite %0, ra, tmp;\n"
               "}\n"
               : "=r"(res)
               : "r"(src), "r"(static_cast<uint32_t>(sf.storage)));
#endif

  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t qmul4E4m3E2m1E4m3Higher4Elts(uint32_t src, cutlass::float_e4m3_t sf) {
  uint32_t res;
  // {$nv-internal-release begin}
  // NOTE(tizheng): This following PTX should issue QMUL4.E4M3x4.E2M1x4.N3210.E4M3x4.B0.
  // However, due to the PTX optimization and exposure problem, it does not generate
  // single QMUL4 instruction as expected. Compiler team is working on this:
  // - https://nvbugspro.nvidia.com/bug/5221546
  // - https://nvbugspro.nvidia.com/bug/5229203

  // For now, we create two paths: 1) functionality is not correct but delivers the expected
  // performance; 2) functionality is correct but performance is low. We use 1) for benchmark and 2)
  // for validation.
  // {$nv-internal-release end}
#ifdef TLLM_BENCHMARK_E2M1_KV_CACHE
  // {$nv-internal-release begin}
  // The intended SASS is QMUL4.E4M3x4.E2M1x4.N7654.E4M3x4.B0. We encode .N7654 into ctype as
  // .e3m4x4. We do not encode into atype because we run out of 4-bit atype. If using 8-bit atype,
  // the generated iswzA and iswzB will both become B3210. The SASS editing will be more than a
  // simple string replacement.
  // {$nv-internal-release end}
  asm volatile("{\n"
               "mul.e3m4x4.e2m1x4.e3m2x4.satfinite %0, %1, %2;\n"
               "}\n"
               : "=r"(res)
               : "r"(src), "r"(static_cast<uint32_t>(sf.storage)));
#else
  asm volatile("{\n"
               ".reg .b32 tmp;\n"
               "prmt.b32 tmp, %2, 0, 0;\n"
               ".reg .b8 byte0, byte1, byte2, byte3;\n"
               "mov.b32 {byte0, byte1, byte2, byte3}, %1;\n"
               ".reg .b32 ra;\n"
               "mov.b32 ra, {_, _, byte2, byte3};\n"
               "mul.e4m3x4.e2m1x4.e4m3x4.satfinite %0, ra, tmp;\n"
               "}\n"
               : "=r"(res)
               : "r"(src), "r"(static_cast<uint32_t>(sf.storage)));
#endif
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cute::uint128_t convertE2m1ToE4m3(uint64_t srcX16, cutlass::float_e4m3_t sf) {
  // This function converts 16 E2m1 to E4m3.
  // Inputs:
  // - src: 16 E2M1 values packed in uint64_t.
  // - sf: The scaling factor of the vector, in E4M3 format.

  // The array to hold the 16 FP8 results.
  uint32_t dst[4];

  // Only Blackwell has the qmul4 instruction.
#if !(__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
  // Convert SF from e4m3 to fp16x2 by packing the byte twice, then converting.
  uint16_t sfPacked = uint16_t(sf.storage) * 256u + uint16_t(sf.storage);
  uint32_t sfFp16x2;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfFp16x2) : "h"(sfPacked));

  uint32_t(&srcWords)[2] = reinterpret_cast<uint32_t(&)[2]>(srcX16);

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    asm volatile("{\n"
                 ".reg .b8 b0, b1, b2, b3;\n"
                 ".reg .b32 h0, h1, h2, h3;\n"
                 ".reg .b16 e0, e1, e2, e3;\n"
                 "mov.b32 {b0, b1, b2, b3}, %2;\n"
                 "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
                 "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
                 "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
                 "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
                 "mul.rn.f16x2 h0, h0, %3;\n"
                 "mul.rn.f16x2 h1, h1, %3;\n"
                 "mul.rn.f16x2 h2, h2, %3;\n"
                 "mul.rn.f16x2 h3, h3, %3;\n"
                 "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
                 "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
                 "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
                 "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
                 "mov.b32 %0, {e0, e1};\n"
                 "mov.b32 %1, {e2, e3};\n"
                 "}\n"
                 : "=r"(dst[ii * 2 + 0]), "=r"(dst[ii * 2 + 1])
                 : "r"(srcWords[ii]), "r"(sfFp16x2));
  }
#else
  // Cast the src into uint32_t array of 2 elements.
  uint32_t(&e2m1x8Array)[2] = reinterpret_cast<uint32_t(&)[2]>(srcX16);

  // Call QMUL4 to convert E2M1 to E4M3.
  dst[0] = qmul4E4m3E2m1E4m3Lower4Elts(e2m1x8Array[0], sf);
  dst[1] = qmul4E4m3E2m1E4m3Higher4Elts(e2m1x8Array[0], sf);
  dst[2] = qmul4E4m3E2m1E4m3Lower4Elts(e2m1x8Array[1], sf);
  dst[3] = qmul4E4m3E2m1E4m3Higher4Elts(e2m1x8Array[1], sf);
#endif

  return reinterpret_cast<cute::uint128_t&>(dst);
}
// {$nv-internal-release end}
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeNvFp4SfAndOutputScale(float& outputScale,
                                                    cutlass::float_e4m3_t& sfOut,
                                                    float const& amax,
                                                    float const& sfScale) {
  // The reciprocal of E2M1_MAX.
  float constexpr reciprocalOfE2m1Max = 1.f / 6.f;
  // The FP32 sf.
  float sfVal = sfScale * amax * reciprocalOfE2m1Max;
  // The FP8 sf.
  sfOut = cutlass::float_e4m3_t{sfVal};
  // Cast back to FP32.
  sfVal = static_cast<float>(sfOut);

  // The output scale.
  // outputScale = reciprocal(fp32(fp8(sfVal * sfScale))) * reciprocal(sfScale))
  // {$nv-internal-release begin}
  // Same as
  // https://gitlab-master.nvidia.com/ftp/tekit/-/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh?ref_type=heads#L443
  // {$nv-internal-release end}
  outputScale =
    sfVal != 0.f ? reciprocal_approximate_ftz(sfVal * reciprocal_approximate_ftz(sfScale)) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeNvFp4SfAndOutputScale(float& outputScale,
                                                    cutlass::float_e4m3_t& sfOut,
                                                    float const& amax,
                                                    float const& sfScale,
                                                    float const& sfScaleInv) {
  // The reciprocal of E2M1_MAX.
  float constexpr reciprocalOfE2m1Max = 1.f / 6.f;
  // The FP32 sf.
  float sfVal = sfScale * amax * reciprocalOfE2m1Max;
  // The FP8 sf.
  sfOut = cutlass::float_e4m3_t{sfVal};
  // Cast back to FP32.
  sfVal = static_cast<float>(sfOut);

  // The output scale.
  // outputScale = reciprocal(fp32(fp8(sfVal * sfScale))) * reciprocal(sfScale))
  // Same as
  // {$nv-internal-release begin}
  // https://gitlab-master.nvidia.com/ftp/tekit/-/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh?ref_type=heads#L443
  // {$nv-internal-release end}
  outputScale = sfVal != 0.f ? reciprocal_approximate_ftz(sfVal * sfScaleInv) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float16 type Tensor to E2m1.
template <int NumEltsPerThread, typename OutT>
inline __device__ void convertFp16ToE2m1(OutT& out,
                                         cutlass::float_e4m3_t& sfOut,
                                         cutlass::half_t const (&in)[NumEltsPerThread],
                                         float sfScale) {

  // This function converts fp16 elements to E2m1.
  // Inputs:
  // - in: fp16 values.
  // - sfScale: The global scaling factor.
  // Outputs:
  // - return value: The casted E2m1 values.
  // - sfOut: The local scaling factor.

  // The number of half2 elements per thread.
  int32_t constexpr NumHalf2PerThread = NumEltsPerThread / 2;
  // The number of threads per vector of 16 elements.
  int32_t constexpr NumThreadsPerVec = 16 / NumEltsPerThread;

  // Cast the input to half2 array.
  half2 const(&inH2)[NumHalf2PerThread] = reinterpret_cast<half2 const(&)[NumHalf2PerThread]>(in);

  // Get absolute maximum values among the local 8 values.
  auto localAmax = __habs2(inH2[0]);
#pragma unroll
  for (int32_t ii = 1; ii < NumHalf2PerThread; ++ii) {
    localAmax = __hmax2(localAmax, __habs2(inH2[ii]));
  }

  // Perform warp-level reduction to achieve the amax of the vector of 16 elements.
  if constexpr (NumThreadsPerVec > 1) {
    static_assert(NumThreadsPerVec == 2 || NumThreadsPerVec == 4, "Not supported.");
    for (int32_t step = 1; step < NumThreadsPerVec; step *= 2) {
      localAmax = __hmax2(__shfl_xor_sync(uint32_t(-1), localAmax, step), localAmax);
    }
  }

  // Get the final absolute maximum values.
  float vecAmax = float(__hmax(localAmax.x, localAmax.y));

  // Compute the sf and output scale.
  float outputScale;
  computeNvFp4SfAndOutputScale(outputScale, sfOut, vecAmax, sfScale);

  // Apply the output scale.
  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int32_t ii = 0; ii < NumHalf2PerThread; ++ii) {
    float2 tmp = __half22float2(inH2[ii]);
    scaled[ii * 2 + 0] = tmp.x * outputScale;
    scaled[ii * 2 + 1] = tmp.y * outputScale;
  }

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(scaled);
  // Make sure the type size is as expected.
  static_assert(sizeof(arrFp4) == NumEltsPerThread / 2 && sizeof(arrFp4) == sizeof(OutT));
  // Cast to E2m1.
  out = reinterpret_cast<OutT const&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to E2m1.
template <int NumEltsPerThread, typename OutT>
inline __device__ void convertFloatToE2m1(OutT& out,
                                          cutlass::float_e4m3_t& sfOut,
                                          float const (&in)[NumEltsPerThread],
                                          float sfScale) {

  // This function converts float to E2m1.
  // Inputs:
  // - in: The tensor holding float values.
  // - sfScale: The global scaling factor.
  // Outputs:
  // - return value: The casted E2m1 values.
  // - sfOut: The local scaling factor.

  // The number of threads per vector of 16 elements.
  int32_t constexpr NumThreadsPerVec = 16 / NumEltsPerThread;

  // Find the local amax.
  float localAmax = 0;
#pragma unroll
  for (int ii = 0; ii < NumEltsPerThread; ii++) {
    localAmax = fmaxf(localAmax, fabsf(in[ii]));
  }

  // Get the absolute maximum among all 16 values.
  if constexpr (NumThreadsPerVec > 1) {
    static_assert(NumThreadsPerVec == 2 || NumThreadsPerVec == 4, "Not supported.");
#pragma unroll
    for (int32_t step = 1; step < NumThreadsPerVec; step *= 2) {
      localAmax = fmaxf(__shfl_xor_sync(uint32_t(-1), localAmax, step), localAmax);
    }
  }

  // Compute the sf and output scale.
  float outputScale;
  computeNvFp4SfAndOutputScale(outputScale, sfOut, localAmax, sfScale);

  // Apply the output scale.
  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int ii = 0; ii < NumEltsPerThread; ii++) {
    scaled[ii] = in[ii] * outputScale;
  }

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(scaled);
  // Cast to E2m1.
  out = reinterpret_cast<OutT&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to E2m1.
template <int NumEltsPerThread, typename OutT>
inline __device__ void convertFloatToE2m1(OutT& out, float const (&in)[NumEltsPerThread]) {
  // Apply the output scale.
  auto arrFp32{reinterpret_cast<cutlass::Array<float, NumEltsPerThread> const&>(in[0])};

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(arrFp32);
  // Cast to E2m1.
  out = reinterpret_cast<OutT&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to E2m1.
inline __device__ uint32_t convertFloat8ToE2m1(float in0,
                                               float in1,
                                               float in2,
                                               float in3,
                                               float in4,
                                               float in5,
                                               float in6,
                                               float in7) {

  cutlass::Array<float, 8> arrFp32{in0, in1, in2, in3, in4, in5, in6, in7};

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(arrFp32);
  // Cast to uint32_t.
  return reinterpret_cast<uint32_t&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF tensor where
// dataRowIdx and dataBlkColIdx are the respective indices of the row and the block of 16 elts
// from the K dim in the tensor of data.
inline __device__ int64_t getSfOffset(int32_t dataRowIdx,
                                      int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {

  // The number of rows of SF per block.
  static int32_t constexpr NumRowsPerSfBlock = 128;
  // The number of cols of SF per block.
  static int32_t constexpr NumColsPerSfBlock = 4;
  // The size of each SF block.
  static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

  // The number of rows of data per SF block.
  static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
  // The number of cols of blocks of data per SF block.
  static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;

  // The row of the SF block in the SF tensor.
  int sfBlkRowIdx = dataRowIdx / NumDataRowsPerSfBlock;
  // The col of the SF block in the SF tensor.
  int sfBlkColIdx = dataBlkColIdx / NumDataBlkColsPerSfBlock;
  // The blocks are stored row-major in the tensor of scaling factors.
  int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumDataBlkColsPerSfBlock + sfBlkColIdx;

  // Find the row in the SF block.
  int sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % NumDataRowsPerSfBlock) / 32;
  // Find the col in the SF block.
  int sfColIdx = (dataBlkColIdx % 4);

  // Compute the offset in bytes.
  return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given the GMEM address of an output element, compute the offset of the corresponding scaling
// factor in the SF tensor. Optionally, a startTokenIndex can be provided if the first token is not
// the start token in the SF tensor. This is useful when inflight batching is enabled in TRT-LLM,
// where the context and generation output are stored as one output tensor. In this case, the
// generation output may not start with zero offset in the SF output tensor.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(int64_t gmemOffsetInBytes,
                                      int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  // The number of elements per sf.
  int32_t constexpr NumEltsPerSf = 16;
  // The GMEM offset of the output element.
  int64_t gmemOffset = gmemOffsetInBytes * 8 /*bits*/ / NumBitsPerElt;
  // The row/col indices of the corresponding SF element.
  int32_t sfRowIdx = gmemOffset / hiddenDim + startTokenIdx;
  int32_t sfColIdx = (gmemOffset % hiddenDim) / NumEltsPerSf;
  // Compute the SF offset.
  return getSfOffset(sfRowIdx, sfColIdx, hiddenDim / NumEltsPerSf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(tizheng): Refactor to track gmem offset instead of doing pointer subtraction.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(void const* gmemOutPtr,
                                      void const* gmemBasePtr,
                                      int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  return getSfOffset<NumBitsPerElt>(reinterpret_cast<char const*>(gmemOutPtr) -
                                      reinterpret_cast<char const*>(gmemBasePtr),
                                    hiddenDim,
                                    startTokenIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// {$nv-internal-release begin}
template <int sfByteIdx>
inline __device__ void placeholderMxFp4x8ToBf16x8(uint32_t* eltsBf16x2,
                                                  uint32_t elts,
                                                  uint32_t sfE8x4) {
  // This function generates placeholder instructions that will be replaced through SASS patching.
  // LOP3.LUT will be replaced with F2FP.BF16.E2M1.UNPACK_B, which does fused conversion of scaling
  // of E2m1 elements and E8 scaling factors to Bf16.
  //
  // I (lsugy) chose his instruction as it can be quite robustly generated and shares a similar
  // signature with the target instruction. The LUT immediate is used to encode the Bx selectors for
  // the e2m1 element in B and the scaling factor in C. The LUT values meet the following criteria:
  //  - Make all three LOP3 operands necessary so the optimizer can't remove any of them or reorder
  //    them (in the two-operand variant, I've seen it switch operands and change the LUT).
  //  - Minimize the probability of side effects by avoiding common LUT values (and, or, xor, ...).
  // At the moment the compiler is not bothered by the fact that Ra and Rb are the same, if that
  // changes we may need to feed another value into the unused Ra.
  //
  // The mapping is as follows:
  //        | Rc.B0 | Rc.B1 | Rc.B2 | Rc.B3 |
  //  Rb.B0 |  0x40 |  0x41 |  0x42 |  0x43 |
  //  Rb.B1 |  0x60 |  0x61 |  0x62 |  0x63 |
  //  Rb.B2 |  0xb0 |  0xb1 |  0xb2 |  0xb3 |
  //  Rb.B3 |  0xd0 |  0xd1 |  0xd2 |  0xd3 |
  //
  // (70-73 and 80-83 should also be usable in case there's a conflict with any of these)

  asm volatile("lop3.b32 %0, %1, %1, %2, %3;"
               : "=r"(eltsBf16x2[0])
               : "r"(elts), "r"(sfE8x4), "n"(0x40 + sfByteIdx));
  asm volatile("lop3.b32 %0, %1, %1, %2, %3;"
               : "=r"(eltsBf16x2[1])
               : "r"(elts), "r"(sfE8x4), "n"(0x60 + sfByteIdx));
  asm volatile("lop3.b32 %0, %1, %1, %2, %3;"
               : "=r"(eltsBf16x2[2])
               : "r"(elts), "r"(sfE8x4), "n"(0xb0 + sfByteIdx));
  asm volatile("lop3.b32 %0, %1, %1, %2, %3;"
               : "=r"(eltsBf16x2[3])
               : "r"(elts), "r"(sfE8x4), "n"(0xd0 + sfByteIdx));
}
// {$nv-internal-release end}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm

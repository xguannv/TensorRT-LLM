/*
 * Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef __cuda_packing_h__
#define __cuda_packing_h__

#include "cuda_stdint.h"

#if defined(__LP64__) || defined(_WIN64)
// 64-bit
#define CU_32_BIT_PAD_ON_32_BIT_BUILDS(name)
#define CU_32_BIT_PAD_ON_64_BIT_BUILDS(name) uint32_t name;
#else
// 32-bit
#define CU_32_BIT_PAD_ON_32_BIT_BUILDS(name) uint32_t name;
#define CU_32_BIT_PAD_ON_64_BIT_BUILDS(name)
#endif

#if !defined(CU_SPECIFY_CUSTOM_PACKING_MACROS)

// CU_PACK_BEGIN/CU_PACK_END:
// Cross platform macros for defining packed structures.
// If you use BEGIN, don't forget to use END!

#ifdef _MSC_VER // MSVC
#define CU_PACK_BEGIN __pragma(pack(push, 1))

// Note:  Do not remove preceding blank line!
#define CU_PACK_END __pragma(pack(pop))

// Note:  Do not remove preceding blank line!
#elif defined(__GNUC__)
#if (__GNUC__ < 3)
// Old versions of GCC, do nothing.  While
// there is support for per-struct packing,
// there is no support for file-wide push &
// pop.  Assume if you're using old GCC that
// you're building everything with the same
// compiler and same packing semantics.
#define CU_PACK_BEGIN
#define CU_PACK_END
#else
#define CU_PACK_BEGIN _Pragma("pack(push, 1)")

// Note:  Do not remove preceding blank line!
#define CU_PACK_END _Pragma("pack(pop)")

// Note:  Do not remove preceding blank line!
#endif
#endif

// Implementation notes:
//
// #pragma is not usable within preprocessor macros, but
// every modern compiler supports a macro-compatible way
// to specify pragmas.  MSVC uses __pragma(...) and C99
// uses _Pragma("...").  Also, the original attempt to
// implement this had the packing size (hard-coded to 1
// above) as a parameter.  Unfortunately, GCC fails to
// preprocess in the correct order, so stringizing and
// string literal concatenation don't work for the
// parameter to _Pragma, preventing anything but hard-
// coded parameter strings.  Feel free to add another
// macro here for defining custom packing if you can
// find a well-supported way to do it.  Also, do not
// remove the blank lines in these macros!  They are
// protecting code that comes after the CU_PACK macros
// from ending up on the same line as macro-compatible
// pragmas, which may be bad on some compilers.

#endif // !defined(CU_SPECIFY_CUSTOM_PACKING_MACROS)

#endif // file guard

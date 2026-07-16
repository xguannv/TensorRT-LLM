/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : cudaNVCFG.h
 *
 *  Description              :
 *      NVCONFIG support; g_nvconfig.h should be pre-included via
 *      extra CFLAGS to the overall build. If not, then all configurations
 *      evaluate to 'enabled'
 */

#ifndef cudaNVCFG_INCLUDED
#define cudaNVCFG_INCLUDED
#ifndef NVCFG
#define NVCFG(x) 1
#endif
#endif

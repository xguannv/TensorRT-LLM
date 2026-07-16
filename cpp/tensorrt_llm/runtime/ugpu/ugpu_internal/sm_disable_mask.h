/*
 * Copyright 1993-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_sm_disable_mask_h__
#define __cuda_etbl_sm_disable_mask_h__

#include "cuda.h"
#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    //------------------------------------------------------------------
    // Cuda Private API for sm disable mask.
    //------------------------------------------------------------------

    // {8B7E90EB-8CF2-4a00-B1BD-08AA535590DB}
    CU_DEFINE_UUID(CU_ETID_SmDisableMask, 0x8b7e90eb, 0x8cf2, 0x4a00, 0xb1, 0xd1, 0x08, 0xaa, 0x53, 0x55, 0x90, 0xdb);

    typedef struct CUetblSmDisableMask_st
    {
        // Size of this structure
        size_t struct_size;

        // Set the valid sm disable mask for this stream
        // \param smDisableMask  String input either starts with prefix hex ("0x") or binary("0b")
        //        For example, smDisableMask = "0xFFFFFFFF" or smDisableMask = "0b11110000"
        CUresult(CUDAAPI* cuStreamSetDisableMask)(CUstream hStream, char const* smDisableMask);

        // Set the valid sm disable mask globally
        // \param smDisableMask  String input either starts with prefix hex ("0x") or binary("0b")
        //        For example, smDisableMask = "0xFFFFFFFF" or smDisableMask = "0b11110000"
        CUresult(CUDAAPI* cuGlobalSetDisableMask)(char const* smDisableMask);

        // Set the valid GPC/TPC selection mask.
        // \param qmdTpcSelMask The array of TPC selection masks, one mask for each GPC.
        // \param numGPC The size of the array.
        CUresult(CUDAAPI* cuSetTpcSelMask)(unsigned int* qmdTpcSelMask, unsigned int numGPC);

    } CUetblSmDisableMask;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard

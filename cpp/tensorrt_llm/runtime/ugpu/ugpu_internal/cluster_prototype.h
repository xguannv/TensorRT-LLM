/*
 * Copyright 2020-2022 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __CLUSTER_PROTOTYPE_H__
#define __CLUSTER_PROTOTYPE_H__

#include "g_nvconfig.h"

#include "cuda.h"

#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    CU_DEFINE_UUID(
        CU_ETID_CLUSTER_PROTOTYPE, 0x26dc3417, 0x0d80, 0x4547, 0x87, 0x26, 0xc0, 0xf1, 0xe7, 0xdd, 0x8b, 0xca);

#define CU_ETID_ClusterPrototype CU_ETID_CLUSTER_PROTOTYPE

    typedef struct CUetblClusterPrototype_st
    {
        size_t struct_size;

        // \brief Set the default cluster dimensions of a function. All
        // subsequent launches will default to use the set cluster size.
        // To reset, set all dimensions to 0.
        //
        CUresult(CUDAAPI* SetFunctionClusterDim)(CUfunction func, int clusterDimX, int clusterDimY, int clusterDimZ);

        // \brief Set the default cluster scheduling policy. All
        // subsequent launches of this kernel function will default to use
        // the set policy.
        // To reset, set the policy to CU_CLUSTER_SCHEDULING_POLICY_DEFAULT.
        //
        CUresult(CUDAAPI* SetFunctionClusterSchedulingPolicy)(CUfunction func, int policy);

        // \brief Allow the function to be launched with non-portable
        // cluster size.
        //
        CUresult(CUDAAPI* SetFunctionClusterNonPortableSizeSupport)(CUfunction func, int enable);

        // \brief Return the number of plural TPCs for skyline testing
        //
        CUresult(CUDAAPI* GetTpcIDtovGPCmTPCMapping)(unsigned char** vgpc, unsigned char** mtpc);

        // \brief Return the cluster dimensions of a function.
        //
        CUresult(CUDAAPI* GetFunctionClusterDim)(CUfunction func, int* clusterDimX, int* clusterDimY, int* clusterDimZ);

        // \brief Return the scheduling policy of a function.
        //
        CUresult(CUDAAPI* GetFunctionClusterSchedulingPolicy)(CUfunction func, int* policy);

        // \brief Return the ability of a function to be launched with a non-portable cluster size.
        //
        CUresult(CUDAAPI* GetFunctionClusterNonPortableSizeSupport)(CUfunction func, int* enable);

        // \brief Return the ability of a device to launch a clustered kernel.
        //
        CUresult(CUDAAPI* GetDeviceClusterLaunchSupported)(CUdevice dev, int* clusterLaunchSupported);

        // \brief Return the max potential cluster size in blocks based on function attribute settings
        CUresult(CUDAAPI* GetFunctionClusterMaxPotentialClusterSize)(
            CUfunction function, int blockSize, size_t dynamicSMemSize, int* maxClusterSize);

        // \brief Return the max number of active clusters on the GPU based on function attribute settings
        CUresult(CUDAAPI* GetFunctionClusterMaxNumActiveClusters)(
            CUfunction function, int blockSize, size_t dynamicSMemSize, int clusterSize, int* numClusters);

        // \brief Override maximum per device cluster size and return old value.
        uint32_t(CUDAAPI* OverrideMaximumPerDeviceClusterSize)(int device, uint32_t size);

        // \brief Return the requirement for a function's cluster dimension to be set.
        //
        CUresult(CUDAAPI* GetFunctionClusterDimMustBeSet)(CUfunction func, int* clusterDimMustBeSet);

        // \brief Free vgpc and mtpc pointers allocated during GetTpcIDtovGPCmTPCMapping
        // This is required on windows as it doesn't allow free() across shared libraries
        CUresult(CUDAAPI* FreeTpcIDtovGPCmTPCMappingPointers)(unsigned char* vgpc, unsigned char* mtpc);

        // \brief Return the drivers support for CNP cluster attributes.
        //
        CUresult(CUDAAPI* GetCnpClusterAttrsSupported)(int* cnpClusterAttrsSupported);

        // \brief Return the vGPC ID of the first singleton vGPC.
        //
        CUresult(CUDAAPI* GetSingletonVGPCBase)(CUdevice dev, unsigned char* singletonVGPCBase);

        // \brief Return the drivers support for preferred cluster dimensions.
        //
        CUresult(CUDAAPI* GetPreferredClusterSupported)(CUdevice dev, int* preferredClusterSupported);
    } CUetblClusterPrototype;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __CLUSTER_PROTOTYPE_H__

/*
 * Copyright 1993-2024 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_ugpu_localization_h__
#define __cuda_etbl_ugpu_localization_h__

#include "cuda.h"
#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    //------------------------------------------------------------------
    // Cuda Private API for uGPU localization
    //------------------------------------------------------------------

    // {592407f4-6b1b-4d1c-97ed-1f13ddb56c22}
    CU_DEFINE_UUID(
        CU_ETID_uGpuLocalization, 0x592407f4, 0x6b1b, 0x4d1c, 0x97, 0xed, 0x1f, 0x13, 0xdd, 0xb5, 0x6c, 0x22);

    // Localized memory locations
    typedef enum
    {
        CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_ANY = 0,
        CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU0 = 1,
        CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU1 = 2,
        CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_MAX
    } CUetblUGpuLocalizationMemoryLocation;

// Minimum string size for SM disable mask is the size of the mask + 2 (0x prefix) + 1 (null terminating character)
#define CU_UGPU_LOCALIZATION_SM_DISABLE_MASK_HEXADECIMAL_LENGTH 64
#define CU_UGPU_LOCALIZATION_SM_DISABLE_MASK_STRING_LENGTH CU_UGPU_LOCALIZATION_SM_DISABLE_MASK_HEXADECIMAL_LENGTH + 3

    typedef struct CUetblUGpuLocalization_st
    {
        // Size of this structure
        size_t struct_size;

        /// \brief Build the uGPU SM disable mask
        /// \param ctx              CUDA ctx which will run the uGPU ID discovery kernel
        CUresult(CUDAAPI* cuBuildUGpuSmDisableMask)(CUcontext ctx);

        /// \brief Get uGPU SM disable mask
        /// \param ctx              CUDA ctx which previously ran the uGPU ID discovery kernel
        /// \param location         uGPU partition to get the SM disable for
        /// \param smDisableMask    String representing SM disable mask, as a hex number
        ///                         This should be large enough to store the entire string
        CUresult(CUDAAPI* cuGetUGpuSmDisableMask)(
            CUcontext ctx, CUetblUGpuLocalizationMemoryLocation location, char* smDisableMask);

        /// \brief Set the SM disable mask for the specified uGPU
        /// \param ctx      CUDA ctx which previously ran the uGPU ID discovery kernel
        /// \param location uGPU partition to set the SM disable for
        CUresult(CUDAAPI* cuSetUGpuSmDisableMask)(CUcontext ctx, CUetblUGpuLocalizationMemoryLocation location);

        /// \brief Set the SM disable mask for the specified TPC
        /// \param ctx  CUDA ctx which previously ran the uGPU ID discovery kernel
        /// \param tpc  TPC to enable (the rest will be disabled)
        CUresult(CUDAAPI* cuSetSingleSmDisableMask)(CUcontext ctx, unsigned int tpc);

        /// \brief Allocate memory on a specified localized uGPU partition
        /// \param ctx          CUDA ctx which will own the returned allocation
        /// \param dptr         Pointer to be updated with the allocation
        /// \param sizeInBytes  Size in bytes
        /// \param location     uGPU partition to get the allocation from
        CUresult(CUDAAPI* cuMemDeviceAllocLocalized)(
            CUcontext ctx, void** dptr, size_t sizeInBytes, CUetblUGpuLocalizationMemoryLocation location);

        /// \brief Set the SM disable mask for the specified uGPU on a stream
        /// \param stream   CUDA stream to localize launches to
        /// \param location uGPU partition to the SM disable for
        CUresult(CUDAAPI* cuSetUGpuSmDisableMaskStream)(CUstream stream, CUetblUGpuLocalizationMemoryLocation location);

        /// \brief Query the localization support
        /// \param device                   The device for which we would like to query the localization support
        /// \param isLocalizationSupported  Returns whether localization is supported
        CUresult(CUDAAPI* cuGetDeviceLocalizationSupport)(CUdevice device, int* isLocalizationSupported);

        /// \brief Create a CUDA handle representing a localized memory allocation of a given size described by the
        /// given properties. \param handle   Value of handle returned. All operations on this allocation are to be
        /// performed using this handle. \param size     Size of the allocation requested \param prop     Properties of
        /// the allocation to create \param flags    For future use, must be zero \param location Localized memory
        /// location to create the allocation in
        CUresult(CUDAAPI* cuMemCreateLocalized)(CUmemGenericAllocationHandle* handle, size_t size,
            CUmemAllocationProp const* prop, unsigned long long flags, CUetblUGpuLocalizationMemoryLocation location);

    } CUetblUGpuLocalization;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard

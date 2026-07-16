/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#define CU_INIT_UUID
#include "cbl2_cuda_base_driver.h"
#include "green_context.h"
#include "sm_disable_mask.h"
#include "tools_device.h"
#include "ugpu_localization.h"
#undef CU_INIT_UUID

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include "tensorrt_llm/runtime/ugpu/ugpu_utils.h"

namespace tensorrt_llm
{

namespace ugpu
{

namespace
{

int popcount64(uint64_t value)
{
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(value));
#else
    return __builtin_popcountll(static_cast<unsigned long long>(value));
#endif
}

} // namespace

// Stream creation method for uGPU localization
enum class UgpuStreamCreateMethod
{
    GreenContext, // Use Green Context to create stream
    uGPU,         // Use uGPU Localization API to create stream
    Balanced      // Use SmDisableMask API with balanced TPC distribution
};

// Parse environment variable to get stream create method
static UgpuStreamCreateMethod parseStreamCreateMethod()
{
    char const* env = std::getenv("TLLM_UGPU_STREAM_CREATE_METHOD");
    if (env == nullptr)
    {
        return UgpuStreamCreateMethod::GreenContext; // Default
    }

    std::string method(env);
    if (method == "GreenContext" || method == "greencontext" || method == "green")
    {
        return UgpuStreamCreateMethod::GreenContext;
    }
    else if (method == "uGPU" || method == "ugpu" || method == "UGPU")
    {
        return UgpuStreamCreateMethod::uGPU;
    }
    else if (method == "Balanced" || method == "balanced" || method == "BALANCED")
    {
        return UgpuStreamCreateMethod::Balanced;
    }

    TLLM_LOG_WARNING("[UgpuLocalization] Unknown TLLM_UGPU_STREAM_CREATE_METHOD=%s, using default (GreenContext)", env);
    return UgpuStreamCreateMethod::GreenContext;
}

struct GpuResourceMask
{
    static constexpr int kMaxResourceCount = 512;
    static constexpr int kMaxMaskArraySize = (kMaxResourceCount + 64 - 1) / 64;
    uint64_t mMaskArray[kMaxMaskArraySize];

    bool isMaskSet(int resourceId) const
    {
        return mMaskArray[resourceId / 64] & (1ULL << (resourceId % 64));
    }

    void setMask(int resourceId)
    {
        mMaskArray[resourceId / 64] |= (1ULL << (resourceId % 64));
    }

    void clearMask(int resourceId)
    {
        mMaskArray[resourceId / 64] &= ~(1ULL << (resourceId % 64));
    }

    void setAllMasks()
    {
        for (int i = 0; i < kMaxMaskArraySize; i++)
        {
            mMaskArray[i] = 0xFFFFFFFFFFFFFFFFULL;
        }
    }

    void clearAllMasks()
    {
        for (int i = 0; i < kMaxMaskArraySize; i++)
        {
            mMaskArray[i] = 0ULL;
        }
    }

    int countZeros() const
    {
        int zeroCount = 0;
        for (int i = 0; i < kMaxMaskArraySize; i++)
        {
            zeroCount += popcount64(~mMaskArray[i]);
        }
        return zeroCount;
    }

    int countOnes() const
    {
        int oneCount = 0;
        for (int i = 0; i < kMaxMaskArraySize; i++)
        {
            oneCount += popcount64(mMaskArray[i]);
        }
        return oneCount;
    }

    // Get all enabled resource IDs (where mask bit is 0, meaning not disabled)
    std::vector<int> getEnabledIds(int maxId) const
    {
        std::vector<int> enabledIds;
        for (int i = 0; i < maxId; i++)
        {
            if (!isMaskSet(i))
            {
                enabledIds.push_back(i);
            }
        }
        return enabledIds;
    }
};

// Single TPC detailed information from CBL2
struct TpcDetailInfo
{
    int gpcId = -1;
    int virtualGpcId = -1;
    int migratableTpcId = -1;
    int globalTpcId = -1;

    std::string toString() const
    {
        std::stringstream ss;
        ss << "globalTpcId=" << globalTpcId << ", gpcId=" << gpcId << ", virtualGpcId=" << virtualGpcId
           << ", migratableTpcId=" << migratableTpcId;
        return ss.str();
    }
};

// GPC topology information
struct GpcInfo
{
    int gpcId;
    std::vector<int> tpcIds; // TPC IDs within this GPC (global TPC index)

    int getTpcCount() const
    {
        return static_cast<int>(tpcIds.size());
    }
};

// GPU topology detection class
class GpuTopology
{
public:
    GpuTopology()
        : mNumGpcs(0)
        , mNumTpcs(0)
        , mMaxTpcsPerGpc(0)
        , mIsInitialized(false)
        , mHasTpcInfo(false)
    {
    }

    ~GpuTopology() {}

    // Detect TPC info using CBL2 cblCudaGetTpcInfo API
    // This provides accurate TPC-to-GPC mapping
    bool detectTpcInfoFromCbl2()
    {
        // Get CBL2 export table
        CUetblCBL2CudaBaseDriver* etblCbl2 = nullptr;
        auto result = cuGetExportTable((void const**) &etblCbl2, &CU_ETID_CBL2CudaBaseDriver);
        if (result != CUDA_SUCCESS || etblCbl2 == nullptr)
        {
            TLLM_LOG_WARNING("[GpuTopology] CBL2 export table not available, will use fallback mapping");
            return false;
        }

        // Create base driver for the current context
        CUcontext ctx = nullptr;
        TLLM_CU_CHECK(cuCtxGetCurrent(&ctx));
        if (ctx == nullptr)
        {
            TLLM_LOG_WARNING("[GpuTopology] No CUDA context available for CBL2");
            return false;
        }

        cblCudaCreateBaseDriverParams createParams = {};
        createParams.hContext = ctx;
        createParams.cudaContextDestroyCallback = nullptr;

        result = etblCbl2->cblCudaCreateBaseDriver(&createParams);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("[GpuTopology] Failed to create CBL2 base driver");
            return false;
        }

        // Get TPC info
        cblCudaGetTpcInfoParams tpcInfoParams = {};
        tpcInfoParams.hBaseDriver = createParams.hBaseDriver;

        result = etblCbl2->cblCudaGetTpcInfo(&tpcInfoParams);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("[GpuTopology] cblCudaGetTpcInfo failed");
            etblCbl2->cblCudaDestroyBaseDriver(createParams.hBaseDriver);
            return false;
        }

        // Store the TPC info
        mNumGpcs = static_cast<int>(tpcInfoParams.numGpcs);
        mNumTpcs = static_cast<int>(tpcInfoParams.numTpcInfo);
        mMaxTpcsPerGpc = static_cast<int>(tpcInfoParams.numTpcsPerGpc);

        // Build TPC-to-GPC mapping and store full TPC details from tpcInfoArray
        // globalTpcId is the TPC ID used in tpcDisableMask
        mTpcToGpc.clear();
        mTpcToGpc.resize(CBL_CUDA_TPCINFO_ARRAY_SIZE, -1);
        mTpcDetails.clear();
        mTpcDetails.resize(CBL_CUDA_TPCINFO_ARRAY_SIZE);

        for (NvU32 i = 0; i < tpcInfoParams.numTpcInfo; i++)
        {
            int globalTpcId = tpcInfoParams.tpcInfoArray[i].globalTpcId;
            int gpcId = tpcInfoParams.tpcInfoArray[i].gpcId;

            if (globalTpcId >= 0 && globalTpcId < CBL_CUDA_TPCINFO_ARRAY_SIZE)
            {
                mTpcToGpc[globalTpcId] = gpcId;

                // Store full TPC details
                mTpcDetails[globalTpcId].globalTpcId = globalTpcId;
                mTpcDetails[globalTpcId].gpcId = gpcId;
                mTpcDetails[globalTpcId].virtualGpcId = tpcInfoParams.tpcInfoArray[i].virtualGpcId;
                mTpcDetails[globalTpcId].migratableTpcId = tpcInfoParams.tpcInfoArray[i].migratableTpcId;
            }
        }

        TLLM_LOG_INFO("[GpuTopology] TPC info from CBL2: NumGPCs=%d, NumTPCs=%d, NumTpcsPerGPC=%d", mNumGpcs, mNumTpcs,
            mMaxTpcsPerGpc);

        // Log detailed TPC mapping
        for (NvU32 i = 0; i < tpcInfoParams.numTpcInfo; i++)
        {
            TLLM_LOG_DEBUG("[GpuTopology] TPC[%d]: gpcId=%d, virtualGpcId=%d, migratableTpcId=%d, globalTpcId=%d", i,
                tpcInfoParams.tpcInfoArray[i].gpcId, tpcInfoParams.tpcInfoArray[i].virtualGpcId,
                tpcInfoParams.tpcInfoArray[i].migratableTpcId, tpcInfoParams.tpcInfoArray[i].globalTpcId);
        }

        // Cleanup
        etblCbl2->cblCudaDestroyBaseDriver(createParams.hBaseDriver);

        mHasTpcInfo = true;
        return true;
    }

    // Detect basic GPU topology parameters using tools_device export table
    // This only gets the basic parameters, not the per-GPC distribution
    bool detectBasicParams(int deviceId = -1)
    {
        TLLM_CU_CHECK(cuInit(0));
        TLLM_CUDA_CHECK(cudaFree(0));

        if (deviceId < 0)
        {
            TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
        }
        else
        {
            mDeviceId = deviceId;
        }

        // First try to get accurate TPC info from CBL2
        if (detectTpcInfoFromCbl2())
        {
            TLLM_LOG_INFO("[GpuTopology] Using accurate TPC-to-GPC mapping from CBL2");
            return true;
        }

        // Fallback: Get export table for tools device
        TLLM_LOG_INFO("[GpuTopology] Falling back to tools_device for basic params");

        CUetblToolsDevice* etblToolsDevice = nullptr;
        auto result = cuGetExportTable((void const**) &etblToolsDevice, &CU_ETID_ToolsDevice);
        TLLM_CHECK_WITH_INFO(result == CUDA_SUCCESS && etblToolsDevice != nullptr,
            "[GpuTopology] Failed to get ToolsDevice export table");

        CUdevice cuDevice;
        TLLM_CU_CHECK(cuDeviceGet(&cuDevice, mDeviceId));

        // Query GPC/TPC topology information
        CUtoolsVariant value;

        // Get number of GPCs
        result = etblToolsDevice->DeviceGetAttributeProperty(
            cuDevice, CU_TOOLS_DEVICE_ATTRIBUTE_LIMITS_NUM_GPCS, CU_TOOLS_DEVICE_ATTRIBUTE_PROPERTY_VALUE, &value);
        TLLM_CHECK_WITH_INFO(result == CUDA_SUCCESS, "[GpuTopology] Failed to get LIMITS_NUM_GPCS");
        mNumGpcs = static_cast<int>(value.data.s64);

        // Get number of TPCs
        result = etblToolsDevice->DeviceGetAttributeProperty(
            cuDevice, CU_TOOLS_DEVICE_ATTRIBUTE_LIMITS_NUM_TPCS, CU_TOOLS_DEVICE_ATTRIBUTE_PROPERTY_VALUE, &value);
        TLLM_CHECK_WITH_INFO(result == CUDA_SUCCESS, "[GpuTopology] Failed to get LIMITS_NUM_TPCS");
        mNumTpcs = static_cast<int>(value.data.s64);

        // Get max TPCs per GPC
        result = etblToolsDevice->DeviceGetAttributeProperty(cuDevice,
            CU_TOOLS_DEVICE_ATTRIBUTE_LIMITS_MAX_TPCS_PER_GPC, CU_TOOLS_DEVICE_ATTRIBUTE_PROPERTY_VALUE, &value);
        TLLM_CHECK_WITH_INFO(result == CUDA_SUCCESS, "[GpuTopology] Failed to get LIMITS_MAX_TPCS_PER_GPC");
        mMaxTpcsPerGpc = static_cast<int>(value.data.s64);

        TLLM_LOG_INFO(
            "[GpuTopology] Basic params: NumGPCs=%d, NumTPCs=%d, MaxTPCsPerGPC=%d", mNumGpcs, mNumTpcs, mMaxTpcsPerGpc);

        // Validate parameters
        TLLM_CHECK_WITH_INFO(mNumGpcs > 0, "[GpuTopology] Invalid NumGPCs");
        TLLM_CHECK_WITH_INFO(mNumTpcs > 0, "[GpuTopology] Invalid NumTPCs");
        TLLM_CHECK_WITH_INFO(mMaxTpcsPerGpc > 0, "[GpuTopology] Invalid MaxTpcsPerGpc");

        return true;
    }

    // Get GPC ID for a given TPC ID using accurate mapping if available
    int getGpcIdForTpcInternal(int tpcId) const
    {
        // Use accurate mapping from CBL2 if available
        if (mHasTpcInfo && tpcId >= 0 && tpcId < static_cast<int>(mTpcToGpc.size()) && mTpcToGpc[tpcId] >= 0)
        {
            return mTpcToGpc[tpcId];
        }
        // Fallback: estimate using MaxTpcsPerGpc
        if (mMaxTpcsPerGpc > 0)
        {
            return tpcId / mMaxTpcsPerGpc;
        }
        return -1;
    }

    // Build GPC topology from tpcDisableMask
    // NOTE: The "smDisableMask" from uGPU localization is actually a TPC-level mask!
    //       Each bit represents a TPC, not an individual SM.
    //       bit=1 means TPC is disabled, bit=0 means TPC is enabled
    // TPC to GPC mapping: Uses accurate CBL2 mapping if available, otherwise GPC_ID = TPC_ID / MaxTpcsPerGpc
    void buildTopologyFromTpcDisableMask(char const* tpcDisableMask)
    {
        TLLM_CHECK_WITH_INFO(mNumGpcs > 0 && mMaxTpcsPerGpc > 0,
            "[GpuTopology] Must call detectBasicParams() before buildTopologyFromTpcDisableMask()");

        // Parse the tpcDisableMask (each bit represents one TPC)
        GpuResourceMask tpcMask;
        parseTpcDisableMask(tpcDisableMask, &tpcMask);

        // Initialize GPC structures
        mGpcs.clear();
        mGpcs.resize(mNumGpcs);
        for (int gpcId = 0; gpcId < mNumGpcs; gpcId++)
        {
            mGpcs[gpcId].gpcId = gpcId;
        }

        // Iterate through all TPCs and assign them to GPCs
        // Uses accurate CBL2 mapping if available
        for (int tpcId = 0; tpcId < mNumTpcs; tpcId++)
        {
            int gpcId = getGpcIdForTpcInternal(tpcId);

            if (gpcId >= 0 && gpcId < mNumGpcs)
            {
                // Check if this TPC is enabled (bit=0 means enabled)
                if (!tpcMask.isMaskSet(tpcId))
                {
                    mGpcs[gpcId].tpcIds.push_back(tpcId);
                }
            }
        }

        // Sort TPC IDs within each GPC for consistent output
        for (auto& gpc : mGpcs)
        {
            std::sort(gpc.tpcIds.begin(), gpc.tpcIds.end());
        }

        // Validate the result
        int totalEnabledTpcs = 0;
        for (auto const& gpc : mGpcs)
        {
            totalEnabledTpcs += gpc.getTpcCount();
        }

        TLLM_LOG_WARNING("[GpuTopology] Built topology from tpcDisableMask: %d enabled TPCs (using %s mapping)",
            totalEnabledTpcs, mHasTpcInfo ? "CBL2 accurate" : "estimated");

        // Print the per-GPC distribution
        printPerGpcDistribution();

        mIsInitialized = true;
    }

    // Full topology detection (for standalone use without tpcDisableMask)
    // Assumes all TPCs are enabled
    bool detect(int deviceId = -1)
    {
        if (mIsInitialized)
        {
            return true;
        }

        // Get basic parameters first
        detectBasicParams(deviceId);

        // Build topology with all TPCs enabled
        // Uses accurate CBL2 mapping if available
        mGpcs.clear();
        mGpcs.resize(mNumGpcs);

        for (int gpcId = 0; gpcId < mNumGpcs; gpcId++)
        {
            mGpcs[gpcId].gpcId = gpcId;
        }

        for (int tpcId = 0; tpcId < mNumTpcs; tpcId++)
        {
            int gpcId = getGpcIdForTpcInternal(tpcId);

            if (gpcId >= 0 && gpcId < mNumGpcs)
            {
                mGpcs[gpcId].tpcIds.push_back(tpcId);
            }
        }

        // Sort for consistent output
        for (auto& gpc : mGpcs)
        {
            std::sort(gpc.tpcIds.begin(), gpc.tpcIds.end());
        }

        TLLM_LOG_INFO("[GpuTopology] Detected topology using %s mapping", mHasTpcInfo ? "CBL2 accurate" : "estimated");

        // Print the per-GPC distribution
        printPerGpcDistribution();

        mIsInitialized = true;
        return true;
    }

    // Print topology information
    void print() const
    {
        TLLM_LOG_INFO("[GpuTopology] Device %d Topology:", mDeviceId);
        TLLM_LOG_INFO("  NumGPCs: %d", mNumGpcs);
        TLLM_LOG_INFO("  NumTPCs: %d", mNumTpcs);
        TLLM_LOG_INFO("  MaxTPCsPerGPC: %d", mMaxTpcsPerGpc);

        for (auto const& gpc : mGpcs)
        {
            std::stringstream tpcStr;
            tpcStr << "TPCs[";
            for (size_t i = 0; i < gpc.tpcIds.size(); i++)
            {
                if (i > 0)
                    tpcStr << ",";
                tpcStr << gpc.tpcIds[i];
            }
            tpcStr << "]";

            TLLM_LOG_INFO("  GPC%d: %s", gpc.gpcId, tpcStr.str().c_str());
        }
    }

    // Get topology as string for logging
    std::string toString() const
    {
        std::stringstream ss;
        ss << "Device " << mDeviceId << " Topology:\n";
        ss << "  NumGPCs=" << mNumGpcs << ", NumTPCs=" << mNumTpcs;
        ss << ", MaxTPCsPerGPC=" << mMaxTpcsPerGpc << "\n";

        for (auto const& gpc : mGpcs)
        {
            ss << "  GPC" << gpc.gpcId << ": " << gpc.tpcIds.size() << " TPCs\n";
        }
        return ss.str();
    }

    // Getters
    int getNumGpcs() const
    {
        return mNumGpcs;
    }

    int getNumTpcs() const
    {
        return mNumTpcs;
    }

    int getMaxTpcsPerGpc() const
    {
        return mMaxTpcsPerGpc;
    }

    std::vector<GpcInfo> const& getGpcs() const
    {
        return mGpcs;
    }

    bool isInitialized() const
    {
        return mIsInitialized;
    }

    bool hasTpcInfo() const
    {
        return mHasTpcInfo;
    }

    // Get detailed TPC info (only valid if hasTpcInfo() returns true)
    TpcDetailInfo const* getTpcDetail(int tpcId) const
    {
        if (mHasTpcInfo && tpcId >= 0 && tpcId < static_cast<int>(mTpcDetails.size())
            && mTpcDetails[tpcId].globalTpcId >= 0)
        {
            return &mTpcDetails[tpcId];
        }
        return nullptr;
    }

    // Get all TPC details
    std::vector<TpcDetailInfo> const& getTpcDetails() const
    {
        return mTpcDetails;
    }

    // Get GPC ID for a given TPC ID
    // Uses accurate CBL2 mapping if available, otherwise searches built topology
    int getGpcIdForTpc(int tpcId) const
    {
        // First try accurate mapping from CBL2
        if (mHasTpcInfo && tpcId >= 0 && tpcId < static_cast<int>(mTpcToGpc.size()) && mTpcToGpc[tpcId] >= 0)
        {
            return mTpcToGpc[tpcId];
        }
        // Fallback: search in built topology
        for (auto const& gpc : mGpcs)
        {
            for (int tpc : gpc.tpcIds)
            {
                if (tpc == tpcId)
                {
                    return gpc.gpcId;
                }
            }
        }
        return -1;
    }

    // Get TPC count for a specific GPC
    int getTpcCountForGpc(int gpcId) const
    {
        if (gpcId >= 0 && gpcId < static_cast<int>(mGpcs.size()))
        {
            return mGpcs[gpcId].getTpcCount();
        }
        return -1;
    }

    // Print per-GPC TPC distribution
    void printPerGpcDistribution() const
    {
        TLLM_LOG_INFO("[GpuTopology] Per-GPC TPC Distribution:");
        for (auto const& gpc : mGpcs)
        {
            std::stringstream ss;
            ss << "  GPC" << gpc.gpcId << ": " << gpc.getTpcCount() << " TPCs (";
            for (size_t i = 0; i < gpc.tpcIds.size(); i++)
            {
                if (i > 0)
                    ss << ",";
                ss << gpc.tpcIds[i];
            }
            ss << ")";
            TLLM_LOG_INFO("%s", ss.str().c_str());
        }
    }

private:
    // Parse TPC disable mask from hex string (e.g., "0xFFFF0000...")
    // Each bit represents one TPC: bit=1 means TPC disabled, bit=0 means TPC enabled
    static void parseTpcDisableMask(char const* disableMask, GpuResourceMask* gpuResourceMask)
    {
        char const* ptr = disableMask;
        gpuResourceMask->setAllMasks(); // Default: all disabled

        int maskStrLen = strlen(ptr);
        if (maskStrLen < 2 || ptr[0] != '0' || ptr[1] != 'x')
        {
            return;
        }

        ptr += 2;
        maskStrLen -= 2;

        if (maskStrLen * 4 > GpuResourceMask::kMaxResourceCount)
        {
            return;
        }

        int bitIndex = 0;
        for (int i = maskStrLen - 1; i >= 0; i--)
        {
            char c = ptr[i];
            int hexValue = 0;

            if (c >= '0' && c <= '9')
            {
                hexValue = c - '0';
            }
            else if (c >= 'A' && c <= 'F')
            {
                hexValue = c - 'A' + 10;
            }
            else if (c >= 'a' && c <= 'f')
            {
                hexValue = c - 'a' + 10;
            }
            else
            {
                return;
            }

            for (int j = 0; j < 4; j++)
            {
                if (bitIndex >= GpuResourceMask::kMaxResourceCount)
                    break;
                if (hexValue & (1 << j))
                {
                    gpuResourceMask->setMask(bitIndex);
                }
                else
                {
                    gpuResourceMask->clearMask(bitIndex);
                }
                bitIndex++;
            }
        }
    }

    int mDeviceId = 0;
    int mNumGpcs;
    int mNumTpcs;
    int mMaxTpcsPerGpc;
    std::vector<GpcInfo> mGpcs;
    std::vector<int> mTpcToGpc;             // Accurate TPC-to-GPC mapping from CBL2
    std::vector<TpcDetailInfo> mTpcDetails; // Full TPC details from CBL2
    bool mIsInitialized;
    bool mHasTpcInfo;                       // True if accurate TPC info from CBL2 is available
};

// Global storage for GPU topologies (singleton per device)
static std::map<int, std::unique_ptr<GpuTopology>> gGpuTopologyMap;

// Build GPU topology from tpcDisableMask - this is the preferred method
// NOTE: The "smDisableMask" from uGPU localization is actually a TPC-level mask!
//       Each bit represents one TPC (containing SmPerTpc SMs, typically 2)
//       bit=1 means TPC disabled, bit=0 means TPC enabled
GpuTopology* buildGpuTopologyFromTpcDisableMask(int deviceId, char const* tpcDisableMask)
{
    if (deviceId < 0)
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    }

    TLLM_CHECK_WITH_INFO(tpcDisableMask != nullptr, "[GpuTopology] tpcDisableMask cannot be null");

    // Create new topology and build from tpcDisableMask
    auto topology = std::make_unique<GpuTopology>();

    // First detect basic parameters
    topology->detectBasicParams(deviceId);

    // Then build the topology using tpcDisableMask
    topology->buildTopologyFromTpcDisableMask(tpcDisableMask);

    // Store and return
    gGpuTopologyMap[deviceId] = std::move(topology);
    return gGpuTopologyMap[deviceId].get();
}

// Get existing GPU topology (must have been built first with buildGpuTopologyFromTpcDisableMask)
GpuTopology* getGpuTopology(int deviceId = -1)
{
    if (deviceId < 0)
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    }

    auto it = gGpuTopologyMap.find(deviceId);
    if (it != gGpuTopologyMap.end())
    {
        return it->second.get();
    }

    // If not found, fail - we require explicit building from tpcDisableMask
    TLLM_CHECK_WITH_INFO(false,
        "[GpuTopology] Topology not found for device %d. "
        "Must call buildGpuTopologyFromTpcDisableMask() first.",
        deviceId);
    return nullptr;
}

// Print GPU topology for current device
void printGpuTopology()
{
    int deviceId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));

    auto it = gGpuTopologyMap.find(deviceId);
    if (it != gGpuTopologyMap.end() && it->second != nullptr)
    {
        it->second->print();
    }
    else
    {
        TLLM_LOG_WARNING("[GpuTopology] No topology available for device %d", deviceId);
    }
}

static void parseDisableMask(char const* disableMask, GpuResourceMask* gpuResourceMask)
{
    char const* ptr = disableMask;
    gpuResourceMask->setAllMasks();
    // The disable mask is like "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7E9D3A74E8E8E8E8E8"
    int maskStrLen = strlen(ptr);

    // Check if the string starts with "0x"
    TLLM_CHECK(maskStrLen >= 2 && ptr[0] == '0' && ptr[1] == 'x');

    // Skip "0x" prefix
    ptr += 2;
    maskStrLen -= 2;

    // Check if the hex string is too long (each hex char represents 4 bits)
    TLLM_CHECK(maskStrLen * 4 <= GpuResourceMask::kMaxResourceCount);

    // Parse from the end of the string (low bits are at the end)
    int bitIndex = 0;

    for (int i = maskStrLen - 1; i >= 0; i--)
    {
        char c = ptr[i];
        int hexValue = 0;

        // Parse hexadecimal character
        if (c >= '0' && c <= '9')
        {
            hexValue = c - '0';
        }
        else if (c >= 'A' && c <= 'F')
        {
            hexValue = c - 'A' + 10;
        }
        else if (c >= 'a' && c <= 'f')
        {
            hexValue = c - 'a' + 10;
        }
        else
        {
            // Invalid character, fail
            TLLM_CHECK(false);
        }

        // Each hexadecimal character represents 4 bits
        for (int j = 0; j < 4; j++)
        {
            TLLM_CHECK(bitIndex < GpuResourceMask::kMaxResourceCount);
            if (hexValue & (1 << j))
            {
                gpuResourceMask->setMask(bitIndex);
            }
            else
            {
                gpuResourceMask->clearMask(bitIndex);
            }
            bitIndex++;
        }
    }
}

class UgpuLocalizationGreenContext
{
public:
    UgpuLocalizationGreenContext()
    {
        Init();
    }

    ~UgpuLocalizationGreenContext()
    {
        destroyGreenContexts();
    }

    CUstream createLocalizedStream(int ugpuId)
    {
        TLLM_CHECK(ugpuId >= 0 && ugpuId < static_cast<int>(mUgpuCount));
        CUstream stream{};
        auto result = cuGreenCtxStreamCreate(&stream, mGreenCtx[ugpuId], CU_STREAM_NON_BLOCKING, 0);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("[GreenContext] failed to create localized stream; falling back to uGPU method");
            return nullptr;
        }
        return stream;
    }

    bool supported() const
    {
        return mSupported && mUgpuCount > 1;
    }

private:
    static constexpr int mMaxUgpuCount = 2;
    CUetblGreenContext* mEtblGreenContext = NULL;
    CUcontext mCtx = NULL;
    CUgreenCtx mGreenCtx[mMaxUgpuCount]{};

    CUdevResourceDesc mLocalizedDesc[mMaxUgpuCount];
    CUdevResource mLocalizedResources[mMaxUgpuCount];

    int mDev = 0;
    unsigned int mUgpuCount = 0;
    bool mSupported = false;

    void destroyGreenContexts()
    {
        for (unsigned int i = 0; i < std::min(mUgpuCount, static_cast<unsigned int>(mMaxUgpuCount)); i++)
        {
            if (mGreenCtx[i] != nullptr)
            {
                auto result = cuGreenCtxDestroy(mGreenCtx[i]);
                if (result != CUDA_SUCCESS)
                {
                    TLLM_LOG_WARNING("[GreenContext] failed to destroy green context %u", i);
                }
                mGreenCtx[i] = nullptr;
            }
        }
        mSupported = false;
    }

    void Init()
    {
        TLLM_CU_CHECK(cuInit(0));
        TLLM_CUDA_CHECK(cudaFree(0));
        TLLM_CUDA_CHECK(cudaGetDevice(&mDev));
        TLLM_CU_CHECK(cuCtxGetCurrent(&mCtx));

        auto result = cuGetExportTable((void const**) &mEtblGreenContext, &CU_ETID_GreenContext);
        if (result != CUDA_SUCCESS || mEtblGreenContext == NULL)
        {
            TLLM_LOG_WARNING("[GreenContext] export table not available; GreenContext stream creation disabled");
            return;
        }

        CUdevResource smResources{};
        result = cuDeviceGetDevResource(mDev, &smResources, CU_DEV_RESOURCE_TYPE_SM);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("[GreenContext] failed to get SM resources; GreenContext stream creation disabled");
            return;
        }

        mUgpuCount = 0;
        CUdevResource remainingResources{};
        result = mEtblGreenContext->cuDevSmResourceSplitByUgpu(NULL, &mUgpuCount, &smResources, NULL);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("[GreenContext] failed to query uGPU split; GreenContext stream creation disabled");
            return;
        }
        if (mUgpuCount <= 1)
        {
            return;
        }
        if (mUgpuCount > mMaxUgpuCount)
        {
            TLLM_LOG_WARNING(
                "[GreenContext] expected at most %d uGPU partitions, got %u; GreenContext stream creation disabled",
                mMaxUgpuCount, mUgpuCount);
            return;
        }
        result = mEtblGreenContext->cuDevSmResourceSplitByUgpu(
            &mLocalizedResources[0], &mUgpuCount, &smResources, &remainingResources);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING(
                "[GreenContext] failed to split SM resources by uGPU; GreenContext stream creation disabled");
            return;
        }

        TLLM_LOG_WARNING("[GreenContext] smResources: totalCount=%d, uGPU0 count=%d, uGPU1 count=%d, remaining=%d",
            smResources.sm.smCount, mLocalizedResources[0].sm.smCount, mLocalizedResources[1].sm.smCount,
            remainingResources.sm.smCount);

        for (unsigned int i = 0; i < mUgpuCount; i++)
        {
            result = cuDevResourceGenerateDesc(&mLocalizedDesc[i], &mLocalizedResources[i], 1);
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "[GreenContext] failed to generate resource descriptor; GreenContext stream creation disabled");
                destroyGreenContexts();
                return;
            }
            result = cuGreenCtxCreate(&mGreenCtx[i], mLocalizedDesc[i], mDev, CU_GREEN_CTX_DEFAULT_STREAM);
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "[GreenContext] failed to create green context; GreenContext stream creation disabled");
                destroyGreenContexts();
                return;
            }
        }
        mSupported = true;
    }
};

class UgpuLocalization
{
public:
    UgpuLocalization()
    {
        Init();
    }

    ~UgpuLocalization() {}

    CUresult localizedDeviceAlloc(void** localizedDevPtr, size_t size, int ugpuId)
    {
        if (ugpuId == -1 || !mIsLocalizationSupported)
        {
            return cuMemAlloc(reinterpret_cast<CUdeviceptr*>(localizedDevPtr), size);
        }
        checkCurrentContext();
        TLLM_CHECK(ugpuId >= 0 && ugpuId < mUgpuCount);
        auto localizationFlag = ugpuIdToLocalization(ugpuId);
        auto cuResult = mEtblUGpuLocalization->cuMemDeviceAllocLocalized(mCtx, localizedDevPtr, size, localizationFlag);
        return cuResult;
    }

    CUresult localizedDeviceFree(void* localizedDevPtr)
    {
        checkCurrentContext();
        TLLM_CU_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(localizedDevPtr)));
        return CUDA_SUCCESS;
    }

    CUresult createLocalizedAllocationHandle(CUmemGenericAllocationHandle* handle, size_t size, int ugpuId,
        unsigned int requestedHandleTypes, bool gpuDirectRDMACapable)
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = mDev;
        prop.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(requestedHandleTypes);
        prop.allocFlags.gpuDirectRDMACapable = gpuDirectRDMACapable ? 1 : 0;

        if (ugpuId == -1 || !mIsLocalizationSupported)
        {
            return cuMemCreate(handle, size, &prop, 0);
        }
        checkCurrentContext();
        TLLM_CHECK(ugpuId >= 0 && ugpuId < mUgpuCount);
        auto localizationFlag = ugpuIdToLocalization(ugpuId);
        return mEtblUGpuLocalization->cuMemCreateLocalized(handle, size, &prop, 0, localizationFlag);
    }

    CUstream createLocalizedStream(int ugpuId, UgpuStreamCreateMethod method)
    {
        CUstream stream = nullptr;

        switch (method)
        {
        case UgpuStreamCreateMethod::GreenContext:
            if ((mGreenContextUtils.get() != nullptr) && mGreenContextUtils->supported())
            {
                stream = mGreenContextUtils->createLocalizedStream(ugpuId);
                if (stream != nullptr)
                {
                    return stream;
                }
            }
            TLLM_LOG_WARNING("[UgpuLocalization] GreenContext not supported, falling back to uGPU method");
            // Fall through to uGPU method
            [[fallthrough]];

        case UgpuStreamCreateMethod::uGPU: return createUgpuLocalizedStream(ugpuId);

        case UgpuStreamCreateMethod::Balanced:
            if (mEtblSmDisableMask != nullptr && mBalancedMaskInitialized)
            {
                return createBalancedLocalizedStream(ugpuId);
            }
            TLLM_LOG_WARNING("[UgpuLocalization] Balanced mode not available, falling back to uGPU method");
            return createUgpuLocalizedStream(ugpuId);

        default: return createUgpuLocalizedStream(ugpuId);
        }
    }

    bool isLocalizationSupported() const
    {
        return mIsLocalizationSupported;
    }

    int getUgpuTpcCount(int ugpuId) const
    {
        TLLM_CHECK(ugpuId >= 0 && mIsLocalizationSupported);
        TLLM_CHECK(ugpuId < mUgpuCount);
        return mGpuResourceMasks[ugpuId].countZeros();
    }

    // Get GPU topology
    GpuTopology const* getTopology() const
    {
        return mTopology;
    }

    // Print detailed uGPU partition info with GPC breakdown
    void printUgpuPartitionTopology() const
    {
        if (!mIsLocalizationSupported || mTopology == nullptr)
        {
            TLLM_LOG_WARNING("[UgpuLocalization] Localization not supported or topology not available");
            return;
        }

        TLLM_LOG_WARNING("[UgpuLocalization] uGPU Partition Topology:");
        TLLM_LOG_WARNING("  Total GPCs: %d, Total TPCs: %d, MaxTPCsPerGPC: %d", mTopology->getNumGpcs(),
            mTopology->getNumTpcs(), mTopology->getMaxTpcsPerGpc());

        for (int ugpuId = 0; ugpuId < mUgpuCount; ugpuId++)
        {
            TLLM_LOG_WARNING("  uGPU%d:", ugpuId);

            // Count enabled TPCs per GPC for this uGPU partition
            std::map<int, std::vector<int>> gpcToTpcs;

            auto const& gpcs = mTopology->getGpcs();
            for (auto const& gpc : gpcs)
            {
                for (int tpcId : gpc.tpcIds)
                {
                    // Check if this TPC is enabled in this uGPU partition
                    // In disable mask: bit=0 means enabled, bit=1 means disabled
                    if (!mGpuResourceMasks[ugpuId].isMaskSet(tpcId))
                    {
                        gpcToTpcs[gpc.gpcId].push_back(tpcId);
                    }
                }
            }

            int totalEnabledTpcs = 0;
            for (auto const& [gpcId, tpcs] : gpcToTpcs)
            {
                TLLM_LOG_WARNING("    GPC%d: %zu TPCs", gpcId, tpcs.size());
                totalEnabledTpcs += tpcs.size();
            }
            TLLM_LOG_WARNING("    Total: %d TPCs", totalEnabledTpcs);
        }
    }

    // Get which GPCs are used by a specific uGPU partition
    std::vector<int> getUgpuGpcs(int ugpuId) const
    {
        std::vector<int> gpcs;
        if (!mIsLocalizationSupported || mTopology == nullptr || ugpuId < 0 || ugpuId >= mUgpuCount)
        {
            return gpcs;
        }

        auto const& allGpcs = mTopology->getGpcs();
        for (auto const& gpc : allGpcs)
        {
            for (int tpcId : gpc.tpcIds)
            {
                if (!mGpuResourceMasks[ugpuId].isMaskSet(tpcId))
                {
                    if (std::find(gpcs.begin(), gpcs.end(), gpc.gpcId) == gpcs.end())
                    {
                        gpcs.push_back(gpc.gpcId);
                    }
                    break;
                }
            }
        }
        return gpcs;
    }

    // Get which TPCs are used by a specific uGPU partition
    std::vector<int> getUgpuTpcs(int ugpuId) const
    {
        std::vector<int> tpcs;
        if (!mIsLocalizationSupported || mTopology == nullptr || ugpuId < 0 || ugpuId >= mUgpuCount)
        {
            return tpcs;
        }

        int numTpcs = mTopology->getNumTpcs();
        for (int tpcId = 0; tpcId < numTpcs; tpcId++)
        {
            if (!mGpuResourceMasks[ugpuId].isMaskSet(tpcId))
            {
                tpcs.push_back(tpcId);
            }
        }
        return tpcs;
    }

    static UgpuLocalization* getUgpuLocalization()
    {
        int devId = -1;
        TLLM_CUDA_CHECK(cudaGetDevice(&devId));
        // TODO: do we need support multiple context?
        auto key = devId;
        std::lock_guard<std::mutex> lock(UgpuLocalization::mUgpuLocalizationMutex);
        if (UgpuLocalization::mUgpuLocalizationMap.find(key) == UgpuLocalization::mUgpuLocalizationMap.end())
        {
            UgpuLocalization::mUgpuLocalizationMap[key] = new UgpuLocalization();
        }
        return UgpuLocalization::mUgpuLocalizationMap[key];
    }

private:
    static std::map<int, UgpuLocalization*> mUgpuLocalizationMap;
    static std::mutex mUgpuLocalizationMutex;

    CUstream createUgpuLocalizedStream(int ugpuId)
    {
        CUstream stream = nullptr;
        TLLM_CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CUresult result = CUDA_SUCCESS;
        try
        {
            result = setStreamLocalization(stream, ugpuId);
        }
        catch (...)
        {
            destroyStreamNoThrow(stream, "uGPU localization");
            throw;
        }
        if (result != CUDA_SUCCESS)
        {
            destroyStreamNoThrow(stream, "uGPU localization");
            TLLM_CU_CHECK(result);
        }
        return stream;
    }

    CUstream createBalancedLocalizedStream(int ugpuId)
    {
        CUstream stream = nullptr;
        TLLM_CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CUresult result = CUDA_SUCCESS;
        try
        {
            result = setStreamBalancedMask(stream, ugpuId);
        }
        catch (...)
        {
            destroyStreamNoThrow(stream, "balanced mask localization");
            throw;
        }
        if (result != CUDA_SUCCESS)
        {
            destroyStreamNoThrow(stream, "balanced mask localization");
            TLLM_CU_CHECK(result);
        }
        return stream;
    }

    static void destroyStreamNoThrow(CUstream stream, char const* context)
    {
        if (stream == nullptr)
        {
            return;
        }
        auto result = cuStreamDestroy(stream);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("[UgpuLocalization] failed to destroy stream after %s failure", context);
        }
    }

    void Init()
    {
        TLLM_CU_CHECK(cuInit(0));
        TLLM_CUDA_CHECK(cudaFree(0));
        TLLM_CUDA_CHECK(cudaGetDevice(&mDev));
        TLLM_CU_CHECK(cuCtxGetCurrent(&mCtx));

        // Get export tables
        auto result = cuGetExportTable((void const**) &mEtblUGpuLocalization, &CU_ETID_uGpuLocalization);

        if (result == CUDA_SUCCESS && mEtblUGpuLocalization != NULL)
        {
            mIsEtblSupported = true;
            mGreenContextUtils = std::make_unique<UgpuLocalizationGreenContext>();
        }
        else
        {
            mIsEtblSupported = false;
            mIsLocalizationSupported = false;
            return;
        }

        // Get SmDisableMask export table for Balanced mode
        result = cuGetExportTable((void const**) &mEtblSmDisableMask, &CU_ETID_SmDisableMask);
        if (result != CUDA_SUCCESS || mEtblSmDisableMask == NULL)
        {
            TLLM_LOG_WARNING(
                "[UgpuLocalization] SmDisableMask export table not available, Balanced mode will not work");
            mEtblSmDisableMask = nullptr;
        }

        int isLocalizationSupport = 0;
        mEtblUGpuLocalization->cuGetDeviceLocalizationSupport(mDev, &isLocalizationSupport);
        mIsLocalizationSupported = (isLocalizationSupport > 0);
        if (!mIsLocalizationSupported)
        {
            return;
        }

        TLLM_CU_CHECK(mEtblUGpuLocalization->cuBuildUGpuSmDisableMask(mCtx));

        // Get tpcDisableMask for ALL locations: ANY, UGPU0, UGPU1
        // NOTE: The API name says "SmDisableMask" but it's actually TPC-level!
        TLLM_CU_CHECK(mEtblUGpuLocalization->cuGetUGpuSmDisableMask(
            mCtx, CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_ANY, &mAnyTpcDisableMask[0]));
        TLLM_CU_CHECK(mEtblUGpuLocalization->cuGetUGpuSmDisableMask(
            mCtx, CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU0, &mOriginalTpcDisableMask[0][0]));
        TLLM_CU_CHECK(mEtblUGpuLocalization->cuGetUGpuSmDisableMask(
            mCtx, CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU1, &mOriginalTpcDisableMask[1][0]));

        parseDisableMask(&mAnyTpcDisableMask[0], &mAnyGpuResourceMask);
        parseDisableMask(&mOriginalTpcDisableMask[0][0], &mGpuResourceMasks[0]);
        parseDisableMask(&mOriginalTpcDisableMask[1][0], &mGpuResourceMasks[1]);

        // Count enabled TPCs for each location
        int anyTpcCount = mAnyGpuResourceMask.countZeros();
        int ugpu0TpcCount = getUgpuTpcCount(0);
        int ugpu1TpcCount = getUgpuTpcCount(1);

        TLLM_LOG_WARNING("ANY   TPC count=%d\n", anyTpcCount);
        TLLM_LOG_WARNING("uGPU0 TPC count=%d\n", ugpu0TpcCount);
        TLLM_LOG_WARNING("uGPU1 TPC count=%d\n", ugpu1TpcCount);
        TLLM_LOG_WARNING("ANY   TPC disable mask: %s\n", &mAnyTpcDisableMask[0]);
        TLLM_LOG_WARNING("uGPU0 TPC disable mask: %s\n", &mOriginalTpcDisableMask[0][0]);
        TLLM_LOG_WARNING("uGPU1 TPC disable mask: %s\n", &mOriginalTpcDisableMask[1][0]);

        // Initialize GPU topology detection using ANY tpcDisableMask
        // ANY mask shows all available TPCs on this GPU (full topology)
        mTopology = buildGpuTopologyFromTpcDisableMask(mDev, &mAnyTpcDisableMask[0]);
        TLLM_CHECK_WITH_INFO(
            mTopology != nullptr, "[UgpuLocalization] Failed to build GPU topology from tpcDisableMask");

        TLLM_LOG_WARNING("[UgpuLocalization] GPU Topology built from ANY tpcDisableMask:");
        TLLM_LOG_WARNING("  %s", mTopology->toString().c_str());

        // Print detailed info for unassigned TPCs (not in UGPU0 or UGPU1)
        // These are TPCs that physically exist (from CBL2) but not assigned to any uGPU partition
        if (mTopology->hasTpcInfo())
        {
            TLLM_LOG_WARNING("[UgpuLocalization] Unassigned TPCs (not in UGPU0 or UGPU1, CBL2 details):");
            int numTpcs = mTopology->getNumTpcs();
            int unassignedCount = 0;
            for (int tpcId = 0; tpcId < numTpcs; tpcId++)
            {
                // Check if this TPC is not assigned to either UGPU0 or UGPU1
                bool enabledInUgpu0 = !mGpuResourceMasks[0].isMaskSet(tpcId);
                bool enabledInUgpu1 = !mGpuResourceMasks[1].isMaskSet(tpcId);

                if (!enabledInUgpu0 && !enabledInUgpu1)
                {
                    TpcDetailInfo const* detail = mTopology->getTpcDetail(tpcId);
                    if (detail != nullptr)
                    {
                        TLLM_LOG_WARNING("  Unassigned TPC[%d]: %s", tpcId, detail->toString().c_str());
                    }
                    else
                    {
                        TLLM_LOG_WARNING("  Unassigned TPC[%d]: (no CBL2 detail available)", tpcId);
                    }
                    unassignedCount++;
                }
            }
            if (unassignedCount == 0)
            {
                TLLM_LOG_WARNING("  No unassigned TPCs (all TPCs assigned to UGPU0 or UGPU1)");
            }
            else
            {
                TLLM_LOG_WARNING("  Total unassigned TPCs: %d", unassignedCount);
            }
        }

        // Print detailed partition info
        printUgpuPartitionTopology();

        // Build balanced masks for Balanced mode
        if (mEtblSmDisableMask != nullptr)
        {
            buildBalancedMasks();
        }
    }

    CUresult setStreamLocalization(CUstream stream, int ugpuId)
    {
        if (ugpuId == -1 || !mIsLocalizationSupported)
        {
            // if no localization needed or not supported, do nothing.
            return CUDA_SUCCESS;
        }
        checkCurrentContext();
        TLLM_CHECK(ugpuId >= 0 && ugpuId < mUgpuCount);
        auto localizationFlag = ugpuIdToLocalization(ugpuId);
        return mEtblUGpuLocalization->cuSetUGpuSmDisableMaskStream(stream, localizationFlag);
    }

    CUresult setStreamBalancedMask(CUstream stream, int ugpuId)
    {
        if (ugpuId == -1 || !mBalancedMaskInitialized || mEtblSmDisableMask == nullptr)
        {
            return CUDA_SUCCESS;
        }
        checkCurrentContext();
        TLLM_CHECK(ugpuId >= 0 && ugpuId < mUgpuCount);
        return mEtblSmDisableMask->cuStreamSetDisableMask(stream, mBalancedTpcDisableMask[ugpuId]);
    }

    CUetblUGpuLocalizationMemoryLocation ugpuIdToLocalization(int ugpuId)
    {
        if (ugpuId < 0 || ugpuId >= mUgpuCount)
        {
            return CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_ANY;
        }
        return ugpuId == 0 ? CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU0 : CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU1;
    }

    void checkCurrentContext()
    {
        CUcontext ctx = NULL;
        TLLM_CU_CHECK(cuCtxGetCurrent(&ctx));
        TLLM_CHECK(ctx == mCtx);
    }

    // Convert GpuResourceMask to hex string for sm_disable_mask API
    static void maskToHexString(GpuResourceMask const& mask, int numTpcs, char* outStr)
    {
        // The mask is stored with bit 0 at the lowest position
        // We need to output as hex string "0xXXXX..." where leftmost hex digit is highest bits
        outStr[0] = '0';
        outStr[1] = 'x';

        int numHexDigits = (numTpcs + 3) / 4; // Round up to cover all TPCs
        for (int i = 0; i < numHexDigits; i++)
        {
            int bitOffset = (numHexDigits - 1 - i) * 4;
            int hexValue = 0;
            for (int j = 0; j < 4; j++)
            {
                int bitIndex = bitOffset + j;
                if (bitIndex < GpuResourceMask::kMaxResourceCount && mask.isMaskSet(bitIndex))
                {
                    hexValue |= (1 << j);
                }
            }
            outStr[2 + i] = (hexValue < 10) ? ('0' + hexValue) : ('A' + hexValue - 10);
        }
        outStr[2 + numHexDigits] = '\0';
    }

    // Build balanced masks by distributing unassigned TPCs evenly
    void buildBalancedMasks()
    {
        if (mTopology == nullptr)
        {
            TLLM_LOG_WARNING("[UgpuLocalization] Cannot build balanced masks without topology");
            return;
        }

        int numTpcs = mTopology->getNumTpcs();

        // Find unassigned TPCs: physically present (from CBL2/tools_device) but not in UGPU0 or UGPU1
        // Note: numTpcs is from CBL2's cblCudaGetTpcInfo or tools_device's LIMITS_NUM_TPCS
        // A TPC is unassigned if it's disabled in both UGPU0 and UGPU1 masks
        std::vector<int> unassignedTpcs;
        for (int tpcId = 0; tpcId < numTpcs; tpcId++)
        {
            // bit=0 means TPC is enabled (belongs to that uGPU)
            // bit=1 means TPC is disabled (does not belong to that uGPU)
            bool enabledInUgpu0 = !mGpuResourceMasks[0].isMaskSet(tpcId);
            bool enabledInUgpu1 = !mGpuResourceMasks[1].isMaskSet(tpcId);

            // Unassigned = not in UGPU0 AND not in UGPU1
            if (!enabledInUgpu0 && !enabledInUgpu1)
            {
                unassignedTpcs.push_back(tpcId);
            }
        }

        TLLM_LOG_WARNING(
            "[UgpuLocalization] Building balanced masks: %zu unassigned TPCs found", unassignedTpcs.size());

        // Start with original uGPU masks
        mBalancedGpuResourceMasks[0] = mGpuResourceMasks[0];
        mBalancedGpuResourceMasks[1] = mGpuResourceMasks[1];

        // Distribute unassigned TPCs based on their GPC's uGPU affinity
        // Step 1: Determine which uGPU each GPC belongs to based on majority TPC assignment
        int numGpcs = mTopology->getNumGpcs();
        std::map<int, int> gpcToUgpu; // gpcId -> ugpuId (0 or 1)
        for (int gpcId = 0; gpcId < numGpcs; gpcId++)
        {
            int ugpu0Count = 0;
            int ugpu1Count = 0;

            for (int tpcId = 0; tpcId < numTpcs; tpcId++)
            {
                if (mTopology->getGpcIdForTpc(tpcId) == gpcId)
                {
                    // bit=0 means enabled
                    if (!mGpuResourceMasks[0].isMaskSet(tpcId))
                    {
                        ugpu0Count++;
                    }
                    else if (!mGpuResourceMasks[1].isMaskSet(tpcId))
                    {
                        ugpu1Count++;
                    }
                }
            }

            // Assign GPC to uGPU with more TPCs (tie goes to uGPU0)
            // If no TPCs assigned, this is an error - every GPC should have at least one TPC assigned
            TLLM_CHECK_WITH_INFO(
                ugpu0Count > 0 || ugpu1Count > 0, "[UgpuLocalization] GPC %d has no TPCs assigned to any uGPU", gpcId);
            gpcToUgpu[gpcId] = (ugpu0Count >= ugpu1Count) ? 0 : 1;
        }

        // Step 2: Calculate how many TPCs each uGPU still needs to reach balance
        int originalTpcCount0 = getUgpuTpcCount(0);
        int originalTpcCount1 = getUgpuTpcCount(1);
        int totalTpcs = originalTpcCount0 + originalTpcCount1 + static_cast<int>(unassignedTpcs.size());
        int targetPerUgpu = totalTpcs / 2;

        // Calculate remaining quota for each uGPU (how many more TPCs needed)
        int remainingQuota0 = std::max(0, targetPerUgpu - originalTpcCount0);
        int remainingQuota1 = std::max(0, targetPerUgpu - originalTpcCount1);

        TLLM_LOG_WARNING("[UgpuLocalization] Quota calculation: target=%d, uGPU0 needs %d more, uGPU1 needs %d more",
            targetPerUgpu, remainingQuota0, remainingQuota1);

        // Track which TPCs are assigned to each uGPU for logging
        std::vector<int> assignedToUgpu0;
        std::vector<int> assignedToUgpu1;
        std::vector<int> skippedTpcs; // TPCs that couldn't be assigned (both quotas exhausted)

        // Step 3: Assign each unassigned TPC based on its GPC's uGPU affinity
        for (int tpcId : unassignedTpcs)
        {
            int gpcId = mTopology->getGpcIdForTpc(tpcId);
            int preferredUgpu = gpcToUgpu[gpcId];
            int otherUgpu = 1 - preferredUgpu;

            int& preferredQuota = (preferredUgpu == 0) ? remainingQuota0 : remainingQuota1;
            int& otherQuota = (otherUgpu == 0) ? remainingQuota0 : remainingQuota1;

            if (preferredQuota > 0)
            {
                // Preferred uGPU still has quota, assign to it
                mBalancedGpuResourceMasks[preferredUgpu].clearMask(tpcId);
                preferredQuota--;
                if (preferredUgpu == 0)
                {
                    assignedToUgpu0.push_back(tpcId);
                }
                else
                {
                    assignedToUgpu1.push_back(tpcId);
                }
            }
            else if (otherQuota > 0)
            {
                // Preferred uGPU full, but other uGPU has quota, assign to other
                mBalancedGpuResourceMasks[otherUgpu].clearMask(tpcId);
                otherQuota--;
                if (otherUgpu == 0)
                {
                    assignedToUgpu0.push_back(tpcId);
                }
                else
                {
                    assignedToUgpu1.push_back(tpcId);
                }
            }
            else
            {
                // Both quotas exhausted, skip this TPC
                skippedTpcs.push_back(tpcId);
            }
        }

        // Convert to hex strings
        maskToHexString(mBalancedGpuResourceMasks[0], numTpcs, mBalancedTpcDisableMask[0]);
        maskToHexString(mBalancedGpuResourceMasks[1], numTpcs, mBalancedTpcDisableMask[1]);

        // Log the results
        int balancedTpcCount0 = mBalancedGpuResourceMasks[0].countZeros();
        int balancedTpcCount1 = mBalancedGpuResourceMasks[1].countZeros();

        TLLM_LOG_WARNING("[UgpuLocalization] Balanced TPC distribution (GPC-affinity aware):");
        TLLM_LOG_WARNING("  uGPU0: %d -> %d TPCs (+%zu from unassigned)", originalTpcCount0, balancedTpcCount0,
            assignedToUgpu0.size());
        TLLM_LOG_WARNING("  uGPU1: %d -> %d TPCs (+%zu from unassigned)", originalTpcCount1, balancedTpcCount1,
            assignedToUgpu1.size());
        TLLM_LOG_WARNING("  Balanced uGPU0 mask: %s", mBalancedTpcDisableMask[0]);
        TLLM_LOG_WARNING("  Balanced uGPU1 mask: %s", mBalancedTpcDisableMask[1]);

        // Print which TPCs were assigned to each uGPU
        if (!unassignedTpcs.empty())
        {
            std::stringstream ss0, ss1;
            ss0 << "  uGPU0 newly assigned TPCs: [";
            ss1 << "  uGPU1 newly assigned TPCs: [";
            for (size_t i = 0; i < assignedToUgpu0.size(); i++)
            {
                if (i > 0)
                    ss0 << ", ";
                ss0 << assignedToUgpu0[i];
            }
            for (size_t i = 0; i < assignedToUgpu1.size(); i++)
            {
                if (i > 0)
                    ss1 << ", ";
                ss1 << assignedToUgpu1[i];
            }
            ss0 << "]";
            ss1 << "]";
            TLLM_LOG_WARNING("%s", ss0.str().c_str());
            TLLM_LOG_WARNING("%s", ss1.str().c_str());

            // Log skipped TPCs if any (both quotas exhausted)
            if (!skippedTpcs.empty())
            {
                std::stringstream ssSkipped;
                ssSkipped << "  Skipped TPCs (quotas exhausted): [";
                for (size_t i = 0; i < skippedTpcs.size(); i++)
                {
                    if (i > 0)
                        ssSkipped << ", ";
                    ssSkipped << skippedTpcs[i];
                }
                ssSkipped << "]";
                TLLM_LOG_WARNING("%s", ssSkipped.str().c_str());
            }
        }

        mBalancedMaskInitialized = true;
    }

    static constexpr int mMaxUgpuCount = 2;
    static constexpr int mTpcDisableMaskSize = 256;

    bool mIsEtblSupported = false;
    bool mIsLocalizationSupported = false;
    bool mBalancedMaskInitialized = false;

    std::unique_ptr<UgpuLocalizationGreenContext> mGreenContextUtils;

    CUetblUGpuLocalization* mEtblUGpuLocalization = NULL;
    CUetblSmDisableMask* mEtblSmDisableMask = NULL;
    CUcontext mCtx = NULL;

    // NOTE: The "smDisableMask" from uGPU API is actually a TPC-level mask!
    // Each bit represents one TPC, not an individual SM.
    char mOriginalTpcDisableMask[mMaxUgpuCount][mTpcDisableMaskSize]; // UGPU0, UGPU1
    char mAnyTpcDisableMask[mTpcDisableMaskSize];                     // ANY (all TPCs enabled)
    char mBalancedTpcDisableMask[mMaxUgpuCount][mTpcDisableMaskSize]; // Balanced masks

    GpuResourceMask mGpuResourceMasks[mMaxUgpuCount];
    GpuResourceMask mAnyGpuResourceMask;
    GpuResourceMask mBalancedGpuResourceMasks[mMaxUgpuCount];

    GpuTopology* mTopology = nullptr;

    int mDev = 0;
    int mUgpuCount = mMaxUgpuCount;
};

std::map<int, UgpuLocalization*> UgpuLocalization::mUgpuLocalizationMap{};
std::mutex UgpuLocalization::mUgpuLocalizationMutex{};

// UgpuLocalizationHandle class implementation
UgpuLocalizationHandle::UgpuLocalizationHandle()
    : mImpl(UgpuLocalization::getUgpuLocalization())
{
}

UgpuLocalizationHandle::~UgpuLocalizationHandle() {}

UgpuLocalizationHandle::UgpuLocalizationHandle(UgpuLocalizationHandle&& other) noexcept
    : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

UgpuLocalizationHandle& UgpuLocalizationHandle::operator=(UgpuLocalizationHandle&& other) noexcept
{
    if (this != &other)
    {
        mImpl = other.mImpl;
        other.mImpl = nullptr;
    }
    return *this;
}

bool UgpuLocalizationHandle::supportsUgpuLocalization() const
{
    TLLM_CHECK(mImpl != nullptr);
    return mImpl->isLocalizationSupported();
}

void UgpuLocalizationHandle::ugpuMalloc(void** localizedDevPtr, size_t size, int ugpuId)
{
    TLLM_CHECK(mImpl != nullptr);
    TLLM_CU_CHECK(mImpl->localizedDeviceAlloc(localizedDevPtr, size, ugpuId));
}

void UgpuLocalizationHandle::ugpuFree(void* localizedDevPtr)
{
    TLLM_CHECK(mImpl != nullptr);
    TLLM_CU_CHECK(mImpl->localizedDeviceFree(localizedDevPtr));
}

CUmemGenericAllocationHandle UgpuLocalizationHandle::createUgpuLocalizedAllocationHandle(
    size_t size, int ugpuId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable)
{
    TLLM_CHECK(mImpl != nullptr);
    CUmemGenericAllocationHandle handle{};
    TLLM_CU_CHECK(
        mImpl->createLocalizedAllocationHandle(&handle, size, ugpuId, requestedHandleTypes, gpuDirectRDMACapable));
    return handle;
}

CUstream UgpuLocalizationHandle::createUgpuLocalizedStream(int ugpuId)
{
    TLLM_CHECK(mImpl != nullptr);
    UgpuStreamCreateMethod method = parseStreamCreateMethod();
    return mImpl->createLocalizedStream(ugpuId, method);
}

} // namespace ugpu

} // namespace tensorrt_llm

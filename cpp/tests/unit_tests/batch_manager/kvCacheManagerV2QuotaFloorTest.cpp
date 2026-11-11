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

#include "kvCacheManagerV2TestUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;

// makeConfig(): 4 tokens per block, one full-attention layer of 4 KiB per block, 4 MiB GPU quota.
constexpr size_t kConfiguredQuota = 4UL << 20;
constexpr int kTokensPerBlock = 4;

class KvCacheManagerV2QuotaFloorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    //! The manager, and what its construction logged. The logger writes every message to stdout or to stderr
    //! depending on its own level, not the message's, so both are captured.
    static std::pair<std::shared_ptr<KvCacheManager>, std::string> build(KVCacheManagerConfig const& config)
    {
        ::testing::internal::CaptureStdout();
        ::testing::internal::CaptureStderr();
        auto manager = std::make_shared<KvCacheManager>(config);
        auto log = ::testing::internal::GetCapturedStderr();
        log += ::testing::internal::GetCapturedStdout();
        return {std::move(manager), std::move(log)};
    }

    static KVCacheManagerConfig withSlidingLayer(int window)
    {
        auto config = makeConfig();
        AttentionLayerConfig layer;
        layer.layerId = 1;
        layer.buffers.push_back(BufferConfig{"key", 8192, std::nullopt});
        layer.slidingWindowSize = window;
        config.layers.emplace_back(std::move(layer));
        return config;
    }
};

TEST_F(KvCacheManagerV2QuotaFloorTest, FloorsWithinTheQuotaAreSilent)
{
    auto config = makeConfig();
    config.constraints.push_back(BatchDesc{{KVCacheDesc{64 * kTokensPerBlock, 0}}});
    auto [manager, log] = build(config);
    EXPECT_LE(manager->getQuota(kHotLevel), kConfiguredQuota);
    EXPECT_EQ(log.find("slot floors"), std::string::npos) << log;
}

TEST_F(KvCacheManagerV2QuotaFloorTest, AConstraintAboveTheQuotaRaisesItAndSaysSo)
{
    // 8192 blocks of 4 KiB is 32 MiB, eight times the quota.
    auto config = makeConfig();
    config.constraints.push_back(BatchDesc{{KVCacheDesc{8192 * kTokensPerBlock, 0}}});
    auto [manager, log] = build(config);

    // Reported, not prevented: the level is still sized at the floors.
    EXPECT_GE(manager->getQuota(kHotLevel), size_t{8192} * 4096);
    EXPECT_NE(log.find("KV cache GPU quota of " + std::to_string(kConfiguredQuota) + " bytes is below the"),
        std::string::npos)
        << log;
    EXPECT_NE(log.find("pool group 0 (life cycles 0 [full attention])"), std::string::npos) << log;
}

TEST_F(KvCacheManagerV2QuotaFloorTest, AWideWindowNamesItsLifeCycle)
{
    // No constraint at all: one request at the widest window is a floor for every attention life
    // cycle, so a window far beyond what the quota holds raises the quota on its own. This is the
    // shape of a window used only to give layers their own pool and sized from an unreachable length.
    int const window = 1 << 16;
    auto [manager, log] = build(withSlidingLayer(window));

    EXPECT_GT(manager->getQuota(kHotLevel), kConfiguredQuota);
    EXPECT_NE(log.find("[window " + std::to_string(window) + "]"), std::string::npos) << log;
    EXPECT_NE(log.find("[full attention]"), std::string::npos) << log;
}

TEST_F(KvCacheManagerV2QuotaFloorTest, AWindowThatFitsIsSilent)
{
    auto [manager, log] = build(withSlidingLayer(64 * kTokensPerBlock));
    EXPECT_LE(manager->getQuota(kHotLevel), kConfiguredQuota);
    EXPECT_EQ(log.find("slot floors"), std::string::npos) << log;
}

} // namespace

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

// KvCache::setAbsentBefore(): a sliding-window cache that starts at a block-aligned position
// with no pages before it. Window 12 at 4 tokens per block keeps 3 live blocks; the absent
// prefix is 16 tokens (blocks 0-3) of a 32-token (8-block) sequence.

#include "kvCacheManagerV2TestUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/common/assert.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeHybridTieredConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeTieredConfig;

constexpr int kTokensPerBlock = 4;
constexpr int kWindow = 12;
constexpr int kAbsent = 16;
constexpr int kLength = 32;

class AbsentPrefixTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        for (int i = 0; i < kLength; ++i)
        {
            mTokens.emplace_back(TokenId{i});
        }
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    static std::shared_ptr<KvCacheManager> makeSwaManager(
        std::optional<int> window = kWindow, std::function<void(KVCacheManagerConfig&)> const& edit = {})
    {
        auto config = makeConfig();
        EXPECT_EQ(config.tokensPerBlock, kTokensPerBlock);
        std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = window;
        if (edit)
        {
            edit(config);
        }
        return std::make_shared<KvCacheManager>(config);
    }

    // The absent blocks of a cache claimed to kLength read as bad page indices, the rest as pages.
    static void expectAbsentIndices(KvCacheManager& manager, KvCache const& cache)
    {
        auto const indices = cache.getBasePageIndices(manager.getLayerGroupId(LayerId{0}));
        for (int ord = 0; ord < kLength / kTokensPerBlock; ++ord)
        {
            SCOPED_TRACE(ord);
            EXPECT_EQ(indices[ord] == kBadPageIndex.value(), ord < kAbsent / kTokensPerBlock);
        }
    }

    // The whole prompt matches through its window, and its absent blocks carry no page.
    void expectAbsentTree(KvCacheManager& manager, bool windowOnly = false) const
    {
        auto const [hasPage, matched] = treePages(manager, span(0, kLength));
        EXPECT_EQ(matched, kLength);
        ASSERT_EQ(hasPage.size(), static_cast<size_t>(kLength / kTokensPerBlock));
        for (size_t ord = 0; ord < hasPage.size(); ++ord)
        {
            SCOPED_TRACE(ord);
            if (ord < static_cast<size_t>(kAbsent / kTokensPerBlock))
            {
                EXPECT_FALSE(hasPage[ord]);
            }
            else if (!windowOnly || ord >= static_cast<size_t>((kLength + 1 - kWindow) / kTokensPerBlock))
            {
                EXPECT_TRUE(hasPage[ord]);
            }
        }
    }

    TokenSpan span(int begin, int end) const
    {
        return TokenSpan{mTokens.data() + begin, end - begin};
    }

    // For each block of the window-aware match: whether it carries a page; and the matched length.
    static std::pair<std::vector<bool>, int> treePages(KvCacheManager& manager, TokenSpan tokens)
    {
        auto const lc = manager.getLayerGroupId(LayerId{0});
        auto const match = manager.radixTree().match({}, tokens, false, true);
        std::vector<bool> hasPage;
        for (auto const* block : match.blocks)
        {
            hasPage.push_back(block->getPage(lc) != nullptr);
        }
        return {hasPage, match.numTokens};
    }

    cudaStream_t mStream{};
    std::vector<TokenIdExt> mTokens;
};

// A fresh cache claims capacity past the absent prefix, then its history jumps over it in one
// update, as on a context engine whose target reused [0, kAbsent). Without the option the
// blocks live at history 0..kAbsent are allocated, never written, and committed into the tree.
TEST_F(AbsentPrefixTest, AbsentPrefixIsNeverAllocatedOrCommitted)
{
    auto manager = makeSwaManager();
    auto const lc = manager->getLayerGroupId(LayerId{0});
    auto const freeBefore = manager->storage().getStatistics().free;

    auto cache = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(cache->resume(mStream));
    cache->setAbsentBefore(kAbsent);
    EXPECT_EQ(cache->absentBefore(), kAbsent);
    ASSERT_TRUE(cache->resize(kLength));
    EXPECT_EQ(freeBefore - manager->storage().getStatistics().free, (kLength - kAbsent) / kTokensPerBlock);
    auto const indices = cache->getBasePageIndices(lc);
    for (int ord = 0; ord < kLength / kTokensPerBlock; ++ord)
    {
        SCOPED_TRACE(ord);
        EXPECT_EQ(indices[ord] == kBadPageIndex.value(), ord < kAbsent / kTokensPerBlock);
    }

    ASSERT_TRUE(cache->resize(std::nullopt, kLength));
    cache->commit(span(0, kLength));
    cache->stopCommitting();

    auto const [hasPage, matched] = treePages(*manager, span(0, kLength));
    EXPECT_EQ(matched, kLength);
    ASSERT_EQ(hasPage.size(), static_cast<size_t>(kLength / kTokensPerBlock));
    for (size_t ord = 0; ord < hasPage.size(); ++ord)
    {
        SCOPED_TRACE(ord);
        EXPECT_EQ(hasPage[ord], ord >= static_cast<size_t>(kAbsent / kTokensPerBlock));
    }
    // A shorter prompt ends where the window would need absent blocks: nothing is reusable.
    EXPECT_EQ(manager->radixTree().match({}, span(0, 24), false, true).numTokens, 0);

    cache->close();
    manager->shutdown();
}

// The same calls without the option: every block is allocated and every block's page reaches
// the tree, so the option is the only thing that changes the result above.
TEST_F(AbsentPrefixTest, UnusedOptionLeavesSlidingWindowUnchanged)
{
    auto manager = makeSwaManager();
    auto const freeBefore = manager->storage().getStatistics().free;

    auto cache = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(cache->resume(mStream));
    EXPECT_EQ(cache->absentBefore(), 0);
    cache->setAbsentBefore(0);
    ASSERT_TRUE(cache->resize(kLength));
    EXPECT_EQ(freeBefore - manager->storage().getStatistics().free, kLength / kTokensPerBlock);
    ASSERT_TRUE(cache->resize(std::nullopt, kLength));
    cache->commit(span(0, kLength));
    cache->stopCommitting();

    auto const [hasPage, matched] = treePages(*manager, span(0, kLength));
    EXPECT_EQ(matched, kLength);
    ASSERT_EQ(hasPage.size(), static_cast<size_t>(kLength / kTokensPerBlock));
    for (size_t ord = 0; ord < hasPage.size(); ++ord)
    {
        SCOPED_TRACE(ord);
        EXPECT_TRUE(hasPage[ord]);
    }
    EXPECT_EQ(manager->radixTree().match({}, span(0, 24), false, true).numTokens, 24);

    cache->close();
    manager->shutdown();
}

// History advances in two chunks with a commit after each. Neither commit gives an absent block a
// page (checked by block index; no KV data is written); after the second, the full prompt is
// reusable through its window.
TEST_F(AbsentPrefixTest, TwoChunkCommitsLeaveEveryAbsentBlockPageless)
{
    auto manager = makeSwaManager();
    auto cache = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(cache->resume(mStream));
    cache->setAbsentBefore(kAbsent);

    ASSERT_TRUE(cache->resize(24));
    ASSERT_TRUE(cache->resize(std::nullopt, 24));
    cache->commit(span(0, 24));
    {
        auto const [hasPage, matched] = treePages(*manager, span(0, 24));
        EXPECT_EQ(matched, 0);
        EXPECT_EQ(hasPage.size(), 0U);
        auto const match = manager->radixTree().match({}, span(0, 24), false, true);
        EXPECT_EQ(match.numReusableTokensBeforePruning, 24);
    }

    ASSERT_TRUE(cache->resize(kLength));
    ASSERT_TRUE(cache->resize(std::nullopt, kLength));
    cache->commit(span(24, kLength));
    cache->stopCommitting();
    auto const [hasPage, matched] = treePages(*manager, span(0, kLength));
    EXPECT_EQ(matched, kLength);
    ASSERT_EQ(hasPage.size(), static_cast<size_t>(kLength / kTokensPerBlock));
    for (size_t ord = 0; ord < hasPage.size(); ++ord)
    {
        SCOPED_TRACE(ord);
        EXPECT_EQ(hasPage[ord], ord >= static_cast<size_t>(kAbsent / kTokensPerBlock));
    }

    // A later sequence on the same prompt reuses the window, not the absent prefix.
    auto reader = manager->createKvCache({}, span(0, kLength), 2);
    EXPECT_EQ(reader->numCommittedTokens(), kLength);
    ASSERT_TRUE(reader->resume(mStream));
    reader->close();
    cache->close();
    manager->shutdown();
}

// A capacity below the absent prefix adds no block (the new-block count would be negative), and
// the pool's accounting stays exact through growth and close.
TEST_F(AbsentPrefixTest, CapacityBelowAbsent)
{
    auto manager = makeSwaManager();
    auto const freeBefore = manager->storage().getStatistics().free;
    {
        auto cache = manager->createKvCache({}, {}, 1);
        ASSERT_TRUE(cache->resume(mStream));
        cache->setAbsentBefore(kAbsent);
        ASSERT_TRUE(cache->resize(8));
        EXPECT_EQ(manager->storage().getStatistics().free, freeBefore);
        ASSERT_TRUE(cache->resize(kLength));
        EXPECT_EQ(freeBefore - manager->storage().getStatistics().free, (kLength - kAbsent) / kTokensPerBlock);
        expectAbsentIndices(*manager, *cache);
        cache->close();
        EXPECT_EQ(manager->storage().getStatistics().free, freeBefore);
    }
    {
        auto cache = manager->createKvCache({}, {}, 2);
        ASSERT_TRUE(cache->resume(mStream));
        cache->setAbsentBefore(kAbsent);
        ASSERT_TRUE(cache->resize(8));
        cache->close();
        EXPECT_EQ(manager->storage().getStatistics().free, freeBefore);
    }
    auto cache = manager->createKvCache({}, {}, 3);
    ASSERT_TRUE(cache->resume(mStream));
    ASSERT_TRUE(cache->resize(kLength));
    EXPECT_EQ(freeBefore - manager->storage().getStatistics().free, kLength / kTokensPerBlock);
    cache->close();
    manager->shutdown();
}

// A commit that ends inside the absent prefix leaves committed and uncommitted absent blocks side
// by side. Run this binary with TLLM_DEBUG_MODE=1 too: the cache's sanity check then runs after
// every resize and commit (the mode is read once, when the library loads).
TEST_F(AbsentPrefixTest, PartialCommitInsideAbsentPrefix)
{
    bool const debugChecks = tensorrt_llm::DebugConfig::isCheckDebugEnabled();
    std::cout << "[ absent ] debug checks " << (debugChecks ? "on" : "off") << std::endl;
    RecordProperty("debug_checks", debugChecks ? "on" : "off");
    auto manager = makeSwaManager();
    auto cache = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(cache->resume(mStream));
    cache->setAbsentBefore(kAbsent);
    ASSERT_TRUE(cache->resize(kLength));
    cache->commit(span(0, kTokensPerBlock));
    EXPECT_EQ(cache->numCommittedTokens(), kTokensPerBlock);
    EXPECT_EQ(cache->historyLength(), kTokensPerBlock);
    expectAbsentIndices(*manager, *cache);
    cache->commit(span(kTokensPerBlock, kAbsent + 2));
    cache->commit(span(kAbsent + 2, kLength));
    cache->stopCommitting();
    expectAbsentTree(*manager);
    cache->close();
    manager->shutdown();
}

// commit_min_snapshot (the manager config when block reuse is not ALL_REUSABLE): stale pages are
// released rather than held for commit, and the absent prefix stays page-less either way.
TEST_F(AbsentPrefixTest, CommitMinSnapshot)
{
    auto manager = makeSwaManager(kWindow, [](KVCacheManagerConfig& config) { config.commitMinSnapshot = true; });
    auto cache = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(cache->resume(mStream));
    cache->setAbsentBefore(kAbsent);
    ASSERT_TRUE(cache->resize(kLength));
    expectAbsentIndices(*manager, *cache);
    ASSERT_TRUE(cache->resize(std::nullopt, kLength));
    cache->commit(span(0, kLength));
    cache->stopCommitting();
    expectAbsentTree(*manager, /*windowOnly=*/true);
    cache->close();
    manager->shutdown();
}

TEST_F(AbsentPrefixTest, SuspendAndResumeKeepThePrefixAbsent)
{
    auto manager = makeSwaManager();
    auto const freeBefore = manager->storage().getStatistics().free;
    auto cache = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(cache->resume(mStream));
    cache->setAbsentBefore(kAbsent);
    ASSERT_TRUE(cache->resize(kLength));
    cache->suspend();
    ASSERT_TRUE(cache->resume(mStream));
    EXPECT_EQ(cache->absentBefore(), kAbsent);
    EXPECT_EQ(freeBefore - manager->storage().getStatistics().free, (kLength - kAbsent) / kTokensPerBlock);
    expectAbsentIndices(*manager, *cache);
    ASSERT_TRUE(cache->resize(std::nullopt, kLength));
    cache->commit(span(0, kLength));
    cache->suspend();
    ASSERT_TRUE(cache->resume(mStream));
    cache->stopCommitting();
    expectAbsentTree(*manager);
    cache->close();
    manager->shutdown();
}

// A request without the option computes [0, kAbsent) and commits it; an absent cache then commits
// the whole prompt on top. The tree path carries the first request's pages before kAbsent and the
// absent cache's after it, so a match ending anywhere in the prompt finds a whole window.
TEST_F(AbsentPrefixTest, AbsentCacheCommitsOnTopOfAWrittenPrefix)
{
    auto manager = makeSwaManager();
    {
        auto first = manager->createKvCache({}, {}, 1);
        ASSERT_TRUE(first->resume(mStream));
        ASSERT_TRUE(first->resize(kAbsent));
        first->commit(span(0, kAbsent));
        first->stopCommitting();
        first->close();
    }
    auto cache = manager->createKvCache({}, {}, 2);
    ASSERT_TRUE(cache->resume(mStream));
    cache->setAbsentBefore(kAbsent);
    ASSERT_TRUE(cache->resize(kLength));
    expectAbsentIndices(*manager, *cache);
    ASSERT_TRUE(cache->resize(std::nullopt, kLength));
    cache->commit(span(0, kLength));
    cache->stopCommitting();

    auto const [hasPage, matched] = treePages(*manager, span(0, kLength));
    EXPECT_EQ(matched, kLength);
    ASSERT_EQ(hasPage.size(), static_cast<size_t>(kLength / kTokensPerBlock));
    for (size_t ord = 0; ord < hasPage.size(); ++ord)
    {
        SCOPED_TRACE(ord);
        EXPECT_TRUE(hasPage[ord]);
    }
    // The window at 24 spans the first request's block 3 and the absent cache's blocks 4-5.
    EXPECT_EQ(manager->radixTree().match({}, span(0, 24), false, true).numTokens, 24);
    cache->close();
    manager->shutdown();
}

TEST_F(AbsentPrefixTest, RejectsInvalidUse)
{
    {
        auto manager = makeSwaManager();
        auto cache = manager->createKvCache({}, {}, 1);
        ASSERT_TRUE(cache->resume(mStream));
        EXPECT_THROW(cache->setAbsentBefore(kTokensPerBlock + 1), std::invalid_argument);
        EXPECT_THROW(cache->setAbsentBefore(-kTokensPerBlock), std::invalid_argument);
        ASSERT_TRUE(cache->resize(8));
        EXPECT_THROW(cache->setAbsentBefore(kAbsent), std::invalid_argument);
        EXPECT_EQ(cache->absentBefore(), 0);
        cache->close();
        EXPECT_ANY_THROW(cache->setAbsentBefore(kAbsent));
        manager->shutdown();
    }
    {
        // Once set: no beam width change and no SWA scratch reuse.
        auto manager = makeSwaManager(
            kWindow, [](KVCacheManagerConfig& config) { config.swaScratchReuse = SwaScratchReuseConfig{}; });
        auto cache = manager->createKvCache({}, {}, 1);
        ASSERT_TRUE(cache->resume(mStream));
        // A scratch-enabled cache is rejected until scratch reuse is turned off.
        ASSERT_TRUE(cache->isSwaScratchReuseEnabled());
        EXPECT_THROW(cache->setAbsentBefore(kAbsent), std::invalid_argument);
        cache->setEnableSwaScratchReuse(false);
        cache->setAbsentBefore(kAbsent);
        EXPECT_THROW(cache->setEnableSwaScratchReuse(true), std::invalid_argument);
        EXPECT_THROW(cache->setBeamWidth(BeamIndex{2}), std::invalid_argument);
        EXPECT_EQ(cache->absentBefore(), kAbsent);
        cache->close();
        manager->shutdown();
    }
    // Life cycles that cannot have an absent prefix: full attention, sinks, OPTIONAL / PRIVATE
    // reuse beside a REQUIRED window, sparse, and recurrent state.
    std::vector<std::pair<char const*, std::function<KVCacheManagerConfig()>>> const rejected{
        {"full_attention",
            []
            {
                auto config = makeConfig();
                std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = std::nullopt;
                return config;
            }},
        {"sinks",
            []
            {
                auto config = makeConfig();
                auto& layer = std::get<AttentionLayerConfig>(config.layers.front());
                layer.slidingWindowSize = kWindow;
                layer.numSinkTokens = kTokensPerBlock;
                return config;
            }},
        {"optional",
            []
            {
                auto config = makeConfig();
                std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = kWindow;
                auto layer = std::get<AttentionLayerConfig>(config.layers.front());
                layer.layerId = 1;
                layer.reusePolicy = AttentionReusePolicy::OPTIONAL;
                config.layers.emplace_back(layer);
                return config;
            }},
        {"private",
            []
            {
                auto config = makeConfig();
                std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = kWindow;
                auto layer = std::get<AttentionLayerConfig>(config.layers.front());
                layer.layerId = 1;
                layer.reusePolicy = AttentionReusePolicy::PRIVATE;
                config.layers.emplace_back(layer);
                return config;
            }},
        {"sparse",
            []
            {
                auto config = makeTieredConfig();
                std::get<AttentionLayerConfig>(config.layers.front()).buffers.front().isSparse = true;
                return config;
            }},
        {"ssm", [] { return makeHybridTieredConfig(); }},
    };
    for (auto const& [name, make] : rejected)
    {
        SCOPED_TRACE(name);
        auto manager = std::make_shared<KvCacheManager>(make());
        auto cache = manager->createKvCache({}, {}, 1);
        ASSERT_TRUE(cache->resume(mStream));
        EXPECT_THROW(cache->setAbsentBefore(kAbsent), std::invalid_argument);
        EXPECT_EQ(cache->absentBefore(), 0);
        cache->close();
        manager->shutdown();
    }
}

} // namespace

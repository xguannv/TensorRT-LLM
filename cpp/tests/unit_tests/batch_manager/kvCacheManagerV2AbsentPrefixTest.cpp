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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;

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

    static std::shared_ptr<KvCacheManager> makeSwaManager(std::optional<int> window = kWindow)
    {
        auto config = makeConfig();
        EXPECT_EQ(config.tokensPerBlock, kTokensPerBlock);
        std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = window;
        return std::make_shared<KvCacheManager>(config);
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

// History advances in two chunks with a commit after each. Each commit publishes the written
// blocks only; after the second, the full prompt is reusable through its window.
TEST_F(AbsentPrefixTest, CommitAfterPartialWritesPublishesOnlyWrittenPages)
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
        manager->shutdown();
    }
    {
        // Full attention needs every block, so it cannot have an absent prefix.
        auto manager = makeSwaManager(std::nullopt);
        auto cache = manager->createKvCache({}, {}, 1);
        ASSERT_TRUE(cache->resume(mStream));
        EXPECT_THROW(cache->setAbsentBefore(kAbsent), std::invalid_argument);
        cache->close();
        manager->shutdown();
    }
}

} // namespace

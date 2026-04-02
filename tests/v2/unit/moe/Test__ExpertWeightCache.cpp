/**
 * @file Test__ExpertWeightCache.cpp
 * @brief Unit tests for ExpertWeightCache with mock IExpertWeightStorage
 */

#include <gtest/gtest.h>
#include "execution/moe/ExpertWeightCache.h"
#include <unordered_set>
#include <vector>

using namespace llaminar2;

// ============================================================================
// Mock weight storage
// ============================================================================

class MockExpertWeightStorage : public IExpertWeightStorage
{
public:
    explicit MockExpertWeightStorage(size_t bytes_per_expert = 1024)
        : bytes_per_expert_(bytes_per_expert) {}

    ExpertWeightHandle load(const ExpertWeightKey &key) override
    {
        load_calls_.push_back(key);
        // Return fake pointers (just need non-null for testing)
        auto ptr = reinterpret_cast<float *>(static_cast<uintptr_t>(
            key.layer_idx * 1000 + key.expert_id + 1));
        return ExpertWeightHandle{
            .key = key,
            .gate_w = ptr,
            .up_w = ptr + 1,
            .down_w = ptr + 2,
            .total_bytes = bytes_per_expert_,
        };
    }

    void release(const ExpertWeightHandle &handle) override
    {
        release_calls_.push_back(handle.key);
    }

    size_t bytesPerExpert() const override
    {
        return bytes_per_expert_;
    }

    const std::vector<ExpertWeightKey> &loadCalls() const { return load_calls_; }
    const std::vector<ExpertWeightKey> &releaseCalls() const { return release_calls_; }

    void reset()
    {
        load_calls_.clear();
        release_calls_.clear();
    }

private:
    size_t bytes_per_expert_;
    std::vector<ExpertWeightKey> load_calls_;
    std::vector<ExpertWeightKey> release_calls_;
};

// ============================================================================
// Tests
// ============================================================================

TEST(Test__ExpertWeightCache, BasicGetAndCache)
{
    auto storage = std::make_shared<MockExpertWeightStorage>(1024);
    ExpertWeightCache cache(storage, /*max_bytes=*/8192, EvictionPolicy::LRU);

    ExpertWeightKey key{0, 5};
    auto handle = cache.get(key);

    EXPECT_NE(handle.gate_w, nullptr);
    EXPECT_NE(handle.up_w, nullptr);
    EXPECT_NE(handle.down_w, nullptr);
    EXPECT_EQ(storage->loadCalls().size(), 1u);

    // Second get should be a cache hit
    auto handle2 = cache.get(key);
    EXPECT_EQ(storage->loadCalls().size(), 1u); // No new load
    EXPECT_EQ(handle2.gate_w, handle.gate_w);   // Same pointer
}

TEST(Test__ExpertWeightCache, CacheStats)
{
    auto storage = std::make_shared<MockExpertWeightStorage>(1024);
    ExpertWeightCache cache(storage, 8192, EvictionPolicy::LRU);

    cache.get({0, 0}); // miss
    cache.get({0, 0}); // hit
    cache.get({0, 1}); // miss
    cache.get({0, 0}); // hit

    auto stats = cache.stats();
    EXPECT_EQ(stats.hits, 2u);
    EXPECT_EQ(stats.misses, 2u);
    EXPECT_EQ(stats.current_entries, 2u);
    EXPECT_EQ(stats.current_bytes, 2048u);
    EXPECT_NEAR(stats.hitRate(), 0.5f, 1e-6f);
}

TEST(Test__ExpertWeightCache, LRU_Eviction)
{
    auto storage = std::make_shared<MockExpertWeightStorage>(1024);
    // Only room for 2 entries
    ExpertWeightCache cache(storage, 2048, EvictionPolicy::LRU);

    cache.get({0, 0}); // miss → load
    cache.get({0, 1}); // miss → load
    cache.get({0, 0}); // hit → {0,0} is most recently used

    // This should evict {0,1} (least recently used)
    cache.get({0, 2}); // miss → eviction

    EXPECT_TRUE(cache.isCached({0, 0}));
    EXPECT_FALSE(cache.isCached({0, 1})); // evicted
    EXPECT_TRUE(cache.isCached({0, 2}));

    auto stats = cache.stats();
    EXPECT_EQ(stats.evictions, 1u);
    EXPECT_EQ(stats.current_entries, 2u);
}

TEST(Test__ExpertWeightCache, FIFO_Eviction)
{
    auto storage = std::make_shared<MockExpertWeightStorage>(1024);
    ExpertWeightCache cache(storage, 2048, EvictionPolicy::FIFO);

    cache.get({0, 0}); // miss → oldest
    cache.get({0, 1}); // miss
    cache.get({0, 0}); // hit (doesn't change FIFO order)

    // Evict oldest → {0,0}
    cache.get({0, 2}); // miss → eviction

    EXPECT_FALSE(cache.isCached({0, 0})); // evicted (first in)
    EXPECT_TRUE(cache.isCached({0, 1}));
    EXPECT_TRUE(cache.isCached({0, 2}));
}

TEST(Test__ExpertWeightCache, Release_CallsStorage)
{
    auto storage = std::make_shared<MockExpertWeightStorage>(1024);
    ExpertWeightCache cache(storage, 4096, EvictionPolicy::LRU);

    cache.get({0, 0});
    cache.get({0, 1});
    cache.evict({0, 0});

    EXPECT_FALSE(cache.isCached({0, 0}));
    EXPECT_EQ(storage->releaseCalls().size(), 1u);
    EXPECT_EQ(storage->releaseCalls()[0].expert_id, 0);
}

TEST(Test__ExpertWeightCache, Clear)
{
    auto storage = std::make_shared<MockExpertWeightStorage>(1024);
    ExpertWeightCache cache(storage, 8192, EvictionPolicy::LRU);

    cache.get({0, 0});
    cache.get({0, 1});
    cache.get({0, 2});
    cache.clear();

    auto stats = cache.stats();
    EXPECT_EQ(stats.current_entries, 0u);
    EXPECT_EQ(stats.current_bytes, 0u);
    EXPECT_EQ(storage->releaseCalls().size(), 3u);
}

TEST(Test__ExpertWeightCache, Prefetch)
{
    auto storage = std::make_shared<MockExpertWeightStorage>(1024);
    ExpertWeightCache cache(storage, 8192, EvictionPolicy::LRU);

    cache.prefetch({{0, 5}});
    EXPECT_TRUE(cache.isCached({0, 5}));

    // Subsequent get should be a hit
    cache.get({0, 5});
    auto stats = cache.stats();
    EXPECT_EQ(stats.hits, 1u);
    EXPECT_EQ(stats.misses, 1u); // prefetch counts as miss
}

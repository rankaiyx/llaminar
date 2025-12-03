/**
 * @file Test__KernelFactoryCacheInvalidation.cpp
 * @brief Unit tests for automatic KernelFactory cache invalidation
 * @author David Sanftenberg
 *
 * Tests verify that when a tensor is destroyed, its entry in the
 * KernelFactory cache is automatically removed. This prevents use-after-free
 * bugs when a new tensor is allocated at the same memory address as a
 * previously destroyed tensor.
 *
 * The fix works by having TensorBase's destructor call
 * KernelFactory::clearCacheFor(this).
 */

#include <gtest/gtest.h>
#include "../../src/v2/kernels/KernelFactory.h"
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/backends/ComputeBackend.h"
#include <memory>

using namespace llaminar::v2::kernels;
using namespace llaminar2;

class Test__KernelFactoryCacheInvalidation : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure DeviceManager is initialized
        DeviceManager::instance();
        // Start with empty cache
        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        // Clean up after tests
        KernelFactory::clearCache();
    }
};

// Helper to create a minimal IQ4_NL tensor (smallest quantized format)
static std::unique_ptr<IQ4_NLTensor> createTestTensor()
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18; // IQ4_NL block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    return std::make_unique<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw_data);
}

// ============================================================================
// Basic Cache Invalidation Tests
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, CacheEmptyAfterClear)
{
    auto [cache_size, packed_bytes] = KernelFactory::cacheStats();
    EXPECT_EQ(cache_size, 0u);
}

TEST_F(Test__KernelFactoryCacheInvalidation, CacheGrowsAfterGetOrCreateGemm)
{
    auto tensor = createTestTensor();

    // Verify cache starts empty
    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_before, 0u);

    // Create a GEMM kernel via factory (should cache it)
    auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
    ASSERT_NE(kernel, nullptr);

    // Verify cache now has one entry
    auto [size_after, bytes] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 1u);
    // Note: bytes may be 0 for some kernel types that don't track packed data size
}

TEST_F(Test__KernelFactoryCacheInvalidation, CacheAutoInvalidatesOnTensorDestruction)
{
    // This is the KEY test for the automatic invalidation feature

    // Create tensor in a scope
    {
        auto tensor = createTestTensor();

        // Add to cache
        auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
        ASSERT_NE(kernel, nullptr);

        // Verify it's in the cache
        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size_during, 1u);

        // tensor goes out of scope here -> destructor runs
    }

    // After tensor destruction, cache entry should be automatically removed
    auto [size_after, bytes_after] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be empty after tensor destruction";
    EXPECT_EQ(bytes_after, 0u) << "Packed bytes should be zero after tensor destruction";
}

TEST_F(Test__KernelFactoryCacheInvalidation, MultipleTensors_IndependentInvalidation)
{
    // Create two tensors
    auto tensor1 = createTestTensor();
    auto tensor2 = createTestTensor();

    // Add both to cache
    KernelFactory::getOrCreateGemm(tensor1.get());
    KernelFactory::getOrCreateGemm(tensor2.get());

    // Should have 2 entries
    auto [size_both, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_both, 2u);

    // Destroy tensor1
    tensor1.reset();

    // Should have 1 entry remaining (tensor2)
    auto [size_one, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_one, 1u) << "Only tensor2's kernel should remain in cache";

    // Destroy tensor2
    tensor2.reset();

    // Cache should now be empty
    auto [size_none, ___] = KernelFactory::cacheStats();
    EXPECT_EQ(size_none, 0u) << "Cache should be empty after all tensors destroyed";
}

TEST_F(Test__KernelFactoryCacheInvalidation, ClearCacheForNonExistentTensor_NoOp)
{
    // Create and cache a tensor
    auto tensor = createTestTensor();
    KernelFactory::getOrCreateGemm(tensor.get());

    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_before, 1u);

    // Create a different tensor (not cached) and clear its entry
    // This should be a no-op - the cache should still have the original entry
    auto tensor2 = createTestTensor();
    KernelFactory::clearCacheFor(tensor2.get());

    // Cache should still have the first tensor's entry
    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 1u) << "Cache should be unaffected by clearCacheFor with different tensor";
}

TEST_F(Test__KernelFactoryCacheInvalidation, CacheHit_SamePointerReturnssamKernel)
{
    auto tensor = createTestTensor();

    // First call creates and caches
    auto *kernel1 = KernelFactory::getOrCreateGemm(tensor.get());
    ASSERT_NE(kernel1, nullptr);

    // Second call should return the SAME kernel (cache hit)
    auto *kernel2 = KernelFactory::getOrCreateGemm(tensor.get());
    EXPECT_EQ(kernel1, kernel2) << "Cache should return the same kernel pointer";

    // Should still have only 1 cache entry
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 1u);
}

// ============================================================================
// Regression Test: Memory Reuse Scenario
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, MemoryReuse_NoStaleKernel)
{
    // This test simulates the exact bug that was occurring:
    // 1. Create tensor A, cache its kernel
    // 2. Destroy tensor A
    // 3. Create tensor B (may get same address as A)
    // 4. Query cache for tensor B -> should NOT get A's stale kernel

    // We can't guarantee memory reuse, but we can verify that after
    // destruction, no stale entries exist

    const TensorBase *captured_ptr = nullptr;

    // Create and cache tensor
    {
        auto tensor = createTestTensor();
        captured_ptr = tensor.get();

        KernelFactory::getOrCreateGemm(tensor.get());

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size_during, 1u);
    }

    // After destruction, even if we query with the old pointer address,
    // we should NOT find any entry (cache should be clean)
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be empty after tensor destruction";

    // Note: We can't call getOrCreateGemm(captured_ptr) because that would be UB.
    // The important thing is that the cache is clean, so if a new tensor
    // happens to be allocated at the same address, it won't hit a stale entry.
}

// ============================================================================
// FP32 Tensor (non-quantized) Test
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, FP32Tensor_AutoInvalidation)
{
    // FP32 tensors also go through the cache
    {
        const size_t rows = 32;
        const size_t cols = 32;
        // FP32Tensor constructor takes shape and optional device_idx (defaults to -1 for CPU)
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});

        auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
        ASSERT_NE(kernel, nullptr);

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size_during, 1u);
    }

    // After destruction, cache should be empty
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, TensorWithoutKernel_DestructorSafe)
{
    // Creating and destroying a tensor that was NEVER added to cache
    // should not cause any issues
    {
        auto tensor = createTestTensor();
        // Don't call getOrCreateGemm - just destroy it
    }

    // Should be a no-op, no crash
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 0u);
}

TEST_F(Test__KernelFactoryCacheInvalidation, RapidCreateDestroy_CacheStaysClean)
{
    // Rapidly create and destroy many tensors
    for (int i = 0; i < 100; ++i)
    {
        auto tensor = createTestTensor();
        KernelFactory::getOrCreateGemm(tensor.get());
    }

    // After all destroys, cache should be empty
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 0u) << "Cache should be clean after rapid create/destroy cycles";
}

/**
 * @file Test__PreparedEmbeddingWeightsLifecycle.cpp
 * @brief Unit tests for PreparedEmbeddingWeights registry lifecycle
 *
 * Tests verify:
 * 1. Registry starts empty and can be cleared
 * 2. FP32 tensors skip preparation (no IINT8Unpackable)
 * 3. Q8_0 tensors get proper EmbedQ8 repack + handle metadata
 * 4. Cache hits return same handle pointer
 * 5. Different devices get separate entries
 * 6. Null tensor input throws
 * 7. Lookup-only (getPreparedEmbeddingWeights) does not create entries
 * 8. Move semantics are safe (no double-free)
 * 9. Multiple quantized formats (Q4_0, Q8_0) all work
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "kernels/common/PreparedEmbeddingWeights.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "backends/BackendManager.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__PreparedEmbeddingWeightsLifecycle : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);

        // Ensure CPU backend is available for host-side allocations
        if (!getCPUBackend())
            initCPUBackend(0);

        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }

    // Small embedding dimensions for fast tests
    static constexpr size_t kVocabSize = 128;
    static constexpr size_t kDModel = 64;
    static constexpr int kDModelInt = 64;
};

// ============================================================================
// Registry State Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, RegistryStartsEmpty)
{
    EXPECT_EQ(KernelFactory::preparedEmbeddingRegistrySize(), 0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, ClearAllCachesReleasesEntries)
{
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());
    ASSERT_NE(handle, nullptr);
    EXPECT_GT(KernelFactory::preparedEmbeddingRegistrySize(), 0u);

    KernelFactory::clearCache();

    EXPECT_EQ(KernelFactory::preparedEmbeddingRegistrySize(), 0u);
    EXPECT_EQ(KernelFactory::getPreparedEmbeddingWeights(tensor.get(), DeviceId::cpu()), nullptr);
}

// ============================================================================
// FP32 Skip Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FP32TensorSkipsPreparation)
{
    auto tensor = TestTensorFactory::createFP32Random({kVocabSize, kDModel});
    auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(KernelFactory::preparedEmbeddingRegistrySize(), 0u);
}

// ============================================================================
// Quantized Tensor Preparation Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, Q8_0TensorGetsPreparation)
{
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());

    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle->tensor, tensor.get());
    EXPECT_EQ(handle->device_id, DeviceId::cpu());
    ASSERT_NE(handle->weights, nullptr);
    EXPECT_NE(handle->weights->device_data, nullptr);
    EXPECT_GT(handle->weights->byte_size, 0u);
    EXPECT_EQ(handle->weights->vocab_size, kVocabSize);
    EXPECT_EQ(handle->weights->d_model, kDModelInt);
    EXPECT_GT(handle->weights->blocks_per_row, 0u);
    EXPECT_EQ(KernelFactory::preparedEmbeddingRegistrySize(), 1u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, Q4_0TensorGetsPreparation)
{
    auto tensor = TestTensorFactory::createQ4_0Random({kVocabSize, kDModel});
    auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());

    ASSERT_NE(handle, nullptr);
    EXPECT_NE(handle->weights->device_data, nullptr);
    EXPECT_EQ(handle->weights->vocab_size, kVocabSize);
    EXPECT_EQ(handle->weights->d_model, kDModelInt);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, HandleMetadataCorrect)
{
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());

    ASSERT_NE(handle, nullptr);

    // blocks_per_row should be ceil(kDModel / 32) for Q8_0 → EmbedQ8
    const size_t expected_blocks_per_row = (kDModel + 31) / 32;
    EXPECT_EQ(handle->weights->blocks_per_row, expected_blocks_per_row);

    // byte_size should be vocab_size * blocks_per_row * sizeof(EmbedQ8Block)
    // EmbedQ8Block = uint16_t(d) + uint16_t(m) + int8_t[32] = 36 bytes
    const size_t expected_byte_size = kVocabSize * expected_blocks_per_row * 36;
    EXPECT_EQ(handle->weights->byte_size, expected_byte_size);
}

// ============================================================================
// Cache Hit Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, CacheHitReturnsSameHandle)
{
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});

    auto *handle1 = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());
    auto *handle2 = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());

    ASSERT_NE(handle1, nullptr);
    EXPECT_EQ(handle1, handle2);
    EXPECT_EQ(KernelFactory::preparedEmbeddingRegistrySize(), 1u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, DifferentTensorsGetSeparateEntries)
{
    auto tensor_a = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel}, 42);
    auto tensor_b = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel}, 99);

    auto *handle_a = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor_a.get(), kDModelInt, DeviceId::cpu());
    auto *handle_b = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor_b.get(), kDModelInt, DeviceId::cpu());

    ASSERT_NE(handle_a, nullptr);
    ASSERT_NE(handle_b, nullptr);
    EXPECT_NE(handle_a, handle_b);
    EXPECT_EQ(KernelFactory::preparedEmbeddingRegistrySize(), 2u);
}

// ============================================================================
// Lookup-Only Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, LookupOnlyDoesNotCreate)
{
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});

    auto *result = KernelFactory::getPreparedEmbeddingWeights(
        tensor.get(), DeviceId::cpu());
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(KernelFactory::preparedEmbeddingRegistrySize(), 0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, LookupFindsCreatedEntry)
{
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});

    KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());

    auto *found = KernelFactory::getPreparedEmbeddingWeights(
        tensor.get(), DeviceId::cpu());
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->tensor, tensor.get());
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, LookupMissesWrongDevice)
{
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});

    KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModelInt, DeviceId::cpu());

    auto *found = KernelFactory::getPreparedEmbeddingWeights(
        tensor.get(), DeviceId(DeviceType::CPU, 1));
    EXPECT_EQ(found, nullptr);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, NullTensorThrows)
{
    EXPECT_THROW(
        KernelFactory::getOrCreatePreparedEmbeddingWeights(
            nullptr, kDModelInt, DeviceId::cpu()),
        std::runtime_error);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, MoveConstructorTransfersOwnership)
{
    PreparedEmbeddingWeights src;
    // Simulate an allocation using CPU backend (malloc)
    src.device_data = malloc(256);
    src.byte_size = 256;
    src.blocks_per_row = 2;
    src.vocab_size = 4;
    src.d_model = 32;
    src.device_id = DeviceId::cpu();

    void *original_ptr = src.device_data;

    PreparedEmbeddingWeights dst(std::move(src));

    EXPECT_EQ(dst.device_data, original_ptr);
    EXPECT_EQ(dst.byte_size, 256u);
    EXPECT_EQ(dst.blocks_per_row, 2u);
    EXPECT_EQ(dst.vocab_size, 4u);
    EXPECT_EQ(dst.d_model, 32);

    // Source must be nullified
    EXPECT_EQ(src.device_data, nullptr);
    EXPECT_EQ(src.byte_size, 0u);

    // dst destructor will call backend->free() which is free() for CPU
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, MoveAssignmentTransfersOwnership)
{
    PreparedEmbeddingWeights src;
    src.device_data = malloc(128);
    src.byte_size = 128;
    src.device_id = DeviceId::cpu();

    void *original_ptr = src.device_data;

    PreparedEmbeddingWeights dst;
    dst = std::move(src);

    EXPECT_EQ(dst.device_data, original_ptr);
    EXPECT_EQ(dst.byte_size, 128u);
    EXPECT_EQ(src.device_data, nullptr);
    EXPECT_EQ(src.byte_size, 0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, DestructorSafeOnNullData)
{
    // Should not crash — device_data is nullptr by default
    {
        PreparedEmbeddingWeights w;
    }
    SUCCEED();
}

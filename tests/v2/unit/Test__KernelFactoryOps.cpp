/**
 * @file Test__KernelFactoryOps.cpp
 * @brief Unit tests for KernelFactory create methods for ops kernels
 * @author David Sanftenberg
 *
 * Tests cover:
 * 1. createRMSNorm() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 2. createSwiGLU() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 3. createRoPE() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 4. createSoftmax() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 5. createAttention() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 6. createEmbedding() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 7. createResidualAdd() for FP32, BF16, FP16 tensors (CPU path)
 * 8. Generic TensorBase* dispatch for all ops
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__KernelFactoryOps : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure DeviceManager is initialized
        DeviceManager::instance();
    }
};

// ============================================================================
// createRMSNorm() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_GenericDispatch_BF16)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createSwiGLU() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_Q8_1)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createRoPE() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateRoPE_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRoPE(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createSoftmax() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateSoftmax_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSoftmax_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSoftmax_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSoftmax_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

// ============================================================================
// createAttention() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateAttention_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createAttention(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createEmbedding() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateEmbedding_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

// ============================================================================
// createResidualAdd() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_GenericDispatch_BF16)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_GenericDispatch_FP16)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createRMSNorm(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createSwiGLU(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createRoPE(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateAttention_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createAttention(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createResidualAdd(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

// ============================================================================
// Unsupported Tensor Type Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_Q8_1_UnsupportedType_Throws)
{
    // Q8_1 is NOT supported for ResidualAdd (only FP32, BF16, FP16)
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    EXPECT_THROW(
        KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU),
        std::runtime_error);
}

// ============================================================================
// Functional Tests - Verify kernels actually work
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP32_FunctionalTest)
{
    // Create test tensors
    auto input = TestTensorFactory::createFP32({4, 4});
    auto residual = TestTensorFactory::createFP32({4, 4});
    auto output = TestTensorFactory::createFP32({4, 4});

    // Initialize with known values
    float *input_data = input->mutable_data();
    float *residual_data = residual->mutable_data();
    for (size_t i = 0; i < 16; ++i)
    {
        input_data[i] = static_cast<float>(i);
        residual_data[i] = static_cast<float>(i * 2);
    }

    // Create and apply kernel
    auto kernel = KernelFactory::createResidualAdd(input.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    bool success = kernel->apply(
        input->data(),
        residual->data(),
        output->mutable_data(),
        16,
        nullptr,
        -1);
    ASSERT_TRUE(success);

    // Verify output = input + residual
    const float *output_data = output->data();
    for (size_t i = 0; i < 16; ++i)
    {
        float expected = static_cast<float>(i) + static_cast<float>(i * 2);
        EXPECT_FLOAT_EQ(output_data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP32_apply_tensor_Test)
{
    // Create test tensors
    auto input = TestTensorFactory::createFP32Random({8, 16});
    auto residual = TestTensorFactory::createFP32Random({8, 16});
    auto output = TestTensorFactory::createFP32({8, 16});

    // Create kernel
    auto kernel = KernelFactory::createResidualAdd(input.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    bool success = kernel->apply_tensor(
        input.get(),
        residual.get(),
        output.get(),
        8 * 16,
        nullptr,
        -1);
    ASSERT_TRUE(success);

    // Verify output = input + residual
    const float *in_data = input->data();
    const float *res_data = residual->data();
    const float *out_data = output->data();
    for (size_t i = 0; i < 128; ++i)
    {
        EXPECT_FLOAT_EQ(out_data[i], in_data[i] + res_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP32_FunctionalTest)
{
    // Create test tensors with RMSNorm-compatible dimensions
    const int rows = 4;
    const int hidden_dim = 64;
    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(rows), static_cast<size_t>(hidden_dim)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(hidden_dim)});
    auto gamma = TestTensorFactory::createFP32({static_cast<size_t>(hidden_dim)});

    // Initialize gamma to 1.0 for simple test
    float *gamma_data = gamma->mutable_data();
    for (int i = 0; i < hidden_dim; ++i)
    {
        gamma_data[i] = 1.0f;
    }

    // Create kernel
    auto kernel = KernelFactory::createRMSNorm(input.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    float eps = 1e-6f;
    bool success = kernel->apply(
        input->data(),
        gamma->data(),
        output->mutable_data(),
        rows,
        hidden_dim,
        eps,
        false,   // use_bf16
        nullptr, // mpi_ctx
        -1);     // device_idx
    ASSERT_TRUE(success);

    // Verify output is not all zeros (basic sanity check)
    const float *out_data = output->data();
    float sum = 0.0f;
    for (int i = 0; i < rows * hidden_dim; ++i)
    {
        sum += std::abs(out_data[i]);
    }
    EXPECT_GT(sum, 0.0f) << "RMSNorm output should not be all zeros";
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP32_FunctionalTest)
{
    // Create test tensors
    const int rows = 4;
    const int cols = 64;
    auto gate = TestTensorFactory::createFP32Random({static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto up = TestTensorFactory::createFP32Random({static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Create kernel
    auto kernel = KernelFactory::createSwiGLU(gate.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    bool success = kernel->apply(
        gate->data(),
        up->data(),
        output->mutable_data(),
        rows,
        cols,
        false, // add_residual
        nullptr,
        -1);
    ASSERT_TRUE(success);

    // Verify output is not all zeros (basic sanity check)
    const float *out_data = output->data();
    float sum = 0.0f;
    for (int i = 0; i < rows * cols; ++i)
    {
        sum += std::abs(out_data[i]);
    }
    EXPECT_GT(sum, 0.0f) << "SwiGLU output should not be all zeros";
}

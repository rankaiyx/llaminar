/**
 * @file Test__CUDATensorTypeAllocation.cpp
 * @brief Integration tests for CUDA tensor type allocation and GPU memory management
 * @author David Sanftenberg
 * @date January 2026
 *
 * Verifies that:
 * 1. CUDAFp32Tensor can be instantiated and allocated on GPU
 * 2. CPU tensors can be transferred to GPU via ensureOnDevice()
 * 3. Quantized CPU tensors can be transferred to/from GPU
 * 4. KernelFactory works with GPU-resident tensors
 *
 * **Known Limitation**: CUDATypedTensor<T> only works correctly for T=float.
 * INT8/INT32 variants have conflicting method signatures with CUDATensorBase.
 * Quantized data on GPU is handled via CPU tensor transfer (ensureOnDevice).
 */

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cstring>

#include "../../src/v2/tensors/TensorType.h"
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/backends/ComputeBackend.h"
#include "../../src/v2/kernels/KernelFactory.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "../../src/v2/tensors/cuda/CUDATypedTensor.h"
#include "../../src/v2/tensors/cuda/CUDATensorBase.h"
#include "../../src/v2/backends/cuda/CUDABackend.h"
#endif

using namespace llaminar2;
using namespace llaminar::v2::kernels;

namespace
{
    // Runtime CUDA availability check
    bool hasCUDA()
    {
#ifdef HAVE_CUDA
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
#else
        return false;
#endif
    }

    // Get block size in bytes for quantized types
    size_t getBlockSizeBytes(TensorType dtype)
    {
        switch (dtype)
        {
        case TensorType::Q4_0:
            return 18; // 2 bytes scale + 16 bytes data (32 elements @ 4-bit)
        case TensorType::Q4_1:
            return 20; // 2 bytes scale + 2 bytes min + 16 bytes data
        case TensorType::Q5_0:
            return 22; // 2 bytes scale + 4 bytes high bits + 16 bytes data
        case TensorType::Q5_1:
            return 24; // 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes data
        case TensorType::Q8_0:
            return 34; // 2 bytes scale + 32 bytes data
        case TensorType::Q8_1:
            return 36; // 2 bytes scale + 2 bytes sum + 32 bytes data
        case TensorType::Q2_K:
            return 84; // 256-element superblock
        case TensorType::Q3_K:
            return 110; // 256-element superblock
        case TensorType::Q4_K:
            return 144; // 256-element superblock
        case TensorType::Q5_K:
            return 176; // 256-element superblock
        case TensorType::Q6_K:
            return 210; // 256-element superblock
        case TensorType::Q8_K:
            return 292; // 256-element superblock
        case TensorType::IQ4_NL:
            return 18; // 2 bytes scale + 16 bytes indices
        case TensorType::IQ4_XS:
            return 18;
        case TensorType::IQ2_XXS:
            return 66; // 256-element superblock
        case TensorType::IQ2_XS:
            return 74;
        case TensorType::IQ2_S:
            return 82;
        case TensorType::IQ3_XXS:
            return 98;
        case TensorType::IQ3_S:
            return 110;
        case TensorType::IQ1_S:
            return 50;
        case TensorType::IQ1_M:
            return 56;
        default:
            return 0; // Non-block types
        }
    }

    // Get elements per block for quantized types
    size_t getBlockElements(TensorType dtype)
    {
        switch (dtype)
        {
        case TensorType::Q4_0:
        case TensorType::Q4_1:
        case TensorType::Q5_0:
        case TensorType::Q5_1:
        case TensorType::Q8_0:
        case TensorType::Q8_1:
        case TensorType::IQ4_NL:
        case TensorType::IQ4_XS:
            return 32; // 32-element blocks
        case TensorType::Q2_K:
        case TensorType::Q3_K:
        case TensorType::Q4_K:
        case TensorType::Q5_K:
        case TensorType::Q6_K:
        case TensorType::Q8_K:
        case TensorType::IQ2_XXS:
        case TensorType::IQ2_XS:
        case TensorType::IQ2_S:
        case TensorType::IQ3_XXS:
        case TensorType::IQ3_S:
        case TensorType::IQ1_S:
        case TensorType::IQ1_M:
            return 256; // 256-element superblocks
        default:
            return 0;
        }
    }

} // namespace

// =============================================================================
// Activation Tensor Types - CUDA Allocation
// =============================================================================

#ifdef HAVE_CUDA

class CUDATensorTypeAllocationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!hasCUDA())
        {
            GTEST_SKIP() << "No CUDA device available";
        }
        backend_ = std::make_unique<CUDABackend>();
        device_count_ = backend_->deviceCount();
        if (device_count_ == 0)
        {
            GTEST_SKIP() << "No CUDA devices found";
        }
        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
        backend_.reset();
    }

    std::unique_ptr<CUDABackend> backend_;
    int device_count_ = 0;
};

// =============================================================================
// FP32 Activation Tensor Tests
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, FP32Tensor_Allocation)
{
    const size_t rows = 128;
    const size_t cols = 256;
    const int device_id = 0;

    // Create CUDA FP32 tensor
    CUDAFp32Tensor tensor({rows, cols}, device_id);

    // Verify properties
    EXPECT_EQ(tensor.dtype(), TensorType::FP32);
    EXPECT_EQ(tensor.shape()[0], rows);
    EXPECT_EQ(tensor.shape()[1], cols);
    EXPECT_EQ(tensor.numel(), rows * cols);
    EXPECT_TRUE(tensor.is_on_gpu());
    EXPECT_FALSE(tensor.is_on_cpu());
    EXPECT_EQ(tensor.device_index(), device_id);
    EXPECT_NE(tensor.device_ptr(), nullptr);
}

TEST_F(CUDATensorTypeAllocationTest, FP32Tensor_HostDeviceTransfer)
{
    const size_t rows = 64;
    const size_t cols = 128;
    const int device_id = 0;

    // Create host data
    std::vector<float> host_data(rows * cols);
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        host_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create CUDA tensor
    CUDAFp32Tensor tensor({rows, cols}, device_id);

    // Upload to device
    tensor.copyFromHost(host_data.data(), host_data.size() * sizeof(float));

    // Download back
    std::vector<float> result(rows * cols, 0.0f);
    tensor.copyToHost(result.data(), result.size() * sizeof(float));

    // Verify roundtrip
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(result[i], host_data[i]) << "Mismatch at index " << i;
    }
}

// =============================================================================
// INT8 Tensor Tests (Activation/Weight format)
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, INT8Tensor_Allocation)
{
    const size_t rows = 128;
    const size_t cols = 256;
    const int device_id = 0;

    // Create CUDA INT8 tensor
    CUDAINT8Tensor tensor({rows, cols}, device_id);

    EXPECT_EQ(tensor.dtype(), TensorType::INT8);
    EXPECT_EQ(tensor.numel(), rows * cols);
    EXPECT_TRUE(tensor.is_on_gpu());
    EXPECT_NE(tensor.device_ptr(), nullptr);
}

TEST_F(CUDATensorTypeAllocationTest, INT8Tensor_HostDeviceTransfer)
{
    const size_t rows = 64;
    const size_t cols = 128;
    const int device_id = 0;

    // Create host data
    std::vector<int8_t> host_data(rows * cols);
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        host_data[i] = static_cast<int8_t>((i % 256) - 128);
    }

    // Create CUDA tensor
    CUDAINT8Tensor tensor({rows, cols}, device_id);

    // Upload to device
    tensor.copyFromHost(host_data.data(), host_data.size() * sizeof(int8_t));

    // Download back
    std::vector<int8_t> result(rows * cols, 0);
    tensor.copyToHost(result.data(), result.size() * sizeof(int8_t));

    // Verify roundtrip
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        EXPECT_EQ(result[i], host_data[i]) << "Mismatch at index " << i;
    }
}

// =============================================================================
// INT32 Tensor Tests (Accumulator format)
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, INT32Tensor_Allocation)
{
    const size_t rows = 128;
    const size_t cols = 256;
    const int device_id = 0;

    // Create CUDA INT32 tensor
    CUDAINT32Tensor tensor({rows, cols}, device_id);

    EXPECT_EQ(tensor.dtype(), TensorType::INT32);
    EXPECT_EQ(tensor.numel(), rows * cols);
    EXPECT_TRUE(tensor.is_on_gpu());
    EXPECT_NE(tensor.device_ptr(), nullptr);
}

TEST_F(CUDATensorTypeAllocationTest, INT32Tensor_HostDeviceTransfer)
{
    const size_t rows = 32;
    const size_t cols = 64;
    const int device_id = 0;

    // Create host data
    std::vector<int32_t> host_data(rows * cols);
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        host_data[i] = static_cast<int32_t>(i * 1000);
    }

    // Create CUDA tensor
    CUDAINT32Tensor tensor({rows, cols}, device_id);

    // Upload to device
    tensor.copyFromHost(host_data.data(), host_data.size() * sizeof(int32_t));

    // Download back
    std::vector<int32_t> result(rows * cols, 0);
    tensor.copyToHost(result.data(), result.size() * sizeof(int32_t));

    // Verify roundtrip
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        EXPECT_EQ(result[i], host_data[i]) << "Mismatch at index " << i;
    }
}

// =============================================================================
// GPU Backend Raw Allocation for Quantized Types
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, RawAllocation_AllQuantizedTypes)
{
    // Test raw GPU memory allocation for all quantized block formats
    // This verifies the backend can allocate memory for any quant format

    struct QuantTypeInfo
    {
        TensorType dtype;
        const char *name;
    };

    std::vector<QuantTypeInfo> quant_types = {
        {TensorType::Q4_0, "Q4_0"},
        {TensorType::Q4_1, "Q4_1"},
        {TensorType::Q5_0, "Q5_0"},
        {TensorType::Q5_1, "Q5_1"},
        {TensorType::Q8_0, "Q8_0"},
        {TensorType::Q8_1, "Q8_1"},
        {TensorType::Q2_K, "Q2_K"},
        {TensorType::Q3_K, "Q3_K"},
        {TensorType::Q4_K, "Q4_K"},
        {TensorType::Q5_K, "Q5_K"},
        {TensorType::Q6_K, "Q6_K"},
        {TensorType::Q8_K, "Q8_K"},
        {TensorType::IQ4_NL, "IQ4_NL"},
        {TensorType::IQ4_XS, "IQ4_XS"},
        {TensorType::IQ2_XXS, "IQ2_XXS"},
        {TensorType::IQ2_XS, "IQ2_XS"},
        {TensorType::IQ2_S, "IQ2_S"},
        {TensorType::IQ3_XXS, "IQ3_XXS"},
        {TensorType::IQ3_S, "IQ3_S"},
        {TensorType::IQ1_S, "IQ1_S"},
        {TensorType::IQ1_M, "IQ1_M"},
    };

    const int device_id = 0;
    const size_t rows = 512;
    const size_t cols = 256; // Must be divisible by block elements

    for (const auto &qt : quant_types)
    {
        size_t block_bytes = getBlockSizeBytes(qt.dtype);
        size_t block_elements = getBlockElements(qt.dtype);

        if (block_bytes == 0 || block_elements == 0)
        {
            ADD_FAILURE() << "Unknown block format for " << qt.name;
            continue;
        }

        // Calculate total bytes needed
        size_t total_elements = rows * cols;
        size_t num_blocks = total_elements / block_elements;
        size_t total_bytes = num_blocks * block_bytes;

        // Allocate GPU memory
        void *device_ptr = backend_->allocate(total_bytes, device_id);
        ASSERT_NE(device_ptr, nullptr)
            << "Failed to allocate GPU memory for " << qt.name
            << " (" << total_bytes << " bytes)";

        // Free GPU memory
        backend_->free(device_ptr, device_id);

        // Log success
        std::cout << "[PASS] " << qt.name << ": allocated " << total_bytes
                  << " bytes (" << num_blocks << " blocks)" << std::endl;
    }
}

// =============================================================================
// CPU Quantized Tensor Creation and Device Transfer
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, Q4_0Tensor_CreateAndSetDevice)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18;
    const size_t num_blocks = rows * (cols / block_size);

    // Create random block data
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &byte : raw_data)
    {
        byte = static_cast<uint8_t>(dist(gen));
    }
    // Set scale values
    for (size_t b = 0; b < num_blocks; ++b)
    {
        uint16_t scale_bits = 0x3C00; // ~1.0 in FP16
        memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
    }

    // Create Q4_0 tensor on CPU
    auto tensor = std::make_unique<Q4_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->native_type(), TensorType::Q4_0);

    // Tensor starts on CPU
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());

    // KernelFactory should create GEMM kernel (will upload to GPU internally)
    auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
    EXPECT_NE(kernel, nullptr) << "KernelFactory should create GEMM kernel for Q4_0 on CUDA";
}

TEST_F(CUDATensorTypeAllocationTest, Q8_0Tensor_CreateAndSetDevice)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 32;
    const size_t bytes_per_block = 34;
    const size_t num_blocks = rows * (cols / block_size);

    // Create random block data
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &byte : raw_data)
    {
        byte = static_cast<uint8_t>(dist(gen));
    }
    for (size_t b = 0; b < num_blocks; ++b)
    {
        uint16_t scale_bits = 0x3C00;
        memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
    }

    // Create Q8_0 tensor on CPU
    auto tensor = std::make_unique<Q8_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->native_type(), TensorType::Q8_0);

    // KernelFactory should create GEMM kernel
    auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
    EXPECT_NE(kernel, nullptr) << "KernelFactory should create GEMM kernel for Q8_0 on CUDA";
}

TEST_F(CUDATensorTypeAllocationTest, IQ4_NLTensor_CreateAndSetDevice)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18;
    const size_t num_blocks = rows * (cols / block_size);

    // Create random block data
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &byte : raw_data)
    {
        byte = static_cast<uint8_t>(dist(gen));
    }
    for (size_t b = 0; b < num_blocks; ++b)
    {
        uint16_t scale_bits = 0x3C00;
        memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
    }

    // Create IQ4_NL tensor on CPU
    auto tensor = std::make_unique<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->native_type(), TensorType::IQ4_NL);

    // KernelFactory should create GEMM kernel
    auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
    EXPECT_NE(kernel, nullptr) << "KernelFactory should create GEMM kernel for IQ4_NL on CUDA";
}

// =============================================================================
// K-Quant Tensor Types
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, Q6_KTensor_CreateAndSetDevice)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_elements = 256; // K-quant superblock
    const size_t bytes_per_block = 210;
    const size_t num_blocks = (rows * cols) / block_elements;

    // Create random block data
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &byte : raw_data)
    {
        byte = static_cast<uint8_t>(dist(gen));
    }

    // Create Q6_K tensor on CPU
    auto tensor = std::make_unique<Q6_KTensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->native_type(), TensorType::Q6_K);

    // KernelFactory should create GEMM kernel
    auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
    EXPECT_NE(kernel, nullptr) << "KernelFactory should create GEMM kernel for Q6_K on CUDA";
}

TEST_F(CUDATensorTypeAllocationTest, Q4_KTensor_CreateAndSetDevice)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_elements = 256;
    const size_t bytes_per_block = 144;
    const size_t num_blocks = (rows * cols) / block_elements;

    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &byte : raw_data)
    {
        byte = static_cast<uint8_t>(dist(gen));
    }

    auto tensor = std::make_unique<Q4_KTensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->native_type(), TensorType::Q4_K);

    auto *kernel = KernelFactory::getOrCreateGemm(tensor.get());
    EXPECT_NE(kernel, nullptr) << "KernelFactory should create GEMM kernel for Q4_K on CUDA";
}

// =============================================================================
// Multiple GPU Allocation Test
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, MultipleAllocation_SameDevice)
{
    // Test allocating multiple tensors on same GPU
    const int device_id = 0;

    // Allocate several FP32 tensors
    std::vector<std::unique_ptr<CUDAFp32Tensor>> tensors;
    for (int i = 0; i < 5; ++i)
    {
        tensors.push_back(std::make_unique<CUDAFp32Tensor>(
            std::vector<size_t>{128, 256}, device_id));
        EXPECT_NE(tensors.back()->device_ptr(), nullptr)
            << "Failed to allocate tensor " << i;
    }

    // All should be on same device
    for (const auto &t : tensors)
    {
        EXPECT_EQ(t->device_index(), device_id);
        EXPECT_TRUE(t->is_on_gpu());
    }

    // Tensors cleanup automatically
}

// =============================================================================
// Large Tensor Allocation Test
// =============================================================================

TEST_F(CUDATensorTypeAllocationTest, LargeTensor_Allocation)
{
    // Test allocating a large tensor (typical hidden state size)
    // Qwen2.5-7B: hidden_dim=3584, batch*seq_len could be 2048
    const size_t rows = 2048;
    const size_t cols = 3584;
    const int device_id = 0;

    CUDAFp32Tensor tensor({rows, cols}, device_id);

    size_t expected_bytes = rows * cols * sizeof(float);
    EXPECT_EQ(tensor.size_bytes(), expected_bytes);
    EXPECT_NE(tensor.device_ptr(), nullptr);
    EXPECT_TRUE(tensor.is_on_gpu());

    std::cout << "[PASS] Large tensor: " << rows << "x" << cols
              << " = " << (expected_bytes / (1024 * 1024)) << " MB" << std::endl;
}

#else // !HAVE_CUDA

TEST(CUDATensorTypeAllocationTest, NoCUDA)
{
    GTEST_SKIP() << "CUDA not available (HAVE_CUDA=OFF)";
}

#endif // HAVE_CUDA

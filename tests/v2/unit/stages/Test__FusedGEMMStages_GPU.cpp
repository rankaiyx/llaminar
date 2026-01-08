/**
 * @file Test__FusedGEMMStages_GPU.cpp
 * @brief GPU dispatch tests for FusedQKVGEMMStage and FusedGateUpGEMMStage
 *
 * **Purpose**: Validate that fused GEMM stages correctly dispatch to CUDA kernels
 * when configured with GPU tensors.
 *
 * **Phase 4.3**: Part of GPU Inference Integration
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "execution/compute_stages/stages/FusedGateUpGEMMStage.h"
#include "tensors/Tensors.h"
#include "tensors/cuda/CUDATypedTensor.h"
#include "utils/Logger.h"
#include "kernels/KernelFactory.h"
#include "backends/ComputeBackend.h"
#include "execution/DeviceContext.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace llaminar2;

#ifdef HAVE_CUDA

namespace
{
    // =============================================================================
    // Helper Functions
    // =============================================================================

    bool hasCUDA()
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }

    std::vector<float> generateRandomFP32(size_t count, unsigned seed)
    {
        std::vector<float> data(count);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : data)
        {
            v = dist(gen);
        }
        return data;
    }

    // Create FP32 weight tensor (CPU)
    std::unique_ptr<FP32Tensor> createFP32Weights(size_t rows, size_t cols, unsigned seed)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        auto data = generateRandomFP32(rows * cols, seed);
        std::memcpy(tensor->mutable_data(), data.data(), rows * cols * sizeof(float));
        return tensor;
    }

    // Create Q4_0 weight tensor (CPU)
    std::unique_ptr<Q4_0Tensor> createQ4_0Weights(size_t rows, size_t cols, unsigned seed)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 18; // Q4_0: 2 bytes scale + 16 bytes data
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
        {
            byte = static_cast<uint8_t>(dist(gen));
        }
        // Set reasonable scale values
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00; // 1.0 in FP16
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q4_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    // Create GPU FP32 tensor with random data
    std::unique_ptr<CUDAFp32Tensor> createGPUTensor(int rows, int cols, unsigned seed)
    {
        auto cpu_data = generateRandomFP32(rows * cols, seed);
        std::vector<size_t> shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
        auto tensor = std::make_unique<CUDAFp32Tensor>(shape, 0);
        tensor->copyFromHost(cpu_data.data(), cpu_data.size() * sizeof(float));
        return tensor;
    }

    // Create empty GPU tensor for output
    std::unique_ptr<CUDAFp32Tensor> createGPUOutputTensor(int rows, int cols)
    {
        std::vector<size_t> shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
        auto tensor = std::make_unique<CUDAFp32Tensor>(shape, 0);
        cudaMemset(tensor->raw_mutable_data(), 0, rows * cols * sizeof(float));
        return tensor;
    }
} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__FusedGEMMStages_GPU : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!hasCUDA())
        {
            GTEST_SKIP() << "No CUDA GPU available";
        }
    }
};

// ============================================================================
// FusedQKVGEMMStage GPU Tests
// ============================================================================

TEST_F(Test__FusedGEMMStages_GPU, FusedQKV_SupportsGPUBackend)
{
    // Create minimal params just to test supportsBackend()
    const int m = 4, k = 64, n_q = 64, n_k = 64, n_v = 64;

    auto wq = createFP32Weights(n_q, k, 100);
    auto wk = createFP32Weights(n_k, k, 101);
    auto wv = createFP32Weights(n_v, k, 102);

    auto input = createGPUTensor(m, k, 200);
    auto output_q = createGPUOutputTensor(m, n_q);
    auto output_k = createGPUOutputTensor(m, n_k);
    auto output_v = createGPUOutputTensor(m, n_v);

    FusedQKVGEMMStage::Params params;
    params.input = input.get();
    params.m = m;
    params.k = k;
    params.wq = wq.get();
    params.output_q = output_q.get();
    params.n_q = n_q;
    params.wk = wk.get();
    params.output_k = output_k.get();
    params.n_k = n_k;
    params.wv = wv.get();
    params.output_v = output_v.get();
    params.n_v = n_v;

    FusedQKVGEMMStage stage(params);

    // Should report support for GPU backends
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(Test__FusedGEMMStages_GPU, FusedQKV_GPUTensors_Execute)
{
    // Test: QKV GEMM with GPU activations, CPU weights
    // Dimensions similar to Qwen2 0.5B attention
    const int m = 16;    // sequence length
    const int k = 896;   // d_model
    const int n_q = 896; // Q output dim
    const int n_k = 128; // K output dim (GQA)
    const int n_v = 128; // V output dim (GQA)

    // Create CPU weights (Q4_0 quantized)
    auto wq = createQ4_0Weights(n_q, k, 100);
    auto wk = createQ4_0Weights(n_k, k, 101);
    auto wv = createQ4_0Weights(n_v, k, 102);

    // Create GPU input and outputs
    auto input = createGPUTensor(m, k, 200);
    auto output_q = createGPUOutputTensor(m, n_q);
    auto output_k = createGPUOutputTensor(m, n_k);
    auto output_v = createGPUOutputTensor(m, n_v);

    LOG_INFO("[FusedQKV_GPUTensors] Created tensors: input=" << input->is_on_gpu()
                                                             << " output_q=" << output_q->is_on_gpu());

    FusedQKVGEMMStage::Params params;
    params.input = input.get();
    params.m = m;
    params.k = k;
    params.wq = wq.get();
    params.output_q = output_q.get();
    params.n_q = n_q;
    params.wk = wk.get();
    params.output_k = output_k.get();
    params.n_k = n_k;
    params.wv = wv.get();
    params.output_v = output_v.get();
    params.n_v = n_v;

    FusedQKVGEMMStage stage(params);

    // Create device context for execution
    // device_idx=1 (GPU 0 after CPU), cuda_device_id=0
    CUDADeviceContext cuda_ctx(1, 0);

    // Execute
    bool success = stage.execute(&cuda_ctx);
    ASSERT_TRUE(success) << "FusedQKVGEMMStage GPU execution failed";

    // Verify outputs are non-zero
    std::vector<float> q_output(m * n_q);
    std::vector<float> k_output(m * n_k);
    std::vector<float> v_output(m * n_v);

    cudaMemcpy(q_output.data(), output_q->raw_data(), m * n_q * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_output.data(), output_k->raw_data(), m * n_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_output.data(), output_v->raw_data(), m * n_v * sizeof(float), cudaMemcpyDeviceToHost);

    // Check Q output
    float q_sum = 0.0f;
    for (float v : q_output)
        q_sum += std::abs(v);
    EXPECT_GT(q_sum, 0.0f) << "Q output is all zeros";

    // Check K output
    float k_sum = 0.0f;
    for (float v : k_output)
        k_sum += std::abs(v);
    EXPECT_GT(k_sum, 0.0f) << "K output is all zeros";

    // Check V output
    float v_sum = 0.0f;
    for (float v : v_output)
        v_sum += std::abs(v);
    EXPECT_GT(v_sum, 0.0f) << "V output is all zeros";

    LOG_INFO("[FusedQKV_GPUTensors] PASSED - Q_sum=" << q_sum
                                                     << " K_sum=" << k_sum << " V_sum=" << v_sum);
}

// ============================================================================
// FusedGateUpGEMMStage GPU Tests
// ============================================================================

TEST_F(Test__FusedGEMMStages_GPU, FusedGateUp_SupportsGPUBackend)
{
    const int m = 4, k = 64, n_gate = 128, n_up = 128;

    auto w_gate = createFP32Weights(n_gate, k, 100);
    auto w_up = createFP32Weights(n_up, k, 101);

    auto input = createGPUTensor(m, k, 200);
    auto output_gate = createGPUOutputTensor(m, n_gate);
    auto output_up = createGPUOutputTensor(m, n_up);

    FusedGateUpGEMMStage::Params params;
    params.input = input.get();
    params.m = m;
    params.k = k;
    params.w_gate = w_gate.get();
    params.output_gate = output_gate.get();
    params.n_gate = n_gate;
    params.w_up = w_up.get();
    params.output_up = output_up.get();
    params.n_up = n_up;

    FusedGateUpGEMMStage stage(params);

    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(Test__FusedGEMMStages_GPU, FusedGateUp_GPUTensors_Execute)
{
    // Test: Gate/Up GEMM with GPU activations, CPU weights
    // Use smaller dimensions for unit test
    const int m = 16;       // sequence length
    const int k = 128;      // d_model (reduced for test)
    const int n_gate = 256; // intermediate_size (reduced for test)
    const int n_up = 256;

    // Create CPU weights (Q4_0 quantized)
    auto w_gate = createQ4_0Weights(n_gate, k, 100);
    auto w_up = createQ4_0Weights(n_up, k, 101);

    // Create GPU input and outputs
    auto input = createGPUTensor(m, k, 200);
    auto output_gate = createGPUOutputTensor(m, n_gate);
    auto output_up = createGPUOutputTensor(m, n_up);

    LOG_INFO("[FusedGateUp_GPUTensors] Created tensors: input=" << input->is_on_gpu()
                                                                << " output_gate=" << output_gate->is_on_gpu());

    FusedGateUpGEMMStage::Params params;
    params.input = input.get();
    params.m = m;
    params.k = k;
    params.w_gate = w_gate.get();
    params.output_gate = output_gate.get();
    params.n_gate = n_gate;
    params.w_up = w_up.get();
    params.output_up = output_up.get();
    params.n_up = n_up;

    FusedGateUpGEMMStage stage(params);

    // Create device context for execution
    CUDADeviceContext cuda_ctx(1, 0);

    // Execute
    bool success = stage.execute(&cuda_ctx);
    ASSERT_TRUE(success) << "FusedGateUpGEMMStage GPU execution failed";

    // Verify outputs are non-zero
    std::vector<float> gate_output(m * n_gate);
    std::vector<float> up_output(m * n_up);

    cudaMemcpy(gate_output.data(), output_gate->raw_data(), m * n_gate * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(up_output.data(), output_up->raw_data(), m * n_up * sizeof(float), cudaMemcpyDeviceToHost);

    // Check gate output
    float gate_sum = 0.0f;
    for (float v : gate_output)
        gate_sum += std::abs(v);
    EXPECT_GT(gate_sum, 0.0f) << "Gate output is all zeros";

    // Check up output
    float up_sum = 0.0f;
    for (float v : up_output)
        up_sum += std::abs(v);
    EXPECT_GT(up_sum, 0.0f) << "Up output is all zeros";

    LOG_INFO("[FusedGateUp_GPUTensors] PASSED - gate_sum=" << gate_sum
                                                           << " up_sum=" << up_sum);
}

TEST_F(Test__FusedGEMMStages_GPU, FusedGateUp_DecodeSize)
{
    // Test with decode-size (m=1)
    const int m = 1;
    const int k = 128;      // d_model (reduced for test)
    const int n_gate = 256; // intermediate_size (reduced for test)
    const int n_up = 256;

    auto w_gate = createQ4_0Weights(n_gate, k, 100);
    auto w_up = createQ4_0Weights(n_up, k, 101);

    auto input = createGPUTensor(m, k, 200);
    auto output_gate = createGPUOutputTensor(m, n_gate);
    auto output_up = createGPUOutputTensor(m, n_up);

    FusedGateUpGEMMStage::Params params;
    params.input = input.get();
    params.m = m;
    params.k = k;
    params.w_gate = w_gate.get();
    params.output_gate = output_gate.get();
    params.n_gate = n_gate;
    params.w_up = w_up.get();
    params.output_up = output_up.get();
    params.n_up = n_up;

    FusedGateUpGEMMStage stage(params);
    CUDADeviceContext cuda_ctx(1, 0);

    bool success = stage.execute(&cuda_ctx);
    ASSERT_TRUE(success) << "FusedGateUpGEMMStage decode-size GPU execution failed";

    // Verify non-zero output
    std::vector<float> gate_output(m * n_gate);
    cudaMemcpy(gate_output.data(), output_gate->raw_data(), m * n_gate * sizeof(float), cudaMemcpyDeviceToHost);

    float gate_sum = 0.0f;
    for (float v : gate_output)
        gate_sum += std::abs(v);
    EXPECT_GT(gate_sum, 0.0f) << "Decode-size gate output is all zeros";

    LOG_INFO("[FusedGateUp_DecodeSize] PASSED - gate_sum=" << gate_sum);
}

#else // !HAVE_CUDA

TEST(Test__FusedGEMMStages_GPU, SkipWithoutCUDA)
{
    GTEST_SKIP() << "CUDA not available";
}

#endif // HAVE_CUDA

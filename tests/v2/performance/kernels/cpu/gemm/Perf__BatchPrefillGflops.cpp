/**
 * @file Perf__BatchPrefillGflops.cpp
 * @brief Benchmark peak GFLOPS for batch prefill scenarios
 *
 * Tests various batch sizes with Qwen 0.5B dimensions to find peak throughput.
 * Uses the best-performing kernel configuration for each shape.
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#include "../../src/v2/kernels/cuda/CudaGemmAutoTuner.h"
#include "../../src/v2/kernels/cuda/CudaGemmVariantsBaseline.h"
#include "../../src/v2/kernels/cuda/IQ4_NL_BlockDecoder.h"
#include "../../src/v2/tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace llaminar2::cuda;

/**
 * @brief Test fixture for batch prefill benchmarking
 */
class BatchPrefillGflops : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cudaGetDevice(&device_id_);
        cudaGetDeviceProperties(&device_props_, device_id_);

        cudaStreamCreate(&stream_);
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);

        // More iterations for accurate peak measurement
        warmup_iterations_ = 5;
        benchmark_iterations_ = 20;
    }

    void TearDown() override
    {
        if (stream_)
            cudaStreamDestroy(stream_);
        if (start_event_)
            cudaEventDestroy(start_event_);
        if (stop_event_)
            cudaEventDestroy(stop_event_);

        if (test_A_device_)
            cudaFree(test_A_device_);
        if (test_B_device_)
            cudaFree(test_B_device_);
        if (test_C_device_)
            cudaFree(test_C_device_);
    }

    void allocateTestData(int m, int n, int k)
    {
        if (m > allocated_m_ || n > allocated_n_ || k > allocated_k_)
        {
            if (test_A_device_)
                cudaFree(test_A_device_);
            if (test_B_device_)
                cudaFree(test_B_device_);
            if (test_C_device_)
                cudaFree(test_C_device_);

            cudaMalloc(&test_A_device_, m * k * sizeof(float));
            cudaMalloc(&test_B_device_, n * (k / 32) * sizeof(IQ4_NLBlock));
            cudaMalloc(&test_C_device_, m * n * sizeof(float));

            allocated_m_ = m;
            allocated_n_ = n;
            allocated_k_ = k;

            // Initialize with random data
            std::vector<float> A_host(m * k);
            std::vector<IQ4_NLBlock> B_host(n * (k / 32));

            for (auto &val : A_host)
                val = static_cast<float>(rand()) / RAND_MAX;
            for (auto &block : B_host)
            {
                block.d = llaminar2::fp32_to_fp16(1.0f);
                for (int i = 0; i < 16; ++i)
                    block.qs[i] = rand() % 256;
            }

            cudaMemcpy(test_A_device_, A_host.data(), A_host.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(test_B_device_, B_host.data(), B_host.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
        }
    }

    /**
     * @brief Find best config for a given shape by testing all
     */
    CudaGemmConfig findBestConfig(int m, int n, int k)
    {
        auto &tuner = CudaGemmAutoTuner::instance();
        auto all_configs = tuner.getAvailableConfigs();

        CudaGemmConfig best_config;
        double best_gflops = 0.0;

        std::cout << "  Testing " << all_configs.size() << " configs to find best...";
        std::cout.flush();

        for (const auto &config : all_configs)
        {
            // Quick single-iteration test
            auto err = launchIQ4NLGemmVariant(test_A_device_, test_B_device_, test_C_device_,
                                              m, n, k, config, stream_);
            if (err != cudaSuccess)
                continue;

            cudaStreamSynchronize(stream_);

            // Time it
            cudaEventRecord(start_event_, stream_);
            launchIQ4NLGemmVariant(test_A_device_, test_B_device_, test_C_device_,
                                   m, n, k, config, stream_);
            cudaEventRecord(stop_event_, stream_);
            cudaEventSynchronize(stop_event_);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);

            double flops = 2.0 * m * n * k;
            double gflops = (flops / 1e9) / (elapsed_ms / 1000.0);

            if (gflops > best_gflops)
            {
                best_gflops = gflops;
                best_config = config;
            }
        }

        std::cout << " Found best: " << best_config.id() << " (" << best_gflops << " GFLOPS)\n";
        return best_config;
    }

    /**
     * @brief Benchmark a configuration with high precision
     */
    double benchmarkConfigAccurate(const CudaGemmConfig &config, int m, int n, int k)
    {
        // Warmup
        for (int i = 0; i < warmup_iterations_; ++i)
        {
            launchIQ4NLGemmVariant(test_A_device_, test_B_device_, test_C_device_,
                                   m, n, k, config, stream_);
        }
        cudaStreamSynchronize(stream_);

        // Timed runs
        cudaEventRecord(start_event_, stream_);
        for (int i = 0; i < benchmark_iterations_; ++i)
        {
            launchIQ4NLGemmVariant(test_A_device_, test_B_device_, test_C_device_,
                                   m, n, k, config, stream_);
        }
        cudaEventRecord(stop_event_, stream_);
        cudaEventSynchronize(stop_event_);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);

        double avg_time_ms = elapsed_ms / benchmark_iterations_;
        double flops = 2.0 * m * n * k;
        return (flops / 1e9) / (avg_time_ms / 1000.0);
    }

    int device_id_ = 0;
    cudaDeviceProp device_props_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    float *test_A_device_ = nullptr;
    IQ4_NLBlock *test_B_device_ = nullptr;
    float *test_C_device_ = nullptr;

    int allocated_m_ = 0;
    int allocated_n_ = 0;
    int allocated_k_ = 0;

    int warmup_iterations_ = 5;
    int benchmark_iterations_ = 20;
};

/**
 * @brief Benchmark batch prefill with Qwen 0.5B Q/K/V dimensions
 */
TEST_F(BatchPrefillGflops, Qwen_0_5B_QKV_BatchSweep)
{
    const int n = 896; // d_model for Qwen 0.5B
    const int k = 896;

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Qwen 0.5B Q/K/V Projection - Batch Prefill Performance      ║\n";
    std::cout << "║ Matrix: [batch × 896 × 896]                                 ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Batch │ Config              │ GFLOPS │ Time/Batch │ Speedup ║\n";
    std::cout << "╠═══════╪═════════════════════╪════════╪════════════╪═════════╣\n";

    // Test batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    double baseline_gflops = 0.0;

    for (int batch : batch_sizes)
    {
        const int m = batch;

        std::cout << "║ " << std::setw(5) << batch << " │ ";
        std::cout.flush();

        allocateTestData(m, n, k);

        // Find best config for this batch size
        auto best_config = findBestConfig(m, n, k);

        // Benchmark it accurately
        double gflops = benchmarkConfigAccurate(best_config, m, n, k);

        if (batch == 1)
            baseline_gflops = gflops;

        double speedup = gflops / baseline_gflops;
        double time_per_batch_ms = (2.0 * m * n * k / 1e9) / gflops * 1000.0;

        std::cout << "\r║ " << std::setw(5) << batch << " │ ";
        std::cout << std::setw(19) << best_config.id() << " │ ";
        std::cout << std::fixed << std::setprecision(1) << std::setw(6) << gflops << " │ ";
        std::cout << std::setw(9) << std::setprecision(3) << time_per_batch_ms << "ms │ ";
        std::cout << std::setw(6) << std::setprecision(1) << speedup << "× ║\n";
    }

    std::cout << "╚═══════╧═════════════════════╧════════╧════════════╧═════════╝\n\n";
}

/**
 * @brief Benchmark batch prefill with Qwen 0.5B FFN gate dimensions
 */
TEST_F(BatchPrefillGflops, Qwen_0_5B_FFN_Gate_BatchSweep)
{
    const int n = 4864; // d_ff for Qwen 0.5B
    const int k = 896;  // d_model

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Qwen 0.5B FFN Gate - Batch Prefill Performance              ║\n";
    std::cout << "║ Matrix: [batch × 4864 × 896]                                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Batch │ Config              │ GFLOPS │ Time/Batch │ Speedup ║\n";
    std::cout << "╠═══════╪═════════════════════╪════════╪════════════╪═════════╣\n";

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    double baseline_gflops = 0.0;

    for (int batch : batch_sizes)
    {
        const int m = batch;

        std::cout << "║ " << std::setw(5) << batch << " │ ";
        std::cout.flush();

        allocateTestData(m, n, k);
        auto best_config = findBestConfig(m, n, k);
        double gflops = benchmarkConfigAccurate(best_config, m, n, k);

        if (batch == 1)
            baseline_gflops = gflops;

        double speedup = gflops / baseline_gflops;
        double time_per_batch_ms = (2.0 * m * n * k / 1e9) / gflops * 1000.0;

        std::cout << "\r║ " << std::setw(5) << batch << " │ ";
        std::cout << std::setw(19) << best_config.id() << " │ ";
        std::cout << std::fixed << std::setprecision(1) << std::setw(6) << gflops << " │ ";
        std::cout << std::setw(9) << std::setprecision(3) << time_per_batch_ms << "ms │ ";
        std::cout << std::setw(6) << std::setprecision(1) << speedup << "× ║\n";
    }

    std::cout << "╚═══════╧═════════════════════╧════════╧════════════╧═════════╝\n\n";
}

/**
 * @brief Benchmark batch prefill with Qwen 0.5B FFN down dimensions
 */
TEST_F(BatchPrefillGflops, Qwen_0_5B_FFN_Down_BatchSweep)
{
    const int n = 896;  // d_model
    const int k = 4864; // d_ff

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Qwen 0.5B FFN Down - Batch Prefill Performance              ║\n";
    std::cout << "║ Matrix: [batch × 896 × 4864]                                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Batch │ Config              │ GFLOPS │ Time/Batch │ Speedup ║\n";
    std::cout << "╠═══════╪═════════════════════╪════════╪════════════╪═════════╣\n";

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    double baseline_gflops = 0.0;

    for (int batch : batch_sizes)
    {
        const int m = batch;

        std::cout << "║ " << std::setw(5) << batch << " │ ";
        std::cout.flush();

        allocateTestData(m, n, k);
        auto best_config = findBestConfig(m, n, k);
        double gflops = benchmarkConfigAccurate(best_config, m, n, k);

        if (batch == 1)
            baseline_gflops = gflops;

        double speedup = gflops / baseline_gflops;
        double time_per_batch_ms = (2.0 * m * n * k / 1e9) / gflops * 1000.0;

        std::cout << "\r║ " << std::setw(5) << batch << " │ ";
        std::cout << std::setw(19) << best_config.id() << " │ ";
        std::cout << std::fixed << std::setprecision(1) << std::setw(6) << gflops << " │ ";
        std::cout << std::setw(9) << std::setprecision(3) << time_per_batch_ms << "ms │ ";
        std::cout << std::setw(6) << std::setprecision(1) << speedup << "× ║\n";
    }

    std::cout << "╚═══════╧═════════════════════╧════════╧════════════╧═════════╝\n\n";
}

/**
 * @file Perf__OneDNN_Conversion.cpp
 * @brief Benchmarks IActivationTensor::to_int8_activation_pack conversion performance.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;
using namespace std::chrono;

namespace
{
    std::unique_ptr<FP32Tensor> generate_random_activations(int M, int K)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> values(static_cast<size_t>(M) * static_cast<size_t>(K));
        for (auto &v : values)
        {
            v = dist(rng);
        }

        std::memcpy(tensor->mutable_data(), values.data(), values.size() * sizeof(float));
        return tensor;
    }

    std::unique_ptr<FP16Tensor> generate_random_fp16_activations(int M, int K)
    {
        auto fp32_tensor = generate_random_activations(M, K);
        auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        tensor->copyFrom(fp32_tensor.get());
        return tensor;
    }

    std::unique_ptr<BF16Tensor> generate_random_bf16_activations(int M, int K)
    {
        auto fp32_tensor = generate_random_activations(M, K);
        auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        tensor->copyFrom(fp32_tensor.get());
        return tensor;
    }
}

class OneDNNConversionPerf : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI if not already initialized (handled by main, but good practice)
    }

    void TearDown() override
    {
    }

    void RunBenchmark(const std::string &name, int M, int K, int iters = 100)
    {
        auto tensor = generate_random_activations(M, K);

        // Warmup
        for (int i = 0; i < 10; ++i)
        {
            auto pack = tensor->to_int8_activation_pack(M, K);
            (void)pack;
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            auto pack = tensor->to_int8_activation_pack(M, K);
            // Prevent optimization
            if (pack.data.empty())
                std::cerr << "Empty pack\n";
        }
        auto end = high_resolution_clock::now();

        double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avg_ms = total_ms / iters;

        // Calculate throughput (GB/s)
        // Read: M*K * 4 bytes (FP32)
        // Write: M*K * 1 byte (INT8) + M * 4 bytes (scales)
        double bytes_processed = static_cast<double>(M) * K * 4.0;        // Just counting input read for now as primary metric
        double gb_per_sec = (bytes_processed * iters) / (total_ms * 1e6); // bytes / (ms * 10^6) * 10^9 = GB/s? No.
        // bytes / (ms * 1e-3) = bytes/s. / 1e9 = GB/s.
        // (bytes * iters) / (total_ms * 1e-3) / 1e9 = (bytes * iters) / (total_ms * 1e6)

        std::cout << std::left << std::setw(40) << name
                  << " M=" << std::setw(5) << M
                  << " K=" << std::setw(5) << K
                  << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " | Throughput: " << std::setprecision(2) << gb_per_sec << " GB/s (Read)"
                  << std::endl;
    }

    void RunBenchmarkFP16(const std::string &name, int M, int K, int iters = 100)
    {
        auto tensor = generate_random_fp16_activations(M, K);

        // Warmup
        for (int i = 0; i < 10; ++i)
        {
            auto pack = tensor->to_int8_activation_pack(M, K);
            (void)pack;
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            auto pack = tensor->to_int8_activation_pack(M, K);
            if (pack.data.empty())
                std::cerr << "Empty pack\n";
        }
        auto end = high_resolution_clock::now();

        double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avg_ms = total_ms / iters;

        // FP16 is 2 bytes per element
        double bytes_processed = static_cast<double>(M) * K * 2.0;
        double gb_per_sec = (bytes_processed * iters) / (total_ms * 1e6);

        std::cout << std::left << std::setw(40) << name
                  << " M=" << std::setw(5) << M
                  << " K=" << std::setw(5) << K
                  << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " | Throughput: " << std::setprecision(2) << gb_per_sec << " GB/s (Read)"
                  << std::endl;
    }

    void RunBenchmarkBF16(const std::string &name, int M, int K, int iters = 100)
    {
        auto tensor = generate_random_bf16_activations(M, K);

        // Warmup
        for (int i = 0; i < 10; ++i)
        {
            auto pack = tensor->to_int8_activation_pack(M, K);
            (void)pack;
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            auto pack = tensor->to_int8_activation_pack(M, K);
            if (pack.data.empty())
                std::cerr << "Empty pack\n";
        }
        auto end = high_resolution_clock::now();

        double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avg_ms = total_ms / iters;

        // BF16 is 2 bytes per element
        double bytes_processed = static_cast<double>(M) * K * 2.0;
        double gb_per_sec = (bytes_processed * iters) / (total_ms * 1e6);

        std::cout << std::left << std::setw(40) << name
                  << " M=" << std::setw(5) << M
                  << " K=" << std::setw(5) << K
                  << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " | Throughput: " << std::setprecision(2) << gb_per_sec << " GB/s (Read)"
                  << std::endl;
    }

    void RunBenchmarkFromInt32(const std::string &name, int M, int K, int iters = 100)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::vector<int32_t> accum(M * K, 1000);
        std::vector<float> row_scales(M, 0.01f);
        std::vector<float> col_scales(K, 0.01f);
        std::vector<float> bias(K, 0.1f);

        // Warmup
        for (int i = 0; i < 10; ++i)
        {
            tensor->from_int32_with_scales(accum.data(), M, K, row_scales.data(), col_scales.data(), bias.data());
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            tensor->from_int32_with_scales(accum.data(), M, K, row_scales.data(), col_scales.data(), bias.data());
        }
        auto end = high_resolution_clock::now();

        double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avg_ms = total_ms / iters;

        // Read: M*K * 4 bytes (INT32)
        // Write: M*K * 4 bytes (FP32)
        double bytes_processed = static_cast<double>(M) * K * 8.0;
        double gb_per_sec = (bytes_processed * iters) / (total_ms * 1e6);

        std::cout << std::left << std::setw(40) << name
                  << " M=" << std::setw(5) << M
                  << " K=" << std::setw(5) << K
                  << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " | Throughput: " << std::setprecision(2) << gb_per_sec << " GB/s (R+W)"
                  << std::endl;
    }

    void RunBenchmarkFromInt32FP16(const std::string &name, int M, int K, int iters = 100)
    {
        auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::vector<int32_t> accum(M * K, 1000);
        std::vector<float> row_scales(M, 0.01f);
        std::vector<float> col_scales(K, 0.01f);
        std::vector<float> bias(K, 0.1f);

        // Warmup
        for (int i = 0; i < 10; ++i)
        {
            tensor->from_int32_with_scales(accum.data(), M, K, row_scales.data(), col_scales.data(), bias.data());
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            tensor->from_int32_with_scales(accum.data(), M, K, row_scales.data(), col_scales.data(), bias.data());
        }
        auto end = high_resolution_clock::now();

        double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avg_ms = total_ms / iters;

        // Read: M*K * 4 bytes (INT32)
        // Write: M*K * 2 bytes (FP16)
        double bytes_processed = static_cast<double>(M) * K * 6.0;
        double gb_per_sec = (bytes_processed * iters) / (total_ms * 1e6);

        std::cout << std::left << std::setw(40) << name
                  << " M=" << std::setw(5) << M
                  << " K=" << std::setw(5) << K
                  << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " | Throughput: " << std::setprecision(2) << gb_per_sec << " GB/s (R+W)"
                  << std::endl;
    }

    void RunBenchmarkFromInt32BF16(const std::string &name, int M, int K, int iters = 100)
    {
        auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::vector<int32_t> accum(M * K, 1000);
        std::vector<float> row_scales(M, 0.01f);
        std::vector<float> col_scales(K, 0.01f);
        std::vector<float> bias(K, 0.1f);

        // Warmup
        for (int i = 0; i < 10; ++i)
        {
            tensor->from_int32_with_scales(accum.data(), M, K, row_scales.data(), col_scales.data(), bias.data());
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            tensor->from_int32_with_scales(accum.data(), M, K, row_scales.data(), col_scales.data(), bias.data());
        }
        auto end = high_resolution_clock::now();

        double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avg_ms = total_ms / iters;

        // Read: M*K * 4 bytes (INT32)
        // Write: M*K * 2 bytes (BF16)
        double bytes_processed = static_cast<double>(M) * K * 6.0;
        double gb_per_sec = (bytes_processed * iters) / (total_ms * 1e6);

        std::cout << std::left << std::setw(40) << name
                  << " M=" << std::setw(5) << M
                  << " K=" << std::setw(5) << K
                  << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " | Throughput: " << std::setprecision(2) << gb_per_sec << " GB/s (R+W)"
                  << std::endl;
    }
};

TEST_F(OneDNNConversionPerf, FP32_to_INT8_Small)
{
    RunBenchmark("FP32->INT8 (Small)", 32, 4096);
}

TEST_F(OneDNNConversionPerf, FP32_to_INT8_Medium)
{
    RunBenchmark("FP32->INT8 (Medium)", 128, 4096);
}

TEST_F(OneDNNConversionPerf, FP32_to_INT8_Large)
{
    RunBenchmark("FP32->INT8 (Large)", 1024, 4096); // Prefill-like
}

TEST_F(OneDNNConversionPerf, FP32_to_INT8_Huge)
{
    RunBenchmark("FP32->INT8 (Huge)", 4096, 4096);
}

TEST_F(OneDNNConversionPerf, FP16_to_INT8_Small)
{
    RunBenchmarkFP16("FP16->INT8 (Small)", 32, 4096);
}

TEST_F(OneDNNConversionPerf, FP16_to_INT8_Medium)
{
    RunBenchmarkFP16("FP16->INT8 (Medium)", 128, 4096);
}

TEST_F(OneDNNConversionPerf, FP16_to_INT8_Large)
{
    RunBenchmarkFP16("FP16->INT8 (Large)", 1024, 4096);
}

TEST_F(OneDNNConversionPerf, FP16_to_INT8_Huge)
{
    RunBenchmarkFP16("FP16->INT8 (Huge)", 4096, 4096);
}

TEST_F(OneDNNConversionPerf, BF16_to_INT8_Small)
{
    RunBenchmarkBF16("BF16->INT8 (Small)", 32, 4096);
}

TEST_F(OneDNNConversionPerf, BF16_to_INT8_Medium)
{
    RunBenchmarkBF16("BF16->INT8 (Medium)", 128, 4096);
}

TEST_F(OneDNNConversionPerf, BF16_to_INT8_Large)
{
    RunBenchmarkBF16("BF16->INT8 (Large)", 1024, 4096);
}

TEST_F(OneDNNConversionPerf, BF16_to_INT8_Huge)
{
    RunBenchmarkBF16("BF16->INT8 (Huge)", 4096, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_FP32_Small)
{
    RunBenchmarkFromInt32("INT32->FP32 (Small)", 32, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_FP32_Medium)
{
    RunBenchmarkFromInt32("INT32->FP32 (Medium)", 128, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_FP32_Large)
{
    RunBenchmarkFromInt32("INT32->FP32 (Large)", 1024, 4096); // Prefill-like
}

TEST_F(OneDNNConversionPerf, INT32_to_FP32_Huge)
{
    RunBenchmarkFromInt32("INT32->FP32 (Huge)", 4096, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_FP16_Small)
{
    RunBenchmarkFromInt32FP16("INT32->FP16 (Small)", 32, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_FP16_Medium)
{
    RunBenchmarkFromInt32FP16("INT32->FP16 (Medium)", 128, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_FP16_Large)
{
    RunBenchmarkFromInt32FP16("INT32->FP16 (Large)", 1024, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_FP16_Huge)
{
    RunBenchmarkFromInt32FP16("INT32->FP16 (Huge)", 4096, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_BF16_Small)
{
    RunBenchmarkFromInt32BF16("INT32->BF16 (Small)", 32, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_BF16_Medium)
{
    RunBenchmarkFromInt32BF16("INT32->BF16 (Medium)", 128, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_BF16_Large)
{
    RunBenchmarkFromInt32BF16("INT32->BF16 (Large)", 1024, 4096);
}

TEST_F(OneDNNConversionPerf, INT32_to_BF16_Huge)
{
    RunBenchmarkFromInt32BF16("INT32->BF16 (Huge)", 4096, 4096);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}

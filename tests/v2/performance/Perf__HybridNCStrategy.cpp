/**
 * @file Perf__HybridNCStrategy.cpp
 * @brief Benchmark to validate hybrid NC strategy (adaptive NC selection)
 *
 * Tests both small batch (NC=128) and large batch (NC=64) configurations
 * to verify the adaptive strategy selects optimal NC based on batch size.
 *
 * @author David Sanftenberg
 */

#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/QuantizedGemm.h"
#include "kernels/cpu/QuantizedGemmL1Opt.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace llaminar2;

class HybridNCStrategyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Load model once
        std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
        loader_ = std::make_unique<ModelLoader>();

        if (rank_ == 0)
        {
            std::cout << "[Hybrid NC Test] Loading model: " << model_path << std::endl;
        }

        if (!loader_->loadModel(model_path))
        {
            if (rank_ == 0)
            {
                std::cerr << "[Hybrid NC Test] Failed to load model" << std::endl;
            }
            GTEST_SKIP() << "Model not available";
        }

        if (rank_ == 0)
        {
            std::cout << "[Hybrid NC Test] Model loaded successfully\n" << std::endl;
        }
    }

    std::shared_ptr<IQ4_NLTensor> getWeightTensor()
    {
        auto weight_base = loader_->loadTensor("blk.0.attn_q.weight", -1);
        if (!weight_base)
        {
            return nullptr;
        }
        return std::dynamic_pointer_cast<IQ4_NLTensor>(weight_base);
    }

    std::shared_ptr<FP32Tensor> createFP32Activation(int m, int k)
    {
        auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)});
        float *data = tensor->mutable_data();
        for (int i = 0; i < m * k; ++i)
        {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        return tensor;
    }

    struct BenchResult
    {
        double mean_ms;
        double stddev_ms;
        double mean_gflops;
        double cv_percent;
    };

    BenchResult benchmarkKernel(ITensorGemm *gemm, const float *A, float *C,
                                int m, int n, int k,
                                int num_trials = 5, int num_iters = 50)
    {
        std::vector<double> times_ms;

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            gemm->multiply(A, C, m, n, k, true, 1.0f, 0.0f, nullptr, -1);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        for (int trial = 0; trial < num_trials; ++trial)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < num_iters; ++iter)
            {
                gemm->multiply(A, C, m, n, k, true, 1.0f, 0.0f, nullptr, -1);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            times_ms.push_back(elapsed / num_iters);
        }

        // Compute statistics
        double sum = 0.0, sum_sq = 0.0;
        for (double t : times_ms)
        {
            sum += t;
            sum_sq += t * t;
        }

        double mean_ms = sum / num_trials;
        double variance = (sum_sq / num_trials) - (mean_ms * mean_ms);
        double stddev_ms = std::sqrt(std::max(0.0, variance));

        double flops = 2.0 * m * n * k;
        double mean_gflops = flops / (mean_ms * 1e6);
        double cv_percent = (stddev_ms / mean_ms) * 100.0;

        return {mean_ms, stddev_ms, mean_gflops, cv_percent};
    }

    int rank_;
    int world_size_;
    std::unique_ptr<ModelLoader> loader_;
};

TEST_F(HybridNCStrategyTest, SmallBatch_NC128)
{
    // Small batch (m=64) should use NC=128 for better throughput
    const int m = 64;
    const int n = 896;
    const int k = 896;

    auto weight = getWeightTensor();
    auto activation = createFP32Activation(m, k);
    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});

    const float *A = activation->data();
    float *C = output->mutable_data();

    QuantizedGemmL1Opt kernel(weight.get());

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ Hybrid NC Strategy - Small Batch (64×896×896)                 ║" << std::endl;
        std::cout << "║ Expected: NC=128 (NC_LARGE) for better throughput             ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;
    }

    auto result = benchmarkKernel(&kernel, A, C, m, n, k);

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Time:       " << result.mean_ms << " ms" << std::endl;
        std::cout << "Throughput: " << result.mean_gflops << " GFLOPS" << std::endl;
        std::cout << "CV:         " << result.cv_percent << "%" << std::endl;
    }
}

TEST_F(HybridNCStrategyTest, LargeBatch_NC64)
{
    // Large batch (m=512) should use NC=64 for better L1 locality
    const int m = 512;
    const int n = 896;
    const int k = 896;

    auto weight = getWeightTensor();
    auto activation = createFP32Activation(m, k);
    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});

    const float *A = activation->data();
    float *C = output->mutable_data();

    QuantizedGemmL1Opt kernel(weight.get());

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ Hybrid NC Strategy - Large Batch (512×896×896)                ║" << std::endl;
        std::cout << "║ Expected: NC=64 (NC_SMALL) for better L1 cache locality       ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;
    }

    auto result = benchmarkKernel(&kernel, A, C, m, n, k);

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Time:       " << result.mean_ms << " ms" << std::endl;
        std::cout << "Throughput: " << result.mean_gflops << " GFLOPS" << std::endl;
        std::cout << "CV:         " << result.cv_percent << "%" << std::endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

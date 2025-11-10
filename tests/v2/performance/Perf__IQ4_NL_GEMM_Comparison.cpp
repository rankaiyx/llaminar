/**
 * @file Perf__IQ4_NL_GEMM_Comparison.cpp
 * @brief Compare IQ4_NLQuantizedGemm vs AutoTunedQuantizedGemm performance
 *
 * This benchmark directly compares two GEMM implementations for IQ4_NL tensors:
 * 1. IQ4_NLQuantizedGemm: Specialized implementation with adaptive cache tiling
 * 2. AutoTunedQuantizedGemm: Generic auto-tuned implementation via ITensorGemmTileDataProvider
 *
 * Goal: Determine which implementation performs better for IQ4_NL tensors across
 * various workload sizes (single token, small batch, medium batch, large batch).
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// V2 includes
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"
#include "backends/ComputeBackend.h"
#include "kernels/cpu/gemm/quantized/QuantizedGemm.h" // For createQuantizedGemm

using namespace llaminar2;

/**
 * @brief Benchmark result for a single GEMM implementation
 */
struct GemmBenchmark
{
    std::string name;        ///< Implementation name
    double time_ms;          ///< Average time per iteration (ms)
    double gflops;           ///< Throughput (GFLOPS)
    double relative_speedup; ///< Relative speedup vs baseline (1.0 = baseline)
};

/**
 * @brief IQ4_NL GEMM Comparison Test Fixture
 */
class IQ4_NL_GEMM_Comparison : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::unique_ptr<ModelLoader> loader_;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        DeviceManager::instance().initialize(-1);

        // Verify OpenMP configured
        int max_threads = omp_get_max_threads();
        if (rank_ == 0)
        {
            std::cout << "[Comparison] OpenMP threads: " << max_threads << std::endl;
        }
        ASSERT_GT(max_threads, 1) << "OpenMP not configured!";

        // Load model
        std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
        try
        {
            loader_ = std::make_unique<ModelLoader>();
            if (!loader_->loadModel(model_path))
            {
                throw std::runtime_error("Failed to load model");
            }
            if (rank_ == 0)
            {
                std::cout << "[Comparison] Model loaded: " << model_path << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            if (rank_ == 0)
            {
                std::cerr << "[Comparison] Model unavailable: " << e.what() << std::endl;
            }
            GTEST_SKIP() << "Model not available: " << model_path;
        }
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        loader_.reset();
    }

    /**
     * @brief Get a weight tensor from the loaded model (layer 0, Q projection)
     */
    std::shared_ptr<IQ4_NLTensor> getWeightTensor()
    {
        if (!loader_)
        {
            throw std::runtime_error("Model loader not initialized");
        }

        // Get first layer Q projection weight
        // Format: "blk.0.attn_q.weight"
        std::string weight_name = "blk.0.attn_q.weight";

        auto weight = loader_->loadTensor(weight_name, -1); // -1 = CPU
        if (!weight)
        {
            throw std::runtime_error("Failed to load weight tensor: " + weight_name);
        }

        // Cast to IQ4_NLTensor
        auto iq4nl_weight = std::dynamic_pointer_cast<IQ4_NLTensor>(weight);
        if (!iq4nl_weight)
        {
            throw std::runtime_error("Weight is not IQ4_NL format: " + weight_name);
        }

        return iq4nl_weight;
    }

    /**
     * @brief Create FP32 activation tensor
     */
    std::shared_ptr<FP32Tensor> createActivation(int seq_len, int features)
    {
        auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(seq_len),
            static_cast<size_t>(features)});

        // Initialize with realistic values
        float *data = tensor->mutable_data();
        const auto &shape = tensor->shape();
        size_t total_elements = shape[0] * shape[1];
        for (size_t i = 0; i < total_elements; ++i)
        {
            data[i] = static_cast<float>(i % 100) / 100.0f - 0.5f;
        }

        return tensor;
    }

    /**
     * @brief Benchmark a single GEMM implementation with MPI distribution
     *
     * Each rank computes half the output rows (m/2), then Allgather at the end.
     */
    GemmBenchmark benchmarkGemm(
        const std::string &name,
        std::unique_ptr<ITensorGemm> &gemm,
        int m, int n, int k,
        int warmup_iters,
        int bench_iters)
    {
        // Distribute rows across ranks
        int local_m = m / world_size_;
        int local_start = rank_ * local_m;

        // Create tensors (local portion for this rank)
        auto activation = createActivation(local_m, k);
        auto local_output = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(local_m),
            static_cast<size_t>(n)});

        // Full output for Allgather
        auto full_output = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(m),
            static_cast<size_t>(n)});

        // Warmup
        for (int i = 0; i < warmup_iters; ++i)
        {
            bool success = gemm->multiply(
                activation->data(),
                local_output->mutable_data(),
                local_m, n, k,
                true, // transpose_B
                1.0f, // alpha
                0.0f, // beta
                nullptr,
                -1);
            if (!success)
            {
                throw std::runtime_error("GEMM failed during warmup: " + name);
            }

            // Allgather results (warmup MPI too)
            MPI_Allgather(
                local_output->data(), local_m * n, MPI_FLOAT,
                full_output->mutable_data(), local_m * n, MPI_FLOAT,
                MPI_COMM_WORLD);
        }

        // Benchmark
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < bench_iters; ++i)
        {
            // Compute local portion
            gemm->multiply(
                activation->data(),
                local_output->mutable_data(),
                local_m, n, k,
                true, 1.0f, 0.0f, nullptr, -1);

            // Allgather to reconstruct full result
            MPI_Allgather(
                local_output->data(), local_m * n, MPI_FLOAT,
                full_output->mutable_data(), local_m * n, MPI_FLOAT,
                MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_time_ms = elapsed_ms / bench_iters;

        // Calculate GFLOPS
        double flops = 2.0 * m * n * k;
        double gflops = (flops / avg_time_ms) / 1e6;

        return GemmBenchmark{
            .name = name,
            .time_ms = avg_time_ms,
            .gflops = gflops,
            .relative_speedup = 1.0 // Will be filled in by caller
        };
    }

    /**
     * @brief Compare both implementations and print results
     */
    void compareImplementations(int m, int n, int k, const std::string &description)
    {
        if (rank_ != 0)
        {
            // Non-rank-0 processes participate in computation but don't print
            auto weight = getWeightTensor();
            auto specialized_gemm = weight->createGemm();
            auto generic_gemm = createQuantizedGemm(weight.get());

            benchmarkGemm("IQ4_NLQuantizedGemm", specialized_gemm, m, n, k, 5, 50);
            benchmarkGemm("AutoTunedQuantizedGemm", generic_gemm, m, n, k, 5, 50);
            return;
        }

        // Rank 0: benchmark and print results
        auto weight = getWeightTensor();

        // Create both implementations
        auto specialized_gemm = weight->createGemm();          // IQ4_NLQuantizedGemm
        auto generic_gemm = createQuantizedGemm(weight.get()); // AutoTunedQuantizedGemm

        // Benchmark both
        auto specialized_result = benchmarkGemm("IQ4_NLQuantizedGemm (Specialized)", specialized_gemm, m, n, k, 5, 50);
        auto generic_result = benchmarkGemm("AutoTunedQuantizedGemm (Generic)", generic_gemm, m, n, k, 5, 50);

        // Calculate relative speedups (baseline = specialized)
        specialized_result.relative_speedup = 1.0;
        generic_result.relative_speedup = specialized_result.time_ms / generic_result.time_ms;

        // Print comparison
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(62) << description << " ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Configuration: [" << m << " × " << n << "] @ k=" << k << std::string(62 - 24 - std::to_string(m).length() - std::to_string(n).length() - std::to_string(k).length(), ' ') << "║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Implementation                    Time (ms)   GFLOPS  Speedup  ║\n";
        std::cout << "╟────────────────────────────────────────────────────────────────╢\n";

        // Print specialized (baseline)
        std::cout << "║ " << std::left << std::setw(30) << specialized_result.name << "  "
                  << std::right << std::setw(7) << std::fixed << std::setprecision(2) << specialized_result.time_ms << "   "
                  << std::setw(7) << std::setprecision(1) << specialized_result.gflops << "  "
                  << std::setw(6) << std::setprecision(2) << specialized_result.relative_speedup << "x ║\n";

        // Print generic with comparison
        std::cout << "║ " << std::left << std::setw(30) << generic_result.name << "  "
                  << std::right << std::setw(7) << std::fixed << std::setprecision(2) << generic_result.time_ms << "   "
                  << std::setw(7) << std::setprecision(1) << generic_result.gflops << "  "
                  << std::setw(6) << std::setprecision(2) << generic_result.relative_speedup << "x ║\n";

        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

        // Winner annotation
        if (generic_result.relative_speedup > 1.05)
        {
            std::cout << "✅ Winner: AutoTunedQuantizedGemm ("
                      << std::setprecision(1) << (generic_result.relative_speedup - 1.0) * 100.0
                      << "% faster)\n";
        }
        else if (specialized_result.relative_speedup > 1.05)
        {
            std::cout << "✅ Winner: IQ4_NLQuantizedGemm ("
                      << std::setprecision(1) << (1.0 / generic_result.relative_speedup - 1.0) * 100.0
                      << "% faster)\n";
        }
        else
        {
            std::cout << "🟰 Tie: Performance within 5% margin\n";
        }
        std::cout << std::endl;
    }
};

// =============================================================================
// Comparison Test Cases
// =============================================================================

TEST_F(IQ4_NL_GEMM_Comparison, SingleToken_QProjection)
{
    compareImplementations(
        1,   // m = single token
        896, // n = d_model (Q-projection output)
        896, // k = d_model (Q-projection input)
        "Single Token Decode (1×896×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, SmallBatch_QProjection)
{
    compareImplementations(
        32,  // m = small batch
        896, // n = d_model
        896, // k = d_model
        "Small Batch (32×896×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, MediumBatch_QProjection)
{
    compareImplementations(
        128, // m = medium batch
        896, // n = d_model
        896, // k = d_model
        "Medium Batch (128×896×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, LargeBatch_QProjection)
{
    compareImplementations(
        512, // m = large batch
        896, // n = d_model
        896, // k = d_model
        "Large Batch (512×896×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, SmallBatch_FFN_Up)
{
    compareImplementations(
        16,   // m = small batch
        4864, // n = intermediate_size (FFN up projection)
        896,  // k = d_model
        "Small Batch FFN-Up (16×4864×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, MediumBatch_FFN_Up)
{
    compareImplementations(
        128,  // m = medium batch
        4864, // n = intermediate_size
        896,  // k = d_model
        "Medium Batch FFN-Up (128×4864×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, LargeBatch_FFN_Down)
{
    compareImplementations(
        256,  // m = large batch
        896,  // n = d_model (FFN down projection)
        4864, // k = intermediate_size
        "Large Batch FFN-Down (256×896×4864)");
}

// ============================================================================
// Prefill Benchmarks (Long Context)
// ============================================================================

TEST_F(IQ4_NL_GEMM_Comparison, Prefill_1024_QProjection)
{
    compareImplementations(
        1024, // m = 1K token prefill
        896,  // n = d_model
        896,  // k = d_model
        "Prefill 1024 Tokens Q-Projection (1024×896×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, Prefill_2048_QProjection)
{
    compareImplementations(
        2048, // m = 2K token prefill
        896,  // n = d_model
        896,  // k = d_model
        "Prefill 2048 Tokens Q-Projection (2048×896×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, Prefill_4096_QProjection)
{
    compareImplementations(
        4096, // m = 4K token prefill
        896,  // n = d_model
        896,  // k = d_model
        "Prefill 4096 Tokens Q-Projection (4096×896×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, Prefill_1024_FFN_Up)
{
    compareImplementations(
        1024, // m = 1K token prefill
        4864, // n = intermediate_size
        896,  // k = d_model
        "Prefill 1024 Tokens FFN-Up (1024×4864×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, Prefill_2048_FFN_Up)
{
    compareImplementations(
        2048, // m = 2K token prefill
        4864, // n = intermediate_size
        896,  // k = d_model
        "Prefill 2048 Tokens FFN-Up (2048×4864×896)");
}

TEST_F(IQ4_NL_GEMM_Comparison, Prefill_4096_FFN_Up)
{
    compareImplementations(
        4096, // m = 4K token prefill
        4864, // n = intermediate_size
        896,  // k = d_model
        "Prefill 4096 Tokens FFN-Up (4096×4864×896)");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}

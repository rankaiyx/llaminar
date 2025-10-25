/**
 * @file Perf__Qwen2E2EPerformance.cpp
 * @brief End-to-end performance benchmarks for Qwen2Pipeline (Phase 3c)
 * @author David Sanftenberg
 *
 * Measures full pipeline performance including:
 * - Model loading time
 * - Prefill throughput (tokens/sec)
 * - Decode throughput (tokens/sec)
 * - MPI tensor-parallel speedup
 * - Memory usage
 *
 * Requirements:
 * - Real Qwen 2.5 0.5B model
 * - MPI support (1 rank and 2 rank comparison)
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <memory>
#include <vector>
#include <iomanip>

#include "../../../src/v2/loaders/ModelContext.h"
#include "../../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../../src/v2/utils/MPIContext.h"
#include "../../../src/v2/utils/Logger.h"

using namespace llaminar2;
using namespace std::chrono;

/**
 * @brief Test fixture for Qwen2 end-to-end performance
 */
class Qwen2E2EPerformance : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        if (world_size_ > 1)
        {
            mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
        }

        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_.reset();
    }

    struct BenchmarkResult
    {
        double load_time_ms = 0.0;
        double prefill_time_ms = 0.0;
        double decode_time_ms = 0.0;
        int prefill_tokens = 0;
        int decode_tokens = 0;
        double prefill_throughput = 0.0; // tokens/sec
        double decode_throughput = 0.0;  // tokens/sec
        int world_size = 1;
    };

    void printBenchmarkResult(const BenchmarkResult &result)
    {
        if (rank_ != 0)
            return;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ QWEN 2.5 0.5B E2E PERFORMANCE (Ranks: " << std::setw(2) << result.world_size << ")           ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ MODEL LOADING                                                ║\n";
        std::cout << "║   Time:          " << std::setw(8) << result.load_time_ms << " ms                              ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ PREFILL PHASE                                                ║\n";
        std::cout << "║   Tokens:        " << std::setw(8) << result.prefill_tokens << " tokens                          ║\n";
        std::cout << "║   Time:          " << std::setw(8) << result.prefill_time_ms << " ms                              ║\n";
        std::cout << "║   Throughput:    " << std::setw(8) << result.prefill_throughput << " tok/s                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ DECODE PHASE                                                 ║\n";
        std::cout << "║   Tokens:        " << std::setw(8) << result.decode_tokens << " tokens                          ║\n";
        std::cout << "║   Time:          " << std::setw(8) << result.decode_time_ms << " ms                              ║\n";
        std::cout << "║   Throughput:    " << std::setw(8) << result.decode_throughput << " tok/s                          ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<ModelContext> model_ctx_;
    std::string model_path_;
    int rank_ = 0;
    int world_size_ = 1;
};

/**
 * @brief Benchmark: Model loading time
 *
 * Measures time to load GGUF model from disk.
 */
TEST_F(Qwen2E2EPerformance, ModelLoadingTime)
{
    BenchmarkResult result;
    result.world_size = world_size_;

    auto t0 = high_resolution_clock::now();

    model_ctx_ = ModelContext::create(model_path_);
    ASSERT_TRUE(model_ctx_) << "Failed to load model: " << model_path_;

    auto t1 = high_resolution_clock::now();
    result.load_time_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;

    if (rank_ == 0)
    {
        const auto &model = model_ctx_->model();
        std::cout << "[Perf] Model loaded in " << result.load_time_ms << " ms" << std::endl;
        std::cout << "[Perf] n_layers=" << model.block_count
                  << " n_heads=" << model.head_count
                  << " d_model=" << model.embedding_length
                  << " vocab_size=" << model.vocab_size << std::endl;
    }
}

/**
 * @brief Benchmark: Single token decode throughput
 *
 * Measures pure decode performance (1 token → 1 token).
 * This is the typical autoregressive generation scenario.
 */
TEST_F(Qwen2E2EPerformance, SingleTokenDecode)
{
    // Load model
    model_ctx_ = ModelContext::create(model_path_);
    ASSERT_TRUE(model_ctx_);

    // Create pipeline
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr);

    BenchmarkResult result;
    result.world_size = world_size_;
    result.decode_tokens = 50; // Benchmark 50 decode steps

    // Warmup: Single token
    std::vector<int> tokens = {151644}; // BOS
    bool success = pipeline->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success);

    if (mpi_ctx_)
        mpi_ctx_->barrier();

    // Benchmark: 50 single-token decode steps
    auto t0 = high_resolution_clock::now();

    for (int i = 0; i < result.decode_tokens; ++i)
    {
        tokens[0] = 151644 + i; // Varying token
        success = pipeline->forward(tokens.data(), 1);
        ASSERT_TRUE(success);
    }

    if (mpi_ctx_)
        mpi_ctx_->barrier();
    auto t1 = high_resolution_clock::now();

    result.decode_time_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
    result.decode_throughput = (result.decode_tokens * 1000.0) / result.decode_time_ms;

    if (rank_ == 0)
    {
        std::cout << "[Perf] Decode: " << result.decode_tokens << " tokens in "
                  << result.decode_time_ms << " ms ("
                  << result.decode_throughput << " tok/s)" << std::endl;
    }
}

/**
 * @brief Benchmark: Multi-token prefill throughput
 *
 * Measures prefill performance with various prompt lengths.
 * Tests scaling with sequence length.
 */
TEST_F(Qwen2E2EPerformance, MultiTokenPrefill)
{
    // Load model
    model_ctx_ = ModelContext::create(model_path_);
    ASSERT_TRUE(model_ctx_);

    // Create pipeline
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr);

    // Test various prefill lengths
    std::vector<int> prefill_lengths = {8, 32, 128, 512};

    for (int seq_len : prefill_lengths)
    {
        BenchmarkResult result;
        result.world_size = world_size_;
        result.prefill_tokens = seq_len;

        // Create token sequence
        std::vector<int> tokens(seq_len);
        for (int i = 0; i < seq_len; ++i)
        {
            tokens[i] = 151644 + (i % 1000);
        }

        // Warmup
        bool success = pipeline->forward(tokens.data(), tokens.size());
        ASSERT_TRUE(success);

        if (mpi_ctx_)
            mpi_ctx_->barrier();

        // Benchmark: 3 iterations
        auto t0 = high_resolution_clock::now();

        for (int iter = 0; iter < 3; ++iter)
        {
            success = pipeline->forward(tokens.data(), tokens.size());
            ASSERT_TRUE(success);
        }

        if (mpi_ctx_)
            mpi_ctx_->barrier();
        auto t1 = high_resolution_clock::now();

        result.prefill_time_ms = duration_cast<microseconds>(t1 - t0).count() / (3.0 * 1000.0);
        result.prefill_throughput = (result.prefill_tokens * 1000.0) / result.prefill_time_ms;

        if (rank_ == 0)
        {
            std::cout << "[Perf] Prefill (" << seq_len << " tokens): "
                      << result.prefill_time_ms << " ms ("
                      << result.prefill_throughput << " tok/s)" << std::endl;
        }
    }
}

/**
 * @brief Benchmark: MPI tensor-parallel speedup
 *
 * Compares single-rank vs multi-rank performance.
 * Measures actual speedup from tensor parallelization.
 *
 * NOTE: Requires running test twice (once with 1 rank, once with 2 ranks).
 */
TEST_F(Qwen2E2EPerformance, DISABLED_TensorParallelSpeedup)
{
    // This test requires manual execution with different MPI configurations
    // Run separately and compare results
    GTEST_SKIP() << "Manual comparison test (run with 1 rank, then 2 ranks)";
}

/**
 * @brief Benchmark: Full generation pipeline
 *
 * End-to-end test: prefill + autoregressive decode.
 * Measures total time for realistic generation scenario.
 */
TEST_F(Qwen2E2EPerformance, DISABLED_FullGeneration)
{
    // Disabled until KV cache is implemented
    GTEST_SKIP() << "KV cache required for autoregressive decode";
}

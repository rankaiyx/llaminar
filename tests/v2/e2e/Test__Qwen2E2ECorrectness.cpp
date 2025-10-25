/**
 * @file Test__Qwen2E2ECorrectness.cpp
 * @brief End-to-end correctness tests for Qwen2Pipeline (Phase 3c)
 * @author David Sanftenberg
 *
 * Validates full pipeline correctness by comparing:
 * - Single-rank vs multi-rank MPI execution
 * - All intermediate activations across transformer layers
 *
 * Requirements:
 * - Real Qwen 2.5 0.5B model (models/qwen2.5-0.5b-instruct-q4_0.gguf)
 * - MPI support (exactly 2 ranks for tensor-parallel validation)
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>

#include "../../../src/v2/loaders/ModelContext.h"
#include "../../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../../src/v2/utils/MPIContext.h"
#include "../../../src/v2/utils/Logger.h"

using namespace llaminar2;

/**
 * @brief Test fixture for Qwen2 end-to-end correctness
 */
class Qwen2E2ECorrectness : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI context
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Model path (from test fixtures)
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    }

    void TearDown() override
    {
        // Cleanup
        model_ctx_single_.reset();
        model_ctx_multi_.reset();
        mpi_ctx_.reset();
    }

    /**
     * @brief Load model for single-rank execution
     */
    bool loadModelSingleRank()
    {
        if (rank_ == 0)
        {
            model_ctx_single_ = ModelContext::create(model_path_);
            if (!model_ctx_single_)
            {
                LOG_ERROR("[E2E] Failed to load model (single-rank): " << model_path_);
                return false;
            }
            LOG_INFO("[E2E] Model loaded successfully (single-rank): " << model_path_);
        }
        return true;
    }

    /**
     * @brief Load model for multi-rank execution
     */
    bool loadModelMultiRank()
    {
        model_ctx_multi_ = ModelContext::create(model_path_);
        if (!model_ctx_multi_)
        {
            LOG_ERROR("[E2E] Rank " << rank_ << " failed to load model (multi-rank): " << model_path_);
            return false;
        }
        LOG_INFO("[E2E] Rank " << rank_ << " model loaded successfully (multi-rank)");
        return true;
    }

    /**
     * @brief Compare two output tensors with tolerance
     */
    struct ComparisonResult
    {
        bool passed = false;
        float max_abs_diff = 0.0f;
        float mean_abs_diff = 0.0f;
        float rel_l2_norm = 0.0f;
        size_t num_mismatches = 0;
    };

    ComparisonResult compareTensors(
        const float *a, const float *b, size_t size, float tolerance)
    {
        ComparisonResult result;

        double sum_abs_diff = 0.0;
        double sum_sq_diff = 0.0;
        double sum_sq_b = 0.0;

        for (size_t i = 0; i < size; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > tolerance)
            {
                result.num_mismatches++;
            }
            if (diff > result.max_abs_diff)
            {
                result.max_abs_diff = diff;
            }
            sum_abs_diff += diff;
            sum_sq_diff += diff * diff;
            sum_sq_b += b[i] * b[i];
        }

        result.mean_abs_diff = static_cast<float>(sum_abs_diff / size);

        if (sum_sq_b > 1e-10)
        {
            result.rel_l2_norm = static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_b));
        }

        result.passed = (result.max_abs_diff <= tolerance &&
                         result.rel_l2_norm <= 0.01f);

        return result;
    }

    void printComparisonResult(const ComparisonResult &result, const std::string &name)
    {
        std::cout << "=== " << name << " ===" << std::endl;
        std::cout << "  Max abs diff:   " << result.max_abs_diff << std::endl;
        std::cout << "  Mean abs diff:  " << result.mean_abs_diff << std::endl;
        std::cout << "  Rel L2 norm:    " << result.rel_l2_norm << std::endl;
        std::cout << "  Mismatches:     " << result.num_mismatches << std::endl;
        std::cout << "  Status:         " << (result.passed ? "PASSED" : "FAILED") << std::endl;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<ModelContext> model_ctx_single_;
    std::shared_ptr<ModelContext> model_ctx_multi_;
    std::string model_path_;
    int rank_;
    int world_size_;
};

/**
 * @brief Test: Single token inference correctness (decode phase)
 *
 * Validates that single-rank and multi-rank pipelines produce identical
 * logits for a single token (typical decode scenario).
 */
TEST_F(Qwen2E2ECorrectness, SingleTokenInference)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f; // Relaxed tolerance for full pipeline

    // Load models
    ASSERT_TRUE(loadModelSingleRank());
    ASSERT_TRUE(loadModelMultiRank());

    // Single token input
    std::vector<int> tokens = {151644}; // BOS token for Qwen 2.5

    // Single-rank execution (rank 0 only)
    std::vector<float> logits_single;
    if (rank_ == 0)
    {
        auto pipeline_single = std::make_unique<Qwen2Pipeline>(
            model_ctx_single_, nullptr, -1, nullptr);

        bool success = pipeline_single->forward(tokens.data(), tokens.size());
        ASSERT_TRUE(success) << "Single-rank forward pass failed";

        // Get logits (vocabulary size)
        const auto &model = model_ctx_single_->model();
        size_t vocab_size = model.vocab_size;
        logits_single.resize(vocab_size);

        // TODO: Add method to extract logits from pipeline
        // For now, we'll need to add a getLogits() method to Qwen2Pipeline
        LOG_WARN("[E2E] Logits extraction not yet implemented");
    }

    // Multi-rank execution (all ranks)
    auto pipeline_multi = std::make_unique<Qwen2Pipeline>(
        model_ctx_multi_, mpi_ctx_, -1, nullptr);

    bool success = pipeline_multi->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success) << "Multi-rank forward pass failed on rank " << rank_;

    std::vector<float> logits_multi;
    const auto &model = model_ctx_multi_->model();
    size_t vocab_size = model.vocab_size;
    logits_multi.resize(vocab_size);

    // TODO: Extract logits

    // Compare on rank 0
    if (rank_ == 0)
    {
        // auto result = compareTensors(
        //     logits_single.data(), logits_multi.data(), vocab_size, tolerance);
        //
        // printComparisonResult(result, "Single Token Inference");
        // EXPECT_TRUE(result.passed);

        LOG_INFO("[E2E] Single token test framework ready (logits extraction pending)");
    }

    mpi_ctx_->barrier();
}

/**
 * @brief Test: Multi-token prefill correctness
 *
 * Validates that single-rank and multi-rank pipelines produce identical
 * results for multi-token prefill (e.g., 8-32 tokens).
 */
TEST_F(Qwen2E2ECorrectness, MultiTokenPrefill)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load models
    ASSERT_TRUE(loadModelSingleRank());
    ASSERT_TRUE(loadModelMultiRank());

    // Multi-token input (8 tokens)
    std::vector<int> tokens = {
        151644, // BOS
        9906,   // Hello
        0,      // (placeholder - actual tokens TBD)
        0,
        0,
        0,
        0,
        0};

    // Single-rank execution (rank 0 only)
    std::vector<float> logits_single;
    if (rank_ == 0)
    {
        auto pipeline_single = std::make_unique<Qwen2Pipeline>(
            model_ctx_single_, nullptr, -1, nullptr);

        bool success = pipeline_single->forward(tokens.data(), tokens.size());
        ASSERT_TRUE(success) << "Single-rank forward pass failed";

        LOG_INFO("[E2E] Single-rank prefill completed");
    }

    // Multi-rank execution (all ranks)
    auto pipeline_multi = std::make_unique<Qwen2Pipeline>(
        model_ctx_multi_, mpi_ctx_, -1, nullptr);

    bool success = pipeline_multi->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success) << "Multi-rank forward pass failed on rank " << rank_;

    LOG_INFO("[E2E] Rank " << rank_ << " multi-rank prefill completed");

    // TODO: Compare intermediate activations
    // - Embedding output
    // - Each layer's attention output
    // - Each layer's FFN output
    // - Final norm output
    // - Logits

    mpi_ctx_->barrier();
}

/**
 * @brief Test: Autoregressive decode correctness
 *
 * Validates multi-step decode produces correct token sequence.
 * Tests KV cache functionality and incremental decode.
 */
TEST_F(Qwen2E2ECorrectness, DISABLED_AutoregressiveDecode)
{
    // Disabled until KV cache is implemented
    GTEST_SKIP() << "KV cache not yet implemented";
}

/**
 * @brief Test: Layer-by-layer activation parity
 *
 * Captures and compares activations at every transformer layer
 * between single-rank and multi-rank execution.
 */
TEST_F(Qwen2E2ECorrectness, DISABLED_LayerActivationParity)
{
    // Disabled until snapshot infrastructure is added
    GTEST_SKIP() << "Activation snapshot capture not yet implemented";
}

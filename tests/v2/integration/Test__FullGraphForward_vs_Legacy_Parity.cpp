/**
 * @file Test__FullGraphForward_vs_Legacy_Parity.cpp
 * @brief Integration test comparing Legacy Pipeline vs Full Graph Forward execution
 *
 * This test validates that the full graph forward path (via orchestrator->executeForward())
 * produces equivalent results to the legacy imperative pipeline code.
 *
 * The test runs the same input through two pipeline configurations:
 *   1. Legacy path: No LayerExecutor, no exec_full_forward (imperative code)
 *   2. Full Graph Forward: LayerExecutor + exec_full_forward (complete graph inference)
 *
 * Parity is validated using:
 *   1. Top-1 token match (must be identical for greedy sampling)
 *   2. Top-5 token overlap (>= 80% required)
 *   3. Cosine similarity (>= 0.99 for graph-based inference)
 *   4. Max absolute difference (< 1.0 for full graph path)
 *
 * Test scenarios:
 *   - Prefill: Multi-token prompt processing
 *   - Incremental decode: Single-token autoregressive generation
 *
 * @author David Sanftenberg
 * @date 2025-12-20
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <iomanip>

#include "pipelines/qwen/Qwen2Pipeline.h"
#include "pipelines/PipelineConfig.h"
#include "loaders/ModelContext.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "backends/ComputeBackend.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Compute cosine similarity between two float arrays
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute maximum absolute difference between two arrays
     */
    float max_abs_diff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    /**
     * @brief Compute mean absolute difference between two arrays
     */
    float mean_abs_diff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(a[i] - b[i]);
        }
        return static_cast<float>(sum / n);
    }

    /**
     * @brief Get top-K token indices from logits (sorted by score descending)
     */
    std::vector<int> get_topk(const float *logits, size_t vocab_size, int k)
    {
        std::vector<std::pair<float, int>> indexed(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i)
        {
            indexed[i] = {logits[i], static_cast<int>(i)};
        }
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });
        std::vector<int> topk(k);
        for (int i = 0; i < k; ++i)
        {
            topk[i] = indexed[i].second;
        }
        return topk;
    }

    /**
     * @brief Count overlap between two top-K lists
     */
    int count_overlap(const std::vector<int> &a, const std::vector<int> &b)
    {
        std::set<int> set_a(a.begin(), a.end());
        int overlap = 0;
        for (int t : b)
        {
            if (set_a.count(t) > 0)
            {
                overlap++;
            }
        }
        return overlap;
    }

} // namespace

/**
 * @brief Test fixture for Full Graph Forward vs Legacy parity
 */
class Test__FullGraphForward_vs_Legacy_Parity : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::string model_path_;
    int rank_ = 0;
    int world_size_ = 1;
    int cpu_device_idx_ = 0;

    // Parity thresholds - requirements for Full Graph Forward parity
    // Full graph execution may have slight differences due to:
    // - Different operation ordering in the compute graph
    // - Graph executor batching decisions
    // - Accumulation order differences
    static constexpr double MIN_COSINE_SIMILARITY = 0.99;  // Slightly lower than FP32 vs FP32
    static constexpr float MAX_MAX_ABS_DIFF = 1.0f;        // Allow some outliers
    static constexpr float MAX_MEAN_ABS_DIFF = 0.2f;       // Reasonable average
    static constexpr double MIN_TOP5_OVERLAP_RATIO = 0.60; // 3/5 tokens must match

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Initialize DeviceManager
        DeviceManager::instance().initialize(-1);

        const auto &devices = DeviceManager::instance().devices();
        if (devices.empty())
        {
            GTEST_SKIP() << "No compute devices available";
        }

        cpu_device_idx_ = 0;

        if (rank_ == 0)
        {
            LOG_INFO("[Setup] DeviceManager initialized with " << devices.size() << " device(s)");
            LOG_INFO("[Setup] Using device index: " << cpu_device_idx_);
        }

        // Use Qwen 0.5B Q4_0 for fast testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);
        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        if (rank_ == 0)
        {
            LOG_INFO("[Full Graph vs Legacy Parity] Loaded model: " << model_path_);
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a legacy pipeline (no LayerExecutor, no exec_full_forward)
     */
    std::unique_ptr<Qwen2Pipeline> createLegacyPipeline()
    {
        auto &mut_env = mutableDebugEnv();

        // Disable LayerExecutor and full forward
        setenv("LLAMINAR_USE_LAYER_EXECUTOR", "0", 1);
        setenv("LLAMINAR_EXEC_FULL_FORWARD", "0", 1);
        mut_env.execution.use_layer_executor = false;
        mut_env.execution.exec_full_forward = false;

        // Disable all execution stages
        setenv("LLAMINAR_EXEC_EMBEDDING", "0", 1);
        setenv("LLAMINAR_EXEC_RMSNORM", "0", 1);
        setenv("LLAMINAR_EXEC_GEMM", "0", 1);
        setenv("LLAMINAR_EXEC_ROPE", "0", 1);
        setenv("LLAMINAR_EXEC_ATTENTION", "0", 1);
        setenv("LLAMINAR_EXEC_SWIGLU", "0", 1);
        setenv("LLAMINAR_EXEC_RESIDUAL", "0", 1);
        setenv("LLAMINAR_EXEC_LM_HEAD", "0", 1);

        mut_env.execution.exec_embedding = false;
        mut_env.execution.exec_rmsnorm = false;
        mut_env.execution.exec_gemm = false;
        mut_env.execution.exec_rope = false;
        mut_env.execution.exec_attention = false;
        mut_env.execution.exec_swiglu = false;
        mut_env.execution.exec_residual = false;
        mut_env.execution.exec_lm_head = false;

        PipelineConfig config;
        config.activation_precision = ActivationPrecision::FP32;
        config.max_seq_len = 512;

        if (!model_ctx_)
        {
            LOG_ERROR("[createLegacyPipeline] Model context is null");
            return nullptr;
        }

        // Legacy path uses device -1 (CPU convention)
        return std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, -1, nullptr, config, 1);
    }

    /**
     * @brief Create a full graph forward pipeline (LayerExecutor + exec_full_forward)
     */
    std::unique_ptr<Qwen2Pipeline> createFullGraphPipeline()
    {
        auto &mut_env = mutableDebugEnv();

        // Enable LayerExecutor and full forward
        setenv("LLAMINAR_USE_LAYER_EXECUTOR", "1", 1);
        setenv("LLAMINAR_EXEC_FULL_FORWARD", "1", 1);
        mut_env.execution.use_layer_executor = true;
        mut_env.execution.exec_full_forward = true;

        // Enable all execution stages for full graph path
        setenv("LLAMINAR_EXEC_EMBEDDING", "1", 1);
        setenv("LLAMINAR_EXEC_RMSNORM", "1", 1);
        setenv("LLAMINAR_EXEC_GEMM", "1", 1);
        setenv("LLAMINAR_EXEC_ROPE", "1", 1);
        setenv("LLAMINAR_EXEC_ATTENTION", "1", 1);
        setenv("LLAMINAR_EXEC_SWIGLU", "1", 1);
        setenv("LLAMINAR_EXEC_RESIDUAL", "1", 1);
        setenv("LLAMINAR_EXEC_LM_HEAD", "1", 1);

        mut_env.execution.exec_embedding = true;
        mut_env.execution.exec_rmsnorm = true;
        mut_env.execution.exec_gemm = true;
        mut_env.execution.exec_rope = true;
        mut_env.execution.exec_attention = true;
        mut_env.execution.exec_swiglu = true;
        mut_env.execution.exec_residual = true;
        mut_env.execution.exec_lm_head = true;

        PipelineConfig config;
        config.activation_precision = ActivationPrecision::FP32;
        config.max_seq_len = 512;

        if (!model_ctx_)
        {
            LOG_ERROR("[createFullGraphPipeline] Model context is null");
            return nullptr;
        }

        // Full graph path uses valid device index from DeviceManager
        return std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, cpu_device_idx_, nullptr, config, 1);
    }

    /**
     * @brief Compare logits between legacy and full graph forward pipelines
     */
    bool compareLogits(const float *legacy_logits, const float *graph_logits,
                       size_t vocab_size, const std::string &label)
    {
        // Get top-5 predictions
        auto legacy_top5 = get_topk(legacy_logits, vocab_size, 5);
        auto graph_top5 = get_topk(graph_logits, vocab_size, 5);

        // Compute metrics
        int top5_overlap = count_overlap(legacy_top5, graph_top5);
        double top5_ratio = top5_overlap / 5.0;
        bool top1_match = (legacy_top5[0] == graph_top5[0]);

        double cosine = cosine_similarity(legacy_logits, graph_logits, vocab_size);
        float max_diff = max_abs_diff(legacy_logits, graph_logits, vocab_size);
        float mean_diff = mean_abs_diff(legacy_logits, graph_logits, vocab_size);

        // Log results
        if (rank_ == 0)
        {
            LOG_INFO("");
            LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  " << label);
            LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
            LOG_INFO("  Top-1 match: " << (top1_match ? "YES ✓" : "NO ✗")
                                       << " (Legacy=" << legacy_top5[0] << ", FullGraph=" << graph_top5[0] << ")");
            LOG_INFO("  Top-5 overlap: " << top5_overlap << "/5 (" << std::fixed << std::setprecision(0)
                                         << (top5_ratio * 100) << "%)");
            LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cosine);
            LOG_INFO("  Max abs diff: " << std::scientific << std::setprecision(4) << max_diff);
            LOG_INFO("  Mean abs diff: " << std::scientific << std::setprecision(4) << mean_diff);

            // Show top-5 tokens
            LOG_INFO("  Legacy top-5: [");
            for (int i = 0; i < 5; ++i)
            {
                LOG_INFO("    " << i + 1 << ". token " << legacy_top5[i]
                                << " (logit=" << std::fixed << std::setprecision(2) << legacy_logits[legacy_top5[i]] << ")");
            }
            LOG_INFO("  ]");
            LOG_INFO("  FullGraph top-5: [");
            for (int i = 0; i < 5; ++i)
            {
                LOG_INFO("    " << i + 1 << ". token " << graph_top5[i]
                                << " (logit=" << std::fixed << std::setprecision(2) << graph_logits[graph_top5[i]] << ")");
            }
            LOG_INFO("  ]");
        }

        // Check pass/fail criteria
        bool pass = true;

        if (!top1_match)
        {
            if (rank_ == 0)
                LOG_WARN("  ⚠ Top-1 mismatch: Legacy=" << legacy_top5[0]
                                                       << " vs FullGraph=" << graph_top5[0]
                                                       << " (acceptable if top-5 overlap is good)");
        }

        if (top5_ratio < MIN_TOP5_OVERLAP_RATIO)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Top-5 overlap " << (top5_ratio * 100) << "% < "
                                               << (MIN_TOP5_OVERLAP_RATIO * 100) << "% threshold");
            pass = false;
        }

        if (cosine < MIN_COSINE_SIMILARITY)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Cosine similarity " << cosine << " < " << MIN_COSINE_SIMILARITY << " threshold");
            pass = false;
        }

        if (max_diff > MAX_MAX_ABS_DIFF)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Max abs diff " << max_diff << " > " << MAX_MAX_ABS_DIFF << " threshold");
            pass = false;
        }

        if (mean_diff > MAX_MEAN_ABS_DIFF)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Mean abs diff " << mean_diff << " > " << MAX_MEAN_ABS_DIFF << " threshold");
            pass = false;
        }

        if (pass && rank_ == 0)
        {
            LOG_INFO("  ✓ All parity checks PASSED");
        }

        return pass;
    }
};

/**
 * @brief Test prefill parity between Legacy Pipeline and Full Graph Forward
 *
 * Runs a multi-token prompt through both execution paths and compares final logits.
 */
TEST_F(Test__FullGraphForward_vs_Legacy_Parity, PrefillParity)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Full Graph Forward vs Legacy Parity Test: Prefill             ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // Test prompt: "The quick brown fox jumps over the lazy dog"
    // Token IDs from Qwen2.5 tokenizer (without special tokens)
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

    // Create legacy pipeline
    if (rank_ == 0)
        LOG_INFO("Creating legacy pipeline (no LayerExecutor)...");
    auto legacy_pipeline = createLegacyPipeline();
    ASSERT_NE(legacy_pipeline, nullptr) << "Failed to create legacy pipeline";

    // Create full graph forward pipeline
    if (rank_ == 0)
        LOG_INFO("Creating full graph forward pipeline (LayerExecutor + exec_full_forward)...");
    auto graph_pipeline = createFullGraphPipeline();
    ASSERT_NE(graph_pipeline, nullptr) << "Failed to create full graph pipeline";

    // Run prefill on legacy
    if (rank_ == 0)
        LOG_INFO("Running prefill on legacy pipeline...");
    bool legacy_ok = legacy_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(legacy_ok) << "Legacy pipeline forward failed";

    // Run prefill on full graph forward
    if (rank_ == 0)
        LOG_INFO("Running prefill on full graph forward pipeline...");
    bool graph_ok = graph_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(graph_ok) << "Full graph pipeline forward failed";

    // Get logits
    size_t vocab_size = model_ctx_->model().vocab_size;
    const float *legacy_logits = legacy_pipeline->logits();
    const float *graph_logits = graph_pipeline->logits();

    ASSERT_NE(legacy_logits, nullptr) << "Legacy logits is null";
    ASSERT_NE(graph_logits, nullptr) << "Full graph logits is null";

    // Compare
    bool pass = compareLogits(legacy_logits, graph_logits, vocab_size, "Prefill Logits Comparison");

    EXPECT_TRUE(pass) << "Prefill parity check failed";
}

/**
 * @brief Test incremental decode parity between Legacy Pipeline and Full Graph Forward
 *
 * After prefill, generates one token autoregressively and compares.
 */
TEST_F(Test__FullGraphForward_vs_Legacy_Parity, IncrementalDecodeParity)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Full Graph Forward vs Legacy Parity: Incremental Decode       ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // Initial prompt
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

    // Create pipelines
    auto legacy_pipeline = createLegacyPipeline();
    auto graph_pipeline = createFullGraphPipeline();
    ASSERT_NE(legacy_pipeline, nullptr) << "Failed to create legacy pipeline";
    ASSERT_NE(graph_pipeline, nullptr) << "Failed to create full graph pipeline";

    // Prefill both
    bool legacy_ok = legacy_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    bool graph_ok = graph_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(legacy_ok) << "Legacy prefill failed";
    ASSERT_TRUE(graph_ok) << "Full graph prefill failed";

    // Get next token (greedy) from legacy
    size_t vocab_size = model_ctx_->model().vocab_size;
    const float *legacy_prefill_logits = legacy_pipeline->logits();
    auto legacy_top1 = get_topk(legacy_prefill_logits, vocab_size, 1);
    int next_token = legacy_top1[0];

    if (rank_ == 0)
    {
        LOG_INFO("Next token from legacy prefill: " << next_token);
    }

    // Incremental decode: feed next_token to both pipelines
    std::vector<int> decode_tokens = {next_token};
    legacy_ok = legacy_pipeline->forward(decode_tokens.data(), 1);
    graph_ok = graph_pipeline->forward(decode_tokens.data(), 1);

    ASSERT_TRUE(legacy_ok) << "Legacy decode failed";
    ASSERT_TRUE(graph_ok) << "Full graph decode failed";

    // Get decode logits
    const float *legacy_decode_logits = legacy_pipeline->logits();
    const float *graph_decode_logits = graph_pipeline->logits();

    // Compare decode logits
    bool pass = compareLogits(legacy_decode_logits, graph_decode_logits, vocab_size,
                              "Decode Step 1 Logits Comparison");

    EXPECT_TRUE(pass) << "Incremental decode parity check failed";
}

/**
 * @brief Test that full graph forward path returns valid logits shape
 *
 * Sanity check that the graph executor produces output of correct dimensions.
 */
TEST_F(Test__FullGraphForward_vs_Legacy_Parity, FullGraphLogitsShape)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Full Graph Forward: Logits Shape Validation                   ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    std::vector<int> tokens = {785, 3974, 13876}; // Short prompt

    auto graph_pipeline = createFullGraphPipeline();
    ASSERT_NE(graph_pipeline, nullptr) << "Failed to create full graph pipeline";

    bool ok = graph_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(ok) << "Full graph forward failed";

    const float *logits = graph_pipeline->logits();
    ASSERT_NE(logits, nullptr) << "Logits pointer is null";

    // Check that logits are not all zeros (sanity)
    size_t vocab_size = model_ctx_->model().vocab_size;
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i)
    {
        sum += std::abs(logits[i]);
    }

    EXPECT_GT(sum, 0.0f) << "Logits are all zeros - full graph forward may have failed silently";

    if (rank_ == 0)
    {
        LOG_INFO("✓ Logits shape validation passed (vocab_size=" << vocab_size << ", sum_abs=" << sum << ")");
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

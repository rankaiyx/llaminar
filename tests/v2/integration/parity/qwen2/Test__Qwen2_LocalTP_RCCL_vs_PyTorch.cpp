/**
 * @file Test__Qwen2_LocalTP_RCCL_vs_PyTorch.cpp
 * @brief Integration: True LOCAL TP Parity Test with MultiDeviceOrchestrator (ROCm/RCCL)
 *
 * This test performs ACTUAL LOCAL tensor parallelism with:
 *   - 2 ROCm devices (rocm:0, rocm:1)
 *   - Weight sharding via MultiDeviceOrchestrator
 *   - RCCL backend for collective operations
 *   - PyTorch reference comparison
 *
 * This is distinct from Test__Qwen2_TP_RCCL_vs_PyTorch.cpp which only tests
 * single-device ROCm inference with RCCL availability check.
 *
 * Key differences from the existing RCCL parity test:
 *   - Uses MultiDeviceOrchestrator instead of InferenceRunnerFactory
 *   - Creates per-device DeviceGraphOrchestrator instances
 *   - Performs real weight sharding with LOCAL TP
 *   - Uses RCCL AllReduce for collective operations
 *
 * Test requirements:
 *   - At least 2 ROCm devices available
 *   - RCCL library compiled in (HAVE_RCCL)
 *   - Requires models/qwen2.5-0.5b-instruct-q4_0.gguf
 *
 * Expected behavior:
 *   - Slightly higher numerical divergence than single-device due to:
 *     - Reduced precision in sharded computations
 *     - AllReduce numerical differences
 *   - ROCm may have different numerical precision than CUDA
 *   - Token predictions should still match PyTorch reference
 *
 * @author David Sanftenberg (AI-assisted)
 * @date 2026-01-25
 */

#include "Qwen2ParityTestBase.h"
#include "models/qwen/Qwen2Schema.h"
#include "loaders/WeightManager.h"
#include "execution/MultiDeviceOrchestrator.h"
#include "execution/DeviceGraphOrchestrator.h"
#include "collective/ILocalTPContext.h"
#include "collective/LocalTPContext.h"
#include "collective/BackendRouter.h"
#include "collective/DeviceGroup.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#include <hip/hip_runtime.h>
#endif

// Note: We do NOT include <rccl.h> directly here because:
// 1. The actual RCCL calls are handled internally by LocalTPContext
// 2. We only need the HAVE_RCCL macro to check backend availability

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

/**
 * @brief Test fixture for TRUE LOCAL TP RCCL parity testing
 *
 * Uses MultiDeviceOrchestrator to perform actual tensor parallelism
 * across 2 ROCm GPUs with RCCL as the collective backend.
 */
class Test__Qwen2_LocalTP_RCCL_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    bool rccl_available_ = false;
    bool dual_rocm_available_ = false;
    int rocm_device_0_ = 0;
    int rocm_device_1_ = 1;
    int rocm_count_ = 0;

    // Multi-device orchestrator (replaces single-device runner_)
    std::unique_ptr<MultiDeviceOrchestrator> multi_orch_;

    // ==========================================================================
    // Qwen2ParityTestBase overrides
    // ==========================================================================

    BackendThresholds getBackendThresholds() override
    {
        // LOCAL TP with ROCm/RCCL has slightly higher divergence due to:
        // - Weight sharding introduces numerical differences
        // - AllReduce operations have finite precision
        // - Column-parallel outputs are combined via RCCL
        // - ROCm may have different FP32/FP16 rounding than CUDA
        //
        // Thresholds are relaxed compared to single-device ROCm:
        //   - Single-device ROCm: cosine >= 0.90, KL < 0.12
        //   - LOCAL TP (2 ROCm): cosine >= 0.88, KL < 0.20
        return BackendThresholds{
            .cosine_threshold = 0.88f, // Relaxed for LOCAL TP + ROCm precision
            .decode_cosine_threshold = 0.88f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4, // Allow 2 early layer failures for TP
            .kl_threshold = 0.25f,        // Relaxed KL threshold for ROCm + TP
            .excluded_stages = {
                // Column-parallel projections produce partial outputs per-device
                // that can't be directly compared to full PyTorch outputs
                "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
                // RoPE outputs derived from sharded Q/K (half the heads)
                "Q_ROPE", "K_ROPE",
                // Attention context has sharded head dimension
                "ATTENTION_CONTEXT",
                // FFN gate/up are column-parallel in TP
                "FFN_GATE", "FFN_UP",
                // SwiGLU output is also column-parallel (half intermediate_size)
                "FFN_SWIGLU"}};
    }

    std::string getBackendName() override
    {
        return "LOCAL_TP_RCCL(2xROCm)";
    }

    DeviceId getDeviceForRank() override
    {
        // For LOCAL TP, the "primary" device for snapshot comparison is rocm:0.
        // The MultiDeviceOrchestrator manages both devices internally.
        return DeviceId::rocm(rocm_device_0_);
    }

    // ==========================================================================
    // ParityTestBase overrides for tensor parallelism
    // ==========================================================================

    WeightDistributionStrategy getWeightStrategy() override
    {
        return WeightDistributionStrategy::SHARDED;
    }

    void configureModel(std::shared_ptr<ModelContext> model_ctx) override
    {
        // Configure weight sharding from Qwen2 schema
        Qwen2SchemaFactory schema_factory;
        model_ctx->weightManager()->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
    }

    // ==========================================================================
    // SetUp / TearDown
    // ==========================================================================

    void SetUp() override
    {
        // Check for ROCm GPUs
#ifdef HAVE_ROCM
        if (hipGetDeviceCount(&rocm_count_) != hipSuccess)
        {
            rocm_count_ = 0;
        }

        if (rocm_count_ >= 2)
        {
            dual_rocm_available_ = true;
            rocm_device_0_ = 0;
            rocm_device_1_ = 1;

            LOG_INFO("[LocalTP RCCL Parity] Found " << rocm_count_ << " ROCm devices");
        }
        else
        {
            LOG_WARN("[LocalTP RCCL Parity] Need at least 2 ROCm devices (found " << rocm_count_ << ")");
        }
#else
        LOG_WARN("[LocalTP RCCL Parity] ROCm not compiled in (HAVE_ROCM not defined)");
#endif

        // Check RCCL availability
#ifdef HAVE_RCCL
        rccl_available_ = true;
        LOG_INFO("[LocalTP RCCL Parity] RCCL backend available");
#else
        LOG_WARN("[LocalTP RCCL Parity] RCCL not compiled in (HAVE_RCCL not defined)");
#endif

        if (!dual_rocm_available_ || !rccl_available_)
        {
            GTEST_SKIP() << "Test requires 2 ROCm devices + RCCL";
        }

        // For LOCAL scope, we run single-rank but with 2 devices
        int rank = 0, world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_size != 1)
        {
            GTEST_SKIP() << "LOCAL TP test must run with -np 1 (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║        TRUE LOCAL TENSOR PARALLELISM (RCCL) PARITY TEST          ║");
        LOG_INFO("╠══════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  Device 0: ROCm:" << rocm_device_0_ << "                                              ║");
        LOG_INFO("║  Device 1: ROCm:" << rocm_device_1_ << "                                              ║");
        LOG_INFO("║  Backend: RCCL (GPU-native collectives for ROCm)                 ║");
        LOG_INFO("║  Scope: LOCAL (single process, 2 devices)                        ║");
        LOG_INFO("║  Weight Sharding: ENABLED (Megatron-style TP)                    ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");

        // Call parent setup (regenerates PyTorch snapshots)
        Qwen2ParityTestBase::SetUp();
    }

    void TearDown() override
    {
        // Clean up multi-device orchestrator first
        multi_orch_.reset();

        // Call parent teardown
        Qwen2ParityTestBase::TearDown();
    }

    // ==========================================================================
    // Multi-Device Pipeline Setup (overrides single-device setupPipeline)
    // ==========================================================================

    /**
     * @brief Setup the LOCAL TP inference pipeline using MultiDeviceOrchestrator
     *
     * This overrides the single-device setupPipeline() from ParityTestBase
     * to create a multi-device orchestrator with RCCL backend.
     */
    bool setupLocalTPPipeline()
    {
        DeviceManager::instance().initialize(-1);

        // Load model with weight sharding enabled
        model_ctx_ = ModelContext::create(
            config_.model_path,
            mpi_ctx_,
            nullptr, // placement_map
            nullptr, // factory
            WeightDistributionStrategy::SHARDED);

        if (!model_ctx_)
        {
            LOG_ERROR("[LocalTP Parity] Failed to load model");
            return false;
        }

        // Configure weight sharding schema
        configureModel(model_ctx_);

        // Create LOCAL TP context with 2 ROCm devices
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::rocm(rocm_device_0_),
            GlobalDeviceAddress::rocm(rocm_device_1_)};

        auto tp_ctx = createLocalTPContext(
            devices,
            {0.5f, 0.5f}, // Equal weights for equal GPUs
            CollectiveBackendType::RCCL);

        if (!tp_ctx)
        {
            LOG_ERROR("[LocalTP Parity] Failed to create LocalTPContext");
            return false;
        }

        LOG_INFO("[LocalTP Parity] LocalTPContext created: degree=" << tp_ctx->degree()
                                                                    << ", backend=" << static_cast<int>(tp_ctx->backend()));

        // Create MultiDeviceOrchestrator configuration
        MultiDeviceOrchestrator::Config config;
        config.devices = devices;
        config.weights = {0.5f, 0.5f};
        config.backend = CollectiveBackendType::RCCL;
        config.max_seq_len = 4096;
        config.batch_size = 1;

        // Create the multi-device orchestrator
        multi_orch_ = std::make_unique<MultiDeviceOrchestrator>(
            model_ctx_,
            std::move(tp_ctx),
            config);

        if (!multi_orch_)
        {
            LOG_ERROR("[LocalTP Parity] Failed to create MultiDeviceOrchestrator");
            return false;
        }

        // Enable snapshot capture for parity comparison
        multi_orch_->enableSnapshotCapture();

        LOG_INFO("[LocalTP Parity] MultiDeviceOrchestrator created with "
                 << multi_orch_->device_count() << " devices");

        // Set runner_ to point to multi_orch_ for ParityTestBase compatibility
        // This allows reusing existing parity test infrastructure
        // Note: multi_orch_ implements IInferenceRunner interface
        runner_.reset(multi_orch_.release());
        multi_orch_ = nullptr; // Ownership transferred to runner_

        return true;
    }
};

// =============================================================================
// Hardware Detection Test
// =============================================================================

/**
 * @brief Test: Verify RCCL backend is selected for dual-ROCm LOCAL group
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, BackendSelection_IsRCCL)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Build a LOCAL device group with 2 ROCm devices
    DeviceGroupBuilder builder;
    auto group = builder
                     .setName("rccl_test_group")
                     .setScope(CollectiveScope::LOCAL)
                     .addDevice(DeviceId::rocm(rocm_device_0_))
                     .addDevice(DeviceId::rocm(rocm_device_1_))
                     .setLocalRank(0)
                     .build();

    // Verify group properties
    EXPECT_TRUE(group.allROCm()) << "Group should be all-ROCm";
    EXPECT_FALSE(group.isHeterogeneous()) << "Group should be homogeneous";
    EXPECT_TRUE(group.isLocal()) << "Group should be LOCAL scope";
    EXPECT_EQ(group.rocm_count, 2) << "Should have 2 ROCm devices";

    LOG_INFO("[LocalTP RCCL Test] Group: " << group.toString());
    LOG_INFO("[LocalTP RCCL Test] Expected backend: RCCL (all ROCm, LOCAL scope)");
}

/**
 * @brief Test: Verify LocalTPContext creation with RCCL backend
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, LocalTPContext_Creation)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(rocm_device_0_),
        GlobalDeviceAddress::rocm(rocm_device_1_)};

    auto tp_ctx = createLocalTPContext(devices, {0.5f, 0.5f}, CollectiveBackendType::RCCL);

    ASSERT_NE(tp_ctx, nullptr) << "Failed to create LocalTPContext";
    EXPECT_EQ(tp_ctx->degree(), 2) << "TP degree should be 2";
    EXPECT_EQ(tp_ctx->devices().size(), 2u) << "Should have 2 devices";
    EXPECT_EQ(tp_ctx->backend(), CollectiveBackendType::RCCL) << "Backend should be RCCL";

    // Verify weights
    auto weights = tp_ctx->weights();
    EXPECT_NEAR(weights[0], 0.5f, 0.01f) << "Weight 0 should be 0.5";
    EXPECT_NEAR(weights[1], 0.5f, 0.01f) << "Weight 1 should be 0.5";

    LOG_INFO("[LocalTP RCCL Test] LocalTPContext created successfully");
}

// =============================================================================
// Parity Tests
// =============================================================================

/**
 * @brief Test: Prefill parity with LOCAL TP vs PyTorch
 *
 * Runs full prefill with MultiDeviceOrchestrator and compares
 * layer-by-layer outputs against PyTorch reference.
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, PrefillParity_LocalTP)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Use our LOCAL TP pipeline instead of single-device
    ASSERT_TRUE(setupLocalTPPipeline()) << "LOCAL TP pipeline setup failed";

    // Run standard prefill parity test
    auto summary = runPrefillParity();
    assertParity(summary);
}

/**
 * @brief Test: Decode parity with LOCAL TP vs PyTorch
 *
 * Tests incremental decode with MultiDeviceOrchestrator.
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, DecodeParity_LocalTP)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Use our LOCAL TP pipeline instead of single-device
    ASSERT_TRUE(setupLocalTPPipeline()) << "LOCAL TP pipeline setup failed";

    // Run standard decode parity test
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

/**
 * @brief Test: Verify snapshot infrastructure works with LOCAL TP
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, SnapshotInfrastructure_LocalTP)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Use our LOCAL TP pipeline
    ASSERT_TRUE(setupLocalTPPipeline()) << "LOCAL TP pipeline setup failed";

    // Load PyTorch reference snapshot
    auto embedding = loadPyTorchSnapshot("EMBEDDING");
    ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";

    // Run forward pass
    ASSERT_TRUE(runner_ != nullptr);
    runner_->forward(config_.token_ids.data(), config_.token_ids.size());

    // Verify we captured snapshots
    auto keys = runner_->getSnapshotKeys();
    EXPECT_GT(keys.size(), 0) << "No snapshots captured with LOCAL TP";

    bool has_embedding = std::find(keys.begin(), keys.end(), "EMBEDDING") != keys.end();
    bool has_lm_head = std::find(keys.begin(), keys.end(), "LM_HEAD") != keys.end();
    EXPECT_TRUE(has_embedding) << "Missing EMBEDDING snapshot with LOCAL TP";
    EXPECT_TRUE(has_lm_head) << "Missing LM_HEAD snapshot with LOCAL TP";

    LOG_INFO("[LocalTP RCCL Test] Captured " << keys.size() << " snapshots");
}

/**
 * @brief Test: Verify logits are reasonable (not NaN/Inf, proper distribution)
 *
 * This is a sanity check that LOCAL TP produces valid output even if
 * exact parity thresholds are not met.
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, LogitsSanityCheck_LocalTP)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Use our LOCAL TP pipeline
    ASSERT_TRUE(setupLocalTPPipeline()) << "LOCAL TP pipeline setup failed";

    // Run forward pass
    ASSERT_TRUE(runner_ != nullptr);
    bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
    ASSERT_TRUE(success) << "Forward pass failed";

    // Get logits
    const float *logits = runner_->logits();
    ASSERT_NE(logits, nullptr) << "Logits are null";

    int vocab_size = runner_->vocab_size();
    EXPECT_GT(vocab_size, 0) << "Invalid vocab size";

    // Check for NaN/Inf
    bool has_nan = false;
    bool has_inf = false;
    float sum = 0.0f;
    float min_val = logits[0];
    float max_val = logits[0];

    for (int i = 0; i < vocab_size; ++i)
    {
        if (std::isnan(logits[i]))
            has_nan = true;
        if (std::isinf(logits[i]))
            has_inf = true;
        sum += logits[i];
        min_val = std::min(min_val, logits[i]);
        max_val = std::max(max_val, logits[i]);
    }

    EXPECT_FALSE(has_nan) << "Logits contain NaN values";
    EXPECT_FALSE(has_inf) << "Logits contain Inf values";

    // Verify logits have reasonable range (not all zeros, not all same value)
    float mean = sum / vocab_size;
    EXPECT_NE(min_val, max_val) << "All logits are the same value (no variance)";

    LOG_INFO("[LocalTP RCCL Test] Logits sanity check passed:");
    LOG_INFO("  vocab_size=" << vocab_size);
    LOG_INFO("  min=" << min_val << ", max=" << max_val << ", mean=" << mean);

    // Find argmax (predicted token)
    int argmax = 0;
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i)
    {
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
            argmax = i;
        }
    }
    LOG_INFO("  predicted_token=" << argmax << " (logit=" << max_logit << ")");
}

/**
 * @brief Test: Multi-token sequence generation with LOCAL TP
 *
 * Tests that LOCAL TP can generate a sequence of tokens without errors.
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, MultiToken_Sequence_LocalTP)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Use our LOCAL TP pipeline
    ASSERT_TRUE(setupLocalTPPipeline()) << "LOCAL TP pipeline setup failed";

    ASSERT_TRUE(runner_ != nullptr);

    // Run prefill
    bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
    ASSERT_TRUE(success) << "Prefill failed";

    // Generate 5 tokens autoregressively
    std::vector<int> generated_tokens;
    const int num_tokens_to_generate = 5;

    for (int i = 0; i < num_tokens_to_generate; ++i)
    {
        const float *logits = runner_->logits();
        ASSERT_NE(logits, nullptr) << "Logits null at step " << i;

        int vocab_size = runner_->vocab_size();

        // Greedy decode (argmax)
        int next_token = 0;
        float max_logit = logits[0];
        for (int v = 1; v < vocab_size; ++v)
        {
            if (logits[v] > max_logit)
            {
                max_logit = logits[v];
                next_token = v;
            }
        }

        generated_tokens.push_back(next_token);

        // Forward the new token (decode step)
        success = runner_->forward(&next_token, 1);
        ASSERT_TRUE(success) << "Decode step " << i << " failed";
    }

    EXPECT_EQ(generated_tokens.size(), static_cast<size_t>(num_tokens_to_generate))
        << "Did not generate expected number of tokens";

    LOG_INFO("[LocalTP RCCL Test] Generated " << generated_tokens.size() << " tokens:");
    std::ostringstream oss;
    for (int tok : generated_tokens)
    {
        oss << tok << " ";
    }
    LOG_INFO("  tokens: " << oss.str());
}

/**
 * @brief Test: Verify token prediction matches PyTorch reference
 *
 * This test checks that the top-1 token prediction matches what PyTorch predicts,
 * which is the ultimate validation of LOCAL TP correctness.
 */
TEST_F(Test__Qwen2_LocalTP_RCCL_vs_PyTorch, TokenPrediction_MatchesPyTorch)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Use our LOCAL TP pipeline
    ASSERT_TRUE(setupLocalTPPipeline()) << "LOCAL TP pipeline setup failed";

    // Run forward pass
    ASSERT_TRUE(runner_ != nullptr);
    bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
    ASSERT_TRUE(success) << "Forward pass failed";

    // Get Llaminar logits
    const float *logits = runner_->logits();
    ASSERT_NE(logits, nullptr) << "Logits are null";
    int vocab_size = runner_->vocab_size();

    // Find argmax
    int llaminar_token = 0;
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i)
    {
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
            llaminar_token = i;
        }
    }

    // Load PyTorch LM_HEAD logits
    auto pytorch_logits = loadPyTorchSnapshot("LM_HEAD");
    ASSERT_FALSE(pytorch_logits.empty()) << "Failed to load PyTorch LM_HEAD snapshot";

    // Find PyTorch argmax
    int pytorch_token = 0;
    float pytorch_max = pytorch_logits[0];
    for (size_t i = 1; i < pytorch_logits.size(); ++i)
    {
        if (pytorch_logits[i] > pytorch_max)
        {
            pytorch_max = pytorch_logits[i];
            pytorch_token = static_cast<int>(i);
        }
    }

    LOG_INFO("[LocalTP RCCL Test] Token prediction comparison:");
    LOG_INFO("  Llaminar token: " << llaminar_token << " (logit=" << max_logit << ")");
    LOG_INFO("  PyTorch token:  " << pytorch_token << " (logit=" << pytorch_max << ")");

    // Token predictions should match
    EXPECT_EQ(llaminar_token, pytorch_token)
        << "Token prediction mismatch: Llaminar=" << llaminar_token
        << " vs PyTorch=" << pytorch_token;
}

// =============================================================================
// Main (MPI wrapper)
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "WARNING: MPI does not provide MPI_THREAD_MULTIPLE support" << std::endl;
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}

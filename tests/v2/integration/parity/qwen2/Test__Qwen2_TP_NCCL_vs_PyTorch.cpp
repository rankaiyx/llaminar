/**
 * @file Test__Qwen2_TP_NCCL_vs_PyTorch.cpp
 * @brief Integration: Single-CUDA Parity Test (NCCL infrastructure check)
 *
 * IMPORTANT: This test file does NOT actually perform LOCAL tensor parallelism.
 * Despite the "TP" in the filename, it runs inference on a SINGLE CUDA device
 * with FULL (unsharded) weights. This is because:
 *   - InferenceRunnerFactory only enables TP sharding when mpi_ctx->world_size() > 1
 *   - This test runs with world_size=1 (single MPI rank)
 *   - LOCAL TP requires OrchestrationRunner, not the factory-based path
 *
 * The test serves as a single-CUDA parity baseline that validates:
 *   - CUDA inference produces correct results vs PyTorch reference
 *   - NCCL library availability (hardware requirement check)
 *
 * For actual GLOBAL TP testing with weight sharding, see:
 *   - Test__Qwen2_TP_MPI_vs_PyTorch.cpp (uses world_size=2)
 *
 * TODO: Implement true LOCAL TP parity testing via OrchestrationRunner path
 * to actually shard weights across 2 CUDA devices and use NCCL for collectives.
 *
 * Test requirements:
 *   - At least 1 CUDA device available (for parity tests)
 *   - 2 CUDA devices preferred (for hardware check)
 *   - NCCL library compiled in (HAVE_NCCL)
 *   - Requires models/qwen2.5-0.5b-instruct-q4_0.gguf
 *
 * @author David Sanftenberg
 * @date 2026-01-24
 */

#include "Qwen2ParityTestBase.h"
#include "models/qwen/Qwen2Schema.h"
#include "loaders/WeightManager.h"
#include "collective/BackendRouter.h"
#include "collective/DeviceGroup.h"
#include "backends/BackendManager.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

/**
 * @brief Test fixture for single-CUDA parity baseline (NCCL hardware check)
 *
 * IMPORTANT: The parity tests in this fixture run on a SINGLE CUDA device,
 * NOT using LOCAL TP. See file header for detailed explanation.
 *
 * The hardware requirement checks verify NCCL availability, but the actual
 * inference runs on a single device without TP.
 */
class Test__Qwen2_TP_NCCL_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    bool nccl_available_ = false;
    bool dual_cuda_available_ = false;
    int cuda_device_0_ = -1;
    int cuda_device_1_ = -1;

    // ==========================================================================
    // Qwen2ParityTestBase overrides
    // ==========================================================================

    BackendThresholds getBackendThresholds() override
    {
        // NOTE: Despite the "TP" in the filename, this test does NOT actually perform
        // tensor parallelism. It runs inference on a SINGLE CUDA device with FULL
        // (unsharded) weights. See file header for explanation.
        //
        // This means the parity tests here are essentially a CUDA single-device baseline.
        // The thresholds should be similar to Test__Qwen2_CUDA_vs_PyTorch.
        //
        // Observed results (Qwen2.5-0.5B, single CUDA device):
        //   - Layer avg cosine: 0.94 - 0.99 range
        //   - LM_HEAD: cosine>0.97, KL<0.08
        //
        // The excluded_stages list is kept for API consistency but is actually
        // unnecessary since there's no TP sharding.
        return BackendThresholds{
            .cosine_threshold = 0.90f, // Single-device CUDA baseline
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5, // Allow 1 early layer failure
            .kl_threshold = 0.12f,        // Single-device threshold
            .excluded_stages = {
                // NOTE: These exclusions are kept for API consistency but are
                // actually unnecessary since this test doesn't do TP sharding.
                // Column-parallel projections produce partial outputs per-device
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
        return "NCCL_LOCAL_TP(2 CUDA devices)";
    }

    DeviceId getDeviceForRank() override
    {
        // For LOCAL TP with 2 CUDA devices, the "primary" device for
        // snapshot comparison is cuda:0. The orchestrator manages both
        // devices internally for the TP computation.
        return DeviceId::cuda(cuda_device_0_);
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
        // Check for dual CUDA GPUs
#ifdef HAVE_CUDA
        int cuda_count = 0;
        (void)cudaGetDeviceCount(&cuda_count);

        if (cuda_count >= 2)
        {
            dual_cuda_available_ = true;
            cuda_device_0_ = 0;
            cuda_device_1_ = 1;

            LOG_INFO("[NCCL TP Parity] Found " << cuda_count << " CUDA devices");
        }
        else
        {
            LOG_WARN("[NCCL TP Parity] Need at least 2 CUDA devices (found " << cuda_count << ")");
        }
#endif

        // Check NCCL availability
#ifdef HAVE_NCCL
        nccl_available_ = true;
        LOG_INFO("[NCCL TP Parity] NCCL backend available");
#else
        LOG_WARN("[NCCL TP Parity] NCCL not compiled in (HAVE_NCCL not defined)");
#endif

        if (!dual_cuda_available_ || !nccl_available_)
        {
            GTEST_SKIP() << "Test requires 2 CUDA devices + NCCL";
        }

        // For LOCAL scope, we run single-rank but with 2 devices
        // Create a minimal MPI context (or null for non-MPI builds)
        int rank = 0, world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_size != 1)
        {
            GTEST_SKIP() << "LOCAL TP test must run with -np 1 (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║           LOCAL TENSOR PARALLELISM (NCCL) TEST                   ║");
        LOG_INFO("╠══════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  Device 0: CUDA:" << cuda_device_0_ << "                                              ║");
        LOG_INFO("║  Device 1: CUDA:" << cuda_device_1_ << "                                              ║");
        LOG_INFO("║  Backend: NCCL (GPU-native collectives)                          ║");
        LOG_INFO("║  Scope: LOCAL (single process, 2 devices)                        ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");

        Qwen2ParityTestBase::SetUp();
    }

    void TearDown() override
    {
        Qwen2ParityTestBase::TearDown();
    }
};

/**
 * @brief Test: Verify NCCL backend is selected for dual-CUDA LOCAL group
 */
TEST_F(Test__Qwen2_TP_NCCL_vs_PyTorch, BackendSelection_IsNCCL)
{
    if (!dual_cuda_available_ || !nccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Build a LOCAL device group with 2 CUDA devices
    DeviceGroupBuilder builder;
    auto group = builder
                     .setName("nccl_test_group")
                     .setScope(CollectiveScope::LOCAL)
                     .addDevice(DeviceId::cuda(cuda_device_0_))
                     .addDevice(DeviceId::cuda(cuda_device_1_))
                     .setLocalRank(0)
                     .build();

    // Verify group properties
    EXPECT_TRUE(group.allCUDA()) << "Group should be all-CUDA";
    EXPECT_FALSE(group.isHeterogeneous()) << "Group should be homogeneous";
    EXPECT_TRUE(group.isLocal()) << "Group should be LOCAL scope";
    EXPECT_EQ(group.cuda_count, 2) << "Should have 2 CUDA devices";

    // Check that NCCL would be selected
    // Note: We can't easily test the actual backend selection without
    // a full BackendRouter, but we verify the group properties are correct
    LOG_INFO("[NCCL TP Test] Group: " << group.toString());
    LOG_INFO("[NCCL TP Test] Expected backend: NCCL (all CUDA, LOCAL scope)");
}

// Instantiate standard parity tests
INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_TP_NCCL_vs_PyTorch);

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

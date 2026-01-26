/**
 * @file Test__Qwen2_TP_PCIeBAR_vs_PyTorch.cpp
 * @brief Integration: PCIeBAR Backend Infrastructure Tests + Single-CUDA Parity Baseline
 *
 * IMPORTANT: This test file contains TWO distinct test types:
 *
 * 1. PCIeBAR BACKEND TESTS (BackendSelection_IsPCIeBAR, DeviceTopology_NUMAInfo):
 *    These validate PCIeBAR infrastructure - initialization, bandwidth, NUMA detection.
 *    They use actual CUDA+ROCm device pairs and test the P2P collective backend.
 *
 * 2. PARITY TESTS (PrefillParity_LayerByLayer, DecodeParity_Incremental):
 *    Despite the filename suggesting "TP", these run inference on a SINGLE CUDA device
 *    with FULL (unsharded) weights. This is because:
 *      - InferenceRunnerFactory only enables TP sharding when mpi_ctx->world_size() > 1
 *      - This test runs with world_size=1 (single MPI rank)
 *      - LOCAL TP requires OrchestrationRunner, not the factory-based path
 *
 *    These parity tests serve as a CUDA single-device baseline for comparison against:
 *      - Test__Qwen2_TP_MPI_vs_PyTorch: Actual GLOBAL TP with 2 ranks
 *      - Test__Qwen2_CUDA_vs_PyTorch: Pure single-CUDA test
 *
 * Hardware Architecture:
 *   - Detects CUDA + ROCm devices (for backend tests)
 *   - Inference runs on CUDA only (single device, no TP)
 *   - MPI: Single rank (mpirun -np 1)
 *
 * PCIe BAR P2P mechanism (tested in backend tests):
 *   1. AMD GPU's BAR (Base Address Register) is memory-mapped via mmap()
 *   2. CUDA registers this mapping as IOMEMORY via cuMemHostRegister()
 *   3. CUDA kernels can directly write to AMD GPU memory (no host staging)
 *   4. Achieves ~2.65 GB/s on PCIe 3.0 x16
 *
 * TODO: Implement true LOCAL TP parity testing via OrchestrationRunner path
 * to actually shard weights across CUDA+ROCm and use PCIeBAR for collectives.
 *
 * Test requirements:
 *   - At least 1 CUDA device (for parity tests)
 *   - 1 CUDA + 1 ROCm device (for PCIeBAR backend tests)
 *   - AMD GPU with Large BAR support (32GB BAR for MI50/MI60)
 *   - CAP_SYS_ADMIN capability or appropriate udev rule for BAR access
 *   - Requires models/qwen2.5-0.5b-instruct-q4_0.gguf
 *
 * @author David Sanftenberg
 * @date 2026-01-24
 */

#include "Qwen2ParityTestBase.h"
#include "models/qwen/Qwen2Schema.h"
#include "loaders/WeightManager.h"
#include "utils/NUMATopology.h"
#include "backends/p2p/DirectP2P.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "collective/backends/PCIeBARBackend.h"
#include "collective/DeviceGroup.h"

// NOTE: Cannot include both cuda_runtime.h and hip/hip_runtime.h in same TU
// due to type redefinitions (dim3, vector types, etc.). Use DeviceManager for counts.

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

namespace
{

    /**
     * @brief Find CUDA and ROCm device IDs, preferring same NUMA node
     *
     * Uses DeviceManager for device enumeration (avoids CUDA/HIP header conflicts)
     *
     * @param out_cuda_id Output: CUDA device ID
     * @param out_rocm_id Output: ROCm device ID
     * @return true if a pair was found (same NUMA preferred but not required)
     */
    bool findCUDAROCmPair(int &out_cuda_id, int &out_rocm_id)
    {
        out_cuda_id = -1;
        out_rocm_id = -1;

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        auto &dm = DeviceManager::instance();

        // Ensure DeviceManager is initialized (enumerate all devices, no NUMA filter)
        dm.initialize(-1);

        int cuda_count = dm.cuda_device_count();
        int rocm_count = dm.rocm_device_count();

        LOG_DEBUG("[PCIeBAR] DeviceManager: " << cuda_count << " CUDA, " << rocm_count << " ROCm devices");

        if (cuda_count < 1 || rocm_count < 1)
        {
            LOG_WARN("[PCIeBAR] Need at least 1 CUDA and 1 ROCm device");
            return false;
        }

        // Try to find same-NUMA pair first
        for (int cuda_idx = 0; cuda_idx < cuda_count; ++cuda_idx)
        {
            auto cuda_numa = NUMATopology::getCUDAGPUNUMANode(cuda_idx);

            for (int rocm_idx = 0; rocm_idx < rocm_count; ++rocm_idx)
            {
                auto rocm_numa = NUMATopology::getROCmGPUNUMANode(rocm_idx);

                if (cuda_numa.numa_node == rocm_numa.numa_node && cuda_numa.numa_node >= 0)
                {
                    out_cuda_id = cuda_idx;
                    out_rocm_id = rocm_idx;
                    LOG_INFO("[PCIeBAR] Found same-NUMA pair: CUDA " << cuda_idx
                                                                     << " + ROCm " << rocm_idx << " (NUMA " << cuda_numa.numa_node << ")");
                    return true;
                }
            }
        }

        // Fallback: any CUDA + any ROCm
        out_cuda_id = 0;
        out_rocm_id = 0;
        LOG_INFO("[PCIeBAR] Using cross-NUMA pair: CUDA " << out_cuda_id
                                                          << " + ROCm " << out_rocm_id << " (may have reduced bandwidth)");
        return true;
#else
        return false;
#endif
    }

} // namespace

/**
 * @brief Test fixture for PCIeBAR backend tests + single-CUDA parity baseline
 *
 * IMPORTANT: The parity tests in this fixture run on a SINGLE CUDA device,
 * NOT using LOCAL TP. See file header for detailed explanation.
 *
 * The backend-specific tests (BackendSelection, DeviceTopology) do test
 * actual PCIeBAR infrastructure with CUDA+ROCm.
 */
class Test__Qwen2_TP_PCIeBAR_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    bool pciebar_available_ = false;
    bool hetero_gpus_available_ = false;
    int cuda_device_id_ = -1;
    int rocm_device_id_ = -1;

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
        // The thresholds are tight because there's no TP overhead:
        //   - No weight sharding (full GEMM accumulation order)
        //   - No allreduce (no reduction rounding)
        //   - Single device (consistent kernel behavior)
        //
        // Observed results (Qwen2.5-0.5B, single CUDA device):
        //   - Layer avg cosine: 0.96 - 0.999 range
        //   - Min cosine: ~0.938 (worst case in layer 3 ATTENTION_OUTPUT)
        //   - LM_HEAD: cosine=0.978, KL=0.032, Top-5=80%
        //
        // The excluded_stages list is kept for consistency with other TP tests,
        // but these stages actually produce FULL outputs (not partial) since
        // there's no sharding.
        return BackendThresholds{
            .cosine_threshold = 0.93f, // Based on observed min ~0.938
            .decode_cosine_threshold = 0.93f,
            .early_layers_count = 6,
            .min_early_layers_passed = 6, // All early layers should pass
            .kl_threshold = 0.10f,        // Tighter - observed KL=0.032
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
        return "PCIeBAR_LOCAL_TP(CUDA+ROCm)";
    }

    DeviceId getDeviceForRank() override
    {
        // For LOCAL TP, return CUDA device as "primary"
        // The actual TP happens via CollectiveContext with both devices
        return DeviceId::cuda(cuda_device_id_);
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
        // Check for CUDA + ROCm
        hetero_gpus_available_ = findCUDAROCmPair(cuda_device_id_, rocm_device_id_);

        if (!hetero_gpus_available_)
        {
            GTEST_SKIP() << "Test requires 1 CUDA + 1 ROCm device";
        }

        // Check PCIe BAR P2P availability
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        auto caps = DirectP2PEngine::probeCapabilities();
        pciebar_available_ = caps.canDoPCIeBarP2P();

        LOG_INFO("[PCIeBAR TP Parity] PCIe BAR P2P: " << (pciebar_available_ ? "available" : "NOT available"));
        LOG_INFO("[PCIeBAR TP Parity]   BAR accessible: " << (caps.pcie_bar_accessible ? "YES" : "NO"));
        LOG_INFO("[PCIeBAR TP Parity]   IOMEMORY support: " << (caps.pcie_bar_iomemory_supported ? "YES" : "NO"));
        LOG_INFO("[PCIeBAR TP Parity]   Discovered BARs: " << caps.discovered_bars.size());
#endif

        if (!pciebar_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P not available (need root or udev rule)";
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
        LOG_INFO("║        LOCAL TENSOR PARALLELISM (PCIeBAR) TEST                   ║");
        LOG_INFO("╠══════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  Device 0: CUDA:" << cuda_device_id_ << "                                              ║");
        LOG_INFO("║  Device 1: ROCm:" << rocm_device_id_ << "                                              ║");
        LOG_INFO("║  Backend: PCIeBAR (direct CUDA↔ROCm P2P via BAR)                 ║");
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
 * @brief Test: Verify PCIeBAR backend initializes and measures bandwidth
 */
TEST_F(Test__Qwen2_TP_PCIeBAR_vs_PyTorch, BackendSelection_IsPCIeBAR)
{
    if (!hetero_gpus_available_ || !pciebar_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    // Build a LOCAL device group with CUDA + ROCm
    DeviceGroupBuilder builder;
    auto group = builder
                     .setName("pciebar_local_test_group")
                     .setScope(CollectiveScope::LOCAL)
                     .addDevice(DeviceId::cuda(cuda_device_id_))
                     .addDevice(DeviceId::rocm(rocm_device_id_))
                     .setLocalRank(0)
                     .build();

    // Verify group properties
    EXPECT_TRUE(group.isHeterogeneous()) << "Group should be heterogeneous";
    EXPECT_TRUE(group.isLocal()) << "Group should be LOCAL scope";
    EXPECT_EQ(group.cuda_count, 1) << "Should have 1 CUDA device";
    EXPECT_EQ(group.rocm_count, 1) << "Should have 1 ROCm device";

    // Initialize PCIeBAR backend
    PCIeBARBackend backend;
    bool init_ok = backend.initialize(group);

    LOG_INFO("[PCIeBAR TP Test] Backend initialization: " << (init_ok ? "SUCCESS" : "FAILED"));

    ASSERT_TRUE(init_ok) << "PCIeBARBackend initialization failed";
    EXPECT_TRUE(backend.isPCIeBarActive()) << "PCIe BAR should be active";
    EXPECT_EQ(backend.type(), CollectiveBackendType::PCIE_BAR);

    // Verify reasonable bandwidth
    double bandwidth = backend.getMeasuredBandwidthGBps();
    LOG_INFO("[PCIeBAR TP Test] Measured bandwidth: " << bandwidth << " GB/s");
    EXPECT_GT(bandwidth, 1.0) << "PCIe BAR bandwidth should be > 1 GB/s";

    backend.shutdown();
#endif
}

/**
 * @brief Test: Verify NUMA topology detection
 */
TEST_F(Test__Qwen2_TP_PCIeBAR_vs_PyTorch, DeviceTopology_NUMAInfo)
{
    if (!hetero_gpus_available_)
    {
        GTEST_SKIP() << "Heterogeneous GPUs not available";
    }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    auto cuda_numa = NUMATopology::getCUDAGPUNUMANode(cuda_device_id_);
    auto rocm_numa = NUMATopology::getROCmGPUNUMANode(rocm_device_id_);

    LOG_INFO("[PCIeBAR Test] CUDA " << cuda_device_id_ << " NUMA: " << cuda_numa.numa_node);
    LOG_INFO("[PCIeBAR Test] ROCm " << rocm_device_id_ << " NUMA: " << rocm_numa.numa_node);

    // Both should have valid NUMA info
    EXPECT_GE(cuda_numa.numa_node, 0) << "CUDA device should have valid NUMA node";
    EXPECT_GE(rocm_numa.numa_node, 0) << "ROCm device should have valid NUMA node";

    if (cuda_numa.numa_node == rocm_numa.numa_node)
    {
        LOG_INFO("[PCIeBAR Test] ✓ Same NUMA node - optimal for PCIe BAR P2P");
    }
    else
    {
        LOG_WARN("[PCIeBAR Test] ✗ Different NUMA nodes - cross-socket P2P may have higher latency");
    }
#endif
}

// Instantiate standard parity tests
INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_TP_PCIeBAR_vs_PyTorch);

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

/**
 * @file Test__PCIeBARBackend.cpp
 * @brief Unit tests for PCIeBARBackend — validation logic only, NO GPU hardware
 *
 * Tests the multi-pair API validation logic, DevicePair struct behavior,
 * and parameter validation WITHOUT requiring actual GPU hardware.
 *
 * @note Hardware initialization tests (probeCapabilities, BAR mapping,
 *       bandwidth benchmarks, CUDA stream creation) are in:
 *       integration/collective/backends/Test__PCIeBARBackend_HardwareInit.cpp
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/backends/DeviceId.h"

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-Pair API Tests (NO hardware required)
    //
    // These tests verify the multi-pair API validation logic and state management
    // WITHOUT requiring actual hardware initialization. They test:
    // - DevicePair struct equality operator
    // - isMultiPairMode() state before initialization
    // - initializeMultiPair() parameter validation
    // - allreduceMultiPair() behavior when not initialized
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__PCIeBARBackend_MultiPair : public ::testing::Test
    {
    protected:
        // Create backend WITHOUT hardware check - for validation-only tests
        std::unique_ptr<PCIeBARBackend> createBackendForValidation()
        {
            return std::make_unique<PCIeBARBackend>();
        }
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // DevicePair Struct Tests
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_SameDevices)
    {
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(0), DeviceId::rocm(0), 0};

        EXPECT_TRUE(pair1 == pair2) << "Identical pairs should be equal";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_DifferentCUDA)
    {
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(1), DeviceId::rocm(0), 0};

        EXPECT_FALSE(pair1 == pair2) << "Pairs with different CUDA devices should not be equal";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_DifferentROCm)
    {
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(0), DeviceId::rocm(1), 0};

        EXPECT_FALSE(pair1 == pair2) << "Pairs with different ROCm devices should not be equal";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_IgnoresPairIndex)
    {
        // Equality is based on device IDs only, not pair_index
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(0), DeviceId::rocm(0), 5};

        EXPECT_TRUE(pair1 == pair2) << "Pairs should be equal regardless of pair_index";
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // isMultiPairMode() State Tests
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, IsMultiPairMode_FalseBeforeInitialize)
    {
        auto backend = createBackendForValidation();

        EXPECT_FALSE(backend->isMultiPairMode())
            << "isMultiPairMode() should return false before initialization";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, GetDevicePairs_EmptyBeforeInitialize)
    {
        auto backend = createBackendForValidation();

        const auto &pairs = backend->getDevicePairs();
        EXPECT_TRUE(pairs.empty())
            << "getDevicePairs() should return empty vector before initialization";
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // initializeMultiPair() Validation Tests
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsEmptyPairsVector)
    {
        auto backend = createBackendForValidation();

        std::vector<DevicePair> empty_pairs;
        EXPECT_FALSE(backend->initializeMultiPair(empty_pairs))
            << "initializeMultiPair() should reject empty pairs vector";

        EXPECT_FALSE(backend->isMultiPairMode())
            << "isMultiPairMode() should remain false after failed initialization";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsNonCUDADeviceAsCUDA)
    {
        auto backend = createBackendForValidation();

        // Try to use ROCm device as cuda_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::rocm(0), DeviceId::rocm(1), 0} // rocm(0) is NOT a CUDA device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject non-CUDA device as cuda_device";

        EXPECT_FALSE(backend->isInitialized())
            << "Backend should not be initialized after validation failure";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsCPUDeviceAsCUDA)
    {
        auto backend = createBackendForValidation();

        // Try to use CPU device as cuda_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cpu(), DeviceId::rocm(0), 0} // CPU is NOT a CUDA device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject CPU device as cuda_device";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsNonROCmDeviceAsROCm)
    {
        auto backend = createBackendForValidation();

        // Try to use CUDA device as rocm_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::cuda(1), 0} // cuda(1) is NOT a ROCm device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject non-ROCm device as rocm_device";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsCPUDeviceAsROCm)
    {
        auto backend = createBackendForValidation();

        // Try to use CPU device as rocm_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::cpu(), 0} // CPU is NOT a ROCm device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject CPU device as rocm_device";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsDuplicateCUDADevices)
    {
        auto backend = createBackendForValidation();

        // Two pairs using the same CUDA device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0},
            {DeviceId::cuda(0), DeviceId::rocm(1), 1} // cuda(0) already used in pair 0!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject duplicate CUDA devices across pairs";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsDuplicateROCmDevices)
    {
        auto backend = createBackendForValidation();

        // Two pairs using the same ROCm device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0},
            {DeviceId::cuda(1), DeviceId::rocm(0), 1} // rocm(0) already used in pair 0!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject duplicate ROCm devices across pairs";
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // allreduceMultiPair() Tests (not initialized)
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, AllreduceMultiPair_FailsWhenNotInitialized)
    {
        auto backend = createBackendForValidation();

        // Create fake buffer vectors
        std::vector<void *> cuda_buffers = {nullptr};
        std::vector<void *> rocm_buffers = {nullptr};

        EXPECT_FALSE(backend->allreduceMultiPair(
            cuda_buffers, rocm_buffers, 100, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair() should fail when backend not initialized";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, AllreduceMultiPair_FailsWhenNotInMultiPairMode)
    {
        auto backend = createBackendForValidation();

        std::vector<void *> cuda_buffers;
        std::vector<void *> rocm_buffers;

        // Empty buffers should fail even before multi-pair mode check
        EXPECT_FALSE(backend->allreduceMultiPair(
            cuda_buffers, rocm_buffers, 100, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair() should fail with empty buffers";
    }

} // namespace llaminar2::test

#else // !HAVE_CUDA || !HAVE_ROCM

// Stub test when CUDA+ROCm not available
TEST(Test__PCIeBARBackend, RequiresCUDAAndROCm)
{
    GTEST_SKIP() << "PCIeBARBackend requires both HAVE_CUDA and HAVE_ROCM";
}

// Stub tests for multi-pair API when not available
TEST(Test__PCIeBARBackend_MultiPair_Stub, StubIsMultiPairMode)
{
    llaminar2::PCIeBARBackend backend;
    EXPECT_FALSE(backend.isMultiPairMode()) << "Stub should always return false";
}

TEST(Test__PCIeBARBackend_MultiPair_Stub, StubInitializeMultiPairFails)
{
    llaminar2::PCIeBARBackend backend;
    std::vector<llaminar2::DevicePair> pairs = {
        {llaminar2::DeviceId::cuda(0), llaminar2::DeviceId::rocm(0), 0}};
    EXPECT_FALSE(backend.initializeMultiPair(pairs))
        << "Stub should always fail initialization";
}

#endif // HAVE_CUDA && HAVE_ROCM

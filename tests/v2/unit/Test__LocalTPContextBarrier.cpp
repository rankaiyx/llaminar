/**
 * @file Test__LocalTPContextBarrier.cpp
 * @brief Unit tests for LocalTPContext barrier synchronization
 *
 * Tests the NCCL-style barrier mechanism for PCIeBAR heterogeneous allreduce
 * where multiple device threads must rendezvous before the actual data transfer.
 *
 * NOTE: The barrier mechanism is only active for PCIeBAR backend with CUDA+ROCm.
 * Single-device tests verify basic functionality, while PCIeBAR tests verify
 * the actual barrier synchronization.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#include "collective/LocalTPContext.h"
#include "collective/ICollectiveBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/Tensors.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @class Test__LocalTPContextBarrier
 * @brief Test fixture for LocalTPContext barrier synchronization
 *
 * Tests the barrier mechanism using realistic device configurations.
 * - Single-device tests verify no-op behavior
 * - PCIeBAR tests verify actual barrier rendezvous (requires CUDA+ROCm)
 */
class Test__LocalTPContextBarrier : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices - use CUDA + ROCm for realistic heterogeneous setup
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        rocm0_ = GlobalDeviceAddress::rocm(0, 0);
        cpu0_ = GlobalDeviceAddress::cpu(0);
    }

    void TearDown() override
    {
        // Ensure all threads are joined
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress rocm0_;
    GlobalDeviceAddress cpu0_;
};

// =============================================================================
// Single Device Tests (No Barrier Needed)
// =============================================================================

/**
 * @test Single device allreduce should be a no-op (no barrier needed)
 */
TEST_F(Test__LocalTPContextBarrier, SingleDeviceNoBarrier)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);

    auto tensor = TestTensorFactory::createFP32({1024});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Single device - should return immediately with no barrier
    bool result = ctx->allreduce(tensor.get());
    EXPECT_TRUE(result);
}

/**
 * @test Single CPU device allreduce should work
 */
TEST_F(Test__LocalTPContextBarrier, SingleCPUDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);

    auto tensor = TestTensorFactory::createFP32({256});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = 1.0f;
    }

    // Single device - should return immediately
    bool result = ctx->allreduce(tensor.get());
    EXPECT_TRUE(result);
}

/**
 * @test Verify context state initialization
 */
TEST_F(Test__LocalTPContextBarrier, ContextStateInitialization)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    // Context should be created without errors
    EXPECT_EQ(ctx->degree(), 1);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/**
 * @test Null tensor should return false
 */
TEST_F(Test__LocalTPContextBarrier, NullTensorReturnsError)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    bool result = ctx->allreduce(nullptr);
    EXPECT_FALSE(result);
}

/**
 * @test Rapid sequential allreduces from single device context
 */
TEST_F(Test__LocalTPContextBarrier, RapidSequentialAllreducesSingleDevice)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    const int num_iterations = 100;
    auto tensor = TestTensorFactory::createFP32({512});

    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        tensor->mutable_data()[i] = static_cast<float>(i);
    }

    for (int i = 0; i < num_iterations; ++i)
    {
        bool result = ctx->allreduce(tensor.get());
        EXPECT_TRUE(result) << "Failed at iteration " << i;
    }
}

/**
 * @test Out-of-place allreduce with single device
 */
TEST_F(Test__LocalTPContextBarrier, OutOfPlaceAllreduceSingleDevice)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    auto input = TestTensorFactory::createFP32({256});
    auto output = TestTensorFactory::createFP32({256});

    for (size_t i = 0; i < input->numel(); ++i)
    {
        input->mutable_data()[i] = static_cast<float>(i);
        output->mutable_data()[i] = 0.0f;
    }

    bool result = ctx->allreduce(input.get(), output.get());
    EXPECT_TRUE(result);

    // Output should have been populated (copied from input for single device)
    bool has_nonzero = false;
    for (size_t i = 0; i < output->numel(); ++i)
    {
        if (output->data()[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// =============================================================================
// PCIeBAR-Specific Tests (Conditional - requires CUDA+ROCm)
// =============================================================================

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

/**
 * @test PCIeBAR barrier requires both devices to participate
 *
 * This test is only compiled when both CUDA and ROCm are available.
 * It verifies the actual barrier behavior with PCIeBAR backend.
 */
TEST_F(Test__LocalTPContextBarrier, PCIeBarBarrierRendezvous)
{
    // Create context with PCIeBAR backend
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {}, CollectiveBackendType::PCIE_BAR);

    if (!ctx || ctx->backend() != CollectiveBackendType::PCIE_BAR)
    {
        GTEST_SKIP() << "PCIeBAR backend not available";
    }

    ASSERT_EQ(ctx->degree(), 2);

    std::atomic<int> arrived_count{0};
    std::atomic<int> completed_count{0};
    std::atomic<bool> thread1_arrived{false};
    std::atomic<bool> thread2_arrived{false};
    std::mutex arrival_mutex;
    std::condition_variable arrival_cv;

    auto tensor1 = TestTensorFactory::createFP32({1024});
    auto tensor2 = TestTensorFactory::createFP32({1024});

    for (size_t i = 0; i < tensor1->numel(); ++i)
    {
        tensor1->mutable_data()[i] = 1.0f;
        tensor2->mutable_data()[i] = 2.0f;
    }

    // Thread 1: CUDA device
    std::thread cuda_thread([&]()
                            {
        thread1_arrived = true;
        arrived_count++;
        arrival_cv.notify_all();
        
        // Call allreduce - should block until ROCm thread also calls
        bool result = ctx->allreduce(tensor1.get());
        completed_count++;
        EXPECT_TRUE(result); });

    // Thread 2: ROCm device (delayed start to test waiting)
    std::thread rocm_thread([&]()
                            {
        // Wait a bit to ensure CUDA thread arrives first
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        thread2_arrived = true;
        arrived_count++;
        arrival_cv.notify_all();
        
        // Call allreduce - should unblock the CUDA thread
        bool result = ctx->allreduce(tensor2.get());
        completed_count++;
        EXPECT_TRUE(result); });

    // Wait for threads with timeout
    cuda_thread.join();
    rocm_thread.join();

    EXPECT_EQ(arrived_count.load(), 2);
    EXPECT_EQ(completed_count.load(), 2);
}

/**
 * @test PCIeBAR multiple barrier cycles
 *
 * Verifies the barrier can be reused multiple times.
 */
TEST_F(Test__LocalTPContextBarrier, PCIeBarMultipleBarrierCycles)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {}, CollectiveBackendType::PCIE_BAR);

    if (!ctx || ctx->backend() != CollectiveBackendType::PCIE_BAR)
    {
        GTEST_SKIP() << "PCIeBAR backend not available";
    }

    const int num_cycles = 5;
    std::atomic<int> cycle_count{0};

    auto tensor1 = TestTensorFactory::createFP32({512});
    auto tensor2 = TestTensorFactory::createFP32({512});

    for (size_t i = 0; i < tensor1->numel(); ++i)
    {
        tensor1->mutable_data()[i] = 1.0f;
        tensor2->mutable_data()[i] = 2.0f;
    }

    std::thread cuda_thread([&]()
                            {
        for (int c = 0; c < num_cycles; ++c) {
            bool result = ctx->allreduce(tensor1.get());
            EXPECT_TRUE(result) << "CUDA thread failed at cycle " << c;
            cycle_count++;
        } });

    std::thread rocm_thread([&]()
                            {
        for (int c = 0; c < num_cycles; ++c) {
            bool result = ctx->allreduce(tensor2.get());
            EXPECT_TRUE(result) << "ROCm thread failed at cycle " << c;
            cycle_count++;
        } });

    cuda_thread.join();
    rocm_thread.join();

    EXPECT_EQ(cycle_count.load(), num_cycles * 2);
}

/**
 * @test PCIeBAR stress test with many barrier cycles
 *
 * Ensures no deadlocks or race conditions under load.
 */
TEST_F(Test__LocalTPContextBarrier, PCIeBarStressTest)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {}, CollectiveBackendType::PCIE_BAR);

    if (!ctx || ctx->backend() != CollectiveBackendType::PCIE_BAR)
    {
        GTEST_SKIP() << "PCIeBAR backend not available";
    }

    const int num_cycles = 20;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    auto tensor1 = TestTensorFactory::createFP32({256});
    auto tensor2 = TestTensorFactory::createFP32({256});

    for (size_t i = 0; i < tensor1->numel(); ++i)
    {
        tensor1->mutable_data()[i] = 1.0f;
        tensor2->mutable_data()[i] = 2.0f;
    }

    std::thread cuda_thread([&]()
                            {
        for (int c = 0; c < num_cycles; ++c) {
            if (ctx->allreduce(tensor1.get())) {
                success_count++;
            } else {
                failure_count++;
            }
        } });

    std::thread rocm_thread([&]()
                            {
        for (int c = 0; c < num_cycles; ++c) {
            if (ctx->allreduce(tensor2.get())) {
                success_count++;
            } else {
                failure_count++;
            }
        } });

    cuda_thread.join();
    rocm_thread.join();

    // All operations should succeed
    EXPECT_EQ(success_count.load(), num_cycles * 2);
    EXPECT_EQ(failure_count.load(), 0);
}

/**
 * @test First arrival waits, second triggers execution
 *
 * Verifies the barrier timing behavior.
 */
TEST_F(Test__LocalTPContextBarrier, FirstArrivalWaitsSecondTriggers)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {}, CollectiveBackendType::PCIE_BAR);

    if (!ctx || ctx->backend() != CollectiveBackendType::PCIE_BAR)
    {
        GTEST_SKIP() << "PCIeBAR backend not available";
    }

    std::atomic<bool> first_thread_started{false};
    std::atomic<bool> first_thread_completed{false};
    std::atomic<bool> second_thread_started{false};

    auto tensor1 = TestTensorFactory::createFP32({128});
    auto tensor2 = TestTensorFactory::createFP32({128});

    for (size_t i = 0; i < tensor1->numel(); ++i)
    {
        tensor1->mutable_data()[i] = 1.0f;
        tensor2->mutable_data()[i] = 2.0f;
    }

    // First thread: starts immediately
    std::thread first_thread([&]()
                             {
        first_thread_started = true;
        ctx->allreduce(tensor1.get());
        first_thread_completed = true; });

    // Wait for first thread to start and enter barrier
    while (!first_thread_started)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // First thread should NOT have completed yet (waiting at barrier)
    EXPECT_FALSE(first_thread_completed.load());

    // Second thread: starts after delay
    std::thread second_thread([&]()
                              {
        second_thread_started = true;
        ctx->allreduce(tensor2.get()); });

    // Both threads should complete now
    first_thread.join();
    second_thread.join();

    EXPECT_TRUE(first_thread_completed.load());
}

#endif // HAVE_CUDA && HAVE_ROCM

// =============================================================================
// Backend Fallback Tests
// =============================================================================

/**
 * @test Verify backend auto-detection doesn't affect single-device behavior
 */
TEST_F(Test__LocalTPContextBarrier, AutoDetectedBackendWorks)
{
    // AUTO should work for single device
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    auto tensor = TestTensorFactory::createFP32({256});
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        tensor->mutable_data()[i] = static_cast<float>(i);
    }

    bool result = ctx->allreduce(tensor.get());
    EXPECT_TRUE(result);
}

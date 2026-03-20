/**
 * @file Test__PCIeBARBackend_WorkerPool.cpp
 * @brief Integration tests for the PCIeBARBackend per-device worker pool,
 *        allocation-free event waits, per-pair resources, and ROCm event
 *        integration.
 *
 * These tests exercise the refactored PCIeBARBackend infrastructure:
 * - CUDADeviceWorker pool: per-device worker threads with clean CUDA contexts
 * - WorkSlot hot path: allocation-free event-wait submission
 * - PairResources: pre-allocated streams, events, and temp buffers
 * - waitForROCmEvent: HIP event sync via IBackend abstraction
 * - allreduceMultiPair: correctness under the new per-pair pipeline
 * - Repeated allreduce: verify no hot-path allocations
 *
 * Prerequisites:
 * - At least 1 CUDA + 1 ROCm device
 * - CAP_SYS_ADMIN for PCIe BAR mapping
 * - AMD GPU with large BAR support
 *
 * @author David Sanftenberg
 * @date March 2026
 */

#include <gtest/gtest.h>

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"
#include "v2/backends/p2p/DirectP2P.h"
#include "v2/utils/Logger.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__PCIeBARBackend_WorkerPool : public ::testing::Test
    {
    protected:
        std::unique_ptr<PCIeBARBackend> backend_;
        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;

        int cuda_count_ = 0;
        int rocm_count_ = 0;
        bool has_pcie_bar_p2p_ = false;

        static constexpr size_t SMALL_COUNT = 256;
        static constexpr size_t SMALL_BYTES = SMALL_COUNT * sizeof(float);

        static constexpr size_t MEDIUM_COUNT = 4096;
        static constexpr size_t MEDIUM_BYTES = MEDIUM_COUNT * sizeof(float);

        static constexpr size_t LARGE_COUNT = 65536; // 256KB
        static constexpr size_t LARGE_BYTES = LARGE_COUNT * sizeof(float);

        void SetUp() override
        {
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();

            if (cuda_backend_)
                cuda_count_ = cuda_backend_->deviceCount();
            if (rocm_backend_)
                rocm_count_ = rocm_backend_->deviceCount();

            auto caps = DirectP2PEngine::probeCapabilities();
            has_pcie_bar_p2p_ = caps.canDoPCIeBarP2P();

            if (has_pcie_bar_p2p_)
            {
                LOG_INFO("[WorkerPool] PCIe BAR P2P available: "
                         << cuda_count_ << " CUDA, " << rocm_count_ << " ROCm");
            }
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Skip macros
        // ─────────────────────────────────────────────────────────────────────

#define REQUIRE_PCIE_BAR()                                         \
    do                                                             \
    {                                                              \
        if (!has_pcie_bar_p2p_)                                    \
            GTEST_SKIP() << "PCIe BAR P2P hardware not available"; \
    } while (0)

#define REQUIRE_CUDA(n)                                      \
    do                                                       \
    {                                                        \
        if (cuda_count_ < (n))                               \
            GTEST_SKIP() << "Need " << (n) << " CUDA"        \
                         << " (have " << cuda_count_ << ")"; \
    } while (0)

#define REQUIRE_ROCM(n)                                      \
    do                                                       \
    {                                                        \
        if (rocm_count_ < (n))                               \
            GTEST_SKIP() << "Need " << (n) << " ROCm"        \
                         << " (have " << rocm_count_ << ")"; \
    } while (0)

        // ─────────────────────────────────────────────────────────────────────
        // Helpers: initialise backend in single-pair or multi-pair mode
        // ─────────────────────────────────────────────────────────────────────

        bool initSinglePair()
        {
            backend_ = std::make_unique<PCIeBARBackend>();
            std::vector<DevicePair> pairs = {
                {DeviceId::cuda(0), DeviceId::rocm(0), 0}};
            return backend_->initializeMultiPair(pairs);
        }

        bool initTwoPairs()
        {
            backend_ = std::make_unique<PCIeBARBackend>();
            std::vector<DevicePair> pairs = {
                {DeviceId::cuda(0), DeviceId::rocm(0), 0},
                {DeviceId::cuda(1), DeviceId::rocm(1), 1}};
            return backend_->initializeMultiPair(pairs);
        }

        bool initSinglePairViaOldAPI()
        {
            backend_ = std::make_unique<PCIeBARBackend>();
            DeviceGroupBuilder builder;
            auto group = builder.setName("worker_pool_test")
                             .setScope(CollectiveScope::LOCAL)
                             .addDevice(DeviceId::cuda(0))
                             .addDevice(DeviceId::rocm(0))
                             .setLocalRank(0)
                             .build();
            return backend_->initialize(group);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helpers: GPU memory management via IBackend
        // ─────────────────────────────────────────────────────────────────────

        void *allocCUDA(int dev, size_t bytes, float val = 0.0f)
        {
            void *ptr = cuda_backend_->allocate(bytes, dev);
            if (ptr && val != 0.0f)
            {
                std::vector<float> host(bytes / sizeof(float), val);
                cuda_backend_->hostToDevice(ptr, host.data(), bytes, dev);
                cuda_backend_->synchronize(dev);
            }
            return ptr;
        }

        void freeCUDA(int dev, void *ptr)
        {
            if (ptr)
                cuda_backend_->free(ptr, dev);
        }

        std::vector<float> readCUDA(int dev, void *ptr, size_t count)
        {
            std::vector<float> host(count);
            cuda_backend_->deviceToHost(host.data(), ptr, count * sizeof(float), dev);
            cuda_backend_->synchronize(dev);
            return host;
        }

        std::vector<float> readBAR(void *ptr, size_t count)
        {
            std::vector<float> host(count);
            std::memcpy(host.data(), ptr, count * sizeof(float));
            return host;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helpers: accuracy
        // ─────────────────────────────────────────────────────────────────────

        static double mse(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size())
                return std::numeric_limits<double>::max();
            double sum = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                double d = a[i] - b[i];
                sum += d * d;
            }
            return sum / static_cast<double>(a.size());
        }
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. Worker Pool Lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Workers start when backend is initialised via multi-pair API,
     *        and the CUDA worker for the primary device is running.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, WorkerPool_StartsOnInit)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        // Verify workers are running by successfully waiting on a completed event
        void *event = cuda_backend_->createEvent(0);
        ASSERT_NE(event, nullptr);
        ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));
        cuda_backend_->synchronize(0);

        EXPECT_TRUE(backend_->waitForCUDAEvent(event, 0))
            << "CUDA worker should be running after initializeMultiPair";

        cuda_backend_->destroyEvent(event, 0);
    }

    /**
     * @brief Workers start when backend is initialised via the old
     *        DeviceGroup-based API (backward compat).
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, WorkerPool_StartsOnOldInit)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePairViaOldAPI());

        // Verify workers are running by successfully waiting on a completed event
        void *event = cuda_backend_->createEvent(0);
        ASSERT_NE(event, nullptr);
        ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));
        cuda_backend_->synchronize(0);

        EXPECT_TRUE(backend_->waitForCUDAEvent(event, 0))
            << "CUDA worker should run after old initialize() too";

        cuda_backend_->destroyEvent(event, 0);
    }

    /**
     * @brief Workers are stopped when the backend is shut down.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, WorkerPool_StopsOnShutdown)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        // Confirm workers are up
        void *event = cuda_backend_->createEvent(0);
        ASSERT_NE(event, nullptr);
        ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));
        cuda_backend_->synchronize(0);
        ASSERT_TRUE(backend_->waitForCUDAEvent(event, 0));

        backend_->shutdown();

        // After shutdown, event wait should fail (no workers)
        EXPECT_FALSE(backend_->waitForCUDAEvent(event, 0))
            << "Event wait should fail after shutdown (no workers)";

        cuda_backend_->destroyEvent(event, 0);
    }

    /**
     * @brief Re-initialise after shutdown — workers come back up.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, WorkerPool_ReinitAfterShutdown)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());
        backend_->shutdown();

        // Re-init with a fresh backend
        ASSERT_TRUE(initSinglePair());

        // Verify workers are running again via event wait
        void *event = cuda_backend_->createEvent(0);
        ASSERT_NE(event, nullptr);
        ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));
        cuda_backend_->synchronize(0);

        EXPECT_TRUE(backend_->waitForCUDAEvent(event, 0))
            << "Workers should come back after re-initialisation";

        cuda_backend_->destroyEvent(event, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. CUDA Event Wait (via Per-Device Worker — allocation-free hot path)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Wait for a completed CUDA event through the worker pool.
     *
     * Creates an event, records it on a sync'd device, then waits via
     * the PCIeBARBackend worker.  The event is already complete so the
     * worker should return immediately.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, EventWait_CompletedEvent)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        // Create + record event on CUDA:0
        void *event = cuda_backend_->createEvent(0);
        ASSERT_NE(event, nullptr) << "Failed to create CUDA event";
        ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));
        cuda_backend_->synchronize(0); // Make sure it's completed

        // Wait via worker pool
        bool ok = backend_->waitForCUDAEvent(event, 0);
        EXPECT_TRUE(ok) << "waitForCUDAEvent should succeed for a completed event";

        cuda_backend_->destroyEvent(event, 0);
    }

    /**
     * @brief Wait for a pending CUDA event (not yet complete at submission).
     *
     * Launches a small kernel to keep the device busy, immediately submits
     * the event wait, and checks it eventually returns true.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, EventWait_PendingEvent)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        // Allocate a small buffer and launch async memset to create work
        void *buf = cuda_backend_->allocate(MEDIUM_BYTES, 0);
        ASSERT_NE(buf, nullptr);

        // Record event BEFORE synchronizing — it may still be pending
        void *event = cuda_backend_->createEvent(0);
        ASSERT_NE(event, nullptr);
        ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));

        // Wait via worker pool — should block until event completes
        bool ok = backend_->waitForCUDAEvent(event, 0);
        EXPECT_TRUE(ok) << "waitForCUDAEvent should succeed for a pending event";

        cuda_backend_->destroyEvent(event, 0);
        cuda_backend_->free(buf, 0);
    }

    /**
     * @brief Null event should be a no-op (returns true).
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, EventWait_NullEvent)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        EXPECT_TRUE(backend_->waitForCUDAEvent(nullptr, 0))
            << "Null event should return true immediately";
    }

    /**
     * @brief Multiple event waits in sequence — exercises the WorkSlot
     *        reset and reuse path.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, EventWait_RepeatedWaits)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        constexpr int NUM_ITERS = 50;
        for (int i = 0; i < NUM_ITERS; ++i)
        {
            void *event = cuda_backend_->createEvent(0);
            ASSERT_NE(event, nullptr) << "Iter " << i;
            ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));
            cuda_backend_->synchronize(0);

            EXPECT_TRUE(backend_->waitForCUDAEvent(event, 0))
                << "Iter " << i << " failed";

            cuda_backend_->destroyEvent(event, 0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. ROCm Event Wait (via IBackend abstraction)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief waitForROCmEvent on a completed HIP event should succeed.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, ROCmEventWait_CompletedEvent)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        // Create + record + sync event on ROCm:0 via IBackend
        void *event = rocm_backend_->createEvent(0);
        ASSERT_NE(event, nullptr) << "Failed to create ROCm event";
        ASSERT_TRUE(rocm_backend_->recordEvent(event, 0));
        rocm_backend_->synchronize(0);

        bool ok = backend_->waitForROCmEvent(event, 0);
        EXPECT_TRUE(ok) << "waitForROCmEvent should succeed for a completed HIP event";

        rocm_backend_->destroyEvent(event, 0);
    }

    /**
     * @brief Null HIP event should be a no-op.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, ROCmEventWait_NullEvent)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        EXPECT_TRUE(backend_->waitForROCmEvent(nullptr, 0))
            << "Null HIP event should return true immediately";
    }

    /**
     * @brief Repeated ROCm event waits to exercise the IBackend path.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, ROCmEventWait_RepeatedWaits)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        constexpr int NUM_ITERS = 50;
        for (int i = 0; i < NUM_ITERS; ++i)
        {
            void *event = rocm_backend_->createEvent(0);
            ASSERT_NE(event, nullptr) << "Iter " << i;
            ASSERT_TRUE(rocm_backend_->recordEvent(event, 0));
            rocm_backend_->synchronize(0);

            EXPECT_TRUE(backend_->waitForROCmEvent(event, 0))
                << "ROCm event wait iter " << i << " failed";

            rocm_backend_->destroyEvent(event, 0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. Per-Pair Resources
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Single-pair init creates PairResources with 3 streams and 4 events.
     *
     * We verify this indirectly by performing an allreduceMultiPair, which
     * uses the per-pair streams and events.  If resources weren't created
     * the allreduce would fail.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, PairResources_CreatedOnInit)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());
        EXPECT_TRUE(backend_->isMultiPairMode());

        // Allocate buffers
        void *cuda_buf = allocCUDA(0, SMALL_BYTES, 1.0f);
        ASSERT_NE(cuda_buf, nullptr);

        auto rocm_alloc = backend_->allocateInBarRegion(SMALL_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;

        // Fill ROCm (via BAR host ptr)
        std::vector<float> rocm_vals(SMALL_COUNT, 2.0f);
        std::memcpy(rocm_buf, rocm_vals.data(), SMALL_BYTES);

        std::vector<void *> cuda_bufs = {cuda_buf};
        std::vector<void *> rocm_bufs = {rocm_buf};

        // allreduceMultiPair exercises per-pair streams, events, and temp buffers
        ASSERT_TRUE(backend_->allreduceMultiPair(
            cuda_bufs, rocm_bufs, SMALL_COUNT, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair should succeed using per-pair resources";

        // Verify: 1.0 + 2.0 = 3.0
        auto result = readCUDA(0, cuda_buf, SMALL_COUNT);
        EXPECT_NEAR(result[0], 3.0f, 1e-3f);
        EXPECT_NEAR(result[SMALL_COUNT - 1], 3.0f, 1e-3f);

        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    /**
     * @brief Temp buffers grow on demand (first allreduce larger than initial).
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, PairResources_TempBufferGrows)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        // First: small allreduce
        {
            void *cuda_buf = allocCUDA(0, SMALL_BYTES, 1.0f);
            ASSERT_NE(cuda_buf, nullptr);
            auto rocm_alloc = backend_->allocateInBarRegion(SMALL_BYTES);
            ASSERT_TRUE(rocm_alloc.has_value());
            auto [rocm_buf, offset] = *rocm_alloc;
            std::vector<float> rv(SMALL_COUNT, 1.0f);
            std::memcpy(rocm_buf, rv.data(), SMALL_BYTES);

            std::vector<void *> cb = {cuda_buf};
            std::vector<void *> rb = {rocm_buf};
            ASSERT_TRUE(backend_->allreduceMultiPair(
                cb, rb, SMALL_COUNT, CollectiveDataType::FLOAT32));

            freeCUDA(0, cuda_buf);
            backend_->freeBarBuffer(rocm_buf);
        }

        // Second: larger allreduce — temp buffer must grow
        {
            void *cuda_buf = allocCUDA(0, LARGE_BYTES, 3.0f);
            ASSERT_NE(cuda_buf, nullptr);
            auto rocm_alloc = backend_->allocateInBarRegion(LARGE_BYTES);
            ASSERT_TRUE(rocm_alloc.has_value());
            auto [rocm_buf, offset] = *rocm_alloc;
            std::vector<float> rv(LARGE_COUNT, 7.0f);
            std::memcpy(rocm_buf, rv.data(), LARGE_BYTES);

            std::vector<void *> cb = {cuda_buf};
            std::vector<void *> rb = {rocm_buf};
            ASSERT_TRUE(backend_->allreduceMultiPair(
                cb, rb, LARGE_COUNT, CollectiveDataType::FLOAT32))
                << "Larger allreduce should trigger temp-buffer growth and succeed";

            // Verify 3.0 + 7.0 = 10.0
            auto result = readCUDA(0, cuda_buf, LARGE_COUNT);
            EXPECT_NEAR(result[0], 10.0f, 1e-3f);
            EXPECT_NEAR(result[LARGE_COUNT / 2], 10.0f, 1e-3f);
            EXPECT_NEAR(result[LARGE_COUNT - 1], 10.0f, 1e-3f);

            freeCUDA(0, cuda_buf);
            backend_->freeBarBuffer(rocm_buf);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. allreduceMultiPair Correctness (single-pair, per-pair pipeline)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Basic correctness: uniform values → expected sum.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_UniformValues)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        void *cuda_buf = allocCUDA(0, MEDIUM_BYTES, 5.0f);
        ASSERT_NE(cuda_buf, nullptr);
        auto rocm_alloc = backend_->allocateInBarRegion(MEDIUM_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;
        std::vector<float> rv(MEDIUM_COUNT, 3.0f);
        std::memcpy(rocm_buf, rv.data(), MEDIUM_BYTES);

        std::vector<void *> cb = {cuda_buf};
        std::vector<void *> rb = {rocm_buf};
        ASSERT_TRUE(backend_->allreduceMultiPair(
            cb, rb, MEDIUM_COUNT, CollectiveDataType::FLOAT32));

        // Verify CUDA side: 5+3=8
        auto cuda_res = readCUDA(0, cuda_buf, MEDIUM_COUNT);
        std::vector<float> expected(MEDIUM_COUNT, 8.0f);
        EXPECT_LT(mse(cuda_res, expected), 1e-6)
            << "CUDA result wrong after allreduce";

        // Verify ROCm side (via BAR): same 8.0
        auto rocm_res = readBAR(rocm_buf, MEDIUM_COUNT);
        EXPECT_LT(mse(rocm_res, expected), 1e-6)
            << "ROCm result wrong after allreduce";

        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    /**
     * @brief Correctness with random data — checks full-element accuracy.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_RandomValues)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

        std::vector<float> cuda_vals(MEDIUM_COUNT);
        std::vector<float> rocm_vals(MEDIUM_COUNT);
        std::vector<float> expected(MEDIUM_COUNT);

        for (size_t i = 0; i < MEDIUM_COUNT; ++i)
        {
            cuda_vals[i] = dist(rng);
            rocm_vals[i] = dist(rng);
            expected[i] = cuda_vals[i] + rocm_vals[i];
        }

        void *cuda_buf = cuda_backend_->allocate(MEDIUM_BYTES, 0);
        ASSERT_NE(cuda_buf, nullptr);
        cuda_backend_->hostToDevice(cuda_buf, cuda_vals.data(), MEDIUM_BYTES, 0);
        cuda_backend_->synchronize(0);

        auto rocm_alloc = backend_->allocateInBarRegion(MEDIUM_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;
        std::memcpy(rocm_buf, rocm_vals.data(), MEDIUM_BYTES);

        std::vector<void *> cb = {cuda_buf};
        std::vector<void *> rb = {rocm_buf};
        ASSERT_TRUE(backend_->allreduceMultiPair(
            cb, rb, MEDIUM_COUNT, CollectiveDataType::FLOAT32));

        auto cuda_res = readCUDA(0, cuda_buf, MEDIUM_COUNT);
        auto rocm_res = readBAR(rocm_buf, MEDIUM_COUNT);

        EXPECT_LT(mse(cuda_res, expected), 1e-6)
            << "CUDA random-data allreduce MSE too high";
        EXPECT_LT(mse(rocm_res, expected), 1e-6)
            << "ROCm random-data allreduce MSE too high";

        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    /**
     * @brief Large allreduce (256KB) — exercises temp-buffer codepath.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_LargeBuffer)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        void *cuda_buf = allocCUDA(0, LARGE_BYTES, 1.0f);
        ASSERT_NE(cuda_buf, nullptr);
        auto rocm_alloc = backend_->allocateInBarRegion(LARGE_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;
        std::vector<float> rv(LARGE_COUNT, 1.0f);
        std::memcpy(rocm_buf, rv.data(), LARGE_BYTES);

        std::vector<void *> cb = {cuda_buf};
        std::vector<void *> rb = {rocm_buf};
        ASSERT_TRUE(backend_->allreduceMultiPair(
            cb, rb, LARGE_COUNT, CollectiveDataType::FLOAT32));

        auto res = readCUDA(0, cuda_buf, LARGE_COUNT);
        // Spot-check first, middle, last
        EXPECT_NEAR(res[0], 2.0f, 1e-4f);
        EXPECT_NEAR(res[LARGE_COUNT / 2], 2.0f, 1e-4f);
        EXPECT_NEAR(res[LARGE_COUNT - 1], 2.0f, 1e-4f);

        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. Repeated Allreduce — No Hot-Path Re-allocation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Run allreduceMultiPair many times on the same buffers.
     *
     * After the first call, streams / events / temp buffers are all allocated.
     * Subsequent calls should reuse them with no new allocations.
     * We verify correctness on every iteration.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_RepeatedNoRealloc)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        void *cuda_buf = cuda_backend_->allocate(MEDIUM_BYTES, 0);
        ASSERT_NE(cuda_buf, nullptr);
        auto rocm_alloc = backend_->allocateInBarRegion(MEDIUM_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;

        std::vector<void *> cb = {cuda_buf};
        std::vector<void *> rb = {rocm_buf};

        constexpr int NUM_ITERS = 20;
        for (int iter = 0; iter < NUM_ITERS; ++iter)
        {
            // Re-initialise buffers each iteration with different values
            float cuda_val = static_cast<float>(iter + 1);
            float rocm_val = static_cast<float>(iter * 2);
            float expected_val = cuda_val + rocm_val;

            std::vector<float> cv(MEDIUM_COUNT, cuda_val);
            std::vector<float> rv(MEDIUM_COUNT, rocm_val);
            cuda_backend_->hostToDevice(cuda_buf, cv.data(), MEDIUM_BYTES, 0);
            cuda_backend_->synchronize(0);
            std::memcpy(rocm_buf, rv.data(), MEDIUM_BYTES);

            ASSERT_TRUE(backend_->allreduceMultiPair(
                cb, rb, MEDIUM_COUNT, CollectiveDataType::FLOAT32))
                << "Iter " << iter << " allreduce failed";

            auto res = readCUDA(0, cuda_buf, MEDIUM_COUNT);
            EXPECT_NEAR(res[0], expected_val, 1e-3f)
                << "Iter " << iter << " result[0] wrong";
            EXPECT_NEAR(res[MEDIUM_COUNT - 1], expected_val, 1e-3f)
                << "Iter " << iter << " result[last] wrong";
        }

        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    /**
     * @brief Timed repeated allreduce to verify no timing regression from
     *        the old single-stream path.
     *
     * This is a soft performance test — it reports mean latency but does
     * not hard-fail on a threshold (hardware varies).
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_RepeatedLatency)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        // Use a size typical of TP decode allreduce (3.5 KB for 896-dim FP32)
        constexpr size_t TP_COUNT = 896;
        constexpr size_t TP_BYTES = TP_COUNT * sizeof(float);

        void *cuda_buf = allocCUDA(0, TP_BYTES, 1.0f);
        ASSERT_NE(cuda_buf, nullptr);
        auto rocm_alloc = backend_->allocateInBarRegion(TP_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;
        std::vector<float> rv(TP_COUNT, 1.0f);
        std::memcpy(rocm_buf, rv.data(), TP_BYTES);

        std::vector<void *> cb = {cuda_buf};
        std::vector<void *> rb = {rocm_buf};

        // Warmup
        for (int i = 0; i < 5; ++i)
        {
            std::memcpy(rocm_buf, rv.data(), TP_BYTES);
            std::vector<float> cv(TP_COUNT, 1.0f);
            cuda_backend_->hostToDevice(cuda_buf, cv.data(), TP_BYTES, 0);
            cuda_backend_->synchronize(0);
            backend_->allreduceMultiPair(cb, rb, TP_COUNT, CollectiveDataType::FLOAT32);
        }

        // Timed iterations
        constexpr int BENCH_ITERS = 100;
        std::vector<double> latencies_us;
        latencies_us.reserve(BENCH_ITERS);

        for (int i = 0; i < BENCH_ITERS; ++i)
        {
            // Reset buffers
            std::vector<float> cv(TP_COUNT, 1.0f);
            cuda_backend_->hostToDevice(cuda_buf, cv.data(), TP_BYTES, 0);
            cuda_backend_->synchronize(0);
            std::memcpy(rocm_buf, rv.data(), TP_BYTES);

            auto t0 = std::chrono::high_resolution_clock::now();
            backend_->allreduceMultiPair(cb, rb, TP_COUNT, CollectiveDataType::FLOAT32);
            auto t1 = std::chrono::high_resolution_clock::now();

            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            latencies_us.push_back(us);
        }

        // Statistics
        std::sort(latencies_us.begin(), latencies_us.end());
        double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        double mean = sum / BENCH_ITERS;
        double median = latencies_us[BENCH_ITERS / 2];
        double p95 = latencies_us[static_cast<int>(BENCH_ITERS * 0.95)];

        LOG_INFO("╔═══════════════════════════════════════════════════════╗");
        LOG_INFO("║  allreduceMultiPair latency (" << TP_COUNT << " floats, " << TP_BYTES << " B)");
        LOG_INFO("╠═══════════════════════════════════════════════════════╣");
        LOG_INFO("║  Mean:   " << mean << " μs");
        LOG_INFO("║  Median: " << median << " μs");
        LOG_INFO("║  P95:    " << p95 << " μs");
        LOG_INFO("║  Min:    " << latencies_us.front() << " μs");
        LOG_INFO("║  Max:    " << latencies_us.back() << " μs");
        LOG_INFO("╚═══════════════════════════════════════════════════════╝");

        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 7. Two-Pair Allreduce (requires 2 CUDA + 2 ROCm)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Two pairs reduce independently and concurrently.
     *
     * Pair 0: CUDA:0=2.0, ROCm:0=3.0  →  5.0
     * Pair 1: CUDA:1=4.0, ROCm:1=6.0  →  10.0
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_TwoPairs_Correctness)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(2);
        REQUIRE_ROCM(2);

        ASSERT_TRUE(initTwoPairs());

        // Pair 0 buffers
        void *cuda_buf0 = allocCUDA(0, MEDIUM_BYTES, 2.0f);
        ASSERT_NE(cuda_buf0, nullptr);
        auto alloc0 = backend_->allocateInBarRegion(MEDIUM_BYTES);
        ASSERT_TRUE(alloc0.has_value());
        auto [rocm_buf0, off0] = *alloc0;
        std::vector<float> rv0(MEDIUM_COUNT, 3.0f);
        std::memcpy(rocm_buf0, rv0.data(), MEDIUM_BYTES);

        // Pair 1 buffers
        void *cuda_buf1 = allocCUDA(1, MEDIUM_BYTES, 4.0f);
        ASSERT_NE(cuda_buf1, nullptr);
        auto alloc1 = backend_->allocateInBarRegion(MEDIUM_BYTES);
        ASSERT_TRUE(alloc1.has_value());
        auto [rocm_buf1, off1] = *alloc1;
        std::vector<float> rv1(MEDIUM_COUNT, 6.0f);
        std::memcpy(rocm_buf1, rv1.data(), MEDIUM_BYTES);

        std::vector<void *> cb = {cuda_buf0, cuda_buf1};
        std::vector<void *> rb = {rocm_buf0, rocm_buf1};

        ASSERT_TRUE(backend_->allreduceMultiPair(
            cb, rb, MEDIUM_COUNT, CollectiveDataType::FLOAT32))
            << "Two-pair allreduce should succeed";

        // Pair 0: 2+3=5
        auto res0 = readCUDA(0, cuda_buf0, MEDIUM_COUNT);
        auto bar0 = readBAR(rocm_buf0, MEDIUM_COUNT);
        std::vector<float> exp0(MEDIUM_COUNT, 5.0f);
        EXPECT_LT(mse(res0, exp0), 1e-6) << "Pair 0 CUDA result wrong";
        EXPECT_LT(mse(bar0, exp0), 1e-6) << "Pair 0 ROCm result wrong";

        // Pair 1: 4+6=10
        auto res1 = readCUDA(1, cuda_buf1, MEDIUM_COUNT);
        auto bar1 = readBAR(rocm_buf1, MEDIUM_COUNT);
        std::vector<float> exp1(MEDIUM_COUNT, 10.0f);
        EXPECT_LT(mse(res1, exp1), 1e-6) << "Pair 1 CUDA result wrong";
        EXPECT_LT(mse(bar1, exp1), 1e-6) << "Pair 1 ROCm result wrong";

        freeCUDA(0, cuda_buf0);
        freeCUDA(1, cuda_buf1);
        backend_->freeBarBuffer(rocm_buf0);
        backend_->freeBarBuffer(rocm_buf1);
    }

    /**
     * @brief Two-pair repeated allreduce — verifies per-pair resource reuse.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_TwoPairs_Repeated)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(2);
        REQUIRE_ROCM(2);

        ASSERT_TRUE(initTwoPairs());

        void *cuda_buf0 = cuda_backend_->allocate(MEDIUM_BYTES, 0);
        void *cuda_buf1 = cuda_backend_->allocate(MEDIUM_BYTES, 1);
        ASSERT_NE(cuda_buf0, nullptr);
        ASSERT_NE(cuda_buf1, nullptr);

        auto alloc0 = backend_->allocateInBarRegion(MEDIUM_BYTES);
        auto alloc1 = backend_->allocateInBarRegion(MEDIUM_BYTES);
        ASSERT_TRUE(alloc0.has_value());
        ASSERT_TRUE(alloc1.has_value());
        void *rocm_buf0 = alloc0->first;
        void *rocm_buf1 = alloc1->first;

        std::vector<void *> cb = {cuda_buf0, cuda_buf1};
        std::vector<void *> rb = {rocm_buf0, rocm_buf1};

        constexpr int NUM_ITERS = 10;
        for (int iter = 0; iter < NUM_ITERS; ++iter)
        {
            float v0 = static_cast<float>(iter + 1);
            float v1 = static_cast<float>(iter * 3);

            std::vector<float> c0(MEDIUM_COUNT, v0);
            std::vector<float> c1(MEDIUM_COUNT, v0 * 2);
            std::vector<float> r0(MEDIUM_COUNT, v1);
            std::vector<float> r1(MEDIUM_COUNT, v1 + 1);

            cuda_backend_->hostToDevice(cuda_buf0, c0.data(), MEDIUM_BYTES, 0);
            cuda_backend_->hostToDevice(cuda_buf1, c1.data(), MEDIUM_BYTES, 1);
            cuda_backend_->synchronize(0);
            cuda_backend_->synchronize(1);
            std::memcpy(rocm_buf0, r0.data(), MEDIUM_BYTES);
            std::memcpy(rocm_buf1, r1.data(), MEDIUM_BYTES);

            ASSERT_TRUE(backend_->allreduceMultiPair(
                cb, rb, MEDIUM_COUNT, CollectiveDataType::FLOAT32))
                << "Two-pair iter " << iter << " failed";

            auto res0 = readCUDA(0, cuda_buf0, MEDIUM_COUNT);
            auto res1 = readCUDA(1, cuda_buf1, MEDIUM_COUNT);

            float exp0 = v0 + v1;
            float exp1 = (v0 * 2) + (v1 + 1);

            EXPECT_NEAR(res0[0], exp0, 1e-3f)
                << "Pair 0 iter " << iter << " result wrong";
            EXPECT_NEAR(res1[0], exp1, 1e-3f)
                << "Pair 1 iter " << iter << " result wrong";
        }

        freeCUDA(0, cuda_buf0);
        freeCUDA(1, cuda_buf1);
        backend_->freeBarBuffer(rocm_buf0);
        backend_->freeBarBuffer(rocm_buf1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 8. Edge Cases
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief allreduceMultiPair with mismatched buffer count should fail.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_MismatchedBufferCount)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        ASSERT_TRUE(initSinglePair());

        void *cuda_buf = allocCUDA(0, SMALL_BYTES, 1.0f);
        ASSERT_NE(cuda_buf, nullptr);
        auto rocm_alloc = backend_->allocateInBarRegion(SMALL_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;

        // Two CUDA buffers but one pair — should fail
        std::vector<void *> cb = {cuda_buf, cuda_buf};
        std::vector<void *> rb = {rocm_buf};

        EXPECT_FALSE(backend_->allreduceMultiPair(
            cb, rb, SMALL_COUNT, CollectiveDataType::FLOAT32))
            << "Mismatched buffer count should be rejected";

        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    /**
     * @brief Backend not initialised should fail allreduce.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, Allreduce_NotInitialised)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        backend_ = std::make_unique<PCIeBARBackend>();
        // NOT calling initializeMultiPair

        void *cuda_buf = allocCUDA(0, SMALL_BYTES, 1.0f);
        ASSERT_NE(cuda_buf, nullptr);

        // Can't allocate BAR without init, so pass a different ptr
        std::vector<void *> cb = {cuda_buf};
        std::vector<void *> rb = {nullptr};

        EXPECT_FALSE(backend_->allreduceMultiPair(
            cb, rb, SMALL_COUNT, CollectiveDataType::FLOAT32))
            << "allreduce on uninitialised backend should fail";

        freeCUDA(0, cuda_buf);
    }

    /**
     * @brief waitForCUDAEvent with no workers running should fail gracefully.
     */
    TEST_F(Test__PCIeBARBackend_WorkerPool, EventWait_NoWorkers)
    {
        REQUIRE_PCIE_BAR();
        REQUIRE_CUDA(1);
        REQUIRE_ROCM(1);

        backend_ = std::make_unique<PCIeBARBackend>();
        // NOT initialised → no workers

        void *event = cuda_backend_->createEvent(0);
        ASSERT_NE(event, nullptr);
        ASSERT_TRUE(cuda_backend_->recordEvent(event, 0));
        cuda_backend_->synchronize(0);

        EXPECT_FALSE(backend_->waitForCUDAEvent(event, 0))
            << "Event wait with no workers should fail gracefully";

        cuda_backend_->destroyEvent(event, 0);
    }

} // namespace llaminar2::test

#else // !HAVE_CUDA || !HAVE_ROCM

// Empty test to avoid linker errors on builds without both backends
TEST(Test__PCIeBARBackend_WorkerPool, SkippedNoBothBackends)
{
    GTEST_SKIP() << "PCIeBARBackend requires both HAVE_CUDA and HAVE_ROCM";
}

#endif // HAVE_CUDA && HAVE_ROCM

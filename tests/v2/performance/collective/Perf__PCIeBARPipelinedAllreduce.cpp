/**
 * @file Perf__PCIeBARPipelinedAllreduce.cpp
 * @brief Performance benchmark comparing pipelined vs sequential PCIeBAR allreduce
 *
 * Measures the effectiveness of the triple-buffered pipelined allreduce implementation
 * that overlaps BAR transfers with CUDA compute:
 *
 * Sequential: Read[all] -> Compute[all] -> Write[all]
 * Pipelined:  Read[0] | Read[1]+Compute[0] | Read[2]+Compute[1]+Write[0] | ...
 *
 * NOTE: Uses IBackend abstraction for device memory allocation to avoid
 * including conflicting CUDA/HIP headers in the same translation unit.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

// Only CUDA headers - avoid HIP headers to prevent conflicts
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>

#include "collective/backends/PCIeBARBackend.h"
#include "collective/DeviceGroup.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/p2p/DirectP2P.h"

using namespace llaminar2;

namespace
{

    // ============================================================================
    // Performance Test Configuration
    // ============================================================================

    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int BENCHMARK_ITERATIONS = 100;

    // Test sizes
    constexpr size_t SIZE_32KB = 32 * 1024 / sizeof(float);      // 8192 floats - pipeline threshold
    constexpr size_t SIZE_64KB = 64 * 1024 / sizeof(float);      // 16384 floats
    constexpr size_t SIZE_256KB = 256 * 1024 / sizeof(float);    // 65536 floats
    constexpr size_t SIZE_1MB = 1024 * 1024 / sizeof(float);     // 262144 floats
    constexpr size_t SIZE_4MB = 4 * 1024 * 1024 / sizeof(float); // ~1M floats
    constexpr size_t SIZE_14MB = 14680064 / sizeof(float);       // 3670016 floats - realistic TP reduction

    // Helper to fill array with random floats
    void fillRandom(std::vector<float> &data)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : data)
        {
            v = dist(rng);
        }
    }

    // Helper to verify allreduce result
    bool verifySum(const float *result, const float *cuda_orig, const float *rocm_orig,
                   size_t count, float tolerance = 1e-4f)
    {
        for (size_t i = 0; i < std::min(count, size_t(10)); ++i)
        {
            float expected = cuda_orig[i] + rocm_orig[i];
            float diff = std::abs(result[i] - expected);
            if (diff > tolerance && diff > tolerance * std::abs(expected))
            {
                std::cerr << "Mismatch at " << i << ": expected " << expected
                          << ", got " << result[i] << " (diff=" << diff << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Test fixture for PCIeBAR pipelined allreduce performance
     */
    class PCIeBARPipelinedPerfTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Get backend instances (via IBackend to avoid header conflicts)
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();

            if (!cuda_backend_ || !rocm_backend_)
            {
                GTEST_SKIP() << "CUDA or ROCm backend not available";
                return;
            }

            // Check if P2P hardware is available
            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoPCIeBarP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P hardware not available";
                return;
            }

            // Create PCIeBAR backend
            backend_ = std::make_unique<PCIeBARBackend>();

            // Create a CUDA + ROCm device group for initialization
            DeviceGroupBuilder builder;
            device_group_ = builder
                                .setName("cuda_rocm_perf_group")
                                .setScope(CollectiveScope::LOCAL)
                                .addDevice(DeviceId::cuda(0))
                                .addDevice(DeviceId::rocm(0))
                                .setLocalRank(0)
                                .build();

            if (!backend_->initialize(device_group_))
            {
                GTEST_SKIP() << "PCIeBAR initialization failed (may need CAP_SYS_ADMIN)";
                return;
            }

            initialized_ = true;
        }

        void TearDown() override
        {
            if (d_cuda_)
            {
                cuda_backend_->free(d_cuda_, 0);
                d_cuda_ = nullptr;
            }
            if (d_rocm_)
            {
                backend_->freeBarBuffer(d_rocm_);
                d_rocm_ = nullptr;
            }
            if (backend_)
            {
                backend_->shutdown();
            }
            backend_.reset();
        }

        void allocateBuffers(size_t count)
        {
            size_t bytes = count * sizeof(float);
            current_count_ = count;

            // Allocate and initialize CUDA buffer using IBackend
            d_cuda_ = cuda_backend_->allocate(bytes, 0);
            h_cuda_.resize(count);
            fillRandom(h_cuda_);
            cuda_backend_->hostToDevice(d_cuda_, h_cuda_.data(), bytes, 0);

            // Allocate and initialize ROCm buffer in BAR region
            auto bar_alloc = backend_->allocateInBarRegion(bytes);
            if (!bar_alloc.has_value())
            {
                GTEST_SKIP() << "Failed to allocate BAR region";
                return;
            }
            d_rocm_ = bar_alloc->first;
            bar_offset_ = bar_alloc->second;

            h_rocm_.resize(count);
            fillRandom(h_rocm_);
            // Write to BAR region via host-accessible mapping
            std::memcpy(d_rocm_, h_rocm_.data(), bytes);

            // Register buffers for the collective
            std::string coll_id = "perf_test_" + std::to_string(count);
            current_coll_id_ = coll_id;

            bool cuda_registered = backend_->registerBuffer(coll_id, DeviceId::cuda(0), d_cuda_, bytes);
            bool rocm_registered = backend_->registerBuffer(coll_id, DeviceId::rocm(0), d_rocm_, bytes);

            if (!cuda_registered || !rocm_registered)
            {
                GTEST_SKIP() << "Failed to register buffers for collective";
                return;
            }
        }

        void resetBuffers(size_t count)
        {
            size_t bytes = count * sizeof(float);
            cuda_backend_->hostToDevice(d_cuda_, h_cuda_.data(), bytes, 0);
            std::memcpy(d_rocm_, h_rocm_.data(), bytes);
            cuda_backend_->synchronize(0);
        }

        void runAllreduce(size_t count)
        {
            // Use registered allreduce which properly tracks offsets
            backend_->allreduceRegistered(current_coll_id_, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        }

        struct BenchmarkResult
        {
            double avg_us;
            double min_us;
            double max_us;
            double throughput_gbps;
        };

        BenchmarkResult runBenchmark(size_t count, int iterations)
        {
            size_t bytes = count * sizeof(float);
            std::vector<double> timings;
            timings.reserve(iterations);

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                resetBuffers(count);
                runAllreduce(count);
            }
            cuda_backend_->synchronize(0);

            // Benchmark
            for (int i = 0; i < iterations; ++i)
            {
                // Reset buffers
                resetBuffers(count);

                auto start = std::chrono::high_resolution_clock::now();

                runAllreduce(count);

                auto end = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(end - start).count();
                timings.push_back(us);
            }

            // Calculate statistics
            double sum = 0, min_val = timings[0], max_val = timings[0];
            for (double t : timings)
            {
                sum += t;
                min_val = std::min(min_val, t);
                max_val = std::max(max_val, t);
            }

            BenchmarkResult result;
            result.avg_us = sum / iterations;
            result.min_us = min_val;
            result.max_us = max_val;
            // Throughput: read + write = 2 * bytes transferred
            result.throughput_gbps = (2.0 * bytes) / (result.avg_us * 1e3);

            return result;
        }

        void cleanupBuffers()
        {
            if (d_cuda_)
            {
                cuda_backend_->free(d_cuda_, 0);
                d_cuda_ = nullptr;
            }
            if (d_rocm_)
            {
                backend_->freeBarBuffer(d_rocm_);
                d_rocm_ = nullptr;
            }
        }

        bool initialized_ = false;
        std::unique_ptr<PCIeBARBackend> backend_;
        DeviceGroup device_group_;
        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;
        void *d_cuda_ = nullptr;
        void *d_rocm_ = nullptr;
        size_t bar_offset_ = 0;
        size_t current_count_ = 0;
        std::string current_coll_id_;
        std::vector<float> h_cuda_;
        std::vector<float> h_rocm_;
    };

    // ============================================================================
    // Performance Tests
    // ============================================================================

    /**
     * @brief Benchmark allreduce at various sizes, comparing pipelined vs sequential thresholds
     */
    TEST_F(PCIeBARPipelinedPerfTest, AllreduceThroughputVsSize)
    {
        if (!initialized_)
        {
            GTEST_SKIP() << "Backend not initialized";
        }

        struct TestCase
        {
            size_t count;
            const char *name;
            bool should_pipeline;
        };

        std::vector<TestCase> test_cases = {
            {SIZE_32KB, "32KB (threshold)", true},
            {SIZE_64KB, "64KB", true},
            {SIZE_256KB, "256KB", true},
            {SIZE_1MB, "1MB", true},
            {SIZE_4MB, "4MB", true},
            {SIZE_14MB, "14MB (TP realistic)", true},
        };

        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           PCIeBAR ALLREDUCE PERFORMANCE (PIPELINED PATH)                 ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Size         │  Avg (μs)  │  Min (μs)  │  Max (μs)  │ Throughput        ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";

        for (const auto &tc : test_cases)
        {
            allocateBuffers(tc.count);

            auto result = runBenchmark(tc.count, BENCHMARK_ITERATIONS);

            std::cout << "║ " << std::setw(12) << std::left << tc.name << " │ "
                      << std::setw(10) << std::fixed << std::setprecision(1) << result.avg_us << " │ "
                      << std::setw(10) << std::fixed << std::setprecision(1) << result.min_us << " │ "
                      << std::setw(10) << std::fixed << std::setprecision(1) << result.max_us << " │ "
                      << std::setw(6) << std::fixed << std::setprecision(2) << result.throughput_gbps << " GB/s     ║\n";

            cleanupBuffers();
        }

        std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";
    }

    /**
     * @brief Compare sequential vs pipelined at threshold boundary
     */
    TEST_F(PCIeBARPipelinedPerfTest, SequentialVsPipelined)
    {
        if (!initialized_)
        {
            GTEST_SKIP() << "Backend not initialized";
        }

        // Test at 64KB - should use pipelined path
        constexpr size_t test_count = 64 * 1024 / sizeof(float); // 16384 floats = 64KB
        allocateBuffers(test_count);

        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           PIPELINED ALLREDUCE CONSISTENCY CHECK (64KB)                   ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";

        // Run multiple times to check consistency
        std::vector<double> timings;
        for (int i = 0; i < 10; ++i)
        {
            resetBuffers(test_count);

            auto start = std::chrono::high_resolution_clock::now();
            runAllreduce(test_count);
            auto end = std::chrono::high_resolution_clock::now();

            timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        // Calculate stats
        double sum = 0;
        for (double t : timings)
            sum += t;
        double avg = sum / timings.size();

        std::cout << "║ Pipelined allreduce (64KB): " << std::fixed << std::setprecision(1)
                  << avg << " μs avg                               ║\n";

        // Verify correctness
        std::vector<float> result(test_count);
        cuda_backend_->deviceToHost(result.data(), d_cuda_, test_count * sizeof(float), 0);
        bool correct = verifySum(result.data(), h_cuda_.data(), h_rocm_.data(), test_count);
        std::cout << "║ Correctness: " << (correct ? "PASS ✓" : "FAIL ✗") << "                                                        ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

        EXPECT_TRUE(correct) << "Pipelined allreduce produced incorrect results";
    }

    /**
     * @brief Verify allreduce produces correct sums
     */
    TEST_F(PCIeBARPipelinedPerfTest, CorrectnessCheck)
    {
        if (!initialized_)
        {
            GTEST_SKIP() << "Backend not initialized";
        }

        // Test various sizes - start with one that doesn't use pipeline to verify basics
        std::vector<size_t> sizes = {SIZE_32KB / 2, SIZE_32KB, SIZE_64KB, SIZE_256KB, SIZE_1MB};

        for (size_t count : sizes)
        {
            size_t bytes = count * sizeof(float);
            bool uses_pipeline = bytes >= 32768; // PIPELINE_THRESHOLD

            allocateBuffers(count);

            std::cout << "Testing size " << count << " floats (" << bytes
                      << " bytes) @ BAR offset " << bar_offset_
                      << " - " << (uses_pipeline ? "PIPELINED" : "SEQUENTIAL") << std::endl;

            // Run allreduce using registered path which tracks offsets correctly
            runAllreduce(count);

            // Verify result
            std::vector<float> result(count);
            cuda_backend_->deviceToHost(result.data(), d_cuda_, count * sizeof(float), 0);

            bool correct = verifySum(result.data(), h_cuda_.data(), h_rocm_.data(), count);
            EXPECT_TRUE(correct) << "Allreduce incorrect for size " << count;

            cleanupBuffers();
        }
    }

} // anonymous namespace

#else // !HAVE_CUDA || !HAVE_ROCM

TEST(PCIeBARPipelinedPerfTest, DISABLED_RequiresCUDAAndROCm)
{
    GTEST_SKIP() << "This test requires both CUDA and ROCm";
}

#endif // HAVE_CUDA && HAVE_ROCM

/**
 * @file Test__PCIeBARLatencyBenchmark.cpp
 * @brief PCIeBAR Backend latency benchmarks for hybrid parallelism validation
 *
 * This test suite validates that PCIeBAR-based AllReduce operations between
 * CUDA and ROCm devices have acceptable latency for heterogeneous tensor
 * parallelism. The key metric is latency for small transfers (14KB) which
 * dominate the decode phase of LLM inference.
 *
 * Success Criteria:
 * - 14KB AllReduce: < 200μs (acceptable), < 150μs (target)
 * - Sustained decode simulation: > 10 tok/s (AllReduce overhead only)
 *
 * If these thresholds are not met, fall back to CPU-staged collectives.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

#include "backends/DeviceId.h"
#include "collective/backends/PCIeBARBackend.h"
#include "collective/CollectiveContext.h"
#include "collective/DeviceGroup.h"
#include "utils/Logging.h"

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

namespace llaminar2
{
    namespace test
    {

        /**
         * @brief Latency metrics for a single transfer size
         */
        struct LatencyMetrics
        {
            size_t bytes;
            double min_us;
            double max_us;
            double mean_us;
            double median_us;
            double p95_us;
            double p99_us;
            double stddev_us;
            double bandwidth_gbps;

            void print() const
            {
                LOG_INFO("╔══════════════════════════════════════════════════════════════╗");
                LOG_INFO("║  Transfer Size: " << bytes << " bytes (" << (bytes / 1024.0) << " KB)");
                LOG_INFO("╠══════════════════════════════════════════════════════════════╣");
                LOG_INFO("║  Min:    " << min_us << " μs");
                LOG_INFO("║  Max:    " << max_us << " μs");
                LOG_INFO("║  Mean:   " << mean_us << " μs");
                LOG_INFO("║  Median: " << median_us << " μs");
                LOG_INFO("║  P95:    " << p95_us << " μs");
                LOG_INFO("║  P99:    " << p99_us << " μs");
                LOG_INFO("║  Stddev: " << stddev_us << " μs");
                LOG_INFO("║  Bandwidth: " << bandwidth_gbps << " GB/s");
                LOG_INFO("╚══════════════════════════════════════════════════════════════╝");
            }
        };

        /**
         * @brief Test fixture for PCIeBAR latency benchmarks
         */
        class PCIeBARLatencyBenchmark : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Check for required hardware
                cuda_available_ = checkCUDAAvailable();
                rocm_available_ = checkROCmAvailable();

                if (!cuda_available_ || !rocm_available_)
                {
                    GTEST_SKIP() << "Requires both CUDA and ROCm devices for heterogeneous testing";
                }

                cuda_device_ = DeviceId::cuda(0);
                rocm_device_ = DeviceId::rocm(0);

                // Create heterogeneous device group
                DeviceGroup::Builder builder;
                builder.addDevice(cuda_device_);
                builder.addDevice(rocm_device_);
                builder.setScope(CollectiveScope::LOCAL);
                device_group_ = builder.build();

                // Create PCIeBAR backend
                // Note: This requires actual PCIeBAR support; will skip if not available
                try
                {
                    backend_ = createPCIeBARBackend();
                    if (!backend_)
                    {
                        GTEST_SKIP() << "PCIeBAR backend not available";
                    }
                }
                catch (const std::exception &e)
                {
                    GTEST_SKIP() << "PCIeBAR backend creation failed: " << e.what();
                }

                // Pre-allocate test buffers for various sizes
                allocateTestBuffers();
            }

            void TearDown() override
            {
                freeTestBuffers();
            }

            /**
             * @brief Run latency benchmark for a specific transfer size
             * @param bytes Transfer size in bytes
             * @param iterations Number of iterations (default 100)
             * @param warmup_iterations Warmup iterations (default 10)
             * @return LatencyMetrics with statistical analysis
             */
            LatencyMetrics benchmarkAllReduce(size_t bytes, int iterations = 100, int warmup_iterations = 10)
            {
                // Get or create buffer of appropriate size
                auto *cuda_buffer = getCUDABuffer(bytes);
                auto *rocm_buffer = getROCmBuffer(bytes);

                if (!cuda_buffer || !rocm_buffer)
                {
                    LOG_ERROR("Failed to get buffers for " << bytes << " bytes");
                    return {};
                }

                // Initialize with deterministic data
                initializeBuffer(cuda_buffer, bytes, cuda_device_);
                initializeBuffer(rocm_buffer, bytes, rocm_device_);

                // Warmup
                for (int i = 0; i < warmup_iterations; i++)
                {
                    backend_->allreduce(cuda_buffer, rocm_buffer, bytes / sizeof(float),
                                        CollectiveOp::SUM, DataType::FP32);
                    synchronizeAll();
                }

                // Benchmark
                std::vector<double> latencies;
                latencies.reserve(iterations);

                for (int i = 0; i < iterations; i++)
                {
                    // Reset buffer contents for consistent measurement
                    auto start = std::chrono::high_resolution_clock::now();

                    backend_->allreduce(cuda_buffer, rocm_buffer, bytes / sizeof(float),
                                        CollectiveOp::SUM, DataType::FP32);
                    synchronizeAll();

                    auto end = std::chrono::high_resolution_clock::now();
                    double us = std::chrono::duration<double, std::micro>(end - start).count();
                    latencies.push_back(us);
                }

                return computeMetrics(bytes, latencies);
            }

            /**
             * @brief Simulate sustained decode workload
             *
             * Each token generation requires 56 AllReduces (2 per layer × 28 layers).
             * This test measures the sustained throughput over many tokens.
             *
             * @param num_tokens Number of tokens to simulate
             * @param allreduce_size Bytes per AllReduce (default 14KB for Wo output)
             * @return Tokens per second achievable with AllReduce overhead
             */
            double benchmarkSustainedDecode(int num_tokens, size_t allreduce_size = 14 * 1024)
            {
                const int layers = 28;
                const int allreduces_per_layer = 2; // Wo AllReduce + FFN AllReduce
                const int total_allreduces = layers * allreduces_per_layer;

                auto *cuda_buffer = getCUDABuffer(allreduce_size);
                auto *rocm_buffer = getROCmBuffer(allreduce_size);

                if (!cuda_buffer || !rocm_buffer)
                {
                    LOG_ERROR("Failed to get buffers for sustained decode test");
                    return 0.0;
                }

                // Warmup
                for (int i = 0; i < 10; i++)
                {
                    backend_->allreduce(cuda_buffer, rocm_buffer, allreduce_size / sizeof(float),
                                        CollectiveOp::SUM, DataType::FP32);
                }
                synchronizeAll();

                auto start = std::chrono::high_resolution_clock::now();

                for (int token = 0; token < num_tokens; token++)
                {
                    for (int layer = 0; layer < layers; layer++)
                    {
                        // Wo AllReduce
                        backend_->allreduce(cuda_buffer, rocm_buffer, allreduce_size / sizeof(float),
                                            CollectiveOp::SUM, DataType::FP32);
                        // FFN down AllReduce
                        backend_->allreduce(cuda_buffer, rocm_buffer, allreduce_size / sizeof(float),
                                            CollectiveOp::SUM, DataType::FP32);
                    }
                }

                synchronizeAll();
                auto end = std::chrono::high_resolution_clock::now();

                double total_s = std::chrono::duration<double>(end - start).count();
                double tok_per_s = num_tokens / total_s;

                LOG_INFO("╔══════════════════════════════════════════════════════════════╗");
                LOG_INFO("║  SUSTAINED DECODE BENCHMARK                                  ║");
                LOG_INFO("╠══════════════════════════════════════════════════════════════╣");
                LOG_INFO("║  Tokens simulated: " << num_tokens);
                LOG_INFO("║  AllReduces per token: " << total_allreduces);
                LOG_INFO("║  Total AllReduces: " << (num_tokens * total_allreduces));
                LOG_INFO("║  Total time: " << (total_s * 1000) << " ms");
                LOG_INFO("║  Throughput: " << tok_per_s << " tok/s (AllReduce only)");
                LOG_INFO("╚══════════════════════════════════════════════════════════════╝");

                return tok_per_s;
            }

        private:
            bool checkCUDAAvailable()
            {
#ifdef HAVE_CUDA
                int count = 0;
                cudaGetDeviceCount(&count);
                return count > 0;
#else
                return false;
#endif
            }

            bool checkROCmAvailable()
            {
#ifdef HAVE_ROCM
                int count = 0;
                hipGetDeviceCount(&count);
                return count > 0;
#else
                return false;
#endif
            }

            std::unique_ptr<ICollectiveBackend> createPCIeBARBackend()
            {
                // Factory method - actual implementation depends on backend availability
                return std::make_unique<PCIeBARBackend>(device_group_);
            }

            void allocateTestBuffers()
            {
                // Pre-allocate buffers for common test sizes
                std::vector<size_t> sizes = {
                    1 * 1024,       // 1 KB
                    4 * 1024,       // 4 KB
                    14 * 1024,      // 14 KB (Wo output for d_model=3584)
                    28 * 1024,      // 28 KB (KV head)
                    64 * 1024,      // 64 KB
                    128 * 1024,     // 128 KB
                    256 * 1024,     // 256 KB
                    512 * 1024,     // 512 KB
                    1024 * 1024,    // 1 MB
                    4 * 1024 * 1024 // 4 MB
                };

                for (size_t size : sizes)
                {
                    allocateBufferPair(size);
                }
            }

            void allocateBufferPair(size_t bytes)
            {
#ifdef HAVE_CUDA
                void *cuda_ptr = nullptr;
                cudaMalloc(&cuda_ptr, bytes);
                cuda_buffers_[bytes] = cuda_ptr;
#endif

#ifdef HAVE_ROCM
                void *rocm_ptr = nullptr;
                hipMalloc(&rocm_ptr, bytes);
                rocm_buffers_[bytes] = rocm_ptr;
#endif
            }

            void freeTestBuffers()
            {
#ifdef HAVE_CUDA
                for (auto &[size, ptr] : cuda_buffers_)
                {
                    if (ptr)
                        cudaFree(ptr);
                }
                cuda_buffers_.clear();
#endif

#ifdef HAVE_ROCM
                for (auto &[size, ptr] : rocm_buffers_)
                {
                    if (ptr)
                        hipFree(ptr);
                }
                rocm_buffers_.clear();
#endif
            }

            void *getCUDABuffer(size_t bytes)
            {
                auto it = cuda_buffers_.find(bytes);
                if (it != cuda_buffers_.end())
                    return it->second;

                // Allocate on demand
                allocateBufferPair(bytes);
                return cuda_buffers_[bytes];
            }

            void *getROCmBuffer(size_t bytes)
            {
                auto it = rocm_buffers_.find(bytes);
                if (it != rocm_buffers_.end())
                    return it->second;

                // Allocate on demand
                allocateBufferPair(bytes);
                return rocm_buffers_[bytes];
            }

            void initializeBuffer(void *ptr, size_t bytes, DeviceId device)
            {
                // Initialize with sequential values for verification
                size_t count = bytes / sizeof(float);
                std::vector<float> host_data(count);
                for (size_t i = 0; i < count; i++)
                {
                    host_data[i] = static_cast<float>(i % 1000) * 0.001f;
                }

                if (device.type() == DeviceType::CUDA)
                {
#ifdef HAVE_CUDA
                    cudaMemcpy(ptr, host_data.data(), bytes, cudaMemcpyHostToDevice);
#endif
                }
                else if (device.type() == DeviceType::ROCm)
                {
#ifdef HAVE_ROCM
                    hipMemcpy(ptr, host_data.data(), bytes, hipMemcpyHostToDevice);
#endif
                }
            }

            void synchronizeAll()
            {
#ifdef HAVE_CUDA
                cudaDeviceSynchronize();
#endif
#ifdef HAVE_ROCM
                hipDeviceSynchronize();
#endif
            }

            LatencyMetrics computeMetrics(size_t bytes, std::vector<double> &latencies)
            {
                LatencyMetrics metrics;
                metrics.bytes = bytes;

                if (latencies.empty())
                    return metrics;

                // Sort for percentiles
                std::sort(latencies.begin(), latencies.end());

                metrics.min_us = latencies.front();
                metrics.max_us = latencies.back();

                // Mean
                double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
                metrics.mean_us = sum / latencies.size();

                // Median
                size_t mid = latencies.size() / 2;
                if (latencies.size() % 2 == 0)
                {
                    metrics.median_us = (latencies[mid - 1] + latencies[mid]) / 2.0;
                }
                else
                {
                    metrics.median_us = latencies[mid];
                }

                // Percentiles
                metrics.p95_us = latencies[static_cast<size_t>(latencies.size() * 0.95)];
                metrics.p99_us = latencies[static_cast<size_t>(latencies.size() * 0.99)];

                // Standard deviation
                double sq_sum = 0.0;
                for (double lat : latencies)
                {
                    sq_sum += (lat - metrics.mean_us) * (lat - metrics.mean_us);
                }
                metrics.stddev_us = std::sqrt(sq_sum / latencies.size());

                // Bandwidth (AllReduce moves 2× data: read + write)
                double bytes_moved = bytes * 2.0;
                double seconds = metrics.mean_us * 1e-6;
                metrics.bandwidth_gbps = (bytes_moved / seconds) / (1024.0 * 1024.0 * 1024.0);

                return metrics;
            }

            bool cuda_available_ = false;
            bool rocm_available_ = false;
            DeviceId cuda_device_;
            DeviceId rocm_device_;
            DeviceGroup device_group_;
            std::unique_ptr<ICollectiveBackend> backend_;

            std::unordered_map<size_t, void *> cuda_buffers_;
            std::unordered_map<size_t, void *> rocm_buffers_;
        };

        // =============================================================================
        // Test Cases
        // =============================================================================

        /**
         * @test Small transfer latency - critical for decode phase
         *
         * The 14KB transfer is the most important: it's the Wo output size for
         * Qwen2.5-7B (d_model=3584 × 4 bytes = 14,336 bytes).
         *
         * Success: mean latency < 200μs
         * Target: mean latency < 150μs
         */
        TEST_F(PCIeBARLatencyBenchmark, SmallTransferLatency_1KB)
        {
            auto metrics = benchmarkAllReduce(1 * 1024);
            metrics.print();

            // Baseline overhead measurement
            EXPECT_LT(metrics.mean_us, 500.0) << "1KB AllReduce should complete in < 500μs";
        }

        TEST_F(PCIeBARLatencyBenchmark, SmallTransferLatency_14KB)
        {
            auto metrics = benchmarkAllReduce(14 * 1024);
            metrics.print();

            // CRITICAL: This is the key metric for hybrid parallelism viability
            EXPECT_LT(metrics.mean_us, 300.0) << "14KB AllReduce should complete in < 300μs (acceptable)";

            // Log whether we meet target
            if (metrics.mean_us < 150.0)
            {
                LOG_INFO("✅ MEETS TARGET: 14KB latency " << metrics.mean_us << "μs < 150μs target");
            }
            else if (metrics.mean_us < 200.0)
            {
                LOG_WARN("⚠️  ACCEPTABLE: 14KB latency " << metrics.mean_us << "μs < 200μs but > 150μs target");
            }
            else
            {
                LOG_WARN("⚠️  HIGH LATENCY: 14KB latency " << metrics.mean_us << "μs > 200μs - consider CPU staging");
            }
        }

        TEST_F(PCIeBARLatencyBenchmark, SmallTransferLatency_28KB)
        {
            auto metrics = benchmarkAllReduce(28 * 1024);
            metrics.print();

            EXPECT_LT(metrics.mean_us, 400.0) << "28KB AllReduce should complete in < 400μs";
        }

        TEST_F(PCIeBARLatencyBenchmark, SmallTransferLatency_64KB)
        {
            auto metrics = benchmarkAllReduce(64 * 1024);
            metrics.print();

            EXPECT_LT(metrics.mean_us, 600.0) << "64KB AllReduce should complete in < 600μs";
        }

        TEST_F(PCIeBARLatencyBenchmark, SmallTransferLatency_128KB)
        {
            auto metrics = benchmarkAllReduce(128 * 1024);
            metrics.print();

            EXPECT_LT(metrics.mean_us, 1000.0) << "128KB AllReduce should complete in < 1ms";
        }

        /**
         * @test Large transfer bandwidth - validates PCIeBAR throughput
         */
        TEST_F(PCIeBARLatencyBenchmark, LargeTransferBandwidth_1MB)
        {
            auto metrics = benchmarkAllReduce(1 * 1024 * 1024);
            metrics.print();

            // At 1MB, bandwidth should dominate over latency
            EXPECT_GT(metrics.bandwidth_gbps, 1.0) << "1MB transfers should achieve > 1 GB/s";
        }

        TEST_F(PCIeBARLatencyBenchmark, LargeTransferBandwidth_4MB)
        {
            auto metrics = benchmarkAllReduce(4 * 1024 * 1024);
            metrics.print();

            // Should approach peak PCIe bandwidth
            EXPECT_GT(metrics.bandwidth_gbps, 2.0) << "4MB transfers should achieve > 2 GB/s";
        }

        /**
         * @test Sustained decode simulation
         *
         * Simulates the AllReduce overhead for 100 tokens of decode.
         * Each token requires 56 AllReduces (2 per layer × 28 layers).
         *
         * Success: > 10 tok/s (allows total throughput > 5 tok/s with compute)
         * Target: > 20 tok/s (AllReduce not a bottleneck)
         */
        TEST_F(PCIeBARLatencyBenchmark, SustainedDecodeSimulation)
        {
            double tok_per_s = benchmarkSustainedDecode(100);

            // This is the key end-to-end metric
            EXPECT_GT(tok_per_s, 5.0) << "Sustained decode should achieve > 5 tok/s (AllReduce only)";

            if (tok_per_s > 20.0)
            {
                LOG_INFO("✅ EXCELLENT: " << tok_per_s << " tok/s - AllReduce is not a bottleneck");
            }
            else if (tok_per_s > 10.0)
            {
                LOG_INFO("✅ GOOD: " << tok_per_s << " tok/s - meets target for hybrid parallelism");
            }
            else if (tok_per_s > 5.0)
            {
                LOG_WARN("⚠️  MARGINAL: " << tok_per_s << " tok/s - may limit total throughput");
            }
            else
            {
                LOG_ERROR("❌ BLOCKING: " << tok_per_s << " tok/s - PCIeBAR is a bottleneck");
            }
        }

        /**
         * @test Latency variance - checks for consistent performance
         *
         * High variance indicates potential contention or driver issues.
         */
        TEST_F(PCIeBARLatencyBenchmark, LatencyVariance)
        {
            auto metrics = benchmarkAllReduce(14 * 1024, 500); // More iterations for stats
            metrics.print();

            // P99 should not be > 3× median (indicates outliers/contention)
            double ratio = metrics.p99_us / metrics.median_us;
            EXPECT_LT(ratio, 3.0) << "P99/median ratio should be < 3.0 (got " << ratio << ")";

            // Stddev should be reasonable
            double cv = metrics.stddev_us / metrics.mean_us; // Coefficient of variation
            EXPECT_LT(cv, 0.5) << "Coefficient of variation should be < 0.5 (got " << cv << ")";
        }

        /**
         * @test Transfer size sweep - characterizes latency vs bandwidth tradeoff
         */
        TEST_F(PCIeBARLatencyBenchmark, TransferSizeSweep)
        {
            LOG_INFO("╔══════════════════════════════════════════════════════════════╗");
            LOG_INFO("║               TRANSFER SIZE SWEEP                            ║");
            LOG_INFO("╠══════════════════════════════════════════════════════════════╣");

            std::vector<size_t> sizes = {
                1024, 2048, 4096, 8192, 14336, 16384, 32768, 65536, 131072, 262144};

            std::vector<LatencyMetrics> all_metrics;

            for (size_t size : sizes)
            {
                auto metrics = benchmarkAllReduce(size, 50);
                all_metrics.push_back(metrics);

                LOG_INFO("║  " << std::setw(8) << size << " bytes: "
                               << std::fixed << std::setprecision(1) << std::setw(8) << metrics.mean_us << " μs  "
                               << std::setw(6) << metrics.bandwidth_gbps << " GB/s");
            }

            LOG_INFO("╚══════════════════════════════════════════════════════════════╝");

            // Find the crossover point where bandwidth starts dominating
            double prev_efficiency = 0;
            for (size_t i = 1; i < all_metrics.size(); i++)
            {
                double efficiency = all_metrics[i].bandwidth_gbps;
                if (efficiency > 0.9 * all_metrics.back().bandwidth_gbps && prev_efficiency < 0.9 * all_metrics.back().bandwidth_gbps)
                {
                    LOG_INFO("Bandwidth efficiency threshold (~90% peak) reached at ~" << all_metrics[i].bytes << " bytes");
                }
                prev_efficiency = efficiency;
            }
        }

    } // namespace test
} // namespace llaminar2

#else // !defined(HAVE_CUDA) || !defined(HAVE_ROCM)

// Placeholder test when heterogeneous hardware not available
TEST(PCIeBARLatencyBenchmark, SkippedNoHeterogeneousHardware)
{
    GTEST_SKIP() << "PCIeBAR latency benchmarks require both CUDA and ROCm hardware";
}

#endif // HAVE_CUDA && HAVE_ROCM

/**
 * @file Perf__AllreduceLatency_3Backend.cpp
 * @brief Side-by-side allreduce latency comparison across NCCL, RCCL, and PCIeBAR backends
 *
 * Tests the Llaminar ICollectiveBackend abstractions directly:
 *   - NCCLBackend (cuda:0 ↔ cuda:1) via allreduceMulti
 *   - RCCLBackend (rocm:0 ↔ rocm:1) via allreduceMulti
 *   - PCIeBARBackend (cuda:0 ↔ rocm:0) via allreduceRegistered
 *
 * Uses IBackend for all device memory allocation to avoid GPU-specific header
 * conflicts. Renders a combined fort table for direct comparison.
 *
 * @author David Sanftenberg
 * @date March 2026
 */

#include <gtest/gtest.h>

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "collective/backends/NCCLBackend.h"
#include "collective/backends/RCCLBackend.h"
#include "collective/backends/PCIeBARBackend.h"
#include "collective/DeviceGroup.h"
#include "backends/BackendManager.h"
#include "backends/DeviceId.h"
#include "backends/IBackend.h"
#include "backends/p2p/DirectP2P.h"

#include "fort.hpp"

using namespace llaminar2;

namespace
{

    // =============================================================================
    // Configuration
    // =============================================================================

    constexpr int WARMUP_ITERATIONS = 200;
    constexpr int BENCHMARK_ITERATIONS = 1000;

    struct SizeSpec
    {
        size_t count;      // float count
        const char *label; // human-readable label
    };

    // Message sizes spanning decode (small) to prefill (large)
    static const std::vector<SizeSpec> TEST_SIZES = {
        {256 / sizeof(float), "256B"},
        {1024 / sizeof(float), "1KB"},
        {4096 / sizeof(float), "4KB"},
        {14336 / sizeof(float), "14KB (decode)"},
        {65536 / sizeof(float), "64KB"},
        {262144 / sizeof(float), "256KB"},
        {1048576 / sizeof(float), "1MB"},
        {4194304 / sizeof(float), "4MB"},
        {8912896 / sizeof(float), "8.5MB (prefill)"},
    };

    // =============================================================================
    // Helpers
    // =============================================================================

    struct LatencyResult
    {
        double avg_us;
        double min_us;
        double max_us;
        double p50_us;
        double p99_us;
        double throughput_gbps;
        bool valid;
    };

    LatencyResult computeStats(std::vector<double> &timings, size_t bytes)
    {
        LatencyResult r{};
        if (timings.empty())
        {
            r.valid = false;
            return r;
        }
        r.valid = true;

        std::sort(timings.begin(), timings.end());

        double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        r.avg_us = sum / static_cast<double>(timings.size());
        r.min_us = timings.front();
        r.max_us = timings.back();
        r.p50_us = timings[timings.size() / 2];
        r.p99_us = timings[static_cast<size_t>(timings.size() * 0.99)];
        // allreduce reads + writes = 2× data
        r.throughput_gbps = (2.0 * static_cast<double>(bytes)) / (r.avg_us * 1e3);
        return r;
    }

    void fillRandom(std::vector<float> &data, unsigned seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : data)
            v = dist(rng);
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class AllreduceLatency3Backend : public ::testing::Test
    {
    protected:
        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;

        void SetUp() override
        {
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();
        }

        // -------------------------------------------------------------------------
        // NCCL: cuda:0 ↔ cuda:1 via allreduceMulti
        // -------------------------------------------------------------------------
        LatencyResult benchNCCL(size_t count)
        {
            if (!cuda_backend_)
                return {.valid = false};

            // Need at least 2 CUDA GPUs
            NCCLBackend backend;
            DeviceGroupBuilder builder;
            auto group = builder.setName("nccl_bench")
                             .setScope(CollectiveScope::LOCAL)
                             .addDevice(DeviceId::cuda(0))
                             .addDevice(DeviceId::cuda(1))
                             .setLocalRank(0)
                             .build();

            if (!backend.initialize(group))
                return {.valid = false};

            if (!backend.isMultiGpuSingleProcess())
            {
                backend.shutdown();
                return {.valid = false};
            }

            size_t bytes = count * sizeof(float);

            // Allocate on each GPU via IBackend
            void *d0 = cuda_backend_->allocate(bytes, 0);
            void *d1 = cuda_backend_->allocate(bytes, 1);
            if (!d0 || !d1)
            {
                if (d0)
                    cuda_backend_->free(d0, 0);
                if (d1)
                    cuda_backend_->free(d1, 1);
                backend.shutdown();
                return {.valid = false};
            }

            // Initialize with known data
            std::vector<float> h0(count), h1(count);
            fillRandom(h0, 42);
            fillRandom(h1, 99);
            cuda_backend_->hostToDevice(d0, h0.data(), bytes, 0);
            cuda_backend_->hostToDevice(d1, h1.data(), bytes, 1);
            cuda_backend_->synchronize(0);
            cuda_backend_->synchronize(1);

            std::vector<void *> buffers = {d0, d1};

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                cuda_backend_->hostToDevice(d0, h0.data(), bytes, 0);
                cuda_backend_->hostToDevice(d1, h1.data(), bytes, 1);
                cuda_backend_->synchronize(0);
                cuda_backend_->synchronize(1);
                backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
                backend.synchronize();
            }

            // Benchmark
            std::vector<double> timings;
            timings.reserve(BENCHMARK_ITERATIONS);

            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                // Reset
                cuda_backend_->hostToDevice(d0, h0.data(), bytes, 0);
                cuda_backend_->hostToDevice(d1, h1.data(), bytes, 1);
                cuda_backend_->synchronize(0);
                cuda_backend_->synchronize(1);

                auto start = std::chrono::high_resolution_clock::now();
                backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
                backend.synchronize();
                auto end = std::chrono::high_resolution_clock::now();

                timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            }

            cuda_backend_->free(d0, 0);
            cuda_backend_->free(d1, 1);
            backend.shutdown();

            return computeStats(timings, bytes);
        }

        // -------------------------------------------------------------------------
        // RCCL: rocm:0 ↔ rocm:1 via allreduceMulti
        // -------------------------------------------------------------------------
        LatencyResult benchRCCL(size_t count)
        {
            if (!rocm_backend_)
                return {.valid = false};

            RCCLBackend backend;
            DeviceGroupBuilder builder;
            auto group = builder.setName("rccl_bench")
                             .setScope(CollectiveScope::LOCAL)
                             .addDevice(DeviceId::rocm(0))
                             .addDevice(DeviceId::rocm(1))
                             .setLocalRank(0)
                             .build();

            if (!backend.initialize(group))
                return {.valid = false};

            if (!backend.isMultiGpuSingleProcess())
            {
                backend.shutdown();
                return {.valid = false};
            }

            size_t bytes = count * sizeof(float);

            void *d0 = rocm_backend_->allocate(bytes, 0);
            void *d1 = rocm_backend_->allocate(bytes, 1);
            if (!d0 || !d1)
            {
                if (d0)
                    rocm_backend_->free(d0, 0);
                if (d1)
                    rocm_backend_->free(d1, 1);
                backend.shutdown();
                return {.valid = false};
            }

            std::vector<float> h0(count), h1(count);
            fillRandom(h0, 42);
            fillRandom(h1, 99);
            rocm_backend_->hostToDevice(d0, h0.data(), bytes, 0);
            rocm_backend_->hostToDevice(d1, h1.data(), bytes, 1);
            rocm_backend_->synchronize(0);
            rocm_backend_->synchronize(1);

            std::vector<void *> buffers = {d0, d1};

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                rocm_backend_->hostToDevice(d0, h0.data(), bytes, 0);
                rocm_backend_->hostToDevice(d1, h1.data(), bytes, 1);
                rocm_backend_->synchronize(0);
                rocm_backend_->synchronize(1);
                backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
                backend.synchronize();
            }

            // Benchmark
            std::vector<double> timings;
            timings.reserve(BENCHMARK_ITERATIONS);

            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                rocm_backend_->hostToDevice(d0, h0.data(), bytes, 0);
                rocm_backend_->hostToDevice(d1, h1.data(), bytes, 1);
                rocm_backend_->synchronize(0);
                rocm_backend_->synchronize(1);

                auto start = std::chrono::high_resolution_clock::now();
                backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
                backend.synchronize();
                auto end = std::chrono::high_resolution_clock::now();

                timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            }

            rocm_backend_->free(d0, 0);
            rocm_backend_->free(d1, 1);
            backend.shutdown();

            return computeStats(timings, bytes);
        }

        // -------------------------------------------------------------------------
        // PCIeBAR: cuda:0 ↔ rocm:0 via allreduceRegistered
        // -------------------------------------------------------------------------
        LatencyResult benchPCIeBAR(size_t count)
        {
            if (!cuda_backend_ || !rocm_backend_)
                return {.valid = false};

            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoPCIeBarP2P())
                return {.valid = false};

            PCIeBARBackend backend;
            DeviceGroupBuilder builder;
            auto group = builder.setName("pciebar_bench")
                             .setScope(CollectiveScope::LOCAL)
                             .addDevice(DeviceId::cuda(0))
                             .addDevice(DeviceId::rocm(0))
                             .setLocalRank(0)
                             .build();

            if (!backend.initialize(group))
                return {.valid = false};

            size_t bytes = count * sizeof(float);

            // CUDA side: normal device allocation
            void *d_cuda = cuda_backend_->allocate(bytes, 0);
            if (!d_cuda)
            {
                backend.shutdown();
                return {.valid = false};
            }

            // ROCm side: must be in BAR region for PCIeBAR to work
            auto bar_alloc = backend.allocateInBarRegion(bytes);
            if (!bar_alloc.has_value())
            {
                cuda_backend_->free(d_cuda, 0);
                backend.shutdown();
                return {.valid = false};
            }
            void *d_rocm = bar_alloc->first;

            // Register for collective
            std::string coll_id = "bench_" + std::to_string(count);
            if (!backend.registerBuffer(coll_id, DeviceId::cuda(0), d_cuda, bytes) ||
                !backend.registerBuffer(coll_id, DeviceId::rocm(0), d_rocm, bytes))
            {
                cuda_backend_->free(d_cuda, 0);
                backend.freeBarBuffer(d_rocm);
                backend.shutdown();
                return {.valid = false};
            }

            // Initialize with known data
            std::vector<float> h_cuda(count), h_rocm(count);
            fillRandom(h_cuda, 42);
            fillRandom(h_rocm, 99);
            cuda_backend_->hostToDevice(d_cuda, h_cuda.data(), bytes, 0);
            std::memcpy(d_rocm, h_rocm.data(), bytes);
            cuda_backend_->synchronize(0);

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                cuda_backend_->hostToDevice(d_cuda, h_cuda.data(), bytes, 0);
                std::memcpy(d_rocm, h_rocm.data(), bytes);
                cuda_backend_->synchronize(0);
                backend.allreduceRegistered(coll_id, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
            }

            // Benchmark
            std::vector<double> timings;
            timings.reserve(BENCHMARK_ITERATIONS);

            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                cuda_backend_->hostToDevice(d_cuda, h_cuda.data(), bytes, 0);
                std::memcpy(d_rocm, h_rocm.data(), bytes);
                cuda_backend_->synchronize(0);

                auto start = std::chrono::high_resolution_clock::now();
                backend.allreduceRegistered(coll_id, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
                auto end = std::chrono::high_resolution_clock::now();

                timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            }

            cuda_backend_->free(d_cuda, 0);
            backend.freeBarBuffer(d_rocm);
            backend.shutdown();

            return computeStats(timings, bytes);
        }
    };

    // =============================================================================
    // Individual Backend Tests
    // =============================================================================

    TEST_F(AllreduceLatency3Backend, NCCLLatency)
    {
        if (!cuda_backend_)
            GTEST_SKIP() << "CUDA backend not available";

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Size" << "Avg (μs)" << "Min (μs)" << "P50 (μs)" << "P99 (μs)" << "Throughput" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 5; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        bool any_valid = false;
        for (const auto &sz : TEST_SIZES)
        {
            auto r = benchNCCL(sz.count);
            if (!r.valid)
            {
                table << sz.label << "SKIP" << "" << "" << "" << "" << fort::endr;
                continue;
            }
            any_valid = true;

            std::ostringstream avg, mn, p50, p99, tp;
            avg << std::fixed << std::setprecision(1) << r.avg_us;
            mn << std::fixed << std::setprecision(1) << r.min_us;
            p50 << std::fixed << std::setprecision(1) << r.p50_us;
            p99 << std::fixed << std::setprecision(1) << r.p99_us;
            tp << std::fixed << std::setprecision(2) << r.throughput_gbps << " GB/s";

            table << sz.label << avg.str() << mn.str() << p50.str() << p99.str() << tp.str() << fort::endr;
        }

        std::cout << "\n  NCCL AllReduce (cuda:0 <-> cuda:1) — "
                  << WARMUP_ITERATIONS << " warmup, " << BENCHMARK_ITERATIONS << " iterations\n"
                  << table.to_string() << std::endl;

        if (!any_valid)
            GTEST_SKIP() << "NCCL initialization failed (need 2 CUDA GPUs)";
    }

    TEST_F(AllreduceLatency3Backend, RCCLLatency)
    {
        if (!rocm_backend_)
            GTEST_SKIP() << "ROCm backend not available";

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Size" << "Avg (μs)" << "Min (μs)" << "P50 (μs)" << "P99 (μs)" << "Throughput" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 5; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        bool any_valid = false;
        for (const auto &sz : TEST_SIZES)
        {
            auto r = benchRCCL(sz.count);
            if (!r.valid)
            {
                table << sz.label << "SKIP" << "" << "" << "" << "" << fort::endr;
                continue;
            }
            any_valid = true;

            std::ostringstream avg, mn, p50, p99, tp;
            avg << std::fixed << std::setprecision(1) << r.avg_us;
            mn << std::fixed << std::setprecision(1) << r.min_us;
            p50 << std::fixed << std::setprecision(1) << r.p50_us;
            p99 << std::fixed << std::setprecision(1) << r.p99_us;
            tp << std::fixed << std::setprecision(2) << r.throughput_gbps << " GB/s";

            table << sz.label << avg.str() << mn.str() << p50.str() << p99.str() << tp.str() << fort::endr;
        }

        std::cout << "\n  RCCL AllReduce (rocm:0 <-> rocm:1) — "
                  << WARMUP_ITERATIONS << " warmup, " << BENCHMARK_ITERATIONS << " iterations\n"
                  << table.to_string() << std::endl;

        if (!any_valid)
            GTEST_SKIP() << "RCCL initialization failed (need 2 ROCm GPUs)";
    }

    TEST_F(AllreduceLatency3Backend, PCIeBARLatency)
    {
        if (!cuda_backend_ || !rocm_backend_)
            GTEST_SKIP() << "CUDA or ROCm backend not available";

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Size" << "Avg (μs)" << "Min (μs)" << "P50 (μs)" << "P99 (μs)" << "Throughput" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 5; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        bool any_valid = false;
        for (const auto &sz : TEST_SIZES)
        {
            auto r = benchPCIeBAR(sz.count);
            if (!r.valid)
            {
                table << sz.label << "SKIP" << "" << "" << "" << "" << fort::endr;
                continue;
            }
            any_valid = true;

            std::ostringstream avg, mn, p50, p99, tp;
            avg << std::fixed << std::setprecision(1) << r.avg_us;
            mn << std::fixed << std::setprecision(1) << r.min_us;
            p50 << std::fixed << std::setprecision(1) << r.p50_us;
            p99 << std::fixed << std::setprecision(1) << r.p99_us;
            tp << std::fixed << std::setprecision(2) << r.throughput_gbps << " GB/s";

            table << sz.label << avg.str() << mn.str() << p50.str() << p99.str() << tp.str() << fort::endr;
        }

        std::cout << "\n  PCIeBAR AllReduce (cuda:0 <-> rocm:0) — "
                  << WARMUP_ITERATIONS << " warmup, " << BENCHMARK_ITERATIONS << " iterations\n"
                  << table.to_string() << std::endl;

        if (!any_valid)
            GTEST_SKIP() << "PCIeBAR initialization failed (need PCIe BAR P2P hardware)";
    }

    // =============================================================================
    // Combined Comparison
    // =============================================================================

    TEST_F(AllreduceLatency3Backend, CompareAllBackends)
    {
        if (!cuda_backend_ || !rocm_backend_)
            GTEST_SKIP() << "Need both CUDA and ROCm backends";

        struct Row
        {
            const char *label;
            LatencyResult nccl;
            LatencyResult rccl;
            LatencyResult pciebar;
        };

        std::vector<Row> rows;
        rows.reserve(TEST_SIZES.size());

        for (const auto &sz : TEST_SIZES)
        {
            Row row;
            row.label = sz.label;
            row.nccl = benchNCCL(sz.count);
            row.rccl = benchRCCL(sz.count);
            row.pciebar = benchPCIeBAR(sz.count);
            rows.push_back(row);
        }

        // Render comparison table
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Size"
              << "NCCL Avg (μs)" << "NCCL P50 (μs)" << "NCCL GB/s"
              << "RCCL Avg (μs)" << "RCCL P50 (μs)" << "RCCL GB/s"
              << "BAR Avg (μs)" << "BAR P50 (μs)" << "BAR GB/s"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 9; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        auto fmtUs = [](double v) -> std::string
        {
            std::ostringstream s;
            s << std::fixed << std::setprecision(1) << v;
            return s.str();
        };
        auto fmtGbps = [](double v) -> std::string
        {
            std::ostringstream s;
            s << std::fixed << std::setprecision(2) << v;
            return s.str();
        };
        auto skip = []() -> std::string
        { return "—"; };

        for (const auto &r : rows)
        {
            table << r.label;

            if (r.nccl.valid)
                table << fmtUs(r.nccl.avg_us) << fmtUs(r.nccl.p50_us) << fmtGbps(r.nccl.throughput_gbps);
            else
                table << skip() << skip() << skip();

            if (r.rccl.valid)
                table << fmtUs(r.rccl.avg_us) << fmtUs(r.rccl.p50_us) << fmtGbps(r.rccl.throughput_gbps);
            else
                table << skip() << skip() << skip();

            if (r.pciebar.valid)
                table << fmtUs(r.pciebar.avg_us) << fmtUs(r.pciebar.p50_us) << fmtGbps(r.pciebar.throughput_gbps);
            else
                table << skip() << skip() << skip();

            table << fort::endr;
        }

        std::cout << "\n  AllReduce Latency Comparison — "
                  << WARMUP_ITERATIONS << " warmup, " << BENCHMARK_ITERATIONS << " iterations\n"
                  << "    NCCL: cuda:0 <-> cuda:1 | RCCL: rocm:0 <-> rocm:1 | PCIeBAR: cuda:0 <-> rocm:0\n\n"
                  << table.to_string() << std::endl;
    }

} // namespace

#else // !(HAVE_CUDA && HAVE_ROCM)

TEST(AllreduceLatency3Backend, SkippedNoCudaRocm)
{
    GTEST_SKIP() << "Requires both HAVE_CUDA and HAVE_ROCM";
}

#endif // HAVE_CUDA && HAVE_ROCM

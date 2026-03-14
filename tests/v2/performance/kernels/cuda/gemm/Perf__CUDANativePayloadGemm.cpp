/**
 * @file Perf__CUDANativePayloadGemm.cpp
 * @brief Correctness and performance harness for tensor-core native-payload prefill GEMM.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <thread>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "CUDANativePayloadGemmPerfCommon.h"

using namespace llaminar2::test::native_payload_gemm_perf;

namespace
{
    class CUDANativePayloadGemmPerf : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
                GTEST_SKIP() << "No CUDA devices available";

            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            device_count_ = device_count;
            peak_tc_tops_ = estimatePeakInt8TensorCoreTops();
            if (peak_tc_tops_ > 0.0)
                std::fprintf(stderr, "[CUDANativePayloadGemm] Peak dense INT8 tensor-core throughput: %.1f TOPS\n", peak_tc_tops_);
        }

        double peak_tc_tops_ = 0.0;
        int device_count_ = 0;
    };

    TEST_F(CUDANativePayloadGemmPerf, Correctness_AllFormats_KeyShapes)
    {
        const RunConfig cfg = loadRunConfig();
        int executed_cases = 0;
        bool stop = false;

        for (const auto &format : kFormats)
        {
            if (!shouldRunName(cfg.format_filters, format.name))
                continue;

            for (const auto &shape : kQwenShapes)
            {
                if (!shouldRunName(cfg.shape_filters, shape.name))
                    continue;
                if ((shape.k % 32) != 0)
                    continue;
                if (executed_cases >= cfg.max_cases)
                {
                    stop = true;
                    break;
                }

                std::fprintf(stderr,
                             "[CUDANativePayloadGemm][Correctness] format=%s shape=%s prefill_m=%d\n",
                             format.name.c_str(), shape.name.c_str(), cfg.correctness_prefill_m);

                auto cutlass_weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult cutlass = runKernel(cutlass_weights.get(), cfg.correctness_prefill_m, shape.n, shape.k, RunPath::CutlassFallback, 0, 1, 0);

                auto native_payload_weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult native_payload = runKernel(native_payload_weights.get(), cfg.correctness_prefill_m, shape.n, shape.k, RunPath::NativePayloadTensorCore, 0, 1, 0);

                EXPECT_GE(cosineSimilarity(cutlass.output, native_payload.output), kCosineGate)
                    << format.name << " prefill " << shape.name;
                ++executed_cases;
            }

            if (stop)
                break;
        }

        ASSERT_GT(executed_cases, 0) << "No correctness cases selected. Check LLAMINAR_CUDA_NATIVE_GEMM_FORMATS / LLAMINAR_CUDA_NATIVE_GEMM_SHAPES.";
    }

    TEST_F(CUDANativePayloadGemmPerf, Performance_AllFormats_AllShapes)
    {
        const RunConfig cfg = loadRunConfig();

        struct PerfTask
        {
            const FormatSpec *format = nullptr;
            const Shape *shape = nullptr;
        };

        struct PerfRow
        {
            std::string format_name;
            uint8_t codebook_id = 0;
            std::string shape_name;
            int m = 0;
            int n = 0;
            int k = 0;
            int warmup_runs = 0;
            int bench_runs = 0;
            size_t weight_bytes = 0;
            double cutlass_min_us = 0.0;
            double cutlass_mean_us = 0.0;
            double cutlass_tops = 0.0;
            double cutlass_pct_tc_peak = 0.0;
            std::string native_family;
            double native_min_us = 0.0;
            double native_mean_us = 0.0;
            double native_tops = 0.0;
            double native_pct_tc_peak = 0.0;
            double speedup_vs_cutlass = 0.0;
        };

        std::vector<PerfTask> tasks;
        tasks.reserve(static_cast<size_t>(cfg.max_cases));
        for (const auto &format : kFormats)
        {
            if (!shouldRunName(cfg.format_filters, format.name))
                continue;

            for (const auto &shape : kQwenShapes)
            {
                if (!shouldRunName(cfg.shape_filters, shape.name))
                    continue;
                if ((shape.k % 32) != 0)
                    continue;
                if (static_cast<int>(tasks.size()) >= cfg.max_cases)
                    break;
                tasks.push_back(PerfTask{&format, &shape});
            }

            if (static_cast<int>(tasks.size()) >= cfg.max_cases)
                break;
        }

        ASSERT_FALSE(tasks.empty()) << "No performance cases selected. Check LLAMINAR_CUDA_NATIVE_GEMM_FORMATS / LLAMINAR_CUDA_NATIVE_GEMM_SHAPES.";

        int worker_count = std::min(2, std::max(1, device_count_));
        if (cfg.performance_workers > 0)
            worker_count = std::min(worker_count, cfg.performance_workers);

        std::fprintf(stderr,
                     "[CUDANativePayloadGemm][Perf] using %d worker thread(s) across %d CUDA device(s)\n",
                     worker_count,
                     worker_count);

        std::vector<std::vector<PerfRow>> task_rows(tasks.size());
        std::atomic<size_t> next_task{0};
        std::mutex log_mutex;

        auto worker = [&](int worker_index)
        {
            const int device_id = worker_index;

            while (true)
            {
                const size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_index >= tasks.size())
                    break;

                const PerfTask &task = tasks[task_index];
                const auto &format = *task.format;
                const auto &shape = *task.shape;

                std::vector<PerfRow> rows;
                rows.reserve(cfg.performance_prefill_m.size());

                for (int m : cfg.performance_prefill_m)
                {
                    auto cutlass_weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                    const size_t weight_bytes = cutlass_weights->size_bytes();
                    const RunResult cutlass = runKernel(cutlass_weights.get(), m, shape.n, shape.k, RunPath::CutlassFallback, cfg.warmup_runs, cfg.bench_runs, device_id);

                    auto native_payload_weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                    const RunResult native_payload = runKernel(native_payload_weights.get(), m, shape.n, shape.k, RunPath::NativePayloadTensorCore, cfg.warmup_runs, cfg.bench_runs, device_id);

                    const auto cutlass_metrics = computeGemmThroughputMetrics(m, shape.n, shape.k, cutlass.min_us, peak_tc_tops_);
                    const auto native_payload_metrics = computeGemmThroughputMetrics(m, shape.n, shape.k, native_payload.min_us, peak_tc_tops_);

                    {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        std::fprintf(stderr,
                                     "[CUDANativePayloadGemm][Perf][gpu=%d] format=%s codebook=%u shape=%s M=%d N=%d K=%d warmup=%d bench=%d "
                                     "cutlass_min_us=%.3f cutlass_tops=%.3f cutlass_pct_tc_peak=%.1f%% "
                                     "native_payload_family=%s native_payload_min_us=%.3f native_payload_tops=%.3f native_payload_pct_tc_peak=%.1f%% "
                                     "speedup_vs_cutlass=%.3fx\n",
                                     device_id,
                                     format.name.c_str(),
                                     static_cast<unsigned>(format.codebook_id),
                                     shape.name.c_str(),
                                     m,
                                     shape.n,
                                     shape.k,
                                     cfg.warmup_runs,
                                     cfg.bench_runs,
                                     cutlass.min_us,
                                     cutlass_metrics.achieved_tops,
                                     cutlass_metrics.pct_tc_peak,
                                     native_payload.native_family.c_str(),
                                     native_payload.min_us,
                                     native_payload_metrics.achieved_tops,
                                     native_payload_metrics.pct_tc_peak,
                                     cutlass.min_us / native_payload.min_us);
                    }

                    rows.push_back(PerfRow{
                        format.name,
                        format.codebook_id,
                        shape.name,
                        m,
                        shape.n,
                        shape.k,
                        cfg.warmup_runs,
                        cfg.bench_runs,
                        weight_bytes,
                        cutlass.min_us,
                        cutlass.mean_us,
                        cutlass_metrics.achieved_tops,
                        cutlass_metrics.pct_tc_peak,
                        native_payload.native_family,
                        native_payload.min_us,
                        native_payload.mean_us,
                        native_payload_metrics.achieved_tops,
                        native_payload_metrics.pct_tc_peak,
                        cutlass.min_us / native_payload.min_us,
                    });
                }

                task_rows[task_index] = std::move(rows);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (int worker_index = 0; worker_index < worker_count; ++worker_index)
            workers.emplace_back(worker, worker_index);
        for (auto &thread : workers)
            thread.join();

        std::FILE *csv = nullptr;
        if (!cfg.csv_path.empty())
        {
            csv = std::fopen(cfg.csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open CSV: " << cfg.csv_path;
            std::fprintf(
                csv,
                "format,codebook,shape,m,n,k,warmup_runs,bench_runs,weight_bytes,cutlass_min_us,cutlass_mean_us,cutlass_tops,cutlass_pct_tc_peak,native_family,native_min_us,native_mean_us,native_tops,native_pct_tc_peak,speedup_vs_cutlass\n");
        }

        int executed_cases = 0;
        for (const auto &rows : task_rows)
        {
            if (rows.empty())
                continue;

            for (const auto &row : rows)
            {
                if (csv)
                {
                    std::fprintf(
                        csv,
                        "%s,%u,%s,%d,%d,%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.1f,%s,%.3f,%.3f,%.3f,%.1f,%.3f\n",
                        row.format_name.c_str(),
                        static_cast<unsigned>(row.codebook_id),
                        row.shape_name.c_str(),
                        row.m,
                        row.n,
                        row.k,
                        row.warmup_runs,
                        row.bench_runs,
                        row.weight_bytes,
                        row.cutlass_min_us,
                        row.cutlass_mean_us,
                        row.cutlass_tops,
                        row.cutlass_pct_tc_peak,
                        row.native_family.c_str(),
                        row.native_min_us,
                        row.native_mean_us,
                        row.native_tops,
                        row.native_pct_tc_peak,
                        row.speedup_vs_cutlass);
                }
                ++executed_cases;
            }
        }

        if (csv)
        {
            std::fclose(csv);
            std::fprintf(stderr,
                         "[CUDANativePayloadGemm][Perf] wrote sweep CSV to %s\n",
                         cfg.csv_path.c_str());
        }

        ASSERT_GT(executed_cases, 0) << "No performance cases selected. Check LLAMINAR_CUDA_NATIVE_GEMM_FORMATS / LLAMINAR_CUDA_NATIVE_GEMM_SHAPES.";
    }
}

#endif
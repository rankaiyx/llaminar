/**
 * @file Perf__CPUNativeVNNI_Sweep.cpp
 * @brief Comprehensive performance benchmark for CPU NativeVNNI GEMV across
 *        all 16 quantization formats, all Qwen model shapes, with bandwidth
 *        roofline analysis and CPUQuantisedGemmKernel baseline comparison.
 *
 * System roofline: 2x Intel Xeon Gold 6238R, 6-channel DDR4-2933 per socket.
 * Measured STREAM read BW: ~117 GB/s (both sockets).
 *
 * For each (format, shape):
 *   1. Pack weights via NativeVNNI (native nibble-LUT or INT8 pre-decoded)
 *   2. Pack weights via CPUQuantisedGemmKernel (INT8 requantize baseline)
 *   3. Benchmark M=1 GEMV, report time, effective BW, roofline %, speedup
 *
 * @note Run with Release build: ctest -R V2_Perf_CPUNativeVNNI_Sweep
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/gemm/CPUQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "fort.hpp"

#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // System bandwidth roofline (measured via STREAM on this machine)
    // =========================================================================

    // Measured on 2x Xeon Gold 6238R (56 physical cores, DDR4-2933, 6ch/socket)
    // STREAM read-only: ~117 GB/s, STREAM copy: ~147 GB/s
    static constexpr double MEASURED_READ_BW_GBS = 117.0;

    // =========================================================================
    // Model shape definitions
    // =========================================================================

    struct ModelConfig
    {
        std::string name;
        int d_model;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int d_ff;
    };

    static const ModelConfig kQwen05B = {"0.5B", 896, 14, 2, 64, 4864};
    static const ModelConfig kQwen15B = {"1.5B", 1536, 12, 2, 128, 8960};
    static const ModelConfig kQwen3B = {"3B", 2048, 16, 2, 128, 11008};
    static const ModelConfig kQwen7B = {"7B", 3584, 28, 4, 128, 18944};

    struct GEMVShape
    {
        std::string name;
        std::string category;
        int N;
        int K;
    };

    static std::vector<GEMVShape> buildShapes(const ModelConfig &m)
    {
        std::vector<GEMVShape> shapes;
        int n_q = m.n_heads * m.head_dim;
        int n_kv = m.n_kv_heads * m.head_dim;

        shapes.push_back({m.name + "_Q_proj", "Attn", n_q, m.d_model});
        shapes.push_back({m.name + "_K_proj", "Attn", n_kv, m.d_model});
        shapes.push_back({m.name + "_V_proj", "Attn", n_kv, m.d_model});
        shapes.push_back({m.name + "_Wo_proj", "Attn", m.d_model, n_q});
        shapes.push_back({m.name + "_FFN_Gate", "FFN", m.d_ff, m.d_model});
        shapes.push_back({m.name + "_FFN_Up", "FFN", m.d_ff, m.d_model});
        shapes.push_back({m.name + "_FFN_Down", "FFN", m.d_model, m.d_ff});

        return shapes;
    }

    // =========================================================================
    // Format definitions
    // =========================================================================

    struct FormatSpec
    {
        std::string name;
        bool is_nibble_lut; // true = 4-bit vpshufb path, false = INT8 pre-decoded
    };

    // All 16 supported NativeVNNI formats
    static const std::vector<FormatSpec> ALL_FORMATS = {
        // Nibble-LUT path (4-bit formats)
        {"Q4_0", true},
        {"Q4_1", true},
        {"IQ4_NL", true},
        {"IQ4_XS", true},
        // INT8 pre-decoded path
        {"Q5_0", false},
        {"Q5_1", false},
        {"Q6_K", false},
        {"Q3_K", false},
        {"Q2_K", false},
        {"IQ3_S", false},
        {"IQ3_XXS", false},
        {"IQ2_S", false},
        {"IQ2_XS", false},
        {"IQ2_XXS", false},
        {"IQ1_S", false},
        {"IQ1_M", false},
        // INT8 pre-decoded path (8-bit formats)
        {"Q8_0", false},
        {"Q8_1", false},
    };

    // Key formats for the larger cross-model sweep
    static const std::vector<FormatSpec> KEY_FORMATS = {
        {"Q4_0", true},
        {"IQ4_NL", true},
        {"Q4_1", true},
        {"Q5_0", false},
        {"Q5_1", false},
        {"Q6_K", false},
        {"Q3_K", false},
        {"Q2_K", false},
    };

    // =========================================================================
    // Tensor factory dispatch
    // =========================================================================

    std::unique_ptr<TensorBase> createWeights(const std::string &fmt, size_t N, size_t K)
    {
        if (fmt == "Q4_0") return TestTensorFactory::createQ4_0Random({N, K});
        if (fmt == "IQ4_NL") return TestTensorFactory::createIQ4_NLRandom({N, K});
        if (fmt == "Q4_1") return TestTensorFactory::createQ4_1Random({N, K});
        if (fmt == "IQ4_XS") return TestTensorFactory::createIQ4_XSRandom({N, K});
        if (fmt == "Q5_0") return TestTensorFactory::createQ5_0Random({N, K});
        if (fmt == "Q5_1") return TestTensorFactory::createQ5_1Random({N, K});
        if (fmt == "Q6_K") return TestTensorFactory::createQ6_KRandom({N, K});
        if (fmt == "Q3_K") return TestTensorFactory::createQ3_KRandom({N, K});
        if (fmt == "Q2_K") return TestTensorFactory::createQ2_KRandom({N, K});
        if (fmt == "IQ3_S") return TestTensorFactory::createIQ3_SRandom({N, K});
        if (fmt == "IQ3_XXS") return TestTensorFactory::createIQ3_XXSRandom({N, K});
        if (fmt == "IQ2_S") return TestTensorFactory::createIQ2_SRandom({N, K});
        if (fmt == "IQ2_XS") return TestTensorFactory::createIQ2_XSRandom({N, K});
        if (fmt == "IQ2_XXS") return TestTensorFactory::createIQ2_XXSRandom({N, K});
        if (fmt == "IQ1_S") return TestTensorFactory::createIQ1_SRandom({N, K});
        if (fmt == "IQ1_M") return TestTensorFactory::createIQ1_MRandom({N, K});
        if (fmt == "Q8_0") return TestTensorFactory::createQ8_0Random({N, K});
        if (fmt == "Q8_1") return TestTensorFactory::createQ8_1Random({N, K});
        return nullptr;
    }

    // =========================================================================
    // Benchmark infrastructure
    // =========================================================================

    static const int WARMUP_ITERS = 50;
    static const int BENCH_ITERS = 200;

    struct BenchResult
    {
        std::string format_name;
        std::string shape_name;
        std::string category;
        int N, K;
        double packed_bytes;      // Total packed weight buffer size
        double native_us;         // NativeVNNI p10 (us)
        double baseline_us;       // CPUQuantisedGemmKernel p10 (us)
        double native_bw_gbs;     // Effective BW for NativeVNNI (GB/s)
        double baseline_bw_gbs;   // Effective BW for baseline (GB/s)
        double roofline_pct;      // native BW / measured BW x 100
        double speedup;           // baseline / native
        bool is_nibble_lut;
    };

    /**
     * @brief Benchmark a kernel, return p10 latency in microseconds.
     */
    double benchKernel(ITensorGemm *kernel, const float *A, float *C,
                       int M, int N, int K, int warmup, int iters)
    {
        for (int i = 0; i < warmup; ++i)
            kernel->multiply(A, C, M, N, K);

        std::vector<double> times(iters);
        for (int i = 0; i < iters; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            kernel->multiply(A, C, M, N, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        std::sort(times.begin(), times.end());
        int p10_idx = std::max(0, (int)(iters * 0.1) - 1);
        return times[p10_idx];
    }

    /**
     * @brief Compute effective bandwidth in GB/s.
     *
     * For GEMV (M=1), the dominant data movement is reading the weight matrix.
     * We use the actual packed buffer size for the bandwidth calculation.
     */
    double computeBandwidth(double packed_bytes, double time_us)
    {
        return packed_bytes / (time_us * 1e-6) / 1e9;
    }

    // =========================================================================
    // Test fixture
    // =========================================================================

    class CPUNativeVNNISweepTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                MPI_Init(nullptr, nullptr);
        }
    };

    // =========================================================================
    // Helper: run one (format, shape) benchmark pair
    // =========================================================================

    bool benchmarkPair(const FormatSpec &fmt, const GEMVShape &shape,
                       BenchResult &result)
    {
        auto weights = createWeights(fmt.name, shape.N, shape.K);
        if (!weights)
            return false;

        // NativeVNNI kernel
        CPUNativeVNNIGemmKernel native_kernel(weights.get());
        if (!native_kernel.isValid())
            return false;

        // Baseline: CPUQuantisedGemmKernel (INT8 requantize path)
        auto baseline_kernel = std::make_unique<llaminar2::gemm::CPUQuantisedGemmKernel>(
            weights.get());

        // Random activations
        std::vector<float> A(shape.K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(shape.N, 0.0f);
        std::vector<float> C_baseline(shape.N, 0.0f);

        // Benchmark both kernels
        double native_us = benchKernel(
            &native_kernel, A.data(), C_native.data(), 1, shape.N, shape.K,
            WARMUP_ITERS, BENCH_ITERS);
        double baseline_us = benchKernel(
            baseline_kernel.get(), A.data(), C_baseline.data(), 1, shape.N, shape.K,
            WARMUP_ITERS, BENCH_ITERS);

        // Compute packed buffer sizes for bandwidth calculation.
        // NativeVNNI interleaved buffer is the dominant data read:
        //   nibble-LUT: stride=1024 per chunk per K-block
        //   INT8:       stride=2048 per chunk per K-block
        // Plus scales (4B/col/kblock) + compensation (4B/col/kblock)
        const auto &packed = native_kernel.packedWeights();
        int N_chunks = (shape.N + 63) / 64;
        int bpr = packed.blocks_per_row;
        double interleaved_bytes = (double)N_chunks * bpr * packed.interleaved_block_stride;
        double meta_bytes = (double)N_chunks * bpr * 64 * (4 + 4); // scales + comp
        double min_bytes = packed.is_asymmetric ? (double)N_chunks * bpr * 64 * 4 : 0;
        double total_packed_bytes = interleaved_bytes + meta_bytes + min_bytes;

        // Baseline always reads INT8 (1 byte/elem) + Q8_1 metadata (~12.5% overhead)
        double baseline_total_bytes = (double)shape.N * shape.K * 1.125;

        result.format_name = fmt.name;
        result.shape_name = shape.name;
        result.category = shape.category;
        result.N = shape.N;
        result.K = shape.K;
        result.packed_bytes = total_packed_bytes;
        result.native_us = native_us;
        result.baseline_us = baseline_us;
        result.native_bw_gbs = computeBandwidth(total_packed_bytes, native_us);
        result.baseline_bw_gbs = computeBandwidth(baseline_total_bytes, baseline_us);
        result.roofline_pct = result.native_bw_gbs / MEASURED_READ_BW_GBS * 100.0;
        result.speedup = baseline_us / native_us;
        result.is_nibble_lut = fmt.is_nibble_lut;

        return true;
    }

    // =========================================================================
    // Helper: render results table with bandwidth analysis
    // =========================================================================

    void renderResultsTable(const std::string &title,
                            const std::vector<BenchResult> &results)
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "Path" << "Shape" << "N" << "K"
              << "Wt MB" << "Nat us" << "Base us"
              << "Nat BW" << "Base BW" << "Roof%"
              << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::left);
        for (int c = 5; c <= 11; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        double total_native_us = 0, total_baseline_us = 0;
        double total_packed_bytes = 0;

        std::string last_format;
        for (const auto &r : results)
        {
            if (!last_format.empty() && r.format_name != last_format)
                table << fort::separator;
            last_format = r.format_name;

            char wt_buf[16], nat_buf[16], base_buf[16];
            char nat_bw_buf[16], base_bw_buf[16], roof_buf[16], spd_buf[16];

            std::snprintf(wt_buf, sizeof(wt_buf), "%.1f", r.packed_bytes / 1e6);
            std::snprintf(nat_buf, sizeof(nat_buf), "%.0f", r.native_us);
            std::snprintf(base_buf, sizeof(base_buf), "%.0f", r.baseline_us);
            std::snprintf(nat_bw_buf, sizeof(nat_bw_buf), "%.1f", r.native_bw_gbs);
            std::snprintf(base_bw_buf, sizeof(base_bw_buf), "%.1f", r.baseline_bw_gbs);
            std::snprintf(roof_buf, sizeof(roof_buf), "%.0f%%", r.roofline_pct);
            std::snprintf(spd_buf, sizeof(spd_buf), "%.2fx", r.speedup);

            table << r.format_name
                  << (r.is_nibble_lut ? "Nib" : "I8")
                  << r.shape_name << r.N << r.K
                  << wt_buf << nat_buf << base_buf
                  << nat_bw_buf << base_bw_buf << roof_buf
                  << spd_buf
                  << fort::endr;

            total_native_us += r.native_us;
            total_baseline_us += r.baseline_us;
            total_packed_bytes += r.packed_bytes;
        }

        table << fort::separator;
        char total_nat_buf[16], total_base_buf[16], total_spd_buf[16];
        char total_nbw_buf[16], total_roof_buf[16];
        std::snprintf(total_nat_buf, sizeof(total_nat_buf), "%.0f", total_native_us);
        std::snprintf(total_base_buf, sizeof(total_base_buf), "%.0f", total_baseline_us);
        std::snprintf(total_spd_buf, sizeof(total_spd_buf), "%.2fx",
                      total_baseline_us / total_native_us);
        double avg_native_bw = total_packed_bytes / (total_native_us * 1e-6) / 1e9;
        std::snprintf(total_nbw_buf, sizeof(total_nbw_buf), "%.1f", avg_native_bw);
        std::snprintf(total_roof_buf, sizeof(total_roof_buf), "%.0f%%",
                      avg_native_bw / MEASURED_READ_BW_GBS * 100.0);

        table << "TOTAL" << "" << "" << "" << ""
              << "" << total_nat_buf << total_base_buf
              << total_nbw_buf << "" << total_roof_buf
              << total_spd_buf
              << fort::endr;

        std::cout << "\n" << title << "\n";
        std::cout << "Roofline: " << MEASURED_READ_BW_GBS << " GB/s (STREAM read, 56 cores)\n\n";
        std::cout << table.to_string() << std::endl;
    }

    // =========================================================================
    // Per-format summary table
    // =========================================================================

    void renderFormatSummary(const std::vector<BenchResult> &results)
    {
        struct FormatSummary
        {
            std::string name;
            std::string path;
            int count = 0;
            double total_native_us = 0;
            double total_baseline_us = 0;
            double total_bytes = 0;
            double min_bw = 1e9, max_bw = 0;
            double min_roof = 1e9, max_roof = 0;
        };

        std::vector<FormatSummary> summaries;
        std::string last_fmt;
        FormatSummary *cur = nullptr;

        for (const auto &r : results)
        {
            if (r.format_name != last_fmt)
            {
                summaries.push_back({});
                cur = &summaries.back();
                cur->name = r.format_name;
                cur->path = r.is_nibble_lut ? "NibbleLUT" : "INT8";
                last_fmt = r.format_name;
            }
            cur->count++;
            cur->total_native_us += r.native_us;
            cur->total_baseline_us += r.baseline_us;
            cur->total_bytes += r.packed_bytes;
            cur->min_bw = std::min(cur->min_bw, r.native_bw_gbs);
            cur->max_bw = std::max(cur->max_bw, r.native_bw_gbs);
            cur->min_roof = std::min(cur->min_roof, r.roofline_pct);
            cur->max_roof = std::max(cur->max_roof, r.roofline_pct);
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "Path" << "Shapes"
              << "Avg BW" << "Min BW" << "Max BW"
              << "Min Roof%" << "Max Roof%"
              << "Avg Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        for (int c = 3; c <= 8; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &s : summaries)
        {
            double avg_bw = s.total_bytes / (s.total_native_us * 1e-6) / 1e9;
            double avg_speedup = s.total_baseline_us / s.total_native_us;

            char avg_bw_buf[16], min_bw_buf[16], max_bw_buf[16];
            char min_r_buf[16], max_r_buf[16], spd_buf[16];
            std::snprintf(avg_bw_buf, sizeof(avg_bw_buf), "%.1f", avg_bw);
            std::snprintf(min_bw_buf, sizeof(min_bw_buf), "%.1f", s.min_bw);
            std::snprintf(max_bw_buf, sizeof(max_bw_buf), "%.1f", s.max_bw);
            std::snprintf(min_r_buf, sizeof(min_r_buf), "%.0f%%", s.min_roof);
            std::snprintf(max_r_buf, sizeof(max_r_buf), "%.0f%%", s.max_roof);
            std::snprintf(spd_buf, sizeof(spd_buf), "%.2fx", avg_speedup);

            table << s.name << s.path << s.count
                  << avg_bw_buf << min_bw_buf << max_bw_buf
                  << min_r_buf << max_r_buf
                  << spd_buf
                  << fort::endr;
        }

        std::cout << "\n=== Format Summary ===\n";
        std::cout << table.to_string() << std::endl;
    }

    // =========================================================================
    // TEST 1: Q4_0 full shape sweep (M=1 decode, all models)
    //
    // Primary 4-bit format -- highest bandwidth efficiency at 0.5 B/elem.
    // =========================================================================

    TEST_F(CPUNativeVNNISweepTest, Q4_0_Decode_AllModels)
    {
        std::vector<BenchResult> results;

        for (const auto &model : {kQwen05B, kQwen15B, kQwen3B, kQwen7B})
        {
            auto shapes = buildShapes(model);
            for (const auto &shape : shapes)
            {
                BenchResult r;
                if (benchmarkPair(ALL_FORMATS[0], shape, r))
                    results.push_back(r);
            }
        }

        renderResultsTable(
            "==============================================================\n"
            "  CPU NativeVNNI Q4_0 GEMV (M=1) Full Sweep\n"
            "  NativeVNNI (0.5 B/elem) vs Baseline (1.0 B/elem)\n"
            "==============================================================",
            results);
    }

    // =========================================================================
    // TEST 2: All key formats x Qwen 3B shapes (M=1 decode)
    //
    // 8 representative formats across the largest model.
    // Compares nibble-LUT vs INT8 path performance.
    // =========================================================================

    TEST_F(CPUNativeVNNISweepTest, AllKeyFormats_Decode_7B)
    {
        std::vector<BenchResult> results;
        auto shapes = buildShapes(kQwen7B);

        for (const auto &fmt : KEY_FORMATS)
        {
            for (const auto &shape : shapes)
            {
                BenchResult r;
                if (benchmarkPair(fmt, shape, r))
                    results.push_back(r);
            }
        }

        renderResultsTable(
            "==============================================================\n"
            "  CPU NativeVNNI Key Formats x Qwen 7B (M=1 Decode)\n"
            "  8 formats x 7 shapes = 56 benchmark points\n"
            "  FFN weights ~32-65 MB: DRAM-bound (exceeds 38.5 MB L3)\n"
            "==============================================================",
            results);

        renderFormatSummary(results);
    }

    TEST_F(CPUNativeVNNISweepTest, AllKeyFormats_Decode_3B)
    {
        std::vector<BenchResult> results;
        auto shapes = buildShapes(kQwen3B);

        for (const auto &fmt : KEY_FORMATS)
        {
            for (const auto &shape : shapes)
            {
                BenchResult r;
                if (benchmarkPair(fmt, shape, r))
                    results.push_back(r);
            }
        }

        renderResultsTable(
            "==============================================================\n"
            "  CPU NativeVNNI Key Formats x Qwen 3B (M=1 Decode)\n"
            "  8 formats x 7 shapes = 56 benchmark points\n"
            "==============================================================",
            results);

        renderFormatSummary(results);
    }

    // =========================================================================
    // TEST 3: All 16 formats x Qwen 0.5B shapes (quick full-format sweep)
    //
    // Smaller model for faster iteration. Tests all formats including
    // ultra-low-bit (IQ1, IQ2) to measure relative bandwidth utilization.
    // =========================================================================

    TEST_F(CPUNativeVNNISweepTest, AllFormats_Decode_05B)
    {
        std::vector<BenchResult> results;
        auto shapes = buildShapes(kQwen05B);

        for (const auto &fmt : ALL_FORMATS)
        {
            for (const auto &shape : shapes)
            {
                BenchResult r;
                if (benchmarkPair(fmt, shape, r))
                    results.push_back(r);
            }
        }

        renderResultsTable(
            "==============================================================\n"
            "  CPU NativeVNNI ALL 16 Formats x Qwen 0.5B (M=1)\n"
            "  16 formats x 7 shapes = 112 benchmark points\n"
            "==============================================================",
            results);

        renderFormatSummary(results);
    }

    // =========================================================================
    // TEST 4: Shape-category analysis (Attention vs FFN bandwidth patterns)
    //
    // For Q4_0, compare bandwidth utilization across shape categories
    // to identify whether small Attention shapes underutilize bandwidth.
    // =========================================================================

    TEST_F(CPUNativeVNNISweepTest, Q4_0_ShapeCategory_Analysis)
    {
        std::vector<BenchResult> results;

        for (const auto &model : {kQwen05B, kQwen15B, kQwen3B, kQwen7B})
        {
            auto shapes = buildShapes(model);
            for (const auto &shape : shapes)
            {
                BenchResult r;
                if (benchmarkPair(ALL_FORMATS[0], shape, r))
                    results.push_back(r);
            }
        }

        // Group by category
        struct CatSummary
        {
            std::string name;
            int count = 0;
            double total_us = 0, total_bytes = 0;
            double min_bw = 1e9, max_bw = 0;
        };

        std::map<std::string, CatSummary> cats;
        for (const auto &r : results)
        {
            auto &c = cats[r.category];
            c.name = r.category;
            c.count++;
            c.total_us += r.native_us;
            c.total_bytes += r.packed_bytes;
            c.min_bw = std::min(c.min_bw, r.native_bw_gbs);
            c.max_bw = std::max(c.max_bw, r.native_bw_gbs);
        }

        fort::utf8_table cat_table;
        cat_table.set_border_style(FT_DOUBLE2_STYLE);
        cat_table << fort::header
                  << "Category" << "Shapes"
                  << "Avg BW (GB/s)" << "Min BW" << "Max BW"
                  << "Avg Roof%" << "Min Roof%" << "Max Roof%"
                  << fort::endr;

        cat_table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 2; c <= 7; ++c)
            cat_table.column(c).set_cell_text_align(fort::text_align::right);

        for (auto &[cat_name, c] : cats)
        {
            double avg_bw = c.total_bytes / (c.total_us * 1e-6) / 1e9;
            char avg_buf[16], min_buf[16], max_buf[16];
            char avg_r[16], min_r[16], max_r[16];
            std::snprintf(avg_buf, sizeof(avg_buf), "%.1f", avg_bw);
            std::snprintf(min_buf, sizeof(min_buf), "%.1f", c.min_bw);
            std::snprintf(max_buf, sizeof(max_buf), "%.1f", c.max_bw);
            std::snprintf(avg_r, sizeof(avg_r), "%.0f%%", avg_bw / MEASURED_READ_BW_GBS * 100);
            std::snprintf(min_r, sizeof(min_r), "%.0f%%", c.min_bw / MEASURED_READ_BW_GBS * 100);
            std::snprintf(max_r, sizeof(max_r), "%.0f%%", c.max_bw / MEASURED_READ_BW_GBS * 100);

            cat_table << cat_name << c.count
                      << avg_buf << min_buf << max_buf
                      << avg_r << min_r << max_r
                      << fort::endr;
        }

        std::cout << "\n==============================================================\n"
                  << "  Q4_0 Shape Category Analysis (Bandwidth per Category)\n"
                  << "==============================================================\n\n";
        std::cout << cat_table.to_string() << std::endl;

        // Per-shape detail sorted by bandwidth (highest first)
        std::sort(results.begin(), results.end(),
                  [](const BenchResult &a, const BenchResult &b)
                  { return a.native_bw_gbs > b.native_bw_gbs; });

        fort::utf8_table detail;
        detail.set_border_style(FT_DOUBLE2_STYLE);
        detail << fort::header
               << "Shape" << "Cat" << "N" << "K" << "Wt MB"
               << "Nat us" << "BW GB/s" << "Roof%"
               << "Speedup vs Base"
               << fort::endr;

        detail.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 4; c <= 8; ++c)
            detail.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &r : results)
        {
            char wt_buf[16], nat_buf[16], bw_buf[16], roof_buf[16], spd_buf[16];
            std::snprintf(wt_buf, sizeof(wt_buf), "%.1f", r.packed_bytes / 1e6);
            std::snprintf(nat_buf, sizeof(nat_buf), "%.0f", r.native_us);
            std::snprintf(bw_buf, sizeof(bw_buf), "%.1f", r.native_bw_gbs);
            std::snprintf(roof_buf, sizeof(roof_buf), "%.0f%%", r.roofline_pct);
            std::snprintf(spd_buf, sizeof(spd_buf), "%.2fx", r.speedup);

            detail << r.shape_name << r.category << r.N << r.K
                   << wt_buf << nat_buf << bw_buf << roof_buf << spd_buf
                   << fort::endr;
        }

        std::cout << "\nPer-shape detail (sorted by bandwidth, highest first):\n";
        std::cout << detail.to_string() << std::endl;
    }

    // =========================================================================
    // TEST 5: Prefill sweep (M=4,16,64) for Q4_0
    //
    // GEMM performance for small batch sizes.
    // =========================================================================

    TEST_F(CPUNativeVNNISweepTest, Q4_0_Prefill_SmallM)
    {
        std::vector<int> M_values = {4, 16, 64};
        auto shapes = buildShapes(kQwen3B);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "M" << "N" << "K"
              << "Nat us" << "Base us" << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 4; c <= 6; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (int M : M_values)
        {
            for (const auto &shape : shapes)
            {
                auto weights = TestTensorFactory::createQ4_0Random(
                    {(size_t)shape.N, (size_t)shape.K});
                if (!weights)
                    continue;

                CPUNativeVNNIGemmKernel native_kernel(weights.get());
                if (!native_kernel.isValid())
                    continue;

                auto baseline = std::make_unique<llaminar2::gemm::CPUQuantisedGemmKernel>(
                    weights.get());

                std::vector<float> A(M * shape.K);
                std::mt19937 rng(42);
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (auto &v : A)
                    v = dist(rng);

                std::vector<float> C_native(M * shape.N, 0.0f);
                std::vector<float> C_baseline(M * shape.N, 0.0f);

                double native_us = benchKernel(
                    &native_kernel, A.data(), C_native.data(), M, shape.N, shape.K,
                    20, 50);
                double baseline_us = benchKernel(
                    baseline.get(), A.data(), C_baseline.data(), M, shape.N, shape.K,
                    20, 50);

                char n_buf[16], b_buf[16], s_buf[16];
                std::snprintf(n_buf, sizeof(n_buf), "%.0f", native_us);
                std::snprintf(b_buf, sizeof(b_buf), "%.0f", baseline_us);
                std::snprintf(s_buf, sizeof(s_buf), "%.2fx", baseline_us / native_us);

                table << shape.name << M << shape.N << shape.K
                      << n_buf << b_buf << s_buf << fort::endr;
            }
            table << fort::separator;
        }

        std::cout << "\n==============================================================\n"
                  << "  CPU NativeVNNI Q4_0 Prefill (M=4,16,64) x Qwen 3B\n"
                  << "==============================================================\n\n";
        std::cout << table.to_string() << std::endl;
    }

} // anonymous namespace

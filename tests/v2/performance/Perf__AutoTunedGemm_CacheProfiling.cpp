/**
 * @file Perf__AutoTunedGemm_CacheProfiling.cpp
 * @brief Micro-benchmark harness for auto-tuned GEMM with cache profiling
 *
 * This benchmark tests the auto-tuned GEMM kernel with IQ4_NL weights
 * under large batch prefill conditions and profiles L1/L2/L3 cache behavior.
 *
 * Features:
 * - Tests multiple prefill batch sizes (32, 64, 128, 256, 512, 1024, 2048)
 * - Profiles cache misses using Linux perf_event_open
 * - Reports GFLOPS, bandwidth, cache miss rates, and CPI
 * - Validates auto-tuner variant selection
 * - Supports manual perf integration for detailed analysis
 *
 * Usage:
 *   # Standard run (built-in perf counters)
 *   ./run_benchmark.sh v2_perf_autotuned_gemm_cache
 *
 *   # Detailed perf profiling (external perf tool)
 *   perf stat -e cycles,instructions,cache-references,cache-misses,\
 *       L1-dcache-loads,L1-dcache-load-misses,\
 *       L1-icache-load-misses,\
 *       LLC-loads,LLC-load-misses,\
 *       dTLB-load-misses,iTLB-load-misses \
 *       ./build_v2_release/tests/v2/performance/v2_perf_autotuned_gemm_cache \
 *       --gtest_filter='*Prefill512*'
 *
 *   # perf record for hotspot analysis
 *   perf record -g -e cycles:pp \
 *       ./build_v2_release/tests/v2/performance/v2_perf_autotuned_gemm_cache \
 *       --gtest_filter='*Prefill1024*'
 *   perf report
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

// Linux perf_event support
#ifdef __linux__
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <asm/unistd.h>
#include <cstring>
#include <cerrno>
#endif

// V2 includes
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"
#include "backends/ComputeBackend.h"
#include "kernels/cpu/gemm/GemmAutoTuner.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

/**
 * @brief Hardware performance counters (Linux perf_event)
 */
struct PerfCounters
{
    long long cycles = 0;
    long long instructions = 0;
    long long cache_references = 0;
    long long cache_misses = 0;
    long long l1d_loads = 0;
    long long l1d_misses = 0;
    long long l1i_misses = 0;
    long long llc_loads = 0;
    long long llc_misses = 0;

    // Derived metrics
    double ipc() const { return cycles > 0 ? (double)instructions / cycles : 0.0; }
    double cpi() const { return instructions > 0 ? (double)cycles / instructions : 0.0; }
    double cache_miss_rate() const { return cache_references > 0 ? (double)cache_misses / cache_references : 0.0; }
    double l1d_miss_rate() const { return l1d_loads > 0 ? (double)l1d_misses / l1d_loads : 0.0; }
    double llc_miss_rate() const { return llc_loads > 0 ? (double)llc_misses / llc_loads : 0.0; }
};

/**
 * @brief RAII wrapper for perf_event file descriptors
 */
class PerfEventCounter
{
public:
#ifdef __linux__
    PerfEventCounter(uint32_t type, uint64_t config)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = type;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 0; // Include kernel events
        pe.exclude_hv = 1;     // Exclude hypervisor

        // Open perf event (pid=0 means current process, cpu=-1 means any CPU)
        fd_ = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        if (fd_ < 0)
        {
            valid_ = false;
            // Don't throw - gracefully degrade if perf not available
        }
        else
        {
            valid_ = true;
        }
    }

    ~PerfEventCounter()
    {
        if (fd_ >= 0)
        {
            close(fd_);
        }
    }

    void start()
    {
        if (valid_)
        {
            ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
        }
    }

    void stop()
    {
        if (valid_)
        {
            ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
        }
    }

    long long read()
    {
        if (!valid_)
            return 0;

        long long count = 0;
        ssize_t bytes = ::read(fd_, &count, sizeof(long long));
        if (bytes != sizeof(long long))
        {
            return 0;
        }
        return count;
    }

    bool is_valid() const { return valid_; }

private:
    int fd_ = -1;
    bool valid_ = false;
#else
    PerfEventCounter(uint32_t, uint64_t) {}
    void start() {}
    void stop() {}
    long long read() { return 0; }
    bool is_valid() const { return false; }
#endif
};

/**
 * @brief Benchmark configuration
 */
struct BenchmarkConfig
{
    int seq_len;             ///< Sequence length (m dimension) - prefill batch size
    int in_features;         ///< Input features (k dimension) - d_model
    int out_features;        ///< Output features (n dimension) - d_model
    int warmup_iters;        ///< Warmup iterations
    int bench_iters;         ///< Timed benchmark iterations
    bool enable_perf;        ///< Enable perf counter profiling
    std::string description; ///< Description
};

/**
 * @brief Benchmark results with cache profiling
 */
struct CacheProfBenchmarkResult
{
    double time_ms;
    double gflops;
    double bandwidth_gb;
    PerfCounters perf;
};

/**
 * @brief Auto-tuned GEMM cache profiling test fixture
 */
class AutoTunedGemmCacheProfiling : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::unique_ptr<ModelLoader> loader_;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        DeviceManager::instance().initialize(-1);

        // Verify OpenMP configuration
        int max_threads = omp_get_max_threads();
        if (rank_ == 0)
        {
            std::cout << "[Cache Profiling] OpenMP threads: " << max_threads << std::endl;
            std::cout << "[Cache Profiling] MPI ranks: " << world_size_ << std::endl;

#ifdef __linux__
            std::cout << "[Cache Profiling] perf_event support: enabled" << std::endl;
#else
            std::cout << "[Cache Profiling] perf_event support: disabled (not Linux)" << std::endl;
#endif
        }

        ASSERT_GT(max_threads, 1) << "OpenMP not configured! Set OMP_NUM_THREADS";

        // Load IQ4_NL model
        std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";

        try
        {
            loader_ = std::make_unique<ModelLoader>();

            if (rank_ == 0)
            {
                std::cout << "[Cache Profiling] Loading model: " << model_path << std::endl;
            }

            if (!loader_->loadModel(model_path))
            {
                throw std::runtime_error("Failed to load model");
            }

            if (rank_ == 0)
            {
                std::cout << "[Cache Profiling] Model loaded successfully" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            if (rank_ == 0)
            {
                std::cerr << "[Cache Profiling] Model load failed: " << e.what() << std::endl;
            }
            GTEST_SKIP() << "Model not available: " << model_path;
        }
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        loader_.reset();
    }

    /**
     * @brief Create FP32 activation tensor
     */
    std::shared_ptr<FP32Tensor> createActivation(int seq_len, int features)
    {
        auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(seq_len),
            static_cast<size_t>(features)});

        float *data = tensor->mutable_data();
        size_t total = seq_len * features;

        // Initialize with pseudo-random data
        for (size_t i = 0; i < total; ++i)
        {
            data[i] = (static_cast<float>(i % 1000) / 1000.0f) - 0.5f;
        }

        return tensor;
    }

    /**
     * @brief Get IQ4_NL weight tensor from model
     */
    std::shared_ptr<IQ4_NLTensor> getWeightTensor()
    {
        // Use first layer Q projection
        std::string weight_name = "blk.0.attn_q.weight";
        auto weight = loader_->loadTensor(weight_name, -1 /* CPU */);

        if (!weight)
        {
            throw std::runtime_error("Failed to load weight: " + weight_name);
        }

        auto iq4nl = std::dynamic_pointer_cast<IQ4_NLTensor>(weight);
        if (!iq4nl)
        {
            throw std::runtime_error("Weight is not IQ4_NL format");
        }

        return iq4nl;
    }

    /**
     * @brief Run benchmark with cache profiling
     */
    CacheProfBenchmarkResult runBenchmark(const BenchmarkConfig &config)
    {
        CacheProfBenchmarkResult result = {};

        // Create tensors
        auto activation = createActivation(config.seq_len, config.in_features);
        auto weight = getWeightTensor();
        auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(config.seq_len),
            static_cast<size_t>(config.out_features)});

        // Create GEMM kernel (auto-tuned)
        auto gemm = weight->createGemm();

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            gemm->multiply(
                activation->data(),
                output->mutable_data(),
                config.seq_len,
                config.out_features,
                config.in_features,
                true, // transpose_B
                1.0f, 0.0f,
                nullptr, -1);
        }

        // Initialize perf counters
#ifdef __linux__
        PerfEventCounter cycles_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        PerfEventCounter insns_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        PerfEventCounter cache_refs_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
        PerfEventCounter cache_miss_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);

        // L1 data cache counters
        uint64_t l1d_read_access = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
        uint64_t l1d_read_miss = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        PerfEventCounter l1d_loads_counter(PERF_TYPE_HW_CACHE, l1d_read_access);
        PerfEventCounter l1d_misses_counter(PERF_TYPE_HW_CACHE, l1d_read_miss);

        // L1 instruction cache counters
        uint64_t l1i_read_miss = (PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        PerfEventCounter l1i_misses_counter(PERF_TYPE_HW_CACHE, l1i_read_miss);

        // LLC (L3) counters
        uint64_t llc_read_access = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
        uint64_t llc_read_miss = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        PerfEventCounter llc_loads_counter(PERF_TYPE_HW_CACHE, llc_read_access);
        PerfEventCounter llc_misses_counter(PERF_TYPE_HW_CACHE, llc_read_miss);

        bool perf_available = cycles_counter.is_valid() && insns_counter.is_valid();

        if (config.enable_perf && !perf_available && rank_ == 0)
        {
            std::cout << "[WARNING] perf counters not available (need root or kernel.perf_event_paranoid <= 2)" << std::endl;
            std::cout << "[INFO] To enable: sudo sysctl -w kernel.perf_event_paranoid=1" << std::endl;
        }
#else
        bool perf_available = false;
#endif

        // Benchmark iterations with perf profiling
        MPI_Barrier(MPI_COMM_WORLD);

#ifdef __linux__
        if (config.enable_perf && perf_available)
        {
            cycles_counter.start();
            insns_counter.start();
            cache_refs_counter.start();
            cache_miss_counter.start();
            l1d_loads_counter.start();
            l1d_misses_counter.start();
            l1i_misses_counter.start();
            llc_loads_counter.start();
            llc_misses_counter.start();
        }
#endif

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < config.bench_iters; ++i)
        {
            gemm->multiply(
                activation->data(),
                output->mutable_data(),
                config.seq_len,
                config.out_features,
                config.in_features,
                true, 1.0f, 0.0f,
                nullptr, -1);
        }

        auto end = std::chrono::high_resolution_clock::now();

#ifdef __linux__
        if (config.enable_perf && perf_available)
        {
            cycles_counter.stop();
            insns_counter.stop();
            cache_refs_counter.stop();
            cache_miss_counter.stop();
            l1d_loads_counter.stop();
            l1d_misses_counter.stop();
            l1i_misses_counter.stop();
            llc_loads_counter.stop();
            llc_misses_counter.stop();

            result.perf.cycles = cycles_counter.read();
            result.perf.instructions = insns_counter.read();
            result.perf.cache_references = cache_refs_counter.read();
            result.perf.cache_misses = cache_miss_counter.read();
            result.perf.l1d_loads = l1d_loads_counter.read();
            result.perf.l1d_misses = l1d_misses_counter.read();
            result.perf.l1i_misses = l1i_misses_counter.read();
            result.perf.llc_loads = llc_loads_counter.read();
            result.perf.llc_misses = llc_misses_counter.read();
        }
#endif

        MPI_Barrier(MPI_COMM_WORLD);

        // Calculate performance metrics
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.time_ms = elapsed_ms / config.bench_iters;

        // GFLOPS = 2 * M * N * K / (time_seconds * 1e9)
        double flops = 2.0 * config.seq_len * config.out_features * config.in_features;
        result.gflops = flops / (result.time_ms / 1000.0 * 1e9);

        // Bandwidth (GB/s)
        size_t weight_bytes = (config.out_features * config.in_features) / 2; // ~4 bits/element
        size_t act_bytes = config.seq_len * config.in_features * sizeof(float);
        size_t out_bytes = config.seq_len * config.out_features * sizeof(float);
        double total_bytes = weight_bytes + act_bytes + out_bytes;
        result.bandwidth_gb = (total_bytes / result.time_ms) / 1e6;

        return result;
    }

    /**
     * @brief Print benchmark results
     */
    void printResult(const BenchmarkConfig &config, const CacheProfBenchmarkResult &result)
    {
        if (rank_ != 0)
            return;

        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(62) << config.description << " ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Configuration:                                                 ║\n";
        std::cout << "║   Sequence Length:  " << std::setw(10) << config.seq_len << "                                      ║\n";
        std::cout << "║   Input Features:   " << std::setw(10) << config.in_features << "                                      ║\n";
        std::cout << "║   Output Features:  " << std::setw(10) << config.out_features << "                                      ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Performance:                                                   ║\n";
        std::cout << "║   Time per iter:    " << std::setw(10) << std::fixed << std::setprecision(3) << result.time_ms << " ms                                   ║\n";
        std::cout << "║   Throughput:       " << std::setw(10) << std::fixed << std::setprecision(2) << result.gflops << " GFLOPS                               ║\n";
        std::cout << "║   Bandwidth:        " << std::setw(10) << std::fixed << std::setprecision(2) << result.bandwidth_gb << " GB/s                                ║\n";

        if (config.enable_perf && result.perf.cycles > 0)
        {
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Cache Profiling:                                               ║\n";
            std::cout << "║   Cycles:           " << std::setw(15) << result.perf.cycles << "                                   ║\n";
            std::cout << "║   Instructions:     " << std::setw(15) << result.perf.instructions << "                                   ║\n";
            std::cout << "║   IPC:              " << std::setw(10) << std::fixed << std::setprecision(2) << result.perf.ipc() << "                                         ║\n";
            std::cout << "║   CPI:              " << std::setw(10) << std::fixed << std::setprecision(2) << result.perf.cpi() << "                                         ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Cache Statistics:                                              ║\n";
            std::cout << "║   Cache Refs:       " << std::setw(15) << result.perf.cache_references << "                                   ║\n";
            std::cout << "║   Cache Misses:     " << std::setw(15) << result.perf.cache_misses << "                                   ║\n";
            std::cout << "║   Miss Rate:        " << std::setw(10) << std::fixed << std::setprecision(2) << (result.perf.cache_miss_rate() * 100) << "%                                       ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ L1D Cache:                                                     ║\n";
            std::cout << "║   Loads:            " << std::setw(15) << result.perf.l1d_loads << "                                   ║\n";
            std::cout << "║   Misses:           " << std::setw(15) << result.perf.l1d_misses << "                                   ║\n";
            std::cout << "║   Miss Rate:        " << std::setw(10) << std::fixed << std::setprecision(2) << (result.perf.l1d_miss_rate() * 100) << "%                                       ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ LLC (L3) Cache:                                                ║\n";
            std::cout << "║   Loads:            " << std::setw(15) << result.perf.llc_loads << "                                   ║\n";
            std::cout << "║   Misses:           " << std::setw(15) << result.perf.llc_misses << "                                   ║\n";
            std::cout << "║   Miss Rate:        " << std::setw(10) << std::fixed << std::setprecision(2) << (result.perf.llc_miss_rate() * 100) << "%                                       ║\n";
        }

        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    }

    /**
     * @brief Query auto-tuner for selected variant
     */
    void printAutoTunerSelection(int m, int n, int k)
    {
        if (rank_ != 0)
            return;

        auto &tuner = GemmAutoTuner::instance();
        auto *kernel = tuner.getOptimalKernel(m, n, k);

        if (kernel)
        {
            auto config = kernel->config();
            std::cout << "\n[Auto-Tuner Selection]\n";
            std::cout << "  Shape: [" << m << ", " << n << ", " << k << "]\n";
            std::cout << "  Variant: " << config.id() << "\n";
            std::cout << "  Tile M×N: " << config.tile_m << "×" << config.tile_n << "\n";
            std::cout << "  Unroll Factor: " << config.unroll_factor << "\n";
            std::cout << "  Prefetch Blocks: " << config.prefetch_blocks << "\n";
            std::cout << "\n";
        }
    }
};

// ============================================================
// Test Cases: Large Batch Prefill Scenarios
// ============================================================

/**
 * @brief Small batch prefill (32 tokens)
 */
TEST_F(AutoTunedGemmCacheProfiling, Prefill32)
{
    BenchmarkConfig config = {
        .seq_len = 32,
        .in_features = 896, // Qwen 2.5 0.5B d_model
        .out_features = 896,
        .warmup_iters = 5,
        .bench_iters = 100,
        .enable_perf = true,
        .description = "Prefill Batch Size: 32 tokens"};

    printAutoTunerSelection(config.seq_len, config.out_features, config.in_features);
    CacheProfBenchmarkResult result = runBenchmark(config);
    printResult(config, result);

    EXPECT_GT(result.gflops, 0.0) << "GEMM produced zero throughput";
}

/**
 * @brief Medium batch prefill (128 tokens)
 */
TEST_F(AutoTunedGemmCacheProfiling, Prefill128)
{
    BenchmarkConfig config = {
        .seq_len = 128,
        .in_features = 896,
        .out_features = 896,
        .warmup_iters = 5,
        .bench_iters = 50,
        .enable_perf = true,
        .description = "Prefill Batch Size: 128 tokens"};

    printAutoTunerSelection(config.seq_len, config.out_features, config.in_features);
    CacheProfBenchmarkResult result = runBenchmark(config);
    printResult(config, result);

    EXPECT_GT(result.gflops, 0.0);
}

/**
 * @brief Large batch prefill (512 tokens) - typical long context
 */
TEST_F(AutoTunedGemmCacheProfiling, Prefill512)
{
    BenchmarkConfig config = {
        .seq_len = 512,
        .in_features = 896,
        .out_features = 896,
        .warmup_iters = 3,
        .bench_iters = 20,
        .enable_perf = true,
        .description = "Prefill Batch Size: 512 tokens (long context)"};

    printAutoTunerSelection(config.seq_len, config.out_features, config.in_features);
    CacheProfBenchmarkResult result = runBenchmark(config);
    printResult(config, result);

    EXPECT_GT(result.gflops, 0.0);
}

/**
 * @brief Very large batch prefill (1024 tokens) - stress test L3 cache
 */
TEST_F(AutoTunedGemmCacheProfiling, Prefill1024)
{
    BenchmarkConfig config = {
        .seq_len = 1024,
        .in_features = 896,
        .out_features = 896,
        .warmup_iters = 2,
        .bench_iters = 10,
        .enable_perf = true,
        .description = "Prefill Batch Size: 1024 tokens (L3 stress)"};

    printAutoTunerSelection(config.seq_len, config.out_features, config.in_features);
    CacheProfBenchmarkResult result = runBenchmark(config);
    printResult(config, result);

    EXPECT_GT(result.gflops, 0.0);
}

/**
 * @brief Extreme batch prefill (2048 tokens) - DRAM bandwidth bound
 */
TEST_F(AutoTunedGemmCacheProfiling, Prefill2048)
{
    BenchmarkConfig config = {
        .seq_len = 2048,
        .in_features = 896,
        .out_features = 896,
        .warmup_iters = 2,
        .bench_iters = 5,
        .enable_perf = true,
        .description = "Prefill Batch Size: 2048 tokens (DRAM bound)"};

    printAutoTunerSelection(config.seq_len, config.out_features, config.in_features);
    CacheProfBenchmarkResult result = runBenchmark(config);
    printResult(config, result);

    EXPECT_GT(result.gflops, 0.0);
}

/**
 * @brief Maximum batch prefill (4096 tokens) - extreme memory pressure
 */
TEST_F(AutoTunedGemmCacheProfiling, Prefill4096)
{
    BenchmarkConfig config = {
        .seq_len = 4096,
        .in_features = 896,
        .out_features = 896,
        .warmup_iters = 1,
        .bench_iters = 3,
        .enable_perf = true,
        .description = "Prefill Batch Size: 4096 tokens (extreme)"};

    printAutoTunerSelection(config.seq_len, config.out_features, config.in_features);
    CacheProfBenchmarkResult result = runBenchmark(config);
    printResult(config, result);

    EXPECT_GT(result.gflops, 0.0);
}

/**
 * @brief Comparative test: scaling across batch sizes
 */
TEST_F(AutoTunedGemmCacheProfiling, BatchScaling)
{
    if (rank_ != 0)
        return;

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Batch Size Scaling Analysis                                    ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Batch  │ Time(ms) │ GFLOPS │ BW(GB/s) │  L1D Miss  │ LLC Miss  ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";

    std::vector<int> batch_sizes = {32, 64, 128, 256, 512, 1024, 2048};

    for (int batch : batch_sizes)
    {
        BenchmarkConfig config = {
            .seq_len = batch,
            .in_features = 896,
            .out_features = 896,
            .warmup_iters = 3,
            .bench_iters = std::max(3, 100 / (batch / 32)),
            .enable_perf = true,
            .description = ""};

        CacheProfBenchmarkResult result = runBenchmark(config);

        std::cout << "║ " << std::setw(6) << batch << " │ ";
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << result.time_ms << " │ ";
        std::cout << std::setw(6) << std::fixed << std::setprecision(1) << result.gflops << " │ ";
        std::cout << std::setw(8) << std::fixed << std::setprecision(1) << result.bandwidth_gb << " │ ";

        if (result.perf.l1d_loads > 0)
        {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << (result.perf.l1d_miss_rate() * 100) << "% │ ";
        }
        else
        {
            std::cout << "      N/A  │ ";
        }

        if (result.perf.llc_loads > 0)
        {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << (result.perf.llc_miss_rate() * 100) << "% ║\n";
        }
        else
        {
            std::cout << "    N/A  ║\n";
        }
    }

    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}

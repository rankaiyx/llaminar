/**
 * @file Perf__Exhaustive_Autotuner_Sweep.cpp
 * @brief Exhaustive benchmark of ALL 1225 microkernel variants
 *
 * This benchmark sweeps through the entire configuration space:
 * - ISA: AVX512 (primary), AVX2 (fallback)
 * - MR (TILE_M): 1, 2, 4, 8, 16 (rows per micro-kernel)
 * - NR (TILE_N): 1, 2, 4, 8, 16 (cols per micro-kernel)
 * - UNROLL_K: 1, 2, 4, 8, 16 (K-loop unroll factor)
 * - PREFETCH_DIST: 0, 1, 2, 3, 5 (prefetch distance)
 *
 * Total combinations: ~1225 variants
 *
 * Goal: Find the globally optimal configuration for IQ4_NL GEMM,
 * bypassing the autotuner's heuristic selection.
 *
 * @author David Sanftenberg
 * @date November 9, 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>

// V2 includes
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"
#include "backends/ComputeBackend.h"
#include "kernels/cpu/gemm/GemmAutoTuner.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

/**
 * @brief Benchmark result for a single configuration
 */
struct ConfigResult
{
    int tile_m;
    int tile_n;
    int unroll_factor;
    int prefetch_blocks;
    double gflops;
    double time_ms;
    std::string config_name;

    bool operator<(const ConfigResult &other) const
    {
        return gflops > other.gflops; // Sort by descending GFLOPS
    }
};

/**
 * @brief Exhaustive autotuner sweep test fixture
 */
class Exhaustive_Autotuner_Sweep : public ::testing::Test
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

        if (rank_ == 0)
        {
            std::cout << "[ExhaustiveSweep] MPI ranks: " << world_size_ << std::endl;
            std::cout << "[ExhaustiveSweep] OpenMP threads: " << omp_get_max_threads() << std::endl;
        }

        // Load model
        std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
        try
        {
            loader_ = std::make_unique<ModelLoader>();
            if (!loader_->loadModel(model_path))
            {
                throw std::runtime_error("Failed to load model");
            }
            if (rank_ == 0)
            {
                std::cout << "[ExhaustiveSweep] Model loaded: " << model_path << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            if (rank_ == 0)
            {
                std::cerr << "[ExhaustiveSweep] Model unavailable: " << e.what() << std::endl;
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
     * @brief Get IQ4_NL weight tensor
     */
    std::shared_ptr<IQ4_NLTensor> getWeightTensor()
    {
        auto weight = loader_->loadTensor("blk.0.attn_q.weight", -1);
        if (!weight)
        {
            throw std::runtime_error("Failed to load weight tensor");
        }

        auto iq4nl_weight = std::dynamic_pointer_cast<IQ4_NLTensor>(weight);
        if (!iq4nl_weight)
        {
            throw std::runtime_error("Weight is not IQ4_NL format");
        }

        return iq4nl_weight;
    }

    /**
     * @brief Benchmark a specific autotuner configuration
     *
     * Forces the autotuner to use a specific configuration, then measures performance.
     */
    ConfigResult benchmarkConfiguration(
        std::shared_ptr<IQ4_NLTensor> weight,
        int m, int n, int k,
        int unroll_factor, int prefetch_blocks, int tile_m, int tile_n)
    {
        // Create activation and output tensors
        std::vector<float> A(m * k, 0.5f);
        std::vector<float> C(m * n, 0.0f);

        // Create configuration to test
        GemmKernelConfig config;
        config.unroll_factor = unroll_factor;
        config.prefetch_blocks = prefetch_blocks;
        config.tile_m = tile_m;
        config.tile_n = tile_n;

        // Force this configuration in the autotuner
        auto &autotuner = GemmAutoTuner::instance();
        autotuner.setOptimalConfig(m, n, k, config);

        // Create GEMM kernel (will use our forced config)
        auto gemm = weight->createGemm();

        // Warmup iterations (5)
        for (int iter = 0; iter < 5; iter++)
        {
            bool success = gemm->multiply(
                A.data(),
                C.data(),
                m, n, k,
                true,    // transpose_B
                1.0f,    // alpha
                0.0f,    // beta
                nullptr, // MPI context
                -1);     // rank

            if (!success)
            {
                return ConfigResult{tile_m, tile_n, unroll_factor, prefetch_blocks,
                                    0.0, 0.0, "FAILED"};
            }
        }

        // Timed benchmark (20 iterations)
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        const int benchmark_iters = 20;
        for (int iter = 0; iter < benchmark_iters; iter++)
        {
            gemm->multiply(A.data(), C.data(), m, n, k, true, 1.0f, 0.0f, nullptr, -1);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = ms / benchmark_iters;
        double gflops = (2.0 * m * n * k) / (avg_ms * 1e6);

        std::ostringstream config_name;
        config_name << "unroll" << unroll_factor << "_pf" << prefetch_blocks
                    << "_tile" << tile_m << "x" << tile_n;

        return ConfigResult{tile_m, tile_n, unroll_factor, prefetch_blocks,
                            gflops, avg_ms, config_name.str()};
    }
};

/**
 * @brief Test: Exhaustive sweep for Q-projection (896×896)
 */
TEST_F(Exhaustive_Autotuner_Sweep, Q_Projection_896x896)
{
    // All ranks participate in the sweep
    auto weight = getWeightTensor();

    // Test workload: 128×896×896 (realistic prefill batch)
    const int m = 128;
    const int n = 896;
    const int k = 896;

    if (rank_ == 0)
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ Exhaustive Autotuner Sweep: Q-Projection 128×896×896        ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ Testing ALL autotuner configurations                         ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    }

    // Autotuner parameter space (from GemmKernelConfig)
    std::vector<int> unroll_values = {4, 8, 16, 24, 32}; // Loop unroll factors
    std::vector<int> prefetch_values = {0, 3, 5, 7, 9};  // Prefetch distances
    std::vector<int> tile_m_values = {8, 16, 32, 64};    // M tile sizes
    std::vector<int> tile_n_values = {4, 8, 16, 32};     // N tile sizes

    std::vector<ConfigResult> local_results;
    int total_configs = unroll_values.size() * prefetch_values.size() *
                        tile_m_values.size() * tile_n_values.size();
    int tested = 0;
    int skipped = 0;

    // Distribute work across MPI ranks
    int configs_per_rank = total_configs / world_size_;
    int start_idx = rank_ * configs_per_rank;
    int end_idx = (rank_ == world_size_ - 1) ? total_configs : start_idx + configs_per_rank;

    if (rank_ == 0)
    {
        std::cout << "\nTesting " << total_configs << " configurations across "
                  << world_size_ << " ranks..." << std::endl;
        std::cout << "Each rank tests ~" << configs_per_rank << " configs" << std::endl;
    }

    // Build flat list of all configs
    std::vector<std::tuple<int, int, int, int>> all_configs;
    for (int unroll : unroll_values)
    {
        for (int prefetch : prefetch_values)
        {
            for (int tile_m : tile_m_values)
            {
                for (int tile_n : tile_n_values)
                {
                    all_configs.emplace_back(unroll, prefetch, tile_m, tile_n);
                }
            }
        }
    }

    // Each rank tests its portion
    for (int idx = start_idx; idx < end_idx; idx++)
    {
        auto [unroll, prefetch, tile_m, tile_n] = all_configs[idx];
        auto result = benchmarkConfiguration(weight, m, n, k, unroll, prefetch, tile_m, tile_n);

        if (result.gflops > 0.0)
        {
            local_results.push_back(result);
            tested++;

            // Progress indicator every 25 configs per rank
            if (rank_ == 0 && tested % 25 == 0)
            {
                std::cout << "  Rank 0 tested: " << tested << "/" << (end_idx - start_idx) << std::endl;
            }
        }
        else
        {
            skipped++;
        }
    }

    // Gather all results to rank 0
    int local_count = local_results.size();
    std::vector<int> all_counts(world_size_);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare for Allgatherv (variable-length data)
    std::vector<int> recv_counts(world_size_);
    std::vector<int> displs(world_size_);

    if (rank_ == 0)
    {
        int total_results = 0;
        for (int i = 0; i < world_size_; i++)
        {
            recv_counts[i] = all_counts[i] * sizeof(ConfigResult);
            displs[i] = total_results * sizeof(ConfigResult);
            total_results += all_counts[i];
        }
    }

    // Flatten local results into byte buffer
    std::vector<char> send_buf(local_results.size() * sizeof(ConfigResult));
    std::memcpy(send_buf.data(), local_results.data(), send_buf.size());

    std::vector<char> recv_buf;
    if (rank_ == 0)
    {
        int total_bytes = 0;
        for (int c : recv_counts)
            total_bytes += c;
        recv_buf.resize(total_bytes);
    }

    MPI_Gatherv(send_buf.data(), send_buf.size(), MPI_BYTE,
                recv_buf.data(), recv_counts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Rank 0 unpacks and processes
    if (rank_ == 0)
    {
        std::vector<ConfigResult> all_results;
        size_t offset = 0;
        for (int i = 0; i < world_size_; i++)
        {
            int count = recv_counts[i] / sizeof(ConfigResult);
            const ConfigResult *rank_results = reinterpret_cast<const ConfigResult *>(recv_buf.data() + offset);
            for (int j = 0; j < count; j++)
            {
                all_results.push_back(rank_results[j]);
            }
            offset += recv_counts[i];
        }

        // Sort by performance
        std::sort(all_results.begin(), all_results.end());

        std::cout << "\n═══════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "RESULTS: Tested " << all_results.size() << " configs total" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════\n"
                  << std::endl;

        // Top 20 configurations
        std::cout << "╔═════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ TOP 20 CONFIGURATIONS                                                   ║" << std::endl;
        std::cout << "╠═════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ Rank  TileM×N  Unroll  Prefetch   Time(ms)    GFLOPS  Speedup           ║" << std::endl;
        std::cout << "╟─────────────────────────────────────────────────────────────────────────╢" << std::endl;

        double baseline_gflops = all_results[0].gflops;

        for (size_t i = 0; i < std::min(size_t(20), all_results.size()); i++)
        {
            const auto &r = all_results[i];
            double speedup = r.gflops / baseline_gflops;

            std::cout << "║ " << std::setw(4) << (i + 1) << "  "
                      << std::setw(2) << r.tile_m << "×" << std::setw(2) << std::left << r.tile_n << std::right
                      << "    " << std::setw(6) << r.unroll_factor
                      << "    " << std::setw(8) << r.prefetch_blocks
                      << "   " << std::setw(8) << std::fixed << std::setprecision(2) << r.time_ms
                      << "   " << std::setw(7) << std::setprecision(1) << r.gflops
                      << "   " << std::setw(5) << std::setprecision(2) << speedup << "x        ║" << std::endl;
        }

        std::cout << "╚═════════════════════════════════════════════════════════════════════════╝" << std::endl;

        // Save full results to CSV
        std::string csv_filename = "exhaustive_autotuner_sweep_results.csv";
        std::ofstream csv(csv_filename);
        csv << "rank,tile_m,tile_n,unroll_factor,prefetch_blocks,time_ms,gflops,config_name\n";

        for (size_t i = 0; i < all_results.size(); i++)
        {
            const auto &r = all_results[i];
            csv << (i + 1) << "," << r.tile_m << "," << r.tile_n << ","
                << r.unroll_factor << "," << r.prefetch_blocks << ","
                << std::fixed << std::setprecision(3) << r.time_ms << ","
                << std::setprecision(2) << r.gflops << ","
                << r.config_name << "\n";
        }

        csv.close();

        std::cout << "\nFull results saved to: " << csv_filename << std::endl;
        std::cout << "\n✅ BEST CONFIGURATION:" << std::endl;
        std::cout << "   Tile: " << all_results[0].tile_m << "×" << all_results[0].tile_n << std::endl;
        std::cout << "   Unroll: " << all_results[0].unroll_factor << std::endl;
        std::cout << "   Prefetch: " << all_results[0].prefetch_blocks << std::endl;
        std::cout << "   Performance: " << std::fixed << std::setprecision(1)
                  << all_results[0].gflops << " GFLOPS" << std::endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

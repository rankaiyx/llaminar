/**
 * @file PerfHarness.h
 * @brief Performance benchmarking harness for MPI attention testing (Phase 3b)
 * @author David Sanftenberg
 *
 * Provides multi-trial timing with statistics for performance analysis.
 */

#pragma once

#include <functional>
#include <vector>
#include <chrono>
#include <string>

namespace llaminar2
{
    namespace test
    {
        /**
         * @brief Performance measurement result
         */
        struct PerfResult
        {
            int trials;             ///< Number of measurement trials
            int warmup;             ///< Number of warmup iterations
            double mean_ms;         ///< Mean execution time (milliseconds)
            double min_ms;          ///< Minimum execution time
            double max_ms;          ///< Maximum execution time
            double stddev_ms;       ///< Standard deviation
            double cv_percent;      ///< Coefficient of variation (%)
            double gflops;          ///< Computed GFLOPS (if operation count provided)
            double speedup;         ///< Speedup vs baseline (if baseline provided)
            double efficiency;      ///< Parallel efficiency (speedup / num_processes)
            size_t operation_count; ///< Total operation count (for GFLOPS)
            int num_processes;      ///< Number of MPI processes

            /**
             * @brief Default constructor
             */
            PerfResult()
                : trials(0), warmup(0), mean_ms(0.0), min_ms(0.0), max_ms(0.0),
                  stddev_ms(0.0), cv_percent(0.0), gflops(0.0), speedup(0.0),
                  efficiency(0.0), operation_count(0), num_processes(1)
            {
            }
        };

        /**
         * @brief MPI-specific performance breakdown
         */
        struct MPIProfile
        {
            double compute_time_ms;       ///< Local computation time
            double allreduce_time_ms;     ///< MPI_Allreduce time
            double barrier_time_ms;       ///< MPI_Barrier time
            double total_time_ms;         ///< Total execution time
            double comm_overhead_percent; ///< Communication overhead percentage

            /**
             * @brief Default constructor
             */
            MPIProfile()
                : compute_time_ms(0.0), allreduce_time_ms(0.0), barrier_time_ms(0.0),
                  total_time_ms(0.0), comm_overhead_percent(0.0)
            {
            }
        };

        /**
         * @brief Performance benchmarking harness
         *
         * Provides multi-trial timing with statistics for accurate performance
         * measurement. Handles warmup iterations to stabilize cache/branch predictor.
         */
        class PerfHarness
        {
        public:
            /**
             * @brief Benchmark a function with multiple trials
             *
             * @param fn Function to benchmark (should perform one complete operation)
             * @param trials Number of measurement trials (default: 5)
             * @param warmup Number of warmup iterations (default: 3)
             * @param operation_count Total operation count (for GFLOPS, default: 0)
             * @param num_processes Number of MPI processes (for efficiency, default: 1)
             * @return Performance results with timing statistics
             */
            static PerfResult benchmark(
                std::function<void()> fn,
                int trials = 5,
                int warmup = 3,
                size_t operation_count = 0,
                int num_processes = 1);

            /**
             * @brief Benchmark with baseline comparison
             *
             * @param fn Function to benchmark
             * @param baseline_result Baseline performance result for speedup calculation
             * @param trials Number of measurement trials
             * @param warmup Number of warmup iterations
             * @param operation_count Total operation count
             * @param num_processes Number of MPI processes
             * @return Performance results with speedup computed
             */
            static PerfResult benchmarkWithBaseline(
                std::function<void()> fn,
                const PerfResult &baseline_result,
                int trials = 5,
                int warmup = 3,
                size_t operation_count = 0,
                int num_processes = 1);

            /**
             * @brief Print performance results in human-readable format
             *
             * @param result Performance result to print
             * @param name Optional name/label for the benchmark
             */
            static void printResult(const PerfResult &result, const char *name = "");

            /**
             * @brief Print MPI performance profile
             *
             * @param profile MPI performance breakdown
             * @param name Optional name/label
             */
            static void printMPIProfile(const MPIProfile &profile, const char *name = "");

            /**
             * @brief Calculate GFLOPS from operation count and time
             *
             * @param operation_count Total floating-point operations
             * @param time_ms Execution time in milliseconds
             * @return GFLOPS (billions of operations per second)
             */
            static double calculateGFLOPS(size_t operation_count, double time_ms)
            {
                if (time_ms <= 0.0)
                    return 0.0;
                return (static_cast<double>(operation_count) / 1e9) / (time_ms / 1000.0);
            }

            /**
             * @brief Calculate attention FLOP count
             *
             * For GQA attention: 2 * seq_len * seq_len * head_dim per head
             * (Q·K^T requires seq_len × seq_len × head_dim, attention·V requires same)
             *
             * @param n_heads Number of query heads
             * @param seq_len Sequence length
             * @param head_dim Head dimension
             * @return Total FLOP count
             */
            static size_t calculateAttentionFLOPs(int n_heads, int seq_len, int head_dim)
            {
                // Q·K^T: [seq_len, head_dim] @ [head_dim, seq_len] = seq_len^2 * head_dim MADDs
                // Attention·V: [seq_len, seq_len] @ [seq_len, head_dim] = seq_len^2 * head_dim MADDs
                // Each MADD = 2 FLOPs (multiply + add)
                size_t flops_per_head = 2ULL * 2ULL * seq_len * seq_len * head_dim;
                return flops_per_head * n_heads;
            }

        private:
            /**
             * @brief Compute statistics from timing samples
             */
            static void computeStatistics(
                const std::vector<double> &samples,
                double &mean,
                double &min,
                double &max,
                double &stddev,
                double &cv_percent);
        };

    } // namespace test
} // namespace llaminar2

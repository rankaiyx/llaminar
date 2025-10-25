/**
 * @file PerfHarness.cpp
 * @brief Implementation of performance benchmarking harness
 * @author David Sanftenberg
 */

#include "PerfHarness.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace llaminar2
{
    namespace test
    {
        PerfResult PerfHarness::benchmark(
            std::function<void()> fn,
            int trials,
            int warmup,
            size_t operation_count,
            int num_processes)
        {
            PerfResult result;
            result.trials = trials;
            result.warmup = warmup;
            result.operation_count = operation_count;
            result.num_processes = num_processes;

            // Warmup iterations
            for (int i = 0; i < warmup; ++i)
            {
                fn();
            }

            // Measurement trials
            std::vector<double> samples;
            samples.reserve(trials);

            for (int i = 0; i < trials; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                fn();
                auto end = std::chrono::high_resolution_clock::now();

                double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                samples.push_back(elapsed_ms);
            }

            // Compute statistics
            computeStatistics(samples, result.mean_ms, result.min_ms, result.max_ms, result.stddev_ms, result.cv_percent);

            // Compute GFLOPS if operation count provided
            if (operation_count > 0)
            {
                result.gflops = calculateGFLOPS(operation_count, result.mean_ms);
            }

            return result;
        }

        PerfResult PerfHarness::benchmarkWithBaseline(
            std::function<void()> fn,
            const PerfResult &baseline_result,
            int trials,
            int warmup,
            size_t operation_count,
            int num_processes)
        {
            PerfResult result = benchmark(fn, trials, warmup, operation_count, num_processes);

            // Compute speedup vs baseline
            if (baseline_result.mean_ms > 0.0)
            {
                result.speedup = baseline_result.mean_ms / result.mean_ms;
            }

            // Compute parallel efficiency
            if (num_processes > 1 && result.speedup > 0.0)
            {
                result.efficiency = result.speedup / static_cast<double>(num_processes);
            }

            return result;
        }

        void PerfHarness::printResult(const PerfResult &result, const char *name)
        {
            std::cout << "=== Performance Result";
            if (name && name[0])
            {
                std::cout << ": " << name;
            }
            std::cout << " ===" << std::endl;

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Trials:         " << result.trials << " (warmup: " << result.warmup << ")" << std::endl;
            std::cout << "  Mean time:      " << result.mean_ms << " ms" << std::endl;
            std::cout << "  Min time:       " << result.min_ms << " ms" << std::endl;
            std::cout << "  Max time:       " << result.max_ms << " ms" << std::endl;
            std::cout << "  Std dev:        " << result.stddev_ms << " ms" << std::endl;
            std::cout << "  CV:             " << result.cv_percent << " %" << std::endl;

            if (result.gflops > 0.0)
            {
                std::cout << "  GFLOPS:         " << result.gflops << std::endl;
            }

            if (result.speedup > 0.0)
            {
                std::cout << "  Speedup:        " << result.speedup << "x" << std::endl;
            }

            if (result.efficiency > 0.0)
            {
                std::cout << "  Efficiency:     " << (result.efficiency * 100.0) << " %" << std::endl;
            }

            if (result.num_processes > 1)
            {
                std::cout << "  Processes:      " << result.num_processes << std::endl;
            }
        }

        void PerfHarness::printMPIProfile(const MPIProfile &profile, const char *name)
        {
            std::cout << "=== MPI Performance Profile";
            if (name && name[0])
            {
                std::cout << ": " << name;
            }
            std::cout << " ===" << std::endl;

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Compute time:   " << profile.compute_time_ms << " ms" << std::endl;
            std::cout << "  Allreduce time: " << profile.allreduce_time_ms << " ms" << std::endl;
            std::cout << "  Barrier time:   " << profile.barrier_time_ms << " ms" << std::endl;
            std::cout << "  Total time:     " << profile.total_time_ms << " ms" << std::endl;
            std::cout << "  Comm overhead:  " << profile.comm_overhead_percent << " %" << std::endl;
        }

        void PerfHarness::computeStatistics(
            const std::vector<double> &samples,
            double &mean,
            double &min,
            double &max,
            double &stddev,
            double &cv_percent)
        {
            if (samples.empty())
            {
                mean = min = max = stddev = cv_percent = 0.0;
                return;
            }

            // Mean
            mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

            // Min/Max
            min = *std::min_element(samples.begin(), samples.end());
            max = *std::max_element(samples.begin(), samples.end());

            // Standard deviation
            double variance = 0.0;
            for (double sample : samples)
            {
                double diff = sample - mean;
                variance += diff * diff;
            }
            variance /= samples.size();
            stddev = std::sqrt(variance);

            // Coefficient of variation
            if (mean > 0.0)
            {
                cv_percent = (stddev / mean) * 100.0;
            }
            else
            {
                cv_percent = 0.0;
            }
        }

    } // namespace test
} // namespace llaminar2

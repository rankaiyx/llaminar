/**
 * @file CPUBenchmark.h
 * @brief CPU-specific benchmark implementation
 *
 * Measures:
 * - Memory bandwidth using STREAM-like copy/scale/add operations
 * - FP32 compute using vectorized FMA loops
 * - INT8 compute using VNNI-style dot products (if available)
 *
 * Implementation uses OpenMP for multi-threaded execution to match
 * actual inference workload characteristics.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "DeviceBenchmark.h"

namespace llaminar2
{
    /**
     * @brief CPU benchmark implementation
     *
     * Memory bandwidth test:
     * - Uses first-touch NUMA initialization for accurate results
     * - Tests read, write, and copy separately
     * - Parallelized with OpenMP to saturate memory channels
     *
     * Compute test:
     * - FP32 FMA throughput (matches GEMM workloads)
     * - INT8 dot product throughput (matches quantized inference)
     * - Uses SIMD (AVX2/AVX-512) when available
     */
    class CPUBenchmark : public IDeviceBenchmark
    {
    public:
        explicit CPUBenchmark(const BenchmarkConfig &config = BenchmarkConfig());
        ~CPUBenchmark() override = default;

        DeviceBenchmarkResult run() override;
        DeviceId device() const override { return DeviceId::cpu(); }
        double estimatedDurationMs() const override;

    private:
        BenchmarkConfig config_;

        /// Measure memory read bandwidth (GB/s)
        double benchmarkMemoryRead(float *buffer, size_t num_elements);

        /// Measure memory write bandwidth (GB/s)
        double benchmarkMemoryWrite(float *buffer, size_t num_elements);

        /// Measure memory copy bandwidth (GB/s)
        double benchmarkMemoryCopy(float *src, float *dst, size_t num_elements);

        /// Measure FP32 FMA throughput (GFLOPS)
        double benchmarkComputeFP32(size_t num_ops);

        /// Measure INT8 dot product throughput (GOPS)
        double benchmarkComputeINT8(size_t num_ops);

        /// Get number of threads to use
        int getNumThreads() const;

        /// Allocate NUMA-aware buffer
        static float *allocateNUMABuffer(size_t num_elements);
        static void freeNUMABuffer(float *buffer, size_t num_elements);
    };

} // namespace llaminar2

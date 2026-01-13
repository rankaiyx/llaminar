/**
 * @file ROCmBenchmark.h
 * @brief ROCm GPU benchmark interface
 *
 * Measures:
 * - Device memory bandwidth (HBM)
 * - Host-to-device transfer rates (pinned and pageable)
 * - Device-to-host transfer rates (pinned and pageable)
 * - FP32/FP16 compute throughput
 * - Peer-to-peer transfer rates (xGMI/Infinity Fabric if available)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "DeviceBenchmark.h"
#include <vector>

namespace llaminar2
{
    /**
     * @brief ROCm/HIP GPU benchmark implementation
     *
     * All benchmarks use HIP streams for accurate timing.
     * Memory transfer tests distinguish pinned vs pageable
     * to inform staging buffer allocation decisions.
     */
    class ROCmBenchmark : public IDeviceBenchmark
    {
    public:
        /**
         * @brief Construct benchmark for a specific ROCm device
         * @param device_ordinal HIP device index (0-based)
         * @param config Benchmark configuration
         */
        explicit ROCmBenchmark(int device_ordinal, const BenchmarkConfig &config = BenchmarkConfig());
        ~ROCmBenchmark() override;

        DeviceBenchmarkResult run() override;
        DeviceId device() const override { return DeviceId::rocm(device_ordinal_); }
        double estimatedDurationMs() const override;

        /**
         * @brief Enumerate all available ROCm/HIP devices
         * @return Vector of DeviceId for each ROCm device
         *
         * This static method allows device enumeration without instantiating
         * a benchmark object, and keeps HIP runtime calls isolated to ROCm files.
         */
        static std::vector<DeviceId> enumerateDevices();

        /**
         * @brief Copy data from host to ROCm device
         * @param device_ordinal HIP device index
         * @param h_src Host source buffer
         * @param num_bytes Number of bytes to copy
         * @return Time in milliseconds, or -1 on error
         *
         * Static helper for cross-vendor transfer measurement.
         * Allocates device memory, copies, and frees.
         */
        static double copyHostToDevice(int device_ordinal, const void *h_src, size_t num_bytes);

        /**
         * @brief Copy data from ROCm device to host
         * @param device_ordinal HIP device index
         * @param h_dst Host destination buffer
         * @param num_bytes Number of bytes to copy
         * @return Time in milliseconds, or -1 on error
         *
         * Static helper for cross-vendor transfer measurement.
         * Allocates device memory, initializes, copies, and frees.
         */
        static double copyDeviceToHost(int device_ordinal, void *h_dst, size_t num_bytes);

    private:
        int device_ordinal_;
        BenchmarkConfig config_;

        /// Device memory bandwidth test
        double benchmarkDeviceMemory();

        /// Host-to-device transfer (pinned memory)
        double benchmarkH2DPinned();

        /// Host-to-device transfer (pageable memory)
        double benchmarkH2DPageable();

        /// Device-to-host transfer (pinned memory)
        double benchmarkD2HPinned();

        /// Device-to-host transfer (pageable memory)
        double benchmarkD2HPageable();

        /// FP32 compute throughput
        double benchmarkComputeFP32();

        /// FP16 compute throughput
        double benchmarkComputeFP16();

        /// Peer-to-peer transfer to another GPU
        /// Returns 0.0 if P2P not available
        double benchmarkPeerTransfer(int peer_ordinal);
    };

} // namespace llaminar2

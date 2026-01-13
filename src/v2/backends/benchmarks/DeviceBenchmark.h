/**
 * @file DeviceBenchmark.h
 * @brief Lightweight device benchmarking for real-terms placement decisions
 *
 * Provides actual measurements of:
 * - Memory bandwidth (GB/s) - sequential read/write/copy
 * - Compute throughput (GFLOPS FP32, GOPS INT8)
 * - Host-device transfer rates (GPU only)
 * - Peer-to-peer transfer rates (multi-GPU only)
 * - Cross-vendor transfer rates (CUDA ↔ ROCm via host staging)
 *
 * Design goals:
 * - Lightweight: Each benchmark completes in < 500ms
 * - Accurate: Results useful for placement decisions (not benchmarking precision)
 * - Non-disruptive: Small memory footprint (~16-64 MB per device)
 *
 * Usage:
 * @code
 * auto results = DeviceBenchmarkRunner::runAll();
 * for (const auto& [device, result] : results) {
 *     LOG_INFO("Device " << device.to_string()
 *              << ": " << result.memory_bandwidth_gbps << " GB/s memory, "
 *              << result.compute_fp32_gflops << " GFLOPS FP32");
 * }
 * @endcode
 *
 * Integration:
 * Results are used by PlacementStrategy to replace estimated values with
 * measured ones, improving placement decisions for heterogeneous clusters.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../DeviceId.h"
#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Results from benchmarking a single device
     *
     * All bandwidth values in GB/s, compute values in GFLOPS/GOPS.
     * Values of 0.0 indicate the metric was not measured or not applicable.
     */
    struct DeviceBenchmarkResult
    {
        DeviceId device;

        // =====================================================================
        // Memory Bandwidth (GB/s)
        // =====================================================================

        /// Sequential memory read bandwidth (GB/s)
        double memory_read_gbps = 0.0;

        /// Sequential memory write bandwidth (GB/s)
        double memory_write_gbps = 0.0;

        /// Memory copy bandwidth (GB/s) - most relevant for inference
        double memory_copy_gbps = 0.0;

        /// Effective bandwidth for typical inference workloads
        /// Weighted average: 0.5*copy + 0.3*read + 0.2*write
        double memory_bandwidth_gbps() const
        {
            return 0.5 * memory_copy_gbps + 0.3 * memory_read_gbps + 0.2 * memory_write_gbps;
        }

        // =====================================================================
        // Compute Throughput
        // =====================================================================

        /// FP32 compute throughput (GFLOPS) - FMA-heavy workload
        double compute_fp32_gflops = 0.0;

        /// FP16 compute throughput (GFLOPS) - if supported
        double compute_fp16_gflops = 0.0;

        /// INT8 compute throughput (GOPS) - for quantized inference
        double compute_int8_gops = 0.0;

        // =====================================================================
        // Transfer Rates (GPU only)
        // =====================================================================

        /// Host-to-device transfer rate (GB/s) using pinned memory
        double h2d_pinned_gbps = 0.0;

        /// Device-to-host transfer rate (GB/s) using pinned memory
        double d2h_pinned_gbps = 0.0;

        /// Host-to-device transfer rate (GB/s) using pageable memory
        double h2d_pageable_gbps = 0.0;

        /// Device-to-host transfer rate (GB/s) using pageable memory
        double d2h_pageable_gbps = 0.0;

        // =====================================================================
        // Peer-to-Peer Transfers (multi-GPU only)
        // =====================================================================

        /// Peer device transfer rates (indexed by peer DeviceId)
        /// Non-zero only if direct P2P is enabled between devices
        /// This includes NVLink, xGMI/Infinity Fabric, AND direct PCIe P2P
        /// Zero means transfers must stage through host memory
        std::map<DeviceId, double> peer_transfer_gbps;

        /// Check if direct P2P is available to a specific peer
        bool hasP2P(DeviceId peer) const
        {
            auto it = peer_transfer_gbps.find(peer);
            return it != peer_transfer_gbps.end() && it->second > 0.0;
        }

        /// Get P2P transfer rate to a peer device (GB/s)
        /// Returns measured P2P rate if available, otherwise 0.0
        double getP2PRate(DeviceId peer) const
        {
            auto it = peer_transfer_gbps.find(peer);
            return (it != peer_transfer_gbps.end()) ? it->second : 0.0;
        }

        /// Estimate effective GPU-to-GPU transfer rate (GB/s) when direct P2P is not available
        /// When no direct P2P exists (no NVLink/xGMI/PCIe-P2P), transfers must stage through
        /// host memory: GPU_A -> D2H -> host_buffer -> H2D -> GPU_B
        /// The bottleneck is min(D2H, H2D) since one completes before the other starts
        double estimatedHostStagedRate() const
        {
            // Use pinned memory rates as they're typically used for staging
            return std::min(d2h_pinned_gbps, h2d_pinned_gbps);
        }

        /// Get effective inter-GPU transfer rate to a peer
        /// Returns direct P2P rate if available (NVLink/xGMI/PCIe-P2P)
        /// Otherwise estimates host-staged rate (GPU->host->GPU)
        double effectiveInterGPURate(DeviceId peer, const DeviceBenchmarkResult& peer_result) const
        {
            // Check for direct P2P (any method: NVLink, xGMI, or PCIe P2P)
            if (hasP2P(peer))
            {
                return getP2PRate(peer);
            }
            // No direct P2P - must stage through host memory
            // The bottleneck is min(our_D2H, peer_H2D) for outbound transfers
            return std::min(d2h_pinned_gbps, peer_result.h2d_pinned_gbps);
        }

        // =====================================================================
        // Metadata
        // =====================================================================

        /// Total benchmark duration (milliseconds)
        double benchmark_duration_ms = 0.0;

        /// Number of iterations used for averaging
        int iterations = 0;

        /// Memory used for benchmarking (bytes)
        size_t benchmark_memory_bytes = 0;

        /// Human-readable summary
        std::string toString() const;

        /// Flag indicating benchmark completed successfully
        bool valid = false;

        /// Check if benchmark completed successfully (alias for valid)
        bool isValid() const
        {
            return valid;
        }
    };

    /**
     * @brief Interface for device-specific benchmark implementations
     */
    class IDeviceBenchmark
    {
    public:
        virtual ~IDeviceBenchmark() = default;

        /// Run all benchmarks for this device
        virtual DeviceBenchmarkResult run() = 0;

        /// Get the device being benchmarked
        virtual DeviceId device() const = 0;

        /// Get estimated benchmark duration (ms)
        virtual double estimatedDurationMs() const { return 500.0; }
    };

    /**
     * @brief Configuration for benchmark execution
     */
    struct BenchmarkConfig
    {
        /// Memory size for bandwidth tests (bytes)
        /// Default 16 MB - large enough for accurate measurement, small enough to be fast
        size_t memory_test_bytes = 16 * 1024 * 1024;

        /// Number of iterations for averaging
        int iterations = 5;

        /// Warmup iterations (not counted in results)
        int warmup_iterations = 2;

        /// Enable compute benchmarks (can be slow on CPU)
        bool benchmark_compute = true;

        /// Enable transfer benchmarks (GPU only)
        bool benchmark_transfers = true;

        /// Enable P2P benchmarks (multi-GPU only)
        bool benchmark_p2p = true;

        /// Enable cross-vendor transfer benchmarks (CUDA ↔ ROCm)
        bool benchmark_cross_vendor = true;

        /// Timeout per benchmark phase (ms)
        double timeout_ms = 2000.0;

        /// Verbose logging during benchmark
        bool verbose = false;

        /// Create config for quick benchmarks (faster but less accurate)
        static BenchmarkConfig quick()
        {
            BenchmarkConfig cfg;
            cfg.memory_test_bytes = 8 * 1024 * 1024;
            cfg.iterations = 3;
            cfg.warmup_iterations = 1;
            cfg.benchmark_p2p = false;
            cfg.benchmark_cross_vendor = false;
            return cfg;
        }

        /// Create config for thorough benchmarks (slower but more accurate)
        static BenchmarkConfig thorough()
        {
            BenchmarkConfig cfg;
            cfg.memory_test_bytes = 64 * 1024 * 1024;
            cfg.iterations = 10;
            cfg.warmup_iterations = 3;
            return cfg;
        }
    };

    /**
     * @brief Factory for creating device-specific benchmarks
     */
    class DeviceBenchmarkFactory
    {
    public:
        /// Create benchmark for a specific device
        static std::unique_ptr<IDeviceBenchmark> create(
            DeviceId device,
            const BenchmarkConfig &config = BenchmarkConfig());

        /// Enumerate all available devices in the system
        static std::vector<DeviceId> enumerateDevices();

        /// Check if benchmarking is supported for a device type
        static bool isSupported(DeviceType type);
    };

    /**
     * @brief Runner for executing benchmarks across multiple devices
     */
    class DeviceBenchmarkRunner
    {
    public:
        explicit DeviceBenchmarkRunner(const BenchmarkConfig &config = BenchmarkConfig());

        /// Run benchmarks for all discovered devices
        std::map<DeviceId, DeviceBenchmarkResult> runAll();

        /// Run benchmark for a specific device
        DeviceBenchmarkResult runSingleDevice(DeviceId device);

        /// Run benchmarks for a list of devices
        std::map<DeviceId, DeviceBenchmarkResult> runDevices(const std::vector<DeviceId> &devices);

        /// Estimate total benchmark duration for a list of devices
        double estimateTotalDuration(const std::vector<DeviceId> &devices);

        /**
         * @brief Measure cross-vendor transfer rate between two GPUs
         *
         * This measures the actual throughput when transferring between GPUs
         * of different vendors (e.g., CUDA ↔ ROCm). The transfer goes through
         * host-pinned memory since driver-level cross-vendor P2P isn't supported.
         *
         * @param src Source device
         * @param dst Destination device
         * @return Transfer rate in GB/s, or 0.0 if transfer not possible
         */
        double measureCrossVendorTransfer(DeviceId src, DeviceId dst);

        /**
         * @brief Measure all cross-vendor transfer rates
         *
         * Tests transfers between all CUDA and ROCm GPU pairs.
         * Results are stored in a map keyed by (src, dst) device pair.
         *
         * @return Map of (src, dst) -> transfer rate in GB/s
         */
        std::map<std::pair<DeviceId, DeviceId>, double> measureAllCrossVendorTransfers();

        /// Get configuration
        const BenchmarkConfig &config() const { return config_; }

    private:
        BenchmarkConfig config_;
    };

    /// Print benchmark results in formatted table
    void printBenchmarkResults(const std::map<DeviceId, DeviceBenchmarkResult> &results);

} // namespace llaminar2

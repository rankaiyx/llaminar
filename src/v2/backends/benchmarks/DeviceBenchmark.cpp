/**
 * @file DeviceBenchmark.cpp
 * @brief Device benchmark factory and runner implementation
 *
 * **IMPORTANT**: This file does NOT include GPU runtime headers directly.
 * CUDA and ROCm headers cannot coexist in the same translation unit due to
 * conflicting type definitions (float4, int4, dim3, etc.).
 *
 * Device enumeration is delegated to the backend-specific benchmark classes
 * which live in separate compilation units (.cu for CUDA, .cpp compiled as HIP for ROCm).
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "DeviceBenchmark.h"
#include "CPUBenchmark.h"
#include "utils/Logger.h"

#ifdef HAVE_CUDA
#include "CUDABenchmark.h"
#endif

#ifdef HAVE_ROCM
#include "ROCmBenchmark.h"
#endif

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace llaminar2
{

    //--------------------------------------------------------------------------
    // DeviceBenchmarkFactory Implementation
    //--------------------------------------------------------------------------

    std::unique_ptr<IDeviceBenchmark> DeviceBenchmarkFactory::create(DeviceId device,
                                                                     const BenchmarkConfig &config)
    {
        switch (device.type)
        {
        case DeviceType::CPU:
            return std::make_unique<CPUBenchmark>(config);

        case DeviceType::CUDA:
#ifdef HAVE_CUDA
            return std::make_unique<CUDABenchmark>(device.ordinal, config);
#else
            LOG_WARN("CUDA support not compiled in, cannot benchmark CUDA device");
            return nullptr;
#endif

        case DeviceType::ROCm:
#ifdef HAVE_ROCM
            return std::make_unique<ROCmBenchmark>(device.ordinal, config);
#else
            LOG_WARN("ROCm support not compiled in, cannot benchmark ROCm device");
            return nullptr;
#endif

        default:
            LOG_ERROR("Unknown device type: " << static_cast<int>(device.type));
            return nullptr;
        }
    }

    std::vector<DeviceId> DeviceBenchmarkFactory::enumerateDevices()
    {
        std::vector<DeviceId> devices;

        // CPU is always available
        devices.push_back(DeviceId::cpu());

#ifdef HAVE_CUDA
        // Enumerate CUDA devices via the CUDABenchmark static method
        // This keeps CUDA runtime calls isolated to CUDABenchmark.cu
        auto cuda_devices = CUDABenchmark::enumerateDevices();
        devices.insert(devices.end(), cuda_devices.begin(), cuda_devices.end());
#endif

#ifdef HAVE_ROCM
        // Enumerate ROCm devices via the ROCmBenchmark static method
        // This keeps HIP runtime calls isolated to ROCmBenchmark.cpp
        auto rocm_devices = ROCmBenchmark::enumerateDevices();
        devices.insert(devices.end(), rocm_devices.begin(), rocm_devices.end());
#endif

        return devices;
    }

    //--------------------------------------------------------------------------
    // DeviceBenchmarkRunner Implementation
    //--------------------------------------------------------------------------

    DeviceBenchmarkRunner::DeviceBenchmarkRunner(const BenchmarkConfig &config)
        : config_(config)
    {
    }

    std::map<DeviceId, DeviceBenchmarkResult> DeviceBenchmarkRunner::runAll()
    {
        std::vector<DeviceId> devices = DeviceBenchmarkFactory::enumerateDevices();
        return runDevices(devices);
    }

    std::map<DeviceId, DeviceBenchmarkResult> DeviceBenchmarkRunner::runDevices(
        const std::vector<DeviceId> &devices)
    {
        std::map<DeviceId, DeviceBenchmarkResult> results;

        LOG_INFO("Running device benchmarks for " << devices.size() << " device(s)");

        for (const auto &device : devices)
        {
            auto benchmark = DeviceBenchmarkFactory::create(device, config_);
            if (benchmark)
            {
                LOG_DEBUG("Benchmarking device: " << device);
                auto result = benchmark->run();
                results[device] = std::move(result);
            }
            else
            {
                LOG_WARN("Failed to create benchmark for device: " << device);
            }
        }

        return results;
    }

    DeviceBenchmarkResult DeviceBenchmarkRunner::runSingleDevice(DeviceId device)
    {
        auto benchmark = DeviceBenchmarkFactory::create(device, config_);
        if (benchmark)
        {
            return benchmark->run();
        }

        DeviceBenchmarkResult invalid;
        invalid.device = device;
        invalid.valid = false;
        return invalid;
    }

    double DeviceBenchmarkRunner::estimateTotalDuration(const std::vector<DeviceId> &devices)
    {
        double total_ms = 0.0;

        for (const auto &device : devices)
        {
            auto benchmark = DeviceBenchmarkFactory::create(device, config_);
            if (benchmark)
            {
                total_ms += benchmark->estimatedDurationMs();
            }
        }

        return total_ms;
    }

    //--------------------------------------------------------------------------
    // Utility Functions
    //--------------------------------------------------------------------------

    void printBenchmarkResults(const std::map<DeviceId, DeviceBenchmarkResult> &results)
    {
        LOG_INFO("╔══════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║                          DEVICE BENCHMARK RESULTS                            ║");
        LOG_INFO("╠══════════════════════════════════════════════════════════════════════════════╣");

        for (const auto &[device, result] : results)
        {
            if (!result.valid)
            {
                LOG_INFO("║ Device: " << device << " - BENCHMARK FAILED");
                continue;
            }

            std::string device_str;
            if (device.type == DeviceType::CPU)
            {
                device_str = "CPU";
            }
            else if (device.type == DeviceType::CUDA)
            {
                device_str = "CUDA GPU " + std::to_string(device.ordinal);
            }
            else if (device.type == DeviceType::ROCm)
            {
                device_str = "ROCm GPU " + std::to_string(device.ordinal);
            }

            LOG_INFO("║ " << device_str);
            LOG_INFO("║   Memory Bandwidth:");
            if (result.memory_read_gbps > 0)
            {
                LOG_INFO("║     Read:       " << result.memory_read_gbps << " GB/s");
            }
            if (result.memory_write_gbps > 0)
            {
                LOG_INFO("║     Write:      " << result.memory_write_gbps << " GB/s");
            }
            LOG_INFO("║     Copy:       " << result.memory_copy_gbps << " GB/s");

            LOG_INFO("║   Compute Throughput:");
            LOG_INFO("║     FP32:       " << result.compute_fp32_gflops << " GFLOPS");
            if (result.compute_fp16_gflops > 0)
            {
                LOG_INFO("║     FP16:       " << result.compute_fp16_gflops << " GFLOPS");
            }
            if (result.compute_int8_gops > 0)
            {
                LOG_INFO("║     INT8:       " << result.compute_int8_gops << " GOPS");
            }

            if (result.h2d_pinned_gbps > 0 || result.d2h_pinned_gbps > 0)
            {
                LOG_INFO("║   Host Transfer:");
                LOG_INFO("║     H2D (pin):  " << result.h2d_pinned_gbps << " GB/s");
                LOG_INFO("║     H2D (page): " << result.h2d_pageable_gbps << " GB/s");
                LOG_INFO("║     D2H (pin):  " << result.d2h_pinned_gbps << " GB/s");
                LOG_INFO("║     D2H (page): " << result.d2h_pageable_gbps << " GB/s");
            }

            if (!result.peer_transfer_gbps.empty())
            {
                LOG_INFO("║   Peer-to-Peer Transfer:");
                for (const auto &[peer, gbps] : result.peer_transfer_gbps)
                {
                    LOG_INFO("║     to GPU " << peer.ordinal << ":  " << gbps << " GB/s");
                }
            }

            LOG_INFO("║   Benchmark Time: " << result.benchmark_duration_ms << " ms");
            LOG_INFO("╠══════════════════════════════════════════════════════════════════════════════╣");
        }

        LOG_INFO("╚══════════════════════════════════════════════════════════════════════════════╝");
    }

    //--------------------------------------------------------------------------
    // Cross-Vendor Transfer Measurement
    //--------------------------------------------------------------------------

    double DeviceBenchmarkRunner::measureCrossVendorTransfer(DeviceId src, DeviceId dst)
    {
        // Only meaningful for GPU-to-GPU transfers across vendors
        if (src.type == DeviceType::CPU || dst.type == DeviceType::CPU)
        {
            return 0.0;
        }
        if (src.type == dst.type)
        {
            // Same vendor - use intra-vendor P2P instead
            return 0.0;
        }

        // Cross-vendor transfer: must stage through host memory
        // We measure the actual throughput of: src_gpu -> host -> dst_gpu

        size_t num_bytes = config_.memory_test_bytes;

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Allocate host staging buffer (use aligned malloc for portability)
        // We can't use cudaMallocHost or hipMallocHost since we need it accessible by both.
        void *h_staging = aligned_alloc(4096, num_bytes);
        if (!h_staging)
        {
            LOG_ERROR("Failed to allocate staging buffer for cross-vendor transfer");
            return 0.0;
        }

        // Initialize staging buffer
        std::memset(h_staging, 0xAB, num_bytes);

        double d2h_gbps = 0.0;
        double h2d_gbps = 0.0;

        // Step 1: D2H from source GPU
        if (src.type == DeviceType::CUDA)
        {
            d2h_gbps = CUDABenchmark::copyDeviceToHost(src.ordinal, h_staging, num_bytes);
        }
        else // ROCm
        {
            d2h_gbps = ROCmBenchmark::copyDeviceToHost(src.ordinal, h_staging, num_bytes);
        }

        // Step 2: H2D to destination GPU
        if (dst.type == DeviceType::CUDA)
        {
            h2d_gbps = CUDABenchmark::copyHostToDevice(dst.ordinal, h_staging, num_bytes);
        }
        else // ROCm
        {
            h2d_gbps = ROCmBenchmark::copyHostToDevice(dst.ordinal, h_staging, num_bytes);
        }

        free(h_staging);

        if (d2h_gbps <= 0.0 || h2d_gbps <= 0.0)
        {
            LOG_WARN("Cross-vendor transfer " << src << " -> " << dst << " failed");
            return 0.0;
        }

        // Combined transfer rate: transfers happen sequentially, so rate is harmonic mean
        // If D2H is 10 GB/s and H2D is 5 GB/s, for 1GB: 0.1s + 0.2s = 0.3s, rate = 1/0.3 = 3.33 GB/s
        // Harmonic mean: 2 / (1/d2h + 1/h2d) = 2 * d2h * h2d / (d2h + h2d)
        double effective_gbps = 2.0 * d2h_gbps * h2d_gbps / (d2h_gbps + h2d_gbps);

        LOG_INFO("Cross-vendor " << src << " -> " << dst
                 << ": D2H=" << d2h_gbps << " GB/s, H2D=" << h2d_gbps << " GB/s"
                 << ", effective=" << effective_gbps << " GB/s");

        return effective_gbps;
#else
        (void)num_bytes;
        LOG_DEBUG("Cross-vendor transfer measurement requires both HAVE_CUDA and HAVE_ROCM");
        return 0.0;
#endif
    }

    std::map<std::pair<DeviceId, DeviceId>, double> DeviceBenchmarkRunner::measureAllCrossVendorTransfers()
    {
        std::map<std::pair<DeviceId, DeviceId>, double> results;

        if (!config_.benchmark_cross_vendor)
        {
            return results;
        }

        auto devices = DeviceBenchmarkFactory::enumerateDevices();

        // Find CUDA and ROCm devices
        std::vector<DeviceId> cuda_devices;
        std::vector<DeviceId> rocm_devices;

        for (const auto &dev : devices)
        {
            if (dev.type == DeviceType::CUDA)
            {
                cuda_devices.push_back(dev);
            }
            else if (dev.type == DeviceType::ROCm)
            {
                rocm_devices.push_back(dev);
            }
        }

        if (cuda_devices.empty() || rocm_devices.empty())
        {
            LOG_DEBUG("Cross-vendor transfer test skipped: need both CUDA and ROCm devices");
            return results;
        }

        LOG_INFO("Measuring cross-vendor transfers (" << cuda_devices.size()
                 << " CUDA, " << rocm_devices.size() << " ROCm devices)");

        // Test CUDA -> ROCm
        for (const auto &cuda_dev : cuda_devices)
        {
            for (const auto &rocm_dev : rocm_devices)
            {
                double gbps = measureCrossVendorTransfer(cuda_dev, rocm_dev);
                if (gbps > 0.0)
                {
                    results[{cuda_dev, rocm_dev}] = gbps;
                    LOG_INFO("  CUDA " << cuda_dev.ordinal << " -> ROCm "
                             << rocm_dev.ordinal << ": " << gbps << " GB/s");
                }
            }
        }

        // Test ROCm -> CUDA
        for (const auto &rocm_dev : rocm_devices)
        {
            for (const auto &cuda_dev : cuda_devices)
            {
                double gbps = measureCrossVendorTransfer(rocm_dev, cuda_dev);
                if (gbps > 0.0)
                {
                    results[{rocm_dev, cuda_dev}] = gbps;
                    LOG_INFO("  ROCm " << rocm_dev.ordinal << " -> CUDA "
                             << cuda_dev.ordinal << ": " << gbps << " GB/s");
                }
            }
        }

        return results;
    }

} // namespace llaminar2

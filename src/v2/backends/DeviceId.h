/**
 * @file DeviceId.h
 * @brief Type-safe device identification for heterogeneous compute
 *
 * Replaces magic integer device indices with explicit type+ordinal.
 * - DeviceId::cpu() for CPU execution
 * - DeviceId::cuda(0) for first CUDA GPU
 * - DeviceId::rocm(1) for second ROCm GPU
 */

#pragma once

#include "DeviceType.h" // Shared DeviceType enum
#include <cassert>
#include <string>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Type-safe device identifier
     *
     * Eliminates the legacy convention where device_idx=0 meant CPU and
     * device_idx>=1 meant GPU ordinal (device_idx-1). Now explicit:
     * - DeviceId::cpu() - CPU execution
     * - DeviceId::cuda(n) - CUDA GPU n (0-indexed)
     * - DeviceId::rocm(n) - ROCm GPU n (0-indexed)
     */
    struct DeviceId
    {
        DeviceType type;
        int ordinal; // GPU ordinal (0-based), 0 for CPU

        // Factory methods for clarity
        static DeviceId cpu() { return {DeviceType::CPU, 0}; }
        static DeviceId cuda(int gpu_ordinal) { return {DeviceType::CUDA, gpu_ordinal}; }
        static DeviceId rocm(int gpu_ordinal) { return {DeviceType::ROCm, gpu_ordinal}; }

        // Predicates
        bool is_cpu() const { return type == DeviceType::CPU; }
        bool is_cuda() const { return type == DeviceType::CUDA; }
        bool is_rocm() const { return type == DeviceType::ROCm; }
        bool is_gpu() const { return type != DeviceType::CPU; }

        // Get CUDA device ordinal - asserts if not CUDA
        int cuda_ordinal() const
        {
            assert(type == DeviceType::CUDA && "cuda_ordinal() called on non-CUDA device");
            return ordinal;
        }

        // Get ROCm device ordinal - asserts if not ROCm
        int rocm_ordinal() const
        {
            assert(type == DeviceType::ROCm && "rocm_ordinal() called on non-ROCm device");
            return ordinal;
        }

        // Get GPU ordinal for any GPU type (CUDA or ROCm)
        int gpu_ordinal() const
        {
            assert(is_gpu() && "gpu_ordinal() called on CPU device");
            return ordinal;
        }

        // String representation for logging
        std::string to_string() const
        {
            switch (type)
            {
            case DeviceType::CPU:
                return "CPU";
            case DeviceType::CUDA:
                return "CUDA:" + std::to_string(ordinal);
            case DeviceType::ROCm:
                return "ROCm:" + std::to_string(ordinal);
            default:
                return "Unknown";
            }
        }

        // Equality comparison
        bool operator==(const DeviceId &other) const
        {
            return type == other.type && ordinal == other.ordinal;
        }
        bool operator!=(const DeviceId &other) const { return !(*this == other); }

        // =========================================================================
        // Legacy Conversion (for gradual migration - prefer explicit DeviceId)
        // =========================================================================

        /**
         * @brief Convert from legacy integer index
         * @param legacy_idx Legacy index (-1 or 0 = CPU, 1+ = GPU ordinal)
         * @return DeviceId
         * @deprecated Use explicit DeviceId::cpu(), DeviceId::cuda(n) instead
         * @note Historical convention in this codebase:
         *       - -1 (NOT_ON_GPU): CPU, host-only
         *       - 0: CPU (DeviceManager device 0)
         *       - 1+: GPU ordinals (DeviceManager device 1+ maps to GPU 0+)
         */
        static DeviceId fromLegacyIndex(int legacy_idx)
        {
            if (legacy_idx <= 0)
                return cpu();
            // Legacy: device_idx >= 1 means GPU ordinal (device_idx - 1)
            // Assume CUDA for legacy compatibility
            return cuda(legacy_idx - 1);
        }

        /**
         * @brief Convert to legacy integer index
         * @return Legacy index (-1=CPU, 1+=GPU device index)
         * @deprecated Avoid using legacy indices
         * @note Historical convention: -1=CPU (NOT_ON_GPU), 1=GPU:0, 2=GPU:1, etc.
         */
        int toLegacyIndex() const
        {
            if (is_cpu())
                return -1; // NOT_ON_GPU constant
            // GPU: legacy index = ordinal + 1 (GPU:0=1, GPU:1=2, etc.)
            return ordinal + 1;
        }
    };

} // namespace llaminar2

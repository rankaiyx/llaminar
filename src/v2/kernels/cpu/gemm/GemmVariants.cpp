/**
 * @file GemmVariants.cpp
 * @brief Registration wrapper for microkernel-based GEMM variants
 *
 * This file now uses the MicroKernelRegistry system with 1,225 pre-compiled
 * FP32 template instantiations plus ~400 INT8 VNNI variants across 64
 * translation units. This replaces the old runtime template generation approach.
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include "GemmVariants.h"
#include "GemmMicroKernelAdapter.h"
#include "int8/INT8PackedGemm.h"
#include "SmartGemmSearch.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{
    using llaminar::v2::kernels::IQuantizedGemmVariant;
    using llaminar::v2::kernels::ITensorGemmTileDataProvider;

    /**
     * @brief Register all GEMM variants for auto-tuning (FP32 + INT8)
     *
     * This now uses the MicroKernelRegistry which provides:
     *
     * FP32 Variants (1,225 variants):
     * - ISA: AVX512Tag, AVX2Tag
     * - MR (tile rows): {1, 2, 4, 8, 16}
     * - NR (tile cols): {1, 2, 4, 6, 8, 16, 32, 64}
     * - UNROLL_K: {1, 2, 4, 8}
     * - PREFETCH_DIST: {0, 1, 2}
     * - MC/KC/NC: Default blocking parameters
     *
     * INT8 VNNI Variants (~400 variants):
     * - ISA: AVX512VNNI
     * - MR (tile rows): {1, 2, 4, 8, 16, 32}
     * - NR (tile cols): {1, 2, 4, 6, 8, 16, 32}
     * - UNROLL_K: {1, 2, 4, 8}
     * - PREFETCH_DIST: {0, 1, 2, 3}
     * - Register constraint: MR × NR ≤ 48
     *
     * The auto-tuner will benchmark a subset and cache the optimal
     * configuration for each (m,n,k) triple and data type.
     */
    std::vector<std::unique_ptr<IQuantizedGemmVariant>> registerAllGemmVariants(
        const ITensorGemmTileDataProvider *decoder)
    {
        std::vector<std::unique_ptr<IQuantizedGemmVariant>> all_variants;

        // 1. Register FP32 variants from MicroKernelRegistry
        auto fp32_variants = kernels::gemm::registerMicroKernelVariants(decoder);
        LOG_DEBUG("Registered " << fp32_variants.size() << " FP32 micro-kernel variants");

        // 2. Register INT8 VNNI variants (if supported)
        if (kernels::gemm::isINT8GemmSupported())
        {
            auto int8_variants = kernels::gemm::registerINT8MicroKernelVariants();
            LOG_DEBUG("Registered " << int8_variants.size() << " INT8 VNNI micro-kernel variants");

            // Merge INT8 variants into FP32 variants
            all_variants.reserve(fp32_variants.size() + int8_variants.size());
            for (auto &v : fp32_variants)
            {
                all_variants.push_back(std::move(v));
            }
            for (auto &v : int8_variants)
            {
                all_variants.push_back(std::move(v));
            }

            LOG_INFO("Total GEMM variants: " << all_variants.size()
                                             << " (" << fp32_variants.size() << " FP32 + "
                                             << int8_variants.size() << " INT8)");
        }
        else
        {
            LOG_WARN("INT8 VNNI not supported on this CPU, using FP32 variants only");
            all_variants = std::move(fp32_variants);
        }

        return all_variants;
    }

} // namespace llaminar2

/**
 * @file FP16PackedGemm.h
 * @brief FP16×FP16→FP32 GEMM adapter with AVX512F/F16C backend
 *
 * This file provides a high-level interface for FP16 matrix multiplication
 * using hardware FP16 conversion instructions, with automatic packing.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../../tensors/TensorKernels.h"
#include <memory>
#include <vector>

// Forward declaration for IQuantizedGemmVariant
namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            class IQuantizedGemmVariant;
        }
    }
}

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Factory function to create FP16 GEMM kernel
             *
             * Creates an optimized FP16×FP16→FP32 GEMM kernel using AVX512F or F16C.
             * The kernel handles:
             * - Automatic packing of A and B matrices
             * - FP16 input with FP32 accumulation (numerical stability)
             * - Hardware FP16→FP32 conversion (vcvtph2ps)
             * - Cache blocking for large matrices
             *
             * @param A Tensor containing FP16 matrix A (if available for metadata)
             * @param B Tensor containing FP16 matrix B (if available for metadata)
             * @return Unique pointer to ITensorGemm kernel, or nullptr if FP16 unsupported
             */
            std::unique_ptr<ITensorGemm> createFP16PackedGemm(
                const TensorBase *A = nullptr,
                const TensorBase *B = nullptr);

            /**
             * @brief Check if FP16 GEMM is supported on this CPU
             *
             * Requires one of:
             * - AVX512F (Ice Lake+): Native FP16 conversion with 512-bit vectors
             * - AVX2 + F16C (Haswell+): FP16 conversion with 256-bit vectors
             *
             * @return true if FP16 GEMM is available, false otherwise
             */
            bool isFP16GemmSupported();

            /**
             * @brief Register all FP16 micro-kernel variants for auto-tuning
             *
             * Creates IQuantizedGemmVariant adapters for all FP16 micro-kernel configurations.
             * These variants can be used by the auto-tuner to find optimal tile/unroll/prefetch
             * parameters for FP16×FP16→FP32 GEMM operations.
             *
             * The variants use:
             * - ISA: AVX512F or AVX2+F16C (hardware FP16 conversion)
             * - MR (tile rows): {1, 2, 4, 8, 16, 32}
             * - NR (tile cols): {1, 2, 4, 6, 8, 16, 32}
             * - UNROLL_K: {1, 2, 4, 8}
             * - PREFETCH_DIST: {0, 1, 2, 3}
             * - Register constraint: MR × NR ≤ 32 (FP32 accumulator register file)
             *
             * Performance expectations:
             * - AVX512F: 16 FP16→FP32 conversions per cycle, ~2-3× faster than FP32 GEMM
             * - AVX2+F16C: 8 FP16→FP32 conversions per cycle, ~1.5-2× faster than FP32 GEMM
             * - Scalar: Portable fallback, ~0.5× FP32 speed (conversion overhead)
             *
             * @return Vector of FP16 variant adapters (typically ~400 variants)
             */
            std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>>
            registerFP16MicroKernelVariants();

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2

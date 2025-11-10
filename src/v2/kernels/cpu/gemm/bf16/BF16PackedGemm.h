/**
 * @file BF16PackedGemm.h
 * @brief BF16×BF16→FP32 GEMM using micro-kernel infrastructure (Phase 1: Standalone)
 *
 * **Phase 1 Strategy**: BF16→FP32 conversion during packing, use FP32 micro-kernels
 *
 * Architecture (Standalone - No Auto-Tuner):
 * - Uses hardcoded 8×6 AVX512 micro-kernel (good general-purpose choice)
 * - BF16→FP32 SIMD conversion during pack_A_panel / pack_B_panel
 * - Reuses existing FP32 MicroKernelTemplate infrastructure
 * - Stores B_tensor as member (no global registration complexity)
 *
 * Future (Phase 2): Add auto-tuner integration for optimal variant selection
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "../../../../tensors/Tensors.h"
#include "../../../../tensors/TensorKernels.h"
#include <memory>

namespace llaminar2
{

    /**
     * @brief Factory function to create BF16×BF16→FP32 GEMM kernel
     *
     * Phase 1: Standalone implementation with hardcoded micro-kernel
     *
     * @param A_tensor BF16 activation tensor (not used in Phase 1, for API compatibility)
     * @param B_tensor BF16 weight tensor (stored in kernel)
     * @return Auto-tuned BF16 GEMM kernel
     *
     * Usage:
     * @code
     *   auto kernel = createBF16PackedGemm(A_bf16, B_bf16);
     *   kernel->multiply(A_data, C_data, m, n, k, false, 1.0f, 0.0f, nullptr, -1);
     * @endcode
     */
    std::unique_ptr<ITensorGemm> createBF16PackedGemm(
        const BF16Tensor *A_tensor,
        const BF16Tensor *B_tensor);

} // namespace llaminar2

#pragma once

#include "../../../../tensors/TensorKernels.h"
#include "../../../../tensors/Tensors.h"
#include <memory>

namespace llaminar2
{

    /**
     * @brief Factory function to create BF16×BF16→FP32 GEMM kernel
     *
     * Creates an auto-tuned GEMM kernel that:
     * 1. Packs BF16 matrices A and B with SIMD BF16→FP32 conversion
     * 2. Uses optimal FP32 micro-kernel variant (selected by GemmAutoTuner)
     * 3. Outputs FP32 result matrix C
     *
     * Usage:
     * ```cpp
     * auto bf16_A = std::make_unique<BF16Tensor>(m, k);
     * auto bf16_B = std::make_unique<BF16Tensor>(k, n);  // or [n, k] if transposed
     * auto kernel = createBF16PackedGemm(bf16_A.get(), bf16_B.get());
     *
     * std::vector<float> C(m * n);
     * kernel->multiply(bf16_A->bf16_data(), C.data(), m, n, k,
     *                  transpose_B, alpha, beta);
     * ```
     *
     * @param A_tensor BF16 activation matrix (or nullptr to use B tensor for weights-only mode)
     * @param B_tensor BF16 weight matrix
     * @return Auto-tuned BF16×BF16→FP32 GEMM kernel
     *
     * @note First call auto-tunes for shape (m,n,k), subsequent calls use cached variant
     * @note Conversion overhead typically 10-20% of total time, amortized over micro-kernel
     */
    std::unique_ptr<ITensorGemm> createBF16PackedGemm(
        const BF16Tensor *A_tensor,
        const BF16Tensor *B_tensor);

} // namespace llaminar2

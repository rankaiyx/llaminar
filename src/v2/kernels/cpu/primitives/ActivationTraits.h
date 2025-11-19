/**
 * @file ActivationTraits.h
 * @brief Precision-specific traits for activation tensor operations
 * @author David Sanftenberg
 *
 * Provides compile-time traits for CpuAttentionKernelT<TensorType> to dispatch
 * precision-specific operations (softmax, GEMM, workspace allocation).
 *
 * Pattern:
 *   template<typename TensorType>
 *   struct ActivationTraits {
 *       using ElementType = typename TensorType::value_type;
 *       static void apply_softmax(...);
 *       static std::unique_ptr<ITensorGemm> create_activation_gemm();
 *       static std::shared_ptr<TensorBase> allocate_workspace(...);
 *   };
 *
 * Specializations: FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor
 */
#pragma once

#include <memory>
#include <vector>
#include <cstring>

#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorKernels.h"
#include "v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h"

namespace llaminar2::primitives
{

    // ============================================================================
    // Primary Template (No Implementation - Forces Specialization)
    // ============================================================================

    /**
     * @brief Primary template for ActivationTraits (must be specialized)
     *
     * This template has no implementation. All usage must be via explicit
     * specializations for FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor.
     */
    template <typename TensorType>
    struct ActivationTraits
    {
        using ElementType = typename TensorType::value_type;

        // No default implementations - forces compile-time error if used without specialization
        static void apply_softmax(ElementType *scores, int rows, int cols, bool causal, float scale) = delete;
        static std::unique_ptr<ITensorGemm> create_activation_gemm() = delete;
        static std::shared_ptr<TensorBase> allocate_workspace(const std::vector<size_t> &shape) = delete;
    };

    // ============================================================================
    // FP32Tensor Specialization
    // ============================================================================

    template <>
    struct ActivationTraits<FP32Tensor>
    {
        using ElementType = float;

        /**
         * @brief Apply softmax to FP32 activation scores
         *
         * Uses native FP32 softmax primitives (scalar/AVX2/AVX512).
         */
        static void apply_softmax(float *scores, int rows, int cols, bool causal, float scale)
        {
            softmax_row_major_fp32(scores, rows, cols, causal, scale, true); // parallel=true
        }

        /**
         * @brief Create FP32 GEMM kernel for activations
         *
         * Returns FP32 GEMM kernel (CPU baseline).
         */
        static std::unique_ptr<ITensorGemm> create_activation_gemm()
        {
            // For now, return FP32Tensor's createGemm()
            // In future, could return optimized kernel based on operation size
            FP32Tensor dummy({1, 1});
            return dummy.createGemm();
        }

        /**
         * @brief Allocate FP32 workspace tensor
         */
        static std::shared_ptr<TensorBase> allocate_workspace(const std::vector<size_t> &shape)
        {
            return std::make_shared<FP32Tensor>(shape);
        }
    };

    // ============================================================================
    // BF16Tensor Specialization
    // ============================================================================

    template <>
    struct ActivationTraits<BF16Tensor>
    {
        using ElementType = uint16_t;

        /**
         * @brief Apply softmax to BF16 activation scores
         *
         * Uses BF16 softmax primitives (scalar/AVX2/AVX512).
         * Converts BF16↔FP32 for exp/div operations.
         */
        static void apply_softmax(uint16_t *scores, int rows, int cols, bool causal, float scale)
        {
            softmax_row_major_bf16(scores, rows, cols, causal, scale, true); // parallel=true
        }

        /**
         * @brief Create BF16 GEMM kernel for activations
         *
         * Returns BF16 GEMM kernel (OneDNN bf16bf16f32 or FP32 fallback).
         */
        static std::unique_ptr<ITensorGemm> create_activation_gemm()
        {
            BF16Tensor dummy({1, 1});
            return dummy.createGemm();
        }

        /**
         * @brief Allocate workspace tensor for BF16 activations
         *
         * NOTE: Returns FP32 workspace, not BF16!
         * Rationale: BF16Tensor is immutable (use from_fp32 to update).
         * Intermediate attention computations need mutable workspaces.
         * GEMM kernels handle BF16 input → FP32 output internally.
         */
        static std::shared_ptr<TensorBase> allocate_workspace(const std::vector<size_t> &shape)
        {
            return std::make_shared<FP32Tensor>(shape); // FP32 workspace, not BF16!
        }
    };

    // ============================================================================
    // FP16Tensor Specialization
    // ============================================================================

    template <>
    struct ActivationTraits<FP16Tensor>
    {
        using ElementType = uint16_t;

        /**
         * @brief Apply softmax to FP16 activation scores
         *
         * Uses FP16 softmax primitives (scalar/AVX2 with F16C/AVX512).
         * Converts FP16↔FP32 for exp/div operations.
         */
        static void apply_softmax(uint16_t *scores, int rows, int cols, bool causal, float scale)
        {
            softmax_row_major_fp16(scores, rows, cols, causal, scale, true); // parallel=true
        }

        /**
         * @brief Create FP16 GEMM kernel for activations
         *
         * Returns FP16 GEMM kernel (OneDNN fp16fp16f32 or FP32 fallback).
         */
        static std::unique_ptr<ITensorGemm> create_activation_gemm()
        {
            FP16Tensor dummy({1, 1});
            return dummy.createGemm();
        }

        /**
         * @brief Allocate workspace tensor for FP16 activations
         *
         * NOTE: Returns FP32 workspace, not FP16!
         * Rationale: FP16Tensor is immutable (use from_fp32 to update).
         * Intermediate attention computations need mutable workspaces.
         * GEMM kernels handle FP16 input → FP32 output internally.
         */
        static std::shared_ptr<TensorBase> allocate_workspace(const std::vector<size_t> &shape)
        {
            return std::make_shared<FP32Tensor>(shape); // FP32 workspace, not FP16!
        }
    };

    // ============================================================================
    // INT32Tensor Specialization
    // ============================================================================

    template <>
    struct ActivationTraits<INT32Tensor>
    {
        using ElementType = int32_t;

        /**
         * @brief Apply softmax to INT32 activation scores
         *
         * Converts INT32→FP32, applies FP32 softmax, converts FP32→INT32.
         * This is acceptable because:
         * 1. Softmax requires floating-point exp() and division
         * 2. Softmax is not compute-bound (conversion overhead acceptable)
         * 3. Proven pattern (RMSNorm does the same)
         */
        static void apply_softmax(int32_t *scores, int rows, int cols, bool causal, float scale)
        {
            const int total = rows * cols;

            // Allocate FP32 workspace
            std::vector<float> fp32_scores(total);

            // Convert INT32 → FP32
            for (int i = 0; i < total; ++i)
            {
                fp32_scores[i] = static_cast<float>(scores[i]);
            }

            // Apply FP32 softmax (tested primitives)
            softmax_row_major_fp32(fp32_scores.data(), rows, cols, causal, scale, true);

            // Convert FP32 → INT32 (round to nearest)
            for (int i = 0; i < total; ++i)
            {
                // Softmax outputs are probabilities [0,1]
                // Scale to INT32 range for better precision
                // Assume scores will be used in INT32×INT32 accumulation
                // Scale factor TBD based on quantization scheme
                // For now, just convert directly (loses precision but safe)
                scores[i] = static_cast<int32_t>(std::round(fp32_scores[i]));
            }

            // NOTE: This conversion strategy may need refinement based on
            // the quantization scheme used in the INT8 pipeline.
            // Softmax probabilities [0,1] converted to INT32 [0,1] lose precision.
            // May need to scale probabilities to larger range (e.g., [0, 2^16])
            // for better accumulation accuracy.
        }

        /**
         * @brief Create INT32 GEMM kernel for activations
         *
         * Returns INT32 GEMM kernel (OneDNN s8s8s32 or INT8×INT8→INT32 accumulation).
         */
        static std::unique_ptr<ITensorGemm> create_activation_gemm()
        {
            INT32Tensor dummy({1, 1});
            return dummy.createGemm();
        }

        /**
         * @brief Allocate INT32 workspace tensor
         */
        static std::shared_ptr<TensorBase> allocate_workspace(const std::vector<size_t> &shape)
        {
            return std::make_shared<INT32Tensor>(shape);
        }
    };

    // ============================================================================
    // Q8_1Tensor Specialization
    // ============================================================================

    template <>
    struct ActivationTraits<llaminar2::Q8_1Tensor>
    {
        using ElementType = llaminar2::Q8_1Block;

        /**
         * @brief Apply softmax to Q8_1 activation scores
         *
         * Q8_1 activations must be dequantized to FP32 for softmax.
         */
        static void apply_softmax(llaminar2::Q8_1Block *scores, int rows, int cols, bool causal, float scale)
        {
            throw std::runtime_error("Q8_1 softmax requires FP32 workspace - must dequantize first!");
        }

        /**
         * @brief Create Q8_1 GEMM kernel for activations
         */
        static std::unique_ptr<ITensorGemm> create_activation_gemm()
        {
            // Return dummy or appropriate kernel
            llaminar2::Q8_1Tensor dummy({1, 1}, std::vector<uint8_t>(sizeof(llaminar2::Q8_1Block)));
            return dummy.createGemm();
        }

        /**
         * @brief Allocate Q8_1 workspace tensor
         */
        static std::shared_ptr<TensorBase> allocate_workspace(const std::vector<size_t> &shape)
        {
            size_t n_elems = 1;
            for (auto dim : shape)
            {
                n_elems *= dim;
            }
            size_t n_blocks = (n_elems + 31) / 32;
            size_t n_bytes = n_blocks * sizeof(llaminar2::Q8_1Block);

            std::vector<uint8_t> raw_data(n_bytes, 0);
            return std::make_shared<llaminar2::Q8_1Tensor>(shape, raw_data);
        }
    };

    // ============================================================================
    // Helper: Detect Tensor Type from Pointer (Optional Utility)
    // ============================================================================

    /**
     * @brief Type trait to detect TensorType from ElementType pointer
     *
     * Usage: typename TensorTypeFromElement<float>::type → FP32Tensor
     */
    template <typename ElementType>
    struct TensorTypeFromElement;

    template <>
    struct TensorTypeFromElement<float>
    {
        using type = FP32Tensor;
    };

    // NOTE: No specialization for uint16_t because it's ambiguous (BF16 vs FP16).
    // If you try to use TensorTypeFromElement<uint16_t>, you'll get a compile error.
    // Use explicit TensorType (BF16Tensor or FP16Tensor) instead.

    template <>
    struct TensorTypeFromElement<int32_t>
    {
        using type = INT32Tensor;
    };

} // namespace llaminar2::primitives

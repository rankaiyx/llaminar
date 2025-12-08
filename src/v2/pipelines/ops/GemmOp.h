/**
 * @file GemmOp.h
 * @brief Typed self-validating GEMM (matrix multiplication) operation
 *
 * Template-based operation supporting all activation precisions (FP32, BF16, FP16, Q8_1).
 * Uses compile-time dispatch via ActivationPrecision enum for zero-overhead precision handling.
 *
 * Encapsulates the full weight GEMM workflow:
 * 1. Validate activation/weight/output tensors
 * 2. Get/create GEMM kernel from weight tensor (cached via KernelFactory)
 * 3. Execute kernel with precision-appropriate paths
 * 4. Capture snapshot (if enabled)
 *
 * Supports both:
 * - Weight GEMM: C = A @ W^T (activations × weight tensor)
 * - Activation GEMM: C = A @ B^T (activations × activations)
 *
 * Usage:
 * @code
 * // Create typed op at initialization (based on config.activation_precision)
 * auto gemm = createGemmOp(ActivationPrecision::Q8_1);
 *
 * // Weight projection (polymorphic call, type-specific implementation)
 * TRY_OP(gemm->execute(hidden, layer.wq.get(), Q_buf, seq_len, n_heads * head_dim, d_model,
 *                      nullptr, device));
 *
 * // Legacy operator() syntax still works for backward compatibility
 * GemmOp legacy_gemm;
 * TRY_OP(legacy_gemm(hidden, layer.wq.get(), Q_buf, seq_len, q_dim, k_dim, nullptr, mpi, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "../PipelineConfig.h"
#include "../../kernels/KernelFactory.h"
#include "../../tensors/Tensors.h"
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // Precision-to-Tensor Type Mapping (shared across ops)
    // =========================================================================

    namespace detail
    {
        template <ActivationPrecision P>
        struct PrecisionTensor;

        template <>
        struct PrecisionTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
            using ElementType = float;
        };

        template <>
        struct PrecisionTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
            using ElementType = uint16_t;
        };

        template <>
        struct PrecisionTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
            using ElementType = uint16_t;
        };

        template <>
        struct PrecisionTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
            using ElementType = Q8_1Block;
        };
    } // namespace detail

    // =========================================================================
    // IGemmOp Interface
    // =========================================================================

    /**
     * @brief Interface for typed GEMM operations
     *
     * Provides polymorphic access to precision-specific GEMM implementations.
     * Use createGemmOp() factory to create instances at runtime.
     */
    class IGemmOp
    {
    public:
        virtual ~IGemmOp() = default;

        /**
         * @brief Get the activation precision this op was created for
         */
        virtual ActivationPrecision precision() const = 0;

        /**
         * @brief Execute weight projection: C = A @ W^T
         *
         * @param A Input activations tensor [m, k]
         * @param W Weight tensor [n, k] (transposed storage)
         * @param C Output tensor [m, n]
         * @param m Number of rows (sequence length)
         * @param n Number of output features
         * @param k Number of input features
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         * @param alpha Scale factor for A@W (default: 1.0f)
         * @param beta Scale factor for existing C (default: 0.0f)
         *
         * @return true on success, false on validation or execution failure
         */
        virtual bool execute(
            const TensorBase *A,
            TensorBase *W,
            TensorBase *C,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) = 0;

        /**
         * @brief Execute activation-activation matmul: C = A @ B^T (or A @ B)
         *
         * @param A Left activation tensor [m, k]
         * @param B Right activation tensor [n, k] if transpose_B, [k, n] otherwise
         * @param C Output tensor [m, n]
         * @param m Number of rows in A
         * @param n Number of rows in B (if transpose_B) or columns in B
         * @param k Inner dimension
         * @param transpose_B Whether to transpose B (true for Q@K^T)
         * @param alpha Scale factor (e.g., 1/sqrt(d_k) for attention)
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        virtual bool activations(
            TensorBase *A,
            TensorBase *B,
            TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f,
            float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute weight projection with raw float pointers
         *
         * Alternative for when activation data is not in a TensorBase.
         * Note: Only supports FP32 data.
         */
        virtual bool execute_raw(
            const float *A_data,
            TensorBase *W,
            float *C_data,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) = 0;
    };

    // =========================================================================
    // GemmOpTyped<Precision> Template Implementation
    // =========================================================================

    /**
     * @brief Typed GEMM operation with compile-time precision dispatch
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     */
    template <ActivationPrecision Precision>
    class GemmOpTyped : public IGemmOp, public OpBase
    {
    public:
        using TensorT = typename detail::PrecisionTensor<Precision>::Type;
        using ElementType = typename detail::PrecisionTensor<Precision>::ElementType;

        const char *name() const override { return "GemmOpTyped"; }
        ActivationPrecision precision() const override { return Precision; }

        bool execute(
            const TensorBase *A,
            TensorBase *W,
            TensorBase *C,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) override
        {
            // 1. Validate inputs
            if (!validateTensor(A, "activation (A)"))
                return false;
            if (!validateTensor(W, "weight (W)"))
                return false;
            if (!validateTensor(C, "output (C)"))
                return false;
            if (!validateDimensions(m, k, "input A"))
                return false;
            if (!validateDimensions(m, n, "output C"))
                return false;

            // 2. Get cached kernel from KernelFactory
            auto *gemm_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(W);
            if (!gemm_kernel)
            {
                logError("failed to get GEMM kernel from weight tensor");
                return false;
            }

            // 3. Execute based on precision
            if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                // Q8_1 path: Use native Q8_1 activation inputs, output to Q8_1
                auto *A_q8 = dynamic_cast<const Q8_1Tensor *>(A);
                auto *C_q8 = dynamic_cast<Q8_1Tensor *>(C);

                if (A_q8 && C_q8)
                {
                    // Both A and C are Q8_1: use native Q8_1 output
                    if (!gemm_kernel->multiply_to_q8_1(
                            A->data(), // Dequantized FP32 for computation
                            C_q8->mutable_q8_1_blocks(),
                            m, n, k,
                            mpi_ctx, device_idx))
                    {
                        logError("Q8_1 GEMM kernel execution failed");
                        return false;
                    }
                }
                else if (C_q8)
                {
                    // A is FP32 (e.g., from attention), C is Q8_1: use FP32 input, Q8_1 output
                    if (!gemm_kernel->multiply_to_q8_1(
                            A->data(), // Already FP32
                            C_q8->mutable_q8_1_blocks(),
                            m, n, k,
                            mpi_ctx, device_idx))
                    {
                        logError("Q8_1 GEMM kernel (FP32->Q8_1) execution failed");
                        return false;
                    }
                }
                else if (A_q8)
                {
                    // A is Q8_1, C is FP32: dequantize A, output FP32
                    if (!gemm_kernel->multiply_activations(
                            A->data(), // Dequantized FP32 for computation
                            nullptr,   // B is packed in kernel
                            C->mutable_data(),
                            m, n, k,
                            true, // transpose_B
                            alpha, beta,
                            mpi_ctx,
                            device_idx))
                    {
                        logError("GEMM kernel (Q8_1->FP32) execution failed");
                        return false;
                    }
                }
                else
                {
                    // Both A and C are FP32: standard FP32 path (shouldn't happen in Q8_1 mode, but handle gracefully)
                    if (!gemm_kernel->multiply_activations(
                            A->data(),
                            nullptr, // B is packed in kernel
                            C->mutable_data(),
                            m, n, k,
                            true, // transpose_B
                            alpha, beta,
                            mpi_ctx,
                            device_idx))
                    {
                        logError("FP32 GEMM kernel execution failed");
                        return false;
                    }
                }
            }
            else
            {
                // FP32/BF16/FP16 path: Standard GEMM with type conversion via data()
                if (!gemm_kernel->multiply_activations(
                        A->data(),
                        nullptr, // B is packed in kernel
                        C->mutable_data(),
                        m, n, k,
                        true, // transpose_B
                        alpha, beta,
                        mpi_ctx,
                        device_idx))
                {
                    logError("GEMM kernel execution failed");
                    return false;
                }
            }

            return true;
        }

        bool activations(
            TensorBase *A,
            TensorBase *B,
            TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f,
            float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validateTensor(A, "activation A"))
                return false;
            if (!validateTensor(B, "activation B"))
                return false;
            if (!validateTensor(C, "output C"))
                return false;
            if (!validateDimensions(m, k, "input A"))
                return false;

            // 2. Create kernel from A
            auto gemm_kernel = A->createGemm();
            if (!gemm_kernel)
            {
                logError("failed to create GEMM kernel for activation matmul");
                return false;
            }

            // 3. Execute based on precision
            if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                // Q8_1 activation × activation path
                // Use native Q8_1 paths when available
                auto *A_q8 = dynamic_cast<Q8_1Tensor *>(A);
                auto *B_q8 = dynamic_cast<Q8_1Tensor *>(B);
                auto *C_q8 = dynamic_cast<Q8_1Tensor *>(C);

                if (A_q8 && B_q8 && C_q8)
                {
                    // Native Q8_1 path available
                    if (!gemm_kernel->multiply_activations(
                            A->data(), // Dequant for now (TODO: native Q8_1 × Q8_1)
                            B->data(),
                            C->mutable_data(),
                            m, n, k,
                            transpose_B,
                            alpha, beta,
                            mpi_ctx,
                            device_idx))
                    {
                        logError("Q8_1 activation GEMM execution failed");
                        return false;
                    }

                    // Requantize output
                    C_q8->quantize_from_cache();
                    return true;
                }
            }

            // FP32/BF16/FP16 path: Standard activation GEMM
            if (!gemm_kernel->multiply_activations(
                    A->data(),
                    B->data(),
                    C->mutable_data(),
                    m, n, k,
                    transpose_B,
                    alpha, beta,
                    mpi_ctx,
                    device_idx))
            {
                logError("activation GEMM execution failed");
                return false;
            }

            return true;
        }

        bool execute_raw(
            const float *A_data,
            TensorBase *W,
            float *C_data,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) override
        {
            // 1. Validate inputs
            if (!validatePointer(A_data, "activation data (A)"))
                return false;
            if (!validateTensor(W, "weight (W)"))
                return false;
            if (!validatePointer(C_data, "output data (C)"))
                return false;

            // 2. Get cached kernel from KernelFactory
            auto *gemm_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(W);
            if (!gemm_kernel)
            {
                logError("failed to get GEMM kernel from weight tensor");
                return false;
            }

            // 3. Execute (raw always uses FP32)
            if (!gemm_kernel->multiply_activations(
                    A_data,
                    nullptr,
                    C_data,
                    m, n, k,
                    true,
                    alpha, beta,
                    mpi_ctx,
                    device_idx))
            {
                logError("GEMM kernel execution failed");
                return false;
            }

            return true;
        }
    };

    // =========================================================================
    // Legacy GemmOp (backward compatibility)
    // =========================================================================

    /**
     * @brief Legacy untyped GEMM operation (backward compatibility)
     *
     * Delegates to GemmOpTyped<FP32> for runtime type dispatch.
     * Prefer using createGemmOp() for new code.
     */
    class GemmOp : public OpBase
    {
    public:
        const char *name() const override { return "GemmOp"; }

        bool operator()(
            const TensorBase *A,
            TensorBase *W,
            TensorBase *C,
            int m, int n, int k,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f)
        {
            (void)snapshot_key; // Handled by pipeline
            return impl_.execute(A, W, C, m, n, k, mpi_ctx, device_idx, alpha, beta);
        }

        bool activations(
            TensorBase *A,
            TensorBase *B,
            TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f,
            float beta = 0.0f,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)snapshot_key; // Handled by pipeline
            return impl_.activations(A, B, C, m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
        }

        bool operator()(
            const float *A_data,
            TensorBase *W,
            float *C_data,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f)
        {
            return impl_.execute_raw(A_data, W, C_data, m, n, k, mpi_ctx, device_idx, alpha, beta);
        }

    private:
        GemmOpTyped<ActivationPrecision::FP32> impl_;
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create typed GEMM op for given precision
     *
     * @param precision Activation precision
     * @return Unique pointer to precision-specific GEMM op
     */
    inline std::unique_ptr<IGemmOp> createGemmOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::Q8_1:
            return std::make_unique<GemmOpTyped<ActivationPrecision::Q8_1>>();
        case ActivationPrecision::BF16:
            return std::make_unique<GemmOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<GemmOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::FP32:
        default:
            return std::make_unique<GemmOpTyped<ActivationPrecision::FP32>>();
        }
    }

} // namespace llaminar2

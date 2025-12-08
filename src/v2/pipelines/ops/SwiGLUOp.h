/**
 * @file SwiGLUOp.h
 * @brief Typed self-validating SwiGLU activation operation
 *
 * Template-based operation supporting all activation precisions (FP32, BF16, FP16, Q8_1).
 * Uses compile-time dispatch via ActivationPrecision enum for zero-overhead precision handling.
 *
 * SwiGLU is used in FFN blocks: output = silu(gate) * up
 * Where silu(x) = x * sigmoid(x)
 *
 * Per HuggingFace: down_proj(act_fn(gate_proj(x)) * up_proj(x))
 *
 * Encapsulates the full SwiGLU workflow:
 * 1. Validate gate/up/output tensors
 * 2. Create SwiGLU kernel from activation tensor
 * 3. Execute kernel with precision-appropriate paths
 * 4. Capture snapshot (if enabled)
 *
 * Usage:
 * @code
 * // Create typed op at initialization
 * auto swiglu = createSwiGLUOp(ActivationPrecision::Q8_1);
 *
 * // Execute (polymorphic call, type-specific implementation)
 * TRY_OP(swiglu->execute(gate, up, output, rows, cols, nullptr, device));
 *
 * // Legacy operator() syntax still works
 * SwiGLUOp legacy;
 * TRY_OP(legacy(gate, up, output, rows, cols, nullptr, mpi, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "GemmOp.h" // For detail::PrecisionTensor
#include "../PipelineConfig.h"
#include "../../kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "../../tensors/Tensors.h"
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // ISwiGLUOp Interface
    // =========================================================================

    /**
     * @brief Interface for typed SwiGLU operations
     *
     * Provides polymorphic access to precision-specific SwiGLU implementations.
     * Use createSwiGLUOp() factory to create instances at runtime.
     */
    class ISwiGLUOp
    {
    public:
        virtual ~ISwiGLUOp() = default;

        /**
         * @brief Get the activation precision this op was created for
         */
        virtual ActivationPrecision precision() const = 0;

        /**
         * @brief Execute SwiGLU activation: output = silu(gate) * up
         *
         * @param gate Gate tensor [rows, cols] - gets silu activation
         * @param up Up tensor [rows, cols] - linear term
         * @param output Output tensor [rows, cols] - can be same as up for in-place
         * @param rows Number of rows (sequence length)
         * @param cols Number of columns (FFN intermediate size)
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        virtual bool execute(
            TensorBase *gate,
            TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute SwiGLU with raw float pointers
         *
         * Note: Only supports FP32 data.
         */
        virtual bool execute_raw(
            const float *gate_data,
            const float *up_data,
            float *output_data,
            int rows, int cols,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    // =========================================================================
    // SwiGLUOpTyped<Precision> Template Implementation
    // =========================================================================

    /**
     * @brief Typed SwiGLU operation with compile-time precision dispatch
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     */
    template <ActivationPrecision Precision>
    class SwiGLUOpTyped : public ISwiGLUOp, public OpBase
    {
    public:
        using TensorT = typename detail::PrecisionTensor<Precision>::Type;
        using ElementType = typename detail::PrecisionTensor<Precision>::ElementType;

        const char *name() const override { return "SwiGLUOpTyped"; }
        ActivationPrecision precision() const override { return Precision; }

        bool execute(
            TensorBase *gate,
            TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validateTensor(gate, "gate"))
                return false;
            if (!validateTensor(up, "up"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "SwiGLU"))
                return false;

            // 2. Create kernel from output tensor
            auto *activation = dynamic_cast<IActivationTensor *>(output);
            if (!activation)
            {
                logError("output tensor must be IActivationTensor");
                return false;
            }

            auto kernel = activation->createSwiGLU();
            if (!kernel)
            {
                logError("failed to create SwiGLU kernel");
                return false;
            }

            // 3. Execute based on precision
            bool success = false;

            if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                // Q8_1: Native Q8_1 path
                auto *q8_gate = dynamic_cast<Q8_1Tensor *>(gate);
                auto *q8_up = dynamic_cast<Q8_1Tensor *>(up);
                auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);

                if (!q8_gate || !q8_up || !q8_output)
                {
                    logError("Q8_1 SwiGLU requires Q8_1 tensors");
                    return false;
                }

                success = kernel->apply_q8_1(
                    q8_gate->q8_1_blocks(),
                    q8_up->q8_1_blocks(),
                    q8_output->mutable_q8_1_blocks(),
                    rows, cols, device_idx);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                // BF16: Native BF16 path
                auto *bf16_gate = dynamic_cast<BF16Tensor *>(gate);
                auto *bf16_up = dynamic_cast<BF16Tensor *>(up);
                auto *bf16_output = dynamic_cast<BF16Tensor *>(output);

                if (!bf16_gate || !bf16_up || !bf16_output)
                {
                    logError("BF16 SwiGLU requires BF16 tensors");
                    return false;
                }

                success = kernel->apply_bf16(
                    bf16_gate->bf16_data(),
                    bf16_up->bf16_data(),
                    bf16_output->mutable_bf16_data(),
                    rows, cols, device_idx);
            }
            else
            {
                // FP32/FP16: Standard path (FP16 uses data() conversion)
                success = kernel->apply(
                    gate->data(),
                    up->data(),
                    output->mutable_data(),
                    rows, cols,
                    false, // no residual
                    mpi_ctx,
                    device_idx);
            }

            if (!success)
            {
                logError("kernel execution failed");
            }
            return success;
        }

        bool execute_raw(
            const float *gate_data,
            const float *up_data,
            float *output_data,
            int rows, int cols,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validatePointer(gate_data, "gate data"))
                return false;
            if (!validatePointer(up_data, "up data"))
                return false;
            if (!validatePointer(output_data, "output data"))
                return false;
            if (!validateDimensions(rows, cols, "SwiGLU"))
                return false;

            // 2. Execute using FP32 kernel
            if (!swiglu_kernel_.apply(
                    gate_data,
                    up_data,
                    output_data,
                    rows, cols,
                    false,
                    mpi_ctx,
                    device_idx))
            {
                logError("SwiGLU kernel execution failed");
                return false;
            }

            return true;
        }

    private:
        CPUSwiGLUKernelT<FP32Tensor> swiglu_kernel_;
    };

    // =========================================================================
    // Legacy SwiGLUOp (backward compatibility)
    // =========================================================================

    /**
     * @brief Legacy untyped SwiGLU operation (backward compatibility)
     *
     * Uses runtime type dispatch based on output tensor type.
     * Prefer using createSwiGLUOp() for new code.
     */
    class SwiGLUOp : public OpBase
    {
    public:
        const char *name() const override { return "SwiGLUOp"; }

        bool operator()(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)snapshot_key; // Handled by pipeline

            // 1. Validate inputs
            if (!validateTensor(gate, "gate"))
                return false;
            if (!validateTensor(up, "up"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "SwiGLU"))
                return false;

            // 2. Execute SwiGLU kernel using FP32 path
            if (!swiglu_kernel_.apply(
                    gate->data(),
                    up->data(),
                    output->mutable_data(),
                    rows, cols,
                    false,
                    mpi_ctx,
                    device_idx))
            {
                logError("SwiGLU kernel execution failed");
                return false;
            }

            return true;
        }

        bool operator()(
            const float *gate_data,
            const float *up_data,
            float *output_data,
            int rows, int cols,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            return impl_.execute_raw(gate_data, up_data, output_data, rows, cols, mpi_ctx, device_idx);
        }

    private:
        CPUSwiGLUKernelT<FP32Tensor> swiglu_kernel_;
        SwiGLUOpTyped<ActivationPrecision::FP32> impl_;
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create typed SwiGLU op for given precision
     *
     * @param precision Activation precision
     * @return Unique pointer to precision-specific SwiGLU op
     */
    inline std::unique_ptr<ISwiGLUOp> createSwiGLUOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::Q8_1:
            return std::make_unique<SwiGLUOpTyped<ActivationPrecision::Q8_1>>();
        case ActivationPrecision::BF16:
            return std::make_unique<SwiGLUOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<SwiGLUOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::FP32:
        default:
            return std::make_unique<SwiGLUOpTyped<ActivationPrecision::FP32>>();
        }
    }

} // namespace llaminar2

/**
 * @file RMSNormOp.h
 * @brief Typed self-validating RMS Normalization operation
 *
 * Template-based operation supporting all activation precisions (FP32, BF16, FP16, Q8_1).
 * Uses compile-time dispatch via ActivationPrecision enum for zero-overhead precision handling.
 *
 * Encapsulates the full RMSNorm workflow:
 * 1. Validate input/weight/output tensors
 * 2. Create RMSNorm kernel from activation tensor
 * 3. Execute kernel with precision-appropriate paths
 * 4. Capture snapshot (if enabled)
 *
 * Supports native precision execution:
 * - FP32Tensor: apply() with float*
 * - BF16Tensor: apply_bf16() with uint16_t*
 * - FP16Tensor: apply_fp16() with uint16_t*
 * - Q8_1Tensor: apply_q8_1() with Q8_1Block*
 *
 * Usage:
 * @code
 * // Create typed op at initialization
 * auto rmsnorm = createRMSNormOp(ActivationPrecision::Q8_1);
 *
 * // Execute (polymorphic call, type-specific implementation)
 * TRY_OP(rmsnorm->execute(input, weight, output, rows, cols, eps, nullptr, device));
 *
 * // Legacy operator() syntax still works
 * RMSNormOp legacy;
 * TRY_OP(legacy(input, weight, output, rows, cols, eps, nullptr, mpi, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "GemmOp.h" // For detail::PrecisionTensor
#include "../PipelineConfig.h"
#include "../../tensors/Tensors.h"
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // IRMSNormOp Interface
    // =========================================================================

    /**
     * @brief Interface for typed RMSNorm operations
     *
     * Provides polymorphic access to precision-specific RMSNorm implementations.
     * Use createRMSNormOp() factory to create instances at runtime.
     */
    class IRMSNormOp
    {
    public:
        virtual ~IRMSNormOp() = default;

        /**
         * @brief Get the activation precision this op was created for
         */
        virtual ActivationPrecision precision() const = 0;

        /**
         * @brief Execute RMS normalization
         *
         * @param input Input tensor [rows, cols]
         * @param weight Gamma weights [cols]
         * @param output Output tensor [rows, cols] (can be same as input)
         * @param rows Number of rows (sequence length)
         * @param cols Number of columns (model dimension)
         * @param eps Epsilon for numerical stability
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        virtual bool execute(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute RMS normalization with raw float pointers
         *
         * Note: Only supports FP32 data.
         */
        virtual bool execute_raw(
            const float *input_data,
            const float *weight_data,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    // =========================================================================
    // RMSNormOpTyped<Precision> Template Implementation
    // =========================================================================

    /**
     * @brief Typed RMSNorm operation with compile-time precision dispatch
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     */
    template <ActivationPrecision Precision>
    class RMSNormOpTyped : public IRMSNormOp, public OpBase
    {
    public:
        using TensorT = typename detail::PrecisionTensor<Precision>::Type;
        using ElementType = typename detail::PrecisionTensor<Precision>::ElementType;

        const char *name() const override { return "RMSNormOpTyped"; }
        ActivationPrecision precision() const override { return Precision; }

        bool execute(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(weight, "weight"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "RMSNorm"))
                return false;

            // 2. Create kernel from output tensor
            auto *activation = dynamic_cast<IActivationTensor *>(output);
            if (!activation)
            {
                logError("output tensor must be IActivationTensor");
                return false;
            }

            auto kernel = activation->createRMSNorm();
            if (!kernel)
            {
                logError("failed to create RMSNorm kernel");
                return false;
            }

            // 3. Execute based on precision
            bool success = false;

            if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                // Q8_1: Native Q8_1 path
                auto *q8_input = dynamic_cast<const Q8_1Tensor *>(input);
                auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
                if (!q8_input || !q8_output)
                {
                    logError("Q8_1 RMSNorm requires Q8_1 input and output tensors");
                    return false;
                }
                success = kernel->apply_q8_1(
                    q8_input->q8_1_blocks(),
                    weight->data(),
                    q8_output->mutable_q8_1_blocks(),
                    rows, cols, eps, device_idx);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                // BF16: Native BF16 path
                auto *bf16_input = dynamic_cast<const BF16Tensor *>(input);
                auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
                if (!bf16_input || !bf16_output)
                {
                    logError("BF16 RMSNorm requires BF16 input and output tensors");
                    return false;
                }
                success = kernel->apply_bf16(
                    bf16_input->bf16_data(),
                    weight->data(),
                    bf16_output->mutable_bf16_data(),
                    rows, cols, eps, device_idx);
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                // FP16: Native FP16 path
                auto *fp16_input = dynamic_cast<const FP16Tensor *>(input);
                auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
                if (!fp16_input || !fp16_output)
                {
                    logError("FP16 RMSNorm requires FP16 input and output tensors");
                    return false;
                }
                success = kernel->apply_fp16(
                    fp16_input->fp16_data(),
                    weight->data(),
                    fp16_output->mutable_fp16_data(),
                    rows, cols, eps, device_idx);
            }
            else
            {
                // FP32: Standard path
                success = kernel->apply(
                    input->data(),
                    weight->data(),
                    output->mutable_data(),
                    rows, cols,
                    eps,
                    false, // use_bf16
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
            const float *input_data,
            const float *weight_data,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validatePointer(input_data, "input data"))
                return false;
            if (!validatePointer(weight_data, "weight data"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "RMSNorm"))
                return false;

            // This overload only supports FP32 output
            if (output->native_type() != TensorType::FP32)
            {
                logError("raw float pointer overload only supports FP32 output tensor");
                return false;
            }

            // 2. Create kernel from output tensor
            auto *activation = dynamic_cast<IActivationTensor *>(output);
            if (!activation)
            {
                logError("output tensor must be IActivationTensor");
                return false;
            }

            auto kernel = activation->createRMSNorm();
            if (!kernel)
            {
                logError("failed to create RMSNorm kernel");
                return false;
            }

            // 3. Execute
            if (!kernel->apply(
                    input_data,
                    weight_data,
                    output->mutable_data(),
                    rows, cols,
                    eps,
                    false,
                    mpi_ctx,
                    device_idx))
            {
                logError("kernel execution failed");
                return false;
            }

            return true;
        }
    };

    // =========================================================================
    // Legacy RMSNormOp (backward compatibility)
    // =========================================================================

    /**
     * @brief Legacy untyped RMSNorm operation (backward compatibility)
     *
     * Uses runtime type dispatch based on output tensor type.
     * Prefer using createRMSNormOp() for new code.
     */
    class RMSNormOp : public OpBase
    {
    public:
        const char *name() const override { return "RMSNormOp"; }

        bool operator()(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)snapshot_key; // Handled by pipeline

            // 1. Validate inputs
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(weight, "weight"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "RMSNorm"))
                return false;

            // 2. Create kernel from output tensor
            auto *activation = dynamic_cast<IActivationTensor *>(output);
            if (!activation)
            {
                logError("output tensor must be IActivationTensor");
                return false;
            }

            auto kernel = activation->createRMSNorm();
            if (!kernel)
            {
                logError("failed to create RMSNorm kernel");
                return false;
            }

            // 3. Execute based on output tensor type (runtime dispatch)
            bool success = false;
            const TensorType out_type = output->native_type();

            switch (out_type)
            {
            case TensorType::FP32:
                success = kernel->apply(
                    input->data(),
                    weight->data(),
                    output->mutable_data(),
                    rows, cols,
                    eps,
                    false,
                    mpi_ctx,
                    device_idx);
                break;

            case TensorType::BF16:
            {
                auto *bf16_input = dynamic_cast<const BF16Tensor *>(input);
                auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
                if (!bf16_input || !bf16_output)
                {
                    logError("BF16 RMSNorm requires BF16 input and output tensors");
                    return false;
                }
                success = kernel->apply_bf16(
                    bf16_input->bf16_data(),
                    weight->data(),
                    bf16_output->mutable_bf16_data(),
                    rows, cols, eps, device_idx);
                break;
            }

            case TensorType::FP16:
            {
                auto *fp16_input = dynamic_cast<const FP16Tensor *>(input);
                auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
                if (!fp16_input || !fp16_output)
                {
                    logError("FP16 RMSNorm requires FP16 input and output tensors");
                    return false;
                }
                success = kernel->apply_fp16(
                    fp16_input->fp16_data(),
                    weight->data(),
                    fp16_output->mutable_fp16_data(),
                    rows, cols, eps, device_idx);
                break;
            }

            case TensorType::Q8_1:
            {
                auto *q8_input = dynamic_cast<const Q8_1Tensor *>(input);
                auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
                if (!q8_input || !q8_output)
                {
                    logError("Q8_1 RMSNorm requires Q8_1 input and output tensors");
                    return false;
                }
                success = kernel->apply_q8_1(
                    q8_input->q8_1_blocks(),
                    weight->data(),
                    q8_output->mutable_q8_1_blocks(),
                    rows, cols, eps, device_idx);
                break;
            }

            default:
                logError("unsupported output tensor type for RMSNorm");
                return false;
            }

            if (!success)
            {
                logError("kernel execution failed");
            }
            return success;
        }

        bool operator()(
            const float *input_data,
            const float *weight_data,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)snapshot_key;
            return impl_.execute_raw(input_data, weight_data, output, rows, cols, eps, mpi_ctx, device_idx);
        }

    private:
        RMSNormOpTyped<ActivationPrecision::FP32> impl_;
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create typed RMSNorm op for given precision
     *
     * @param precision Activation precision
     * @return Unique pointer to precision-specific RMSNorm op
     */
    inline std::unique_ptr<IRMSNormOp> createRMSNormOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::Q8_1:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::Q8_1>>();
        case ActivationPrecision::BF16:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::FP32:
        default:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::FP32>>();
        }
    }

} // namespace llaminar2

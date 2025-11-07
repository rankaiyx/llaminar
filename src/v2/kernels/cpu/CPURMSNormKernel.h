/**
 * @file CPURMSNormKernel.h
 * @brief CPU implementation of RMS normalization
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"

namespace llaminar2
{

    /**
     * @brief CPU implementation of RMSNorm kernel
     */
    class CPURMSNormKernel : public ITensorRMSNorm
    {
    public:
        CPURMSNormKernel() = default;
        ~CPURMSNormKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        // FP32 RMSNorm (standard path)
        bool apply(
            const float *input, const float *gamma, float *output,
            int seq_len, int d_model,
            float eps = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        /**
         * @brief INT32 RMSNorm with INT8 output (for full INT8 pipelines)
         *
         * Performs RMS normalization on INT32 accumulator tensors and
         * requantizes output to INT8 with per-row dynamic scaling.
         *
         * Pipeline: INT32 input → normalize → apply gamma → requantize → INT8 output
         *
         * @param input_int32 Input INT32 tensor [seq_len, d_model]
         * @param gamma Scale parameter [d_model] (FP32)
         * @param output_int8 Output INT8 tensor [seq_len, d_model]
         * @param output_row_scales Per-row INT8 scales [seq_len]
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param eps Epsilon for numerical stability
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success
         */
        bool apply_int32_to_int8(
            const int32_t *input_int32,
            const float *gamma,
            int8_t *output_int8,
            float *output_row_scales,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            int device_idx = -1);
    };

} // namespace llaminar2

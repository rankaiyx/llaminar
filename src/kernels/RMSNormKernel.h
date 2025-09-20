#pragma once

#include <vector>
#include <memory>
#include <string>
#include "../kernel_base.h"

namespace llaminar
{

    /**
     * @brief RMS Normalization kernel for transformer layers
     *
     * Implements Root Mean Square Layer Normalization:
     * y = (x / sqrt(mean(x^2) + eps)) * weight
     *
     * Expected inputs:
     * - input: [seq_len, hidden_size] - input tensor
     * - weight: [hidden_size] - learnable scale parameters
     *
     * Expected outputs:
     * - output: [seq_len, hidden_size] - normalized output tensor
     */
    class RMSNormKernel : public KernelBase
    {
    public:
        RMSNormKernel();

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getKernelType() const override { return "RMSNorm"; }
        size_t getExpectedInputCount() const override { return 2; }
        size_t getExpectedOutputCount() const override { return 1; }

        // Configuration
        void setEpsilon(float eps) { epsilon_ = eps; }
        float getEpsilon() const { return epsilon_; }

    private:
        /**
         * @brief Core RMS normalization computation
         * @param input Input data pointer
         * @param weight Weight data pointer
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param hidden_size Hidden dimension size
         */
        void computeRMSNorm(const float *input, const float *weight,
                            float *output, int seq_len, int hidden_size);

        float epsilon_; ///< Small value to prevent division by zero
    };

} // namespace llaminar
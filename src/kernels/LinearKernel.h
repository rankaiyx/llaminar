#pragma once

#include "../tensor.h"
#include "../kernel_base.h"
#include <string>
#include <vector>
#include <memory>

namespace llaminar
{

    /**
     * @brief Linear layer kernel for matrix multiplication with optional bias
     */
    class LinearKernel : public KernelBase
    {
    public:
        LinearKernel() = default;
        ~LinearKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                     std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) const override;

        std::string getKernelType() const override { return "Linear"; }
        size_t getExpectedInputCount() const override { return 2; } // input + weight (bias optional)
        size_t getExpectedOutputCount() const override { return 1; }

    private:
        /**
         * @brief Core linear projection computation
         * @param input Input data pointer
         * @param weight Weight matrix data pointer
         * @param bias Bias vector data pointer (can be nullptr)
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param input_size Input dimension size
         * @param output_size Output dimension size
         */
        void computeLinear(const float *input, const float *weight, const float *bias,
                           float *output, size_t seq_len, size_t input_size, size_t output_size);
    };

} // namespace llaminar
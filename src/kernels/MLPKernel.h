#pragma once

#include "../tensor.h"
#include "../kernel_base.h"
#include <string>
#include <vector>
#include <memory>

namespace llaminar
{

    /**
     * @brief Multi-layer perceptron (MLP) kernel with SwiGLU activation
     */
    class MLPKernel : public KernelBase
    {
    public:
        MLPKernel() = default;
        ~MLPKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                     std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) const override;

        std::string getKernelType() const override { return "MLP"; }
        size_t getExpectedInputCount() const override { return 4; } // input, w_gate, w_up, w_down
        size_t getExpectedOutputCount() const override { return 1; }

    private:
        /**
         * @brief Compute gate and up projections
         * @param input Input data pointer
         * @param w_gate Gate weight matrix
         * @param w_up Up weight matrix
         * @param gate_output Gate projection output
         * @param up_output Up projection output
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param d_ff Feed-forward dimension
         */
        void computeGateAndUp(const float *input, const float *w_gate, const float *w_up,
                              float *gate_output, float *up_output,
                              size_t seq_len, size_t d_model, size_t d_ff);

        /**
         * @brief Apply SwiGLU activation function
         * @param gate_output Gate projection values
         * @param up_output Up projection values
         * @param output Combined output after SwiGLU
         * @param seq_len Sequence length
         * @param d_ff Feed-forward dimension
         */
        void applySwiGLU(const float *gate_output, const float *up_output, float *output,
                         size_t seq_len, size_t d_ff);

        /**
         * @brief Compute down projection
         * @param input Input data pointer
         * @param w_down Down weight matrix
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param d_ff Feed-forward dimension
         * @param d_model Model dimension
         */
        void computeDown(const float *input, const float *w_down, float *output,
                         size_t seq_len, size_t d_ff, size_t d_model);

        /**
         * @brief SiLU (Swish) activation function
         * @param x Input value
         * @return SiLU(x) = x * sigmoid(x)
         */
        float silu(float x) const;
    };

} // namespace llaminar
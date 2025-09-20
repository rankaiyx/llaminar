#include "MLPKernel.h"
#include "../logger.h"
#include <algorithm>
#include <cmath>

namespace llaminar
{

    bool MLPKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                            std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MLPKernel validation failed");
            return false;
        }

        auto input = inputs[0];
        auto w_gate = inputs[1];
        auto w_up = inputs[2];
        auto w_down = inputs[3];
        auto output = outputs[0];

        size_t seq_len = input->shape()[0];
        size_t d_model = input->shape()[1];
        size_t d_ff = w_gate->shape()[1];

        // Allocate temporary buffers
        std::vector<float> gate_proj(seq_len * d_ff);
        std::vector<float> up_proj(seq_len * d_ff);
        std::vector<float> activated(seq_len * d_ff);

        // Compute gate and up projections
        computeGateAndUp(input->data(), w_gate->data(), w_up->data(),
                         gate_proj.data(), up_proj.data(), seq_len, d_model, d_ff);

        // Apply SwiGLU activation
        applySwiGLU(gate_proj.data(), up_proj.data(), activated.data(), seq_len, d_ff);

        // Compute down projection
        computeDown(activated.data(), w_down->data(), output->data(),
                    seq_len, d_ff, d_model);

        return true;
    }

    bool MLPKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                             const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 4 || outputs.size() != 1)
        {
            LOG_ERROR("MLPKernel: Expected 4 inputs and 1 output, got " << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        // Basic null checks
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputs[i])
            {
                LOG_ERROR("MLPKernel: Input " << i << " is null");
                return false;
            }
        }

        if (!outputs[0])
        {
            LOG_ERROR("MLPKernel: Output is null");
            return false;
        }

        auto input = inputs[0];
        auto w_gate = inputs[1];
        auto w_up = inputs[2];
        auto w_down = inputs[3];
        auto output = outputs[0];

        // Check input is 2D [seq_len, d_model]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MLPKernel: Input must be 2D, got " << input->shape().size() << " dimensions");
            return false;
        }

        size_t seq_len = input->shape()[0];
        size_t d_model = input->shape()[1];

        // Check weight dimensions
        if (w_gate->shape().size() != 2 || w_gate->shape()[0] != d_model)
        {
            LOG_ERROR("MLPKernel: Gate weight shape mismatch");
            return false;
        }

        if (w_up->shape().size() != 2 || w_up->shape()[0] != d_model || w_up->shape()[1] != w_gate->shape()[1])
        {
            LOG_ERROR("MLPKernel: Up weight shape mismatch");
            return false;
        }

        size_t d_ff = w_gate->shape()[1];
        if (w_down->shape().size() != 2 || w_down->shape()[0] != d_ff || w_down->shape()[1] != d_model)
        {
            LOG_ERROR("MLPKernel: Down weight shape mismatch");
            return false;
        }

        // Check output shape
        if (output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != d_model)
        {
            LOG_ERROR("MLPKernel: Output shape mismatch");
            return false;
        }

        return true;
    }

    void MLPKernel::computeGateAndUp(const float *input, const float *w_gate, const float *w_up,
                                     float *gate_output, float *up_output,
                                     size_t seq_len, size_t d_model, size_t d_ff)
    {
        // Compute gate projection: gate_output = input * w_gate
        for (size_t i = 0; i < seq_len; ++i)
        {
            for (size_t j = 0; j < d_ff; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < d_model; ++k)
                {
                    sum += input[i * d_model + k] * w_gate[k * d_ff + j];
                }
                gate_output[i * d_ff + j] = sum;
            }
        }

        // Compute up projection: up_output = input * w_up
        for (size_t i = 0; i < seq_len; ++i)
        {
            for (size_t j = 0; j < d_ff; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < d_model; ++k)
                {
                    sum += input[i * d_model + k] * w_up[k * d_ff + j];
                }
                up_output[i * d_ff + j] = sum;
            }
        }
    }

    void MLPKernel::applySwiGLU(const float *gate_output, const float *up_output, float *output,
                                size_t seq_len, size_t d_ff)
    {
        // Apply SwiGLU: output = silu(gate) * up
        for (size_t i = 0; i < seq_len * d_ff; ++i)
        {
            output[i] = silu(gate_output[i]) * up_output[i];
        }
    }

    void MLPKernel::computeDown(const float *input, const float *w_down, float *output,
                                size_t seq_len, size_t d_ff, size_t d_model)
    {
        // Compute down projection: output = input * w_down
        for (size_t i = 0; i < seq_len; ++i)
        {
            for (size_t j = 0; j < d_model; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < d_ff; ++k)
                {
                    sum += input[i * d_ff + k] * w_down[k * d_model + j];
                }
                output[i * d_model + j] = sum;
            }
        }
    }

    float MLPKernel::silu(float x) const
    {
        // SiLU (Swish) activation: x * sigmoid(x)
        return x / (1.0f + std::exp(-x));
    }

} // namespace llaminar
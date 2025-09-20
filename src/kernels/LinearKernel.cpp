#include "LinearKernel.h"
#include "../logger.h"
#include <algorithm>

namespace llaminar
{

    bool LinearKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                               std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("LinearKernel validation failed");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        // Check for optional bias
        const float *bias = nullptr;
        if (inputs.size() >= 3 && inputs[2])
        {
            bias = inputs[2]->data();
        }

        size_t seq_len = input->shape()[0];
        size_t input_size = input->shape()[1];
        size_t output_size = weight->shape()[1];

        computeLinear(input->data(), weight->data(), bias,
                      output->data(), seq_len, input_size, output_size);

        return true;
    }

    bool LinearKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
        {
            LOG_ERROR("LinearKernel: Expected 2-3 inputs and 1 output, got " << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("LinearKernel: Null tensor provided");
            return false;
        }

        // Check input is 2D [seq_len, input_size]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("LinearKernel: Input must be 2D, got " << input->shape().size() << " dimensions");
            return false;
        }

        // Check weight is 2D [input_size, output_size]
        if (weight->shape().size() != 2)
        {
            LOG_ERROR("LinearKernel: Weight must be 2D, got " << weight->shape().size() << " dimensions");
            return false;
        }

        // Check dimensions match
        if (input->shape()[1] != weight->shape()[0])
        {
            LOG_ERROR("LinearKernel: Input size " << input->shape()[1] << " doesn't match weight input size " << weight->shape()[0]);
            return false;
        }

        // Check output is 2D [seq_len, output_size]
        if (output->shape().size() != 2 ||
            output->shape()[0] != input->shape()[0] ||
            output->shape()[1] != weight->shape()[1])
        {
            LOG_ERROR("LinearKernel: Output shape mismatch");
            return false;
        }

        // Check optional bias
        if (inputs.size() == 3 && inputs[2])
        {
            auto bias = inputs[2];
            if (bias->shape().size() != 1 || bias->shape()[0] != weight->shape()[1])
            {
                LOG_ERROR("LinearKernel: Bias shape mismatch");
                return false;
            }
        }

        return true;
    }

    void LinearKernel::computeLinear(const float *input, const float *weight, const float *bias,
                                     float *output, size_t seq_len, size_t input_size, size_t output_size)
    {
        // Perform matrix multiplication: output = input * weight + bias
        for (size_t i = 0; i < seq_len; ++i)
        {
            for (size_t j = 0; j < output_size; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < input_size; ++k)
                {
                    sum += input[i * input_size + k] * weight[k * output_size + j];
                }
                if (bias)
                {
                    sum += bias[j];
                }
                output[i * output_size + j] = sum;
            }
        }
    }

} // namespace llaminar
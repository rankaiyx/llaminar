/**
 * @file IMoEExpertExecutor.cpp
 * @brief CPU reference implementation of expert FFN (SwiGLU)
 */

#include "IMoEExpertExecutor.h"
#include <cmath>
#include <vector>

namespace llaminar2
{

    bool CPUMoEExpertExecutor::executeExpert(
        const float *input,
        float *output,
        const float *gate_w,
        const float *up_w,
        const float *down_w,
        const ExpertBatch &batch,
        int d_model,
        int intermediate_dim)
    {
        if (batch.empty())
            return true;

        // Scratch buffers for one token's intermediate computation
        std::vector<float> gate_out(intermediate_dim);
        std::vector<float> up_out(intermediate_dim);
        std::vector<float> swiglu_out(intermediate_dim);

        for (int i = 0; i < batch.numTokens(); ++i)
        {
            int t = batch.token_indices[i];
            const float *x = input + static_cast<size_t>(t) * d_model;
            float *y = output + static_cast<size_t>(t) * d_model;

            // Gate projection: gate_out = x @ gate_w^T
            // gate_w is [intermediate, d_model], row-major
            for (int j = 0; j < intermediate_dim; ++j)
            {
                float dot = 0.0f;
                const float *w = gate_w + static_cast<size_t>(j) * d_model;
                for (int k = 0; k < d_model; ++k)
                    dot += x[k] * w[k];
                gate_out[j] = dot;
            }

            // Up projection: up_out = x @ up_w^T
            for (int j = 0; j < intermediate_dim; ++j)
            {
                float dot = 0.0f;
                const float *w = up_w + static_cast<size_t>(j) * d_model;
                for (int k = 0; k < d_model; ++k)
                    dot += x[k] * w[k];
                up_out[j] = dot;
            }

            // SwiGLU: swiglu_out = SiLU(gate_out) * up_out
            for (int j = 0; j < intermediate_dim; ++j)
            {
                float silu = gate_out[j] / (1.0f + std::exp(-gate_out[j]));
                swiglu_out[j] = silu * up_out[j];
            }

            // Down projection: y = swiglu_out @ down_w^T
            // down_w is [d_model, intermediate], row-major
            for (int j = 0; j < d_model; ++j)
            {
                float dot = 0.0f;
                const float *w = down_w + static_cast<size_t>(j) * intermediate_dim;
                for (int k = 0; k < intermediate_dim; ++k)
                    dot += swiglu_out[k] * w[k];
                y[j] = dot;
            }
        }

        return true;
    }

} // namespace llaminar2

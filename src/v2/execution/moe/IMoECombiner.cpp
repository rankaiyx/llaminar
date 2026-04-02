/**
 * @file IMoECombiner.cpp
 * @brief CPU reference implementation of MoE combiner
 */

#include "IMoECombiner.h"
#include <cstring>

namespace llaminar2
{

    bool CPUMoECombiner::combine(
        const float *expert_outputs,
        const DispatchPlan &plan,
        float *output,
        int d_model)
    {
        const int seq_len = plan.seq_len;

        // Zero the output first — tokens accumulate from multiple experts
        std::memset(output, 0, static_cast<size_t>(seq_len) * d_model * sizeof(float));

        // For each expert batch, accumulate weighted contributions
        for (const auto &batch : plan.batches)
        {
            for (int i = 0; i < batch.numTokens(); ++i)
            {
                int t = batch.token_indices[i];
                float w = batch.weights[i];

                // expert_outputs[t] was written by this expert
                const float *src = expert_outputs + static_cast<size_t>(t) * d_model;
                float *dst = output + static_cast<size_t>(t) * d_model;

                for (int d = 0; d < d_model; ++d)
                    dst[d] += w * src[d];
            }
        }

        return true;
    }

} // namespace llaminar2

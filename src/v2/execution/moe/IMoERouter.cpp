/**
 * @file IMoERouter.cpp
 * @brief Implementation of SoftmaxTopKRouter
 */

#include "IMoERouter.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace llaminar2
{

    RoutingTable SoftmaxTopKRouter::route(
        const float *gate_logits,
        int seq_len,
        int num_experts,
        int top_k)
    {
        RoutingTable table;
        table.seq_len = seq_len;
        table.top_k = top_k;
        table.entries.reserve(static_cast<size_t>(seq_len) * top_k);

        // Scratch for softmax probabilities per token
        std::vector<float> probs(num_experts);

        // Scratch for top-k selection (index, probability)
        struct IndexProb
        {
            int index;
            float prob;
        };
        std::vector<IndexProb> sorted(num_experts);

        for (int t = 0; t < seq_len; ++t)
        {
            const float *logits = gate_logits + static_cast<size_t>(t) * num_experts;

            // Numerically stable softmax
            float max_val = *std::max_element(logits, logits + num_experts);
            float sum_exp = 0.0f;
            for (int e = 0; e < num_experts; ++e)
            {
                probs[e] = std::exp(logits[e] - max_val);
                sum_exp += probs[e];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int e = 0; e < num_experts; ++e)
                probs[e] *= inv_sum;

            // Populate sortable array
            for (int e = 0; e < num_experts; ++e)
                sorted[e] = {e, probs[e]};

            // Partial sort for top-k
            std::partial_sort(sorted.begin(), sorted.begin() + top_k, sorted.end(),
                              [](const IndexProb &a, const IndexProb &b)
                              { return a.prob > b.prob; });

            // Optionally renormalize top-k weights
            float weight_sum = 0.0f;
            if (normalize_weights_)
            {
                for (int k = 0; k < top_k; ++k)
                    weight_sum += sorted[k].prob;
            }

            // Emit entries
            for (int k = 0; k < top_k; ++k)
            {
                float w = sorted[k].prob;
                if (normalize_weights_ && weight_sum > 0.0f)
                    w /= weight_sum;

                table.entries.push_back(TokenRouteEntry{
                    .token_idx = t,
                    .expert_id = sorted[k].index,
                    .weight = w,
                });
            }
        }

        return table;
    }

} // namespace llaminar2

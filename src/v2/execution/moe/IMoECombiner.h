/**
 * @file IMoECombiner.h
 * @brief Interface for combining expert outputs with routing weights
 *
 * The combiner merges outputs from multiple experts for each token,
 * weighted by the router's selection weights.
 */

#pragma once

#include "MoETypes.h"

namespace llaminar2
{

    /**
     * @brief Interface for combining expert outputs
     */
    class IMoECombiner
    {
    public:
        virtual ~IMoECombiner() = default;

        /**
         * @brief Combine expert outputs into final hidden states
         *
         * For each token t:
         *   output[t] = Σ_{k ∈ top_k} weight[t][k] * expert_output[t]
         *
         * @param expert_outputs Per-expert output buffer [seq_len, d_model].
         *                       Each expert wrote only its assigned rows.
         *                       This is indexed by the dispatch plan.
         * @param plan           The dispatch plan (knows which experts wrote which rows)
         * @param output         Final combined output [seq_len, d_model]
         * @param d_model        Hidden dimension
         * @return true on success
         */
        virtual bool combine(
            const float *expert_outputs,
            const DispatchPlan &plan,
            float *output,
            int d_model) = 0;
    };

    /**
     * @brief CPU reference combiner: weighted sum of expert contributions
     */
    class CPUMoECombiner : public IMoECombiner
    {
    public:
        bool combine(
            const float *expert_outputs,
            const DispatchPlan &plan,
            float *output,
            int d_model) override;
    };

} // namespace llaminar2

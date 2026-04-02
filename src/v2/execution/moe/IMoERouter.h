/**
 * @file IMoERouter.h
 * @brief Interface for MoE routing strategies
 *
 * Decouples the routing algorithm from the stage infrastructure.
 * Enables unit testing of routing logic without compute stages,
 * and allows different routing strategies (softmax top-k, expert choice, etc.)
 */

#pragma once

#include "MoETypes.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Interface for MoE token-to-expert routing
     *
     * Implementations compute which experts each token should be sent to,
     * along with the routing weights.
     *
     * Thread safety: route() must be safe to call from any thread but is
     * NOT required to be reentrant (the pipeline calls it once per layer).
     */
    class IMoERouter
    {
    public:
        virtual ~IMoERouter() = default;

        /**
         * @brief Compute routing decisions from gate logits
         *
         * @param gate_logits Pre-computed gate logits [seq_len, num_experts]
         *                    Row-major: logits[t * num_experts + e]
         * @param seq_len     Number of tokens
         * @param num_experts Total number of experts
         * @param top_k       Number of experts to activate per token
         * @return RoutingTable with seq_len * top_k entries
         */
        virtual RoutingTable route(
            const float *gate_logits,
            int seq_len,
            int num_experts,
            int top_k) = 0;
    };

    /**
     * @brief Standard softmax top-k router
     *
     * 1. Apply softmax to gate_logits per token
     * 2. Select top-k experts per token
     * 3. Optionally renormalize top-k weights to sum to 1
     */
    class SoftmaxTopKRouter : public IMoERouter
    {
    public:
        /**
         * @param normalize_weights If true, top-k weights are renormalized to sum to 1
         */
        explicit SoftmaxTopKRouter(bool normalize_weights = false)
            : normalize_weights_(normalize_weights) {}

        RoutingTable route(
            const float *gate_logits,
            int seq_len,
            int num_experts,
            int top_k) override;

    private:
        bool normalize_weights_;
    };

} // namespace llaminar2

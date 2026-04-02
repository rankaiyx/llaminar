/**
 * @file IMoEDispatcher.h
 * @brief Interface for MoE token dispatch strategies
 *
 * Converts a RoutingTable (per-token expert selection) into a DispatchPlan
 * (per-expert token batches). This separation enables testing dispatch
 * logic independently from the compute stage.
 */

#pragma once

#include "MoETypes.h"

namespace llaminar2
{

    /**
     * @brief Interface for building per-expert batches from routing decisions
     */
    class IMoEDispatcher
    {
    public:
        virtual ~IMoEDispatcher() = default;

        /**
         * @brief Convert routing table to per-expert batched dispatch plan
         *
         * @param table   Routing decisions from IMoERouter
         * @param d_model Hidden dimension (for output sizing)
         * @return DispatchPlan with one ExpertBatch per active expert
         */
        virtual DispatchPlan dispatch(
            const RoutingTable &table,
            int d_model) = 0;
    };

    /**
     * @brief Standard dispatcher: groups tokens by expert
     *
     * Each expert gets a contiguous batch of its assigned tokens.
     * Simple and efficient for CPU execution.
     */
    class StandardMoEDispatcher : public IMoEDispatcher
    {
    public:
        DispatchPlan dispatch(
            const RoutingTable &table,
            int d_model) override;
    };

} // namespace llaminar2

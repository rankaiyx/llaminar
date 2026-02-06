/**
 * @file GlobalPPRankPlanBuilder.h
 * @brief Builds per-rank execution plans from Global PP topology
 *
 * Takes a validated GlobalPPTopology and produces a GlobalPPRankPlan for
 * each rank, including the correct step ordering (execute, send, recv)
 * to avoid deadlocks.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "GlobalPPTopology.h"
#include "GlobalPPRankPlan.h"

namespace llaminar2
{

    /**
     * @brief Builds per-rank execution plans from a Global PP topology
     *
     * The builder walks through stages in order and for each rank determines:
     * - Whether the rank executes the stage
     * - What transfers (send/recv) are needed before/after the stage
     * - Weight shard information for the rank's portion
     *
     * The generated plan is deadlock-free by construction:
     * - Sends and receives are matched (one sender, one receiver, same tag)
     * - Global TP stages are entered by all participants simultaneously
     * - Pipeline ordering is respected
     */
    class GlobalPPRankPlanBuilder
    {
    public:
        /**
         * @brief Build a rank plan from topology
         *
         * @param topology Validated Global PP topology
         * @param rank MPI rank to build plan for
         * @return GlobalPPRankPlan for the given rank
         */
        static GlobalPPRankPlan build(const GlobalPPTopology &topology, int rank);

    private:
        /**
         * @brief Build a RankStageAction for a given stage and rank
         */
        static RankStageAction buildStageAction(const GlobalPPStageSpec &stage, int rank);

        /**
         * @brief Find transfers where this rank is sender or receiver
         *
         * For the transition from stage[i] to stage[i+1], determine what
         * this rank needs to do (send, recv, or nothing).
         */
        static std::vector<RankTransferAction> buildTransfersForRank(
            const GlobalPPTopology &topology,
            int from_stage_id,
            int to_stage_id,
            int rank);
    };

} // namespace llaminar2

/**
 * @file GlobalPPTopology.h
 * @brief Topology description for Global Pipeline Parallelism
 *
 * Describes a per-machine pipeline topology that composes multiple MPI ranks'
 * resources into a single inference pipeline. Each stage in the pipeline is
 * either:
 * - A single-rank stage (one MPI rank executes, optionally with LocalTP/LocalPP)
 * - A global-TP stage (multiple MPI ranks execute in tensor parallel)
 *
 * The topology is created from CLI/YAML configuration and distributed to all
 * ranks. Each rank derives its own GlobalPPRankPlan from the topology.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../../backends/GlobalDeviceAddress.h"
#include <string>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // Enums
    // =========================================================================

    /**
     * @brief Parallelism mode for devices within a single-rank Global PP stage
     */
    enum class InnerParallelism
    {
        SINGLE_DEVICE, ///< One device, no inner parallelism
        LOCAL_TP,      ///< Tensor parallel across local devices
        LOCAL_PP,      ///< Pipeline parallel across local devices
    };

    /**
     * @brief Convert InnerParallelism to string
     */
    inline const char *innerParallelismName(InnerParallelism mode)
    {
        switch (mode)
        {
        case InnerParallelism::SINGLE_DEVICE:
            return "SINGLE_DEVICE";
        case InnerParallelism::LOCAL_TP:
            return "LOCAL_TP";
        case InnerParallelism::LOCAL_PP:
            return "LOCAL_PP";
        default:
            return "UNKNOWN";
        }
    }

    // =========================================================================
    // GlobalPPStageSpec
    // =========================================================================

    /**
     * @brief Specification for one stage in the Global PP pipeline
     *
     * A stage is a contiguous range of transformer layers assigned to specific
     * hardware. Stages execute sequentially in the pipeline, with activations
     * transferred between them via MPI send/recv.
     */
    struct GlobalPPStageSpec
    {
        int stage_id = -1; ///< 0-indexed stage number in the pipeline

        // =====================================================================
        // Layer assignment
        // =====================================================================
        int first_layer = -1;        ///< First layer index (inclusive, 0-based)
        int last_layer = -1;         ///< Last layer index (inclusive, 0-based)
        bool has_embedding = false;  ///< This stage includes the embedding layer
        bool has_lm_head = false;    ///< This stage includes the LM head

        // =====================================================================
        // Ownership
        // =====================================================================
        bool is_global_tp = false;   ///< Multi-rank stage (all participating ranks execute)
        int owning_rank = -1;        ///< For single-rank stages: which MPI rank owns this stage

        // =====================================================================
        // Inner parallelism (single-rank stages only)
        // =====================================================================
        InnerParallelism inner_mode = InnerParallelism::SINGLE_DEVICE;
        std::vector<GlobalDeviceAddress> devices; ///< Devices for this stage
        std::vector<float> tp_weights;            ///< Proportional weights (heterogeneous LocalTP)

        // =====================================================================
        // Global TP config (multi-rank stages only)
        // =====================================================================
        std::vector<int> participating_ranks;     ///< Which MPI ranks participate (sorted)
        GlobalDeviceAddress per_rank_device;       ///< Device to use on each rank (e.g., cpu:0)

        // =====================================================================
        // Helpers
        // =====================================================================

        /** @brief Number of layers in this stage */
        int layerCount() const
        {
            return (first_layer >= 0 && last_layer >= first_layer) ? (last_layer - first_layer + 1) : 0;
        }

        /** @brief Check if a given layer belongs to this stage */
        bool hasLayer(int layer) const
        {
            return layer >= first_layer && layer <= last_layer;
        }

        /** @brief Check if a given rank participates in this stage */
        bool rankParticipates(int rank) const
        {
            if (is_global_tp)
            {
                for (int r : participating_ranks)
                {
                    if (r == rank) return true;
                }
                return false;
            }
            return rank == owning_rank;
        }

        /** @brief Validate this stage spec; returns error messages */
        std::vector<std::string> validate() const;
    };

    // =========================================================================
    // GlobalPPTransfer
    // =========================================================================

    /**
     * @brief Describes a transfer of activations between two adjacent Global PP stages
     *
     * Derived automatically from the stage specifications. Each transfer has a
     * unique sender rank and receiver rank, identified by MPI tag.
     */
    struct GlobalPPTransfer
    {
        int from_stage = -1;     ///< Source stage ID
        int to_stage = -1;       ///< Destination stage ID
        int sender_rank = -1;    ///< MPI rank that sends
        int receiver_rank = -1;  ///< MPI rank that receives (-1 if global TP → same set)
        int mpi_tag = 0;         ///< Unique MPI tag for this transfer

        /** @brief Check if this transfer is a no-op (both ranks are the same) */
        bool isNoop() const { return sender_rank == receiver_rank; }
    };

    // =========================================================================
    // GlobalPPTopology
    // =========================================================================

    /**
     * @brief Full pipeline topology for one machine (across MPI ranks)
     *
     * Created from CLI/YAML configuration and broadcast to all ranks.
     * Each rank uses this to derive its own GlobalPPRankPlan.
     *
     * Invariants:
     * - Stages are sorted by stage_id (0, 1, 2, ...)
     * - Stage layer ranges cover all model layers exactly once
     * - Stage 0 has_embedding = true; last stage has_lm_head = true
     * - Global TP stages have ≥2 participating ranks
     * - Single-rank stages have a valid owning_rank
     * - Transfers are derived deterministically from stages
     */
    struct GlobalPPTopology
    {
        std::vector<GlobalPPStageSpec> stages;      ///< Pipeline stages (ordered)
        std::vector<GlobalPPTransfer> transfers;     ///< Inter-stage transfers (derived)
        int total_layers = 0;                        ///< Total transformer layers in model
        int world_size = 0;                          ///< Number of MPI ranks

        // =====================================================================
        // Factory
        // =====================================================================

        /**
         * @brief Build topology from stage specs and derive transfers
         *
         * Sets up transfer records between adjacent stages based on ownership.
         * Validates the topology and returns errors if invalid.
         *
         * @param specs Stage specifications (will be sorted by stage_id)
         * @param total_layers Total number of transformer layers in the model
         * @param world_size Number of MPI ranks
         * @return Constructed topology
         */
        static GlobalPPTopology build(std::vector<GlobalPPStageSpec> specs,
                                      int total_layers, int world_size);

        // =====================================================================
        // Queries
        // =====================================================================

        /** @brief Get stages that a given rank participates in */
        std::vector<int> stagesForRank(int rank) const;

        /** @brief Get the stage that contains a given layer */
        const GlobalPPStageSpec *stageForLayer(int layer) const;

        /** @brief Check if a rank participates in a given stage */
        bool rankParticipatesInStage(int rank, int stage_id) const;

        /** @brief Get transfer record between two adjacent stages (nullptr if not found) */
        const GlobalPPTransfer *transferBetween(int from_stage, int to_stage) const;

        /** @brief Number of stages */
        int numStages() const { return static_cast<int>(stages.size()); }

        // =====================================================================
        // Validation
        // =====================================================================

        /**
         * @brief Validate the topology for consistency
         * @return List of error messages (empty = valid)
         */
        std::vector<std::string> validate() const;

        /**
         * @brief Human-readable summary for logging
         */
        std::string toString() const;
    };

} // namespace llaminar2

/**
 * @file CompoundShardInfo.h
 * @brief Multi-level shard descriptor for nested parallelism
 *
 * CompoundShardInfo describes the sharding state at a specific leaf device
 * in the parallelism tree. It combines:
 *
 * - **Layer range**: Which transformer layers this device processes (from PP)
 * - **TP shard position**: This device's position within tensor-parallel groups
 * - **Work fraction**: The proportion of work this device handles (for proportional TP)
 *
 * For deeply nested trees with TP at multiple levels, the shard indices and
 * totals are computed by walking from root to leaf and accumulating products.
 *
 * Example: In a tree where root TP has 2 children, each with another TP of 2,
 * the total shards = 4. A leaf at position (1,0) has shard_index = 2.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "ParallelismTree.h"
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Describes how weights are sharded at a specific leaf device
     *
     * This is the compound view of all parallelism levels that affect a leaf.
     * Used by WeightLoader to know exactly which shard of each weight tensor
     * this device should load.
     */
    struct CompoundShardInfo
    {
        int layer_first = -1;       ///< First layer this device processes (inclusive, 0-based)
        int layer_last = -1;        ///< Last layer this device processes (inclusive, 0-based)
        int tp_shard_index = 0;     ///< This device's TP shard index (0-based, global across nested TP)
        int tp_total_shards = 1;    ///< Total TP shards at this level (product of nested TP degrees)
        float work_fraction = 1.0f; ///< Fraction of work (for proportional TP)

        bool has_embedding = false; ///< Does this device handle embedding lookup?
        bool has_lm_head = false;   ///< Does this device handle LM head projection?

        // =====================================================================
        // Factory Methods
        // =====================================================================

        /**
         * @brief Derive shard info from a leaf node directly
         *
         * Simple case: uses the leaf's layer range directly, assumes no nested TP.
         * tp_shard_index and tp_total_shards remain at defaults (0, 1).
         *
         * @param leaf The DEVICE leaf node (must have layers assigned)
         * @return CompoundShardInfo with layer range from the leaf
         */
        static CompoundShardInfo fromLeaf(const ParallelismNode &leaf);

        /**
         * @brief Derive shard info by walking ancestors from root to leaf
         *
         * Full case: walks the path from root to leaf, accumulating TP degrees.
         * For each TP ancestor, multiplies tp_total_shards and computes the
         * leaf's position within that TP level.
         *
         * @param leaf The DEVICE leaf node
         * @param tree The complete tree (for walking ancestors)
         * @return CompoundShardInfo with compound TP indices
         */
        static CompoundShardInfo fromTreePath(const ParallelismNode &leaf,
                                              const ParallelismTree &tree);

        // =====================================================================
        // Queries
        // =====================================================================

        /**
         * @brief Number of layers this device processes
         * @return layer_last - layer_first + 1, or 0 if not assigned
         */
        int layerCount() const
        {
            return (layer_first >= 0 && layer_last >= layer_first)
                       ? (layer_last - layer_first + 1)
                       : 0;
        }

        /**
         * @brief Is this device in a tensor-parallel group?
         * @return true if tp_total_shards > 1
         */
        bool isSharded() const { return tp_total_shards > 1; }

        /**
         * @brief Validate consistency of shard info
         *
         * Checks:
         * - layer_first <= layer_last (if assigned)
         * - tp_shard_index in [0, tp_total_shards)
         * - tp_total_shards >= 1
         * - work_fraction in (0.0, 1.0]
         *
         * @return List of error messages (empty = valid)
         */
        std::vector<std::string> validate() const;

        // =====================================================================
        // Serialization
        // =====================================================================

        /**
         * @brief Human-readable string representation
         * @return Multi-line description of the shard info
         */
        std::string toString() const;

        // =====================================================================
        // Comparison
        // =====================================================================

        /**
         * @brief Equality comparison for testing
         */
        bool operator==(const CompoundShardInfo &other) const;
        bool operator!=(const CompoundShardInfo &other) const { return !(*this == other); }
    };

} // namespace llaminar2

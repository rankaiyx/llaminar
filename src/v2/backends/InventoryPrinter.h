/**
 * @file InventoryPrinter.h
 * @brief Renders hardware inventory tables (CPU, GPU, P2P) using libfort.
 *
 * Operates on ClusterInventory / RankInventory from DeviceInventory.h.
 * Designed to be called once by rank 0 after the AllGather exchange,
 * or locally at startup for single-process mode.
 *
 * @author David Sanftenberg
 * @date 2026-04-11
 */

#pragma once

namespace llaminar2
{
    struct ClusterInventory;
    struct RankInventory;

    /**
     * @brief Print hardware inventory tables via LOG_INFO.
     *
     * For a multi-node cluster, prints one set of tables per unique node.
     * For single-process mode, accepts a single RankInventory.
     */
    namespace InventoryPrinter
    {
        /// Print full cluster inventory (all nodes, called by rank 0)
        void printClusterInventory(const ClusterInventory &inventory);

        /// Print a single rank's inventory (single-process or local mode)
        void printRankInventory(const RankInventory &inventory);
    }

} // namespace llaminar2

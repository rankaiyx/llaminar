/**
 * @file NodeDetection.h
 * @brief Canonical hostname-based MPI node detection
 *
 * Single source of truth for mapping MPI ranks to physical nodes.
 * Uses MPI_Get_processor_name() + AllGather to group ranks by hostname,
 * then assigns sequential node IDs (0, 1, 2, ...) in order of first
 * appearance.
 *
 * When an MPI hostfile is provided (via --mpi-hostfile), the hostfile's
 * hostname ordering determines node IDs, ensuring consistency between
 * the planned topology and runtime detection.
 *
 * This replaces the formula-based approach (rank / ranks_per_node) which
 * assumes contiguous rank placement and breaks with non-round-robin layouts.
 *
 * Usage sites:
 * - MPITopology::detect_topology()
 * - GlobalTPContext::detectNodeIds()
 * - OrchestrationRunner::gatherClusterInventory()
 */

#pragma once

#include <mpi.h>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Result of hostname-based node detection across MPI ranks.
     */
    struct NodeDetectionResult
    {
        /// Node ID for each rank (indexed by rank position in the communicator).
        /// Same hostname → same node_id. IDs are sequential starting from 0.
        std::vector<int> node_ids;

        /// Number of unique nodes detected.
        int node_count = 0;

        /// Hostname for each rank (indexed by rank position).
        std::vector<std::string> hostnames;
    };

    /**
     * @brief Canonical hostname-based node detection.
     *
     * All node identification in Llaminar MUST go through this class
     * to ensure a single consistent formula.
     */
    class NodeDetection
    {
    public:
        /**
         * @brief Detect node placement by gathering hostnames over an MPI communicator.
         *
         * Every rank in @p comm must call this collectively.
         * Uses MPI_Get_processor_name() for the local hostname, then
         * MPI_Allgather to exchange hostnames across all ranks.
         *
         * If @p hostfile_path is non-empty, the hostfile's hostname ordering
         * determines node IDs (hostfile entry order = node ID order). MPI
         * hostnames not found in the hostfile get new sequential IDs after
         * the hostfile entries.
         *
         * @param comm           MPI communicator (e.g. MPI_COMM_WORLD or a domain comm)
         * @param hostfile_path  Optional path to an MPI hostfile for node ordering
         * @return NodeDetectionResult with per-rank node IDs and hostnames
         */
        static NodeDetectionResult detect(MPI_Comm comm,
                                          const std::string &hostfile_path = "");

        /**
         * @brief Assign node IDs from a pre-collected list of hostnames.
         *
         * Non-collective. Useful when hostnames have already been gathered
         * (e.g. from a deserialized ClusterInventory).
         *
         * @param hostnames  One hostname per rank, in rank order
         * @return NodeDetectionResult with per-rank node IDs (hostnames copied in)
         */
        static NodeDetectionResult fromHostnames(const std::vector<std::string> &hostnames);

        /**
         * @brief Assign node IDs from hostnames, using hostfile for node ordering.
         *
         * Non-collective. Hostfile hostnames define the authoritative node ID
         * ordering. MPI hostnames not found in the hostfile get new sequential
         * IDs after the hostfile entries.
         *
         * @param hostnames       One hostname per rank, in rank order
         * @param hostfile_path   Path to an MPI hostfile for node ordering
         * @return NodeDetectionResult with per-rank node IDs
         */
        static NodeDetectionResult fromHostnames(
            const std::vector<std::string> &hostnames,
            const std::string &hostfile_path);

        /**
         * @brief Parse an MPI hostfile and build a hostname→node_id mapping.
         *
         * Reads an OpenMPI-style hostfile (one hostname per line, optional
         * "slots=N"). Each unique hostname in file order gets a sequential
         * node ID. The returned vector maps hostname→node_id.
         *
         * @param hostfile_path  Path to the hostfile
         * @return Vector of (hostname, node_id) pairs in file order (unique hostnames only)
         */
        static std::vector<std::pair<std::string, int>> parseHostfile(
            const std::string &hostfile_path);

    private:
        /**
         * @brief Assign node IDs using an optional hostfile node map.
         *
         * If @p hostfile_nodes is non-empty, hostfile entries define the
         * authoritative hostname→node_id ordering. MPI hostnames not in
         * the hostfile get new sequential IDs.
         *
         * If @p hostfile_nodes is empty, behaves identically to first-appearance
         * ordering (the default fromHostnames behavior).
         */
        static NodeDetectionResult fromHostnamesWithNodeMap(
            const std::vector<std::string> &hostnames,
            const std::vector<std::pair<std::string, int>> &hostfile_nodes);
    };

} // namespace llaminar2

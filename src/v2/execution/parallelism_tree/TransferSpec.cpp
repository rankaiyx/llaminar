/**
 * @file TransferSpec.cpp
 * @brief Implementation of transfer mechanism derivation
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "TransferSpec.h"
#include <algorithm>
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // TransferSpec — derive()
    // =========================================================================

    TransferSpec TransferSpec::derive(const ParallelismNode &from,
                                      const ParallelismNode &to,
                                      int tag_base)
    {
        TransferSpec spec;
        spec.mpi_tag = tag_base;

        // Collect leaf ranks from both subtrees
        auto from_ranks = from.leafRanks();
        auto to_ranks = to.leafRanks();

        // Check for rank intersection (same-rank transfer)
        std::set<int> intersection;
        std::set_intersection(from_ranks.begin(), from_ranks.end(),
                              to_ranks.begin(), to_ranks.end(),
                              std::inserter(intersection, intersection.begin()));

        if (!intersection.empty())
        {
            // LOCAL_PP: same-rank transfer
            spec.mechanism = Mechanism::LOCAL_PP;
            spec.sender_rank = -1;
            spec.receiver_rank = -1;
            spec.local_backend = CollectiveBackendType::AUTO;
            return spec;
        }

        // Cross-rank transfer: determine sender and receiver
        // sender_rank = max of from's leaf ranks (deterministic choice)
        // receiver_rank = min of to's leaf ranks (deterministic choice)
        spec.sender_rank = *std::max_element(from_ranks.begin(), from_ranks.end());
        spec.receiver_rank = *std::min_element(to_ranks.begin(), to_ranks.end());

        // Collect hostnames from leaf devices
        auto from_leaves = from.leafDevices();
        auto to_leaves = to.leafDevices();

        std::set<std::string> from_hosts;
        for (const auto *leaf : from_leaves)
        {
            from_hosts.insert(leaf->device.hostname);
        }

        std::set<std::string> to_hosts;
        for (const auto *leaf : to_leaves)
        {
            to_hosts.insert(leaf->device.hostname);
        }

        // Check for hostname intersection (same machine)
        std::set<std::string> host_intersection;
        std::set_intersection(from_hosts.begin(), from_hosts.end(),
                              to_hosts.begin(), to_hosts.end(),
                              std::inserter(host_intersection, host_intersection.begin()));

        if (!host_intersection.empty())
        {
            // MPI_INTRAHOST: different ranks, same machine
            spec.mechanism = Mechanism::MPI_INTRAHOST;
        }
        else
        {
            // MPI_INTERHOST: different machines
            spec.mechanism = Mechanism::MPI_INTERHOST;
        }

        return spec;
    }

    // =========================================================================
    // TransferSpec — Serialization
    // =========================================================================

    const char *TransferSpec::mechanismName(Mechanism m)
    {
        switch (m)
        {
        case Mechanism::LOCAL_PP:
            return "LOCAL_PP";
        case Mechanism::MPI_INTRAHOST:
            return "MPI_INTRAHOST";
        case Mechanism::MPI_INTERHOST:
            return "MPI_INTERHOST";
        default:
            return "UNKNOWN";
        }
    }

    std::string TransferSpec::toString() const
    {
        std::ostringstream oss;
        oss << "TransferSpec {\n";
        oss << "  mechanism: " << mechanismName(mechanism) << "\n";

        if (mechanism == Mechanism::LOCAL_PP)
        {
            oss << "  local_backend: " << collectiveBackendTypeToString(local_backend) << "\n";
        }
        else
        {
            oss << "  sender_rank: " << sender_rank << "\n";
            oss << "  receiver_rank: " << receiver_rank << "\n";
            oss << "  mpi_tag: " << mpi_tag << "\n";
        }

        oss << "}";
        return oss.str();
    }

    // =========================================================================
    // TransferSpec — Comparison
    // =========================================================================

    bool TransferSpec::operator==(const TransferSpec &other) const
    {
        if (mechanism != other.mechanism)
            return false;

        if (mechanism == Mechanism::LOCAL_PP)
        {
            return local_backend == other.local_backend &&
                   mpi_tag == other.mpi_tag;
        }
        else
        {
            return sender_rank == other.sender_rank &&
                   receiver_rank == other.receiver_rank &&
                   mpi_tag == other.mpi_tag;
        }
    }

} // namespace llaminar2

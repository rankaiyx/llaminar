/**
 * @file NodeDetection.cpp
 * @brief Canonical hostname-based MPI node detection (implementation)
 */

#include "NodeDetection.h"
#include "Logger.h"
#include <cstring>
#include <fstream>
#include <sstream>

namespace llaminar2
{

    NodeDetectionResult NodeDetection::detect(MPI_Comm comm,
                                              const std::string &hostfile_path)
    {
        if (comm == MPI_COMM_NULL)
        {
            return {};
        }

        int comm_size = 0;
        int comm_rank = 0;
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &comm_rank);

        if (comm_size <= 0)
        {
            return {};
        }

        // Get this rank's hostname via the canonical MPI call
        constexpr int kMaxHostnameLen = 256;
        char my_hostname[kMaxHostnameLen];
        std::memset(my_hostname, 0, kMaxHostnameLen);

        int name_len = 0;
        MPI_Get_processor_name(my_hostname, &name_len);

        // AllGather hostnames across all ranks in the communicator
        std::vector<char> all_hostnames(
            static_cast<size_t>(comm_size) * kMaxHostnameLen, '\0');
        MPI_Allgather(my_hostname, kMaxHostnameLen, MPI_CHAR,
                      all_hostnames.data(), kMaxHostnameLen, MPI_CHAR,
                      comm);

        // Extract hostname strings
        std::vector<std::string> hostnames;
        hostnames.reserve(static_cast<size_t>(comm_size));
        for (int i = 0; i < comm_size; ++i)
        {
            const char *h = all_hostnames.data() +
                            static_cast<size_t>(i) * kMaxHostnameLen;
            hostnames.emplace_back(h);
        }

        // If a hostfile is provided, use it to determine node ID ordering
        NodeDetectionResult result;
        if (!hostfile_path.empty())
        {
            auto hostfile_nodes = parseHostfile(hostfile_path);
            if (!hostfile_nodes.empty())
            {
                result = fromHostnamesWithNodeMap(hostnames, hostfile_nodes);
                LOG_DEBUG("[NodeDetection] detect (hostfile): " << result.node_count
                                                                << " node(s) across " << comm_size << " ranks"
                                                                << " (hostfile=" << hostfile_path
                                                                << ", this rank hostname=\"" << hostnames[comm_rank] << "\")");
                return result;
            }
            // Hostfile parse failed or empty, fall through to auto-detect
            LOG_WARN("[NodeDetection] Hostfile '" << hostfile_path
                                                  << "' produced no entries, falling back to auto-detection");
        }

        // Delegate to the shared assignment logic
        result = fromHostnames(hostnames);

        LOG_DEBUG("[NodeDetection] detect: " << result.node_count
                                             << " unique node(s) across " << comm_size
                                             << " ranks (this rank hostname=\""
                                             << hostnames[comm_rank] << "\")");

        return result;
    }

    NodeDetectionResult NodeDetection::fromHostnames(
        const std::vector<std::string> &hostnames)
    {
        // No hostfile node map — assign by first appearance
        return fromHostnamesWithNodeMap(hostnames, {});
    }

    NodeDetectionResult NodeDetection::fromHostnames(
        const std::vector<std::string> &hostnames,
        const std::string &hostfile_path)
    {
        if (hostfile_path.empty())
        {
            return fromHostnames(hostnames);
        }

        auto hostfile_nodes = parseHostfile(hostfile_path);
        if (hostfile_nodes.empty())
        {
            LOG_WARN("[NodeDetection] Hostfile '" << hostfile_path
                                                  << "' produced no entries, falling back to first-appearance ordering");
            return fromHostnames(hostnames);
        }

        return fromHostnamesWithNodeMap(hostnames, hostfile_nodes);
    }

    NodeDetectionResult NodeDetection::fromHostnamesWithNodeMap(
        const std::vector<std::string> &hostnames,
        const std::vector<std::pair<std::string, int>> &hostfile_nodes)
    {
        NodeDetectionResult result;
        const int n = static_cast<int>(hostnames.size());
        result.node_ids.resize(static_cast<size_t>(n));
        result.hostnames = hostnames;

        // Build hostname→node_id lookup from hostfile if available
        // Hostfile entries define the authoritative node ID ordering
        int next_node_id = 0;
        std::vector<std::pair<std::string, int>> seen;

        if (!hostfile_nodes.empty())
        {
            // Seed with hostfile entries (they define the ordering)
            seen = hostfile_nodes;
            for (const auto &[name, nid] : seen)
            {
                if (nid >= next_node_id)
                {
                    next_node_id = nid + 1;
                }
            }
        }

        // Assign node IDs — hostfile entries take priority, then first-appearance
        for (int i = 0; i < n; ++i)
        {
            int assigned_id = -1;
            for (const auto &[name, nid] : seen)
            {
                if (name == hostnames[i])
                {
                    assigned_id = nid;
                    break;
                }
            }
            if (assigned_id < 0)
            {
                assigned_id = next_node_id++;
                seen.emplace_back(hostnames[i], assigned_id);
            }
            result.node_ids[i] = assigned_id;
        }

        result.node_count = next_node_id;
        return result;
    }

    std::vector<std::pair<std::string, int>> NodeDetection::parseHostfile(
        const std::string &hostfile_path)
    {
        std::vector<std::pair<std::string, int>> nodes;

        std::ifstream file(hostfile_path);
        if (!file.is_open())
        {
            LOG_WARN("[NodeDetection] Cannot open hostfile: " << hostfile_path);
            return nodes;
        }

        // Parse OpenMPI-style hostfile: one hostname per line, optional "slots=N"
        // Each unique hostname gets a sequential node ID in file order
        int next_node_id = 0;
        std::string line;
        while (std::getline(file, line))
        {
            // Skip empty lines and comments
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos || line[start] == '#')
            {
                continue;
            }

            // Extract hostname (first token)
            std::istringstream iss(line.substr(start));
            std::string hostname;
            iss >> hostname;

            if (hostname.empty())
            {
                continue;
            }

            // Check if we've already seen this hostname
            bool found = false;
            for (const auto &[name, nid] : nodes)
            {
                if (name == hostname)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                nodes.emplace_back(hostname, next_node_id++);
            }
        }

        LOG_DEBUG("[NodeDetection] parseHostfile: " << nodes.size()
                                                    << " unique node(s) from " << hostfile_path);
        return nodes;
    }

} // namespace llaminar2

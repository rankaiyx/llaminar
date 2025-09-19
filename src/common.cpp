#include "common.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <set>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <unistd.h>
#include <set>
#include <algorithm>

// Function to detect CPU topology using /proc/cpuinfo and /sys/devices/system/cpu
CPUTopology detectCPUTopology(bool use_hyperthreading = false)
{
    CPUTopology topo;
    topo.total_cpus = 0;
    topo.physical_cores = 0;
    topo.sockets = 0;
    topo.cores_per_socket = 0;
    topo.threads_per_core = 0;
    topo.use_hyperthreading = use_hyperthreading;
    topo.hyperthreading_detected = false;

    // Get total number of CPUs
    topo.total_cpus = sysconf(_SC_NPROCESSORS_ONLN);

    // Parse /proc/cpuinfo for detailed topology
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::map<int, std::map<std::string, std::string>> cpu_data;
    int current_processor = -1;

    while (std::getline(cpuinfo, line))
    {
        if (line.empty())
        {
            current_processor = -1;
            continue;
        }

        size_t colon = line.find(':');
        if (colon != std::string::npos)
        {
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);

            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            if (key == "processor")
            {
                current_processor = std::stoi(value);
                cpu_data[current_processor] = std::map<std::string, std::string>();
            }
            else if (current_processor != -1)
            {
                cpu_data[current_processor][key] = value;
            }
        }
    }

    // Analyze topology from collected data
    std::set<int> physical_ids;
    std::set<int> core_ids;
    std::map<std::pair<int, int>, std::vector<int>> socket_core_to_threads;

    for (const auto &cpu : cpu_data)
    {
        int cpu_id = cpu.first;
        const auto &data = cpu.second;

        int physical_id = -1;
        int core_id = -1;

        auto phys_it = data.find("physical id");
        auto core_it = data.find("core id");

        if (phys_it != data.end())
        {
            physical_id = std::stoi(phys_it->second);
            physical_ids.insert(physical_id);
            topo.cpu_to_socket[cpu_id] = physical_id;
            topo.socket_to_cpus[physical_id].push_back(cpu_id);
        }

        if (core_it != data.end())
        {
            core_id = std::stoi(core_it->second);
            core_ids.insert(core_id);
            topo.cpu_to_physical_core[cpu_id] = core_id;
        }

        if (physical_id != -1 && core_id != -1)
        {
            socket_core_to_threads[{physical_id, core_id}].push_back(cpu_id);
        }
    }

    topo.sockets = physical_ids.size();

    // Calculate physical cores and hyperthreading
    std::set<std::pair<int, int>> unique_physical_cores;
    for (const auto &entry : socket_core_to_threads)
    {
        unique_physical_cores.insert(entry.first);

        // Determine primary CPU for each physical core (lowest CPU ID)
        int socket = entry.first.first;
        const auto &thread_list = entry.second;

        if (!thread_list.empty())
        {
            int primary_cpu = *std::min_element(thread_list.begin(), thread_list.end());
            topo.socket_to_primary_cpus[socket].push_back(primary_cpu);
            topo.physical_core_ids.push_back(primary_cpu);
            topo.socket_to_physical_cores[socket].push_back(primary_cpu);
        }

        // Check for hyperthreading
        if (thread_list.size() > 1)
        {
            topo.hyperthreading_detected = true;
        }
    }

    topo.physical_cores = unique_physical_cores.size();
    topo.cores_per_socket = topo.sockets > 0 ? topo.physical_cores / topo.sockets : 0;
    topo.threads_per_core = topo.physical_cores > 0 ? topo.total_cpus / topo.physical_cores : 1;

    // Sort the primary CPU lists for consistent ordering
    for (auto &entry : topo.socket_to_primary_cpus)
    {
        std::sort(entry.second.begin(), entry.second.end());
    }
    for (auto &entry : topo.socket_to_physical_cores)
    {
        std::sort(entry.second.begin(), entry.second.end());
    }
    std::sort(topo.physical_core_ids.begin(), topo.physical_core_ids.end());

    return topo;
}

// Function to print CPU topology information
void printCPUTopology(const CPUTopology &topo)
{
    std::cout << "\n=== CPU Topology Information ===" << std::endl;
    std::cout << "Total CPUs: " << topo.total_cpus << std::endl;
    std::cout << "Physical cores: " << topo.physical_cores << std::endl;
    std::cout << "Sockets: " << topo.sockets << std::endl;
    std::cout << "Cores per socket: " << topo.cores_per_socket << std::endl;
    std::cout << "Threads per core: " << topo.threads_per_core << std::endl;
    std::cout << "Hyperthreading detected: " << (topo.hyperthreading_detected ? "Yes" : "No") << std::endl;
    std::cout << "Using hyperthreading: " << (topo.use_hyperthreading ? "Yes" : "No") << std::endl;

    if (topo.sockets > 0)
    {
        std::cout << "\nSocket distribution:" << std::endl;
        for (const auto &entry : topo.socket_to_cpus)
        {
            int socket = entry.first;
            const auto &cpus = entry.second;
            const auto &primaries = topo.socket_to_primary_cpus.at(socket);

            std::cout << "  Socket " << socket << ": " << cpus.size() << " CPUs";
            std::cout << " (Primary cores: ";
            for (size_t i = 0; i < primaries.size(); ++i)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << primaries[i];
            }
            std::cout << ")" << std::endl;
        }
        std::cout << "===============================" << std::endl;
    }
}
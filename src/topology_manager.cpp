#include "topology_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <set>
#include <algorithm>
#include <cstring>
#include <mpi.h>

#ifdef HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

// For CUDA detection
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
#endif

// For ROCm detection
#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>
#endif

namespace llaminar
{

    TopologyManager::TopologyManager()
    {
        // Initialize any required libraries
#ifdef ENABLE_CUDA
        // Initialize NVML for better GPU info
        nvmlInit();
#endif

#ifdef ENABLE_ROCM
        // Initialize ROCm SMI
        rsmi_init(0);
#endif
    }

    SystemTopology TopologyManager::detectSystemTopology(bool use_hyperthreading, bool detect_gpus)
    {
        SystemTopology topology;

        // Detect CPU topology
        topology.cpu = detectCPUTopology(use_hyperthreading);

        // Detect NUMA topology
        topology.numa = detectNUMATopology();

        // Detect GPU devices if requested
        if (detect_gpus)
        {
            topology.gpus = detectGPUDevices();
        }

        // Check MPI environment
        int provided;
        if (MPI_Query_thread(&provided) == MPI_SUCCESS)
        {
            topology.mpi_enabled = true;
            MPI_Comm_rank(MPI_COMM_WORLD, &topology.mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &topology.mpi_size);
        }
        else
        {
            topology.mpi_enabled = false;
            topology.mpi_rank = 0;
            topology.mpi_size = 1;
        }

        return topology;
    }

    CPUTopology TopologyManager::detectCPUTopology(bool use_hyperthreading)
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

        parseCPUInfo(topo);
        detectSocketTopology(topo);
        detectHyperthreading(topo);

        return topo;
    }

    NUMATopology TopologyManager::detectNUMATopology()
    {
        NUMATopology numa;
        numa.numa_available = false;
        numa.numa_nodes = 0;

#ifdef HAVE_NUMA
        if (numa_available() == 0)
        {
            numa.numa_available = true;
            numa.numa_nodes = numa_max_node() + 1;

            parseNUMANodes(numa);
            mapCPUsToNUMANodes(numa);
        }
#endif

        return numa;
    }

    std::vector<GPUDevice> TopologyManager::detectGPUDevices()
    {
        std::vector<GPUDevice> all_gpus;

        // Detect CUDA devices
        auto cuda_gpus = detectCUDADevices();
        all_gpus.insert(all_gpus.end(), cuda_gpus.begin(), cuda_gpus.end());

        // Detect ROCm devices
        auto rocm_gpus = detectROCmDevices();
        all_gpus.insert(all_gpus.end(), rocm_gpus.begin(), rocm_gpus.end());

        // Detect Intel GPU devices (future implementation)
        auto intel_gpus = detectIntelGPUDevices();
        all_gpus.insert(all_gpus.end(), intel_gpus.begin(), intel_gpus.end());

        return all_gpus;
    }

    void TopologyManager::parseCPUInfo(CPUTopology &topology)
    {
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
                topology.cpu_to_socket[cpu_id] = physical_id;
                topology.socket_to_cpus[physical_id].push_back(cpu_id);
            }

            if (core_it != data.end())
            {
                core_id = std::stoi(core_it->second);
                core_ids.insert(core_id);
                topology.cpu_to_physical_core[cpu_id] = core_id;
            }

            if (physical_id != -1 && core_id != -1)
            {
                socket_core_to_threads[{physical_id, core_id}].push_back(cpu_id);
            }
        }

        topology.sockets = physical_ids.size();
    }

    void TopologyManager::detectSocketTopology(CPUTopology &topology)
    {
        std::set<std::pair<int, int>> unique_physical_cores;
        std::map<std::pair<int, int>, std::vector<int>> socket_core_to_threads;

        // Rebuild socket_core_to_threads from existing data
        for (const auto &cpu_socket : topology.cpu_to_socket)
        {
            int cpu_id = cpu_socket.first;
            int socket_id = cpu_socket.second;

            auto core_it = topology.cpu_to_physical_core.find(cpu_id);
            if (core_it != topology.cpu_to_physical_core.end())
            {
                int core_id = core_it->second;
                socket_core_to_threads[{socket_id, core_id}].push_back(cpu_id);
            }
        }

        for (const auto &entry : socket_core_to_threads)
        {
            unique_physical_cores.insert(entry.first);

            // Determine primary CPU for each physical core (lowest CPU ID)
            int socket = entry.first.first;
            const auto &thread_list = entry.second;

            if (!thread_list.empty())
            {
                int primary_cpu = *std::min_element(thread_list.begin(), thread_list.end());
                topology.socket_to_primary_cpus[socket].push_back(primary_cpu);
                topology.physical_core_ids.push_back(primary_cpu);
                topology.socket_to_physical_cores[socket].push_back(primary_cpu);
            }
        }

        topology.physical_cores = unique_physical_cores.size();
        topology.cores_per_socket = topology.sockets > 0 ? topology.physical_cores / topology.sockets : 0;
        topology.threads_per_core = topology.physical_cores > 0 ? topology.total_cpus / topology.physical_cores : 1;

        // Sort the primary CPU lists for consistent ordering
        for (auto &entry : topology.socket_to_primary_cpus)
        {
            std::sort(entry.second.begin(), entry.second.end());
        }
        for (auto &entry : topology.socket_to_physical_cores)
        {
            std::sort(entry.second.begin(), entry.second.end());
        }
        std::sort(topology.physical_core_ids.begin(), topology.physical_core_ids.end());
    }

    void TopologyManager::detectHyperthreading(CPUTopology &topology)
    {
        // Check if we have more threads than physical cores
        if (topology.total_cpus > topology.physical_cores)
        {
            topology.hyperthreading_detected = true;
        }
    }

    void TopologyManager::parseNUMANodes(NUMATopology &numa)
    {
#ifdef HAVE_NUMA
        for (int node = 0; node < numa.numa_nodes; ++node)
        {
            if (numa_node_size64(node, nullptr) > 0)
            {
                numa.node_memory_gb[node] = numa_node_size64(node, nullptr) / (1024 * 1024 * 1024);
            }
        }
#endif
    }

    void TopologyManager::mapCPUsToNUMANodes(NUMATopology &numa)
    {
#ifdef HAVE_NUMA
        for (int cpu = 0; cpu < sysconf(_SC_NPROCESSORS_ONLN); ++cpu)
        {
            int node = numa_node_of_cpu(cpu);
            if (node >= 0)
            {
                numa.cpu_to_numa_node[cpu] = node;
                numa.node_to_cpus[node].push_back(cpu);
            }
        }
#endif
    }

    std::vector<GPUDevice> TopologyManager::detectCUDADevices()
    {
        std::vector<GPUDevice> cuda_devices;

#ifdef ENABLE_CUDA
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
        {
            for (int i = 0; i < device_count; ++i)
            {
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
                {
                    GPUDevice device;
                    device.device_id = i;
                    device.name = prop.name;
                    device.memory_gb = prop.totalGlobalMem / (1024 * 1024 * 1024);
                    device.compute_capability_major = prop.major;
                    device.compute_capability_minor = prop.minor;
                    device.vendor = GPUVendor::NVIDIA;
                    device.is_available = true;

                    cuda_devices.push_back(device);
                }
            }
        }
#endif

        return cuda_devices;
    }

    std::vector<GPUDevice> TopologyManager::detectROCmDevices()
    {
        std::vector<GPUDevice> rocm_devices;

#ifdef ENABLE_ROCM
        int device_count = 0;
        if (hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0)
        {
            for (int i = 0; i < device_count; ++i)
            {
                hipDeviceProp_t prop;
                if (hipGetDeviceProperties(&prop, i) == hipSuccess)
                {
                    GPUDevice device;
                    device.device_id = i;
                    device.name = prop.name;
                    device.memory_gb = prop.totalGlobalMem / (1024 * 1024 * 1024);
                    device.compute_capability_major = prop.major;
                    device.compute_capability_minor = prop.minor;
                    device.vendor = GPUVendor::AMD;
                    device.is_available = true;

                    rocm_devices.push_back(device);
                }
            }
        }
#endif

        return rocm_devices;
    }

    std::vector<GPUDevice> TopologyManager::detectIntelGPUDevices()
    {
        std::vector<GPUDevice> intel_devices;

        // TODO: Implement Intel GPU detection when support is added
        // This would use Intel's oneAPI Level Zero or similar

        return intel_devices;
    }

    void TopologyManager::printSystemTopology(const SystemTopology &topology)
    {
        std::cout << "\n=== System Topology Information ===" << std::endl;

        printCPUTopology(topology.cpu);
        printNUMATopology(topology.numa);
        printGPUDevices(topology.gpus);

        std::cout << "\n=== MPI Environment ===" << std::endl;
        std::cout << "MPI enabled: " << (topology.mpi_enabled ? "Yes" : "No") << std::endl;
        if (topology.mpi_enabled)
        {
            std::cout << "MPI rank: " << topology.mpi_rank << std::endl;
            std::cout << "MPI size: " << topology.mpi_size << std::endl;
        }
        std::cout << "=================================" << std::endl;
    }

    void TopologyManager::printCPUTopology(const CPUTopology &topology)
    {
        std::cout << "\n=== CPU Topology ===" << std::endl;
        std::cout << "Total CPUs: " << topology.total_cpus << std::endl;
        std::cout << "Physical cores: " << topology.physical_cores << std::endl;
        std::cout << "Sockets: " << topology.sockets << std::endl;
        std::cout << "Cores per socket: " << topology.cores_per_socket << std::endl;
        std::cout << "Threads per core: " << topology.threads_per_core << std::endl;
        std::cout << "Hyperthreading detected: " << (topology.hyperthreading_detected ? "Yes" : "No") << std::endl;
        std::cout << "Using hyperthreading: " << (topology.use_hyperthreading ? "Yes" : "No") << std::endl;

        if (topology.sockets > 0)
        {
            std::cout << "\nSocket distribution:" << std::endl;
            for (const auto &entry : topology.socket_to_cpus)
            {
                int socket = entry.first;
                const auto &cpus = entry.second;
                const auto &primaries = topology.socket_to_primary_cpus.at(socket);

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
        }
    }

    void TopologyManager::printNUMATopology(const NUMATopology &topology)
    {
        std::cout << "\n=== NUMA Topology ===" << std::endl;
        std::cout << "NUMA available: " << (topology.numa_available ? "Yes" : "No") << std::endl;

        if (topology.numa_available)
        {
            std::cout << "NUMA nodes: " << topology.numa_nodes << std::endl;

            for (const auto &entry : topology.node_to_cpus)
            {
                int node = entry.first;
                const auto &cpus = entry.second;

                std::cout << "  Node " << node << ": " << cpus.size() << " CPUs";
                if (topology.node_memory_gb.count(node))
                {
                    std::cout << ", " << topology.node_memory_gb.at(node) << " GB memory";
                }
                std::cout << std::endl;
            }
        }
    }

    void TopologyManager::printGPUDevices(const std::vector<GPUDevice> &gpus)
    {
        std::cout << "\n=== GPU Devices ===" << std::endl;

        if (gpus.empty())
        {
            std::cout << "No GPU devices detected" << std::endl;
            return;
        }

        std::cout << "Found " << gpus.size() << " GPU device(s):" << std::endl;

        for (const auto &gpu : gpus)
        {
            std::cout << "  GPU " << gpu.device_id << ": " << gpu.name;
            std::cout << " (" << gpu.memory_gb << " GB)";

            std::string vendor_str;
            switch (gpu.vendor)
            {
            case GPUVendor::NVIDIA:
                vendor_str = "NVIDIA CUDA";
                break;
            case GPUVendor::AMD:
                vendor_str = "AMD ROCm";
                break;
            case GPUVendor::INTEL:
                vendor_str = "Intel GPU";
                break;
            default:
                vendor_str = "Unknown";
                break;
            }

            std::cout << " [" << vendor_str << "]";
            std::cout << " CC: " << gpu.compute_capability_major << "." << gpu.compute_capability_minor;
            std::cout << std::endl;
        }
    }

    std::vector<int> TopologyManager::getOptimalCPUBinding(const SystemTopology &topology, int num_processes)
    {
        if (topology.numa.numa_available)
        {
            return optimizeCPUBindingForMemory(topology.cpu, topology.numa, num_processes);
        }
        else
        {
            return optimizeCPUBindingForPerformance(topology.cpu, num_processes);
        }
    }

    std::vector<int> TopologyManager::getOptimalGPUSelection(const std::vector<GPUDevice> &gpus, int num_devices)
    {
        std::vector<int> selected;

        // Simple selection: pick the first num_devices available GPUs
        // TODO: Implement more sophisticated selection based on memory, compute capability, etc.
        for (const auto &gpu : gpus)
        {
            if (gpu.is_available && selected.size() < static_cast<size_t>(num_devices))
            {
                selected.push_back(gpu.device_id);
            }
        }

        return selected;
    }

    std::vector<int> TopologyManager::optimizeCPUBindingForPerformance(const CPUTopology &topology, int num_processes)
    {
        std::vector<int> binding;

        // Use physical cores first, then hyperthreads if needed
        const auto &primary_cores = topology.physical_core_ids;

        for (int i = 0; i < num_processes && i < static_cast<int>(primary_cores.size()); ++i)
        {
            binding.push_back(primary_cores[i]);
        }

        // If we need more processes and hyperthreading is available
        if (num_processes > static_cast<int>(primary_cores.size()) && topology.hyperthreading_detected)
        {
            // Add all available CPUs
            for (const auto &socket_cpus : topology.socket_to_cpus)
            {
                for (int cpu : socket_cpus.second)
                {
                    if (binding.size() >= static_cast<size_t>(num_processes))
                        break;

                    // Skip if already in binding
                    if (std::find(binding.begin(), binding.end(), cpu) == binding.end())
                    {
                        binding.push_back(cpu);
                    }
                }
                if (binding.size() >= static_cast<size_t>(num_processes))
                    break;
            }
        }

        return binding;
    }

    std::vector<int> TopologyManager::optimizeCPUBindingForMemory(const CPUTopology &topology, const NUMATopology &numa, int num_processes)
    {
        std::vector<int> binding;

        // Distribute processes across NUMA nodes for better memory bandwidth
        for (int node = 0; node < numa.numa_nodes && binding.size() < static_cast<size_t>(num_processes); ++node)
        {
            auto node_it = numa.node_to_cpus.find(node);
            if (node_it != numa.node_to_cpus.end())
            {
                const auto &node_cpus = node_it->second;

                // Pick one CPU from this node
                if (!node_cpus.empty())
                {
                    binding.push_back(node_cpus[0]);
                }
            }
        }

        // If we need more processes, fill remaining slots
        if (binding.size() < static_cast<size_t>(num_processes))
        {
            auto remaining = optimizeCPUBindingForPerformance(topology, num_processes - binding.size());
            for (int cpu : remaining)
            {
                if (std::find(binding.begin(), binding.end(), cpu) == binding.end())
                {
                    binding.push_back(cpu);
                }
            }
        }

        return binding;
    }

    bool TopologyManager::isHyperthreadingAvailable(const CPUTopology &topology)
    {
        return topology.hyperthreading_detected;
    }

    bool TopologyManager::isNUMAAvailable(const NUMATopology &topology)
    {
        return topology.numa_available;
    }

    bool TopologyManager::areGPUsAvailable(const std::vector<GPUDevice> &gpus)
    {
        return !gpus.empty();
    }

} // namespace llaminar
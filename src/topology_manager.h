#pragma once

#include <string>
#include <vector>
#include <map>

namespace llaminar
{

    // GPU vendor enumeration
    enum class GPUVendor
    {
        NONE,
        NVIDIA, // CUDA
        AMD,    // ROCm
        INTEL   // Intel GPU (future)
    };

    // GPU device information
    struct GPUDevice
    {
        int device_id;
        std::string name;
        size_t memory_gb;
        int compute_capability_major;
        int compute_capability_minor;
        GPUVendor vendor;
        bool is_available;
    };

    // CPU topology structure
    struct CPUTopology
    {
        int total_cpus;
        int physical_cores;
        int sockets;
        int cores_per_socket;
        int threads_per_core;
        bool hyperthreading_detected;
        bool use_hyperthreading;
        std::vector<int> physical_core_ids;
        std::map<int, int> cpu_to_socket;
        std::map<int, int> cpu_to_physical_core;
        std::map<int, std::vector<int>> socket_to_cpus;
        std::map<int, std::vector<int>> socket_to_physical_cores;
        std::map<int, std::vector<int>> socket_to_primary_cpus; // socket_id -> primary CPU per core (no HT siblings)
    };

    // NUMA topology structure
    struct NUMATopology
    {
        bool numa_available;
        int numa_nodes;
        std::map<int, std::vector<int>> node_to_cpus;
        std::map<int, size_t> node_memory_gb;
        std::map<int, int> cpu_to_numa_node;
    };

    // Complete system topology
    struct SystemTopology
    {
        CPUTopology cpu;
        NUMATopology numa;
        std::vector<GPUDevice> gpus;
        bool mpi_enabled;
        int mpi_rank;
        int mpi_size;
    };

    /**
     * TopologyManager - Comprehensive system topology detection and management
     *
     * Handles detection of:
     * - CPU topology (cores, sockets, hyperthreading)
     * - NUMA memory topology
     * - GPU devices (NVIDIA CUDA, AMD ROCm, Intel)
     * - MPI environment
     *
     * Provides optimization recommendations for:
     * - Process/thread binding
     * - Memory allocation strategies
     * - GPU device selection
     */
    class TopologyManager
    {
    public:
        TopologyManager();
        ~TopologyManager() = default;

        // Main detection methods
        SystemTopology detectSystemTopology(bool use_hyperthreading = false, bool detect_gpus = true);

        // Individual topology detection
        CPUTopology detectCPUTopology(bool use_hyperthreading = false);
        NUMATopology detectNUMATopology();
        std::vector<GPUDevice> detectGPUDevices();

        // Topology information display
        void printSystemTopology(const SystemTopology &topology);
        void printCPUTopology(const CPUTopology &topology);
        void printNUMATopology(const NUMATopology &topology);
        void printGPUDevices(const std::vector<GPUDevice> &gpus);

        // Optimization recommendations
        std::vector<int> getOptimalCPUBinding(const SystemTopology &topology, int num_processes = 1);
        std::vector<int> getOptimalGPUSelection(const std::vector<GPUDevice> &gpus, int num_devices = 1);

        // Utility methods
        bool isHyperthreadingAvailable(const CPUTopology &topology);
        bool isNUMAAvailable(const NUMATopology &topology);
        bool areGPUsAvailable(const std::vector<GPUDevice> &gpus);

    private:
        // CPU detection helpers
        void parseCPUInfo(CPUTopology &topology);
        void detectSocketTopology(CPUTopology &topology);
        void detectHyperthreading(CPUTopology &topology);

        // NUMA detection helpers
        void parseNUMANodes(NUMATopology &topology);
        void mapCPUsToNUMANodes(NUMATopology &topology);

        // GPU detection helpers
        std::vector<GPUDevice> detectCUDADevices();
        std::vector<GPUDevice> detectROCmDevices();
        std::vector<GPUDevice> detectIntelGPUDevices();

        // Optimization helpers
        std::vector<int> optimizeCPUBindingForPerformance(const CPUTopology &topology, int num_processes);
        std::vector<int> optimizeCPUBindingForMemory(const CPUTopology &topology, const NUMATopology &numa, int num_processes);
    };

} // namespace llaminar
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unistd.h>

// Function to get NUMA node for current process
int getCurrentNumaNode() {
    // Try to read the NUMA node from /proc/self/numa_maps
    std::ifstream numa_file("/proc/self/numa_maps");
    if (!numa_file.is_open()) {
        return -1; // NUMA info not available
    }
    
    // For simplicity, we'll use the CPU affinity to determine NUMA node
    // Read from /sys/devices/system/node/node*/cpulist
    for (int node = 0; node < 8; ++node) { // Check up to 8 NUMA nodes
        std::string cpu_list_path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
        std::ifstream cpu_file(cpu_list_path);
        if (cpu_file.is_open()) {
            // Get current CPU
            int current_cpu = sched_getcpu();
            if (current_cpu >= 0) {
                std::string cpu_range;
                std::getline(cpu_file, cpu_range);
                
                // Parse CPU range (e.g., "0-27,56-83")
                std::istringstream ss(cpu_range);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    if (token.find('-') != std::string::npos) {
                        // Range format: start-end
                        size_t dash_pos = token.find('-');
                        int start = std::stoi(token.substr(0, dash_pos));
                        int end = std::stoi(token.substr(dash_pos + 1));
                        if (current_cpu >= start && current_cpu <= end) {
                            return node;
                        }
                    } else {
                        // Single CPU
                        int cpu = std::stoi(token);
                        if (current_cpu == cpu) {
                            return node;
                        }
                    }
                }
            }
        }
    }
    return -1; // Could not determine NUMA node
}

// Function to run basic MPI test
void runBasicTest(int rank, int size) {
    std::cout << "Hello from Llaminar! Process " << rank
              << " of " << size << " processes." << std::endl;
}

// Function to run NUMA-aware MPI test
void runNumaTest(int rank, int size) {
    int numa_node = getCurrentNumaNode();
    int current_cpu = sched_getcpu();
    
    std::cout << "Process " << rank << "/" << size 
              << " running on CPU " << current_cpu
              << " (NUMA node " << numa_node << ")" << std::endl;
    
    // Collect NUMA distribution information
    std::vector<int> numa_nodes(size);
    MPI_Allgather(&numa_node, 1, MPI_INT, numa_nodes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Count processes per NUMA node
        std::vector<int> numa_counts(8, 0); // Support up to 8 NUMA nodes
        int unique_numa_nodes = 0;
        
        for (int i = 0; i < size; ++i) {
            if (numa_nodes[i] >= 0 && numa_nodes[i] < 8) {
                if (numa_counts[numa_nodes[i]] == 0) {
                    unique_numa_nodes++;
                }
                numa_counts[numa_nodes[i]]++;
            }
        }
        
        std::cout << "\n=== NUMA Distribution Summary ===" << std::endl;
        std::cout << "Total MPI processes: " << size << std::endl;
        std::cout << "NUMA nodes utilized: " << unique_numa_nodes << std::endl;
        
        for (int i = 0; i < 8; ++i) {
            if (numa_counts[i] > 0) {
                std::cout << "NUMA node " << i << ": " << numa_counts[i] << " processes" << std::endl;
            }
        }
        
        // Check if we have good NUMA distribution
        bool good_distribution = (unique_numa_nodes >= 2) && (size >= unique_numa_nodes);
        if (good_distribution) {
            std::cout << "\n✓ NUMA MPI TEST SUCCESS: Good distribution across NUMA nodes" << std::endl;
        } else {
            std::cout << "\n⚠ NUMA MPI TEST WARNING: Suboptimal NUMA distribution" << std::endl;
        }
        std::cout << "===================================" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check for test mode arguments
    std::string test_mode = "";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test-mode" && i + 1 < argc) {
            test_mode = argv[i + 1];
            break;
        }
    }

    if (test_mode == "basic") {
        runBasicTest(rank, size);
    } else if (test_mode == "numa") {
        runNumaTest(rank, size);
    } else {
        // Default behavior - basic hello world
        runBasicTest(rank, size);
        
        // TODO: Implement LLM inferencing engine with COSMA integration
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
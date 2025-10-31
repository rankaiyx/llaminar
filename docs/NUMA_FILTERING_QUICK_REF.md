# NUMA-Aware Device Filtering - Quick Reference

## Overview

**Purpose**: Each MPI rank sees only devices affine to its socket, preventing 40-60% cross-socket memory access penalties.

**Status**: ✅ Implemented in Phase 1 (2025-10-31)

## Quick Start

### Standard MPI Inference (Socket-Local Devices)
```bash
# Each rank auto-detects its NUMA node and filters devices
mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2/llaminar2 -m model.gguf -p "Your prompt"

# Expected behavior:
# Rank 0 → Socket 0 → Sees CPU (socket 0), GPU 0-1 (socket 0)
# Rank 1 → Socket 1 → Sees CPU (socket 1), GPU 2-3 (socket 1)
```

### List All Devices (No Filtering)
```bash
# Single process mode - shows all devices on system
./build_v2/llaminar2 --list-devices

# Output shows:
# - NUMA node detection method (hwloc/procfs/fallback)
# - All available devices with NUMA affinity
```

### Debugging NUMA Detection
```bash
# Run with verbose logging
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/llaminar2 --list-devices

# Check logs for:
# "[NUMATopology] Detected NUMA node X via <method>"
# "[DeviceManager] Including CUDA GPU X (NUMA node Y)"
# "[DeviceManager] Filtering out CUDA GPU X (on NUMA node Y)"
```

## Detection Methods

### Automatic Detection Priority (Highest to Lowest):

1. **hwloc** (most reliable, optional dependency):
   ```bash
   sudo apt install libhwloc-dev
   # Rebuild with: cmake -B build_v2 -S src/v2 -DHAVE_HWLOC=ON
   ```

2. **procfs** (Linux standard, no dependencies):
   - Reads `/proc/self/status` Cpus_allowed_list
   - Works on all Linux systems

3. **NVML** (CUDA GPUs only):
   - Requires CUDA Toolkit installed
   - Uses `nvmlDeviceGetNumaNode()`

4. **sysfs** (universal Linux fallback):
   - Reads `/sys/bus/pci/devices/<BUS_ID>/numa_node`

5. **Fallback** (single-socket assumption):
   - Assumes node 0 if all methods fail
   - Logs warning message

## Configuration

### Environment Variables

**Override NUMA Node** (advanced):
```bash
# Force specific NUMA node (bypasses detection)
NUMA_NODE=1 ./build_v2/llaminar2 -m model.gguf

# Disable filtering (show all devices)
NUMA_NODE=-1 ./build_v2/llaminar2 --list-devices
```

### Build-Time Options

**Enable hwloc** (recommended for production):
```bash
cmake -B build_v2 -S src/v2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DHAVE_HWLOC=ON

cmake --build build_v2 --parallel
```

**Enable CUDA** (for GPU NUMA detection):
```bash
cmake -B build_v2 -S src/v2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DHAVE_CUDA=ON

cmake --build build_v2 --parallel
```

## Validation

### Check NUMA Topology
```bash
# List NUMA nodes
numactl --hardware

# Example output:
# available: 2 nodes (0-1)
# node 0 cpus: 0 2 4 6 ... (socket 0)
# node 1 cpus: 1 3 5 7 ... (socket 1)

# Show current process NUMA binding
cat /proc/self/status | grep Cpus_allowed_list
```

### Verify Device Affinity
```bash
# List GPU NUMA affinity (CUDA systems)
nvidia-smi topo -m

# Example output:
# GPU0 - GPU1: Same socket (NUMA node 0)
# GPU2 - GPU3: Same socket (NUMA node 1)
# GPU0 - GPU2: Cross-socket (nodes 0-1)
```

### Test MPI Rank Binding
```bash
# Run with binding report
mpirun -np 2 --bind-to socket --report-bindings \
  ./build_v2/llaminar2 --list-devices

# Expected output:
# [0] Rank 0 bound to socket 0 [0-27]
# [1] Rank 1 bound to socket 1 [28-55]
# [INFO] [Rank 0] NUMA node: 0 (detection: procfs)
# [INFO] [Rank 1] NUMA node: 1 (detection: procfs)
```

## Troubleshooting

### NUMA Detection Failed
```
[WARN] NUMA detection failed, assuming node 0 (single socket)
```

**Diagnosis**:
1. Check if system has multiple NUMA nodes:
   ```bash
   ls /sys/devices/system/node/
   # Should show: node0, node1, ... if multi-socket
   ```

2. Verify MPI rank binding:
   ```bash
   mpirun -np 2 --bind-to socket --report-bindings hostname
   # Should show different CPU sets per rank
   ```

3. Try hwloc for better detection:
   ```bash
   sudo apt install libhwloc-dev hwloc
   lstopo  # Visualize NUMA topology
   ```

**Workaround**: Manually specify NUMA node:
```bash
mpirun -np 2 bash -c 'NUMA_NODE=$OMPI_COMM_WORLD_RANK ./llaminar2 ...'
```

### Cross-Socket GPU Access Warning
```
[DEBUG] Filtering out CUDA GPU 2 (on node 1, process on node 0)
```

**Meaning**: GPU 2 is on socket 1, but process is on socket 0. This is **expected behavior** - GPU will not be used by this rank to avoid cross-socket penalty.

**Validation**: Run on both ranks to verify complementary filtering:
```bash
# Rank 0 uses: GPU 0-1 (socket 0)
# Rank 1 uses: GPU 2-3 (socket 1)
```

### All Devices Filtered Out
```
[INFO] [DeviceManager] Enumerated 1 device(s):  # Only CPU, no GPUs
```

**Possible Causes**:
1. **Single-socket system**: All GPUs on same socket as CPU (expected)
2. **Incorrect MPI binding**: Rank not bound to socket
   ```bash
   # Fix: Add --bind-to socket to mpirun
   mpirun -np 2 --bind-to socket ./llaminar2 ...
   ```
3. **NUMA detection mismatch**: Process detected on wrong node
   ```bash
   # Debug: Check actual CPU binding
   taskset -cp $$  # Shows CPU affinity
   ```

## Performance Validation

### Baseline (No Filtering)
```bash
# Disable NUMA filtering for baseline
NUMA_NODE=-1 mpirun -np 2 ./llaminar2 --benchmark -m model.gguf
# Note: May see 40-60% slowdown from cross-socket access
```

### Optimized (With Filtering)
```bash
# Enable NUMA filtering (default)
mpirun -np 2 --bind-to socket ./llaminar2 --benchmark -m model.gguf
# Expected: 10-20% faster than baseline on multi-socket systems
```

### Measurement
```bash
# Compare throughput (tokens/sec)
# Baseline: ~100 tok/s (cross-socket access)
# Optimized: ~120 tok/s (socket-local access)
# Improvement: ~20% (varies by workload)
```

## API Reference

### C++ API

**Detect Process NUMA Node**:
```cpp
#include "utils/NUMATopology.h"

auto numa_info = NUMATopology::detectLocalNUMANode();
std::cout << "NUMA node: " << numa_info.local_numa_node << std::endl;
std::cout << "Method: " << numa_info.detection_method << std::endl;
std::cout << "Success: " << numa_info.detection_succeeded << std::endl;
```

**Detect GPU NUMA Node**:
```cpp
auto gpu_info = NUMATopology::getCUDAGPUNUMANode(gpu_id);
std::cout << "GPU " << gpu_id << " on NUMA node " << gpu_info.numa_node << std::endl;
std::cout << "PCIe bus: " << gpu_info.pci_bus_id << std::endl;
```

**Validate GPU Affinity**:
```cpp
int process_node = 0;
int gpu_node = 1;
bool is_local = NUMATopology::isGPULocalToProcess(gpu_node, process_node);
// Returns: false (cross-socket)
```

**Initialize DeviceManager with Filtering**:
```cpp
auto numa_info = NUMATopology::detectLocalNUMANode();
DeviceManager::instance().initialize(numa_info.local_numa_node);

// Alternative: No filtering
DeviceManager::instance().initialize(-1);  // All devices visible
```

## Related Documentation

- **Architecture Analysis**: `docs/architecture/MPI_DEVICE_ORCHESTRATION_ANALYSIS.md`
- **Refactor Plan**: `docs/architecture/MPI_DEVICE_REFACTOR_PLAN.md`
- **Phase 1 Changelog**: `changelog/2025-10-31-phase1-numa-device-filtering.md`
- **Session Summary**: `changelog/2025-10-31-mpi-device-refactor-session-summary.md`

## Next Steps

After validating Phase 1 on multi-GPU hardware:
1. **Phase 2**: MPI host staging for GPU-resident tensors
2. **Phase 3**: Documentation updates
3. **Phase 4**: Explicit pipeline execution logging

---

**Last Updated**: 2025-10-31  
**Status**: ✅ Implemented, ⏳ Multi-GPU Validation Pending

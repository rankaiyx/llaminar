# MPI + Multi-Device Architecture Analysis & Clarification Proposal

**Date**: October 31, 2025  
**Status**: Architecture Gap Analysis  
**Priority**: HIGH - Foundational for heterogeneous execution

---

## Executive Summary

**Problem**: The current V2 architecture has ambiguity around how MPI ranks coordinate with heterogeneous device execution. We create one pipeline per MPI rank, but the relationship between:
1. **MPI rank-level parallelism** (socket-to-socket distribution)
2. **Intra-rank device parallelism** (CPU + multiple GPUs per socket)

is poorly defined, creating confusion about work distribution within each rank.

**Proposed Solution**: Establish a clear **3-tier hierarchy**:
- **Tier 1 (MPI)**: Socket-level parallelism (1 rank per socket)
- **Tier 2 (Pipeline)**: Rank-local orchestration (1 pipeline per rank)
- **Tier 3 (Device)**: Layer-level device affinity within rank

---

## Current Architecture (As Implemented)

### What's Clear

#### 1. MPI Initialization (Main.cpp)
```cpp
// ONE MPI rank per process
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
auto mpi_ctx = MPIContextFactory::global();  // Wraps MPI_COMM_WORLD

// ONE pipeline per rank
auto pipeline = PipelineFactory::create(architecture, model_ctx, mpi_ctx, device_idx, placement_map, config);
```

**Fact**: We create exactly 1 pipeline instance per MPI rank.

#### 2. Device Enumeration (DeviceManager)
```cpp
DeviceManager::initialize() {
    // Enumerates ALL devices visible to this process:
    devices_.push_back({CPU_OPENBLAS, -1, ...});          // CPU always idx 0
    devices_.push_back({GPU_CUDA, 0, "NVIDIA A100", ...}); // GPU 0 (if present)
    devices_.push_back({GPU_CUDA, 1, "NVIDIA A100", ...}); // GPU 1 (if present)
    // etc.
}
```

**Fact**: DeviceManager sees ALL devices on the node, not just those affine to this rank's socket.

#### 3. Weight Placement (DeviceOrchestrator)
```cpp
// Creates placement map: weight_name → device_idx
auto placement_map = orchestrator->createPlacementMap(model_ctx);

// Strategies:
// - ALL_GPU: Everything on device 0
// - LAYER_SPLIT: Layers 0-N on GPU, rest on CPU
// - AUTO: Fit what we can on GPU based on memory
```

**Fact**: PlacementMap is **rank-local** - each rank has its own map assigning weights to devices visible to that rank.

#### 4. Pipeline Execution (Qwen2Pipeline::attention_block)
```cpp
int attn_device = placement_map_->getWeightDevice("attn_q", layer_idx);
TensorBase* input = prepareActivationForDevice(current_hidden_, attn_device, "attn_input");
auto& buffers = getBuffersForDevice(attn_device);
// Execute attention on attn_device...
```

**Fact**: Each **layer** can execute on a different device within the rank.

---

## What's Ambiguous/Broken

### Problem 1: Device Enumeration Doesn't Respect NUMA/Socket Affinity

**Current Behavior**:
```cpp
// Rank 0 (socket 0) sees:
devices_ = [CPU, GPU:0, GPU:1, GPU:2, GPU:3]  // All 4 GPUs visible

// Rank 1 (socket 1) sees:
devices_ = [CPU, GPU:0, GPU:1, GPU:2, GPU:3]  // Same 4 GPUs visible!
```

**Problem**: Both ranks see all GPUs, but GPUs 0-1 are PCIe-attached to socket 0, GPUs 2-3 to socket 1. If Rank 1 uses GPU:0, it crosses QPI/UPI with severe performance penalty.

**Expected Behavior**:
```cpp
// Rank 0 (socket 0) should see:
devices_ = [CPU:0, GPU:0, GPU:1]  // Only socket-local devices

// Rank 1 (socket 1) should see:
devices_ = [CPU:1, GPU:2, GPU:3]  // Only socket-local devices
```

### Problem 2: No Explicit Rank-to-Socket Binding

**Current**: We assume `mpirun --bind-to socket` pins rank N to socket N, but DeviceManager doesn't know which socket it's on.

**Missing**:
```cpp
class DeviceManager {
    int local_socket_id_;  // Which NUMA node/socket am I on?
    
    void initialize(int numa_node) {
        // Only enumerate devices affine to numa_node
    }
};
```

### Problem 3: Intra-Rank Work Distribution is Implicit

**Current**: `PlacementMap` assigns layers to devices, but there's no explicit work scheduler within a rank.

**Example Ambiguity**:
```
Rank 0 placement:
  Layer 0-11: GPU:0
  Layer 12-23: GPU:1
  Layer 24-31: CPU
```

**Questions**:
1. Does the pipeline execute layers **sequentially** (L0 on GPU:0, wait, L12 on GPU:1, wait...)?
2. Or can we execute L0 on GPU:0 **concurrently** with L12 on GPU:1 for different tokens in a batch?
3. Who manages device synchronization between layers?

**Current Answer**: Sequential execution only (no intra-rank parallelism).

### Problem 4: MPI Gather Assumes Homogeneous Output Locations

**Current** (GQAAttention MPI path):
```cpp
// After local computation on rank-local device
MPI_Allgather(local_output, global_output, ...);
```

**Problem**: If Rank 0's output is on GPU:0 and Rank 1's output is on GPU:2, we need:
1. GPU:0 → Host copy (Rank 0)
2. MPI_Allgather (host-to-host)
3. Host → GPU:0 copy (Rank 0)

**Missing**: Explicit device → host → MPI → host → device transfer pipeline.

---

## Proposed Architecture Clarification

### Tier 1: MPI Socket Parallelism

**Principle**: **One MPI rank per socket**, with strict socket affinity.

```cpp
// mpirun binding (external to code)
mpirun -np 2 --bind-to socket --map-by socket \
  --report-bindings ./llaminar2 ...

// Results in:
// Rank 0 → CPU cores 0-27  (socket 0)
// Rank 1 → CPU cores 28-55 (socket 1)
```

**Code Changes**:
```cpp
// Main.cpp: Detect NUMA node for this rank
int numa_node = detectLocalNUMANode();  // hwloc or /proc/self/status
auto& dm = DeviceManager::instance();
dm.initialize(numa_node);  // Only enumerate socket-local devices
```

**DeviceManager Enhancement**:
```cpp
class DeviceManager {
public:
    void initialize(int numa_node = -1) {  // -1 = all devices (testing)
        local_numa_node_ = (numa_node >= 0) ? numa_node : 0;
        
        // Enumerate CPU (always socket-local)
        devices_.push_back(createCPUDevice(local_numa_node_));
        
        // Enumerate GPUs affine to this NUMA node
        #ifdef HAVE_CUDA
        for (int i = 0; i < cudaGetDeviceCount(); ++i) {
            int gpu_numa_node = getGPUNUMANode(i);  // Via NVML
            if (gpu_numa_node == local_numa_node_) {
                devices_.push_back(createCUDADevice(i));
            }
        }
        #endif
    }

private:
    int local_numa_node_ = -1;  // Which socket are we on?
};
```

**Result**: Each rank sees only its socket-local devices.

---

### Tier 2: Pipeline Orchestration (Rank-Local)

**Principle**: **One pipeline per rank**, orchestrates work across rank-local devices.

**Execution Model**: **Sequential layer execution** (for Phase 1), with explicit device placement per layer.

```cpp
// PipelineBase::forward()
for (int layer = 0; layer < n_layers_; ++layer) {
    // Determine execution device for this layer
    int attn_device = getAttentionDevice(layer);
    int ffn_device = getFFNDevice(layer);
    
    // Prepare activation on attention device
    TensorBase* attn_input = prepareActivationForDevice(current_hidden_, attn_device, "attn");
    
    // Execute attention
    if (!attention_block(layer, attn_input, attn_device)) return false;
    
    // Prepare activation on FFN device (may transfer)
    TensorBase* ffn_input = prepareActivationForDevice(attn_output_, ffn_device, "ffn");
    
    // Execute FFN
    if (!ffn_block(layer, ffn_input, ffn_device)) return false;
    
    // Output is now on ffn_device
    current_hidden_ = ffn_output_;
}
```

**Key Point**: Layers execute **sequentially** within a rank, but each layer can use a different device.

**Future (Phase 2)**: Pipeline parallelism across layers (L0 on GPU:0, L12 on GPU:1 concurrently).

---

### Tier 3: Device-Level Execution

**Principle**: Each layer's work executes entirely on **one device** (no mid-layer transfers).

```cpp
// Qwen2Pipeline::attention_block()
bool Qwen2Pipeline::attention_block(int layer_idx, TensorBase* input, int execution_device) {
    // Get device-local buffers
    auto& buffers = getBuffersForDevice(execution_device);
    
    // All work happens on execution_device:
    // 1. RMSNorm on execution_device
    // 2. Q/K/V projections on execution_device (weights already there)
    // 3. Attention on execution_device
    // 4. Output projection on execution_device
    
    // No inter-device transfers during layer execution
}
```

**Contract**: Once a layer starts on device D, all intermediate activations stay on D until layer completes.

---

### MPI Coordination with Heterogeneous Devices

**Challenge**: Ranks may have outputs on different devices (Rank 0 on GPU:0, Rank 1 on GPU:2).

**Solution**: **Explicit staging through host memory**:

```cpp
// GQAAttention::compute_tensor_parallel()
bool compute_with_mpi(MPIContext* mpi_ctx, int output_device) {
    // 1. Compute local heads (on output_device)
    auto local_output = compute_local_heads(output_device);
    
    // 2. Stage to host if on GPU
    std::vector<float> host_buffer;
    if (output_device >= 0) {  // GPU
        host_buffer.resize(local_output->size());
        cudaMemcpy(host_buffer.data(), local_output->data(), ...);
    } else {  // CPU
        host_buffer = local_output->to_vector();
    }
    
    // 3. MPI collective (always host-to-host)
    std::vector<float> global_buffer(n_heads * output_size);
    mpi_ctx->allgather(host_buffer.data(), global_buffer.data(), local_size);
    
    // 4. Transfer back to device if needed
    if (output_device >= 0) {
        cudaMemcpy(final_output->data(), global_buffer.data(), ...);
    } else {
        std::memcpy(final_output->data(), global_buffer.data(), ...);
    }
    
    return true;
}
```

**Principle**: MPI always operates on **host memory**, with explicit device ↔ host transfers.

---

## Implementation Roadmap

### Phase 1: NUMA-Aware Device Enumeration (Immediate)

**Goal**: Each rank only sees socket-local devices.

**Files to Modify**:
1. `src/v2/backends/ComputeBackend.h`:
   ```cpp
   class DeviceManager {
   public:
       void initialize(int numa_node = -1);
       int local_numa_node() const { return local_numa_node_; }
   private:
       int local_numa_node_ = -1;
   };
   ```

2. `src/v2/backends/ComputeBackend.cpp`:
   - Add `detectLocalNUMANode()` using `hwloc` or `/proc/self/status`
   - Add `getGPUNUMANode(int gpu_id)` using NVML/ROCm APIs
   - Filter devices in `initialize()` by NUMA affinity

3. `src/v2/Main.cpp`:
   ```cpp
   int numa_node = detectLocalNUMANode();
   dm.initialize(numa_node);
   ```

**Validation**:
```bash
mpirun -np 2 --bind-to socket ./llaminar2 --list-devices

# Expected output:
# Rank 0:
#   Device 0: CPU (socket 0)
#   Device 1: GPU (CUDA:0) socket 0
#   Device 2: GPU (CUDA:1) socket 0

# Rank 1:
#   Device 0: CPU (socket 1)
#   Device 1: GPU (CUDA:2) socket 1
#   Device 2: GPU (CUDA:3) socket 1
```

---

### Phase 2: Explicit Device Staging for MPI (Short-term)

**Goal**: Add host memory staging to all MPI-aware kernels.

**Pattern**:
```cpp
template<typename T>
class MPIStager {
public:
    // Transfer tensor to host (no-op if already on host)
    static std::vector<T> toHost(const TensorBase* tensor) {
        if (tensor->device_index() < 0) {
            return tensor->to_vector();  // Already on host
        }
        std::vector<T> host(tensor->size());
        tensor->copyToHost(host.data());
        return host;
    }
    
    // Transfer host data back to device
    static void toDevice(const std::vector<T>& host, TensorBase* tensor) {
        if (tensor->device_index() < 0) {
            std::memcpy(tensor->mutable_data(), host.data(), ...);
        } else {
            tensor->copyFromHost(host.data());
        }
    }
};
```

**Apply to**:
- `GQAAttention::compute_tensor_parallel()`
- Any MPI allreduce/allgather/broadcast operations
- Embedding broadcast (if on GPU)

---

### Phase 3: Document Device Execution Contract (Documentation)

**Create**: `docs/architecture/DEVICE_EXECUTION_MODEL.md`

**Contents**:
1. **Tier hierarchy** (MPI → Pipeline → Device)
2. **Device affinity rules**:
   - Weights: Static placement via `PlacementMap`
   - Activations: Dynamic transfers via `prepareActivationForDevice()`
   - Intermediate buffers: Always co-located with execution device
3. **MPI coordination protocol**:
   - Host-only MPI operations
   - Explicit device ↔ host staging
   - Synchronization requirements
4. **Performance implications**:
   - Cross-socket GPU access penalty (~40-60% slower)
   - PCIe transfer latency
   - MPI host staging overhead

---

### Phase 4: Multi-Device Pipeline Parallelism (Future)

**Goal**: Execute multiple layers concurrently on different devices.

**Design**:
```cpp
class PipelinePar allelizerV2 {
public:
    // Partition layers into device-local stages
    struct Stage {
        int start_layer, end_layer;
        int device_idx;
    };
    
    std::vector<Stage> createStages(const PlacementMap& map, int n_layers) {
        // Group consecutive layers on same device into stages
        // Example: [0-11 on GPU:0], [12-23 on GPU:1], [24-31 on CPU]
    }
    
    // Execute stages with pipeline parallelism
    bool execute(const std::vector<Stage>& stages, TensorBase* input, int batch_size) {
        // Microbatch 0: Stage 0 (layers 0-11 on GPU:0)
        // Microbatch 1: Stage 0, then Stage 1 (layers 12-23 on GPU:1)
        // Microbatch 2: Stage 0, then Stage 1, then Stage 2 (layers 24-31 on CPU)
        // ...
    }
};
```

**Prerequisite**: Batch processing with microbatching support.

---

## Testing Strategy

### Unit Tests

1. **DeviceManager NUMA filtering**:
   ```cpp
   TEST(DeviceManager, NUMAFiltering) {
       auto& dm = DeviceManager::instance();
       dm.initialize(0);  // Socket 0
       
       // Should only see socket-0 devices
       for (auto& dev : dm.devices()) {
           EXPECT_EQ(dev.numa_node, 0);
       }
   }
   ```

2. **MPI staging helpers**:
   ```cpp
   TEST(MPIStager, DeviceToHostTransfer) {
       auto gpu_tensor = createGPUTensor({1024, 1024});
       auto host_vec = MPIStager::toHost(gpu_tensor);
       EXPECT_EQ(host_vec.size(), 1024*1024);
   }
   ```

### Integration Tests

1. **Cross-device layer execution**:
   ```cpp
   TEST(Qwen2Pipeline, HeterogeneousExecution) {
       PlacementMap map;
       map.setLayerDevice(0, 0);   // Layer 0 on GPU:0
       map.setLayerDevice(1, -1);  // Layer 1 on CPU
       map.setLayerDevice(2, 1);   // Layer 2 on GPU:1
       
       auto pipeline = Qwen2Pipeline(..., map);
       EXPECT_TRUE(pipeline.forward(tokens, 32));
       // Validate activation transfers happened correctly
   }
   ```

2. **MPI + heterogeneous devices**:
   ```bash
   mpirun -np 2 ./test_mpi_hetero_attention
   # Rank 0: GPU:0
   # Rank 1: GPU:2
   # Validate allgather works correctly
   ```

---

## Performance Validation

### Benchmark Cross-Socket GPU Access

```cpp
// Measure penalty of cross-socket GPU usage
void benchmark_gpu_numa_locality() {
    // Rank 0, GPU:0 (local to socket 0)
    auto t1 = benchmark_matmul(gpu:0);  // ~1.2 TFLOPs
    
    // Rank 0, GPU:2 (on socket 1)
    auto t2 = benchmark_matmul(gpu:2);  // ~0.7 TFLOPs (40% penalty)
    
    LOG_INFO("Cross-socket penalty: " << (1.0 - t2/t1) * 100 << "%");
}
```

**Expected**: 30-60% performance degradation for cross-socket GPU access on 2-socket Xeon systems.

---

## Open Questions for Discussion

1. **Should we support cross-socket GPU access at all?**
   - Pro: Flexibility for unbalanced workloads
   - Con: Severe performance penalty, adds complexity
   - **Recommendation**: Forbid cross-socket access, error if placement tries it

2. **How to handle systems with unbalanced GPU counts?**
   - Example: Socket 0 has 3 GPUs, Socket 1 has 1 GPU
   - **Recommendation**: Orchestrator should fail-fast with clear error message

3. **Should MPI ranks share GPUs?**
   - Example: Both ranks use GPU:0
   - **Recommendation**: No - exclusive GPU ownership per rank

4. **What about single-socket multi-GPU systems?**
   - Current: Rank 0 sees all GPUs (correct behavior)
   - **Recommendation**: Keep as-is, NUMA filtering is no-op for single socket

---

## Summary of Changes

### Immediate (This PR)

1. ✅ **Document this analysis** → `docs/architecture/MPI_DEVICE_ORCHESTRATION_ANALYSIS.md`
2. ⏳ **Add NUMA detection** → `DeviceManager::initialize(numa_node)`
3. ⏳ **Filter devices by socket** → Only enumerate socket-local GPUs
4. ⏳ **Add validation** → Error if placement tries cross-socket access

### Short-term (Next PR)

5. ⏳ **MPI host staging** → Explicit device ↔ host transfers around MPI ops
6. ⏳ **Device execution docs** → `DEVICE_EXECUTION_MODEL.md`
7. ⏳ **Integration tests** → Multi-device, multi-rank correctness

### Long-term (Future)

8. ⏳ **Pipeline parallelism** → Concurrent layer execution on different devices
9. ⏳ **Microbatching** → Enable pipeline parallelism with batch splitting

---

## References

- **Current Code**:
  - `src/v2/backends/ComputeBackend.{h,cpp}` - Device enumeration
  - `src/v2/pipelines/PipelineBase.{h,cpp}` - Multi-device infrastructure
  - `src/v2/loaders/DeviceOrchestrator.{h,cpp}` - Placement strategies
  - `src/v2/Main.cpp` - MPI + device initialization

- **Related Docs**:
  - `.github/instructions/llaminar-v2-architecture.instructions.md` - V2 architecture
  - `changelog/2025-10-24-phase4-device-placement.md` - Multi-device implementation

- **External References**:
  - NUMA API: `man numa(3)`, `hwloc` library
  - NVML: `nvidia-smi topo -m` for GPU-socket affinity
  - Intel MPI: `I_MPI_PIN_DOMAIN` for socket binding

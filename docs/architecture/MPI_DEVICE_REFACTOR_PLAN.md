# MPI + Device Architecture Refactor Plan

**Date**: October 31, 2025  
**Status**: ACTIVE - Phase 1 In Progress  
**Related**: `MPI_DEVICE_ORCHESTRATION_ANALYSIS.md`

---

## Refactor Phases

### Phase 1: NUMA-Aware Device Enumeration ⏳ IN PROGRESS

**Goal**: Each MPI rank only enumerates and uses devices affine to its NUMA node/socket.

**Estimated Effort**: 4-6 hours  
**Risk**: LOW - Isolated to DeviceManager, no pipeline changes

#### Changes Required

1. **Add NUMA detection utilities** (`src/v2/utils/NUMATopology.h/cpp`)
   - `detectLocalNUMANode()` - Get current process's NUMA node
   - `getGPUNUMANode(int cuda_id)` - Get GPU's NUMA affinity via NVML
   - `getCPUSocketCount()` - Total sockets on system
   - Fallback to NUMA node 0 if detection fails (single-socket systems)

2. **Enhance DeviceManager** (`src/v2/backends/ComputeBackend.h/cpp`)
   - Add `initialize(int numa_node)` overload
   - Store `local_numa_node_` member
   - Filter GPU enumeration by NUMA affinity
   - Add `numa_node` field to `ComputeDevice` struct

3. **Update Main.cpp initialization**
   - Detect NUMA node before DeviceManager init
   - Pass to `dm.initialize(numa_node)`
   - Log socket affinity info

4. **Add validation**
   - `DeviceOrchestrator`: Detect cross-socket placement attempts
   - Error with clear message if placement violates socket isolation

#### Testing

```bash
# Unit test: NUMA detection
./build_v2/tests/v2/v2_test_numa_topology

# Integration test: MPI + NUMA
mpirun -np 2 --bind-to socket ./build_v2/tests/v2/v2_test_mpi_device_affinity

# Manual validation
mpirun -np 2 --bind-to socket ./build_v2/llaminar2 --list-devices
# Each rank should only show socket-local devices
```

#### Success Criteria

- ✅ Each MPI rank sees only its socket-local GPUs
- ✅ Device indices are rank-local (both ranks have "GPU:0" for their first GPU)
- ✅ NUMA node is logged for each device
- ✅ No performance regression on single-socket systems
- ✅ Clear error if placement tries cross-socket access

---

### Phase 2: MPI Host Staging ⏳ PLANNED

**Goal**: All MPI collectives operate on host memory with explicit device staging.

**Estimated Effort**: 6-8 hours  
**Risk**: MEDIUM - Touches MPI-aware kernels, need correctness validation

#### Changes Required

1. **Create MPI staging utilities** (`src/v2/utils/MPIStager.h/cpp`)
   ```cpp
   class MPIStager {
   public:
       // Stage tensor to host for MPI operation
       static std::vector<float> toHost(const TensorBase* tensor);
       
       // Stage MPI result back to device
       static void toDevice(const std::vector<float>& host, TensorBase* tensor);
       
       // Typed variants for int, bfloat16, etc.
   };
   ```

2. **Update GQAAttention MPI paths**
   - `compute_mpi()` - Add staging around allreduce
   - `compute_tensor_parallel()` - Add staging around allgather
   - Explicit cudaMemcpy when output_device >= 0

3. **Update other MPI-aware operations**
   - Embedding broadcast (if weights on GPU)
   - Any tensor-parallel GEMMs
   - KV cache synchronization (if applicable)

4. **Add synchronization**
   - `MPI_Barrier` before/after device staging
   - Clear ownership semantics (who owns host buffer?)

#### Testing

```bash
# Correctness: MPI + GPU outputs
mpirun -np 2 ./build_v2/tests/v2/v2_test_mpi_gpu_staging

# Performance: Measure staging overhead
./build_v2/tests/v2/v2_perf_mpi_staging_overhead
```

#### Success Criteria

- ✅ MPI operations work with GPU-resident tensors
- ✅ Correct results with heterogeneous device outputs across ranks
- ✅ No memory leaks in staging buffers
- ✅ Performance penalty <5% vs host-only baseline

---

### Phase 3: Device Execution Contract Documentation ⏳ PLANNED

**Goal**: Formalize and document the device execution model.

**Estimated Effort**: 3-4 hours  
**Risk**: LOW - Documentation only

#### Deliverables

1. **`docs/architecture/DEVICE_EXECUTION_MODEL.md`**
   - 3-tier hierarchy (MPI → Pipeline → Device)
   - Device affinity rules
   - Activation transfer semantics
   - MPI coordination protocol
   - Performance characteristics

2. **Code comments**
   - `PipelineBase::prepareActivationForDevice()` - Contract documentation
   - `DeviceManager::initialize()` - NUMA filtering explanation
   - `GQAAttention::compute_tensor_parallel()` - MPI staging pattern

3. **Developer guide**
   - How to add new MPI-aware kernels
   - Device placement best practices
   - Common pitfalls and debugging

#### Success Criteria

- ✅ New developers can understand device execution model
- ✅ Clear examples for common patterns
- ✅ No contradictions with actual implementation

---

### Phase 4: Pipeline Execution Refactor ⏳ FUTURE

**Goal**: Make sequential layer execution explicit and add device transfer logging.

**Estimated Effort**: 4-6 hours  
**Risk**: LOW - Clarification of existing behavior

#### Changes Required

1. **Explicit execution loop** (`PipelineBase::forward()`)
   ```cpp
   for (int layer = 0; layer < n_layers_; ++layer) {
       int exec_device = getLayerExecutionDevice(layer);
       
       LOG_DEBUG("[Layer " << layer << "] Executing on device " << exec_device);
       
       // Transfer if needed
       if (current_hidden_->device_index() != exec_device) {
           LOG_INFO("[Layer " << layer << "] Transfer: device " 
                    << current_hidden_->device_index() << " → " << exec_device);
           current_hidden_ = prepareActivationForDevice(current_hidden_, exec_device);
       }
       
       // Execute layer
       if (!transformer_layer(layer, seq_len)) return false;
   }
   ```

2. **Device transfer metrics**
   - Track number of transfers per forward pass
   - Log transfer volume (MB)
   - Warn if excessive transfers detected

3. **Validation**
   - Assert layer completes on expected device
   - Detect unexpected mid-layer transfers

#### Success Criteria

- ✅ Clear logging of layer → device mapping
- ✅ Transfer overhead is visible and measurable
- ✅ No hidden device migrations

---

### Phase 5: Multi-Device Pipeline Parallelism ⏳ FUTURE (V3)

**Goal**: Execute multiple layers concurrently on different devices (pipeline parallelism).

**Estimated Effort**: 2-3 weeks  
**Risk**: HIGH - Major architectural change, requires microbatching

#### Design

```cpp
class PipelineParallelExecutor {
    struct Stage {
        int start_layer, end_layer;
        int device_idx;
        std::vector<TensorBase*> activations;  // Per-microbatch
    };
    
    // Partition model into stages (consecutive layers on same device)
    std::vector<Stage> createStages(const PlacementMap& map);
    
    // Execute with pipeline parallelism
    bool execute(int batch_size, int microbatch_size);
};
```

**Prerequisites**:
- Batch processing infrastructure
- Microbatch splitting/merging
- Inter-device queues
- Careful memory management

**Out of Scope**: Not included in current refactor plan.

---

## Implementation Order

### Week 1: Foundation (Phase 1-2)

**Day 1-2**: Phase 1 - NUMA-aware enumeration
- [ ] Create `NUMATopology.h/cpp`
- [ ] Modify `DeviceManager`
- [ ] Update `Main.cpp`
- [ ] Unit tests
- [ ] Integration tests

**Day 3-4**: Phase 2 - MPI host staging
- [ ] Create `MPIStager.h/cpp`
- [ ] Update `GQAAttention` MPI paths
- [ ] Correctness tests
- [ ] Performance validation

**Day 5**: Phase 3 - Documentation
- [ ] Write `DEVICE_EXECUTION_MODEL.md`
- [ ] Add code comments
- [ ] Developer guide

### Week 2: Refinement (Phase 4)

**Day 1-2**: Phase 4 - Explicit execution
- [ ] Refactor `PipelineBase::forward()`
- [ ] Add transfer logging
- [ ] Metrics collection

**Day 3**: Integration testing
- [ ] End-to-end MPI + multi-GPU tests
- [ ] Performance benchmarking
- [ ] Regression testing

**Day 4-5**: Buffer & polish
- [ ] Fix bugs found in testing
- [ ] Optimize hot paths
- [ ] Final documentation pass

---

## Testing Strategy

### Unit Tests (Per Phase)

```cpp
// Phase 1: NUMA detection
TEST(NUMATopology, DetectLocalNode)
TEST(DeviceManager, FilterByNUMA)
TEST(DeviceManager, SingleSocketFallback)

// Phase 2: MPI staging
TEST(MPIStager, CPUToHost)
TEST(MPIStager, GPUToHost)
TEST(MPIStager, HostToGPU)

// Phase 4: Execution logging
TEST(PipelineBase, LayerDeviceMapping)
TEST(PipelineBase, TransferMetrics)
```

### Integration Tests

```cpp
// MPI + heterogeneous devices
TEST(MPIIntegration, TwoRanksTwoSockets)
TEST(MPIIntegration, GPUOutputsAcrossRanks)
TEST(MPIIntegration, CrossLayerDeviceTransfer)

// Full pipeline
TEST(Qwen2Pipeline, HeterogeneousExecution)
TEST(Qwen2Pipeline, MPIWithGPUs)
```

### Performance Benchmarks

```bash
# Baseline (before refactor)
./benchmark_qwen2_inference --device auto --mpi-ranks 2

# After Phase 1 (NUMA filtering)
./benchmark_qwen2_inference --device auto --mpi-ranks 2
# Should be identical or slightly faster (no cross-socket access)

# After Phase 2 (MPI staging)
./benchmark_qwen2_inference --device gpu:0 --mpi-ranks 2
# Measure staging overhead (expect <5% degradation)
```

### Validation Criteria

For each phase:
1. ✅ All existing tests pass
2. ✅ New tests for phase functionality pass
3. ✅ No performance regression on single-socket systems
4. ✅ Performance improvement on multi-socket systems (Phase 1)
5. ✅ Clear error messages for misconfiguration

---

## Rollback Plan

### If Phase 1 Breaks Something

```bash
# Revert DeviceManager changes
git revert <commit_hash>

# Fallback: Initialize without NUMA filtering
DeviceManager::initialize(-1);  // -1 = enumerate all devices
```

### If Phase 2 Has Bugs

```bash
# Disable MPI staging temporarily
#define LLAMINAR_DISABLE_MPI_STAGING 1

# Use host-only tensors for MPI operations
if (mpi_ctx_) {
    output_device = -1;  // Force CPU output for MPI
}
```

### Emergency Kill Switch

```cpp
// Add to PipelineConfig
bool disable_numa_filtering = false;
bool disable_mpi_staging = false;
bool disable_device_transfers = false;
```

---

## Risk Mitigation

### NUMA Detection Failure

**Risk**: `detectLocalNUMANode()` fails on some systems.

**Mitigation**:
- Fallback to NUMA node 0
- Log warning: "NUMA detection failed, assuming single socket"
- Continue execution normally

### GPU NUMA Affinity Unknown

**Risk**: NVML unavailable or GPU affinity unclear.

**Mitigation**:
- Assume all GPUs affine to NUMA node 0
- Only filter if confident in affinity data
- Log warning about unfiltered devices

### MPI Staging Performance

**Risk**: Host staging adds significant overhead.

**Mitigation**:
- Benchmark before/after
- Add `LLAMINAR_DISABLE_MPI_STAGING` env var
- Profile to find hotspots
- Consider CUDA-aware MPI in future (Phase 6)

---

## Success Metrics

### Phase 1 Success

- ✅ 0% cross-socket GPU access in MPI runs
- ✅ 10-20% performance improvement on multi-socket systems
- ✅ No regression on single-socket systems
- ✅ Clear device enumeration in logs

### Phase 2 Success

- ✅ All MPI tests pass with GPU outputs
- ✅ <5% performance overhead from staging
- ✅ No memory leaks (valgrind clean)
- ✅ Correct numerical results (parity tests pass)

### Overall Refactor Success

- ✅ Architectural ambiguities resolved
- ✅ Clear 3-tier execution model
- ✅ No regressions in existing functionality
- ✅ Performance improvement on multi-socket + multi-GPU
- ✅ Developer-friendly documentation

---

## Current Status

**Active Phase**: Phase 1 - NUMA-Aware Device Enumeration  
**Started**: October 31, 2025  
**Next Milestone**: NUMA detection utilities implementation  

**Blocked**: None  
**Risks**: None identified yet  
**Questions**: None pending

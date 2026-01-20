# Hybrid Parallelism Project Plan

## Executive Summary

This document outlines a phased implementation plan for enabling heterogeneous multi-device inference in Llaminar V2, targeting the architecture:

- **Cross-rank**: Pipeline Parallel (layers 0-13 on rank 0, 14-27 on rank 1)
- **Intra-rank**: Pipeline Parallel split between CPU and GPUs
- **Intra-GPU**: Tensor Parallel across NVIDIA + AMD GPUs via PCIeBAR

**Current State**: Infrastructure is ~80% complete but not wired together.

**Target**: 10+ tok/s decode throughput (currently 0.41 tok/s on ROCm)

---

## Research Summary

| Area | Status | Key Finding |
|------|--------|-------------|
| **PCIeBAR Backend** | ✅ Implemented | AllReduce works, needs latency benchmarks for small transfers |
| **Pipeline Parallelism** | ⚠️ Infrastructure exists | Missing P2P MPI primitives, no pipeline scheduler |
| **Heterogeneous TP** | ⚠️ Partial | BackendRouter supports it, but equal-split assumption blocks proportional heads |
| **CollectiveContext Wiring** | ❌ Not connected | 7-hour fix to wire to GraphExecutor |
| **CPU Pipeline Stage** | ✅ Ready | CPU kernels complete, just needs layer assignment config |

---

## Phase 0: Foundation & Validation (Week 1)

### Objective
Wire the existing infrastructure and validate PCIeBAR latency before building on it.

### 0.1 Wire CollectiveContext to AllreduceStage
**Priority**: CRITICAL - Unblocks ROCm performance immediately

**Effort**: 7 hours

**Changes**:
| File | Change |
|------|--------|
| `src/v2/execution/GraphExecutor.h` | Add `collective_ctx_` member and setter |
| `src/v2/execution/GraphExecutor.cpp` | Add `executeCollectiveStage()` intercept for ALLREDUCE/ALLGATHER |
| `src/v2/pipelines/qwen/GraphOrchestrator.cpp` | Wire `injected_collective_ctx_` to executor |
| `InferenceRunnerFactory.cpp` | Create CollectiveContext with ClusterInventory |

**Implementation**:
```cpp
// GraphExecutor.h
class GraphExecutor : public IGraphExecutor {
public:
    void setCollectiveContext(ICollectiveContext* ctx) { collective_ctx_ = ctx; }
private:
    ICollectiveContext* collective_ctx_ = nullptr;
};

// GraphExecutor.cpp - executeNode()
if (collective_ctx_ && node.stage->type() == ComputeStageType::ALLREDUCE) {
    return executeCollectiveStage(node, ctx);  // Delegates to CollectiveContext
}
```

**Success Criteria**:
- ROCm single-GPU uses RCCL instead of MPI_Allreduce
- BackendRouter correctly selects RCCL for ROCm, NCCL for CUDA
- Log output shows "Using RCCL backend" or "Using PCIeBAR backend"

### 0.2 PCIeBAR Latency Benchmark Suite
**Priority**: HIGH - Validates architecture assumption

**Effort**: 2 days

**Why Needed**: Current benchmarks measure bandwidth (2.65 GB/s), but hybrid parallelism needs low **latency** for small 14KB activation transfers. Estimated overhead is 50-155μs but unverified.

**Benchmark Cases**:
| Test | Transfer Size | Expected Latency | Purpose |
|------|--------------|------------------|---------|
| Minimum latency | 1 KB | 45-100 μs | Baseline overhead |
| Activation size | 14 KB | 50-155 μs | Wo output (d_model × 4 bytes) |
| KV head | 28 KB | 60-180 μs | One KV head per layer |
| Full attention | 112 KB | 100-300 μs | All heads one layer |
| Large transfer | 1 MB | 400-600 μs | FFN intermediate |

**Test File**: `tests/v2/integration/Test__PCIeBARLatencyBenchmark.cpp`

**Metrics to Capture**:
```cpp
struct PCIeBARLatencyMetrics {
    double kernel_launch_overhead_us;   // Time before DMA starts
    double dma_latency_us;              // Actual transfer time
    double sync_overhead_us;            // hipDeviceSynchronize / cudaDeviceSynchronize
    double total_round_trip_us;         // End-to-end AllReduce
    double bandwidth_gbps;              // Throughput at this size
};
```

**Success Criteria**:
- 14KB AllReduce completes in < 200μs
- Benchmark produces reproducible results (< 10% variance)
- Document actual vs expected latency for planning

### 0.3 Baseline Performance Measurement
**Priority**: HIGH - Establishes success metrics

**Effort**: 0.5 days

**Tasks**:
1. Benchmark current ROCm decode with MPI AllReduce (expect ~0.41 tok/s)
2. Benchmark with CollectiveContext wired (expect 2-5x improvement)
3. Benchmark CUDA for comparison
4. Document per-stage timing breakdown

**Output**: Baseline metrics in `docs/v2/PERFORMANCE_BASELINES.md`

---

## Phase 1: Tensor Parallel Optimization (Weeks 2-3)

### Objective
Optimize intra-node GPU tensor parallelism with PCIeBAR for NVIDIA↔AMD communication.

### 1.1 PCIeBAR Production Hardening
**Priority**: HIGH

**Effort**: 5 days

**Gap Analysis**:
| Issue | Impact | Fix |
|-------|--------|-----|
| Only FP32 AllReduce | Medium | Add FP16/BF16 support |
| No retry/timeout | High | Add transient error recovery |
| No ReduceScatter | Low | Not needed for current TP |
| Hardcoded 1GB BAR | Medium | Make configurable |

**Tasks**:
1. Add FP16/BF16 reduction kernels (2 days)
2. Add timeout/retry logic with configurable thresholds (1 day)
3. Add error injection tests (1 day)
4. Make BAR size configurable via environment variable (0.5 days)
5. Documentation and examples (0.5 days)

### 1.2 Proportional Head Assignment
**Priority**: MEDIUM - Enables load balancing across heterogeneous GPUs

**Effort**: 1 week

**Current Limitation**: `WeightManager::calculateSlice()` assumes equal splits:
```cpp
// Current code - equal split
size_t local_dim = total_dim / world_size_;
size_t offset = rank_ * local_dim;

// Needed - proportional split
size_t local_dim = assignment.head_count * head_dim;
size_t offset = assignment.head_offset * head_dim;
```

**Required Changes**:

1. **New Configuration Struct**:
```cpp
struct DeviceShardingAssignment {
    DeviceId device;
    int head_count;        // Heads assigned to this device
    int head_offset;       // Starting head index
    float weight_ratio;    // Portion of total work (e.g., 0.73 for NVIDIA)
};

struct TensorParallelConfig {
    std::vector<DeviceShardingAssignment> assignments;
    int total_heads;       // Sum of all head_counts
    ShardingMode mode;     // PROPORTIONAL or EQUAL
};
```

2. **WeightManager Changes**:
   - Accept `TensorParallelConfig` instead of just world_size/rank
   - Slice weights according to `head_count` not `1/world_size`
   - Support variable offsets

3. **Qwen2Graph Changes**:
   - Pass device-specific `local_num_heads` to attention stages
   - Variable KV cache allocation per device

4. **AllGatherV Stage**:
   - Variable-sized AllGather for differently-sharded outputs
   - Requires `sendcounts[]` and `displacements[]` arrays

**Success Criteria**:
- NVIDIA (A100): 20 heads, AMD (MI300): 8 heads (73%/27% split)
- Both GPUs complete attention in similar wall-clock time
- AllGatherV correctly assembles full output

### 1.3 Heterogeneous Device Group Testing
**Priority**: MEDIUM

**Effort**: 3 days

**Test Cases**:
| Configuration | Expected Backend | Test |
|--------------|------------------|------|
| 1× CUDA | NCCL (trivial) | `Test__DeviceGroup_SingleCUDA` |
| 1× ROCm | RCCL (trivial) | `Test__DeviceGroup_SingleROCm` |
| 2× CUDA | NCCL | `Test__DeviceGroup_DualCUDA` |
| 2× ROCm | RCCL | `Test__DeviceGroup_DualROCm` |
| 1× CUDA + 1× ROCm | PCIeBAR | `Test__DeviceGroup_Heterogeneous` |
| 1× CUDA + 1× ROCm + CPU | HOST | `Test__DeviceGroup_FullHeterogeneous` |

---

## Phase 2: Pipeline Parallelism (Weeks 4-6)

### Objective
Enable cross-rank pipeline parallelism for multi-socket/multi-node scaling.

### 2.1 Point-to-Point MPI Primitives
**Priority**: CRITICAL for PP

**Effort**: 3 days

**Current Gap**: `MPIContext` has collective operations but NO point-to-point:
```cpp
// Needed additions to MPIContext
void send(const void* data, size_t count, MPI_Datatype type, int dest, int tag);
void recv(void* data, size_t count, MPI_Datatype type, int src, int tag);
MPI_Request isend(...);  // Async
MPI_Request irecv(...);  // Async
void wait(MPI_Request* request);
void waitAll(std::vector<MPI_Request>& requests);
```

**Implementation Notes**:
- Use MPI tags to distinguish layer boundaries
- Consider persistent requests for decode loop efficiency

### 2.2 Send/Receive Activation Stages
**Priority**: HIGH

**Effort**: 4 days

**New Stages**:
```cpp
class SendActivationsStage : public ComputeStageBase {
    // Sends activations to next pipeline stage (rank)
    Params: buffer, dest_rank, tag, async
};

class ReceiveActivationsStage : public ComputeStageBase {
    // Receives activations from previous pipeline stage
    Params: buffer, src_rank, tag, async
};
```

**Integration with ComputeGraph**:
```cpp
// End of rank 0's layers (after layer 13)
graph.addNode("send_to_rank1", SendActivationsStage({
    .buffer = layer13_output,
    .dest_rank = 1,
    .tag = ACTIVATION_TAG,
}), DeviceId::cpu());  // CPU-staged for portability

// Start of rank 1's layers (before layer 14)
graph.addNode("recv_from_rank0", ReceiveActivationsStage({
    .buffer = layer14_input,
    .src_rank = 0,
    .tag = ACTIVATION_TAG,
}), DeviceId::cpu());
```

### 2.3 Layer Range Configuration
**Priority**: HIGH

**Effort**: 2 days

**Configuration**:
```cpp
struct PipelineParallelConfig {
    int num_pipeline_stages;           // e.g., 2
    std::vector<LayerRange> stage_layers;  // [{0,13}, {14,27}]
    int warmup_micro_batches;          // GPipe warmup count
};

struct LayerRange {
    int first_layer;  // inclusive
    int last_layer;   // inclusive
    int owning_rank;
};
```

**WeightManager Integration**:
- Only load weights for assigned layers
- Memory savings: ~50% per rank for 2-way PP

### 2.4 Basic Pipeline Scheduler (GPipe-style)
**Priority**: MEDIUM

**Effort**: 1 week

**Schedule for 2-stage PP with 4 micro-batches**:
```
Time →
Rank 0: [F0][F1][F2][F3]        [B3][B2][B1][B0]
Rank 1:     [F0][F1][F2][F3][B3][B2][B1][B0]
```

**Implementation**:
1. Partition input into micro-batches
2. Forward pass: Stream micro-batches through pipeline
3. Backward pass: N/A for inference (no gradients)
4. Synchronize between forward completion

**Note**: For inference-only, this simplifies to sequential micro-batch processing with activation buffering.

### 2.5 Pipeline Parallel Testing
**Priority**: HIGH

**Effort**: 3 days

**Test Cases**:
1. 2-rank PP with synchronous send/recv
2. Verify activations match single-rank execution
3. Memory usage verification (should be ~50% per rank)
4. Latency overhead measurement

---

## Phase 3: CPU Pipeline Stage (Weeks 7-8)

### Objective
Enable CPU to process transformer layers as a pipeline stage for memory offloading.

### 3.1 Layer Placement Configuration
**Priority**: MEDIUM

**Effort**: 2 days

**Configuration**:
```cpp
struct LayerPlacementConfig {
    std::vector<LayerDeviceAssignment> assignments;
};

struct LayerDeviceAssignment {
    int layer_index;
    DeviceId device;  // DeviceId::cpu() or DeviceId::cuda(0) etc.
};
```

**Example - CPU handles first 4 layers**:
```cpp
LayerPlacementConfig config;
for (int i = 0; i < 4; i++) {
    config.assignments.push_back({i, DeviceId::cpu()});
}
for (int i = 4; i < 28; i++) {
    config.assignments.push_back({i, DeviceId::cuda(0)});
}
```

### 3.2 CPU↔GPU Activation Transfer Stage
**Priority**: MEDIUM

**Effort**: 3 days

**New Stage**:
```cpp
class TransferActivationsStage : public ComputeStageBase {
    // Transfers activations between CPU and GPU within same rank
    Params: src_buffer, dst_buffer, src_device, dst_device
};
```

**Optimization Options**:
1. **Pageable memory**: Simple, ~15-30 μs for 14KB
2. **Pinned memory pool**: Lower latency, ~5-10 μs for 14KB
3. **Zero-copy mapping**: Near-zero sync, GPU accesses CPU memory directly

**Recommendation**: Start with pageable, add pinned pool if latency is bottleneck.

### 3.3 Async CPU Execution (Optional)
**Priority**: LOW

**Effort**: 1 week

**Current Limitation**: GraphExecutor runs stages sequentially. CPU and GPU stages don't overlap.

**Enhancement**:
```cpp
class AsyncGraphExecutor : public IGraphExecutor {
    // Thread pool for CPU stages
    // CUDA/HIP streams for GPU stages
    // Dependency tracking for overlap
};
```

**Value**: Enables CPU to compute layer N+1 while GPU computes layer N.

**Skip Condition**: If CPU throughput << GPU throughput (likely ~3% TFLOPS contribution), overlap provides minimal benefit. CPU's value is memory offload, not compute.

### 3.4 CPU Pipeline Testing
**Priority**: MEDIUM

**Effort**: 2 days

**Test Cases**:
1. Single layer on CPU, rest on GPU
2. Numerical parity with all-GPU execution
3. Memory usage verification
4. Latency measurement for CPU layers

---

## Phase 4: Integration & Optimization (Weeks 9-10)

### Objective
Combine all parallelism modes and optimize end-to-end performance.

### 4.1 Three-Level Hybrid Mode
**Priority**: HIGH

**Effort**: 1 week

**Target Configuration**:
```
Rank 0 (Socket 0):
├── CPU: Layers 0-3 (pipeline stage 0a)
├── NVIDIA GPU: Layers 4-13, heads 0-19 (TP + pipeline stage 0b)
└── AMD GPU: Layers 4-13, heads 20-27 (TP + pipeline stage 0b)

Rank 1 (Socket 1):
├── CPU: Layers 14-17 (pipeline stage 1a)
├── NVIDIA GPU: Layers 18-27, heads 0-19 (TP + pipeline stage 1b)
└── AMD GPU: Layers 18-27, heads 20-27 (TP + pipeline stage 1b)
```

**Data Flow**:
1. Token embedding (CPU rank 0)
2. CPU layers 0-3 (rank 0)
3. Transfer to GPUs (rank 0)
4. TP across NVIDIA+AMD for layers 4-13 (rank 0)
5. PCIeBAR AllReduce after each TP layer
6. MPI Send to rank 1
7. Repeat on rank 1 with layers 14-27
8. LM head + sampling (rank 1)

### 4.2 Communication Overlap
**Priority**: MEDIUM

**Effort**: 4 days

**Opportunities**:
1. **TP AllReduce overlap**: Start next layer's GEMM while AllReduce completes
2. **PP Send overlap**: Start layer N+1 on sender while receiver processes layer N
3. **Prefetch overlap**: Load next micro-batch while processing current

**Implementation**: CUDA/HIP streams + async MPI operations

### 4.3 Memory Optimization
**Priority**: MEDIUM

**Effort**: 3 days

**Tasks**:
1. Activation checkpointing for PP (recompute vs store)
2. KV cache memory pool shared across layers
3. Weight memory mapping for CPU layers (mmap, no copy)

### 4.4 End-to-End Benchmarking
**Priority**: HIGH

**Effort**: 3 days

**Benchmark Suite**:
| Test | Configuration | Target |
|------|--------------|--------|
| Single GPU decode | 1× ROCm | >5 tok/s |
| TP decode | 1× CUDA + 1× ROCm | >10 tok/s |
| PP decode | 2 ranks | >8 tok/s |
| Full hybrid | 2 ranks × (CPU + CUDA + ROCm) | >15 tok/s |

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PCIeBAR latency too high for small transfers | HIGH | Medium | Fall back to NCCL+RCCL with CPU staging |
| Proportional head assignment breaks numerical parity | MEDIUM | Low | Extensive parity testing |
| Pipeline bubble overhead dominates for short sequences | HIGH | Medium | Micro-batching, skip PP for short prompts |
| CUDA↔ROCm driver conflicts | HIGH | Low | Careful initialization order, isolation testing |
| MPI deadlocks in PP | HIGH | Medium | Extensive testing, tag management |

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Phase 0 | CollectiveContext wired, PCIeBAR latency benchmarks |
| 2-3 | Phase 1 | Proportional head assignment, PCIeBAR hardening |
| 4-6 | Phase 2 | Pipeline parallelism with layer ranges |
| 7-8 | Phase 3 | CPU pipeline stage support |
| 9-10 | Phase 4 | Full hybrid integration, optimization |

**Total Effort**: ~10 weeks for full implementation

---

## Appendix A: File Change Summary

### Phase 0 Files
- `src/v2/execution/GraphExecutor.h` - Add collective_ctx_
- `src/v2/execution/GraphExecutor.cpp` - Add executeCollectiveStage()
- `src/v2/pipelines/qwen/GraphOrchestrator.cpp` - Wire collective_ctx
- `tests/v2/integration/Test__PCIeBARLatencyBenchmark.cpp` - New

### Phase 1 Files
- `src/v2/collective/backends/PCIeBARBackend.cpp` - FP16/BF16, timeout
- `src/v2/loaders/WeightManager.cpp` - Proportional slicing
- `src/v2/loaders/WeightManager.h` - TensorParallelConfig
- `src/v2/models/qwen/Qwen2Graph.cpp` - Variable local_num_heads
- `src/v2/execution/compute_stages/stages/AllGatherVStage.cpp` - New

### Phase 2 Files
- `src/v2/utils/MPIContext.h` - P2P primitives
- `src/v2/utils/MPIContext.cpp` - send/recv implementation
- `src/v2/execution/compute_stages/stages/SendActivationsStage.cpp` - New
- `src/v2/execution/compute_stages/stages/ReceiveActivationsStage.cpp` - New
- `src/v2/pipelines/qwen/PipelineScheduler.cpp` - New

### Phase 3 Files
- `src/v2/pipelines/qwen/LayerPlacementConfig.h` - New
- `src/v2/execution/compute_stages/stages/TransferActivationsStage.cpp` - New

### Phase 4 Files
- `src/v2/pipelines/qwen/HybridOrchestrator.cpp` - New (combines all modes)
- `tests/v2/performance/Perf__HybridParallel.cpp` - New

---

## Appendix B: PCIeBAR Latency Test Plan

### Test Infrastructure
```cpp
// tests/v2/integration/Test__PCIeBARLatencyBenchmark.cpp

class PCIeBARLatencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuda_device_ = DeviceId::cuda(0);
        rocm_device_ = DeviceId::rocm(0);
        
        backend_ = std::make_unique<PCIeBARBackend>(
            ClusterInventory::detectDevices());
    }
    
    void benchmarkAllReduce(size_t bytes, int iterations = 100) {
        auto cuda_buffer = allocateCUDA(bytes);
        auto rocm_buffer = allocateROCm(bytes);
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            backend_->allreduce(cuda_buffer.get(), bytes, DataType::FP32);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            backend_->allreduce(cuda_buffer.get(), bytes, DataType::FP32);
            synchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_op_us = total_us / iterations;
        
        LOG_INFO("PCIeBAR AllReduce " << bytes << " bytes: " 
                 << per_op_us << " μs/op");
    }
};

TEST_F(PCIeBARLatencyTest, SmallTransferLatency) {
    // Critical for hybrid parallelism viability
    benchmarkAllReduce(1 * 1024);    // 1 KB
    benchmarkAllReduce(14 * 1024);   // 14 KB (Wo output)
    benchmarkAllReduce(28 * 1024);   // 28 KB (KV head)
    benchmarkAllReduce(64 * 1024);   // 64 KB
    benchmarkAllReduce(128 * 1024);  // 128 KB
}

TEST_F(PCIeBARLatencyTest, LargeTransferBandwidth) {
    benchmarkAllReduce(1 * 1024 * 1024);   // 1 MB
    benchmarkAllReduce(4 * 1024 * 1024);   // 4 MB
    benchmarkAllReduce(16 * 1024 * 1024);  // 16 MB
}

TEST_F(PCIeBARLatencyTest, SustainedThroughput) {
    // Simulate decode loop: 56 AllReduces per forward pass
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int token = 0; token < 100; token++) {
        for (int layer = 0; layer < 28; layer++) {
            backend_->allreduce(wo_buffer_.get(), 14 * 1024, DataType::FP32);
            backend_->allreduce(ffn_buffer_.get(), 14 * 1024, DataType::FP32);
        }
    }
    synchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(end - start).count();
    double tok_per_s = 100.0 / total_s;
    
    LOG_INFO("PCIeBAR sustained decode: " << tok_per_s << " tok/s (AllReduce only)");
    
    // Target: AllReduce overhead should allow >10 tok/s
    EXPECT_GT(tok_per_s, 10.0);
}
```

### Success Criteria Table

| Transfer Size | Target Latency | Acceptable | Blocking |
|--------------|----------------|------------|----------|
| 1 KB | < 100 μs | < 200 μs | > 500 μs |
| 14 KB | < 150 μs | < 300 μs | > 700 μs |
| 28 KB | < 200 μs | < 400 μs | > 1 ms |
| 128 KB | < 400 μs | < 800 μs | > 2 ms |

**If blocking thresholds are exceeded**: Fall back to hybrid strategy where NVIDIA uses NCCL internally, AMD uses RCCL internally, and cross-vendor communication uses CPU staging.

---

## Appendix C: Decision Tree for Backend Selection

```
                    ┌─────────────────────────┐
                    │ Need collective op?     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Cross-rank (MPI world)? │
                    └───────────┬─────────────┘
                          yes   │   no
                    ┌───────────┴───────────┐
                    ▼                       ▼
              Use MPI                 Check device group
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              All CUDA?            All ROCm?             Mixed CUDA+ROCm?
                    │                      │                      │
                    ▼                      ▼                      ▼
              Use NCCL             Use RCCL              Check PCIeBAR latency
                                                               │
                                           ┌───────────────────┼───────────────────┐
                                           ▼                                       ▼
                                     < 200μs for 14KB?                      >= 200μs?
                                           │                                       │
                                           ▼                                       ▼
                                     Use PCIeBAR                           Use HOST (CPU staging)
```

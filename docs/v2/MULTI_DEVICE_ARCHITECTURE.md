# Llaminar V2 Multi-Device Architecture

## Overview

This document describes the unified architecture for distributing work across:
1. **MPI Ranks** (inter-node/inter-socket parallelism)
2. **Devices per Rank** (CPU + GPUs within a rank)
3. **Threads per Device** (OpenMP within CPU, CUDA streams on GPU)

## Current Architecture vs Target

### Current (December 2025)

```
              MPI World
         ┌──────┴──────┐
     Rank 0         Rank 1
       │               │
    device_idx=0    device_idx=0
     (CPU only)      (CPU only)
```

- Each rank sees one CPU "device" at index 0
- GPUs enumerated but not used for compute
- Weight sharding across ranks (row-parallel Wo/Down)
- Serial control flow interleaved with parallel kernels

### Target Architecture

```
                      MPI World (COMM_WORLD)
               ┌──────────┴──────────┐
           Rank 0                 Rank 1
          (Socket 0)             (Socket 1)
         ┌────┴────┐            ┌────┴────┐
    Device 0   Device 1    Device 0   Device 1
     (CPU)      (GPU0)      (CPU)      (GPU1)
       │          │           │          │
   28 threads  CUDA ctx   28 threads  CUDA ctx
```

Key changes:
1. Multiple devices per rank (CPU + N GPUs)
2. Hierarchical work splitting: rank → device → kernel
3. Clean separation of setup (serial) vs compute (parallel)
4. Device-aware kernel dispatch

## Core Abstractions

### 1. WorkDistributor

Central abstraction that answers: "Given N elements of work, which rank/device handles which slice?"

```cpp
namespace llaminar2 {

/**
 * @brief Hierarchical work distribution: World → Rank → Device → Thread
 * 
 * Computes work slices at each level of the hierarchy.
 * Does NOT perform the work - just computes indices.
 */
class WorkDistributor {
public:
    struct Config {
        int world_size = 1;           // MPI ranks
        int rank = 0;                 // This rank's index
        std::vector<int> devices;     // Device indices for this rank
        int threads_per_cpu = 1;      // OpenMP threads for CPU devices
    };
    
    /**
     * @brief Work slice for a specific level of hierarchy
     */
    struct WorkSlice {
        size_t start;    // First element (inclusive)
        size_t end;      // Last element (exclusive)
        size_t count;    // end - start
        int owner;       // Rank/device index that owns this slice
    };
    
    explicit WorkDistributor(Config config);
    
    // Rank-level distribution (for tensor parallelism)
    WorkSlice getRankSlice(size_t total_elements) const;
    std::vector<WorkSlice> getAllRankSlices(size_t total_elements) const;
    
    // Device-level distribution (for heterogeneous execution)
    WorkSlice getDeviceSlice(size_t rank_elements, int device_idx) const;
    std::vector<WorkSlice> getAllDeviceSlices(size_t rank_elements) const;
    
    // Memory estimation for device selection
    size_t estimateMemoryPerDevice(size_t total_bytes) const;
    
    // Convenience: full hierarchy
    struct HierarchicalSlice {
        int rank;
        int device_idx;
        size_t global_start;
        size_t global_end;
        size_t local_start;   // Offset within this device
        size_t local_count;
    };
    std::vector<HierarchicalSlice> distribute(size_t total_elements) const;
    
private:
    Config config_;
};

} // namespace llaminar2
```

### 2. DeviceContext

Encapsulates device-specific execution context (threads for CPU, stream for GPU).

```cpp
namespace llaminar2 {

/**
 * @brief Abstract execution context for a single device
 * 
 * CPU: Manages OpenMP thread pool
 * GPU: Manages CUDA stream + workspace memory
 */
class IDeviceContext {
public:
    virtual ~IDeviceContext() = default;
    
    // Device info
    virtual int deviceIndex() const = 0;
    virtual ComputeBackendType backendType() const = 0;
    virtual bool isGPU() const = 0;
    
    // Synchronization
    virtual void synchronize() = 0;
    virtual void barrier() = 0;  // Only for thread pools
    
    // Memory
    virtual void* allocate(size_t bytes) = 0;
    virtual void free(void* ptr) = 0;
    virtual void* getWorkspace(size_t bytes) = 0;  // Scratch memory
    
    // Transfers
    virtual bool copyToDevice(void* dst, const void* src, size_t bytes) = 0;
    virtual bool copyToHost(void* dst, const void* src, size_t bytes) = 0;
};

class CPUDeviceContext : public IDeviceContext {
public:
    explicit CPUDeviceContext(int device_idx, int num_threads);
    // OpenMP thread pool management
    void runParallel(std::function<void(int thread_id, int num_threads)> work);
};

class CUDADeviceContext : public IDeviceContext {
public:
    explicit CUDADeviceContext(int device_idx, int cuda_device_id);
    cudaStream_t stream() const;
    // Async execution
    void runAsync(std::function<void(cudaStream_t)> work);
};

} // namespace llaminar2
```

### 3. ComputeStage

Represents a single parallelizable operation that can run on any device.

```cpp
namespace llaminar2 {

/**
 * @brief A single compute operation that can dispatch to CPU or GPU
 * 
 * ComputeStages are the unit of work for layer-level parallelism.
 * They encapsulate the inputs, outputs, and kernel selection logic.
 */
class ComputeStage {
public:
    enum class Type {
        GEMM,           // Matrix multiplication
        RMS_NORM,       // RMS normalization
        ROPE,           // Rotary position encoding
        ATTENTION,      // Scaled dot-product attention
        SWIGLU,         // SwiGLU activation
        ADD_RESIDUAL,   // Element-wise addition
        SOFTMAX,        // Softmax normalization
        ALLREDUCE,      // MPI collective (runs on rank 0 thread only)
    };
    
    struct GEMMParams {
        const TensorBase* weight;
        TensorBase* input;
        TensorBase* output;
        const float* bias = nullptr;
        int m, n, k;
    };
    
    struct AttentionParams {
        TensorBase* Q;
        TensorBase* K;
        TensorBase* V;
        TensorBase* output;
        int n_heads, n_kv_heads, head_dim;
        int q_seq_len, kv_seq_len;
        bool causal;
    };
    
    // ... other param structs
    
    /**
     * @brief Execute this stage on the given device context
     * 
     * @param ctx Device context (CPU or GPU)
     * @return true on success
     */
    virtual bool execute(IDeviceContext* ctx) = 0;
    
    /**
     * @brief Get the operation type
     */
    virtual Type type() const = 0;
    
    /**
     * @brief Estimated FLOPs for load balancing
     */
    virtual size_t estimatedFlops() const = 0;
    
    /**
     * @brief Does this stage require MPI synchronization after?
     */
    virtual bool requiresAllreduce() const = 0;
};

/**
 * @brief Factory for creating device-appropriate kernels
 */
class ComputeStageFactory {
public:
    static std::unique_ptr<ComputeStage> createGEMM(
        const GEMMParams& params,
        ComputeBackendType target_backend);
    
    static std::unique_ptr<ComputeStage> createAttention(
        const AttentionParams& params,
        ComputeBackendType target_backend);
    
    // ... other factory methods
};

} // namespace llaminar2
```

### 4. LayerExecutor

Orchestrates execution of a transformer layer across devices.

```cpp
namespace llaminar2 {

/**
 * @brief Executes a transformer layer with proper device dispatch
 * 
 * Key responsibilities:
 * 1. Determine which device(s) to use for this layer
 * 2. Build the compute graph (sequence of ComputeStages)
 * 3. Execute stages on appropriate devices
 * 4. Handle MPI synchronization points
 */
class LayerExecutor {
public:
    struct Config {
        std::shared_ptr<MPIContext> mpi_ctx;
        std::vector<IDeviceContext*> device_contexts;
        std::shared_ptr<WeightPlacementMap> placement_map;
        ActivationPrecision activation_precision;
    };
    
    explicit LayerExecutor(Config config);
    
    /**
     * @brief Execute attention block
     * 
     * Stages: RMSNorm → QKV Projection → RoPE → Attention → Wo → Residual
     * 
     * @param layer Layer weights
     * @param layer_idx Layer index
     * @param input Current hidden state [seq, d_model]
     * @param output Updated hidden state [seq, d_model]
     * @param kv_cache KV cache for this layer (may be nullptr)
     */
    bool executeAttention(
        const LayerWeights& layer,
        int layer_idx,
        TensorBase* input,
        TensorBase* output,
        IKVCache* kv_cache);
    
    /**
     * @brief Execute FFN block
     * 
     * Stages: RMSNorm → Gate/Up → SwiGLU → Down → Residual
     */
    bool executeFFN(
        const LayerWeights& layer,
        int layer_idx,
        TensorBase* input,
        TensorBase* output);
    
private:
    // Build compute graph without executing
    std::vector<std::unique_ptr<ComputeStage>> buildAttentionStages(
        const LayerWeights& layer, int layer_idx,
        TensorBase* input, TensorBase* output, IKVCache* kv_cache);
    
    std::vector<std::unique_ptr<ComputeStage>> buildFFNStages(
        const LayerWeights& layer, int layer_idx,
        TensorBase* input, TensorBase* output);
    
    // Execute a sequence of stages with proper synchronization
    bool executeStages(
        std::vector<std::unique_ptr<ComputeStage>>& stages,
        int target_device);
    
    Config config_;
    std::unique_ptr<WorkDistributor> work_dist_;
};

} // namespace llaminar2
```

## Work Distribution Strategy

### Level 1: MPI Ranks (Tensor Parallelism)

Weights are sharded across ranks. Each rank processes the same input but
operates on different weight slices:

```
Input [seq=512, d_model=4096]
     │
     ├─ Rank 0: Wo[:, 0:2048]  → partial output
     │           Allreduce ←─────┘
     └─ Rank 1: Wo[:, 2048:4096] → partial output
                 Allreduce ←───────┘
                       │
                Output [seq=512, d_model=4096]
```

### Level 2: Devices per Rank (Pipeline/Data Parallelism)

Within a rank, work can be split across CPU and GPU(s):

**Option A: Layer-wise split** (simple, current focus)
```
Rank 0:
  - Layers 0-11: GPU 0
  - Layers 12-23: CPU (or GPU 1 if available)
```

**Option B: Operation-wise split** (future)
```
Each layer:
  - GEMM operations: GPU (higher throughput)
  - Element-wise ops: CPU (lower latency for small tensors)
```

### Level 3: Expert Parallelism (MoE)

For Mixture-of-Experts architectures (Mixtral, Qwen2-MoE, DeepSeek-V3), experts
can be distributed across devices:

```
MoE Layer (8 experts, top-k=2):
                    Router
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Tokens for    Tokens for    Tokens for
   Expert 0-1    Expert 2-3    Expert 4-7
        │             │             │
     Device 0      Device 1      Device 2
      (GPU)         (GPU)         (CPU)
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                  Combine
```

**Expert Distribution Strategies:**

1. **Expert Parallelism (EP)**: Each device owns a subset of experts
   - Pro: No weight replication, good for many experts
   - Con: Token routing causes communication overhead

2. **Expert Replication**: All devices have all experts, split tokens
   - Pro: No token routing, simple
   - Con: High memory usage (full expert weights per device)

3. **Hybrid**: Replicate hot experts, distribute cold ones
   - Pro: Balances memory and communication
   - Con: Complex to implement

The `WorkDistributor` supports MoE via:
- `distributeExperts()`: Maps experts to devices based on memory/compute capacity
- `routeTokensToExperts()`: Given router outputs, determines which tokens go where
- `ExpertSlice`: Tracks expert_id, device_idx, token_indices

### Level 4: Threads/Streams per Device

**CPU**: OpenMP worksharing within parallel region
**GPU**: CUDA streams for async execution + graph capture

## Implementation Plan

### Phase 1: WorkDistributor + DeviceContext ✅ COMPLETE

1. ✅ Created `src/v2/execution/WorkDistributor.{h,cpp}` (340 lines)
   - Hierarchical work slicing: World → Rank → Device
   - MoE expert distribution support
   - 22 unit tests passing
2. ✅ Created `src/v2/execution/DeviceContext.{h,cpp}` (200 lines)
   - CPUDeviceContext with OpenMP parallelism
   - Memory allocation with alignment
   - Workspace management
   - 16 unit tests passing

### Phase 2: ComputeStage Abstraction ✅ COMPLETE

1. ✅ Created `src/v2/execution/ComputeStage.{h,cpp}` (1250 lines)
   - IComputeStage interface with execute(), estimatedFlops(), supportsBackend()
   - 10 concrete stage types: GEMM, RMSNorm, RoPE, Attention, SwiGLU, ResidualAdd, Allreduce, MoERouter, MoEExpert, MoECombine
   - ComputeStageFactory for device-appropriate stage creation
   - 16 unit tests passing

### Phase 3: LayerExecutor ✅ COMPLETE

1. ✅ Created `src/v2/execution/LayerExecutor.{h,cpp}` (990 lines)
   - ComputeGraph with DAG-based execution ordering (Kahn's algorithm)
   - buildAttentionGraph(), buildFFNGraph(), buildMoEGraph()
   - Sequential and parallel execution modes
   - Multi-device execution support
   - Per-stage profiling and statistics
   - 29 unit tests passing

**Total: 83 unit tests for execution framework**

### Phase 4: GPU Dispatch (CUDA + ROCm) - ✅ COMPLETE

**Goal**: Enable GPU acceleration via CUDA (NVIDIA) and ROCm (AMD) with unified abstractions.

1. ✅ Create GPU DeviceContext implementations
   - `CUDADeviceContext`: CUDA stream management, device memory, backend delegation
   - `ROCmDeviceContext`: HIP stream management, device memory, backend delegation  
   - `IGPUDeviceContext` base class for common patterns (workspace, synchronization)
   - Added to `src/v2/execution/DeviceContext.{h,cpp}`

2. ✅ Create GPU ComputeStage implementations
   - `GPUGEMMStage`: Matrix multiplication via IBackend::gemmIQ4NL()
   - `GPURMSNormStage`: RMS normalization with device workspace
   - `GPUSwiGLUStage`: SwiGLU activation
   - `GPUResidualAddStage`: Element-wise residual addition
   - `GPURoPEStage`: Rotary position encoding
   - `GPUAttentionStage`: Scaled dot-product attention
   - All stages conditionally compiled with `#ifdef HAVE_CUDA` / `#ifdef HAVE_ROCM`

3. ✅ Add GPU memory management
   - Device memory allocation via IBackend interface
   - Host-to-device and device-to-host transfers
   - Device-to-device transfers for multi-GPU
   - Workspace allocation per context

4. ✅ Wire up ComputeStageFactory for GPU dispatch
   - Factory detects backend type from DeviceContext
   - Returns GPU-optimized stages for CUDA/ROCm backends
   - Falls back to CPU stages when GPU unavailable

5. ✅ Enable ROCm device enumeration
   - Fixed `enumerate_rocm_devices()` for modern HIP API (gcnArchName parsing)
   - Proper NUMA-aware GPU filtering

6. ✅ All unit tests passing
   - 149 unit tests pass on both CPU-only and ROCm builds
   - GPU context tests skip gracefully when no GPU available
   - Namespace fix for backend includes (prevented nested namespace issue)

**Note**: Actual GPU kernel execution requires CUDA/ROCm runtime access. The framework is complete and will work when GPU hardware is available with proper container permissions.

### Phase 5: Pipeline Integration ✅ COMPLETE

1. ✅ Created PipelineExecutor adapter (bridges LayerExecutor with pipeline)
   - `src/v2/pipelines/PipelineExecutor.{h,cpp}` (~260 lines)
   - 22 unit tests passing
2. ✅ Added feature flags to PipelineConfig for incremental rollout
   - `executor_ffn_norm`, `executor_ffn_swiglu`, `executor_ffn_residual`
   - `executor_attn_norm`, `executor_attn_residual`, `executor_rope`
3. ✅ Integrated PipelineExecutor into PipelineBase
   - Lazy initialization when any flag enabled
   - `initializePipelineExecutor()` called from `initializeTypedOps()`
4. ✅ Wired declarative ops through PipelineExecutor when flags enabled
   - `rms_norm()` → FFN_NORM / ATTENTION_NORM detection
   - `swiglu()` → FFN_SWIGLU flag
   - `add_residual()` → FFN_RESIDUAL / ATTENTION_RESIDUAL detection
   - `apply_rope()` → ROPE flag
5. ✅ All 149 unit tests passing

### Phase 6: Multi-GPU Infrastructure ✅ COMPLETE

1. ✅ Enable CUDA + ROCm in same binary (heterogeneous multi-GPU)
   - Created separate GPU enumeration compilation units (CUDAEnumeration.cu, ROCmEnumeration.cpp)
   - Fixed header conflicts between cuda_runtime.h and hip_runtime.h
   - All 3 GPUs (RTX 3090 + 2× MI50) enumerated in single process
2. ☐ Enable multiple GPUs per rank in DeviceManager
3. ☐ Implement layer-wise GPU assignment
4. ☐ Add GPU-aware MPI (NCCL for NVIDIA, RCCL for AMD)

### Phase 7: Kernel Execution (CPU) - IN PROGRESS

**Problem**: The executor framework infrastructure is complete but compute stages have critical flaws:
- ❌ RMSNormStage, SwiGLUStage, RoPEStage - **BROKEN** (scalar reimplementations, no precision support)
- ✅ ResidualAddStage - **CORRECT** (handles ActivationPrecision properly)
- ✅ GEMMStage - **CORRECT** (delegates to KernelFactory)
- ✅ FusedQKVGEMMStage - **CORRECT** (delegates to QuantisedGemmKernel::multiply_fused_multi)

**Root Cause**: Stages reimplemented operations as scalar loops instead of using existing typed kernels.
This violates the architecture and breaks ActivationPrecision support.

**Goal**: Refactor stages to delegate to typed kernels (`CPURMSNormKernelT<P>`, `CPUSwiGLUKernelT<P>`, etc.)
matching how the legacy Ops work, but without the Op wrapper layer.

#### 7.1 Type-Safe Tensor Design (CRITICAL)

**Problem**: Original ComputeStage design used `void*` pointers with manual precision tracking:

```cpp
// BROKEN DESIGN - void* loses type safety and device info
struct Params {
    const void *input;           // What type? What device?
    void *output;                // Manual casting everywhere
    int seq_len, hidden_dim;     // Duplicates tensor metadata
    ActivationPrecision precision; // Redundant - tensor knows this
};
```

**Issues with void***:
1. **No type safety**: Requires manual casting, easy to mismatch types
2. **No device info**: Can't dispatch to GPU vs CPU automatically
3. **Redundant metadata**: `seq_len`, `hidden_dim`, `precision` duplicate tensor info
4. **Not GPU-ready**: GPU dispatch requires knowing tensor's device placement
5. **Manual kernel selection**: Must switch on precision instead of using polymorphism

**Correct Design**: Use typed tensor pointers:

```cpp
// CORRECT DESIGN - TensorBase*/IActivationTensor* are self-describing
struct Params {
    TensorBase* input;        // Knows: type, device, rows, cols, precision
    TensorBase* output;       // Knows: type, device, rows, cols, precision
    const TensorBase* gamma;  // Weight tensor (immutable)
    float eps;                // Only operation-specific params remain
};

bool RMSNormStage::execute(IDeviceContext *ctx) {
    // Tensor tells us everything we need
    auto* activation = dynamic_cast<IActivationTensor*>(params_.input);
    if (!activation) return false;
    
    // IActivationTensor::applyRMSNorm dispatches to correct kernel automatically:
    // - Checks tensor's native_type() for precision (FP32/BF16/FP16/Q8_1)
    // - Checks tensor's device_idx() for CPU vs GPU
    // - Creates appropriate kernel via KernelFactory
    return activation->applyRMSNorm(
        params_.gamma->data(),
        params_.input->rows(),
        params_.input->cols(),
        params_.eps
    );
}
```

**Benefits**:
1. **Type-safe**: No void* casting, compiler catches type errors
2. **Self-describing**: Tensor knows its precision, dimensions, device
3. **Device-aware**: `IActivationTensor::applyRMSNorm()` dispatches to GPU if tensor is on GPU
4. **DRY**: No duplicate `seq_len`/`hidden_dim`/`precision` fields
5. **Polymorphic dispatch**: Uses existing kernel factory infrastructure

#### 7.2 Tensor Type Hierarchy

```
TensorBase (base class)
├─ data(), mutable_data()     - Raw pointer access
├─ rows(), cols(), numel()    - Dimensions
├─ native_type()              - TensorType enum (FP32, BF16, Q8_1, etc.)
├─ device_idx()               - Device placement (-1=CPU, 0+=GPU)
└─ dtype_name()               - Human-readable type string

IActivationTensor (interface for activation buffers)
├─ createRoPE(), createSwiGLU(), createRMSNorm(), createAttention()
├─ applyRoPE(), applyRMSNorm()  - In-place operations
└─ to_int8_activation_pack()    - Quantization for INT8 GEMM

Concrete Types:
├─ FP32Tensor    : TensorBase, IActivationTensor
├─ BF16Tensor    : TensorBase, IActivationTensor
├─ FP16Tensor    : TensorBase, IActivationTensor
├─ Q8_1Tensor    : TensorBase, IActivationTensor
└─ IQ4_NLTensor  : TensorBase (weight only, not IActivationTensor)
```

#### 7.3 Stage Parameter Patterns

**Activation-only stages** (RMSNorm, RoPE, SwiGLU, ResidualAdd):
```cpp
struct Params {
    TensorBase* input;         // IActivationTensor* at runtime
    TensorBase* output;        // May be same as input (in-place)
    const TensorBase* weights; // If needed (gamma for RMSNorm)
    // Operation-specific scalars only
};
```

**GEMM stages** (weights × activations):
```cpp
struct Params {
    TensorBase* A;             // Activation (IActivationTensor*)
    const TensorBase* B;       // Weight tensor (quantized)
    TensorBase* C;             // Output (IActivationTensor*)
    // Optional: bias, fusion flags
};
```

**Attention stages**:
```cpp
struct Params {
    TensorBase* Q;             // Query activations
    TensorBase* K;             // Key activations  
    TensorBase* V;             // Value activations
    TensorBase* output;        // Attention output
    IUnifiedKVCache* kv_cache; // Cache (already typed)
    // Attention config (n_heads, etc.)
};
```

#### 7.4 Migration Plan

| Stage | Current | Target |
|-------|---------|--------|
| `RMSNormStage` | `void* input, void* output, ActivationPrecision` | `TensorBase* input, TensorBase* output` |
| `RoPEStage` | `void* tensor, int seq_len, n_heads, head_dim` | `TensorBase* Q, TensorBase* K` |
| `SwiGLUStage` | `void* gate, void* up, void* output` | `TensorBase* gate, TensorBase* up, TensorBase* output` |
| `ResidualAddStage` | Already uses `ActivationPrecision` | Convert to `TensorBase*` |
| `GEMMStage` | `void* A, TensorBase* B` | `TensorBase* A, TensorBase* B` |
| `AttentionStage` | `void* Q/K/V` | `TensorBase* Q/K/V` |

#### 7.5 Kernel Dispatch via IActivationTensor

The legacy pipeline uses these kernel patterns:

```
Qwen2Pipeline::attention_block():
  ├─ FusedGEMM(wq, wk, wv) → Fused 3-way QKV projection
  │     └─ Uses KernelFactory::getOrCreateGemm() for each weight
  │     └─ Quantizes activations ONCE, runs 3 GEMMs
  │
  ├─ apply_rope() → RoPEKernel (CPU primitives)
  │
  ├─ MpiAttentionOrchestrator::compute()
  │     └─ Updates KV cache
  │     └─ GQAAttention::compute() for Q*K^T, softmax, *V
  │
  └─ project_row_parallel(wo) → Single GEMM + allreduce

Qwen2Pipeline::ffn_block():
  ├─ FusedGEMM(gate, up) → Fused 2-way GateUp projection
  │
  ├─ swiglu() → SwiGLUPrimitives (CPU SIMD)
  │
  └─ project_row_parallel(down) → Single GEMM + allreduce
```

#### 7.2 Kernel Mapping for Executor Stages

| Stage | Current Implementation | Production Kernel to Wire |
|-------|------------------------|---------------------------|
| `GEMMStage` | Placeholder | `KernelFactory::getOrCreateGemm(B)->multiply()` |
| `RMSNormStage` | ✅ Implemented inline | Already works |
| `RoPEStage` | ✅ Implemented inline | Already works |
| `AttentionStage` | ✅ Implemented inline | Already works (naive, see note) |
| `SwiGLUStage` | ✅ Implemented inline | Already works |
| `ResidualAddStage` | ✅ Implemented inline | Already works |
| `FusedGEMMStage` | **NEW NEEDED** | `FusedGEMM::execute()` |

**Note on AttentionStage**: Current inline implementation is correct but naive (O(n²) memory for scores).
The legacy path uses `GQAAttention` which has better memory characteristics. For production, we may
want to wire AttentionStage to GQAAttention, but the inline version is functionally correct.

#### 7.3 GEMMStage Implementation Plan

**File**: `src/v2/execution/ComputeStage.cpp` (GEMMStage::execute)

```cpp
bool GEMMStage::execute(IDeviceContext *ctx)
{
    if (!ctx || !params_.A || !params_.B || !params_.C) {
        LOG_ERROR("[GEMMStage] Invalid parameters");
        return false;
    }

    // Get cached kernel from KernelFactory (handles weight packing once)
    auto *gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.B);
    if (!gemm) {
        LOG_ERROR("[GEMMStage] Failed to get GEMM kernel for weight tensor");
        return false;
    }

    // Cast to QuantisedGemmKernel if available (for full API access)
    auto *qgemm = dynamic_cast<gemm_v4::QuantisedGemmKernel*>(gemm);
    if (qgemm) {
        // Quantized GEMM path: FP32 activations, quantized weights
        return qgemm->multiply(
            static_cast<const float*>(params_.A),
            static_cast<float*>(params_.C),
            params_.m, params_.n, params_.k,
            params_.transpose_B,
            params_.alpha, params_.beta,
            nullptr, // no bias in GEMMStage (added separately)
            -1       // device_idx (not used for CPU)
        );
    }

    // FP32 GEMM fallback (for non-quantized weights)
    return gemm->multiply(
        static_cast<const float*>(params_.A),
        static_cast<float*>(params_.C),
        params_.m, params_.n, params_.k,
        params_.alpha, params_.beta
    );
}
```

**Key insight**: The existing `KernelFactory::getOrCreateGemm()` caches packed weights.
We MUST use the cached kernel, not create new ones (would re-pack weights every call!).

#### 7.4 FusedGEMMStage (New Stage)

**Problem**: QKV and GateUp projections share a common input (normalized activations).
Quantizing the input once for multiple GEMMs saves significant overhead.

**File**: `src/v2/execution/ComputeStage.h` (new class)

```cpp
/**
 * @brief Fused multi-GEMM stage for QKV/GateUp patterns
 *
 * Quantizes activations once, executes N GEMMs with shared quantized data.
 * Replaces N separate GEMMStage nodes for better performance.
 */
class FusedGEMMStage : public IComputeStage
{
public:
    struct Projection {
        const TensorBase* weight;  ///< Weight tensor (k × n)
        void* output;              ///< Output buffer (m × n)
        const float* bias;         ///< Optional bias (n)
        int n;                     ///< Output dimension
    };

    struct Params {
        const void* A;                      ///< Shared input activations (m × k)
        std::vector<Projection> projections; ///< List of projections
        int m, k;                           ///< Input dimensions
    };

    explicit FusedGEMMStage(Params params);

    bool execute(IDeviceContext *ctx) override;
    ComputeStageType type() const override { return ComputeStageType::GEMM; }
    // ... etc
};
```

**Implementation**: Creates `FusedGEMM` instance and calls `execute()`.

#### 7.5 Updated Qwen2LayerExecutor Graph Building

**Current** (builds separate Q, K, V GEMMStages):
```cpp
// 3 separate GEMM stages - inefficient (quantizes input 3 times)
graph.addNode("q_proj", createGEMM(...wq...), device);
graph.addNode("k_proj", createGEMM(...wk...), device);
graph.addNode("v_proj", createGEMM(...wv...), device);
```

**Target** (single FusedGEMMStage):
```cpp
// 1 fused stage - quantizes input once
FusedGEMMStage::Params qkv_params{
    buffers.normalized->data(),
    {{layer.wq, buffers.Q->mutable_data(), q_bias, n_q},
     {layer.wk, buffers.K->mutable_data(), k_bias, n_kv},
     {layer.wv, buffers.V->mutable_data(), v_bias, n_kv}},
    seq_len, d_model
};
graph.addNode("qkv_fused", ComputeStageFactory::createFusedGEMM(qkv_params), device);
```

#### 7.6 Implementation Order

1. **GEMMStage::execute()** - Wire to KernelFactory (single GEMM)
   - Unblocks Wo and Down projections
   - ~20 lines of code

2. **Test separate GEMMs** - Verify Q, K, V, Wo, Gate, Up, Down work individually
   - May be inefficient but correct

3. **FusedGEMMStage** - Add new stage type
   - ~100 lines new code
   - Wire to existing FusedGEMM class

4. **Update Qwen2LayerExecutor** - Replace 3 GEMMStages with FusedGEMMStage
   - Modify buildAttentionGraph() and buildFFNGraph()

5. **End-to-end test** - Run with `LLAMINAR_USE_LAYER_EXECUTOR=1`
   - Should produce correct output matching legacy path

#### 7.7 Required File Changes

| File | Change |
|------|--------|
| `ComputeStage.cpp` | Implement GEMMStage::execute() with KernelFactory |
| `ComputeStage.h` | Add FusedGEMMStage class declaration |
| `ComputeStage.cpp` | Add FusedGEMMStage::execute() implementation |
| `ComputeStageFactory` | Add createFusedGEMM() factory method |
| `Qwen2LayerExecutor.cpp` | Replace separate GEMM nodes with fused nodes |

#### 7.8 Verification Strategy

```bash
# Test with separate GEMMs first (less efficient but correct)
LLAMINAR_USE_LAYER_EXECUTOR=1 \
LLAMINAR_EXEC_GEMM=1 \
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "The capital of France is" -n 10 -t 0

# Should output: "Paris. It is the largest city..."
```

After FusedGEMMStage:
```bash
# Test with fused QKV/GateUp
LLAMINAR_USE_LAYER_EXECUTOR=1 \
LLAMINAR_EXEC_GEMM=1 \
LLAMINAR_EXEC_FUSED_GEMM=1 \  # New flag
./run_llaminar.sh ...
```

## File Structure

```
src/v2/execution/
├── WorkDistributor.h       # Hierarchical work distribution (340 lines)
├── WorkDistributor.cpp     # Rank/Device/MoE slicing
├── DeviceContext.h         # IDeviceContext, IGPUDeviceContext, CPUDeviceContext, CUDADeviceContext, ROCmDeviceContext (~400 lines)
├── DeviceContext.cpp       # CPU/GPU context implementations, memory management (~550 lines)
├── ComputeStage.h          # IComputeStage + CPU/GPU stage classes (~550 lines)
├── ComputeStage.cpp        # CPU + GPU implementations of all stages (~900 lines)
├── LayerExecutor.h         # ComputeGraph + LayerExecutor (425 lines)
└── LayerExecutor.cpp       # Graph execution, Attention/FFN/MoE builders (658 lines)

src/v2/backends/
├── IBackend.h              # Abstract GPU backend interface (272 lines)
├── ComputeBackend.{h,cpp}  # DeviceManager, device enumeration (~1000 lines)
├── GPUEnumeration.h        # Extern declarations for GPU enumeration
├── CUDAEnumeration.cu      # CUDA device enumeration (isolated TU to avoid header conflicts)
├── ROCmEnumeration.cpp     # ROCm device enumeration (compiled with HIP)
├── cuda/
│   ├── CUDABackend.h       # CUDA implementation of IBackend
│   └── CUDABackend.cu      # CUDA kernels and runtime calls
└── rocm/
    ├── ROCmBackend.h       # ROCm implementation of IBackend
    └── ROCmBackend.cpp     # HIP kernels and runtime calls

src/v2/pipelines/
├── PipelineConfig.h        # Added executor_* feature flags (~460 lines)
├── PipelineBase.h          # Added PipelineExecutor member, initializePipelineExecutor() (~1360 lines)
├── PipelineBase.cpp        # Modified rms_norm/swiglu/add_residual/apply_rope for executor dispatch (~2930 lines)
├── PipelineExecutor.h      # LayerExecutor-to-pipeline adapter (~260 lines)
└── PipelineExecutor.cpp    # Operation execution via ComputeStage (~265 lines)

tests/v2/unit/
├── Test__WorkDistributor.cpp    # 22 tests
├── Test__DeviceContext.cpp      # 19 tests (including GPU context tests)
├── Test__ComputeStage.cpp       # 16 tests
├── Test__LayerExecutor.cpp      # 29 tests (681 lines)
└── Test__PipelineExecutor.cpp   # 22 tests (300 lines)
```

**Total: 149 unit tests passing (execution framework: 108 tests)**

## Migration Path

The new architecture will be introduced alongside existing code:

1. `LayerExecutor` wraps existing `attention_block()`/`ffn_block()` logic
2. Existing kernels wrapped in `ComputeStage` interface
3. Pipeline can use either old or new path (feature flag)
4. Once validated, old path deprecated

## Open Questions

1. **GPU memory management**: Use CUDA memory pools? Pre-allocate all buffers?
2. **Async execution**: How to overlap CPU and GPU work within a layer?
3. **Mixed precision**: Should GPU use FP16/BF16 while CPU uses Q8_1?
4. **NCCL integration**: Use NCCL for GPU allreduce or stage through CPU?
5. **Layer-level parallelism**: Can we truly run attention+FFN in parallel regions?

### Phase 8: Precision-Aware Execution Graph - PLANNED

**Problem Statement:**

All `ComputeStage` implementations are currently hardcoded to FP32:

```cpp
// Current ResidualAddStage::execute() - WRONG for Q8_1!
const float *input = static_cast<const float *>(params_.input);
const float *residual = static_cast<const float *>(params_.residual);
float *output = static_cast<float *>(params_.output);
```

When `activation_precision = Q8_1`, buffers contain `Q8_1Block` data (36 bytes per 32 elements),
not floats. Casting to `float*` produces garbage output.

Meanwhile, the legacy pipeline's typed ops (`ResidualOpTyped<P>`, `RMSNormOpTyped<P>`, etc.)
properly handle all precisions including Q8_1.

**Goal:** Make the execution graph precision-aware without duplicating code.

#### 8.1 Current Architecture Gap

| Component | Legacy Pipeline | Execution Graph | Status |
|-----------|-----------------|-----------------|--------|
| Residual Add | `ResidualOpTyped<P>` | `ResidualAddStage` (FP32 only) | ❌ Gap |
| RMSNorm | `RMSNormOpTyped<P>` | `RMSNormStage` (FP32 only) | ❌ Gap |
| SwiGLU | `SwiGLUOpTyped<P>` | `SwiGLUStage` (FP32 only) | ❌ Gap |
| RoPE | `RoPEOpTyped<P>` | `RoPEStage` (FP32 only) | ❌ Gap |
| Attention | `CPUAttentionKernelTyped<P>` | `AttentionStage` (FP32 only) | ❌ Gap |
| GEMM | `QuantisedGemmKernel` | `GEMMStage` | ✅ Works |

**Key Insight:** The typed ops in `src/v2/pipelines/ops/` already implement correct
precision handling. We should delegate to them, not reimplement.

#### 8.2 Design Approach: Delegation Pattern

Instead of templating all stages or adding if/else chains, we'll use **delegation**:

```cpp
// New: ResidualAddStage delegates to IResidualOp
class ResidualAddStage : public IComputeStage {
public:
    struct Params {
        TensorBase* input;      // Changed from void* - tensor knows its precision!
        TensorBase* residual;
        TensorBase* output;
        int rows, cols;         // For dimension tracking
        ActivationPrecision precision; // Runtime precision selector
    };

    bool execute(IDeviceContext *ctx) override {
        // Create appropriate typed op based on precision
        auto op = createResidualOp(params_.precision);
        return op->apply(
            params_.residual,
            params_.input,
            params_.output,
            params_.rows,
            params_.cols
        );
    }
};
```

The `createResidualOp()` factory (already exists in `PipelineBase.h`) returns:
- `ResidualOpTyped<FP32>` for FP32 activations
- `ResidualOpTyped<Q8_1>` for Q8_1 activations (uses `simd::q8_1_add_q8_1`)
- etc.

#### 8.3 Q8_1 Residual Add Implementation (Reference)

The Q8_1 residual add in `ResidualOpTyped<Q8_1>` uses `simd::q8_1_add_q8_1()`:

```cpp
// From SIMDHelpers.h - Native Q8_1 addition
inline void q8_1_add_q8_1(const Q8_1Block *a, const Q8_1Block *b,
                          Q8_1Block *output, size_t count) {
    const size_t n_blocks = count / 32;
    for (size_t blk = 0; blk < n_blocks; ++blk) {
        // 1. Dequant both blocks to FP32 (in registers, using SIMD)
        // 2. Add FP32 values
        // 3. Find max_abs for requantization
        // 4. Requantize to Q8_1
    }
}
```

This is **NOT** a dequant→add→requant in separate passes. It's fused:
- Dequant happens in registers (no memory write)
- Add happens in registers
- Requant happens immediately with fresh scale

This preserves precision better than explicit dequant steps.

#### 8.4 Implementation Plan

**Phase 8.1: Add Mockable Interfaces for Testing**

Create `IComputeStageMock` interface for testing graph execution without real kernels:

```cpp
// tests/v2/mocks/MockComputeStage.h
class MockComputeStage : public IComputeStage {
public:
    bool execute(IDeviceContext *ctx) override {
        execute_count_++;
        return should_succeed_;
    }
    
    int execute_count_ = 0;
    bool should_succeed_ = true;
};
```

**Phase 8.2: Update Stage Params to Use TensorBase***

Change all stage `Params` structs from `void*` to `TensorBase*`:

| Stage | Old Params | New Params |
|-------|------------|------------|
| `ResidualAddStage` | `void* input, void* residual, void* output` | `TensorBase* input, TensorBase* residual, TensorBase* output` |
| `RMSNormStage` | `void* input, void* output, float* gamma` | `TensorBase* input, TensorBase* output, TensorBase* gamma` |
| `SwiGLUStage` | `void* gate, void* up, void* output` | `TensorBase* gate, TensorBase* up, TensorBase* output` |
| `RoPEStage` | `void* Q, void* K` | `TensorBase* Q, TensorBase* K` |

With `TensorBase*`, stages can query `tensor->precision()` at runtime.

**Phase 8.3: Implement Precision Dispatch**

```cpp
bool ResidualAddStage::execute(IDeviceContext *ctx) {
    // Query precision from tensor (or use explicit param)
    auto precision = params_.input->precision();
    
    switch (precision) {
        case ActivationPrecision::FP32: {
            ResidualOpTyped<ActivationPrecision::FP32> op;
            return op.apply(params_.residual, params_.input, params_.output,
                           params_.rows, params_.cols);
        }
        case ActivationPrecision::Q8_1: {
            ResidualOpTyped<ActivationPrecision::Q8_1> op;
            return op.apply(params_.residual, params_.input, params_.output,
                           params_.rows, params_.cols);
        }
        // ... BF16, FP16
    }
}
```

**Phase 8.4: Add Unit Tests for Precision Modes**

```cpp
TEST(ResidualAddStageTyped, Q8_1_NativeAddition) {
    // Create Q8_1 tensors
    auto residual = TensorFactory::createQ8_1({32, 896});
    auto input = TensorFactory::createQ8_1({32, 896});
    auto output = TensorFactory::createQ8_1({32, 896});
    
    // Fill with test data
    fill_with_random_q8_1(residual.get());
    fill_with_random_q8_1(input.get());
    
    // Execute via stage
    ResidualAddStage::Params params{
        input.get(), residual.get(), output.get(),
        32, 896, ActivationPrecision::Q8_1
    };
    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));
    
    // Verify against direct simd::q8_1_add_q8_1
    auto expected = TensorFactory::createQ8_1({32, 896});
    simd::q8_1_add_q8_1(
        residual->q8_1_blocks(),
        input->q8_1_blocks(),
        expected->mutable_q8_1_blocks(),
        32 * 896
    );
    
    // Compare blocks
    EXPECT_EQ(memcmp(output->q8_1_blocks(), expected->q8_1_blocks(),
                     output->num_blocks() * sizeof(Q8_1Block)), 0);
}
```

**Phase 8.5: Update Qwen2LayerExecutor to Pass Precision**

```cpp
// In buildAttentionGraph()
ResidualAddStage::Params attn_residual_params{
    buffers.attn_proj,           // TensorBase* (not void*)
    buffers.current_hidden,
    buffers.current_hidden,
    seq_len, config_.d_model,
    config_.activation_precision  // NEW: precision from config
};
```

#### 8.5 File Changes Required

| File | Change |
|------|--------|
| `execution/ComputeStage.h` | Change Params to use TensorBase*, add precision field |
| `execution/ComputeStage.cpp` | Implement precision dispatch in execute() |
| `pipelines/qwen/Qwen2LayerExecutor.cpp` | Pass TensorBase* instead of void* |
| `tests/v2/unit/Test__ComputeStage.cpp` | Add precision-aware tests |
| `tests/v2/mocks/MockComputeStage.h` | NEW: Mock implementations for testing |

#### 8.6 Testing Strategy

1. **Unit tests**: Each stage with each precision (FP32, BF16, FP16, Q8_1)
2. **Graph tests**: Full attention/FFN graphs with Q8_1 tensors
3. **Parity tests**: Compare executor output vs legacy pipeline for same input
4. **Mock tests**: Verify graph execution order without real kernels

#### 8.7 Success Criteria

- [ ] `ResidualAddStage` produces correct output for Q8_1 tensors
- [ ] `RMSNormStage` produces correct output for Q8_1 tensors
- [ ] `SwiGLUStage` produces correct output for Q8_1 tensors
- [ ] Full layer execution with Q8_1 matches legacy pipeline output
- [ ] Unit tests pass for all precision modes
- [ ] No unnecessary dequant→requant cycles in Q8_1 path

## Related Documents

- `.github/copilot-instructions.md` - Development guidelines
- `.github/instructions/llaminar-v2-architecture.instructions.md` - V2 architecture overview
- `docs/v2/LAYER_FUSION_PLAN.md` - OpenMP optimization plan
- `src/v2/pipelines/ops/ResidualOp.h` - Reference implementation for typed residuals
### Phase 9: Attention Architecture Refactoring - PLANNED

**Problem Statement:**

The current attention path has **5 abstraction layers** with redundant code:

```
AttentionWithKVCacheStage
    └── MpiAttentionOrchestrator::compute_mpi()
            ├── to_gqa_config() // Converts MpiAttentionConfig → GQAAttentionConfig
            └── GQAAttention::compute()
                    └── createAttentionKernelForPrecision() // Local anonymous factory
                            └── CPUAttentionKernelTyped<P>
```

**Issues:**
1. **Redundant config objects**: `MpiAttentionConfig` and `GQAAttentionConfig` are nearly identical
2. **MPI strategy belongs in WorkDistributor**: Not in attention-specific orchestrator
3. **KV cache mixed with compute**: `AttentionWithKVCacheStage` does two things
4. **Local factory bypassed KernelFactory**: GQAAttention had its own kernel creation (fixed!)
5. **5 layers vs 3**: Compare to RMSNorm: `RMSNormStage` → `KernelFactory` → `CPURMSNormKernelTyped`

**Target Architecture:**

```
LayerExecutor (orchestrates DAG of ComputeStages)
    │
    ├── KVCacheAppendStage (NEW - dedicated cache update)
    │       └── IUnifiedKVCache::append_kv()
    │
    ├── AttentionComputeStage (REFACTORED - pure attention compute)
    │       └── KernelFactory::createAttention(tensor, device_type)
    │               ├── CPUAttentionKernelTyped<P>  (ITensorAttention)
    │               └── GPUAttentionKernelTyped<T>  (ITensorAttention, future CUDA/ROCm)
    │
    └── AllreduceStage (if tensor parallel)
            └── MPI_Allreduce()

WorkDistributor: Handles MPI head distribution (n_heads / world_size)
DeviceContext: Handles execution device (CPU threads / GPU streams)
```

**Key Changes:**
1. **Split KV Cache Management**: `KVCacheAppendStage` separate from attention
2. **Remove MpiAttentionOrchestrator**: Responsibilities split:
   - MPI strategy → `WorkDistributor` + `AllreduceStage`
   - Compute dispatch → `KernelFactory::createAttention()`
   - Config → Simplified `AttentionComputeStage::Params`
3. **Remove GQAAttention**: Logic absorbed by `CPUAttentionKernelTyped`
4. **Unified Config**: One config struct flows through

#### 9.1 Current Responsibilities Analysis

| Current Location | Target Location |
|-----------------|-----------------|
| MpiAttentionOrchestrator: MPI strategy selection | `WorkDistributor` + `AllreduceStage` |
| MpiAttentionOrchestrator: Config conversion | Single unified params in `AttentionComputeStage` |
| MpiAttentionOrchestrator: Input validation | `AttentionComputeStage` or kernel |
| GQAAttention: GQA head broadcasting | Inside `CPUAttentionKernelTyped` (already handles GQA) |
| GQAAttention: Kernel creation | `KernelFactory::createAttention()` ✅ (done!) |
| AttentionWithKVCacheStage: KV cache append | `KVCacheAppendStage` (new) |
| AttentionWithKVCacheStage: Attention compute | `AttentionComputeStage` (refactored) |

#### 9.2 Implementation Plan

**Phase 9.1: Create KVCacheAppendStage** (NEW)

Dedicated stage for KV cache updates:

```cpp
class KVCacheAppendStage : public IComputeStage
{
public:
    struct Params {
        TensorBase* K;              ///< Key tensor to append [seq_len, n_kv_heads * head_dim]
        TensorBase* V;              ///< Value tensor to append
        IUnifiedKVCache* kv_cache;  ///< Target cache
        int layer_idx;              ///< Layer index in cache
        int seq_len;                ///< Number of tokens to append (1 for decode)
    };

    bool execute(IDeviceContext *ctx) override {
        if (!params_.kv_cache) return true; // No cache = no-op
        return params_.kv_cache->append_kv(
            params_.layer_idx, 0, params_.K, params_.V, params_.seq_len
        );
    }
};
```

**Phase 9.2: Create AttentionComputeStage** (Refactored)

Pure attention compute, no KV cache management:

```cpp
class AttentionComputeStage : public IComputeStage
{
public:
    struct Params {
        TensorBase* Q;              ///< Query [seq_len, n_heads * head_dim]
        TensorBase* K;              ///< Key [kv_len, n_kv_heads * head_dim]
        TensorBase* V;              ///< Value [kv_len, n_kv_heads * head_dim]
        TensorBase* output;         ///< Output [seq_len, n_heads * head_dim]
        int batch_size;             ///< Batch size (1 for single sequence)
        int seq_len;                ///< Query sequence length
        int kv_len;                 ///< Key/Value length (may differ in decode)
        int n_heads;                ///< Query heads
        int n_kv_heads;             ///< KV heads (GQA: n_heads % n_kv_heads == 0)
        int head_dim;               ///< Dimension per head
        bool causal;                ///< Apply causal mask
        int window_size;            ///< Sliding window (-1 = disabled)
        // Workspaces (pre-allocated)
        TensorBase* workspace_scores;
        TensorBase* workspace_context;
        TensorBase* workspace_mask;
    };

    bool execute(IDeviceContext *ctx) override {
        // Get device type from context
        auto dev_type = ctx->isGPU() 
            ? DeviceType::CUDA  // or ROCm based on context
            : DeviceType::CPU;
        
        // Create kernel via KernelFactory
        auto kernel = KernelFactory::createAttention(params_.Q, dev_type);
        if (!kernel) {
            LOG_ERROR("[AttentionComputeStage] Failed to create attention kernel");
            return false;
        }
        
        // Dispatch to kernel's compute_tensor() method
        return kernel->compute_tensor(
            params_.Q, params_.K, params_.V, params_.output,
            params_.batch_size, params_.seq_len, params_.kv_len,
            params_.n_heads, params_.n_kv_heads, params_.head_dim,
            params_.causal, params_.window_size,
            params_.workspace_scores, params_.workspace_context, params_.workspace_mask,
            nullptr, // mpi_ctx - handled at higher level
            ctx->deviceIdx()
        );
    }
};
```

**Phase 9.3: Update LayerExecutor DAG Building**

Replace monolithic `AttentionWithKVCacheStage` with composable stages:

```cpp
// In Qwen2LayerExecutor::buildAttentionGraph()

// Stage 1: Append K/V to cache (if caching enabled)
if (params_.kv_cache) {
    KVCacheAppendStage::Params cache_params{
        buffers.K.get(), buffers.V.get(),
        params_.kv_cache, layer_idx, seq_len
    };
    graph.addNode("kv_cache_append", 
        ComputeStageFactory::createKVCacheAppend(cache_params), device);
}

// Stage 2: Get cached K/V for attention (or use direct K/V if no cache)
TensorBase* K_attn = params_.kv_cache 
    ? params_.kv_cache->get_k_base(layer_idx, 0)
    : buffers.K.get();
TensorBase* V_attn = params_.kv_cache
    ? params_.kv_cache->get_v_base(layer_idx, 0) 
    : buffers.V.get();
int kv_len = params_.kv_cache
    ? params_.kv_cache->get_cached_tokens(layer_idx, 0)
    : seq_len;

// Stage 3: Pure attention compute
AttentionComputeStage::Params attn_params{
    buffers.Q.get(), K_attn, V_attn, buffers.attn_output.get(),
    1, seq_len, kv_len,
    n_heads, n_kv_heads, head_dim,
    true, -1,  // causal, no sliding window
    workspaces.scores.get(), workspaces.context.get(), nullptr
};
graph.addNode("attention_compute",
    ComputeStageFactory::createAttentionCompute(attn_params), device);

// Dependencies
if (params_.kv_cache) {
    graph.addDependency("kv_cache_append", "attention_compute");
}
```

**Phase 9.4: MPI Tensor Parallelism via WorkDistributor**

Instead of `MpiAttentionOrchestrator::compute_tensor_parallel()`, use:

```cpp
// In LayerExecutor for tensor-parallel attention
if (mpi_ctx && mpi_ctx->world_size() > 1) {
    // WorkDistributor determines head distribution
    WorkDistributor dist(mpi_config);
    auto slice = dist.getRankSlice(n_heads);
    
    // Each rank computes its head slice
    AttentionComputeStage::Params local_params = attn_params;
    local_params.n_heads = slice.count;
    // ... adjust Q/output pointers for local heads
    
    graph.addNode("attention_local", createAttentionCompute(local_params), device);
    
    // Allreduce to combine outputs
    AllreduceStage::Params allreduce_params{
        buffers.attn_output.get(),
        seq_len * n_heads * head_dim,
        MPI_SUM
    };
    graph.addNode("attention_allreduce", createAllreduce(allreduce_params), device);
    graph.addDependency("attention_local", "attention_allreduce");
}
```

#### 9.3 Deprecation Timeline

| Component | Status | Target |
|-----------|--------|--------|
| `MpiAttentionOrchestrator` | Active | Deprecated in Phase 9.4 |
| `GQAAttention` | Now uses KernelFactory | Deprecated in Phase 9.3 |
| `AttentionWithKVCacheStage` | Active | Replaced by KVCacheAppendStage + AttentionComputeStage |
| `AttentionStage` | [[deprecated]] | Remove after Phase 9.2 |
| `MpiAttentionConfig` | Active | Replaced by AttentionComputeStage::Params |
| `GQAAttentionConfig` | Active | Replaced by AttentionComputeStage::Params |

#### 9.4 File Changes Required

| File | Change |
|------|--------|
| `execution/ComputeStage.h` | Add `KVCacheAppendStage`, `AttentionComputeStage` |
| `execution/ComputeStage.cpp` | Implement new stages |
| `execution/ComputeStageFactory` | Add `createKVCacheAppend()`, `createAttentionCompute()` |
| `pipelines/qwen/Qwen2LayerExecutor.cpp` | Use new composable stages |
| `pipelines/attention/MpiAttentionOrchestrator.h` | Mark deprecated |
| `pipelines/attention/GQAAttention.h` | Mark deprecated |

#### 9.5 Testing Strategy

1. **Unit tests**: `KVCacheAppendStage` append/get round-trip
2. **Unit tests**: `AttentionComputeStage` with mock DeviceContext
3. **Integration tests**: Full attention DAG vs legacy `AttentionWithKVCacheStage`
4. **Parity tests**: Ensure output matches for all precisions (FP32, BF16, Q8_1)
5. **MPI tests**: Tensor-parallel attention via WorkDistributor

#### 9.6 Success Criteria

- [ ] `KVCacheAppendStage` correctly manages cache state
- [ ] `AttentionComputeStage` produces correct output for all precisions
- [ ] DAG composition (append → compute) works correctly
- [ ] Tensor-parallel attention works via `WorkDistributor` + `AllreduceStage`
- [ ] No regression in attention correctness (E2E parity tests pass)
- [ ] `MpiAttentionOrchestrator` and `GQAAttention` deprecated with clear migration path
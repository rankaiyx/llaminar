# Llaminar V2 Architecture

This document is a high-level but concrete map of the Llaminar V2 stack: tensors, kernels, devices, inference execution, MPI orchestration, and attention. It is intended as a **quick-start reference for future agents** so they can safely modify V2 without re-deriving the architecture from scratch.

---

## 1. Design Goals

V2 is a **kernel-centric, operator-free** architecture:

- **Per-tensor device affinity** – each tensor knows which device it lives on.
- **Unified inference interface** – `IInferenceRunner` abstracts execution strategy (Pipeline or Graph).
- **Heterogeneous execution** – CPU / CUDA / ROCm / (future) backends can be mixed in one run.
- **Quantization-aware kernels** – unified GEMM/attention interfaces that work with FP32/BF16 and quantized formats.
- **MPI-aware orchestration** – multi-rank inference (tensor parallelism) lives in orchestrators, not kernels.
- **Centralized kernel dispatch** – `KernelFactory` provides unified kernel creation with caching.
- **Weight sharding** – automatic tensor parallelism distributes weight matrices across MPI ranks.
- **Declarative compute graphs** – `Qwen2Graph` builds DAGs with automatic dependency tracking and MPI synchronization.

The mental model:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE LAYER                                     │
│                                                                              │
│   createInferenceRunner()  ──→  IInferenceRunner                            │
│                                      │                                       │
│              ┌───────────────────────┴───────────────────────┐              │
│              ▼                                               ▼              │
│    ┌──────────────────┐                        ┌─────────────────────────┐  │
│    │   PipelineBase   │                        │   GraphOrchestrator    │  │
│    │  (Qwen2Pipeline) │                        │  (Graph-based exec)    │  │
│    └──────────────────┘                        └─────────────────────────┘  │
│              │                                               │              │
│              │ direct kernel calls                           │ DAG stages   │
│              ▼                                               ▼              │
│    ┌──────────────────┐                        ┌─────────────────────────┐  │
│    │    Composite     │                        │   ComputeGraph DAG      │  │
│    │   Operations     │                        │  + GraphExecutor        │  │
│    └──────────────────┘                        └─────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KERNEL LAYER                                       │
│                                                                              │
│   KernelFactory  ──→  ITensorGemm, ITensorAttention, IRMSNorm, etc.         │
│                                                                              │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │  CPU Kernels    │  │  CUDA Kernels   │  │  Quantized Kernels         │ │
│   │  (OpenBLAS/MKL) │  │  (future)       │  │  (IQ4_NL, Q8_0, Q6_K...)   │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TENSOR LAYER                                       │
│                                                                              │
│   TensorBase  ──→  FP32Tensor, BF16Tensor, IQ4_NLTensor, Q8_0Tensor, etc.   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Unified Inference Interface

### 2.1 IInferenceRunner

Location: `src/v2/inference/IInferenceRunner.h`

The `IInferenceRunner` interface provides a unified API for inference execution, implemented by both `PipelineBase` and `GraphOrchestrator`:

```cpp
class IInferenceRunner {
public:
    virtual ~IInferenceRunner() = default;
    
    // Core inference
    virtual bool forward(const int* tokens, int seq_len) = 0;
    virtual const float* logits() const = 0;
    
    // State management
    virtual void clear_cache() = 0;
    virtual int get_position() const = 0;
    
    // Metadata
    virtual int vocab_size() const = 0;
    virtual ExecutionPath executionPath() const = 0;
    virtual const char* architecture() const = 0;
};

enum class ExecutionPath {
    PIPELINE,  // Traditional imperative execution
    GRAPH      // Graph-based declarative execution
};
```

### 2.2 Factory Function

Location: `src/v2/inference/InferenceRunner.h`

```cpp
// Factory creates appropriate runner based on config/environment
std::unique_ptr<IInferenceRunner> createInferenceRunner(
    std::shared_ptr<ModelContext> model_ctx,
    std::shared_ptr<MPIContext> mpi_ctx,
    int device_idx,
    const InferenceRunnerConfig& config = {});
```

**Selection Logic:**
1. `config.force_pipeline` → Pipeline path
2. `config.force_graph` → Graph path  
3. `LLAMINAR_EXEC_FULL_FORWARD=1` → Graph path
4. Otherwise → Pipeline path (default)

### 2.3 Usage Pattern

```cpp
// Create runner (automatically selects execution path)
auto runner = createInferenceRunner(model_ctx, mpi_ctx, device_idx, config);

// Check which path was selected
if (runner->executionPath() == ExecutionPath::GRAPH) {
    LOG_INFO("Using graph-based execution");
}

// Inference (works regardless of path)
runner->forward(tokens.data(), seq_len);
const float* logits = runner->logits();

// Sampling
int next_token = argmax(logits, runner->vocab_size());

// Reset for new sequence
runner->clear_cache();
```

---

## 3. GraphOrchestrator System

### 3.1 Architecture Overview

The **GraphOrchestrator** is the graph-based execution system for transformer inference. It orchestrates declarative compute graphs for high-performance execution with:

- **Declarative graph construction** – Build computation as a DAG of stages
- **Automatic dependency tracking** – Stages execute in correct order
- **Graph caching** – Pre-built graphs reused across decode steps
- **MPI-aware execution** – `AllreduceStage` handles tensor-parallel synchronization
- **State management** – Owns KV cache, position tracking, and activation buffers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GraphOrchestrator                                     │
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐  │
│   │  InferenceState │   │  GraphExecutor  │   │   LayerGraphCache       │  │
│   │  - hidden       │   │  - execute()    │   │   - attention_decode    │  │
│   │  - logits       │   │  - topo sort    │   │   - ffn_decode          │  │
│   │  - kv_cache     │   │                 │   │   - per-layer caching   │  │
│   │  - positions    │   │                 │   │                         │  │
│   └─────────────────┘   └─────────────────┘   └─────────────────────────┘  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Qwen2Graph (IGraphBuilder)                                         │   │
│   │  - buildAttentionGraph() → ComputeGraph                             │   │
│   │  - buildFFNGraph() → ComputeGraph                                   │   │
│   │  - Declarative stage definition                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 ComputeGraph and Stages

Location: `src/v2/execution/ComputeGraph.h`, `src/v2/execution/ComputeStage.h`

A **ComputeGraph** is a DAG of `IComputeStage` nodes:

```cpp
class ComputeGraph {
public:
    void addNode(const std::string& name, std::unique_ptr<IComputeStage> stage, int device_idx);
    void addDependency(const std::string& dependent, const std::string& dependency);
    std::vector<std::string> getExecutionOrder() const;  // Topological sort
    void reset();  // Reset all stages for reuse
};
```

**Available Stage Types:**

| Stage Class | Purpose | MPI Sync? |
|-------------|---------|-----------|
| `RMSNormStage` | RMS normalization | No |
| `GEMMStage` | Matrix multiplication | No |
| `SwiGLUStage` | SwiGLU activation | No |
| `RoPEStage` | Rotary position embeddings | No |
| `ResidualAddStage` | Residual connections | No |
| `AttentionComputeStage` | Full attention computation | No |
| `KVCacheAppendStage` | KV cache management | No |
| `AllreduceStage` | **MPI synchronization** | **Yes** |

### 3.3 Execution Flow

```
forward(tokens, seq_len)
        │
        ▼
┌───────────────────────────────────────┐
│  1. Embedding Lookup                  │
│     token_ids → hidden [seq, d_model] │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  2. For each layer (0..N-1):          │
│                                       │
│     ┌─────────────────────────────┐   │
│     │  executeAttention()         │   │
│     │  - Build/cache graph        │   │
│     │  - RMSNorm → Q/K/V → RoPE   │   │
│     │  - KV Cache → Attention     │   │
│     │  - Wo projection            │   │
│     │  - AllreduceStage (if MPI)  │   │
│     │  - Residual add             │   │
│     └─────────────────────────────┘   │
│              │                        │
│              ▼                        │
│     ┌─────────────────────────────┐   │
│     │  executeFFN()               │   │
│     │  - Build/cache graph        │   │
│     │  - RMSNorm → Gate/Up        │   │
│     │  - SwiGLU → Down            │   │
│     │  - AllreduceStage (if MPI)  │   │
│     │  - Residual add             │   │
│     └─────────────────────────────┘   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  3. Final Norm + LM Head              │
│     hidden → logits [seq, vocab]      │
└───────────────────────────────────────┘
        │
        ▼
    return logits
```

### 3.4 MPI Tensor Parallelism

The GraphOrchestrator implements **Megatron-style tensor parallelism** where weight matrices are split across MPI ranks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RANK 0                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GraphOrchestrator (owns InferenceState, GraphExecutor)              │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ComputeGraph (FFN layer)                                     │   │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐ ┌───────┐ │   │    │
│  │  │  │ RMSNorm│→│Gate/Up │→│ SwiGLU │→│ Down GEMM    │→│Allreduce│   │    │
│  │  │  └────────┘ └────────┘ └────────┘ └──────────────┘ └───────┘ │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Weights (SHARD 0):                                                          │
│  • ffn_gate: [d_ff/2, d_model]  ← first half of rows                        │
│  • ffn_up:   [d_ff/2, d_model]  ← first half of rows                        │
│  • ffn_down: [d_model, d_ff/2]  ← first half of columns                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ MPI_Allreduce (SUM)
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RANK 1                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GraphOrchestrator ← SEPARATE INSTANCE, IDENTICAL GRAPH STRUCTURE    │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ComputeGraph (structurally identical)                        │   │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐ ┌───────┐ │   │    │
│  │  │  │ RMSNorm│→│Gate/Up │→│ SwiGLU │→│ Down GEMM    │→│Allreduce│   │    │
│  │  │  └────────┘ └────────┘ └────────┘ └──────────────┘ └───────┘ │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Weights (SHARD 1):                                                          │
│  • ffn_gate: [d_ff/2, d_model]  ← second half of rows                       │
│  • ffn_up:   [d_ff/2, d_model]  ← second half of rows                       │
│  • ffn_down: [d_model, d_ff/2]  ← second half of columns                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Each rank has its own `GraphOrchestrator` instance
- Graphs are structurally identical across ranks
- Weight tensors point to different shards
- `AllreduceStage` synchronizes partial results after Wo and Down projections

### 3.5 Graph Caching (Decode Optimization)

For decode mode (seq_len=1), graphs are cached and reused:

```cpp
bool GraphOrchestrator::executeAttention(...) {
    if (graph_caching_enabled_ && seq_len == 1) {
        auto& cache = layer_graph_cache_[layer_idx];
        
        if (cache.attention_decode && cache.valid) {
            // Update only dynamic parameters (position offset)
            updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);
            
            // Execute cached graph
            bool success = executor_.execute(*cache.attention_decode, ctx);
            cache.attention_decode->reset();
            return success;
        }
        
        // Build graph and cache for future reuse
        cache.attention_decode = std::make_unique<ComputeGraph>(
            graph_builder_->buildAttentionGraph(...));
        cache.valid = true;
    }
    // Execute (cached or freshly built)
}
```

**Benefits:**
- Eliminates graph construction overhead during decode
- Position offset updated via `updateDynamicParams()` without rebuilding
- ~10-20% decode speedup for long sequences

### 3.6 InferenceState

The `GraphOrchestrator` owns all mutable inference state:

```cpp
struct InferenceState {
    // Core buffers
    std::shared_ptr<TensorBase> hidden;    // [batch * seq, d_model]
    std::shared_ptr<TensorBase> logits;    // [batch * seq, vocab_size]
    
    // KV Cache
    std::unique_ptr<IUnifiedKVCache> kv_cache;
    
    // Position tracking
    std::vector<int> positions;           // Per-sequence position offset
    std::vector<int> sequence_lengths;    // For variable-length batches
    
    // Activation buffers
    std::shared_ptr<TensorBase> normalized, residual;
    std::shared_ptr<TensorBase> Q, K, V;
    std::shared_ptr<TensorBase> attn_output, attn_proj;
    std::shared_ptr<TensorBase> gate, up, ffn_output;
    
    // Attention workspace
    std::shared_ptr<TensorBase> workspace_scores, workspace_context, workspace_mask;
};
```

---

## 4. Tensors and Tensor Interfaces

### 4.1 Core Tensor Types

Location: `src/v2/tensors/`

- `TensorBase` – abstract base: shape, dtype, device, virtual hooks to create kernels
- Concrete tensors:
  - `FP32Tensor`, `BF16Tensor`, `FP16Tensor` – dense float tensors
  - Quantized: `IQ4_NLTensor`, `Q4_0Tensor`, `Q6_KTensor`, `Q8_0Tensor`, etc.
  - View/alias tensors for cheap slicing

Each tensor exposes factory methods for kernels:
```cpp
std::unique_ptr<ITensorGemm> createGemm() const;
std::unique_ptr<ITensorAttention> createAttention() const;
```

### 4.2 Tensor Kernel Interfaces

Location: `src/v2/tensors/TensorKernels.h`

- `ITensorGemm` – GEMM between activations and/or weights
- `ITensorAttention` – Attention over Q/K/V
- Other per-op interfaces (RMSNorm, SwiGLU, RoPE)

---

## 5. Kernels and KernelFactory

### 5.1 CPU Kernels

Location: `src/v2/kernels/cpu/`

- GEMM: oneDNN / OpenBLAS / AVX-512 VNNI JIT
- Attention: `CpuAttentionKernelT<T>` for FP32/BF16
- Vectorized primitives in `primitives/`

### 5.2 Quantized GEMM and Block Decoders

Quantized tensors implement `IBlockDecoder`:
```cpp
class IBlockDecoder {
    virtual void decode_block_at(size_t row, size_t k_block, float* out) const = 0;
    virtual size_t block_size() const = 0;
};
```

A generic `QuantizedGemmKernel` uses `IBlockDecoder` strategy pattern:
- One implementation works for all quantized formats
- `decode_block_at` is `always_inline` for zero virtual call overhead

### 5.3 KernelFactory

Location: `src/v2/kernels/KernelFactory.h`

Centralized kernel dispatch with caching:

```cpp
class KernelFactory {
    static DeviceType getDeviceType(int device_idx);
    static ITensorGemm* getOrCreateGemm(const TensorBase* tensor);  // Cached
    static void clearCacheFor(const TensorBase* tensor);            // Auto cleanup
    static void clearCache();                                        // Manual cleanup
};
```

**Usage:**
```cpp
// Preferred: cached kernel (pack once, use many)
ITensorGemm* gemm = KernelFactory::getOrCreateGemm(weight_tensor.get());
gemm->multiply(activations, output, m, n, k);
```

---

## 6. MPI Layer

### 6.1 MPIContext

Location: `src/v2/utils/MPIContext.h`

```cpp
class MPIContext {
    int rank() const;
    int world_size() const;
    void allreduce_sum(float* buffer, size_t count);
    void broadcast(void* data, size_t count, int root);
    void barrier();
};
```

### 6.2 Weight Sharding

Location: `src/v2/loaders/WeightManager.h`

Automatic tensor parallelism:

| Weight Pattern | Sharding Mode | Rationale |
|----------------|---------------|-----------|
| `attn_output.weight` (Wo) | ROW_PARALLEL | Split input dim, allreduce output |
| `ffn_down.weight` | ROW_PARALLEL | Split input dim, allreduce output |
| QKV, Gate/Up, norms | REPLICATE | Full tensors needed for attention |

**Memory Savings (2 ranks):** ~50% reduction for sharded weights

---

## 7. Attention

### 7.1 ITensorAttention

Location: `src/v2/tensors/TensorKernels.h`

```cpp
class ITensorAttention {
    virtual bool compute(
        const float* Q, const float* K, const float* V, float* output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, /* workspaces */, MPIContext* mpi_ctx) = 0;
};
```

### 7.2 GQAAttention

Location: `src/v2/pipelines/attention/GQAAttention.*`

Front-end for GQA/MQA/MHA semantics:
- Validation (shapes, head counts)
- Mask construction (causal, padding)
- Kernel dispatch via `ITensorAttention`

---

## 8. Snapshot System

### 8.1 Overview

The snapshot system captures intermediate tensor states for debugging and parity testing:

- **Compile-time conditional** – Only in Debug/E2ERelease builds
- **In-memory storage** – `std::map<std::string, std::vector<float>>`
- **Zero overhead in Release** – Compiles away to NOOPs

### 8.2 Usage

```cpp
// Enable capture
runner->enableSnapshotCapture("/tmp/snapshots");

// Run inference
runner->forward(tokens, seq_len);

// Retrieve snapshots
auto keys = runner->getSnapshotKeys();
for (const auto& key : keys) {
    size_t size;
    const float* data = runner->getSnapshot(key, size);
    // Compare with reference...
}
```

### 8.3 Snapshot Keys

- **Global:** `EMBEDDING`, `FINAL_NORM`, `LM_HEAD`
- **Per-layer:** `layer{i}_{STAGE}` where stage is:
  - Attention: `ATTENTION_NORM`, `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION`, `Q_ROPE`, `K_ROPE`, `ATTENTION_CONTEXT`, `ATTENTION_OUTPUT`, `ATTENTION_RESIDUAL`
  - FFN: `FFN_NORM`, `FFN_GATE`, `FFN_UP`, `FFN_SWIGLU`, `FFN_DOWN`, `FFN_RESIDUAL`

---

## 9. Quick Reference

### 9.1 File Locations

| Component | Location |
|-----------|----------|
| Inference interface | `src/v2/inference/IInferenceRunner.h` |
| Inference factory | `src/v2/inference/InferenceRunner.{h,cpp}` |
| GraphOrchestrator | `src/v2/pipelines/qwen/GraphOrchestrator.{h,cpp}` |
| Graph builder | `src/v2/pipelines/qwen/Qwen2Graph.{h,cpp}` |
| ComputeGraph/Stages | `src/v2/execution/` |
| KernelFactory | `src/v2/kernels/KernelFactory.{h,cpp}` |
| Tensors | `src/v2/tensors/` |
| CPU kernels | `src/v2/kernels/cpu/` |
| MPI utilities | `src/v2/utils/MPIContext.h` |
| Weight sharding | `src/v2/loaders/WeightManager.{h,cpp}` |

### 9.2 Key Design Rules

1. **Use `IInferenceRunner`** for inference – Don't call `GraphOrchestrator` directly from Main/ChatUI
2. **Keep MPI out of kernels** – MPI lives in `AllreduceStage` and orchestrators
3. **Use KernelFactory** – Prefer `getOrCreateGemm()` for cached kernel access
4. **Graph Executor for new code** – Prefer declarative graphs over imperative kernel calls

### 9.3 Environment Variables

| Variable | Effect |
|----------|--------|
| `LLAMINAR_EXEC_FULL_FORWARD=1` | Force graph execution path |
| `LLAMINAR_USE_LAYER_EXECUTOR=1` | Enable LayerExecutor |
| `LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=1` | Enable automatic buffer aliasing |

### 9.4 Running Tests

```bash
# Unit tests
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests
ctest --test-dir build_v2_release -R "^V2_Integration_" --output-on-failure

# E2E parity tests (requires E2ERelease build)
ctest --test-dir build_v2_e2e_release -R "^V2_E2E_" --output-on-failure
```

---

This architecture is intentionally modular: small, focused abstractions connected by narrow interfaces. The `IInferenceRunner` interface ensures that callers (Main.cpp, ChatUI, tests) remain decoupled from the specific execution strategy, whether using traditional pipelines or the graph-based system.

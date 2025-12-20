# Distributed Architecture Proposal: Tensor-Parallel GraphOrchestrator

**Author**: David Sanftenberg  
**Date**: December 2025  
**Status**: Implementation In Progress

## Executive Summary

This document proposes a redesigned distributed inference architecture for Llaminar V2 that:
1. Eliminates redundant computation across MPI ranks
2. Properly separates physical topology (nodes/sockets) from logical topology (MPI ranks)
3. Uses explicit graph stages for tensor slicing and gathering
4. Removes the legacy `MpiAttentionOrchestrator` in favor of declarative graph operations
5. **ALL ranks (including rank 0) participate in compute by default**
6. **Future support for heterogeneous work distribution by device capability**

## Key Design Decisions

### Decision 1: All Ranks Compute (Including Rank 0)
Unlike traditional coordinator/worker patterns, **rank 0 participates in computation** by default:
- Eliminates idle coordinator waste (especially important for 2-rank systems)
- Uses equal work division initially
- `is_compute_participant()` defaults to `true` for all ranks
- Only explicit opt-out changes this behavior

### Decision 2: Use Existing TensorSlice Class
**Do NOT create a duplicate `TensorSlice` struct.** We already have:
- `src/v2/tensors/TensorSlice.h` - Full tensor slice wrapper class
- `SliceMetadata` struct with `ROW_PARALLEL`, `COLUMN_PARALLEL`, `FULL` modes
- Factory methods: `SliceMetadata::forRowParallel()`, `SliceMetadata::forColumnParallel()`

For simple work ranges (heads, vocab indices), use `WorkRange` struct which is lighter weight.
For tensor parallelism metadata, use `SliceMetadata` from the existing TensorSlice.h.

### Decision 3: Device Capability Exchange at Startup
At construction, `MPITopology`:
1. Detects local devices (CPU, CUDA, ROCm)
2. AllGathers device capabilities from all ranks
3. Stores in `all_placements_` vector
4. Enables future weighted work distribution

## Current State Problems

### Problem 1: Redundant Computation
Currently, most operations are **replicated** across all ranks:
- Embedding lookup: All ranks compute full embeddings
- QKV projections: All ranks compute full Q/K/V
- LM Head: All ranks compute full logits (151K vocab × batch × 4B = massive waste)

Only Wo and Down projections use row-parallel sharding with AllreduceStage.

### Problem 2: Hidden MPI Logic
The `MpiAttentionOrchestrator` (~1600 lines) hides tensor-parallel attention logic inside a monolithic class, making it:
- Hard to debug (batched attention bug we were investigating)
- Hard to compose with other stages
- Inconsistent with the declarative graph approach

### Problem 3: Missing Physical Topology
`NUMATopology` handles intra-node CPU/GPU affinity but there's no abstraction for:
- Multi-node clusters
- Rank-to-node mapping
- Optimized communication patterns (shared memory intra-node vs network inter-node)

## Proposed Architecture

### 1. MPITopology Class

The `MPITopology` class (implemented in `src/v2/utils/MPITopology.h`) provides:

```cpp
// src/v2/utils/MPITopology.h

namespace llaminar2 {

/**
 * @brief Device capability for heterogeneous work distribution
 */
struct DeviceCapability {
    enum class Type { CPU, CUDA, ROCm };
    Type type = Type::CPU;
    int device_id = 0;
    float relative_compute = 1.0f;  ///< Relative compute power (CPU=1.0, GPU~=10.0)
    size_t memory_bytes = 0;
    std::string name;
};

/**
 * @brief Information about a single MPI rank's placement
 */
struct RankPlacement {
    int rank;                    ///< MPI rank (0..world_size-1)
    int node_id;                 ///< Physical node/machine (0..node_count-1)
    int local_rank;              ///< Rank within node (0..ranks_per_node-1)
    int socket_id;               ///< CPU socket within node
    int numa_node;               ///< NUMA node for memory affinity
    std::string hostname;        ///< Machine hostname
    std::vector<DeviceCapability> devices;  ///< All devices available to this rank
};

/**
 * @brief Simple work range (lighter weight than TensorSlice)
 * 
 * For tensor parallelism metadata, use SliceMetadata from tensors/TensorSlice.h
 */
struct WorkRange {
    size_t start;  ///< Start index (inclusive)
    size_t end;    ///< End index (exclusive)
    size_t count;  ///< Number of elements
    
    bool empty() const { return count == 0; }
    
    /// Create equal work range for rank
    static WorkRange for_rank_equal(size_t total, int rank, int world_size);
    
    /// Create weighted work range (future: by device capability)
    static WorkRange for_rank_weighted(size_t total, int rank, int world_size,
                                        const std::vector<float>& weights);
};

/**
 * @brief MPI topology abstraction for distributed inference
 * 
 * Key design principles:
 * 1. ALL ranks (including rank 0) participate in compute by default
 * 2. Equal work division initially; future support for weighted distribution
 * 3. Integrates with existing SliceMetadata from tensors/TensorSlice.h
 */
class MPITopology {
public:
    explicit MPITopology(MPI_Comm comm = MPI_COMM_WORLD);
    
    // Basic queries
    int rank() const;
    int world_size() const;
    bool is_coordinator() const { return rank_ == 0; }
    bool is_compute_participant() const { return compute_participant_; }  // Default: true
    
    // Work distribution (returns WorkRange for simple indices)
    WorkRange get_head_range(int total_heads) const;
    WorkRange get_kv_head_range(int total_kv_heads) const;
    WorkRange get_column_range(size_t total_cols) const;
    WorkRange get_row_range(size_t total_rows) const;
    WorkRange get_vocab_range(size_t vocab_size) const;
    
    // SliceMetadata creation (integrates with TensorSlice.h)
    SliceMetadata createRowParallelMeta(size_t rows, size_t cols, bool presliced = false) const;
    SliceMetadata createColumnParallelMeta(size_t rows, size_t cols, bool presliced = false) const;
    
    // Device capabilities (gathered from all ranks at startup)
    const std::vector<RankPlacement>& all_placements() const;
    std::vector<float> get_compute_weights() const;
    
    // Communicators
    MPI_Comm intra_node_comm() const;
    MPI_Comm inter_node_comm() const;
    MPI_Comm world_comm() const;

private:
    bool compute_participant_ = true;  // ALL ranks compute by default
    std::vector<RankPlacement> all_placements_;  // From AllGather
};

} // namespace llaminar2
```

### 2. Distributed Graph Stages

New stages that make tensor parallelism explicit in the compute graph:

```cpp
// src/v2/execution/DistributedStages.h

namespace llaminar2 {
};

} // namespace llaminar2
```

### 2. Distributed Graph Stages

New stages that make tensor parallelism explicit in the compute graph:

```cpp
// src/v2/execution/DistributedStages.h

namespace llaminar2 {

// =============================================================================
// SliceStage: Extract local portion of a tensor
// =============================================================================

/**
 * @brief Extract a slice of input tensor for local computation
 * 
 * Used at the START of tensor-parallel regions to get this rank's portion.
 */
class SliceStage : public IComputeStage {
public:
    struct Params {
        TensorBase* input;           ///< Full tensor (may be nullptr on non-root ranks)
        TensorBase* output;          ///< Local slice output
        WorkRange range;             ///< Which portion to extract
        int dim;                     ///< Dimension to slice (0=rows, 1=cols)
        const MPITopology* topology; ///< For rank-aware slicing
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::SLICE; }
};

// =============================================================================
// GatherStage: Collect slices from all ranks to one rank
// =============================================================================

/**
 * @brief Gather tensor slices from all ranks to coordinator (rank 0)
 * 
 * Used at the END of tensor-parallel regions to collect results.
 * NOTE: Rank 0 also contributes its computed portion (all ranks compute).
 */
class GatherStage : public IComputeStage {
public:
    struct Params {
        TensorBase* local_input;     ///< This rank's local slice
        TensorBase* gathered_output; ///< Full tensor (only valid on root)
        int root;                    ///< Rank to gather to (usually 0)
        int dim;                     ///< Dimension that was sliced
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::GATHER; }
};

// =============================================================================
// AllGatherStage: Collect slices to ALL ranks
// =============================================================================

/**
 * @brief AllGather tensor slices so all ranks have full tensor
 * 
 * Used when subsequent computation needs full tensor (e.g., attention needs full K/V).
 */
class AllGatherStage : public IComputeStage {
public:
    struct Params {
        TensorBase* local_input;     ///< This rank's local slice  
        TensorBase* full_output;     ///< Full tensor on ALL ranks
        int dim;                     ///< Dimension that was sliced
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::ALLGATHER; }
};

// =============================================================================
// ScatterStage: Distribute tensor from root to all ranks
// =============================================================================

/**
 * @brief Scatter tensor from coordinator to all ranks
 * 
 * Used to distribute input tokens at start of forward pass.
 */
class ScatterStage : public IComputeStage {
public:
    struct Params {
        TensorBase* full_input;      ///< Full tensor (only valid on root)
        TensorBase* local_output;    ///< This rank's slice
        int root;                    ///< Rank that has full input (usually 0)
        int dim;                     ///< Dimension to scatter along
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::SCATTER; }
};

// =============================================================================
// ReduceScatterStage: Reduce + Scatter in one operation
// =============================================================================

/**
 * @brief ReduceScatter for efficient row-parallel GEMM
 * 
 * Combines partial results and distributes slices in one collective.
 * More efficient than Allreduce when subsequent op only needs a slice.
 */
class ReduceScatterStage : public IComputeStage {
public:
    struct Params {
        TensorBase* input;           ///< Full partial result on each rank
        TensorBase* output;          ///< Reduced slice for this rank
        MPI_Op op;                   ///< Reduction operation (usually MPI_SUM)
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::REDUCE_SCATTER; }
};

} // namespace llaminar2
```

### 3. Tensor-Parallel Graph Structure

Here's how a layer would look with explicit tensor parallelism:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TENSOR-PARALLEL TRANSFORMER LAYER                       │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    ATTENTION (Column-Parallel Q/K/V)                       │ │
│  │                                                                            │ │
│  │  hidden ──► RMSNorm ──► normalized                                         │ │
│  │                              │                                             │ │
│  │              ┌───────────────┼───────────────┐                             │ │
│  │              ▼               ▼               ▼                             │ │
│  │         ┌─────────┐    ┌─────────┐    ┌─────────┐                          │ │
│  │         │ Wq GEMM │    │ Wk GEMM │    │ Wv GEMM │  ◄── Column-parallel     │ │
│  │         │ (local) │    │ (local) │    │ (local) │      Each rank has       │ │
│  │         └────┬────┘    └────┬────┘    └────┬────┘      Wq[:, slice]        │ │
│  │              │              │              │                               │ │
│  │              ▼              ▼              ▼                               │ │
│  │         Q_local        K_local        V_local                              │ │
│  │         [m, heads/r]   [m, kv_h/r]    [m, kv_h/r]                          │ │
│  │              │              │              │                               │ │
│  │              ▼              ▼              ▼                               │ │
│  │         ┌─────────┐    ┌─────────┐    ┌─────────┐                          │ │
│  │         │  RoPE   │    │  RoPE   │    │   ---   │  ◄── Local RoPE          │ │
│  │         └────┬────┘    └────┬────┘    └────┬────┘                          │ │
│  │              │              │              │                               │ │
│  │              │         ┌────┴────┐         │                               │ │
│  │              │         │KV Cache │         │  ◄── Local KV cache           │ │
│  │              │         │ Append  │         │      (sharded by heads)       │ │
│  │              │         └────┬────┘         │                               │ │
│  │              │              │              │                               │ │
│  │              └──────────────┼──────────────┘                               │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │ Local Attention │  ◄── Each rank computes             │ │
│  │                    │ (heads/world_sz)│      attention for its heads        │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                      attn_local                                            │ │
│  │                      [m, heads/r × head_dim]                               │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │    Wo GEMM      │  ◄── Row-parallel: Wo[slice, :]     │ │
│  │                    │    (local)      │      Each rank has partial Wo       │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │   AllReduce     │  ◄── Sum partial results            │ │
│  │                    │    (SUM)        │      All ranks get full output      │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                       attn_output                                          │ │
│  │                       [m, d_model]  ◄── Full on all ranks                  │ │
│  │                             │                                              │ │
│  └─────────────────────────────┼──────────────────────────────────────────────┘ │
│                                │                                                │
│                                ▼                                                │
│                         ResidualAdd                                             │
│                                │                                                │
│  ┌─────────────────────────────┼──────────────────────────────────────────────┐ │
│  │                         FFN (Column-Parallel Gate/Up)                      │ │
│  │                                                                            │ │
│  │  hidden ──► RMSNorm ──► normalized                                         │ │
│  │                              │                                             │ │
│  │              ┌───────────────┴───────────────┐                             │ │
│  │              ▼                               ▼                             │ │
│  │         ┌─────────┐                    ┌─────────┐                         │ │
│  │         │Gate GEMM│                    │ Up GEMM │  ◄── Column-parallel    │ │
│  │         │ (local) │                    │ (local) │      Gate[:, slice]     │ │
│  │         └────┬────┘                    └────┬────┘      Up[:, slice]       │ │
│  │              │                              │                              │ │
│  │              └──────────────┬───────────────┘                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │     SwiGLU      │  ◄── Local SwiGLU on slice          │ │
│  │                    │    (local)      │                                     │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │   Down GEMM     │  ◄── Row-parallel: Down[slice, :]   │ │
│  │                    │    (local)      │                                     │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │   AllReduce     │  ◄── Sum partial results            │ │
│  │                    │    (SUM)        │                                     │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                       ffn_output                                           │ │
│  │                       [m, d_model]                                         │ │
│  │                             │                                              │ │
│  └─────────────────────────────┼──────────────────────────────────────────────┘ │
│                                │                                                │
│                                ▼                                                │
│                         ResidualAdd                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. Weight Sharding Strategy (Full)

| Weight | Parallelism | Rank i Has | Collective After |
|--------|-------------|------------|------------------|
| `embed_table` | **Replicated** | Full [V, D] | None (fast lookup) |
| `attn_norm.gamma` | **Replicated** | Full [D] | None |
| `wq` | **Column-parallel** | [D, heads_i × head_dim] | None (concat implicit) |
| `wk` | **Column-parallel** | [D, kv_heads_i × head_dim] | None |
| `wv` | **Column-parallel** | [D, kv_heads_i × head_dim] | None |
| `wo` | **Row-parallel** | [heads_i × head_dim, D] | **AllReduce(SUM)** |
| `ffn_norm.gamma` | **Replicated** | Full [D] | None |
| `gate_proj` | **Column-parallel** | [D, ffn_dim/r] | None |
| `up_proj` | **Column-parallel** | [D, ffn_dim/r] | None |
| `down_proj` | **Row-parallel** | [ffn_dim/r, D] | **AllReduce(SUM)** |
| `final_norm.gamma` | **Replicated** | Full [D] | None |
| `lm_head` | **Column-parallel** | [D, vocab/r] | **AllGather** (rank 0 for sampling) |

### 5. LM Head: Special Handling

The LM head is expensive (vocab_size × d_model ≈ 151K × 896 for Qwen2.5-0.5B). Options:

**Option A: Column-Parallel + AllGather (Full Logits)**
```
Rank 0: lm_head[:, 0:vocab/2]  → logits[:, 0:vocab/2]
Rank 1: lm_head[:, vocab/2:]   → logits[:, vocab/2:]
                                     │
                              AllGather to rank 0
                                     │
                              Full logits on rank 0 for sampling
```

**Option B: Column-Parallel + Reduce to get argmax (Efficient for Greedy)**
```
Each rank: local_argmax = argmax(local_logits)
           local_max = max(local_logits)
                                     │
                              AllReduce(MAX) + voting
                                     │
                              Global argmax without full logits transfer
```

Option B is more efficient for greedy sampling but requires custom reduction.

### 6. KV Cache Sharding

With head-parallel attention, each rank only needs to cache its assigned heads:

```cpp
// Before (replicated):
KVCache: [n_layers, batch_size, max_seq, n_kv_heads, head_dim]
         ≈ 24 × 1 × 2048 × 2 × 64 × 4B = 25MB per rank (duplicated!)

// After (sharded):
KVCache: [n_layers, batch_size, max_seq, n_kv_heads/world_size, head_dim]
         ≈ 24 × 1 × 2048 × 1 × 64 × 4B = 12.5MB per rank (halved)
```

**Challenge**: For GQA, n_kv_heads (2) may be less than world_size (2). In this case, one rank handles all KV heads for half the layers, the other for the other half (pipeline parallelism hybrid).

### 7. Coordinator vs Worker Model

Rather than separate graph classes, use a **role-aware execution model**:

```cpp
enum class RankRole {
    Coordinator,  // Rank 0: handles I/O, sampling, may have extra stages
    Worker        // Ranks 1..n: pure compute
};

class DistributedGraphOrchestrator {
public:
    // All ranks execute the same forward graph
    const float* forward(const int* tokens, int seq_len, int batch_size);
    
private:
    RankRole role_;
    std::shared_ptr<MPITopology> topology_;
    
    // Role-specific behavior encapsulated in stages, not separate graphs
    // - InputStage: Only coordinator reads tokens, then Bcast
    // - OutputStage: Only coordinator samples from gathered logits
};
```

**Graph Structure (same for all ranks):**
```
[InputStage]        ← Coordinator: read tokens; Workers: receive Bcast
    │
[EmbeddingStage]    ← All ranks: lookup (replicated embed table)
    │
[Layer 0..n-1]      ← All ranks: compute local portions
    │
[FinalNormStage]    ← All ranks: local norm (replicated gamma)
    │
[LMHeadStage]       ← All ranks: local GEMM (column-parallel lm_head)
    │
[GatherStage]       ← All ranks → Coordinator: gather logits
    │
[SampleStage]       ← Coordinator: sample next token; Workers: receive Bcast
```

## Migration Path

### Phase 1: MPITopology (Foundation)
1. Create `MPITopology` class with topology detection
2. Update `MPIContext` to use `MPITopology` internally
3. Add work distribution methods (`get_head_slice`, `get_column_slice`, etc.)

### Phase 2: Distributed Stages
1. Implement `SliceStage`, `GatherStage`, `AllGatherStage`, `ScatterStage`
2. Add `ComputeStageType` enum values
3. Create `ComputeStageFactory::create*` methods

### Phase 3: Column-Parallel QKV
1. Update `WeightManager` to shard Wq/Wk/Wv by columns
2. Modify `Qwen2Graph::buildAttentionGraph` to use local heads
3. Remove head extraction from `MpiAttentionOrchestrator`

### Phase 4: Column-Parallel FFN
1. Update `WeightManager` to shard Gate/Up by columns
2. Modify `Qwen2Graph::buildFFNGraph` for local FFN dim

### Phase 5: Column-Parallel LM Head
1. Shard `lm_head` by vocab dimension
2. Add `AllGatherStage` before sampling
3. Implement efficient argmax reduction (optional)

### Phase 6: Sharded KV Cache
1. Modify `UnifiedKVCache` to only store local heads
2. Update cache append/gather stages

### Phase 7: Cleanup
1. Deprecate `MpiAttentionOrchestrator`
2. Remove legacy `use_decomposed_attention` flag
3. Update tests

## Memory Savings Analysis

For Qwen2.5-0.5B with 2 ranks:

| Component | Before (per rank) | After (per rank) | Savings |
|-----------|-------------------|------------------|---------|
| Wq (per layer) | 896 × 896 × 4B = 3.2MB | 896 × 448 × 4B = 1.6MB | 50% |
| Wk (per layer) | 896 × 128 × 4B = 458KB | 896 × 64 × 4B = 229KB | 50% |
| Wv (per layer) | 896 × 128 × 4B = 458KB | 896 × 64 × 4B = 229KB | 50% |
| Gate (per layer) | 896 × 4864 × 4B = 17.4MB | 896 × 2432 × 4B = 8.7MB | 50% |
| Up (per layer) | 896 × 4864 × 4B = 17.4MB | 896 × 2432 × 4B = 8.7MB | 50% |
| LM Head | 896 × 151936 × 4B = 544MB | 896 × 75968 × 4B = 272MB | 50% |
| KV Cache | 24 × 2048 × 128 × 4B = 25MB | 24 × 2048 × 64 × 4B = 12.5MB | 50% |
| **Total Model** | ~2.8GB | ~1.5GB | **46%** |

For larger models (7B+), savings are even more significant as model size dominates.

## Open Questions

1. **GQA with few KV heads**: When `n_kv_heads < world_size`, how to distribute?
   - Option: Hybrid tensor + pipeline parallelism
   - Option: Replicate KV heads, only shard Q heads

2. **Batch parallelism**: Should we also support data parallelism (different sequences on different ranks)?

3. **Sequence parallelism**: For long sequences, split sequence dimension across ranks?

4. **Communication optimization**: Use NCCL for GPU-GPU, MPI for CPU-CPU?

## Conclusion

This proposal outlines a clean tensor-parallel architecture that:
- Makes MPI communication explicit in the compute graph
- Properly separates topology concerns
- Eliminates redundant computation
- Scales memory usage with number of ranks

The migration can be done incrementally, with each phase delivering measurable benefits.

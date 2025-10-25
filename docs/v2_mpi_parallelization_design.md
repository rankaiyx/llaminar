# V2 MPI Parallelization Strategy Design

**Date**: January 2025  
**Status**: Design Proposal  
**Author**: David Sanftenberg

## Executive Summary

This document designs MPI work-splitting strategies for V2 pipelines, enabling distributed model parallelism across multiple ranks. The design adapts proven patterns from V1's extensive MPI infrastructure while maintaining V2's operator-free architecture.

## Background

### Current State (V2)

**Strengths**:
- ✅ MPIContext abstraction with core primitives (allreduce_sum, broadcast, allgather, barrier)
- ✅ get_local_slice() for work distribution
- ✅ mpi_ctx_ available in PipelineBase
- ✅ Performance: 402.89 ± 8.65 GFLOPS with 2 MPI ranks

**Gaps**:
- ❌ No MPI parallelization - each rank computes FULL workload independently
- ❌ No strategy framework for configuring parallelization
- ❌ No weight slicing (weights replicated on all ranks)
- ❌ attention_gqa runs single-rank only (no cross-rank coordination)

### V1 Reference Architecture

V1 has extensive MPI infrastructure we can learn from:

**V1 Features**:
- `DistributionType` enum: `ROW_WISE`, `COL_WISE`, `BLOCK_WISE`, `HEAD_WISE`, `VOCAB_WISE`, `SEQUENCE_WISE`
- `WeightSliceType` enum: `REPLICATED`, `ROW_SLICED`, `COL_SLICED`
- Weight slicing contracts with automatic dimension calculation
- Collective operations: `allReduceTensors()`, `gatherTensors()`, `allGatherTensors()`
- Distribution helpers: `getRowDistribution()`, `getColDistribution()`, `getBlockDistribution()`
- MPI-aware weight loading with automatic slicing

**Key V1 Pattern - Tensor Parallel Attention**:
```cpp
// From V1 changelog: Weight slicing contracts
// Q/K/V: ROW_SLICED by n_head → [local_heads, d_model]
// O:     COL_SLICED by n_head → [d_model, local_heads]
// Each rank computes subset of heads, allreduce outputs
```

## Design Goals

1. **Benefit all pipelines**: Common infrastructure in PipelineBase
2. **Variety of strategies**: User-configurable parallelization approach
3. **Sane defaults**: Auto-select based on model/hardware
4. **Backward compatible**: world_size=1 works unchanged (zero overhead)
5. **Performance**: Linear scaling with number of ranks (target: 2× speedup on 2 ranks)

## Proposed Architecture

### 1. Strategy Enum (Inspired by V1 DistributionType)

```cpp
// src/v2/pipelines/MPIStrategy.h
namespace llaminar2 {

/**
 * @brief MPI parallelization strategies for distributed inference
 */
enum class MPIStrategy {
    /**
     * @brief No parallelization (single rank or disabled)
     * 
     * Each rank operates independently. Use when:
     * - world_size == 1
     * - User explicitly disabled MPI parallelization
     */
    None,

    /**
     * @brief Tensor parallel - split attention heads/features across ranks
     * 
     * Attention: Each rank computes subset of heads, allreduce outputs
     * FFN: Each rank computes subset of intermediate features, allreduce outputs
     * 
     * Communication: 2× allreduce per layer (after attention, after FFN)
     * Best for: Large models, high-bandwidth interconnects
     * 
     * Requirements:
     * - n_heads % world_size == 0 (attention heads evenly divisible)
     * - d_ff % world_size == 0 (FFN intermediate dimension evenly divisible)
     */
    TensorParallel,

    /**
     * @brief Pipeline parallel - split layers across ranks
     * 
     * Rank 0: Layers 0 to (n_layers/world_size - 1)
     * Rank 1: Layers (n_layers/world_size) to (2*n_layers/world_size - 1)
     * ...
     * 
     * Communication: Point-to-point activation passing between ranks
     * Best for: Very deep models, lower bandwidth interconnects
     * 
     * Requirements:
     * - n_layers % world_size == 0 (layers evenly divisible)
     */
    PipelineParallel,

    /**
     * @brief Sequence parallel - split sequence dimension across ranks
     * 
     * Each rank processes subset of tokens. Used for prefill phase.
     * 
     * Communication: Allgather for attention computation, reduce for output
     * Best for: Long sequences (>1024 tokens), prefill optimization
     * 
     * Requirements:
     * - seq_len % world_size == 0 (tokens evenly divisible)
     */
    SequenceParallel,

    /**
     * @brief Hybrid - combination of strategies
     * 
     * Example: Tensor-parallel within node + pipeline-parallel across nodes
     * 
     * Requirements: Advanced configuration (not implemented in initial phase)
     */
    Hybrid
};

/**
 * @brief MPI configuration for pipeline execution
 */
struct MPIConfig {
    MPIStrategy strategy = MPIStrategy::TensorParallel;  // Default strategy
    bool auto_select = true;  // Auto-choose strategy based on model/hardware
    bool validate_divisibility = true;  // Check dimension divisibility

    // Tensor-parallel options
    bool tp_split_attention = true;  // Split attention heads
    bool tp_split_ffn = true;        // Split FFN intermediate dimension

    // Fallback strategy if primary fails validation
    MPIStrategy fallback_strategy = MPIStrategy::None;
};

} // namespace llaminar2
```

### 2. PipelineBase Extensions

```cpp
// src/v2/pipelines/PipelineBase.h (additions)

class PipelineBase {
protected:
    // Existing members
    std::shared_ptr<MPIContext> mpi_ctx_;
    int device_idx_;
    
    // NEW: MPI parallelization state
    MPIConfig mpi_config_;
    MPIStrategy mpi_strategy_ = MPIStrategy::None;
    
    // NEW: MPI-aware attention (dispatches based on strategy)
    bool attention_gqa_mpi(TensorBase *Q, TensorBase *K, TensorBase *V, 
                          TensorBase *output, int n_heads, int n_kv_heads, 
                          int head_dim, bool causal = true, int window_size = -1);
    
    // NEW: Tensor-parallel attention implementation
    bool attention_gqa_tensor_parallel(TensorBase *Q, TensorBase *K, TensorBase *V, 
                                      TensorBase *output, int n_heads, int n_kv_heads, 
                                      int head_dim, bool causal, int window_size);
    
    // NEW: Strategy selection and validation
    MPIStrategy selectOptimalStrategy();
    bool validateStrategy(MPIStrategy strategy);
    
    // NEW: MPI distribution helpers (inspired by V1)
    std::pair<size_t, size_t> getHeadDistribution(int n_heads);  // For tensor-parallel
    std::pair<size_t, size_t> getLayerDistribution(int n_layers); // For pipeline-parallel
    std::pair<size_t, size_t> getTokenDistribution(int seq_len);  // For sequence-parallel
};
```

### 3. Tensor-Parallel Attention (Primary Implementation)

**Algorithm**:
1. **Distribute attention heads** across ranks (each rank computes subset)
2. **Compute Q/K/V projections** for local heads
3. **Compute attention scores** for local heads
4. **Compute attention output** for local heads
5. **Allreduce outputs** to sum across all ranks

**Implementation**:
```cpp
// src/v2/pipelines/PipelineBase.cpp

bool PipelineBase::attention_gqa_tensor_parallel(
    TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
    int n_heads, int n_kv_heads, int head_dim, bool causal, int window_size)
{
    if (!mpi_ctx_) {
        LOG_ERROR("Tensor-parallel attention requires MPI context");
        return false;
    }

    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();

    // 1. Distribute attention heads across ranks
    auto [start_head, local_n_heads] = getHeadDistribution(n_heads);
    
    LOG_DEBUG("Rank " << rank << "/" << world_size 
              << ": Computing heads " << start_head 
              << " to " << (start_head + local_n_heads - 1)
              << " (local_n_heads=" << local_n_heads << ")");

    // 2. GQA: Broadcast K/V heads if fewer than Q heads
    int n_groups = n_heads / n_kv_heads;
    size_t seq_len = Q->shape()[0];
    
    // 3. Compute attention for local heads
    // Allocate local output buffer [seq_len, local_n_heads * head_dim]
    auto local_output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, local_n_heads * head_dim}, 
        device_idx_
    );
    
    // Compute local attention (each rank processes its head subset)
    for (size_t local_h = 0; local_h < local_n_heads; ++local_h) {
        size_t global_h = start_head + local_h;
        
        // Get Q/K/V slices for this head
        float *Q_h = Q->mutable_data() + global_h * head_dim * seq_len;
        
        // GQA: Map to K/V head
        size_t kv_head = global_h / n_groups;
        float *K_h = K->mutable_data() + kv_head * head_dim * seq_len;
        float *V_h = V->mutable_data() + kv_head * head_dim * seq_len;
        
        // Compute scores: Q @ K^T [seq_len, seq_len]
        // ... (standard attention computation)
        
        // Apply causal mask + softmax
        // ... (standard attention computation)
        
        // Compute context: scores @ V [seq_len, head_dim]
        // ... (standard attention computation)
        
        // Store in local output buffer
        float *output_h = local_output->mutable_data() + local_h * head_dim * seq_len;
        // ... copy context to output_h
    }
    
    // 4. Allreduce: Sum local outputs across all ranks
    // CRITICAL: Each rank contributes its local heads, sum gives full output
    size_t output_size = seq_len * n_heads * head_dim;
    
    // Zero-initialize output on all ranks (will accumulate local contributions)
    std::fill(output->mutable_data(), 
              output->mutable_data() + output_size, 
              0.0f);
    
    // Allreduce: Each rank contributes its local heads at correct offset
    // This accumulates contributions from all ranks
    mpi_ctx_->allreduce_sum(
        local_output->data(),  // Local contribution
        output->mutable_data() + start_head * head_dim * seq_len,  // Global position
        local_n_heads * head_dim * seq_len  // Size of this rank's contribution
    );
    
    // 5. Barrier to ensure all ranks complete
    mpi_ctx_->barrier();
    
    return true;
}

std::pair<size_t, size_t> PipelineBase::getHeadDistribution(int n_heads) {
    if (!mpi_ctx_) return {0, n_heads};
    return mpi_ctx_->get_local_slice(n_heads);
}
```

### 4. Strategy Selection Logic

```cpp
// src/v2/pipelines/PipelineBase.cpp

MPIStrategy PipelineBase::selectOptimalStrategy() {
    if (!mpi_ctx_ || mpi_ctx_->world_size() == 1) {
        return MPIStrategy::None;
    }

    int world_size = mpi_ctx_->world_size();

    // Try tensor-parallel first (most common, best performance)
    if (validateStrategy(MPIStrategy::TensorParallel)) {
        return MPIStrategy::TensorParallel;
    }

    // Fallback to pipeline-parallel
    if (validateStrategy(MPIStrategy::PipelineParallel)) {
        return MPIStrategy::PipelineParallel;
    }

    // No valid strategy - disable MPI parallelization
    LOG_WARNING("No valid MPI strategy found, using single-rank execution");
    return MPIStrategy::None;
}

bool PipelineBase::validateStrategy(MPIStrategy strategy) {
    if (!mpi_ctx_) return false;
    int world_size = mpi_ctx_->world_size();

    switch (strategy) {
        case MPIStrategy::TensorParallel:
            // Requires n_heads divisible by world_size
            // Note: n_heads_ set by derived class (Qwen2Pipeline, etc.)
            if (n_heads_ % world_size != 0) {
                LOG_WARNING("Tensor-parallel requires n_heads (" << n_heads_ 
                           << ") divisible by world_size (" << world_size << ")");
                return false;
            }
            return true;

        case MPIStrategy::PipelineParallel:
            // Requires n_layers divisible by world_size
            if (n_layers_ % world_size != 0) {
                LOG_WARNING("Pipeline-parallel requires n_layers (" << n_layers_ 
                           << ") divisible by world_size (" << world_size << ")");
                return false;
            }
            return true;

        case MPIStrategy::SequenceParallel:
            // No validation needed (can split any sequence length)
            return true;

        default:
            return false;
    }
}
```

### 5. Qwen2Pipeline Integration

```cpp
// src/v2/pipelines/Qwen2Pipeline.cpp

Qwen2Pipeline::Qwen2Pipeline(
    std::shared_ptr<ModelContext> model_ctx,
    std::shared_ptr<MPIContext> mpi_ctx,
    int device_idx,
    std::shared_ptr<WeightPlacementMap> placement_map,
    const MPIConfig& mpi_config)  // NEW parameter
    : PipelineBase(model_ctx, mpi_ctx, device_idx, placement_map)
{
    // ... existing architecture setup (n_layers_, n_heads_, etc.) ...
    
    // Configure MPI strategy (NEW)
    mpi_config_ = mpi_config;
    
    if (mpi_ctx_ && mpi_ctx_->world_size() > 1) {
        if (mpi_config_.auto_select) {
            mpi_strategy_ = selectOptimalStrategy();
        } else {
            mpi_strategy_ = mpi_config_.strategy;
            if (!validateStrategy(mpi_strategy_)) {
                LOG_WARNING("User-specified strategy invalid, using fallback");
                mpi_strategy_ = mpi_config_.fallback_strategy;
            }
        }
        
        LOG_INFO("MPI Strategy: " << strategyName(mpi_strategy_) 
                 << " (rank " << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size() << ")");
    }
}

// Update forward() to use MPI-aware attention
bool Qwen2Pipeline::forward(const int *tokens, int seq_len) {
    // ... existing code ...
    
    // Replace existing attention call with MPI-aware version
    // OLD: attention_gqa(Q.get(), K.get(), V.get(), attn_output.get(), ...)
    
    // NEW: Dispatch based on MPI strategy
    if (mpi_strategy_ == MPIStrategy::None) {
        // Fast path: Single-rank attention (existing code)
        attention_gqa(Q.get(), K.get(), V.get(), attn_output.get(), 
                     n_heads_, n_kv_heads_, head_dim_, true);
    } else {
        // MPI-aware attention (tensor-parallel or other strategy)
        attention_gqa_mpi(Q.get(), K.get(), V.get(), attn_output.get(), 
                         n_heads_, n_kv_heads_, head_dim_, true);
    }
    
    // ... rest of forward pass ...
}
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `MPIStrategy.h` with strategy enum and config
- [ ] Add `mpi_config_`, `mpi_strategy_` to PipelineBase
- [ ] Implement strategy selection logic
- [ ] Implement validation logic
- [ ] Add `getHeadDistribution()` helper

**Deliverables**:
- Strategy framework compiles and links
- Tests for strategy validation

### Phase 2: Tensor-Parallel Attention (Week 1-2)
- [ ] Implement `attention_gqa_tensor_parallel()`
- [ ] Add `attention_gqa_mpi()` dispatcher
- [ ] Update Qwen2Pipeline to use MPI-aware attention
- [ ] Add MPI barrier synchronization

**Deliverables**:
- Attention runs distributed across 2 ranks
- Correctness test: Single-rank vs multi-rank outputs match

### Phase 3: Testing and Validation (Week 2)
- [ ] Create `test_mpi_tensor_parallel.cpp`
- [ ] Test with 1, 2, 4 ranks
- [ ] Validate numerical correctness (compare to single-rank)
- [ ] Benchmark performance (measure speedup)

**Expected Results**:
- 2 ranks: 1.8-2.0× speedup (near-linear scaling)
- 4 ranks: 3.5-3.8× speedup (some communication overhead)

### Phase 4: Weight Slicing (Week 3-4)
- [ ] Adapt V1's weight slicing contracts to V2
- [ ] Implement automatic weight slicing in ModelLoader
- [ ] Add `ROW_SLICED`, `COL_SLICED` metadata
- [ ] Update Qwen2Pipeline to load sliced weights

**Deliverables**:
- Weights distributed across ranks (not replicated)
- Memory usage per rank reduced proportionally

### Phase 5: Pipeline-Parallel (Future)
- [ ] Implement layer distribution
- [ ] Add point-to-point activation passing
- [ ] Test on very deep models

### Phase 6: Sequence-Parallel (Future)
- [ ] Implement token distribution
- [ ] Add allgather for attention
- [ ] Optimize for prefill phase

## Success Metrics

### Performance Targets
- **2 ranks**: 1.8-2.0× speedup (vs single-rank baseline)
- **4 ranks**: 3.5-3.8× speedup
- **Communication overhead**: <10% of compute time

### Correctness
- ✅ Numerical agreement with single-rank (max abs diff < 1e-4)
- ✅ All CTest parity tests pass with MPI
- ✅ Multi-trial variance <5% (stable performance)

### Memory Efficiency
- ✅ Weight memory per rank = full_model_size / world_size (with slicing)
- ✅ Activation memory unchanged (distributed computation, not storage)

## Migration Path from V1

V1 has extensive MPI infrastructure we should adapt:

### V1 Components to Adapt
1. **Weight slicing contracts** → V2 ModelLoader
2. **DistributionType enum** → MPIStrategy enum (done)
3. **MPI collective helpers** → MPIContext extensions
4. **Weight validation with MPI** → V2 weight loading tests

### V1 Components NOT Needed
- `MPIOperatorBase` (V2 has no operators)
- `CommunicationPattern` enum (too low-level, MPIContext handles)
- Graph-based execution (V2 is operator-free)

## Risks and Mitigations

### Risk 1: Communication Overhead
**Concern**: Allreduce every layer may dominate small models

**Mitigation**:
- Measure communication time vs compute time
- Only enable MPI for large models (>7B parameters)
- Add environment variable to disable: `LLAMINAR_V2_DISABLE_MPI=1`

### Risk 2: Numerical Divergence
**Concern**: Floating-point errors from different reduction order

**Mitigation**:
- Use deterministic reduction (ensure same order on all ranks)
- Add strict numerical testing (max abs diff threshold)
- Validate against single-rank ground truth

### Risk 3: Complexity Creep
**Concern**: V2 design principle is simplicity, MPI adds complexity

**Mitigation**:
- Keep MPI logic isolated in PipelineBase (not in kernels)
- Maintain single-rank fast path (zero overhead when world_size=1)
- Comprehensive documentation and testing

## Open Questions

1. **How to handle GQA with head distribution?**
   - Answer: Distribute Q heads, broadcast K/V heads (already in design)

2. **Should FFN be tensor-parallel too?**
   - Answer: Yes, split intermediate dimension (d_ff) for consistency

3. **How to benchmark communication overhead?**
   - Answer: Add MPI_Barrier before/after collectives, measure wall time

4. **Should we support mixed strategies (TP for attention, replicated for FFN)?**
   - Answer: Future work, start with consistent strategy across all ops

## References

### V1 Codebase
- `src/v1/MpiKernelBase.h`: Distribution strategies, communication patterns
- `src/v1/WeightContracts.h`: Weight slicing contracts, ROW_SLICED/COL_SLICED
- `src/v1/PipelineBase.h`: Collective operation helpers
- `changelog/2025-01-27_weight_slicing_contracts.md`: V1 weight slicing design

### Literature
- Megatron-LM: Tensor, pipeline, and sequence parallelism patterns
- DeepSpeed: Hybrid parallelism strategies
- FSDP (PyTorch): Fully Sharded Data Parallel

---

**Next Steps**: Review this design with team, then implement Phase 1 (Core Infrastructure).

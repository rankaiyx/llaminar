# Project Plan: Full Graph Framework Inference

**Date**: December 19, 2025  
**Author**: David Sanftenberg  
**Status**: Phase 4 Complete ✅  
**Branch**: `feature/typed-residuals`

## Executive Summary

This project extends the declarative graph framework to handle **complete inference** (embedding → layers → lm_head), enabling the graph path to fully replace the imperative pipeline code. Currently, the graph framework only handles per-layer execution while the pipeline orchestrates embedding, the layer loop, final norm, and LM head.

### Current State

```
Qwen2Pipeline::forward_batch():
├── Batch padding                    [PIPELINE - imperative]
├── Allocation                       [PIPELINE - imperative]
├── Position tracking                [PIPELINE - imperative]
├── EMBEDDING                        [PIPELINE - direct stage call]
├── LOOP × n_layers:
│   └── transformer_layer()          [ORCHESTRATOR - graph execution] ✅
├── FINAL NORM                       [PIPELINE - imperative]
└── LM HEAD                          [PIPELINE - direct stage call]
```

### Target State

```
GraphOrchestrator::executeForward():
├── Build full forward graph         [GRAPH BUILDER - declarative]
├── Execute entire graph             [GRAPH EXECUTOR - single call]
└── Return logits                    [Done]

Qwen2Pipeline becomes thin wrapper:
├── Setup (batch padding, allocation)
└── orchestrator_->executeForward(input, output)
```

---

## Blockers Identified

| # | Blocker | Severity | Est. Effort | Status |
|---|---------|----------|-------------|--------|
| 1 | Embedding/LM head not wired in orchestrator | High | 2-3 hours | ✅ Fixed |
| 2 | Global weights not set in Qwen2Graph | Medium | 1 hour | ✅ Fixed |
| 3 | `executeForward()` not called by pipeline | High | 2-3 hours | ✅ Fixed |
| 4 | Position ID building in pipeline, not graph | Low | 30 min | N/A (handled) |
| 5 | State ownership split between pipeline/orchestrator | Medium | 2-4 hours | Pending |
| 6 | No integration tests for full graph path | High | 2 hours | ✅ Fixed |

---

## Implementation Phases

### Phase 1: Wire Global Weights in Qwen2Graph ✅ COMPLETE

**Goal**: Allow `Qwen2Graph` to build full forward graphs with actual weight pointers.

**Problem**: `buildFullForwardGraph()` exists but `weights_.embedding_table`, `weights_.final_norm`, `weights_.lm_head` are never set when using orchestrator path.

**Solution**:
1. ✅ Add `setWeights()` / `setBuffers()` methods to `GraphOrchestrator` (delegate to Qwen2Graph)
2. ✅ Add `configureOrchestratorWeights()` method to `Qwen2Pipeline` 
3. ✅ Call from `Qwen2Pipeline` after orchestrator initialization
4. ✅ Create layer weight accessor lambda that maps `LayerWeights` → `Qwen2LayerWeights`

**Files Modified**:
- `src/v2/pipelines/qwen/GraphOrchestrator.h` - Added setWeights(), setBuffers(), hasGlobalWeights()
- `src/v2/pipelines/qwen/GraphOrchestrator.cpp` - Implemented methods
- `src/v2/pipelines/qwen/Qwen2Pipeline.h` - Added configureOrchestratorWeights()
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Implemented method, called from init
- `tests/v2/unit/Test__GraphOrchestrator.cpp` - Added 4 new tests

**Tests Added**:
- `SetWeightsDelegatesToGraphBuilder` ✅
- `SetBuffersDelegatesToGraphBuilder` ✅
- `HasGlobalWeightsReturnsFalseWhenNotSet` ✅
- `HasGlobalWeightsReturnsTrueWhenSet` ✅

**Results**: 173 V2 unit tests pass, 25 GraphOrchestrator tests pass

---

### Phase 2: Implement `executeForward()` in GraphOrchestrator ✅ COMPLETE

**Goal**: Single entry point for complete forward pass via graph execution.

**Status**: Merged with Phase 3 - the `executeForward()` implementation already existed and was functional.

---

### Phase 3: Add Full Forward Path to Pipeline ✅ COMPLETE

**Goal**: Pipeline can optionally use `executeForward()` instead of layer loop.

**Solution**:
1. ✅ Added `LLAMINAR_EXEC_FULL_FORWARD=1` environment flag (via `ExecutionConfig.exec_full_forward`)
2. ✅ When enabled, `forward_batch()` calls `orchestrator_->executeForward()`
3. ✅ Legacy path remains as fallback when flag not set

**Actual Implementation** (in `Qwen2Pipeline::forward_batch()`):
```cpp
const auto &exec_env = debugEnv().execution;
if (exec_env.exec_full_forward && orchestrator_) {
    // Allocate logits buffer
    // Configure model buffers via orchestrator_->setBuffers()
    // Build position IDs for all sequences in batch
    // Build Qwen2ForwardInput with tokens, positions, batch info, kv_cache
    // Build Qwen2ForwardOutput with logits, hidden pointers
    // Execute: orchestrator_->executeForward(input, output)
    // Update position counters
    return true;
}
// else: legacy imperative path
```

**Files Modified**:
- `src/v2/utils/DebugEnv.h` - Added `exec_full_forward` flag to `ExecutionConfig` struct (~line 450)
- `src/v2/utils/DebugEnv.h` - Added parsing for `LLAMINAR_EXEC_FULL_FORWARD` env var (~line 507)
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Added ~70 lines for full forward path (~line 548)

**Tests**: 173 unit tests pass

**Usage**:
```bash
LLAMINAR_USE_LAYER_EXECUTOR=1 LLAMINAR_EXEC_FULL_FORWARD=1 ./run_llaminar.sh -m model.gguf -p "prompt"
```

---

### Phase 4: Integration Testing & Parity Verification ✅ COMPLETE

**Goal**: Verify full graph path produces identical results to legacy path.

**Tests Created**:
1. **`PrefillParity`** - 9-token prompt prefill comparison
2. **`IncrementalDecodeParity`** - 5-step incremental decode comparison
3. **`FullGraphLogitsShape`** - Logits shape and sanity validation

**Results**:
- ✅ Cosine similarity: 1.000000 (perfect)
- ✅ Top-5 overlap: 100% (5/5 tokens match)
- ✅ Max abs diff: 0.0000e+00
- ✅ Mean abs diff: 0.0000e+00

**Bugs Fixed During Testing**:
1. **Token ID Type Mismatch** (segfault): `padded.tokens->data()` returns `float*` but was cast to `int*` for embedding kernel
   - Fix: Build `flat_token_ids` vector from `token_batches` (actual int arrays) with proper padding
2. **Missing Layer Buffers**: `Qwen2ModelBuffers::layer_buffers` was not populated from `activation_buffers_`
   - Fix: Populate all layer buffer pointers (normalized, Q, K, V, attn_output, gate, up, ffn_output, etc.)
3. **Missing Workspace Buffers**: Attention requires `workspace_mask`, `workspace_scores`, `workspace_context`
   - Fix: Add workspace buffers from `attention_workspace_*_` members

**Files Created**:
- `tests/v2/integration/Test__FullGraphForward_vs_Legacy_Parity.cpp` (~600 lines)

**Files Modified**:
- `tests/v2/CMakeLists.txt` - Added test target with labels
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Fixed buffer population for full forward path

---

### Phase 5: State Ownership Migration (Optional)

**Goal**: Move state ownership from pipeline to orchestrator for cleaner architecture.

**Current**:
```cpp
// Pipeline owns everything
std::shared_ptr<TensorBase> current_hidden_;
std::unique_ptr<IUnifiedKVCache> kv_cache_;
std::vector<int> current_positions_;
```

**Target**:
```cpp
// Orchestrator owns inference state
class GraphOrchestrator {
    std::shared_ptr<TensorBase> hidden_;
    std::unique_ptr<IUnifiedKVCache> kv_cache_;
    std::vector<int> positions_;
    
    // Pipeline just calls
    bool forward(const int* tokens, int seq_len, float* logits);
};
```

**Risk**: Large refactor, may break existing integrations.
**Recommendation**: Defer until Phase 4 proves parity.

---

### Phase 6: IModelExecutor Interface (Future)

**Goal**: Generic interface for model execution without pipeline dependency.

```cpp
class IModelExecutor {
public:
    virtual ~IModelExecutor() = default;
    
    // High-level API
    virtual bool forward(const int* tokens, int seq_len) = 0;
    virtual const float* logits() const = 0;
    virtual void clear_cache() = 0;
    
    // Configuration
    virtual int vocab_size() const = 0;
    virtual int max_seq_len() const = 0;
};

class Qwen2ModelExecutor : public IModelExecutor {
    std::unique_ptr<GraphOrchestrator> orchestrator_;
    // All state owned here
};
```

**Benefit**: Clean API for inference without pipeline complexity.

---

## Test Plan

### Unit Tests
| Test | Description | Phase |
|------|-------------|-------|
| `Qwen2Graph_SetGlobalWeights` | Verify weight setters work | 1 |
| `Qwen2Graph_BuildFullForward_WithWeights` | Graph has all nodes | 1 |
| `GraphOrchestrator_ExecuteForward_MockBuilder` | Mock execution path | 2 |

### Integration Tests
| Test | Description | Phase |
|------|-------------|-------|
| `FullGraphForward_LogitsShape` | Output has correct shape | 3 |
| `FullGraphForward_ParityWithLegacy` | Same output as legacy path | 4 |
| `FullGraphForward_DecodeMode` | Single token decode works | 4 |

### E2E Tests
| Test | Description | Phase |
|------|-------------|-------|
| `FullGraphForward_TokenPrediction` | Correct next token | 4 |
| `FullGraphForward_MultiToken` | Multi-token generation | 4 |

---

## Success Criteria

| Criterion | Metric |
|-----------|--------|
| Full forward via graph | `executeForward()` handles complete inference |
| Numerical parity | Logits match legacy path within 1e-5 |
| No performance regression | < 5% slower than legacy path |
| Tests pass | All existing + new tests |
| Feature-flagged | Disabled by default, opt-in via env var |

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Global weights | 1-2 hours | None |
| Phase 2: executeForward() | 2-3 hours | Phase 1 |
| Phase 3: Pipeline integration | 2-3 hours | Phase 2 |
| Phase 4: Testing | 2-4 hours | Phase 3 |
| Phase 5: State migration | 4-8 hours | Phase 4 (optional) |
| Phase 6: IModelExecutor | 4-8 hours | Phase 5 (future) |

**Minimum Viable**: Phases 1-4 (~8-12 hours)
**Full Migration**: Phases 1-6 (~20-30 hours)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence | Medium | High | Comprehensive parity tests |
| Performance regression | Low | Medium | Benchmark before/after |
| Breaking existing code | Low | High | Feature flag, parallel paths |
| Graph caching issues | Medium | Medium | Test prefill + decode modes |

---

## Files Affected

### Phase 1
- `src/v2/pipelines/qwen/Qwen2Graph.h`
- `src/v2/pipelines/qwen/Qwen2Graph.cpp`
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`
- `tests/v2/unit/Test__Qwen2Graph.cpp` (new tests)

### Phase 2
- `src/v2/pipelines/qwen/GraphOrchestrator.h`
- `src/v2/pipelines/qwen/GraphOrchestrator.cpp`

### Phase 3
- `src/v2/utils/DebugEnv.h`
- `src/v2/utils/DebugEnv.cpp`
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

### Phase 4
- `tests/v2/integration/Test__FullGraphForwardParity.cpp` (new)

---

## Appendix: Current buildFullForwardGraph Implementation

```cpp
// Qwen2Graph.cpp line ~260
ComputeGraph Qwen2Graph::buildFullForwardGraph(const Qwen2ForwardInput& input, Qwen2ForwardOutput& output) {
    // Requires weights_.embedding_table, weights_.final_norm, weights_.lm_head
    // These are NOT set when using orchestrator path!
    
    if (!weights_.embedding_table || !weights_.final_norm || !weights_.lm_head) {
        LOG_ERROR("Global weights not set - cannot build full forward graph");
        return ComputeGraph{};  // Empty graph = error
    }
    
    // Build embedding → layers → final_norm → lm_head
    // ...
}
```

**Key Insight**: The code exists but weights aren't wired. Phase 1 fixes this.

# V2 MPI Phase 3: Testing Infrastructure Design

**Date**: 2025-10-25  
**Status**: 📋 **DESIGN PHASE** - Infrastructure not yet implemented  
**Dependencies**: Phase 2 (tensor-parallel attention implementation) ✅ Complete

---

## Executive Summary

**Phase 2 delivered** the core tensor-parallel attention algorithm (~450 lines) with strategy selection and MPI distribution. **Phase 3 goals** were numerical correctness validation and performance benchmarking. However, **V2's testing infrastructure is not yet mature enough** to support end-to-end pipeline testing.

This document outlines the **missing infrastructure** and provides a **roadmap** for implementing comprehensive Phase 3 testing when V2 matures.

---

## Current State (Phase 2 Complete)

### ✅ What Works

**Core Implementation** (PipelineBase.cpp, ~450 lines):
- `attention_gqa_mpi()` - Dispatcher for MPI-aware attention
- `attention_gqa_tensor_parallel()` - Distributed attention with head sharding
- `selectOptimalStrategy()` - Automatic strategy selection (tensor/pipeline/sequence)
- `validateStrategy()` - Divisibility checks and validation
- Distribution helpers - `getHeadDistribution()`, `getLayerDistribution()`, etc.

**Qwen2Pipeline Integration** (Qwen2Pipeline.cpp, +48 lines):
- MPI configuration in constructor
- Strategy selection logging
- Dispatcher integration (`attention_gqa_mpi`)

**Strategy Validation Tests** (Test__MPITensorParallel.cpp, 235 lines):
- ✅ 5 tests passing with 2 MPI ranks
- Strategy enum validation
- Head distribution correctness
- Qwen-specific configuration tests

### ❌ What's Missing (Blockers for Phase 3)

**High-Level API Limitations**:
1. **Pipeline Construction**: Qwen2Pipeline requires ModelContext (full GGUF model), can't be instantiated standalone
2. **Tensor API**: `attention_gqa()` expects `TensorBase*` not raw `float*` arrays
3. **Private Members**: `n_heads_`, `mpi_strategy_` are private (no setters)
4. **Method Visibility**: `attention_gqa_mpi()` is protected, not accessible to tests

**Testing Infrastructure Gaps**:
1. **No Mock Pipeline**: Can't create minimal pipeline for unit testing attention
2. **No Tensor Factory**: Can't easily create FP32Tensor for test inputs
3. **No Baseline Runner**: Can't run single-rank reference for comparison
4. **No Performance Harness**: No multi-trial timing framework

**V2 Maturity Status** (from copilot-instructions.md):
- V2 is **"in development"** and **"not feature-complete"**
- V2 is **"not production-ready"**
- V2 testing: **"not validated"**
- Use V2 for: **"experimenting with new architectures"**

---

## Phase 3 Goals (Deferred Until Infrastructure Ready)

### Goal 1: Numerical Correctness Validation

**Objective**: Prove distributed attention produces identical results to single-rank

**Test Cases**:
- Single-token attention (decode phase)
- Multi-token attention (prefill, 32/128/512 tokens)
- Variable sequence lengths (1, 4, 8, 16, 32, 64)
- Edge cases (single head per rank, uneven distribution)

**Success Criteria**:
- Max absolute difference < 1e-4 (FP32 tolerance)
- All elements match within tolerance
- Reproducible across multiple runs

**Metrics to Collect**:
- Max abs diff
- Mean abs diff
- RMSE (root mean square error)
- Relative L2 norm
- Number of mismatches > tolerance

### Goal 2: Performance Benchmarking

**Objective**: Measure speedup, efficiency, and scaling of distributed attention

**Test Cases**:
- Single-token (1 token): Latency-critical decode phase
- Small batch (32 tokens): Small prefill
- Medium batch (128 tokens): Typical prefill
- Large batch (512 tokens): Large prefill
- Scaling analysis (1, 2, 4 ranks)

**Metrics to Collect**:
- Time per iteration (ms)
- GFLOPS (2 × seq_len^2 × n_heads × head_dim)
- Speedup vs baseline (T_1rank / T_nrank)
- Efficiency (speedup / num_ranks)
- Communication overhead (estimate from efficiency loss)

**Multi-Trial Statistics**:
- Mean ± stddev (5 trials minimum)
- Coefficient of variation (CV% < 5% target)
- Min/max range
- Warmup trials (3) excluded from statistics

**Expected Performance** (based on V1 empirical data):
- 2 ranks: 1.8-1.9× speedup (10-15% communication overhead)
- 4 ranks: 3.2-3.6× speedup (15-20% overhead)
- 8 ranks: 5.5-6.5× speedup (20-25% overhead)

### Goal 3: Integration with Existing Framework

**Objective**: Integrate MPI tests into existing V2 performance suite

**Requirements**:
- Use `add_v2_perf_test()` CMake function
- Apply optimal MPI/OpenMP settings from `run_benchmark.sh`
- Label appropriately: `V2;Performance;MPI;TensorParallel`
- Pin to physical cores for consistent measurement
- Report to same CSV format as IQ4_NL GEMM benchmark

---

## Missing Infrastructure Checklist

### 1. Test-Friendly Pipeline API

**Current**: Qwen2Pipeline requires full ModelContext

**Needed**:
```cpp
// Option A: Standalone attention test API
class AttentionTestHarness {
public:
    AttentionTestHarness(int n_heads, int head_dim, MPIStrategy strategy);
    bool runAttention(const float* Q, const float* K, const float* V, 
                     float* output, int seq_len);
};

// Option B: Mock ModelContext
class MockModelContext : public ModelContext {
public:
    MockModelContext(int n_heads, int n_layers);  // Minimal stub
};

// Option C: Public test methods in PipelineBase
class PipelineBase {
public:
    // Expose for testing
    bool test_attention_gqa_mpi(...);  // Public wrapper
};
```

**Recommendation**: Option A (dedicated test harness) - cleanest separation

### 2. Tensor Factory for Tests

**Current**: Can't create TensorBase* from raw float arrays

**Needed**:
```cpp
// Utility for tests
namespace test_utils {
    std::shared_ptr<FP32Tensor> createTensor(const float* data, 
                                            std::vector<size_t> shape);
    void copyToRaw(const TensorBase* tensor, float* output);
}
```

**Alternative**: Accept raw pointers in test-specific attention methods

### 3. Baseline Comparison Framework

**Current**: No way to run single-rank reference

**Needed**:
```cpp
class BaselineRunner {
public:
    BaselineRunner(int n_heads, int head_dim);
    void runAttention(const float* Q, const float* K, const float* V,
                     float* output, int seq_len);
    // Guaranteed single-rank, no MPI
};
```

**Purpose**: Ground truth for correctness comparison

### 4. Performance Measurement Utilities

**Current**: No multi-trial timing framework

**Needed**:
```cpp
struct PerfResult {
    double mean_ms;
    double stddev_ms;
    double cv_percent;
    double gflops;
    double speedup;
    double efficiency;
};

class PerfHarness {
public:
    PerfResult benchmark(std::function<void()> fn, 
                        int trials = 5, int warmup = 3);
};
```

**Alternative**: Use existing `Perf__IQ4_NL_GEMM.cpp` patterns

### 5. MPI Barrier Instrumentation

**Current**: No timing breakdown for communication overhead

**Needed**:
```cpp
struct MPIProfile {
    double compute_time_ms;      // Local computation
    double allreduce_time_ms;    // MPI_Allreduce
    double barrier_time_ms;      // MPI_Barrier
    double total_time_ms;
    double comm_overhead_percent;
};
```

**Purpose**: Diagnose performance bottlenecks

---

## Proposed Implementation Plan

### Phase 3a: Minimal Testing Infrastructure (1-2 days)

**Goal**: Enable basic correctness testing without full pipeline

**Tasks**:
1. Create `AttentionTestHarness` class
   - Wraps `PipelineBase::attention_gqa_tensor_parallel()`
   - Accepts raw float* pointers
   - Handles MPI context internally

2. Create `BaselineRunner` for single-rank reference
   - Calls `PipelineBase::attention_gqa()` directly
   - No MPI overhead

3. Add test utilities:
   - `compareTensors()` - Compute diff metrics
   - `initializeTestData()` - Deterministic test inputs

4. Implement `Test__MPITensorParallelCorrectness`
   - 3 tests: SingleToken, MultiToken, VariableSeqLengths
   - Uses test harness instead of full pipeline

**Deliverable**: Correctness validation without ModelContext dependency

### Phase 3b: Performance Framework (1-2 days)

**Goal**: Measure speedup and efficiency

**Tasks**:
1. Create `PerfHarness` class
   - Multi-trial timing with warmup
   - Statistics (mean/stddev/CV%)
   - GFLOPS calculation

2. Implement `Perf__MPITensorParallel`
   - SingleToken, MultiToken, ScalingAnalysis tests
   - Uses same harness as correctness tests

3. Integrate with CTest
   - Add to performance test suite
   - Apply optimal MPI/OpenMP settings
   - Label appropriately

4. Document expected performance
   - Baseline measurements
   - Speedup targets
   - Known limitations

**Deliverable**: Automated performance benchmarking

### Phase 3c: Full Pipeline Integration (3-5 days)

**Goal**: End-to-end testing with real models

**Prerequisites**:
- V2 model loading complete
- Qwen2Pipeline fully functional
- KV cache implemented

**Tasks**:
1. Create end-to-end correctness test
   - Load real Qwen 2.5 0.5B model
   - Run full forward pass (1 rank vs 2 ranks)
   - Compare all intermediate activations

2. Create end-to-end performance test
   - Measure full pipeline throughput
   - Include model loading overhead
   - Test with various batch sizes

3. Compare V2 vs V1 performance
   - Same model, same hardware
   - Measure V2 overhead
   - Identify optimization opportunities

**Deliverable**: Production-ready validation

---

## Workarounds for Current Testing

### Strategy 1: Unit Test Only (Current Approach)

**What We Can Test Now**:
- ✅ Strategy selection logic (`selectOptimalStrategy`)
- ✅ Strategy validation (`validateStrategy`)
- ✅ Head distribution (`getHeadDistribution`)
- ✅ MPI configuration defaults

**What We Can't Test**:
- ❌ Attention computation correctness
- ❌ Performance speedup
- ❌ Numerical stability

**Status**: Implemented in `Test__MPITensorParallel.cpp` (5/5 passing)

### Strategy 2: Integration Test with V1 Comparison

**Approach**: Use V1 infrastructure to validate V2 algorithm

**Steps**:
1. Extract `attention_gqa_tensor_parallel()` logic into standalone function
2. Create V1 test that calls this function
3. Compare V1 baseline vs tensor-parallel output
4. Port validated algorithm back to V2

**Pros**: Leverage mature V1 testing framework  
**Cons**: Duplicated code, not testing actual V2 implementation

### Strategy 3: Manual Testing Scripts

**Approach**: Write standalone C++ programs outside GTest

**Example**:
```cpp
// manual_mpi_test.cpp
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    // Manually construct minimal pipeline
    // Run attention with 2 ranks
    // Print comparison metrics
    
    MPI_Finalize();
}
```

**Pros**: Full control, no framework dependencies  
**Cons**: Not integrated into CI, manual execution

---

## Recommended Path Forward

### Short-Term (Next 1-2 weeks)

1. ✅ **Accept Phase 2 as complete** - Core algorithm implemented and compiles
2. ✅ **Document limitations** - This design doc serves as specification
3. ⬜ **Implement Phase 3a** - Minimal testing infrastructure (1-2 days)
4. ⬜ **Run correctness tests** - Validate algorithm correctness
5. ⬜ **Document results** - Update changelog with findings

### Medium-Term (Next 1-2 months)

6. ⬜ **Implement Phase 3b** - Performance framework
7. ⬜ **Run performance benchmarks** - Measure speedup
8. ⬜ **Optimize as needed** - Address bottlenecks
9. ⬜ **Update V2 architecture docs** - Add MPI section

### Long-Term (Next 3-6 months)

10. ⬜ **Complete V2 pipeline** - Full model loading, KV cache, etc.
11. ⬜ **Implement Phase 3c** - End-to-end integration testing
12. ⬜ **V2 production readiness** - Full validation suite
13. ⬜ **V1 → V2 migration** - Replace V1 with V2

---

## Success Criteria for Phase 3 (When Infrastructure Ready)

### Correctness Tests
- [ ] All tests pass with max_diff < 1e-4
- [ ] Single-rank vs multi-rank outputs identical
- [ ] No MPI deadlocks or hangs
- [ ] Reproducible across multiple runs
- [ ] Edge cases handled (uneven distribution, single head/rank)

### Performance Tests
- [ ] Speedup > 1.5× on 2 ranks (decode)
- [ ] Speedup > 1.7× on 2 ranks (prefill)
- [ ] Efficiency > 85% on 2 ranks (prefill)
- [ ] CV% < 5% (low variance)
- [ ] Communication overhead < 15%

### Integration
- [ ] Tests run automatically in CTest
- [ ] Performance results logged to CSV
- [ ] Clear pass/fail criteria
- [ ] Documented expected performance
- [ ] Tagged appropriately in CMake

---

## Summary

**Phase 2 Achievement**: ✅ **Complete**
- Tensor-parallel attention algorithm implemented (~450 lines)
- Strategy selection and validation working
- Qwen2Pipeline integrated
- Build successful, strategy tests passing (5/5)

**Phase 3 Status**: 📋 **Blocked on Infrastructure**
- Correctness tests designed but can't be implemented yet
- Performance tests designed but can't be implemented yet
- V2 testing infrastructure too immature for end-to-end testing

**Next Steps**:
1. Implement Phase 3a (minimal testing infrastructure) - 1-2 days
2. Run correctness validation - 1 day
3. Document results and proceed to Phase 3b (performance)

**Alternative**: Accept Phase 2 as milestone, defer Phase 3 until V2 matures (recommended for now)

---

**Last Updated**: 2025-10-25  
**Related Documents**:
- `changelog/2025-01-27-v2-mpi-phase1-infrastructure.md` - Phase 1 design
- `changelog/2025-01-27-v2-mpi-phase2-tensor-parallel-implementation.md` - Phase 2 implementation
- `.github/instructions/llaminar-v2-architecture.instructions.md` - V2 architecture overview

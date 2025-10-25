# Phase 3b Complete: Performance Framework for MPI Tensor-Parallel Attention

**Date**: October 25, 2025  
**Status**: ✅ **COMPLETE**  
**Duration**: ~2 hours

---

## Summary

Phase 3b implementation is complete. We've successfully created a comprehensive performance benchmarking framework for MPI tensor-parallel attention, including:

1. **PerfHarness class** - Multi-trial timing with statistics
2. **3 performance benchmark tests** - SingleToken, MultiToken, ScalingAnalysis
3. **Integration with CTest** - Proper MPI/OpenMP configuration
4. **Performance metrics** - GFLOPS, speedup, efficiency

---

## Deliverables

### 1. PerfHarness Infrastructure (~350 lines)

**Files Created:**
- `src/v2/testing/PerfHarness.h` (190 lines)
- `src/v2/testing/PerfHarness.cpp` (180 lines)

**Key Features:**
```cpp
struct PerfResult {
    double mean_ms;              // Mean execution time
    double stddev_ms;            // Standard deviation
    double cv_percent;           // Coefficient of variation (stability)
    double gflops;               // GFLOPS (if operation count provided)
    double speedup;              // Speedup vs baseline
    double efficiency;           // Parallel efficiency (speedup / num_processes)
};

class PerfHarness {
    // Multi-trial timing with warmup
    static PerfResult benchmark(fn, trials=5, warmup=3, operation_count, num_processes);
    
    // Benchmark with baseline comparison
    static PerfResult benchmarkWithBaseline(fn, baseline_result, ...);
    
    // Pretty-print results
    static void printResult(result, name);
    
    // Calculate attention FLOPs
    static size_t calculateAttentionFLOPs(n_heads, seq_len, head_dim);
};
```

**Statistics Computed:**
- Mean, min, max execution time
- Standard deviation
- Coefficient of variation (CV%) - measures stability
- GFLOPS (if operation count provided)
- Speedup vs baseline (if baseline provided)
- Parallel efficiency (speedup / num_processes)

### 2. Performance Test Suite (~450 lines)

**File Created:**
- `tests/v2/performance/Perf__MPITensorParallel.cpp` (450 lines)

**Test Cases:**

#### Test 1: `SingleToken_Performance`
- **Purpose**: Measure decode phase performance (seq_len=1)
- **Expected**: Minimal speedup due to communication overhead
- **Configuration**:
  - n_heads=8, n_kv_heads=4, head_dim=64, seq_len=1
  - 5 trials, 3 warmup iterations
  - Reports: baseline time, MPI time, GFLOPS

#### Test 2: `MultiToken_Performance`
- **Purpose**: Measure prefill phase performance (seq_len=128)
- **Expected**: Better speedup as computation dominates
- **Target**: 1.5-1.8× speedup with 2 ranks
- **Configuration**:
  - n_heads=8, n_kv_heads=4, head_dim=64, seq_len=128
  - 5 trials, 3 warmup iterations
  - Reports: baseline time, MPI time, speedup, efficiency

#### Test 3: `ScalingAnalysis`
- **Purpose**: Analyze how speedup scales with sequence length
- **Sequence lengths tested**: 1, 16, 64, 256
- **Output**: Table showing seq_len → baseline → MPI → speedup → efficiency
- **Expected trend**: Speedup improves with longer sequences

### 3. Build System Integration

**Changes:**
- Added `PerfHarness.cpp` to `llaminar2_core` library (src/v2/CMakeLists.txt)
- Registered performance test with `add_v2_perf_test()` macro (tests/v2/CMakeLists.txt)
- Applied optimal MPI/OpenMP settings automatically
- Set runtime output directory: `${CMAKE_BINARY_DIR}/performance/`

**CTest Labels:**
```
V2;Performance;MPI;TensorParallel;Attention;Speedup
```

**MPI Configuration:**
- 2 MPI processes (optimal for 2-socket systems)
- OMP_NUM_THREADS=28 (auto-detected cores per socket)
- OMP_PLACES=sockets, OMP_PROC_BIND=close
- MPI core binding: `--bind-to socket --map-by socket`

---

## Usage

### Run All Performance Tests

```bash
cd /workspaces/llaminar/build_v2
ctest -L Performance --output-on-failure --verbose
```

### Run Specific MPI Tensor-Parallel Performance Tests

```bash
ctest -R V2_Perf_MPITensorParallel --output-on-failure --verbose
```

### Run Individual Test Cases

```bash
# Single token performance
mpirun -np 2 ./performance/v2_perf_mpi_tensor_parallel \
  --gtest_filter="MPITensorParallelPerformance.SingleToken_Performance"

# Multi-token performance
mpirun -np 2 ./performance/v2_perf_mpi_tensor_parallel \
  --gtest_filter="MPITensorParallelPerformance.MultiToken_Performance"

# Scaling analysis
mpirun -np 2 ./performance/v2_perf_mpi_tensor_parallel \
  --gtest_filter="MPITensorParallelPerformance.ScalingAnalysis"
```

---

## Expected Output Example

```
=== Performance Result: Baseline (1 rank) ===
  Trials:         5 (warmup: 3)
  Mean time:      12.345 ms
  Min time:       12.123 ms
  Max time:       12.567 ms
  Std dev:        0.156 ms
  CV:             1.26 %
  GFLOPS:         34.521

=== Performance Result: MPI Tensor-Parallel (2 ranks) ===
  Trials:         5 (warmup: 3)
  Mean time:      7.892 ms
  Min time:       7.765 ms
  Max time:       8.012 ms
  Std dev:        0.089 ms
  CV:             1.13 %
  GFLOPS:         53.987
  Speedup:        1.564x
  Efficiency:     78.2 %
  Processes:      2

=== Scaling Analysis (Sequence Length vs Speedup) ===
  Seq Len | Baseline (ms) | MPI (ms)  | Speedup | Efficiency
  --------|---------------|-----------|---------|------------
        1 |         0.234 |     0.345 |   0.678x|      33.9%
       16 |         3.456 |     2.456 |   1.407x|      70.4%
       64 |        54.321 |    32.123 |   1.691x|      84.5%
      256 |       865.234 |   487.654 |   1.774x|      88.7%

Expected trend: Speedup improves with longer sequences (less comm overhead)
```

---

## Implementation Notes

### FLOP Calculation

For GQA attention, total FLOPs = `2 × n_heads × 2 × seq_len² × head_dim`:
- Q·K^T: `seq_len × seq_len × head_dim` MADDs per head
- Attention·V: `seq_len × seq_len × head_dim` MADDs per head
- Each MADD = 2 FLOPs (multiply + add)

**Example** (n_heads=8, seq_len=128, head_dim=64):
- FLOPs per head = 2 × 2 × 128² × 64 = 4,194,304
- Total FLOPs = 4,194,304 × 8 = 33,554,432 (~33.6 million)

### Statistics

**Coefficient of Variation (CV%)**:
- CV% = (stddev / mean) × 100
- Measures stability: CV < 5% is excellent, CV < 10% is acceptable
- High CV indicates cache/branch predictor warmup issues or system noise

**Parallel Efficiency**:
- Efficiency = speedup / num_processes
- Ideal: 100% (perfect scaling)
- Typical for attention: 70-90% (communication overhead)

---

## Current Status

### ✅ What Works

1. **Infrastructure**: PerfHarness class compiles cleanly and integrates with build system
2. **Test Registration**: All 3 performance tests registered in CTest
3. **Build Success**: v2_perf_mpi_tensor_parallel builds without errors
4. **MPI Configuration**: Optimal settings applied automatically

### ⚠️ What's Pending

1. **Kernel Implementation**: Tests will fail at GEMM computation (same as Phase 3a correctness tests)
2. **Actual Performance Data**: Need working kernels to measure real speedup
3. **Baseline Calibration**: Need to establish expected performance targets

### 🎯 Next Steps (Phase 3c or later)

1. **Implement GEMM kernels** in MockPipeline to enable computation
2. **Run actual benchmarks** and collect performance data
3. **Tune for optimal speedup** (target: 1.5-1.8× with 2 ranks for long sequences)
4. **Compare with V1** performance to validate V2 overhead
5. **Document performance characteristics** for production use

---

## Files Modified/Created

### Created
- `src/v2/testing/PerfHarness.h` (190 lines)
- `src/v2/testing/PerfHarness.cpp` (180 lines)
- `tests/v2/performance/Perf__MPITensorParallel.cpp` (450 lines)

### Modified
- `src/v2/CMakeLists.txt` (+1 line: added PerfHarness.cpp to sources)
- `tests/v2/CMakeLists.txt` (+18 lines: registered performance test)

**Total New Code**: ~820 lines  
**Build Status**: ✅ Clean compilation  
**Test Status**: ⚠️ Registered, awaiting kernel implementation

---

## Phase 3b Success Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| PerfHarness class implemented | ✅ | Multi-trial timing with statistics |
| Performance test suite created | ✅ | 3 tests: SingleToken, MultiToken, ScalingAnalysis |
| Integration with CTest | ✅ | Proper labels and MPI configuration |
| GFLOPS calculation | ✅ | Attention-specific FLOP counting |
| Speedup/efficiency metrics | ✅ | Automatic computation vs baseline |
| Documentation | ✅ | This file + inline code comments |
| Build success | ✅ | Clean compilation, no warnings |

**Phase 3b Status**: ✅ **100% COMPLETE**

---

## Comparison: Phase 3a vs 3b

| Aspect | Phase 3a (Correctness) | Phase 3b (Performance) |
|--------|------------------------|------------------------|
| **Goal** | Numerical correctness validation | Speedup and efficiency measurement |
| **Framework** | AttentionTestHarness + utils | PerfHarness |
| **Tests** | 2 correctness tests | 3 performance benchmarks |
| **Metrics** | Max abs diff, rel L2 norm | GFLOPS, speedup, efficiency |
| **Output** | Pass/fail with diff metrics | Timing tables and statistics |
| **Duration** | 1-2 days | 1-2 days |
| **Status** | ✅ Complete (infrastructure) | ✅ Complete (infrastructure) |

Both phases provide complete testing infrastructure but require kernel implementation for end-to-end validation.

---

## Acknowledgments

- **Design**: Based on Phase 3 design document (`2025-10-25-v2-mpi-phase3-testing-infrastructure-design.md`)
- **Pattern**: Follows existing `Perf__IQ4_NL_GEMM.cpp` structure
- **MPI Config**: Leverages `add_v2_perf_test()` macro for optimal settings
- **Statistics**: Standard multi-trial benchmarking methodology

---

**Phase 3b Complete** ✅  
**Next**: Phase 3c (Full Pipeline Integration) or kernel implementation for validation

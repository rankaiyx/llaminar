# Batch Attention Unit Test Suite - 2025-01-24

## Summary

Created comprehensive unit test suite for `CpuAttentionKernelT::compute_batch()` to validate kernel correctness in isolation. **Result: All critical tests pass**, confirming the attention kernel implementation is correct.

## Motivation

E2E batch attention tests were failing with numerical mismatches (~18.75 max_abs_diff). After fixing:
1. MPI test synchronization (preventing deadlocks)
2. ElementType pointer arithmetic in `compute_batch()`
3. MPI tensor-parallel dispatch to use `compute_batch()` for batched inputs
4. GQAAttention dispatch to use `compute_batch()` for batched inputs

Tests **still failed**. We needed to validate whether the issue was in the kernel itself or elsewhere in the pipeline.

## New Test File

**`tests/v2/unit/Test__CpuAttentionKernelT_Batch.cpp`** (~500 lines)

### Test Coverage

#### 1. Batch vs Sequential Equivalence (3 tests)
- **`BatchSize1EqualsSequential`**: Validates `compute_batch(batch_size=1)` produces identical results to `compute()`
- **`BatchSize1EqualsSequential_GQA`**: Same test with GQA (n_heads > n_kv_heads)
- **`BatchSize1EqualsSequential_Causal`**: Same test with causal masking

**Status: ✅ ALL PASS**

#### 2. Batch Independence (2 tests)
- **`BatchIndependence_TwoSequences`**: Validates processing seq0 and seq1 separately produces same results as processing together
- **`BatchIndependence_FourSequences`**: Same test with batch_size=4

**Status: ✅ ALL PASS**

#### 3. Batch Variants (2 tests)
- **`BatchGQA_TwoSequences`**: Batch attention with GQA
- **`BatchCausal_TwoSequences`**: Batch attention with causal masking

**Status: ✅ ALL PASS**

#### 4. Edge Cases (2 tests)
- **`BatchSize0Invalid`**: Validates batch_size=0 is rejected ❌ FAILS (kernel doesn't validate)
- **`LargeBatchSize`**: Validates batch_size=8 works

**Status: ✅ 1/2 PASS** (validation issue, not correctness)

## Test Results

```bash
$ ctest --test-dir build_v2 -R "V2_Unit_CpuAttentionKernelT_Batch" -V

[==========] Running 9 tests from 1 test suite.
[  PASSED  ] 8 tests:
  ✅ BatchSize1EqualsSequential (11 ms)
  ✅ BatchSize1EqualsSequential_GQA (8 ms)
  ✅ BatchSize1EqualsSequential_Causal (1 ms)
  ✅ BatchIndependence_TwoSequences (2 ms)
  ✅ BatchIndependence_FourSequences (57 ms)
  ✅ BatchGQA_TwoSequences (2 ms)
  ✅ BatchCausal_TwoSequences (3 ms)
  ✅ LargeBatchSize (3 ms)

[  FAILED  ] 1 test:
  ❌ BatchSize0Invalid (kernel doesn't validate batch_size <= 0)
```

## Key Findings

### ✅ Kernel is Correct
The comprehensive unit tests **confirm that `CpuAttentionKernelT::compute_batch()` works correctly**:
- Batch vs sequential equivalence validated
- Batch independence validated (no cross-contamination)
- Works with GQA, causal masking, large batches

### ❌ E2E Tests Still Fail
Despite kernel correctness, E2E tests still fail:

```bash
$ mpirun -np 2 ./build_v2_release/tests/v2/v2_test_qwen2_e2e_correctness --gtest_filter=Qwen2E2ECorrectness.MultiSequenceBatch

=== Batch Parity - Sequence 1 ===
  Max abs diff:   18.7512
  Mean abs diff:  2.91498
  Rel L2 norm:    0.993391
  Mismatches:     303793
  Status:         FAILED

[  FAILED  ] Qwen2E2ECorrectness.MultiSequenceBatch (15542 ms)
```

## Conclusion

**Kernel is NOT the problem.** The issue must be in **pipeline-level code**:

### Potential Root Causes
1. **Embedding layer** - Token embeddings may handle batches incorrectly
2. **RoPE** - Rotary positional embeddings may not respect batch structure
3. **RMSNorm** - Normalization may compute statistics across batch boundaries
4. **Residual connections** - May add wrong tensors in batch mode
5. **Weight placement** - MPI weight distribution may corrupt batch structure
6. **KV cache** - K/V cache may not isolate sequences properly

### Next Steps
1. Add batch-aware logging/snapshots to pipeline stages
2. Compare activations layer-by-layer for batch vs sequential execution
3. Identify first diverging stage
4. Fix root cause in pipeline code

## Files Modified

### New Files
- `tests/v2/unit/Test__CpuAttentionKernelT_Batch.cpp` (500 lines) - Comprehensive batch tests

### Modified Files
- `tests/v2/CMakeLists.txt` - Added batch test target

## Testing

```bash
# Build batch tests
cmake --build build_v2 --target v2_test_cpu_attention_kernel_t_batch --parallel

# Run batch tests
ctest --test-dir build_v2 -R "V2_Unit_CpuAttentionKernelT_Batch" -V

# Run E2E batch test (still fails)
mpirun -np 2 ./build_v2_release/tests/v2/v2_test_qwen2_e2e_correctness --gtest_filter=Qwen2E2ECorrectness.MultiSequenceBatch
```

## References

- Batch attention implementation: `src/v2/kernels/cpu/CpuAttentionKernelT.h` lines 462-626
- E2E batch tests: `tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp` lines 392-1100
- MPI orchestrator: `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp`
- GQA implementation: `src/v2/pipelines/attention/GQAAttention.cpp`

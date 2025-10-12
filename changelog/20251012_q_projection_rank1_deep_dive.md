# Q Projection Rank 1 Bug Investigation - Session 2
**Date**: 2025-01-12
**Status**: Bug isolated to execution phase, not weight loading

## Session Summary

After extensive debugging with diagnostic logging, weight file dumps, and cblas parameter verification, we have definitively isolated the bug but not yet identified the root cause.

## Key Findings

### ✅ Verified CORRECT

1. **Weight Loading**: Both ranks load correct weight slices from disk
   - Rank 0: rows [0:448] of wq matrix
   - Rank 1: rows [448:896] of wq matrix
   - Confirmed via `qwen_pipeline.cpp` logging showing correct `row_slice` values

2. **Weight Assignment**: Both ranks correctly assign loaded weights to `local_wq`
   - Rank 0 wq_global: `0x64b44e0070e0` → `[-0.00227356, -0.00500488, ...]`
   - Rank 1 wq_global: `0x5da2f94a7840` → `[-0.0228271, -0.0157471, ...]`
   - Different pointers ✅, Different values ✅, Correct mapping ✅

3. **Input Broadcasting**: Both ranks receive identical input after MPI_Bcast
   - `min=-1.77143 max=0.642602 mean=-0.00290018` (exact match)
   - First 5 values: `[0.041737, -0.00297965, 0.0274317, -0.0102278, -0.011885]` (exact match)

4. **Matmul Parameters**: cblas_sgemm receives correct parameters on both ranks
   - M=5, N=448, K=896 (same for both ranks)
   - Leading dimensions: input=896, weight=896, output=448 (correct for row-major)
   - Weight transpose flag: CblasTrans (correct)

5. **Weight Verification**: Test infrastructure confirmed all 97 tensors match PyTorch
   - Max diff: 0.000000 (FP32 precision perfect match)
   - Includes both rank 0 and rank 1 weight tensors

### ❌ Still WRONG

**Rank 1 Q Projection Output**: Despite all inputs and parameters being correct, rank 1 produces wrong output

**Evidence**:
```
PyTorch head 7 (rank 1's first head):  [ 0.0824,  1.5207,  0.2346,  4.0186, -3.2651]
Llaminar rank 1 head 0 (global head 7): [-0.1942,  0.5540, -0.3729,  1.0078, -15.5776]
Max absolute difference: 12.312474
```

Meanwhile, rank 0 is **perfect**:
```
PyTorch head 0:   [ 0.0826,  0.3144,  0.0390, -0.8743, -15.5209]
Llaminar rank 0:  [ 0.0826,  0.3144,  0.0390, -0.8743, -15.5209]
Max absolute difference: 0.000046 (FP32 precision)
```

## Investigation Tools Created

1. **debug_q_projection_comparison.py**: Compares Llaminar's Q projection output with PyTorch head-by-head
2. **debug_weight_comparison.py**: Compares loaded weight files with PyTorch reference
3. **debug_manual_q_projection.py**: Manually performs matmul with saved weights to isolate bug
4. **Enhanced logging in MPIAttentionKernel.cpp**:
   - `[Q_PROJ_DEBUG]`: Input/weight stats before projection
   - `[Q_GATHER_DEBUG]`: Per-head values before/after MPI gather
   - `[WQ_VERIFY]`: wq_global stats and file dumps
   - `[PTR_DEBUG]`: Pointer assignment verification
   - `[MATMUL_DEBUG]`: Matmul parameters
   - `[CBLAS_DEBUG]`: cblas_sgemm call parameters

## Hypotheses Tested and REJECTED

1. ❌ **Weight slicing bug**: Rank 1 uses wrong row offset
   - REJECTED: Logging shows correct `row_slice=[448:896]`

2. ❌ **Weight loading bug**: Rank 1 loads wrong data from file
   - REJECTED: Weight verification passed for all tensors

3. ❌ **MPI gather bug**: Global tensor assembled incorrectly
   - REJECTED: Bug occurs BEFORE gather (in local_q computation)

4. ❌ **Pointer aliasing**: Ranks share same weight buffer
   - REJECTED: Different pointer addresses confirmed

5. ❌ **Input broadcast bug**: Rank 1 receives different input
   - REJECTED: Input statistics and samples match exactly

6. ❌ **Matmul parameter bug**: Wrong M/N/K or leading dimensions
   - REJECTED: All parameters correct for both ranks

7. ❌ **COSMA vs OpenBLAS**: Rank 1 uses different backend
   - REJECTED: COSMA disabled via `ADAPTIVE_DISABLE_COSMA=1`

8. ❌ **Weight transpose bug**: Weight matrix transposed for rank 1
   - REJECTED: Both ranks show same shape `[448, 896]`

## Remaining Hypotheses

### Most Likely: Memory Corruption During Execution

Despite all inputs being correct BEFORE the matmul call, something corrupts rank 1's computation. Possible causes:

1. **OpenBLAS thread safety issue**: Even with `openblas_set_num_threads(1)`, there might be residual threading bugs when multiple MPI ranks call OpenBLAS simultaneously

2. **Stack/heap corruption**: Some buffer overflow or memory stomp between weight loading and matmul execution

3. **Compiler optimization bug**: Aggressive optimization causing undefined behavior in rank 1's code path

4. **Shared library state**: OpenBLAS internal state corrupted by concurrent calls from both MPI ranks

5. **Memory alignment issue**: Rank 1's weight buffer misaligned causing incorrect SIMD operations

## Next Investigation Steps

### Immediate Actions

1. **Test with explicit memory barriers**:
   ```cpp
   MPI_Barrier(MPI_COMM_WORLD);  // Before weight loading
   // Load weights
   MPI_Barrier(MPI_COMM_WORLD);  // Before matmul
   // Perform matmul
   MPI_Barrier(MPI_COMM_WORLD);  // After matmul
   ```

2. **Verify OpenBLAS is truly single-threaded**:
   ```cpp
   int actual_threads = openblas_get_num_threads();
   LOG_INFO("OpenBLAS threads: " << actual_threads);
   ```

3. **Try alternative BLAS implementation**:
   - Test with reference BLAS (unoptimized but reliable)
   - Test with MKL if available

4. **Serialize MPI rank execution**:
   ```cpp
   for (int r = 0; r < world_size; ++r) {
       if (rank == r) {
           // Perform matmul
       }
       MPI_Barrier(MPI_COMM_WORLD);
   }
   ```

5. **Add memory poisoning checks**:
   ```cpp
   // Fill output buffer with known pattern before matmul
   std::fill_n(local_q->data(), local_q->size(), -99999.0f);
   // Perform matmul
   // Check if any -99999 values remain (indicates incomplete write)
   ```

6. **Test with different compilers/optimization levels**:
   - `-O0` (no optimization)
   - `-O2` vs `-O3`
   - Different compilers (gcc vs clang)

7. **Memory sanitizer run**:
   ```bash
   cmake -DCMAKE_CXX_FLAGS="-fsanitize=memory -fno-omit-frame-pointer" ...
   ```

8. **Valgrind memcheck**:
   ```bash
   mpirun -np 2 valgrind --tool=memcheck --leak-check=full ./build/test_parity_framework
   ```

### Diagnostic Additions

Add logging to verify OpenBLAS actually executes correctly:

```cpp
// Before matmul
float weight_checksum = std::accumulate(weight, weight + N*K, 0.0f);
float input_checksum = std::accumulate(input, input + M*K, 0.0f);

// After matmul
float output_checksum = std::accumulate(output, output + M*N, 0.0f);
LOG_INFO("Checksums: weight=" << weight_checksum << " input=" << input_checksum << " output=" << output_checksum);
```

## Code Modifications Made

### src/qwen_pipeline.cpp
- Line 1132: Changed weight loading log to include rank information
- Now logs: `[WeightLoad] Rank X blk.Y.attn_q.weight row_slice=[...]`

### src/kernels/MPIAttentionKernel.cpp
- Added comprehensive diagnostic logging (lines 693-756)
- Added weight file dumps for comparison
- Added matmul parameter logging
- Added cblas_sgemm parameter logging
- Added pointer assignment verification

## Environment Variables Used

```bash
ADAPTIVE_DISABLE_COSMA=1  # Force OpenBLAS path
LLAMINAR_PARITY_CAPTURE=1  # Enable snapshot capture
```

## Test Execution

```bash
# Standard test run
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch"

# With detailed output
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch" 2>&1 | grep -E "Q_PROJ_DEBUG|CBLAS_DEBUG|PTR_DEBUG"
```

## Critical Data Points

**Rank 0 (CORRECT)**:
- wq pointer: `0x64b44e0070e0`
- wq first 5: `[-0.00227356, -0.00500488, 0.0187988, 0.0124512, 0.00408936]`
- wq stats: `min=-1.03906 max=0.96875 mean=-9.49203e-06`
- Q output[0, :5]: `[0.0826069, 0.314376, 0.0389893, -0.874254, -15.5209]`
- vs PyTorch: **0.000046 max diff** ✅

**Rank 1 (WRONG)**:
- wq pointer: `0x5da2f94a7840`
- wq first 5: `[-0.0228271, -0.0157471, 0.0179443, 0.0490723, 0.043457]`
- wq stats: `min=-1.22656 max=1.17188 mean=-2.38856e-05`
- Q output[0, :5]: `[-0.19424, 0.554032, -0.37286, 1.00782, -15.5776]`
- vs PyTorch: **12.312474 max diff** ❌

**Common (SAME FOR BOTH)**:
- Input first 5: `[0.041737, -0.00297965, 0.0274317, -0.0102278, -0.011885]`
- Input stats: `min=-1.77143 max=0.642602 mean=-0.00290018`
- Matmul shape: M=5, N=448, K=896

## Conclusion

This is a particularly insidious bug because:
1. All inputs are provably correct
2. All parameters are provably correct
3. The bug manifests only during execution
4. Only affects rank 1, not rank 0

The bug is most likely related to:
- OpenBLAS internal state corruption in multi-process environment
- Memory corruption during matmul execution
- Compiler optimization producing incorrect code for rank 1's code path

The fix will likely involve one of:
- Serializing BLAS calls across ranks
- Switching BLAS implementation
- Adding explicit memory barriers
- Disabling compiler optimizations
- Finding and fixing a memory corruption bug in the codebase

## Files Modified

- `src/qwen_pipeline.cpp` (weight loading logging)
- `src/kernels/MPIAttentionKernel.cpp` (extensive diagnostic logging)
- `debug_q_projection_comparison.py` (created)
- `debug_weight_comparison.py` (created)
- `debug_manual_q_projection.py` (created)

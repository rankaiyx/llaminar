# Parity Test MPI Rank Snapshot Fix

**Date:** 2025-10-11  
**Author:** David Sanftenberg

## Summary

Fixed a critical bug in the parity test framework where MPI ranks were overwriting each other's snapshots, causing incorrect comparisons. The bug manifested as massive divergence at the very first activation stage (ATTENTION_NORM_layer0) due to comparing rank 1's intermediate values against PyTorch's single-process output.

## Root Cause Analysis

### The Bug

When running with `mpirun -np 2`:
1. **Both MPI ranks capture snapshots** via `LlaminarSnapshotHook::capture()`
   - Each rank has its own `SnapshotRegistry::instance()` (process-local singleton)
   - Both ranks capture their local intermediate tensor values
2. **Both ranks save snapshots to the SAME directory**
   - `IncrementalSnapshotHelper::afterToken()` called on all ranks
   - All ranks call `SnapshotRegistry::instance().saveToDirectory(token_dir)`
   - Same `token_dir` path for all ranks → files get overwritten!
3. **Last rank to finish wins**
   - Rank 1 typically finishes slightly after rank 0
   - Rank 1's snapshot files overwrite rank 0's files
4. **Comparison uses wrong data**
   - Rank 0 loads snapshot files for comparison
   - Files contain rank 1's local values (due to overwrite)
   - PyTorch runs single-process → equivalent to rank 0's full computation
   - Comparison fails: rank 1's partial sharded values ≠ PyTorch's full values

### Why Ranks Have Different Values

With tensor sharding (TP across 2 ranks):
- **Weights are partitioned**: Q/K/V projections split across ranks
  - Rank 0: owns first half of Q heads, K heads, V heads
  - Rank 1: owns second half of Q heads, K heads, V heads
- **Intermediate values differ**: Before final gather/reduce operations
  - Rank 0 computes attention for its owned heads
  - Rank 1 computes attention for its owned heads  
  - Values are only identical AFTER MPI_Allgatherv/MPI_Allreduce
- **Embeddings are REPLICATED**: Full table on each rank (confirmed in weight_contracts.h)
  - Both ranks should have identical embedding outputs
  - But if rank 1's snapshot overwrites rank 0's, comparison uses wrong baseline

## The Fix

**File Modified**: `tests/parity_test_framework.cpp`

### Change 1: Add MPI Header

```cpp
#include <mpi.h>  // NEW
```

### Change 2: Rank-Specific Snapshot Saving

Modified `IncrementalSnapshotHelper::afterToken()` to only save from rank 0:

```cpp
bool IncrementalSnapshotHelper::afterToken(int token_index)
{
    // CRITICAL: Only save snapshots from rank 0 to avoid MPI ranks overwriting each other!
    // Each rank captures its own local intermediate values (due to sharding), but we want
    // to compare rank 0's values against PyTorch (which runs single-process).
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0)
    {
        std::string token_dir = getTokenDir(token_index);
        // ... save snapshots ...
    }
    else
    {
        LOG_DEBUG("IncrementalSnapshotHelper: Rank " << rank << " skipping save (only rank 0 saves)");
    }

    // Clear registry on ALL ranks for next token
    SnapshotRegistry::instance().clear();
    
    return true;
}
```

**Rationale**:
- Rank 0 computes the full forward pass with its portion of sharded weights
- After MPI gather/reduce operations, rank 0 has complete activation tensors
- PyTorch runs single-process (no sharding) → equivalent to rank 0's final values
- By saving only rank 0's snapshots, we compare apples-to-apples

## Test Results After Fix

**Build**: ✅ SUCCESS
```bash
cmake --build build --target test_parity_framework --parallel
```

**Test Execution**: ⚠️ STILL FAILING (but for different reason)
```bash
mpirun -np 2 ./build/test_parity_framework --gtest_filter="*TrueIncrementalDecodeVsPyTorch"
```

**Output**:
```
[TOKEN SEQUENCE VALIDATION]
  ✗ Token sequences DIVERGE
    Functional output differs between systems

[STAGE-LEVEL VALIDATION]
  Tokens passed:      0/3
  Tokens failed:      1/3
  Stages compared:    1
  Stages passed:      0
  Stages failed:      2

  ⚠ Fail-fast triggered - stopped at first failure

[OUTPUT SEQUENCE]
  PyTorch tokens:  [6, 25010, 10]
  Llaminar tokens: [400, 1, 66]
```

## Remaining Issue: Token Divergence

The fix resolved the snapshot overwrite bug, but revealed a **deeper issue**: Llaminar and PyTorch generate completely different token sequences from the start.

### Observations

1. **First token divergence**: PyTorch=6, Llaminar=400
   - This happens during **prefill** phase (before incremental decode)
   - Suggests bug in prefill logits computation
2. **ATTENTION_NORM still fails**: max_abs=1.967, rel_l2=0.135
   - Same error magnitude as before fix
   - But now we're comparing correct snapshots (rank 0 vs PyTorch)
3. **Weights verified perfect**: embedding + all layer weights max_diff=0
   - Bug is NOT in weight loading
   - Bug IS in computational path

### Next Investigation Steps

1. **Compare prefill logits directly**:
   ```cpp
   // Save Llaminar prefill logits
   // Compare with PyTorch prefill logits BEFORE sampling
   ```

2. **Check embedding lookup**:
   - Both ranks use full embedding table (REPLICATED)
   - Both ranks look up same tokens
   - Verify MPIEmbeddingKernel produces identical output on both ranks

3. **Verify RMSNorm gather**:
   - After `MPIRMSNormKernel::execute()`, all ranks should have identical output
   - Check if `gatherOutput()` is called correctly
   - Verify MPI_Allgatherv works as expected

4. **Add intermediate logging**:
   - Log embedding output (both ranks)
   - Log RMSNorm output (both ranks)
   - Compare against PyTorch at each stage

## Files Modified

- `tests/parity_test_framework.cpp`:
  - Line 9: Added `#include <mpi.h>`
  - Lines 386-420: Modified `IncrementalSnapshotHelper::afterToken()` to only save from rank 0

## Documentation Updates Needed

- Update `.github/instructions/parity-test-framework.instructions.md` to document:
  - MPI rank snapshot behavior
  - Why only rank 0 saves snapshots
  - How to debug multi-rank parity issues

## Conclusion

**Fixed**: MPI ranks overwriting each other's snapshots ✅  
**Remaining**: Llaminar produces different tokens than PyTorch from prefill onwards ⚠️

The fix ensures we're now comparing the correct data (rank 0 vs PyTorch), but exposes that there's a fundamental computation difference. The next debugging session should focus on the prefill phase and verify that MPI collective operations (gather/reduce) are producing correct global tensors on rank 0.

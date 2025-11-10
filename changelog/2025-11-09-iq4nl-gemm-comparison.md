# IQ4_NL GEMM Implementation Performance Comparison

**Date**: November 9, 2025  
**System**: 2-socket server with MPI distribution (2 ranks, 28 OpenMP threads each)  
**Model**: Qwen 2.5 0.5B IQ4_NL quantized weights

## Executive Summary

**Result: Both implementations perform identically** across all workload sizes from single-token decode (1×896×896) to large prefill (4096×4864×896).

All test cases show performance **within 5% margin** (marked as ties), with GFLOPS representing the combined throughput of both MPI ranks working together on the full matrix.

## Recommendation

 **Use AutoTunedQuantizedGemm (generic)** because:
- Same performance as specialized implementation
- Simpler codebase (single implementation for all quantized formats)
- Easier to maintain and extend
- Works via IBlockDecoder strategy pattern

## Performance Results

### Decode Workloads (Small Batch)

| Test Case | m×n×k | Specialized | Generic | Speedup |
|-----------|-------|-------------|---------|---------|
| Single Token | 1×896×896 | 8.88ms (0.2 GFLOPS) | 8.86ms (0.2 GFLOPS) | 1.00x |
| Small Batch | 32×896×896 | 8.92ms (5.8 GFLOPS) | 8.80ms (5.8 GFLOPS) | 1.01x |
| Medium Batch | 128×896×896 | 9.02ms (22.8 GFLOPS) | 8.98ms (22.9 GFLOPS) | 1.00x |
| Large Batch | 512×896×896 | 15.40ms (53.4 GFLOPS) | 15.58ms (52.8 GFLOPS) | 0.99x |

### FFN Workloads (Intermediate Dimension)

| Test Case | m×n×k | Specialized | Generic | Speedup |
|-----------|-------|-------------|---------|---------|
| FFN-Up Small | 16×4864×896 | 8.96ms (15.6 GFLOPS) | 8.86ms (15.7 GFLOPS) | 1.01x |
| FFN-Up Medium | 128×4864×896 | 9.00ms (124.0 GFLOPS) | 8.96ms (124.5 GFLOPS) | 1.00x |

### Prefill Workloads (Long Context)

**Q-Projection (896×896 weights):**

| Tokens | m×n×k | Specialized | Generic | Speedup |
|--------|-------|-------------|---------|---------|
| 1024 | 1024×896×896 | 5.38ms (305.5 GFLOPS) | 5.21ms (315.6 GFLOPS) | 1.03x |
| 2048 | 2048×896×896 | 10.06ms (326.9 GFLOPS) | 9.99ms (329.2 GFLOPS) | 1.01x |
| 4096 | 4096×896×896 | 19.82ms (331.7 GFLOPS) | 20.18ms (325.9 GFLOPS) | 0.98x |

**FFN-Up Projection (4864×896 weights):**

| Tokens | m×n×k | Specialized | Generic | Speedup |
|--------|-------|-------------|---------|---------|
| 1024 | 1024×4864×896 | 20.33ms (439.0 GFLOPS) | 19.98ms (446.7 GFLOPS) | 1.02x |
| 2048 | 2048×4864×896 | 41.22ms (433.0 GFLOPS) | 41.77ms (427.3 GFLOPS) | 0.99x |
| 4096 | 4096×4864×896 | 80.03ms (446.1 GFLOPS) | 82.17ms (434.5 GFLOPS) | 0.97x |

## Key Findings

1. **Peak Performance**: ~450 GFLOPS achieved on large FFN operations (both implementations)
2. **Scaling**: Performance scales well from 1K to 4K tokens (305→331 GFLOPS for Q-proj, 439→446 GFLOPS for FFN-up)
3. **MPI Distribution**: Both ranks cooperate effectively via MPI_Allgather (work division + result aggregation)
4. **No Winner**: Performance difference ranges from -3% to +3% across all cases (measurement noise)

## Architecture Notes

### IQ4_NLQuantizedGemm (Specialized)
- Adaptive cache tiling based on m dimension
- m ≤ 16: Cache-blocked micro-kernel strategy
- m > 16: Row-wise processing
- BF16/INT8 VNNI optimized paths

### AutoTunedQuantizedGemm (Generic)
- Uses GemmAutoTuner infrastructure
- Selects optimal micro-kernel variant per operation
- Works with any ITensorGemmTileDataProvider (IQ4_NL, Q6_K, Q8_0, etc.)
- Single implementation for all quantized formats

## Test Methodology

- **MPI Configuration**: 2 ranks, each processing m/2 rows
- **Aggregation**: MPI_Allgather to combine partial results
- **GFLOPS Calculation**: Based on full matrix (2×m×n×k FLOPs / combined time)
- **Warmup**: 5 iterations per implementation
- **Benchmark**: 50 iterations per implementation
- **Synchronization**: MPI_Barrier before/after timing

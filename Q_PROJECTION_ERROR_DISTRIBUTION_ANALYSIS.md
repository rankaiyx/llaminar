# Q_PROJECTION and K_PROJECTION Error Distribution Analysis

**Date:** October 8, 2025  
**Context:** Investigating whether Q_PROJECTION max_abs=0.11 masks systematic errors that could cause ROPE_APPLICATION failure (max_abs=97.45)

## Executive Summary

✅ **Q_PROJECTION and K_PROJECTION errors are small, well-distributed, and NOT systematic**
- Q_PROJECTION: max=0.11, mean=0.008, median=0.006
- K_PROJECTION: max=0.095, mean=0.009, median=0.005
- 99.8% of Q elements have error <0.1
- 98.75% of K elements have error <0.1

❌ **ROPE_APPLICATION failure (max_abs=97.45) is NOT caused by propagating Q/K projection errors**
- Error positions [3,992] and [4,992] map to K tensor region
- 880x error amplification (0.11 → 97.45) indicates major bug in RoPE or gathering
- Most likely: MPI_Allgather layout mismatch or RoPE distributed computation bug

## Detailed Q_PROJECTION Analysis

### Overall Statistics
```
max_abs:    0.110511
mean_abs:   0.008054
std_abs:    0.007866
median_abs: 0.005992
```

### Error Distribution (Percentiles)
| Percentile | Error Value |
|------------|-------------|
| P50 | 0.005992 |
| P75 | 0.011432 |
| P90 | 0.017619 |
| P95 | 0.022950 |
| P99 | 0.034381 |

**Interpretation:** 99% of elements have error <0.034, showing that the max_abs=0.11 is a genuine outlier, not representative of systematic error.

### Per-Head Error Analysis
Configuration: 14 attention heads × 64 dimensions = 896 total dims

| Head | Max Abs Error |
|------|---------------|
| 12 | 0.110511 (worst) |
| 1 | 0.056027 |
| 2 | 0.054655 |
| 9 | 0.053313 |
| 3 | 0.051629 |

**Interpretation:** Head 12 contains the single outlier (0.11). Most heads have max errors <0.056, which is normal quantization noise for Q4_0 format.

### Error Histogram
| Range | Count | Percentage |
|-------|-------|------------|
| [0.000, 0.001) | 563 | 12.57% |
| [0.001, 0.010) | 2533 | 56.54% |
| [0.010, 0.050) | 1374 | 30.67% |
| [0.050, 0.100) | 9 | 0.20% |
| [0.100, 0.150) | 1 | 0.02% |
| [0.150, inf) | 0 | 0.00% |

**Key Finding:** Only 1 element (0.02%) exceeds 0.1 error. The distribution is dominated by small errors (0.001-0.010 range = 56.54%).

## Detailed K_PROJECTION Analysis

### Overall Statistics
```
max_abs:    0.095051
mean_abs:   0.008602
std_abs:    0.010991
median_abs: 0.004683
```

### Error Distribution (Percentiles)
| Percentile | Error Value |
|------------|-------------|
| P50 | 0.004683 |
| P75 | 0.011385 |
| P90 | 0.020107 |
| P95 | 0.027984 |
| P99 | 0.055763 |

### Per-KV-Head Error Analysis
Configuration: 2 KV heads × 64 dimensions = 128 total dims (GQA)

| KV-Head | Max Abs Error |
|---------|---------------|
| 1 | 0.095051 (worst) |
| 0 | 0.043488 |

**Interpretation:** KV-Head 1 has higher error (0.095), but still within tolerance. Head 0 is very clean (0.043).

### Error Histogram
| Range | Count | Percentage |
|-------|-------|------------|
| [0.000, 0.001) | 105 | 16.41% |
| [0.001, 0.010) | 355 | 55.47% |
| [0.010, 0.050) | 172 | 26.88% |
| [0.050, 0.100) | 8 | 1.25% |
| [0.100, inf) | 0 | 0.00% |

**Key Finding:** K_PROJECTION has slightly more elements in the 0.05-0.1 range (1.25% vs Q's 0.20%), but still well within tolerance.

## ROPE_APPLICATION Failure Analysis

### Top Error Locations
```
[PARITY_TOP_DIFF] ROPE_APPLICATION top_k=5
  [3,992] diff=97.453476 expected=47.602432 actual=-49.851048
  [4,992] diff=89.827995 expected=39.817192 actual=-50.010803
  [2,992] diff=60.283051 expected=11.416624 actual=-48.866428
  [4,8]   diff=45.963718 expected=-17.691280 actual=28.272438
  [2,960] diff=45.006058 expected=48.093239 actual=3.087183
```

### Tensor Layout Analysis
ROPE_APPLICATION format: `[Q | K]` concatenated
- Q section: positions 0-895 (14 heads × 64 dims)
- K section: positions 896-1023 (2 KV heads × 64 dims)

**Error position mapping:**
- Position 992: **K section** (992 - 896 = 96 → KV-Head 1, dimension 32)
- Position 960: **K section** (960 - 896 = 64 → KV-Head 1, dimension 0)
- Position 8: **Q section** (Head 0, dimension 8)

**Critical Finding:** 3 out of 5 top errors are in the K section at position 992 (different sequence positions). This suggests:
1. K tensor at dimension 992-896=96 has wrong values after RoPE
2. OR K tensor gathering concatenates heads in wrong order
3. OR K head 1 has systematic RoPE calculation error

## Conclusions

### Q_PROJECTION and K_PROJECTION are NOT the problem
1. ✅ Error distributions are well-behaved and typical of Q4_0 quantization
2. ✅ Mean errors (0.008-0.009) are 10x smaller than max (0.09-0.11)
3. ✅ 99% of elements have errors <0.056
4. ✅ Errors are NOT systematic - they're random quantization noise
5. ✅ Max values (0.09-0.11) are genuine outliers, not masking hidden patterns

### ROPE_APPLICATION error is introduced AFTER projection
1. ❌ 880x error amplification (0.11 → 97.45) cannot come from projections
2. ❌ Errors concentrated in K section at specific positions (992, 960)
3. ❌ Most likely causes:
   - **MPI_Allgather concatenates K heads in wrong order**
   - **RoPE calculation uses wrong head indices in distributed computation**
   - **K tensor layout mismatch between PyTorch sequential [h0, h1] and MPI gathered [rank0_h0, rank1_h1]**

## Next Steps

### High Priority: Investigate K tensor gathering
1. Verify MPI_Allgather for K uses correct head dimensions (2 KV heads × 64 dims = 128)
2. Check if gathering produces sequential [KV-head0 all dims | KV-head1 all dims] matching PyTorch
3. Add debug logging to show K tensor layout before/after MPI_Allgather

### Medium Priority: Verify RoPE calculation for GQA
1. Check if RoPE applies correct head indices when q_heads (14) ≠ k_heads (2)
2. Verify theta calculation uses correct head_dim for K heads
3. Compare RoPE output for K head 1 specifically (where max errors occur)

### Validation
1. After fix, expect K_PROJECTION → ROPE_APPLICATION K section to maintain <0.1 error
2. ROPE_APPLICATION should pass with max_abs <0.15 (allowing for RoPE numerical precision)
3. Downstream ATTENTION_SCORES should improve dramatically

## Supporting Data

Test command:
```bash
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch"
```

Configuration:
- Model: Qwen-0.5B (14 Q heads, 2 KV heads, 64 head_dim)
- Sequence length: 5 tokens
- MPI ranks: 2 (7 Q heads per rank, 1 KV head per rank)
- Total parameters: Q=896, K=128, V=128

Quantization: Q4_0 for weights (4-bit quantized)

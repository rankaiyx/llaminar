# Q8_1 vs FP32 Parity Analysis

**Date**: December 8, 2025
**Author**: David Sanftenberg

## Summary

Comprehensive layer-by-layer analysis of Q8_1 activation precision vs FP32 ground truth using the new snapshot comparison framework.

## Key Findings

### 1. Divergence Pattern

The Q8_1 path shows significant divergence from FP32 starting at the **ATTENTION_CONTEXT** stage:

| Stage | Cosine Similarity | Status |
|-------|-------------------|--------|
| EMBEDDING | 0.999987 | ✓ OK |
| layer0_Q_PROJECTION | 0.999963 | ✓ OK |
| layer0_K_PROJECTION | 0.999969 | ✓ OK |
| layer0_V_PROJECTION | 0.999877 | ✓ OK |
| layer0_Q_ROPE | 0.998117 | ⚠️ WARN |
| **layer0_ATTENTION_CONTEXT** | **0.859776** | ❌ DIVERGED |
| layer0_ATTENTION_OUTPUT | 0.943706 | ❌ DIVERGED |
| ... | ... | ... |
| layer22_ATTENTION_NORM | -0.086 | ❌ SEVERE |
| FINAL_NORM | 0.602773 | ❌ DIVERGED |
| LM_HEAD | 0.158841 | ❌ DIVERGED |

### 2. Root Cause Analysis

The divergence is caused by the **KV cache forcing FP32 dequantization**:

```
Flow in Q8_1 Pipeline:
1. Q/K/V projections → Q8_1 tensors (via execute_to_q8_1)
2. K/V dequantized → FP32 → stored in KV cache
3. For attention: Q=Q8_1, K/V=FP32 (from cache)
4. GQAAttention detects mixed precision → falls back to FP32 path
5. Q is dequantized to FP32 for attention computation
```

This results in **double quantization noise**:
- Q8_1 → FP32 dequantization for K/V (stored in cache)
- Q8_1 → FP32 dequantization for Q (in attention)

The Q8_1 format has ~8-bit precision (~0.4% error per quantization). With double dequantization and accumulated noise through 24 layers, the error compounds significantly.

### 3. Why K/V Use FP32 Cache

The KV cache stores FP32 for good reasons:
- **Accuracy**: Maintaining full precision for long-context attention
- **Reuse**: Cached K/V are used for all subsequent tokens
- **Standard practice**: Most inference frameworks use FP32/BF16 KV cache

The issue is that Q8_1 activations are incompatible with FP32 KV cache in the current implementation.

### 4. Q/K/V Projection Accuracy

The Q/K/V projections themselves show excellent accuracy (>0.9999 cosine similarity), indicating:
- The Q4_0 weight dequantization is accurate
- The GEMM computation is correct
- The Q8_1 requantization introduces minimal error (~0.01%)

The problem is not in the projection but in the **attention computation path**.

## Test Output

New test added: `Test__Q8_1_vs_FP32_Parity.LayerByLayerSnapshotComparison`

```cpp
// tests/v2/integration/Test__Q8_1_vs_FP32_Parity.cpp
TEST_F(Test__Q8_1_vs_FP32_Parity, LayerByLayerSnapshotComparison)
{
    // Enables snapshot capture on both FP32 and Q8_1 pipelines
    // Compares intermediate activations at each stage
    // Identifies first divergence point
}
```

Run with:
```bash
mpirun -np 2 ./build_v2/tests/v2/v2_integration_q8_1_vs_fp32_parity \
    --gtest_filter="*LayerByLayer*"
```

## Potential Solutions

### Option 1: Q8_1 KV Cache (High Effort)
Store KV cache in Q8_1 format:
- Modify `KVCache` to support Q8_1Tensor storage
- Use Q8_1-native attention kernel when Q/K/V are all Q8_1
- Trade-off: Increased complexity, potential accuracy issues for long sequences

### Option 2: Hybrid Precision Path (Medium Effort)
Keep FP32 KV cache, but dequantize Q once:
- Q is dequantized to FP32 before attention
- K/V remain in FP32 (from cache)
- Pure FP32 attention computation
- Trade-off: No memory bandwidth savings for attention

### Option 3: Accept Current Accuracy (Low Effort)
Document that Q8_1 activation precision has higher error:
- ~15% error at layer 0 attention
- Acceptable for many use cases where token prediction still matches
- Trade-off: May produce different outputs than FP32 for some prompts

### Option 4: Investigate Attention Kernel (Medium Effort)
Current finding suggests attention might have bugs:
- 0.9999 Q projection accuracy shouldn't lead to 0.86 attention accuracy
- May be precision loss in score computation (Q*K^T)
- Worth investigating the attention kernel implementation

## Files Changed

- `tests/v2/integration/Test__Q8_1_vs_FP32_Parity.cpp`: Added `LayerByLayerSnapshotComparison` test
- Added helper functions: `computeCosineSimilarity`, `computeMaxAbsDiff`, `computeMeanAbsDiff`

## Next Steps

1. **Investigate attention kernel**: The 0.9999 → 0.86 drop is suspicious - may indicate a bug
2. **Consider Q8_1 KV cache**: If attention is correct, implement Q8_1 cache for true Q8_1 path
3. **Profile memory bandwidth**: Measure actual savings from Q8_1 vs FP32 activations
4. **Test with larger models**: Q8_1 may be more suitable for memory-bound models (7B+)

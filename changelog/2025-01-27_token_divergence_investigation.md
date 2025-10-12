# Token Divergence Investigation - Prefill Logits Mismatch

**Date**: 2025-01-27 (October 11, 2025 actual)  
**Author**: David Sanftenberg  
**Status**: 🔍 In Progress - Divergence Identified in Prefill Phase

## Summary

Discovered that Llaminar and PyTorch generate completely different token sequences. The divergence starts in the **prefill phase**, not during decode, indicating a fundamental computation error in Llaminar's forward pass.

## Token Divergence Details

### Observed Behavior
- **PyTorch Generated Tokens**: `[6, 25010, 10]`
- **Llaminar Generated Tokens**: `[400, 1, 66]`
- **Divergence Point**: First token after prefill already differs

### Prefill Configuration
```
Model: qwen2.5-0.5b-instruct-q4_0.gguf
Prefill tokens: [1, 2, 3, 4, 5]
Decode tokens: 3 (total 8 tokens to validate)
```

## Root Cause Analysis

### Prefill Logits Comparison

**PyTorch Prefill** (after processing [1,2,3,4,5]):
```
Top 5 tokens: [6, 9, 1, 4, 527]
Selected token: 6 (argmax)
```

**Llaminar Prefill** (after processing [1,2,3,4,5]):
```
Top 5 tokens: [400, 151643, 81, 1080, 3]
Selected token: 400 (argmax)
```

**Key Finding**: The top-5 tokens don't overlap AT ALL. This indicates **significant numerical divergence** in the logits, not just a small precision difference.

## Investigation Plan

### Phase 1: Logits Comparison ✅ DONE
- [x] Run PyTorch snapshot generation  
- [x] Run Llaminar inference
- [x] Compare top-k tokens from prefill logits
- [x] Result: Complete divergence detected

### Phase 2: Layer-by-Layer Comparison 🔄 IN PROGRESS
Working backwards from logits to find first divergence point:

1. **LM Head (Final Projection)**
   - Input: Last layer hidden states [5, 896]
   - Output: Logits [5, 151669]
   - Status: Need to compare

2. **Layer 23 (Final Transformer Layer)**
   - Attention output
   - FFN output
   - Residual connections
   - Status: Not yet compared

3. **Layer 0 (First Transformer Layer)**
   - Q/K/V projections
   - Attention scores
   - Attention output
   - FFN projections
   - Status: Not yet compared

4. **Embedding Layer**
   - Input tokens: [1, 2, 3, 4, 5]
   - Output shape: [5, 896]
   - Status: Not yet compared

### Phase 3: Suspect Areas

Based on recent changes and typical sources of divergence:

#### Highest Priority Suspects
1. **Weight Slicing**  
   - Recently implemented MPI weight slicing
   - Contracts validate correctly, but slicing logic could have bugs
   - Check: Are sliced weights correctly assembled across ranks?

2. **RoPE Application**
   - Known to be sensitive to implementation details
   - Check: Position indices, cos/sin buffer generation
   - Check: RoPE fusion with attention computation

3. **Attention Computation**
   - MPI distribution of attention
   - Check: Softmax normalization across ranks
   - Check: AllReduce operations

4. **RMSNorm**
   - Variance computation across distributed tensors
   - Check: Reduction operations
   - Check: Epsilon handling

#### Medium Priority Suspects
5. **Quantization/Dequantization**
   - Q4_0 quantized weights
   - Check: Dequant kernel correctness
   - Note: Weight contracts validate, so likely not the issue

6. **Tensor Orientation**
   - Row-major vs column-major
   - Transpose operations
   - Check: Consistent orientation across operations

7. **OpenMP Threading**
   - Race conditions in parallel reductions
   - Check: Critical sections, atomic operations

## Debugging Approach

### Step 1: Save Full Activation Tensors
Modified `test_parity_framework.cpp` to save Llaminar's prefill logits:
```cpp
// Save full prefill logits for debugging
std::string prefill_logits_file = llaminar_output_dir + "/prefill_logits.npy";
NpyArray prefill_logits_array;
prefill_logits_array.shape = {static_cast<size_t>(vocab_size)};
prefill_logits_array.data.assign(last_row_logits, last_row_logits + vocab_size);
if (NpzLoader::write_npy(prefill_logits_file, prefill_logits_array.data, prefill_logits_array.shape))
{
    std::cout << "[TRUE_INCR] Saved prefill logits to: " << prefill_logits_file << std::endl;
}
```

### Step 2: Compare Numerically
Create Python script to:
1. Load Llaminar prefill logits (llaminar_incremental_snapshots/prefill_logits.npy)
2. Load PyTorch prefill logits (generate from PyTorch model)
3. Compute differences:
   - Max absolute difference
   - Relative L2 difference
   - Top-k token overlap
4. Visualize difference distribution

### Step 3: Binary Search Through Layers
If logits differ significantly:
1. Compare last layer hidden states
2. If match → bug is in LM head projection
3. If differ → compare layer 22, then 11, then 5... (binary search)
4. Narrow down to specific layer
5. Within that layer, compare:
   - Attention norm input
   - Q/K/V projections
   - Attention scores
   - Attention output
   - FFN norm input
   - FFN gate/up projections
   - FFN output

### Step 4: Targeted Fix
Once divergence source is identified:
1. Review relevant kernel implementation
2. Compare with PyTorch reference
3. Add targeted unit test
4. Fix and verify

## Technical Details

### Sampling Logic (Verified Correct)
Both PyTorch and Llaminar use greedy sampling (argmax):
```python
# PyTorch
next_token = int(logits[0, -1, :].argmax())

# Llaminar (C++)
int next_token = std::distance(last_row_logits, 
                               std::max_element(last_row_logits, 
                                               last_row_logits + vocab_size));
```

This is correct - the issue is not in sampling but in the logits themselves.

### Weight Validation (Passing)
Weight contracts validate successfully on both MPI ranks:
```
[TRUE_INCR] ✓ Weight verification passed
```

This suggests weights are loaded correctly, ruling out simple weight loading bugs.

## Next Steps

1. **Immediate**: Wait for test to complete, save Llaminar prefill logits
2. **Next**: Create comparison script to analyze logits difference
3. **Then**: If logits diverge significantly, start layer-by-layer comparison
4. **Finally**: Fix identified bug and re-run parity test

## Files Modified

- `tests/test_parity_framework.cpp`: Added prefill logits saving
- `python/reference/generate_incremental_decode_snapshots.py`: Fixed corrupted header

## Related Issues

- Weight slicing contracts: Implemented and validated
- PyTorch snapshot generation: Fixed and working
- Token sequence comparison: Now working correctly

The infrastructure is now in place to debug the numerical divergence effectively.


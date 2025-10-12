# Embedding Weight Validation Analysis
**Date**: January 27, 2025  
**Investigation**: Token divergence root cause

## Summary

Token divergence between Llaminar and PyTorch has been traced to the **embedding layer**. The divergence starts immediately after embedding lookup, indicating either:
1. Embedding weights are loaded incorrectly from GGUF
2. Embedding table lookup logic is incorrect
3. Weight slicing/distribution across MPI ranks is wrong

## Evidence from Test Run

### 1. Embedding Configuration (from logs)

```
[22:11:06.789] DEBUG MPIEmbeddingKernel validate: full_table_mode_=1 transposed_=0 table_shape=[151669, 896]
[22:11:06.789] DEBUG MPIEmbeddingKernel execute: full_table_mode=1 transposed=0 shape=[151669, 896] seq_len=5
```

**Key Facts**:
- `full_table_mode=1` means **each MPI rank has the FULL embedding table**
- Not using distributed/sliced embeddings
- Table shape: [151669 vocab × 896 hidden_dim]
- Input sequence: 5 tokens `[1, 2, 3, 4, 5]`

### 2. Weight Loading (from logs)

```
[22:11:03.162] INFO [TRANSPOSE_SKIP] Embedding tensor 'token_embd.weight' dimensions already correct: [151669x896]
[22:11:03.468] INFO Loaded tensor 'token_embd.weight' elements=135895424 first=-0.00982666
[22:11:03.512] INFO [WeightLoad] token_embd.weight shape=151669x896
```

**Observations**:
- Embedding weights loaded from GGUF: 151669 × 896 = 135,895,424 elements
- First element value: -0.00982666
- No transpose needed (dimensions already correct)
- **No weight contract validation logged**

### 3. Embedding Output Divergence

From `compare_embeddings.py`:

```
Token 0 Embedding:
  PyTorch:  shape (1, 1, 896), min=-0.062, max=0.049, mean=-0.000413, std=0.016
  Llaminar: shape (1, 896),    min=-0.043, max=0.038, mean=0.000449,  std=0.015
  
Numerical Comparison:
  Max abs diff: 0.074951
  Mean abs diff: 0.016040
  Relative L2 norm: 1.245663 (124.57%)
  
✗ SIGNIFICANT DIVERGENCE (rel L2 = 124.57%)
```

**This 124% error is MASSIVE** - not a precision issue but fundamentally wrong computation.

### 4. RMSNorm Input Variance (embedding outputs)

From logs after embedding:
```
Rank 0, Row 0: sum_sq=0.239598, variance=0.000267
Rank 0, Row 1: sum_sq=0.206806, variance=0.000231
Rank 0, Row 2: sum_sq=0.217236, variance=0.000242

Rank 1, Row 0: sum_sq=0.197272, variance=0.000220
Rank 1, Row 1: sum_sq=0.188000, variance=0.000210
```

Different statistics between ranks suggest **different embedding outputs per rank** despite `full_table_mode=1`.

## Critical Questions

### Q1: Are embedding weights identical across ranks?

**Status**: ❓ UNKNOWN - need to verify

**Test**: Compare first few bytes of `token_embd.weight` tensor on rank 0 vs rank 1

### Q2: Is embedding table lookup correct?

**Status**: ❓ UNKNOWN

**Test**: 
- Manually verify embedding for token 1: `embedding_table[1, :]`
- Compare against PyTorch reference
- Check if token IDs are being interpreted correctly

### Q3: Are weight slicing contracts being validated?

**Status**: ❌ NO - no validation logs found

**Evidence**: Searched for "contract", "slice.*valid", "weight.*check" in logs - nothing found

**Recommendation**: Enable weight contract validation during model loading

### Q4: Does `full_table_mode` work correctly?

**Status**: ❓ UNKNOWN

**Code Location**: `src/kernels/MPIEmbeddingKernel.h` line 85:
```cpp
bool full_table_mode_ = false; // True if embedding table contains full vocab on each rank
```

**Test**: 
- Verify both ranks have identical embedding tables
- Check if gather/scatter logic is being skipped correctly in full table mode

## Recommendations

### Immediate Actions (Priority 1)

1. **Add embedding weight validation** to test:
   ```cpp
   // In test_parity_framework.cpp, after model load:
   auto emb_rank0 = pipeline->getEmbeddingWeight();  // On rank 0
   auto emb_rank1 = pipeline->getEmbeddingWeight();  // On rank 1
   MPI_Bcast(emb_rank0->data(), size, MPI_FLOAT, 0, MPI_COMM_WORLD);
   // Compare emb_rank0 == emb_rank1
   ```

2. **Enable weight contract logging**:
   ```cpp
   // In qwen_pipeline_adapter.cpp or model_loader.cpp:
   LOG_INFO("Validating embedding weight contract...");
   auto contracts = getQwenWeightContracts();
   bool valid = contracts.global.at("token_embd.weight").validate(embedding_tensor);
   LOG_INFO("Embedding weight contract: " << (valid ? "PASS" : "FAIL"));
   ```

3. **Verify token lookup logic**:
   ```cpp
   // Add debug logging in MPIEmbeddingKernel::execute():
   LOG_DEBUG("Looking up tokens: [" << input_ids[0] << "," << input_ids[1] << ",...] on rank " << rank);
   LOG_DEBUG("Embedding for token " << input_ids[0] << ": [" << output[0] << "," << output[1] << "," << output[2] << ",...]");
   ```

4. **Compare raw weights**:
   - Extract `token_embd.weight` from GGUF file
   - Load same weights from PyTorch model
   - Compare byte-by-byte for tokens [1, 2, 3, 4, 5]

### Follow-up Actions (Priority 2)

1. Check if recent weight slicing changes affected embedding loading
2. Verify tensor transpose logic for embedding weights
3. Test with distributed embedding mode (full_table_mode=0) to isolate issue
4. Add embedding weight checksum to model loading

## Hypothesis

**Most Likely**: Embedding weights are being loaded incorrectly from GGUF file.

**Why**: 
- Divergence starts at first computation (embedding)
- 124% error is too large for quantization or precision
- Different embedding statistics per rank despite `full_table_mode=1`
- No weight contract validation

**Next Step**: Compare raw embedding weights from GGUF vs expected values.

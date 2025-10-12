# Token Sampling Bug Analysis

## Date
2025-10-11

## Problem Statement
PyTorch and Llaminar generate completely different tokens from identical inputs:
- **Input**: Prefill tokens `[1, 2, 3, 4, 5]`
- **PyTorch Output**: `[6, 25010, 10]`  
- **Llaminar Output**: `[13, 101779, 67538]`

## Root Cause Investigation

### 1. Token Processing Flow - FIXED ✅

**Initial Bug:** PyTorch was processing prefill tokens **incrementally** (one at a time), while Llaminar processed them as a **batch**.

- **PyTorch (broken)**: Process [1], then [2], then [3], then [4], then [5] incrementally
- **Llaminar**: Process [1,2,3,4,5] as a single batch

**Impact:** Different attention masking → Different hidden states

**Fix:** Modified `incremental_decode_with_prefill()` to process all prefill tokens as a batch:
```python
# BEFORE: Incremental loop
for token_id in prefill_tokens:
    capture_stages([token_id], past_key_values)

# AFTER: Batch processing  
prefill_snapshots = capturer.capture_stages(prefill_tokens, past_key_values=None)
```

### 2. Re-processing Last Token - FIXED ✅

**Initial Bug:** PyTorch re-processed the last prefill token to get initial logits.

```python
# WRONG: Re-processes token 5
last_prefill_token = prefill_tokens[-1]
initial_snapshots = capturer.capture_stages([last_prefill_token], past_key_values)
logits = initial_snapshots.get('LM_HEAD')
```

**Impact:** Token 5 appears twice in KV cache → Wrong model state

**Fix:** Use logits from prefill output directly:
```python
# CORRECT: Use saved logits from batch prefill
prefill_snapshots = capturer.capture_stages(prefill_tokens, past_key_values=None)
logits = prefill_snapshots.get('LM_HEAD')
```

### 3. Logits Index Bug - FIXED ✅

**Bug:** Sampling from wrong position in logits tensor.

**PyTorch logits shape after batch prefill:** `[1, 5, 151669]` (batch=1, seq_len=5, vocab)

```python
# WRONG: Samples from position 0 (token 1's prediction)
next_token = int(logits[0, 0, :].argmax())

# CORRECT: Samples from position -1 (token 5's prediction)
if logits.shape[1] > 1:
    next_token = int(logits[0, -1, :].argmax())  # Batch: use last position
else:
    next_token = int(logits[0, 0, :].argmax())   # Incremental: use only position
```

## Current Status

### After All Fixes

**PyTorch:**
- Top 5 tokens: `[6, 9, 43239, 59, 26]`
- Selected: `6`

**Llaminar:**
- Top 5 tokens: `[13, 3, 6953, 16, 11]`
- Selected: `13`

### Critical Finding

The two systems produce **completely different logit distributions** despite:
- ✅ Same model file (`models/Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf`)
- ✅ Same input tokens (`[1, 2, 3, 4, 5]`)
- ✅ Same processing mode (batch prefill)
- ✅ Same sampling position (last token)

## Hypothesis: Weight Loading Mismatch

### PyTorch Path
```
GGUF File → GGUFLoader (python/reference/loaders/gguf_loader.py)
         → Extract state_dict + config
         → Create transformers.Qwen2ForCausalLM
         → Load weights into HuggingFace model
```

### Llaminar Path
```
GGUF File → ModelLoader (src/model_loader.cpp)
         → Direct GGUF tensor loading
         → QwenPipeline execution
```

### Potential Issues

1. **Quantization Dequant:**
   - PyTorch's GGUF loader may dequantize weights differently
   - Llaminar uses native Q4_0, Q6_K dequantization
   
2. **Weight Mapping:**
   - HuggingFace Qwen2 layer names vs GGUF tensor names
   - Possible transpose/reshape differences
   
3. **Layer Configuration:**
   - RoPE parameters
   - Attention head splits
   - Layer normalization epsilon

4. **Numerical Precision:**
   - PyTorch: float32 by default
   - Llaminar: May use different precision in kernels

## Debug Strategy

### Immediate Next Steps

1. **Compare Embedding Output**
   ```python
   # Check if embedding layer matches
   pytorch_emb = model.embed_tokens(torch.tensor([[1,2,3,4,5]]))
   llaminar_emb = load_llaminar_embedding([1,2,3,4,5])
   assert np.allclose(pytorch_emb, llaminar_emb, rtol=1e-4)
   ```

2. **Compare First Layer Weights**
   ```python
   # Check Q projection weights for layer 0
   pytorch_q_weight = model.layers[0].self_attn.q_proj.weight
   llaminar_q_weight = load_llaminar_weight("q_proj_layer0")
   assert np.allclose(pytorch_q_weight, llaminar_q_weight)
   ```

3. **Trace Layer-by-Layer**
   - Add verbose logging to both systems
   - Compare hidden states after each layer
   - Find where divergence begins

### Long-Term Solution

**Option A**: Make PyTorch use same quantization as Llaminar
- Modify GGUFLoader to preserve quantization
- Use quantized inference in transformers

**Option B**: Make Llaminar export FP32 for parity testing
- Add FP32 dequantization mode
- Compare against FP32 PyTorch model

**Option C**: Use identical weight source
- Both load from same HuggingFace checkpoint (not GGUF)
- Quantize in Llaminar, keep FP32 in PyTorch for comparison

## Testing Fixes

### Bugs Fixed (Verified)
1. ✅ Batch vs incremental prefill
2. ✅ Re-processing last token
3. ✅ Logits index for sampling

### Bugs Remaining
1. ❌ **Weight loading mismatch** (suspected root cause)
2. ❌ Numerical divergence in forward pass
3. ❌ All stage-level comparisons failing

## Recommendations

1. **Immediate**: Create minimal embedding test
   - Load models
   - Process single token [1]
   - Compare embedding output
   - If embeddings differ → Weight loading issue confirmed

2. **Short-term**: Add weight validation tool
   - Script to compare GGUF weights vs HuggingFace weights
   - Check for transpose/reshape differences
   - Validate quantization round-trip

3. **Long-term**: Unified test framework
   - Both systems use same FP32 reference
   - Separate quantization testing from parity testing
   - Add intermediate layer validation

## Files Modified in This Session

1. `python/reference/generate_incremental_decode_snapshots.py`
   - Fixed batch prefill processing
   - Removed re-processing of last token
   - Fixed logits indexing for sampling
   
2. `tests/test_parity_framework.cpp`
   - Added debug output for logits shape
   - Added top-5 token display

## Conclusion

We've fixed **all token sampling bugs** in the test infrastructure:
- ✅ Prefill processing (batch vs incremental)
- ✅ Token re-processing
- ✅ Logits indexing

However, the systems still generate different tokens because they're computing **different logits** from the same inputs. This indicates a **fundamental model loading or numerical computation difference**, not a test infrastructure bug.

**Next priority**: Validate weight loading and identify where numerical divergence begins.

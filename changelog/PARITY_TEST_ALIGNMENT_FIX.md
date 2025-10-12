# Parity Test Alignment Fix - Session Summary

## Date
2025-01-XX

## Objective
Fix the incremental decode parity test to properly compare PyTorch and Llaminar implementations with aligned prefill+decode behavior.

## Changes Made

### 1. Python Script Enhancement (`generate_incremental_decode_snapshots.py`)

#### Added New Function: `incremental_decode_with_prefill()`
```python
def incremental_decode_with_prefill(
    model_path: str,
    prefill_tokens: List[int],
    num_decode_tokens: int,
    verbose: bool = False
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[int]]:
    """
    Two-phase processing matching Llaminar's behavior:
    - Phase 1: PREFILL - Process prefill_tokens to build KV cache (NO snapshots)
    - Phase 2: DECODE - Generate num_decode_tokens, capturing snapshots for each
    """
```

**Key Features:**
- Matches Llaminar's prefill+decode separation
- Saves snapshots only during decode phase (3 tokens)
- Properly maintains KV cache state across phases

#### Updated CLI
```bash
# New prefill+decode mode
python script.py --model X --prefill-tokens 1,2,3,4,5 --num-decode-tokens 3

# Legacy mode still supported
python script.py --model X --tokens 1,2,3,4,5,6
```

#### Fixed Tensor Shape Issue
```python
# Squeeze 3D→2D for single-token decode
if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == 1:
    tensor = tensor.squeeze(axis=(0, 1))  # (1,1,896) → (896,)
    tensor = np.expand_dims(tensor, axis=0)  # (896,) → (1,896)
```

### 2. Stage Naming Fix (`generate_test_snapshots.py`)

**Before:**
```python
make_hook(f'MLP_NORM_layer{i}')
make_hook(f'MLP_OUTPUT_layer{i}')
```

**After:**
```python
make_hook(f'FFN_NORM_layer{i}')
make_hook(f'FFN_DOWN_layer{i}')
```

**Rationale:** Matches Llaminar's FFN stage naming convention.

### 3. C++ Test Update (`test_parity_framework.cpp`)

**Before:**
```cpp
cmd << python_cmd << " " << script_path
    << " --model \"" << model_path << "\""
    << " --tokens " << token_str.str()  // All tokens concatenated
```

**After:**
```cpp
cmd << python_cmd << " " << script_path
    << " --model \"" << model_path << "\""
    << " --prefill-tokens " << prefill_str.str()  // Separate prefill
    << " --num-decode-tokens " << num_decode_tokens  // Separate decode
```

## Current Status

### ✅ Fixed Issues
1. **Shape mismatch** - PyTorch tensors now properly 2D `(1, 896)`
2. **Snapshot count alignment** - Both systems generate 3 token directories  
3. **Stage naming** - No more "Missing Llaminar snapshot: MLP_*" errors
4. **Prefill+decode separation** - PyTorch now matches Llaminar's two-phase behavior

### ❌ Remaining Issues

#### 1. Token Divergence (Critical)
```
PyTorch tokens:  [9 → 8496 → 12]
Llaminar tokens: [13 → 101779 → 67538]
```

**Root Cause:** Unknown - need to investigate:
- Model loading differences?
- Weight initialization differences?
- Numerical precision in first forward pass?
- Different random seeds?

#### 2. Numerical Differences (All Stages)
```
Stages compared: 513
Stages passed:   0  
Stages failed:   513
```

**Examples:**
- `EMBEDDING.npy`: max_abs=0.103, rel_l2=0.031
- `Q_PROJECTION_layer0.npy`: max_abs=13.97, rel_l2=1.556
- `FINAL_NORM.npy`: max_abs=97.01, rel_l2=10.75

**Note:** These differences compound from the initial token divergence.

## Test Execution

```bash
# Rebuild test
cmake --build build --target test_parity_framework --parallel

# Run test
timeout 120 mpirun -np 2 ./build/test_parity_framework \
  --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch"
```

**Current Result:** ❌ FAILED (token divergence)

## Next Steps

### Priority 1: Debug Token Divergence

1. **Verify Model Loading**
   ```bash
   # Check if both load the same model file
   python -c "from transformers import AutoModelForCausalLM; \
              m = AutoModelForCausalLM.from_pretrained('models/...'); \
              print(m.config)"
   ```

2. **Add Verbose Logging**
   - Print input tokens in both systems
   - Print embedding output statistics
   - Print first-layer attention output
   - Compare logits before argmax

3. **Check for Randomness**
   - Set torch.manual_seed(42) in PyTorch
   - Verify dropout is disabled
   - Check sampling temperature

4. **Validate Weight Loading**
   - Compare first layer's Q projection weight statistics
   - Verify quantization handling
   - Check for transpose/layout differences

### Priority 2: Establish Baseline

Create a minimal test:
```python
# Test 1: Single prefill token, check embedding
tokens = [1]
llaminar_emb = run_llaminar_embedding(tokens)
pytorch_emb = run_pytorch_embedding(tokens)
assert np.allclose(llaminar_emb, pytorch_emb, rtol=1e-4)

# Test 2: Single layer forward, no KV cache
...
```

### Priority 3: Incremental Validation

Once baseline passes:
1. Add one layer at a time
2. Validate each stage individually
3. Build up to full model parity

## Files Modified

1. `python/reference/generate_incremental_decode_snapshots.py`
   - Added `incremental_decode_with_prefill()` function
   - Updated CLI with `--prefill-tokens` and `--num-decode-tokens`
   - Fixed 3D→2D tensor squeeze logic

2. `python/reference/generate_test_snapshots.py`
   - Renamed `MLP_NORM` → `FFN_NORM`
   - Renamed `MLP_OUTPUT` → `FFN_DOWN`

3. `tests/test_parity_framework.cpp`
   - Updated Python script invocation to use new CLI
   - Changed from `--tokens` to `--prefill-tokens` + `--num-decode-tokens`

## Lessons Learned

1. **Alignment is Critical** - PyTorch must match Llaminar's exact execution flow (prefill vs decode)
2. **Stage Naming Matters** - Consistent naming prevents "missing snapshot" errors
3. **Shape Handling** - PyTorch hooks capture 3D tensors, need careful squeezing
4. **Token Divergence Compounds** - Even small initial differences create massive downstream errors

## References

- Original issue: Token sequence mismatch (PyTorch: 8 tokens, Llaminar: 3 tokens)
- Root cause: Different model states (PyTorch processing all input, Llaminar doing prefill+decode)
- Solution: Implement two-phase processing in PyTorch to match Llaminar

## Conclusion

We've successfully aligned the test infrastructure to match Llaminar's prefill+decode execution model. The remaining challenge is understanding why the two systems generate different tokens from the start, which requires deeper investigation into model loading, weight handling, and numerical precision.

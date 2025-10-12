# Parity Test Analysis: Shape Fix + Token Sequence Issue

**Date**: October 11, 2025  
**Investigation**: True Incremental Decode Parity Test

## Problem 1: Shape Mismatch ✅ FIXED

### Root Cause
PyTorch was saving 3D arrays `(1, 1, features)` but C++ comparison code expected 2D arrays `(1, features)`.

### Fix Applied
Modified `python/reference/generate_incremental_decode_snapshots.py`:
```python
# Squeeze tensor to 2D if it's 3D with batch=1 and seq_len=1
if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == 1:
    tensor = tensor.squeeze(axis=(0, 1))  # Remove batch and seq_len dimensions
    tensor = np.expand_dims(tensor, axis=0)  # Expand back to (1, features)
```

### Verification
Before: `(1, 1, 896)` → Shape shown as `1x1` in C++  
After:  `(1, 896)`    → Shape shown correctly as `1x896`

## Problem 2: Token Sequence Mismatch ⚠️ NEEDS FIX

### Current Behavior

**Test Configuration:**
- Prefill tokens: `[1, 2, 3, 4, 5]` (5 tokens)
- Decode tokens: 3 additional tokens
- Total input sequence: `[1, 2, 3, 4, 5, 6, 7, 8]`

**PyTorch Processing:**
- Processes ALL 8 input tokens incrementally
- Generates 8 predictions (one per input token): `[62162, 997, 5, 5, 6, 25010, 1784, 93]`
- Saves snapshots for all 8 tokens: `token_0/` through `token_7/`

**Llaminar Processing:**
- Prefill with `[1, 2, 3, 4, 5]` (no snapshots saved)
- Decode 3 tokens: generates `[6, 13956, 99822]`
- Saves snapshots for only 3 tokens: `token_0/`, `token_1/`, `token_2/`

### The Mismatch

| Aspect | PyTorch | Llaminar |
|--------|---------|----------|
| Tokens processed | 8 (`[1,2,3,4,5,6,7,8]`) | 3 (decode only) |
| Snapshots saved | 8 directories | 3 directories |
| Predictions | 8 tokens | 3 tokens |
| What it's predicting | Next token after each input | Next token during decode |

### Why They Don't Match

1. **PyTorch** is treating the entire sequence `[1,2,3,4,5,6,7,8]` as **input** and predicting what comes **after** each token
2. **Llaminar** is using `[1,2,3,4,5]` as prefill (context) and then **generating** 3 new tokens

These are fundamentally different tasks!

### Correct Interpretation

The "sampled tokens" are the model's **predictions** of what should come next:

**PyTorch:**
- Input token 1 → Predicts: 62162
- Input token 2 → Predicts: 997
- Input token 3 → Predicts: 5
- ... etc

**Llaminar:**
- After prefill [1,2,3,4,5] → Generates: 6
- After token 6 → Generates: 13956
- After token 13956 → Generates: 99822

### The Fix Needed

**Option A: Match PyTorch to Llaminar's behavior**
- PyTorch should:
  1. Process prefill tokens `[1,2,3,4,5]` without saving snapshots
  2. Then generate 3 new tokens, saving snapshots for each generated token
  
**Option B: Match Llaminar to PyTorch's behavior**  
- Llaminar should:
  1. Process all 8 input tokens one-by-one
  2. Save snapshots for all 8
  3. Compare predictions for all 8

**Option C: Hybrid approach (RECOMMENDED)**
- Test BOTH behaviors:
  1. **Prefill parity**: Compare prefill stage (all 5 tokens processed at once)
  2. **Incremental decode parity**: Compare decode stage (3 tokens generated one-by-one)
  
For now, **Option A** is simpler and matches the "true incremental decode" goal.

## Problem 3: Stage Naming Mismatch ⚠️ NEEDS FIX

### Issue
- PyTorch saves: `MLP_NORM_layerN.npy`, `MLP_OUTPUT_layerN.npy`
- Llaminar saves: `FFN_NORM_layerN.npy`, `FFN_SWIGLU_layerN.npy`, etc.

### Impact
- C++ comparison code reports "Missing Llaminar snapshot" for MLP_* stages
- Real snapshots exist but have different names (FFN_*)

### Fix Options
1. **Rename in PyTorch** - change hooks to use FFN_* naming
2. **Rename in Llaminar** - use MLP_* naming (less preferred, FFN is more accurate)
3. **Add name mapping** - translate MLP → FFN during comparison

**Recommended**: Fix #1 - Update PyTorch hooks to use FFN_* naming to match Llaminar.

## Next Steps

### Immediate (Priority 1)
1. ✅ Fix shape issue (DONE)
2. ⏳ Fix token sequence logic - align PyTorch and Llaminar behavior
3. ⏳ Fix stage naming - use consistent FFN_* names

### After Fixes (Priority 2)
4. Re-run test and expect:
   - ✅ Same number of tokens processed
   - ✅ Matching snapshot directories
   - ✅ Matching stage names
   - ✅ Comparable numerical values

### Validation (Priority 3)
5. Document expected vs actual behavior
6. Add test case documentation explaining what's being compared
7. Consider adding separate tests for:
   - Prefill parity (batch processing)
   - Decode parity (incremental generation)

## Current Status

| Issue | Status | Fix |
|-------|--------|-----|
| Shape mismatch | ✅ FIXED | Squeeze 3D → 2D arrays |
| Token count mismatch | 🔧 IN PROGRESS | Align processing logic |
| Stage naming | ⏳ TODO | Standardize on FFN_* |
| Numerical comparison | ⏸️ BLOCKED | Needs above fixes first |

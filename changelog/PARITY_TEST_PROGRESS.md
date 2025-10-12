# Parity Test Progress Report

**Date**: October 11, 2025  
**Test Run**: After shape fix  
**Status**: Partial success, need alignment fix

## ✅ Successes

1. **Shape fix working**: Most stages now have matching shapes (1×896, 1×128, etc.)
2. **Test infrastructure working**: Comparison code runs successfully
3. **Dual-level validation working**: Token comparison + stage comparison both execute

## ❌ Remaining Issues

### Issue 1: Token Index Mismatch

**Problem**: `token_0` means different things to PyTorch vs Llaminar

**PyTorch `token_0`:**
- Snapshots after processing input token `1` (first token of sequence)
- KV cache length: 0 (empty)
- Context: Just this one token

**Llaminar `token_0`:**
- Snapshots after first **decode** step  
- KV cache length: 5 (has prefill tokens [1,2,3,4,5])
- Context: Prefill + first generated token
- Generated token: `6`

**They're comparing different model states!**

### Issue 2: Sequence Length Mismatches

Some stages still have mismatched shapes:
```
ATTENTION_SOFTMAX_layer14: PyTorch=14x1 vs Llaminar=14x6
```

This is because:
- PyTorch is processing 1 new token (seq_len=1)
- Llaminar has 6 tokens in context (5 prefill + 1 decode, so attention over 6 positions)

### Issue 3: Massive Numerical Differences

Examples:
```
FINAL_NORM: max_abs=203.768, rel_l2=11.0009
FFN_NORM_layer6: max_abs=42.2663, rel_l2=1.93822
Q_PROJECTION_layer8: max_abs=24.4912, rel_l2=1.61695
```

These aren't precision errors - these are **completely different values**. This confirms they're in totally different model states.

## 🎯 The Core Problem

**We're comparing snapshots from different points in the inference process:**

| System | What token_0 represents |
|--------|------------------------|
| PyTorch | After processing input token #1 (no KV cache) |
| Llaminar | After decode step #1 (with 5-token prefill in KV cache) |

**This is why:**
- Shapes don't match (different seq_lens)
- Values are completely different (different contexts)
- Token predictions don't align

## 🔧 Solutions

### Option A: Make PyTorch Match Llaminar (RECOMMENDED)

**Change PyTorch script to:**
1. Run prefill with tokens [1,2,3,4,5] → DON'T save snapshots
2. Generate token 6 → Save as `token_0/`
3. Generate token 7 → Save as `token_1/`
4. Generate token 8 → Save as `token_2/`

**Benefits:**
- Aligns with "true incremental decode" test name
- Both systems have same KV cache state
- Direct apples-to-apples comparison

**Implementation:**
```python
# In Python script:
def incremental_decode_with_prefill(prefill_tokens, num_decode_tokens):
    # Phase 1: Prefill (don't save)
    past_key_values = None
    for token in prefill_tokens:
        outputs = model(token, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
    
    # Phase 2: Decode (save snapshots)
    for i in range(num_decode_tokens):
        # Generate next token
        next_token = sample(outputs.logits)
        # Capture snapshots
        snapshots[f'token_{i}'] = capture_stages(next_token, past_key_values)
        # Update KV cache
        outputs = model(next_token, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
```

### Option B: Make Llaminar Match PyTorch

Make Llaminar process all 8 tokens individually and save snapshots for each.

**Problems:**
- Doesn't test real incremental decode behavior
- Misses the prefill optimization
- Not aligned with test name

### Option C: Test Both Modes Separately

1. **Test 1**: Prefill parity (process [1,2,3,4,5] as batch)
2. **Test 2**: Decode parity (generate 3 tokens with KV cache)

**Problems:**
- More complex
- Current test infrastructure not set up for this

## 📋 Recommended Action Plan

**Step 1:** Modify PyTorch script to support prefill+decode mode
- Add `--prefill-tokens` and `--decode-tokens` arguments
- Process prefill silently (no snapshots)
- Save snapshots only for decode tokens

**Step 2:** Update C++ test to pass prefill info to Python
- Already passing token sequence, just need to split it properly

**Step 3:** Re-run test
- Expect: same KV cache states, matching shapes, comparable values

**Step 4:** Fix remaining naming issues (MLP vs FFN)

## Current Test Command

The test currently does:
```bash
python generate_incremental_decode_snapshots.py \
  --model model.gguf \
  --tokens 1,2,3,4,5,6,7,8  # Processes ALL as input
```

Should do:
```bash
python generate_incremental_decode_snapshots.py \
  --model model.gguf \
  --prefill-tokens 1,2,3,4,5 \
  --num-decode-tokens 3  # Generates 3 new tokens
```

## Next Steps

1. ✅ Shape fix (DONE)
2. 🔧 Add prefill/decode split to Python script (IN PROGRESS)
3. ⏳ Update C++ test to use new Python API
4. ⏳ Fix stage naming (MLP → FFN)
5. ⏳ Re-test and validate numerical parity

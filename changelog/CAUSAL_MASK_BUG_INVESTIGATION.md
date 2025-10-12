# ROOT CAUSE FOUND: Causal Mask Bug in Incremental Decode

## Date
2025-10-11

## The Bug

**File**: `src/kernels/common/attention_primitives.cpp`
**Function**: `compute_qk_scores()`
**Line**: 281

```cpp
if (causal && j > i)
{
    score_row[j] = -std::numeric_limits<float>::infinity();
    continue;
}
```

## Problem

The causal mask compares:
- `i`: Query position **relative to current batch** (0 for incremental decode)
- `j`: Key position **relative to KV cache** (0 to cache_len-1)

These are in different coordinate systems!

## Example (Incremental Decode, token_0)

Prefill: 5 tokens (positions 0-4)
Decode: 1 token (position 5)

Cache has 6 positions total (0-5).

Current query `i=0` (relative) is actually at absolute position 5.

The buggy code does:
```
j=0: 0 > 0? NO  → compute score ✓  (CORRECT: can attend to pos 0)
j=1: 1 > 0? YES → MASK ❌          (WRONG: should attend to pos 1)
j=2: 2 > 0? YES → MASK ❌          (WRONG: should attend to pos 2)
j=3: 3 > 0? YES → MASK ❌          (WRONG: should attend to pos 3)
j=4: 4 > 0? YES → MASK ❌          (WRONG: should attend to pos 4)
j=5: 5 > 0? YES → MASK ❌          (WRONG: should attend to pos 5)
```

Result: Only position 0 is unmasked, all others masked to -inf!

This explains perfectly why we see:
```
ATTENTION_SCORES: [89.53, 0, 0, 0, 0, 0]  # Only first position computed
ATTENTION_SOFTMAX: [1.0, 0, 0, 0, 0, 0]   # One-hot after softmax
```

Wait, the scores show position 5 is non-zero, not position 0. Let me re-check...

Actually looking at the data again:
```
ATTENTION_SCORES: [0.0, 0.0, 0.0, 0.0, 0.0, 89.53]
```

Position 5 (last) is non-zero, positions 0-4 are zero.

Let me reconsider... If query is at absolute position 5:
```
j=0: 0 > 0? NO  → compute ✓
j=1: 1 > 0? YES → MASK ❌
j=2: 2 > 0? YES → MASK ❌
j=3: 3 > 0? YES → MASK ❌
j=4: 4 > 0? YES → MASK ❌
j=5: 5 > 0? YES → MASK ❌
```

But this would make position 0 non-zero, not position 5.

Hmm, let me check if there's row/column transposition or different memory layout...

## Actually - Need to Check K Layout

The Q and K might be laid out differently. Let me check the actual tensor layout.

Q: `[i, h, d]` where i=query_pos
K: `[j, h, d]` where j=key_pos

The indexing in the code:
```cpp
const float *qi = q + (size_t)i * heads * head_dim + (size_t)h * head_dim;
const float *kj = k + (size_t)j * heads * head_dim + (size_t)h * head_dim;
```

So Q is `[q_seq_len, heads, head_dim]`
And K is `[k_seq_len, heads, head_dim]`

For incremental decode:
- Q shape: [1, 14, 64]
- K shape: [6, 14, 64] (from cache)

When i=0, j=0: comparing Q[0] with K[0]
When i=0, j=5: comparing Q[0] with K[5]

And we're seeing scores[0, 0] through scores[0, 5] where scores[0, 5] is non-zero.

So the bug is that j=1,2,3,4,5 are all being masked because `j > i` (where i=0).

Only j=0 should survive... but we're seeing j=5 survive instead.

## Wait - Check if scores are reversed

Let me check if the scores array indexing is row-major or if there's a transpose somewhere.

Actually, looking at line 273:
```cpp
float *score_row = scores + (size_t)h * q_seq_len * k_seq_len + (size_t)i * k_seq_len;
```

This gives us the row for head h, query position i.
Then `score_row[j]` is the score for key position j.

So `score_row[0]` should be score for j=0, `score_row[5]` should be score for j=5.

But we observe:
```
scores[0, :] = [0, 0, 0, 0, 0, 89.53]
```

This means scores[0, 5] is non-zero, which is j=5.

With the causal mask `j > i` where i=0:
- j=5: 5 > 0 is TRUE, so should be MASKED to -inf
- j=0: 0 > 0 is FALSE, so should be COMPUTED

But we're seeing the opposite! j=5 is computed, j=0 is masked.

**The mask logic must be inverted!**

Looking again at line 281:
```cpp
if (causal && j > i)
{
    score_row[j] = -std::numeric_limits<float>::infinity();
    continue;
}
```

This says: if j > i, MASK it. For i=0, this masks j=1,2,3,4,5 and computes j=0.

But we're seeing j=5 computed and j=0-4 masked!

So either:
1. The condition should be `j < some_threshold` instead of `j > i`
2. There's a memory layout issue
3. The snapshot is captured incorrectly

Let me check if there's any transposition in the snapshot capture...

---

**Status**: Found causal mask bug in `compute_qk_scores()`, but need to verify exact memory layout to confirm the fix.

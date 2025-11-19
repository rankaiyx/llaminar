# Batch Padding Debug Plan Using Snapshot Infrastructure

**Date**: 2025-01-24  
**Status**: 📋 **PLANNED** - Strategy for debugging remaining batch test failures

## Problem

Equal-length batch tests **PASS** (max_abs_diff = 0.0), but padded sequences **FAIL**:

| Test | Status | Notes |
|------|--------|-------|
| MultiSequenceBatchEqualLength | ✅ **PASS** | 2 sequences, length=2, no padding |
| MultiSequenceBatch | ❌ FAIL | Unequal lengths → padding required |
| BatchScaling | ❌ FAIL | Scaling + padding |
| ComprehensiveBatchParity | ❌ FAIL | Seq0: ✅ (0.0), Seq1: ❌ (~18) |

## Hypothesis

Padding tokens likely need special position handling:
- **Current behavior**: Position IDs may be contiguous across padding boundary
- **Expected**: Padding positions should either be masked or use special values
- **Evidence**: ComprehensiveBatchParity shows Seq0 perfect, Seq1 fails → asymmetric padding effect

## Available Debug Infrastructure

### V2 Snapshot System

**Built-in to PipelineBase** (`src/v2/pipelines/PipelineBase.h`):
```cpp
// Enable snapshot capture (only in builds with ENABLE_PIPELINE_SNAPSHOTS)
void enableSnapshotCapture(const std::string &output_dir = "");

// Retrieve captured snapshots
const float *getSnapshot(const std::string &key, size_t &out_size) const;
std::vector<std::string> getSnapshotKeys() const;
```

**Already Instrumented in Qwen2Pipeline** (`src/v2/pipelines/qwen/Qwen2Pipeline.cpp`):
- ✅ `EMBEDDING` (line 339)
- ✅ `ATTENTION_NORM` (line 466)
- ✅ `Q_PROJECTION` (line 483)
- ✅ `K_PROJECTION` (line 494)
- ✅ `V_PROJECTION` (line 505)
- ✅ `Q_ROPE` (line 567) ⬅️ **Critical for position debugging**
- ✅ `K_ROPE` (line 568) ⬅️ **Critical for position debugging**
- ✅ `ATTENTION_CONTEXT` (line 597)
- ✅ `ATTENTION_OUTPUT` (line 609)
- ✅ `ATTENTION_RESIDUAL` (line 627)
- ✅ `FFN_NORM` (line 678)
- ✅ `FFN_GATE` (line 693)
- ✅ `FFN_UP` (line 704)
- ✅ `FFN_SWIGLU` (line 720)
- ✅ `FINAL_NORM` (line 373)
- ✅ `LM_HEAD` (not captured yet)

### Python Reference Generators

**Attention Snapshots** (`python/reference/generate_attention_snapshots.py`):
- Supports batch inputs: `[batch, seq_len, ...]` tensors
- Captures Q/K/V projections, RoPE outputs, attention scores
- Uses PyTorch's RoPE implementation as ground truth
- **Limitation**: Currently uses `torch.arange(q_len)` for positions (contiguous)

**Pipeline Snapshots** (`python/reference/generate_qwen2_pipeline_snapshots.py`):
- Full pipeline capture with layer-by-layer snapshots
- Supports custom prompts and token sequences
- Uses GGUF loader to ensure weight parity

## Debug Strategy

### Phase 1: Enable Snapshot Capture in Failing Test

Modify `Test__Qwen2E2ECorrectness.cpp` to enable snapshots for `MultiSequenceBatch`:

```cpp
TEST(Qwen2E2ECorrectness, MultiSequenceBatch) {
    // Enable snapshot capture for debug
    pipeline->enableSnapshotCapture("llaminar_batch_snapshots");
    
    // Run batched inference (currently fails)
    // ...
    
    // Examine captured Q_ROPE, K_ROPE to check position handling
    size_t size;
    const float *q_rope = pipeline->getSnapshot("layer0_Q_ROPE", size);
    // Dump for analysis...
}
```

### Phase 2: Generate PyTorch Reference with Batched Positions

Modify `generate_attention_snapshots.py` to support batched position IDs:

```python
# Current (contiguous):
position_ids = torch.arange(q_len).unsqueeze(0)  # [1, q_len]

# New (batched with padding):
def create_batch_position_ids(sequences):
    """
    Args:
        sequences: List of (tokens, padding_length) tuples
    Returns:
        position_ids: [batch, max_len] with per-sequence positions
    """
    batch_size = len(sequences)
    max_len = max(len(tokens) + pad for tokens, pad in sequences)
    position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, (tokens, pad_len) in enumerate(sequences):
        seq_len = len(tokens)
        position_ids[i, :seq_len] = torch.arange(seq_len)
        # Padding positions: options to test
        # Option A: Continue sequence [0,1,2,3,...] (likely wrong)
        # Option B: Freeze at last valid position [0,1,1,1,...] (maybe?)
        # Option C: Use special value like -1 (masked in attention)
    
    return position_ids
```

### Phase 3: Compare Llaminar vs PyTorch Snapshots

Key comparison points:

1. **Position ID Generation**
   - Compare `position_ids` arrays for batched sequences
   - Check how padding affects position assignment

2. **Q_ROPE / K_ROPE Outputs**
   - Compare RoPE-transformed Q/K for padded sequences
   - Identify where divergence begins (likely at padding boundary)

3. **Attention Scores**
   - Check if padding tokens are properly masked in attention
   - Verify attention weights don't leak from padded positions

### Phase 4: Fix Root Cause

Based on snapshot comparison, likely fixes:

**Option A**: Fix Position ID Generation
```cpp
// src/v2/pipelines/qwen/Qwen2Pipeline.cpp
void Qwen2Pipeline::generate_position_ids(..., const std::vector<int> &sequence_lengths) {
    for (int i = 0; i < batch_size; ++i) {
        int seq_len = sequence_lengths[i];
        for (int j = 0; j < seq_len; ++j) {
            position_ids[i * max_len + j] = j;  // Valid positions
        }
        for (int j = seq_len; j < max_len; ++j) {
            position_ids[i * max_len + j] = seq_len - 1;  // Freeze padding?
        }
    }
}
```

**Option B**: Add Attention Masking for Padding
```cpp
// Ensure attention kernel respects padding mask
// May need to extend CpuAttentionKernelT to accept padding_mask parameter
```

**Option C**: Skip RoPE for Padding Tokens
```cpp
// CPURoPEKernel.cpp - check if position is padding before applying RoPE
if (position_ids[tok] >= 0) {  // Negative = padding sentinel
    apply_rotation(...);
}
```

## Implementation Checklist

- [ ] Add batch position ID generation helper to Python reference
- [ ] Modify `generate_attention_snapshots.py` to support batched inputs
- [ ] Enable snapshot capture in `MultiSequenceBatch` test
- [ ] Generate PyTorch reference with matching padding strategy
- [ ] Compare Q_ROPE/K_ROPE snapshots layer-by-layer
- [ ] Identify exact divergence point (position IDs vs RoPE vs attention)
- [ ] Implement fix based on findings
- [ ] Verify all batch tests pass

## Files to Modify

**C++ (Llaminar)**:
- `tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp` - Enable snapshot capture
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Fix position ID generation (if needed)
- `src/v2/kernels/cpu/CPURoPEKernel.cpp` - Handle padding positions (if needed)

**Python (Reference)**:
- `python/reference/generate_attention_snapshots.py` - Add batch position ID support
- New: `python/reference/test_batch_positions.py` - Validate padding strategies

## Next Steps

1. **Examine Current Position ID Generation**: Look at how `Qwen2Pipeline` generates position_ids for batched sequences
2. **Check PyTorch Padding Strategy**: Review how HuggingFace Qwen2 handles padding in attention
3. **Enable Snapshot Capture**: Modify failing test to dump intermediate states
4. **Generate Reference**: Create PyTorch snapshots with matching batch structure

## References

- **Snapshot Infrastructure**: `src/v2/pipelines/PipelineBase.{h,cpp}`
- **Python Generators**: `python/reference/generate_attention_snapshots.py`
- **V2 Architecture**: `.github/instructions/llaminar-architecture-v2.instructions.md`
- **RoPE Refactor**: `changelog/2025-01-24-rope-batched-inference-refactor.md`

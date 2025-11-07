# INT8 Attention Testing Strategy - PyTorch Parity Approach

**Date**: November 6, 2025  
**Context**: V2 INT8 attention kernel testing  
**Status**: Test framework established, ready for FP32 validation

## Problem Statement

Initially created INT8 attention tests that compared against an **untested FP32 implementation**. This is circular validation - if the FP32 implementation is buggy, INT8 tests will fail even if INT8 is correct.

**Original test results**: 6/9 passing, 3 failing with severe errors:
- `AccuracyVsFP32Reference`: **88% relative L2 error** (expected <5%)
- `CausalMasking`: Causal mask not changing output at all
- `SingleSequence`: 39% error (expected <10%)

## Solution: Layered Validation Strategy

Following V1's proven **Parity Test Framework**, we now use:

1. **PyTorch as ground truth** (known correct implementation)
2. **FP32 attention validated against PyTorch** (establishes correct baseline)
3. **INT8 attention validated against validated FP32** (isolates quantization error)

This creates a proper validation chain:
```
PyTorch (ground truth)
    ↓ parity test
FP32 Attention (validated correct)
    ↓ quantization test
INT8 Attention (test quantization accuracy)
```

## Implementation

### 1. PyTorch Snapshot Generation ✅ COMPLETE

**File**: `python/reference/generate_attention_snapshots.py` (400+ lines)

**Captures**:
- `input.npy`: Attention input (post-norm) [batch, seq_len, d_model]
- `Q_projection.npy`: Query projection [batch, seq_len, n_heads * d_head]
- `K_projection.npy`: Key projection [batch, seq_len, n_kv_heads * d_head]
- `V_projection.npy`: Value projection [batch, seq_len, n_kv_heads * d_head]
- `Q_rope.npy`: Query after RoPE [batch, seq_len, n_heads * d_head]
- `K_rope.npy`: Key after RoPE [batch, seq_len, n_kv_heads * d_head]
- `attention_scores.npy`: Pre-softmax scores [batch, n_heads, seq_len, seq_len]
- `attention_weights.npy`: Post-softmax probabilities [batch, n_heads, seq_len, seq_len]
- `attention_output.npy`: Final output [batch, seq_len, d_model]

**Usage**:
```bash
python3 python/reference/generate_attention_snapshots.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Hello world" \
  --layer 0 \
  --output pytorch_attention_snapshots \
  --verbose
```

**Test Output** (verified working):
```
✓ Model loaded:
  n_layers: 24
  n_heads: 14
  n_kv_heads: 2 (GQA - Grouped Query Attention)
  d_model: 896
  d_head: 64

✓ Captured 9 attention stages:
  input               : shape (1, 2, 896), dtype float32
  Q_projection        : shape (1, 2, 896), dtype float32
  K_projection        : shape (1, 2, 128), dtype float32
  V_projection        : shape (1, 2, 128), dtype float32
  Q_rope              : shape (1, 2, 896), dtype float32
  K_rope              : shape (1, 2, 128), dtype float32
  attention_scores    : shape (1, 14, 2, 2), dtype float32
  attention_weights   : shape (1, 14, 2, 2), dtype float32
  attention_output    : shape (1, 2, 896), dtype float32
```

### 2. C++ Snapshot Loading Infrastructure ✅ READY

**File**: `tests/v2/pytorch_parity/NpzLoader.h` (already exists)

**Key Function**:
```cpp
bool NpzLoader::load_npy(const std::string &filepath, NpyArray &out_array);
```

**NpyArray Structure**:
```cpp
struct NpyArray {
    std::vector<size_t> shape;  // e.g., {1, 2, 896}
    std::vector<float> data;    // Flattened FP32 data
    std::string dtype;          // e.g., "<f4" (little-endian float32)
    
    size_t total_elements() const;
    bool is_valid() const;
};
```

**Usage Example**:
```cpp
#include "tests/v2/pytorch_parity/NpzLoader.h"

NpyArray pytorch_q_proj;
if (!NpzLoader::load_npy("pytorch_attention_snapshots/Q_projection.npy", pytorch_q_proj)) {
    LOG_ERROR("Failed to load PyTorch Q_projection snapshot");
    return false;
}

// pytorch_q_proj.shape = {1, 2, 896}
// pytorch_q_proj.data = std::vector<float> with 1*2*896 = 1792 elements
```

### 3. Next Steps (To Be Implemented)

#### Step 3.1: Implement FP32AttentionKernel ⏳ TODO

**File**: `src/v2/kernels/cpu/FP32AttentionKernel.{h,cpp}`

**Requirements**:
- Same API as INT8AttentionKernel but with FP32 inputs/outputs
- No quantization - pure FP32 computation
- Should match INT8AttentionKernel structure for easy comparison

**API**:
```cpp
class FP32AttentionKernel {
public:
    FP32AttentionKernel(int n_heads, int d_head, int device_idx = -1);
    
    bool forward(
        const float *q,      // [batch, seq_len, n_heads*d_head]
        const float *k,      // [batch, seq_len, n_kv_heads*d_head]
        const float *v,      // [batch, seq_len, n_kv_heads*d_head]
        float *output,       // [batch, seq_len, n_heads*d_head] OUT
        int batch, int seq_len,
        bool use_causal_mask = false, float eps = 1e-8f);
};
```

#### Step 3.2: Test FP32 Attention vs PyTorch ⏳ TODO

**File**: `tests/v2/unit/Test__FP32AttentionParity.cpp`

**Test Strategy**:
1. Load PyTorch snapshots for "Hello world" prompt
2. Run FP32AttentionKernel with same Q/K/V inputs
3. Compare outputs with tight tolerances (rel_l2 < 1e-4)
4. Validate intermediate stages (scores, weights, context)

**Expected Tolerances**:
- **Embedding/Projections**: max_abs < 1e-5, rel_l2 < 1e-4
- **Attention Scores**: max_abs < 1e-3, rel_l2 < 1e-3 (larger tensors)
- **Attention Weights**: max_abs < 1e-4, rel_l2 < 1e-4 (probabilities)
- **Final Output**: max_abs < 1e-4, rel_l2 < 1e-4

#### Step 3.3: Update INT8 Tests to Use Validated FP32 ⏳ TODO

**File**: `tests/v2/unit/Test__INT8AttentionKernel.cpp` (modify existing)

**Changes**:
- Remove `reference_attention_fp32()` helper (buggy)
- Replace with calls to validated `FP32AttentionKernel`
- Update tolerances for quantization error (rel_l2 < 0.05 = 5%)

**Test Flow**:
```cpp
// 1. Create FP32 inputs
std::vector<float> q_fp32, k_fp32, v_fp32;
fill_random_fp32(q_fp32, -1.0f, 1.0f);
// ... same for k, v

// 2. Run validated FP32 attention
FP32AttentionKernel fp32_attn(n_heads, d_head);
std::vector<float> output_fp32_ref(batch * seq_len * d_model);
fp32_attn.forward(q_fp32.data(), k_fp32.data(), v_fp32.data(),
                  output_fp32_ref.data(), batch, seq_len, false);

// 3. Quantize FP32 → INT8
std::vector<int8_t> q_int8, k_int8, v_int8;
std::vector<float> q_scales, k_scales, v_scales;
quantize_fp32_to_int8(q_fp32, q_int8, q_scales, batch * seq_len, d_model);
// ... same for k, v

// 4. Run INT8 attention
INT8AttentionKernel int8_attn(n_heads, d_head);
std::vector<int8_t> output_int8(batch * seq_len * d_model);
std::vector<float> output_scales(batch * seq_len);
int8_attn.forward(q_int8.data(), q_scales.data(), /* ... */);

// 5. Dequantize INT8 → FP32
std::vector<float> output_fp32_from_int8;
dequantize_int8_to_fp32(output_int8, output_scales, output_fp32_from_int8, /* ... */);

// 6. Compare against validated FP32 reference
float rel_error = compute_relative_l2(output_fp32_ref, output_fp32_from_int8);
EXPECT_LT(rel_error, 0.05f) << "INT8 quantization error should be <5%";
```

## Benefits of This Approach

1. **Proper Validation Chain**: PyTorch → FP32 → INT8 (each step validated)
2. **Isolates Bugs**: Can distinguish FP32 implementation bugs from INT8 quantization issues
3. **Leverages Existing Infrastructure**: Reuses V1's proven parity framework
4. **Clear Error Attribution**: Know exactly where problems originate
5. **Prevents Circular Validation**: No longer comparing buggy code against itself

## Architecture Details (Qwen 2.5 0.5B)

From PyTorch snapshot generation:
- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **n_layers**: 24
- **n_heads**: 14 (query heads)
- **n_kv_heads**: 2 (key/value heads - GQA)
- **d_model**: 896 (hidden dimension)
- **d_head**: 64 (per-head dimension)
- **GQA ratio**: 14/2 = 7 (7 query heads share 1 key/value head)

**Memory Shapes**:
- Q projection: [batch, seq_len, 896] (14 heads × 64 dim)
- K projection: [batch, seq_len, 128] (2 kv_heads × 64 dim)
- V projection: [batch, seq_len, 128] (2 kv_heads × 64 dim)
- Attention scores: [batch, 14 heads, seq_len, seq_len]
- Output: [batch, seq_len, 896]

## Current Status

✅ **Completed**:
- PyTorch snapshot generation script (400+ lines, tested)
- NpzLoader infrastructure (already existed)
- INT8AttentionKernel implementation (600 lines)
- INT8 test suite (850 lines, 6/9 passing)

⏳ **Next Tasks** (in priority order):
1. Implement FP32AttentionKernel (~400 lines)
2. Create Test__FP32AttentionParity.cpp (~500 lines)
3. Validate FP32 against PyTorch (should pass easily)
4. Update INT8 tests to use validated FP32 as reference
5. Debug any remaining INT8 issues (now isolated to quantization)

## Key Files

**Python**:
- `python/reference/generate_attention_snapshots.py` - Snapshot generator

**C++ Infrastructure**:
- `tests/v2/pytorch_parity/NpzLoader.h` - .npy file loader

**To Be Created**:
- `src/v2/kernels/cpu/FP32AttentionKernel.{h,cpp}` - FP32 attention
- `tests/v2/unit/Test__FP32AttentionParity.cpp` - FP32 vs PyTorch validation

**Existing**:
- `src/v2/kernels/cpu/INT8AttentionKernel.{h,cpp}` - INT8 attention (600 lines)
- `tests/v2/unit/Test__INT8AttentionKernel.cpp` - INT8 tests (850 lines)

## References

- V1 Parity Framework: `docs/parity-test-framework.instructions.md` (2700+ lines)
- V1 Reference Tests: `tests/v1/TestParityFramework.cpp` (3000+ lines)
- Dynamic Thresholds: `docs/DYNAMIC_VARIANCE_THRESHOLDS.md`

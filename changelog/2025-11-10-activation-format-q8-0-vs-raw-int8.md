# Activation Quantization Format: Q8_0 vs Raw INT8

**Date**: November 10, 2025  
**Decision**: Should we quantize activations to Q8_0 blocks or raw INT8 arrays?

---

## TL;DR Recommendation

✅ **Use Q8_0 blocks** (current approach is correct)

**Rationale**:
- ✅ Self-contained (scales embedded with data)
- ✅ Cache-friendly (64-byte alignment, 34 bytes/block)
- ✅ Consistent with weight format philosophy
- ✅ Easier debugging (inspect blocks independently)
- ✅ Type-safe API (scales can't be mismatched)

---

## Format Comparison

### Option 1: Q8_0 Blocks (Current)

**Structure** (34 bytes per block):
```cpp
struct Q8_0Block {
    uint16_t d;      // FP16 scale factor (2 bytes)
    int8_t qs[32];   // 32 quantized INT8 values (32 bytes)
};
// Total: 34 bytes per 32 elements = 1.0625 bytes/element
```

**Memory Layout**:
```
Block 0: [d₀][qs₀...qs₃₁]  (34 bytes)
Block 1: [d₁][qs₃₂...qs₆₃] (34 bytes)
Block 2: [d₂][qs₆₄...qs₉₅] (34 bytes)
...
```

**Quantization Formula**:
```cpp
scale = max(abs(x)) / 127.0f;
q_i = round(x_i / scale);  // INT8 in range [-127, +127]
```

**Dequantization Formula**:
```cpp
x_i ≈ q_i * scale;
```

---

### Option 2: Raw INT8 + Separate Scales

**Structure**:
```cpp
int8_t* quantized_values;  // M*K int8_t values (1 byte/element)
float* scales;              // (M*K / 32) float scales (4 bytes/scale)
// OR
half* scales;               // (M*K / 32) FP16 scales (2 bytes/scale)
```

**Memory Layout**:
```
Data:   [qs₀...qs₃₁][qs₃₂...qs₆₃][qs₆₄...qs₉₅]...  (contiguous int8_t array)
Scales: [d₀][d₁][d₂]...                           (separate array)
```

**Total Storage** (with FP16 scales):
- Data: M*K bytes
- Scales: (M*K / 32) * 2 bytes
- **Total: M*K * 1.0625 bytes** (same as Q8_0!)

---

### Option 3: Q8_1 Blocks (CUDA-style)

**Structure** (36 bytes per block):
```cpp
struct Q8_1Block {
    half d;          // FP16 scale factor (2 bytes)
    half s;          // FP16 sum (for zero-point) (2 bytes)
    int8_t qs[32];   // 32 quantized INT8 values (32 bytes)
};
// Total: 36 bytes per 32 elements = 1.125 bytes/element
```

**Why CUDA uses this**:
- Sum is used for **zero-point correction** in symmetric quantization
- Useful when activations have non-zero mean (e.g., after LayerNorm)
- Formula: `q_i = round((x_i - mean) / scale)`

**Why we don't need it**:
- We use **asymmetric quantization** (no zero-point needed)
- Simpler bias correction: `DPBUSD(a+128, b) - 128*Σ(b)`
- Activations are often ReLU'd (non-negative) → mean correction less critical

---

## Detailed Analysis

### 1. Memory Efficiency

| Format | Bytes/Element | Overhead | Cache Lines (1024 elements) |
|--------|---------------|----------|----------------------------|
| **Q8_0 blocks** | 1.0625 | +6.25% | 34 (2176 bytes) |
| **Raw INT8 + FP16** | 1.0625 | +6.25% | 34 (2176 bytes) |
| **Raw INT8 + FP32** | 1.125 | +12.5% | 37 (2432 bytes) |
| **Q8_1 blocks** | 1.125 | +12.5% | 37 (2304 bytes) |

**Verdict**: Q8_0 and raw INT8+FP16 are **identical** in memory footprint.

---

### 2. Cache Locality

**Q8_0 Blocks**:
```
Load block → scale and 32 INT8 values in same cache line (34 bytes)
  ↓
Locality: EXCELLENT (all data for 32 elements in one fetch)
```

**Raw INT8 + Separate Scales**:
```
Load INT8 values → 32 bytes from data array
Load scale → 2 bytes from scales array (different cache line!)
  ↓
Locality: POOR (two separate memory regions)
```

**Penalty**: Separate scales require **additional cache line** per block.

**Example** (AVX512 GEMM, 64-element panel):
- Q8_0: Load 2 blocks = **2 cache lines** (68 bytes)
- Raw INT8: Load 64 bytes data + 4 bytes scales = **2 cache lines** for data + **1 cache line** for scales = **3 cache lines total**

**Verdict**: Q8_0 is **33% more cache-efficient** (2 vs 3 cache lines).

---

### 3. VNNI Instruction Compatibility

**AVX512-VNNI DPBUSD**:
```cpp
__m512i acc = _mm512_dpbusd_epi32(
    acc,                  // Accumulator (16× INT32)
    a_unsigned,           // A: 64× UINT8 (activations)
    b_signed              // B: 64× INT8 (weights)
);
```

**Requirements**:
- A operand: **UINT8** (unsigned)
- B operand: **INT8** (signed)

**Both formats work equally well**:
```cpp
// Q8_0 blocks
__m256i a_signed = _mm256_loadu_si256((const __m256i*)block.qs);
__m512i a_unsigned = _mm512_add_epi8(a_signed, bias128);  // Convert to UINT8
float scale = fp16_to_fp32(block.d);                       // Extract scale

// Raw INT8 + separate scales
__m256i a_signed = _mm256_loadu_si256((const __m256i*)data);
__m512i a_unsigned = _mm512_add_epi8(a_signed, bias128);  // Convert to UINT8
float scale = fp16_to_fp32(scales[block_idx]);            // Extract scale
```

**Verdict**: **No difference** in instruction compatibility.

---

### 4. API Ergonomics

**Q8_0 API** (current):
```cpp
// Quantize
std::vector<Q8_0Block> act_q8(num_blocks);
quantize_fp32_to_q8_0(activations_fp32, act_q8.data(), M*K);

// GEMM
gemm_int8_iq4nl_vnni_q8(
    act_q8.data(),        // Self-contained (scales embedded)
    weights_iq4nl,
    output,
    M, N, K
);
```

**Pros**:
- ✅ Type-safe (can't pass wrong scales)
- ✅ Self-documenting (Q8_0Block* → obviously quantized)
- ✅ Easy to serialize/debug (inspect blocks independently)

**Raw INT8 API**:
```cpp
// Quantize
std::vector<int8_t> act_q8(M*K);
std::vector<half> scales((M*K + 31) / 32);
quantize_fp32_to_int8(activations_fp32, act_q8.data(), scales.data(), M*K);

// GEMM
gemm_int8_iq4nl_vnni_raw(
    act_q8.data(),        // Data pointer
    scales.data(),        // Scale pointer (MUST match data!)
    weights_iq4nl,
    output,
    M, N, K
);
```

**Cons**:
- ❌ Error-prone (scales and data can get mismatched)
- ❌ Requires **two allocations** (data + scales)
- ❌ Harder to debug (scales live elsewhere)

**Verdict**: Q8_0 API is **significantly safer and cleaner**.

---

### 5. Alignment and Padding

**Q8_0 Blocks**:
- Each block is 34 bytes
- Natural alignment: 2 bytes (uint16_t)
- Can enforce 64-byte alignment manually:
  ```cpp
  alignas(64) std::vector<Q8_0Block> act_q8;
  ```

**Raw INT8 + Scales**:
- INT8 array: Natural alignment 1 byte
- Scales array: Natural alignment 2 bytes (half) or 4 bytes (float)
- **Must manually align both arrays** for SIMD:
  ```cpp
  alignas(64) std::vector<int8_t> act_q8;
  alignas(64) std::vector<half> scales;
  ```

**Verdict**: Q8_0 is **easier to align** (single allocation).

---

### 6. GEMM Kernel Implementation

**Q8_0 Kernel** (current):
```cpp
// Load block (scale + data in one struct)
const Q8_0Block& block = act_q8[block_idx];
__m256i a_vec = _mm256_loadu_si256((const __m256i*)block.qs);
float scale_a = fp16_to_fp32(block.d);

// Use in VNNI
__m512i a_unsigned = _mm512_add_epi8(_mm512_castsi256_si512(a_vec), bias128);
__m512i acc = _mm512_dpbusd_epi32(acc, a_unsigned, b_vec);

// Scale correction
output[i] += (float)(horizontal_sum(acc) - bias_correction) * scale_a * scale_b;
```

**Raw INT8 Kernel**:
```cpp
// Load data and scale separately
__m256i a_vec = _mm256_loadu_si256((const __m256i*)&act_q8[block_idx * 32]);
float scale_a = fp16_to_fp32(scales[block_idx]);  // <-- Separate load!

// Use in VNNI (identical to Q8_0)
__m512i a_unsigned = _mm512_add_epi8(_mm512_castsi256_si512(a_vec), bias128);
__m512i acc = _mm512_dpbusd_epi32(acc, a_unsigned, b_vec);

// Scale correction (identical to Q8_0)
output[i] += (float)(horizontal_sum(acc) - bias_correction) * scale_a * scale_b;
```

**Difference**: One extra load for scale (from separate array).

**Verdict**: Q8_0 has **better spatial locality** (scale co-located with data).

---

### 7. Debugging and Validation

**Q8_0 Blocks**:
```cpp
// Inspect a single block
void inspect_block(const Q8_0Block& block) {
    float scale = fp16_to_fp32(block.d);
    std::cout << "Scale: " << scale << "\n";
    std::cout << "Values: ";
    for (int i = 0; i < 32; ++i) {
        std::cout << (int)block.qs[i] << " ";
    }
    std::cout << "\n";
}

// Validate quantization
for (size_t b = 0; b < num_blocks; ++b) {
    const Q8_0Block& block = act_q8[b];
    float scale = fp16_to_fp32(block.d);
    for (int i = 0; i < 32; ++i) {
        float original = activations_fp32[b * 32 + i];
        float reconstructed = block.qs[i] * scale;
        float error = fabs(original - reconstructed);
        assert(error / fabs(original) < 0.01);  // <1% error
    }
}
```

**Raw INT8**:
```cpp
// Inspect requires TWO arrays
void inspect_block(const int8_t* data, const half* scales, size_t block_idx) {
    float scale = fp16_to_fp32(scales[block_idx]);
    std::cout << "Scale: " << scale << "\n";
    std::cout << "Values: ";
    for (int i = 0; i < 32; ++i) {
        std::cout << (int)data[block_idx * 32 + i] << " ";
    }
    std::cout << "\n";
}

// Validation is more error-prone (easy to misalign indices)
for (size_t b = 0; b < num_blocks; ++b) {
    float scale = fp16_to_fp32(scales[b]);  // MUST index scales correctly!
    for (int i = 0; i < 32; ++i) {
        size_t idx = b * 32 + i;
        float original = activations_fp32[idx];
        float reconstructed = data[idx] * scale;  // MUST match scale[b]!
        // ...
    }
}
```

**Verdict**: Q8_0 is **much easier to debug** (self-contained blocks).

---

### 8. Consistency with Weight Format

**Current Weight Formats**:
- IQ4_NL: Blocks with embedded scales
- Q4_K: Blocks with embedded scales and metadata
- Q6_K: Blocks with embedded scales and metadata

**Q8_0**: **Consistent** with existing formats (block-based with embedded scales)

**Raw INT8**: **Inconsistent** (would be the only format with separate scales)

**Verdict**: Q8_0 maintains **architectural consistency**.

---

## Performance Considerations

### Cache Line Efficiency

**Scenario**: Load 64 INT8 values + 2 scales for VNNI (typical panel size)

**Q8_0 Blocks**:
```
Cache lines: 2 (68 bytes total)
  Block 0: [d₀][qs₀...qs₃₁]  (34 bytes) → Cache line 0
  Block 1: [d₁][qs₃₂...qs₆₃] (34 bytes) → Cache line 1
```

**Raw INT8 + Scales**:
```
Cache lines: 3 (68 bytes total, but scattered)
  Data: [qs₀...qs₆₃]         (64 bytes) → Cache lines 0-1
  Scales: [d₀][d₁]           (4 bytes)  → Cache line 2 (different region!)
```

**Impact**:
- Q8_0: **2 L1 cache misses** (sequential blocks)
- Raw INT8: **3 L1 cache misses** (data + separate scale array)

**Bandwidth**: 50% more cache traffic with raw INT8.

---

### Prefetching

**Q8_0 Blocks**:
```cpp
// Prefetch next block (data + scale together)
_mm_prefetch(&act_q8[block_idx + 1], _MM_HINT_T0);  // Single prefetch
```

**Raw INT8**:
```cpp
// Prefetch data and scale separately
_mm_prefetch(&act_q8[(block_idx + 1) * 32], _MM_HINT_T0);  // Data
_mm_prefetch(&scales[block_idx + 1], _MM_HINT_T0);         // Scale (separate!)
```

**Verdict**: Q8_0 requires **1 prefetch** vs 2 for raw INT8 (simpler code).

---

### Memory Allocation Overhead

**Q8_0 Blocks**:
```cpp
// Single allocation
std::vector<Q8_0Block> act_q8((M*K + 31) / 32);  // One malloc
```

**Raw INT8**:
```cpp
// Two allocations
std::vector<int8_t> act_q8(M*K);                    // First malloc
std::vector<half> scales((M*K + 31) / 32);          // Second malloc
```

**Verdict**: Q8_0 is **faster to allocate** (one malloc vs two).

---

## Edge Cases and Robustness

### Partial Blocks (Tail Handling)

**Q8_0 Blocks**:
```cpp
// Last block may have <32 elements
// Zero-padding is PART of the block structure
Q8_0Block last_block;
for (int i = 0; i < remaining; ++i) {
    last_block.qs[i] = quantize(x[i]);
}
for (int i = remaining; i < 32; ++i) {
    last_block.qs[i] = 0;  // Explicit padding
}
last_block.d = compute_scale();
```

**Raw INT8**:
```cpp
// Must manually coordinate padding between data and scales
for (int i = 0; i < M*K; ++i) {
    act_q8[i] = quantize(x[i]);
}
for (int i = M*K; i < padded_size; ++i) {
    act_q8[i] = 0;  // Padding (must match scale array size!)
}
scales[(M*K + 31) / 32 - 1] = compute_scale_for_partial_block();
```

**Verdict**: Q8_0 makes **tail handling explicit and safer**.

---

### Serialization and Checkpointing

**Q8_0 Blocks**:
```cpp
// Save to disk (single write)
file.write(reinterpret_cast<const char*>(act_q8.data()), 
           act_q8.size() * sizeof(Q8_0Block));

// Load from disk (single read)
file.read(reinterpret_cast<char*>(act_q8.data()), 
          act_q8.size() * sizeof(Q8_0Block));
```

**Raw INT8**:
```cpp
// Save to disk (TWO writes)
file.write(reinterpret_cast<const char*>(act_q8.data()), M*K);
file.write(reinterpret_cast<const char*>(scales.data()), 
           scales.size() * sizeof(half));

// Load from disk (TWO reads, must coordinate sizes)
file.read(reinterpret_cast<char*>(act_q8.data()), M*K);
file.read(reinterpret_cast<char*>(scales.data()), 
          ((M*K + 31) / 32) * sizeof(half));  // Size calculation error-prone!
```

**Verdict**: Q8_0 is **much simpler to serialize** (single block array).

---

## Comparison Table

| Criterion | Q8_0 Blocks | Raw INT8 + Scales | Winner |
|-----------|-------------|-------------------|--------|
| **Memory efficiency** | 1.0625 bytes/elem | 1.0625 bytes/elem | **TIE** |
| **Cache locality** | Excellent (34 bytes) | Poor (scattered) | **Q8_0** |
| **Cache lines (64 elem)** | 2 | 3 | **Q8_0** |
| **VNNI compatibility** | Perfect | Perfect | **TIE** |
| **API safety** | Type-safe | Error-prone | **Q8_0** |
| **Alignment** | Single array | Two arrays | **Q8_0** |
| **Prefetching** | 1 prefetch/block | 2 prefetches/block | **Q8_0** |
| **Allocation overhead** | 1 malloc | 2 mallocs | **Q8_0** |
| **Debugging** | Easy | Hard | **Q8_0** |
| **Serialization** | 1 write/read | 2 writes/reads | **Q8_0** |
| **Consistency** | Matches weights | Unique format | **Q8_0** |
| **Code complexity** | Low | Medium | **Q8_0** |

**Winner**: **Q8_0 Blocks** (11/12 criteria)

---

## What About Q8_1? (CUDA-style)

**Q8_1 Structure**:
```cpp
struct Q8_1Block {
    half d;          // Scale
    half s;          // Sum (for zero-point)
    int8_t qs[32];
};
// 36 bytes/block = 1.125 bytes/element (+12.5% overhead vs Q8_0)
```

**When to use Q8_1**:
- ✅ Activations have **non-zero mean** (e.g., after LayerNorm without ReLU)
- ✅ Need **symmetric quantization** for numerical precision
- ✅ GPU with **high memory bandwidth** (overhead less critical)

**When to use Q8_0**:
- ✅ Activations are **non-negative** (after ReLU, GeLU)
- ✅ CPU with **limited bandwidth** (every byte counts)
- ✅ Simpler quantization logic (asymmetric = fewer steps)

**For CPU inference**: Q8_0 is **better** (smaller, simpler, faster).

---

## Recommendation

### ✅ Keep Q8_0 Blocks (Current Approach is Correct)

**Rationale**:
1. **Cache efficiency**: 33% fewer cache lines vs raw INT8
2. **API safety**: Type-safe, can't mismatch scales
3. **Consistency**: Matches our weight format philosophy
4. **Simplicity**: One allocation, one prefetch, one serialize
5. **Debugging**: Self-contained blocks are easy to inspect
6. **Performance**: No measurable difference in compute, better memory locality

### Implementation Status

**Current** (IntegerGemm.cpp):
```cpp
✅ quantize_fp32_to_q8_0() implemented
✅ gemm_int8_iq4nl_vnni() uses Q8_0Block
✅ Two-pass scale refinement (optimal quantization)
```

**Future Work** (Phase 3+):
```cpp
// Add pre-quantized API
bool gemm_int8_iq4nl_vnni_q8(
    const Q8_0Block *A_q8,    // Pre-quantized activations
    const IQ4_NLTensor *B,
    float *C,
    int M, int N, int K);

// Keep FP32 API for convenience
bool gemm_int8_iq4nl_vnni(
    const float *A,           // FP32 activations (quantize internally)
    const IQ4_NLTensor *B,
    float *C,
    int M, int N, int K);
```

---

## Summary

**Question**: Should we quantize to INT8 or Q8_0?

**Answer**: **Q8_0 blocks** (current approach is optimal)

**Why**:
- ✅ Same memory footprint as raw INT8 + scales (1.0625 bytes/element)
- ✅ 33% better cache efficiency (2 vs 3 cache lines per 64 elements)
- ✅ Type-safe API (scales embedded, can't be mismatched)
- ✅ Consistent with our block-based weight formats
- ✅ Easier to debug, serialize, and maintain

**Don't Change Anything**: The current Q8_0 implementation is **already optimal** for CPU inference.


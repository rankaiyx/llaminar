# Q16_1 Integer-Domain Attention Kernel v2

**Status**: In Progress (Phase 4 - Per-head Scale Normalization)  
**Created**: 2025-12-30  
**Updated**: 2025-12-30  
**Author**: Llaminar Team  
**Supersedes**: [PROJECT_Q16_INTEGER_ATTENTION.md](./PROJECT_Q16_INTEGER_ATTENTION.md)

## Progress

- [x] Phase 0: Rename failed v1 kernel and microkernels to `.deprecated`
  - [x] `Q16FusedAttentionRef.cpp/h` → `.deprecated.cpp/h`
  - [x] `microkernels/*.cpp/h` → `.deprecated.cpp/h` (5 microkernels)
  - [x] Fixed all internal includes to use `.deprecated.h`
  - [x] Added deprecated sources back to CMakeLists for existing infrastructure
- [x] Phase 1: Scaffold new `Q16IntegerAttentionRef.h/.cpp`
  - [x] Variable block size structs (`Q16_1Block_64`, `Q16_1Block_128`, `Q16_1Block_192`)
  - [x] `Q16IntegerAttentionParams` with head scales
  - [x] Integer Q×K^T dot product (templated)
  - [x] Integer P×V accumulation (templated)
  - [x] Integration with new `Exp2FixedSoftmax` microkernel
  - [x] Fresh `Exp2FixedSoftmax.h/.cpp` with cleaner API
  - [x] 29 unit tests for Exp2FixedSoftmax (17 basic + 12 spiky activation tests, all passing)
- [x] Phase 1.5: Codebase audit for block size dependencies
  - [x] Identified 25 files with 40 functions requiring updates
  - [x] Categorized by priority: P0 (blocking) → P3 (integration)
  - [x] See "Q16_1 Codebase Audit" section below
- [x] Phase 2: Core tensor + RoPE block size support (P0) ✅
  - [x] `BlockStructures.h`: Add `Q16_1Block_64/128/192` with `sum_qs` field ✅
  - [x] `BlockStructures.h`: Add `Q16BlockSize` enum and type traits ✅
  - [x] `BlockStructures.h`: Add `optimal_q16_block_size()` function ✅
  - [x] `Test__Q16BlockStructures.cpp`: 17 unit tests all passing ✅
  - [x] `Q16IntegerAttentionRef.h`: Refactored to use canonical types from BlockStructures.h ✅
  - [x] `Q16_1Tensor`: Add `block_size_` member (default: BLOCK_32 for backward compat) ✅
  - [x] `Q16_1Tensor`: Add constructor with `Q16BlockSize` parameter ✅
  - [x] `Q16_1Tensor`: Update `blocks_per_row()` to use `block_size_` ✅
  - [x] `Q16_1Tensor`: Add `q16_block_size()` accessor ✅
  - [x] `Test__Q16_1Tensor.cpp`: 9 new unit tests for variable block sizes (all passing) ✅
  - [x] `RoPEPrimitives`: Templatize functions on BlockType (scalar, AVX2, AVX512) ✅
  - [x] `Test__Q16_1RoPE.cpp`: 25 tests including variable block sizes (all passing) ✅
- [x] Phase 3: Residual + MPI + KV Cache (P1) ✅
  - [x] `SIMDHelpers.h`: Templatize 4 q16_1 operations (q16_add_q16, q16_add_fp32, q16_add_q8, q16_sum_n) ✅
  - [x] `Test__Q16VariableBlockSIMD.cpp`: 19 tests for variable block SIMD operations (all passing) ✅
  - [x] `MPIContext`: Add `allreduce_q16_inplace<BlockType>` template for variable block sizes ✅
  - [x] `BlockStructures.h`: Add `q16_block_size_bytes()` and `q16_block_size_elements()` helpers ✅
  - [x] `TensorFactory`: Add `createQ16_1(shape, block_size, device_idx)` overload ✅
  - [x] `UnifiedKVCache`: Auto-select optimal block size from head_dim, variable-size copy/shift ✅
  - [x] **Unit Test Coverage Audit** (2025-12-30):
    - [x] `Test__Q16BlockStructures.cpp`: Added 4 tests for helper functions (`q16_block_size_bytes`, `q16_block_size_elements`) ✅
    - [x] `Test__TensorFactory_Q16BlockSize.cpp`: NEW - 11 tests for `createQ16_1(shape, block_size)` overload ✅
    - [x] `Test__UnifiedKVCache.cpp`: Added 7 tests for variable block size auto-selection and operations ✅
    - [x] `Test__Q16MPI_Allreduce.cpp`: NEW - 10 MPI integration tests (2-rank) for all block sizes ✅
    - [x] Fixed 2 pre-existing KV cache tests (`AppendQ16_1_MultipleAppends`, `EvictQ16_1`) ✅
    - [x] Fixed `MPIContext::allreduce_q16_inplace` to correctly loop over blocks for `q16_sum_n` ✅
- [ ] Phase 4: Per-head scale normalization
- [ ] Phase 5: Implement Wo projection (VPDPWSSD)
- [ ] Phase 6: Implement integer residual add
- [ ] Phase 7: FA2 Prefill tiled implementation
- [ ] Phase 8: Unit tests for Q16IntegerAttention
- [ ] Phase 9: E2E parity tests

---

## Lessons Learned from v1 Implementation

The original Q16 integer attention implementation **failed to achieve integer-only computation** despite extensive documentation claiming otherwise. Post-mortem analysis revealed:

### What Went Wrong

| Stage | Design Intent | Actual Implementation | Root Cause |
|-------|---------------|----------------------|------------|
| **Q×K^T** | INT16×INT16→INT32 | **FP32** accumulator | Per-block scales forced `float(block_dot) * q.d * k.d` |
| **Softmax** | Exp2FixedSoftmax LUT | **`std::exp()`** | Score range unpredictable for LUT input mapping |
| **P×V** | INT32 accumulator | INT32 + **FP64 scale tracking** | Per-element V scales required double precision |
| **Wo projection** | VPDPWSSD INT32 | **FP32** accumulator | Reference impl took easy path |
| **Residual add** | Q16_1 + Q16_1 | ✓ Q16_1 | Only stage that worked correctly |

### The Fundamental Problem: Per-Block Scales

Q16_1 format stores one `float d` scale per 32 elements. For `head_dim=64`:
- Q has 2 blocks with scales `q.d[0]`, `q.d[1]`  
- K has 2 blocks with scales `k.d[0]`, `k.d[1]`
- V has 2 blocks with scales `v.d[0]`, `v.d[1]`

Computing Q×K^T requires combining **4 different scale pairs** per score:
```
score = (q_block0 · k_block0) × q.d[0] × k.d[0]
      + (q_block1 · k_block1) × q.d[1] × k.d[1]
```

Without normalization, agents fell back to FP32 to handle arbitrary scale combinations.

### Design Oversight: Fixed 32-Element Block Size

The Q16_1 format inherited Q8_1's 32-element block size without considering:

1. **INT16 has 256× more precision** than INT8 - can handle larger block variance
2. **Common head dimensions** (64, 128, 192) would benefit from aligned block sizes
3. **Attention algorithm** specifically needs per-head (not per-block) scales

This wasn't obvious at design time because Q16_1 was designed for **GEMM**, where per-block scales work fine. The problem only manifests in **attention** where multiple scale combinations compound.

---

## v2 Solution Part A: Model-Aware Block Sizes

### The Key Insight: 64 is the GCD

Common head dimensions across model families:
- **64**: Qwen2.5-0.5B, GPT-2
- **128**: Qwen3, MiniMax-M1, Llama-2/3, Mistral
- **192**: DeepSeek V3, Kimi K2 (MLA: 128 nope + 64 rope)

**64 divides all of them evenly**, making it the universal fallback block size.

### Model Survey

| Model | Architecture | Q/K head_dim | V head_dim | 64-block | 128-block | 192-block |
|-------|-------------|--------------|------------|----------|-----------|-----------|
| Qwen2.5-0.5B | Standard | 64 | 64 | ✅ 1/head | ❌ Broken | ❌ Broken |
| Qwen3-8B | Standard | 128 | 128 | 2/head | ✅ 1/head | ❌ Broken |
| MiniMax-M1 | Standard | 128 | 128 | 2/head | ✅ 1/head | ❌ Broken |
| Llama-3-8B | Standard | 128 | 128 | 2/head | ✅ 1/head | ❌ Broken |
| DeepSeek V3 | MLA | **192** | 128 | 3/head | 1.5 ❌ | ✅ 1/head |
| Kimi K2 | MLA | **192** | 128 | 3/head | 1.5 ❌ | ✅ 1/head |

### Block Size Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                   BLOCK SIZE SELECTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AT QUANTIZATION TIME (model-aware):                            │
│                                                                  │
│    if model.architecture == MLA:                                │
│      // Option A: Semantic split (preserves nope/rope)          │
│      Q/K weights → Q16_64 blocks (3 per head, needs normalize) │
│      V weights   → Q16_128 blocks (1 per head)                 │
│                                                                  │
│      // Option B: Maximum performance (loses nope/rope split)   │
│      Q/K weights → Q16_192 blocks (1 per head) ← OPTIMAL!      │
│      V weights   → Q16_128 blocks (1 per head)                 │
│                                                                  │
│    else if model.head_dim == 64:                                │
│      → Q16_64 blocks (1 per head) ← OPTIMAL, no normalization! │
│                                                                  │
│    else if model.head_dim == 128:                               │
│      → Q16_128 blocks (1 per head) ← OPTIMAL                   │
│                                                                  │
│    else:                                                         │
│      → Q16_64 blocks (universal fallback)                      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AT RUNTIME (attention kernel):                                 │
│                                                                  │
│    if blocks_per_head == 1:                                     │
│      → Pure integer path, no normalization overhead             │
│                                                                  │
│    else:  // 2-3 blocks per head                                │
│      → Per-head normalization (Part B of this plan)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MLA Block Size Tradeoffs

For DeepSeek V3 / Kimi K2 with 192-dim Q/K:

| Approach | Block Config | Blocks/Head | Normalization | Nope/Rope Semantics |
|----------|-------------|-------------|---------------|---------------------|
| **64-block + normalize** | Q16_64 | 3 | Required | Preserved (can split 2+1) |
| **192-block optimal** | Q16_192 | **1** | None | Lost (single scale) |

**Recommendation**: Start with 192-block for simplicity and performance. If precision issues arise due to nope/rope distribution differences, fall back to 64-block with normalization.

### Memory Efficiency Comparison

| Block Size | Overhead | Bytes/Value | Use Case |
|------------|----------|-------------|----------|
| 32 (current) | 8B / 32 = 25% | 2.25 | Legacy GEMM |
| 64 | 8B / 64 = 12.5% | 2.125 | head_dim=64, universal fallback |
| 128 | 8B / 128 = 6.25% | 2.0625 | head_dim=128 (most modern models) |
| 192 | 8B / 192 = 4.2% | **2.04** | MLA Q/K (DeepSeek V3, Kimi K2) |

### New Q16 Format Variants

```cpp
// Existing (keep for GEMM compatibility)
struct Q16_1Block_32 {
    float d;
    int32_t sum_qs;
    int16_t qs[32];
};

// New: 64-element blocks (universal for attention)
struct Q16_1Block_64 {
    float d;
    int32_t sum_qs;
    int16_t qs[64];
};

// New: 128-element blocks (optimal for head_dim=128)
struct Q16_1Block_128 {
    float d;
    int32_t sum_qs;
    int16_t qs[128];
};

// New: 192-element blocks (optimal for MLA Q/K)
struct Q16_1Block_192 {
    float d;
    int32_t sum_qs;
    int16_t qs[192];
};

// Tensor metadata includes block size
struct Q16_1TensorMetadata {
    uint32_t block_size;  // 32, 64, 128, or 192
    uint32_t num_blocks;
    // ... existing fields ...
};
```

### MLA Block Size Options

For DeepSeek V3 / Kimi K2 with MLA architecture, two approaches:

#### Option A: 192-Block (Recommended for Performance)

```cpp
// Single 192-element block per Q/K head - optimal path!
struct MLAHeadLayout_192 {
    Q16_1Block_192 qk[1];     // 192-dim as 1×192 block ← OPTIMAL!
    Q16_1Block_128 v[1];      // 128-dim as 1×128 block ← OPTIMAL!
};

// Both Q/K and V get 1-block-per-head optimal integer path
// No normalization overhead at all!
```

#### Option B: 64-Block Split (Preserves Nope/Rope Semantics)

```cpp
// Split into nope (non-positional) and rope (rotary) parts
struct MLAHeadLayout_Split {
    // Q/K: 192 = 128 (nope) + 64 (rope)
    Q16_1Block_64 qk_nope[2];  // 128-dim as 2×64 blocks
    Q16_1Block_64 qk_rope[1];  // 64-dim as 1×64 block
    
    // V: 128-dim
    Q16_1Block_128 v[1];       // 128-dim as 1×128 block (optimal!)
};

// Attention dot product naturally splits:
// score = dot(Q.nope, K.nope) + dot(Q.rope, K.rope)
//       = (nope_dot × nope_scale) + (rope_dot × rope_scale)
//
// Only 2 scale combinations instead of 3!
```

The nope and rope portions have **different scaling characteristics** anyway (rope values are rotated), so treating them separately is semantically correct.

---

## v2 Solution Part B: Per-Head Scale Normalization

### Core Insight

**Normalize Q, K, V to a single scale per head at pipeline boundaries**, not during attention:

| Tensor | Where Normalized | When |
|--------|------------------|------|
| **Q** | FusedQKVGEMMStage (after projection + bias + RoPE) | Every forward pass |
| **K** | KV Cache write | Once per token |
| **V** | KV Cache write | Once per token |

After normalization:
- All blocks within a head share **one scale factor**
- Integer dot products accumulate without per-block scale tracking
- Scales are combined **once** at attention output, not per-element

### Normalized Q16_1 Format

```cpp
// Standard Q16_1Block (per-32-element scale)
struct Q16_1Block {
    float d;           // Per-block scale
    int32_t sum_qs;    // Sum of quantized values
    int16_t qs[32];    // Quantized values
};

// For attention, we store per-head metadata alongside the tensor
struct Q16_1HeadMetadata {
    float head_scale;      // Max |d| across all blocks in this head
    int num_blocks;        // Blocks per head (head_dim / 32)
};

// Normalized representation (logical, not new struct):
// qs_normalized[i] ≈ qs[i] × (block.d / head_scale)
// Actual value = qs_normalized[i] × head_scale
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Q16_1 INTEGER ATTENTION v2                                │
│                (Per-Head Scale Normalization)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  PREPROCESSING: FusedQKVGEMMStage                                      ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  Input × Wq/Wk/Wv → FP32 Q, K, V                                      ║  │
│  ║  + Bias (if present)                                                   ║  │
│  ║  + RoPE                                                                ║  │
│  ║  ───────────────────────────────────────────────────────────────────  ║  │
│  ║  NORMALIZE Q: For each head h:                                         ║  │
│  ║    1. Quantize to Q16_1 blocks                                        ║  │
│  ║    2. Q_head_scale[h] = max(|block.d|) across head's blocks           ║  │
│  ║    3. Requantize: qs_new = qs × (block.d / Q_head_scale) (INT16)     ║  │
│  ║    4. Set all block.d = Q_head_scale[h]                               ║  │
│  ║  Output: Q_normalized[seq_len × n_heads × head_dim] + Q_head_scales   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                              │                                               │
│                              ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  KV CACHE WRITE (K and V normalization)                                ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  For new K, V tokens being cached:                                     ║  │
│  ║    1. Quantize to Q16_1 blocks                                        ║  │
│  ║    2. K_head_scale = max(|block.d|), V_head_scale = max(|block.d|)    ║  │
│  ║    3. Requantize to unified scale (same process as Q)                 ║  │
│  ║    4. Store normalized K, V in cache with head scales                 ║  │
│  ║                                                                        ║  │
│  ║  Note: Head scales may evolve as new tokens added. Options:            ║  │
│  ║    A. Track running max, renormalize when scale increases >2×         ║  │
│  ║    B. Use fixed conservative scale based on model statistics          ║  │
│  ║    C. Normalize per-token-group (e.g., every 64 tokens)               ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                              │                                               │
│                              ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  Q16 INTEGER ATTENTION KERNEL                                          ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                        ║  │
│  ║  INPUTS (all normalized to per-head scales):                           ║  │
│  ║    Q: INT16[seq_len_q × n_heads × head_dim] + Q_head_scale[n_heads]   ║  │
│  ║    K: INT16[kv_len × n_kv_heads × head_dim] + K_head_scale[n_kv_heads]║  │
│  ║    V: INT16[kv_len × n_kv_heads × head_dim] + V_head_scale[n_kv_heads]║  │
│  ║                                                                        ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 1: Q×K^T → INT32 Scores                                    │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ for each query q in [0, seq_len_q):                              │ ║  │
│  ║  │   for each key k in [0, kv_end):                                 │ ║  │
│  ║  │     int32_t score = 0;                                           │ ║  │
│  ║  │     for (int i = 0; i < head_dim; ++i)                           │ ║  │
│  ║  │       score += Q_int16[q][i] × K_int16[k][i];  // VPDPWSSD      │ ║  │
│  ║  │     scores[k] = score;                                           │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ Combined scale = Q_head_scale × K_head_scale × (1/√head_dim)     │ ║  │
│  ║  │ (Applied ONCE at output, not per-element)                        │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 2: Score Normalization + Exp2FixedSoftmax                  │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Find max for numerical stability (INT32)                      │ ║  │
│  ║  │ int32_t max_score = max(scores[0..kv_end]);                      │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Map to LUT input range: INT32 → INT8 [-128, 0]                │ ║  │
│  ║  │ // score_shift chosen so (max_score - min_score) >> shift ≈ 128  │ ║  │
│  ║  │ int score_shift = compute_score_shift(max_score, min_score);     │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ for each k:                                                       │ ║  │
│  ║  │   int32_t relative = scores[k] - max_score;  // Always ≤ 0       │ ║  │
│  ║  │   int8_t lut_idx = clamp(relative >> score_shift, -128, 0);      │ ║  │
│  ║  │   weights_int16[k] = exp2_lut[lut_idx + 128];  // [0, 32767]    │ ║  │
│  ║  │   weight_sum += weights_int16[k];                                │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 3: P×V → INT32 Context (VPDPWSSD)                          │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ int32_t context[head_dim] = {0};                                 │ ║  │
│  ║  │ for each k in [0, kv_end):                                       │ ║  │
│  ║  │   int16_t w = weights_int16[k];                                  │ ║  │
│  ║  │   for (int d = 0; d < head_dim; ++d)                             │ ║  │
│  ║  │     context[d] += w × V_int16[k][d];  // INT16×INT16→INT32      │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // All V values share V_head_scale - no per-element tracking!    │ ║  │
│  ║  │ Context scale = V_head_scale / weight_sum × (1/32767)            │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 4: Wo Projection (VPDPWSSD)                                │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Requantize INT32 context → INT16 for VPDPWSSD                 │ ║  │
│  ║  │ int16_t ctx_int16[n_heads × head_dim];                           │ ║  │
│  ║  │ float ctx_scale = requantize_int32_to_int16(context, ctx_int16); │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Wo projection: INT16 context × INT8 weights (sign-extended)   │ ║  │
│  ║  │ int32_t proj[d_model] = {0};                                     │ ║  │
│  ║  │ for (int n = 0; n < d_model; ++n)                                │ ║  │
│  ║  │   for (int k = 0; k < input_dim; ++k)                            │ ║  │
│  ║  │     proj[n] += ctx_int16[k] × wo_int16[k][n];  // VPDPWSSD      │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Requantize INT32 → Q16_1                                      │ ║  │
│  ║  │ combined_scale = ctx_scale × wo_block_scales[...]               │ ║  │
│  ║  │ requantize_int32_to_q16_1(proj, combined_scale, output);        │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 5: Residual Add (Q16_1 + Q16_1 → Q16_1)                    │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ simd::q16_1_add_q16_1(residual_in, projection, residual_out);   │ ║  │
│  ║  │ // Already implemented correctly in v1                           │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  OUTPUT: Q16_1[seq_len_q × d_model] (residual stream)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Precision Flow Summary

| Stage | Input Types | Accumulator | Output | FP32 Usage |
|-------|-------------|-------------|--------|------------|
| Block size selection | Model config | - | Block size enum | None |
| Q normalization | Q16_1 (block scales) | - | Q16_1 (head scale) | Scale computation only |
| K/V normalization | Q16_1 (block scales) | - | Q16_1 (head scale) | Scale computation only |
| Q×K^T | INT16 × INT16 | **INT32** | INT32 scores | None |
| Softmax | INT32 scores | INT32 | INT16 weights | None (LUT) |
| P×V | INT16 × INT16 | **INT32** | INT32 context | None |
| Wo projection | INT16 × INT16 | **INT32** | Q16_1 | Scale factor only |
| Residual add | Q16_1 + Q16_1 | - | Q16_1 | None |

**FP32 is used ONLY for scale factors, never for data computation.**

---

## Q16_1 Codebase Audit: Files Requiring Block Size Updates

This section documents ALL locations in the codebase where Q16_1 operations assume the fixed 32-element block size. Each must be updated to support the new variable block sizes (64, 128, 192).

### Legend
- ⬜ Not started
- 🟨 In progress
- ✅ Complete

---

### A. Core Block Structure Definition

| File | Function/Struct | Current State | Action Required |
|------|-----------------|---------------|-----------------|
| [BlockStructures.h](../../src/v2/tensors/BlockStructures.h) | `Q16_1Block` | Fixed 32-element | ⬜ Keep for GEMM, add `Q16_1Block_64/128/192` |

---

### B. Q16_1Tensor Class

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | Constructor (5 overloads) | Uses `Q16_1Block::BLOCK_SIZE` | ⬜ Add block_size parameter |
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | `fp32_data()` | Hardcoded 32-element decode | ⬜ Dispatch on block_size |
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | `copyFrom()` | 32-element block loop | ⬜ Dispatch on block_size |
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | `applyRoPE()` | Uses 32-block kernel | ⬜ Add block_size dispatch |
| [TensorFactory.cpp](../../src/v2/tensors/TensorFactory.cpp) | `createQ16_1()` | Default 32-block | ⬜ Add block_size parameter |

---

### C. RoPE Operations (Q8_1 → Q16_1 and Q16_1 → Q16_1)

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [RoPEPrimitives.h](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.h#L569) | `apply_rope_q8_1_to_q16_1()` | Outputs 32-block Q16_1 | ⬜ Add output block_size param |
| [RoPEPrimitives.h](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.h#L521) | `apply_rope_q16_1_integer()` | In-place 32-block | ⬜ Add block_size param |
| [RoPEPrimitives.h](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.h#L465) | `apply_rope_q16_1_integer_head()` | Per-head RoPE (32-block) | ⬜ Templatize on BlockType |
| [RoPEPrimitives.cpp](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp#L3593) | `apply_rope_q8_1_to_q16_1_head()` | Internal impl (32-block) | ⬜ Templatize on BlockType |
| [CPURoPEKernelT.cpp](../../src/v2/kernels/cpu/ops/CPURoPEKernelT.cpp#L644) | `apply_q8_1_to_q16_1()` | Kernel dispatch (32-block) | ⬜ Add block_size dispatch |
| [CPURoPEKernelT.cpp](../../src/v2/kernels/cpu/ops/CPURoPEKernelT.cpp#L717) | `apply_typed()` (Q16_1) | In-place Q16_1 RoPE | ⬜ Add block_size dispatch |
| [RoPEStage.cpp](../../src/v2/execution/compute_stages/stages/RoPEStage.cpp) | Stage execution | Creates 32-block output | ⬜ Pass block_size to kernel |

---

### D. Residual Add Operations

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2211) | `q16_1_add_fp32()` | `n_blocks = count / 32` | ⬜ Templatize on BlockType |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2614) | `q16_1_add_q8_1()` | `n_blocks = count / 32` | ⬜ Templatize on BlockType |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2024) | `q16_1_add_q16_1()` | 32-element loop | ⬜ Templatize on BlockType |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L1781) | `q16_1_sum_multi()` | MPI reduction helper | ⬜ Templatize on BlockType |

---

### E. MPI Allreduce for Q16_1

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [MPIContext.h](../../src/v2/utils/MPIContext.h#L132) | `allreduce_q16_1_inplace()` | `Q16_1Block*` (32-element) | ⬜ Templatize or add block_size |
| [AllreduceStage.cpp](../../src/v2/execution/compute_stages/stages/AllreduceStage.cpp#L86) | Q16_1 path | Uses `mutable_q16_1_blocks()` | ⬜ Get block_size from tensor |

---

### F. KV Cache (Q16_1 Storage)

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [UnifiedKVCache.h](../../src/v2/tensors/UnifiedKVCache.h#L183) | `UnifiedKVCacheTensor<Q16_1>` | Specialization | ⬜ Add block_size to config |
| [UnifiedKVCache.cpp](../../src/v2/tensors/UnifiedKVCache.cpp#L52) | `allocate_tensor()` | 32-block allocation | ⬜ Pass block_size |
| [UnifiedKVCache.cpp](../../src/v2/tensors/UnifiedKVCache.cpp#L111) | `copy_append_data()` | 32-block copy | ⬜ Dispatch on block_size |
| [UnifiedKVCache.cpp](../../src/v2/tensors/UnifiedKVCache.cpp#L172) | `shift_evict_data()` | 32-block shift | ⬜ Dispatch on block_size |
| [KVCacheAppendStage.cpp](../../src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp#L280) | Hardcoded comment | `block_size = 32` | ⬜ Get from tensor |

---

### G. Quantization/Dequantization

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [QuantizeToQ16_1Stage.h](../../src/v2/execution/compute_stages/stages/QuantizeToQ16_1Stage.h) | FP32→Q16_1 | 32-block output | ⬜ Add block_size param |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2447) | `q16_1_to_q8_1()` | 32-block conversion | ⬜ Templatize on BlockType |
| [WoProjectionVNNIRef.deprecated.cpp](../../src/v2/kernels/cpu/attention/q16_1/ref/microkernels/WoProjectionVNNIRef.deprecated.cpp#L158) | `requantize_fp32_to_q16_1()` | 32-block output | ⬜ Fresh impl in v2 kernel |

---

### H. Attention Kernel Infrastructure

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [JitQ16FusedAttention.h](../../src/v2/kernels/cpu/attention/q16_1/jit/JitQ16FusedAttention.h#L131) | `blocks_per_head()` | `Q16_1Block::BLOCK_SIZE` | ⬜ Use configurable block_size |
| [Q16FusedAttentionKernel.cpp](../../src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp) | `convert_params()` | Assumes 32-block | ⬜ Add block_size dispatch |
| [FusedAttentionWoStage.cpp](../../src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp#L194) | Debug dump loop | `Q16_1Block::BLOCK_SIZE` | ⬜ Get from tensor |
| [Q8_1 JitFusedAttentionWo.h](../../src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h#L6443) | Q16_1 residual write | `Q16_1_BLOCK_SIZE = 32` | ⬜ Get from params |

---

### I. Buffer Size Calculations

| File | Location | Current State | Action Required |
|------|----------|---------------|-----------------|
| [BufferRole.h](../../src/v2/execution/BufferRole.h#L328) | `BLOCK_SIZE = 32` | Hardcoded constant | ⬜ Make configurable |
| [EmbeddingStage.cpp](../../src/v2/execution/compute_stages/stages/EmbeddingStage.cpp#L236) | `block_size = 32` | Hardcoded | ⬜ Get from config |

---

### J. RMSNorm for Q16_1 Input

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [KernelFactory.cpp](../../src/v2/kernels/KernelFactory.cpp#L1469) | `createRMSNorm(Q16_1Tensor*)` | Dequants 32-block | ⬜ Dispatch on block_size |
| [CPURMSNormKernelT](../../src/v2/kernels/cpu/ops/) | `<Q16_1>` specialization | 32-block dequant | ⬜ Add block_size dispatch |

---

### K. HybridQ16 Mode Integration

| File | Component | Current State | Action Required |
|------|-----------|---------------|-----------------|
| [HybridPrecisionConfig.h](../../src/v2/execution/HybridPrecisionConfig.h) | Q16_1 config | No block_size field | ⬜ Add block_size config |
| [Qwen2Graph.cpp](../../src/v2/models/qwen/Qwen2Graph.cpp) | HybridQ16 wiring | Assumes 32-block | ⬜ Pass block_size to stages |
| [GraphOrchestrator.cpp](../../src/v2/execution/GraphOrchestrator.cpp) | Buffer allocation | 32-block buffers | ⬜ Use model-aware sizes |

---

### Summary Statistics

| Category | Files | Functions | Priority |
|----------|-------|-----------|----------|
| **Core Tensor** | 3 | 8 | **P0** (blocking) |
| **RoPE** | 3 | 7 | **P0** (blocking) |
| **Residual Add** | 1 | 4 | **P1** (required) |
| **MPI Allreduce** | 2 | 2 | **P1** (required) |
| **KV Cache** | 2 | 5 | **P1** (required) |
| **Quantize/Dequant** | 3 | 3 | **P1** (required) |
| **Attention Kernel** | 4 | 4 | **P2** (v2 kernel) |
| **Buffer Calcs** | 2 | 2 | **P2** (integration) |
| **RMSNorm** | 2 | 2 | **P2** (integration) |
| **HybridQ16** | 3 | 3 | **P3** (after core) |
| **Total** | **25** | **40** | |

### Recommended Implementation Order

1. **Phase A** (Blocking): Core tensor + RoPE
   - `BlockStructures.h`: Add new block types
   - `Q16_1Tensor`: Add block_size member, update constructors
   - `RoPEPrimitives`: Templatize on BlockType

2. **Phase B** (Required): Residual + MPI + KV Cache
   - `SIMDHelpers.h`: Templatize q16_1 operations
   - `MPIContext`: Add block_size to allreduce
   - `UnifiedKVCache`: Add block_size config

3. **Phase C** (v2 Kernel): Fresh attention implementation
   - `Q16IntegerAttentionRef`: Already uses new block types ✅
   - Kernel uses `Q16_1Block_64/128/192` directly

4. **Phase D** (Integration): Wire it all together
   - `HybridPrecisionConfig`: Add block_size config
   - `Qwen2Graph`: Pass block_size through pipeline
   - `GraphOrchestrator`: Model-aware buffer allocation

---

## Implementation Phases

### Phase 0: Variable Block Size Infrastructure

#### Task 0.1: Define Block Size Enum and Formats
```cpp
// src/v2/tensors/Q16BlockFormats.h

enum class Q16BlockSize : uint32_t {
    BLOCK_32 = 32,    // Legacy, GEMM compatibility
    BLOCK_64 = 64,    // Universal for attention (GCD)
    BLOCK_128 = 128,  // Optimal for head_dim=128
    BLOCK_192 = 192,  // Optimal for MLA Q/K (DeepSeek V3, Kimi K2)
};

struct Q16_1Block_64 {
    float d;
    int32_t sum_qs;
    int16_t qs[64];
};

struct Q16_1Block_128 {
    float d;
    int32_t sum_qs;
    int16_t qs[128];
};

struct Q16_1Block_192 {
    float d;
    int32_t sum_qs;
    int16_t qs[192];
};
```

#### Task 0.2: Model-Aware Block Size Selection
```cpp
// src/v2/loaders/BlockSizeSelector.h

struct AttentionBlockSizes {
    Q16BlockSize qk_block_size;  // For Q and K
    Q16BlockSize v_block_size;   // For V (may differ!)
};

AttentionBlockSizes select_attention_block_sizes(const ModelConfig& cfg) {
    if (cfg.architecture == Architecture::MLA) {
        // DeepSeek V3, Kimi K2: 192-dim Q/K, 128-dim V
        return {
            .qk_block_size = Q16BlockSize::BLOCK_192,  // 1 block/head - optimal!
            .v_block_size = Q16BlockSize::BLOCK_128    // 1 block/head - optimal!
        };
    }
    
    switch (cfg.head_dim) {
        case 64:
            return { Q16BlockSize::BLOCK_64, Q16BlockSize::BLOCK_64 };
        case 128:
            return { Q16BlockSize::BLOCK_128, Q16BlockSize::BLOCK_128 };
        default:
            return { Q16BlockSize::BLOCK_64, Q16BlockSize::BLOCK_64 };
    }
}
```

#### Task 0.3: Update Q16 Tensor to Support Variable Block Size
- [ ] Add `block_size` field to Q16_1Tensor metadata
- [ ] Implement `typed_data<BlockType>()` accessor
- [ ] Update quantization/dequantization for each block size

#### Task 0.4: Update GGUF Loader for New Block Formats
- [ ] Register `Q16_1_64` and `Q16_1_128` type codes
- [ ] Add block size detection from tensor metadata
- [ ] Implement conversion path from 32-block to 64/128-block at load time

### Phase 1: Per-Head Scale Metadata Infrastructure

#### Task 1.1: Add Head Scale Storage
- [ ] Add `head_scales` array to attention kernel params
- [ ] Define storage location (alongside tensor? separate buffer?)
- [ ] Update `Q16FusedAttentionWoResidualParams` struct

#### Task 1.2: Exp2FixedSoftmax Integration Check
- [ ] Verify `Exp2FixedSoftmaxRef` exists and is functional
- [ ] Ensure LUT covers INT8 input range [-128, 0] → INT16 [0, 32767]
- [ ] Add score-shift computation helper

### Phase 2: Q Normalization in FusedQKVGEMMStage

#### Task 2.1: Add Normalization Option to FusedQKVGEMMStage
```cpp
struct FusedQKVGEMMStageParams {
    // ... existing fields ...
    Q16BlockSize q_block_size = Q16BlockSize::BLOCK_64;  // NEW
    bool normalize_q_to_head_scale = false;  // NEW
    float* q_head_scales_out = nullptr;      // NEW: [num_heads]
};
```

#### Task 2.2: Implement Q Normalization Logic
```cpp
// After Q projection + bias + RoPE, before storing to output:
if (params.normalize_q_to_head_scale && params.q_head_scales_out) {
    const int blocks_per_head = head_dim / static_cast<int>(params.q_block_size);
    
    for (int h = 0; h < num_heads; ++h) {
        if (blocks_per_head == 1) {
            // OPTIMAL: Single block per head, no normalization needed!
            params.q_head_scales_out[h] = Q_q16_block[h].d;
        } else {
            // Multiple blocks: normalize to head scale
            float head_scale = compute_head_max_scale(Q_q16, h, head_dim);
            params.q_head_scales_out[h] = head_scale;
            normalize_head_to_scale(Q_q16, h, head_dim, head_scale);
        }
    }
}
```

#### Task 2.3: Unit Tests for Q Normalization
- [ ] Test: normalized blocks all have same scale
- [ ] Test: dequantized values match original within tolerance
- [ ] Test: head_scale equals max of original block scales
- [ ] Test: single-block heads skip normalization (optimal path)

### Phase 3: K/V Normalization in KV Cache

#### Task 3.1: Add Normalization to KVCacheUpdateStage
```cpp
struct KVCacheUpdateParams {
    // ... existing fields ...
    Q16BlockSize k_block_size = Q16BlockSize::BLOCK_64;  // NEW
    Q16BlockSize v_block_size = Q16BlockSize::BLOCK_128; // NEW (often different!)
    bool normalize_to_head_scale = false;     // NEW
    float* k_head_scales = nullptr;           // NEW: [num_kv_heads], running max
    float* v_head_scales = nullptr;           // NEW: [num_kv_heads], running max
};
```

#### Task 3.2: Implement K/V Normalization Strategy

**Option A: Conservative Fixed Scale** (Recommended for v2.0)
```cpp
// Use model-wide statistics to set conservative fixed scales
// Avoids renormalization complexity
constexpr float K_FIXED_SCALE = 2.0f;  // Determined from profiling
constexpr float V_FIXED_SCALE = 2.0f;
```

**Option B: Running Max with Renormalization** (Future)
```cpp
// Track running max, renormalize cache when scale increases significantly
if (new_max > current_head_scale * RENORM_THRESHOLD) {
    renormalize_cache_for_head(cache, layer, head, new_max);
    head_scale = new_max;
}
```

#### Task 3.3: Update KV Cache Tensor Type
- [ ] Ensure KV cache stores normalized INT16 + head scale metadata
- [ ] Update `UnifiedKVCache<Q16_1>` to track head scales
- [ ] Support different block sizes for K vs V (important for MLA!)

#### Task 3.4: Unit Tests for K/V Normalization
- [ ] Test: cache entries have consistent head scale
- [ ] Test: dequantized K/V values match non-normalized baseline
- [ ] Test: fixed scale handles typical value ranges without clipping

### Phase 4: Integer-Only Attention Kernel

#### Task 4.1: Create Q16IntegerAttentionRef (New File)
```
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h
```

**Do NOT modify Q16FusedAttentionRef.cpp** - create fresh implementation.

#### Task 4.2: Implement Stage 1 - Integer Q×K^T
```cpp
// Pure INT16×INT16→INT32, no scale tracking during loop
// Works with any block size as long as inputs are pre-normalized
void compute_qk_scores_int32(
    const int16_t* Q,      // Normalized Q [head_dim]
    const int16_t* K,      // Normalized K [kv_len × head_dim]  
    int kv_len,
    int head_dim,
    int32_t* scores_out)   // [kv_len]
{
    for (int kv = 0; kv < kv_len; ++kv) {
        int32_t score = 0;
        for (int d = 0; d < head_dim; ++d) {
            score += static_cast<int32_t>(Q[d]) * static_cast<int32_t>(K[kv * head_dim + d]);
        }
        scores_out[kv] = score;
    }
}
```

#### Task 4.3: Implement Stage 2 - Exp2FixedSoftmax
```cpp
// INT32 scores → INT16 weights via LUT
void exp2_fixed_softmax_int32_to_int16(
    const int32_t* scores,
    int kv_len,
    int16_t* weights_out,
    int32_t* weight_sum_out)
{
    // Find max score
    int32_t max_score = scores[0];
    for (int i = 1; i < kv_len; ++i)
        max_score = std::max(max_score, scores[i]);
    
    // Compute score range for shift calculation
    int32_t min_score = scores[0];
    for (int i = 1; i < kv_len; ++i)
        min_score = std::min(min_score, scores[i]);
    
    // Adaptive shift: map score range to ~128 LUT entries
    int score_shift = compute_adaptive_shift(max_score - min_score);
    
    // Apply LUT
    int32_t sum = 0;
    for (int i = 0; i < kv_len; ++i) {
        int32_t relative = scores[i] - max_score;  // ≤ 0
        int8_t lut_idx = static_cast<int8_t>(
            std::max(-128, std::min(0, static_cast<int>(relative >> score_shift))));
        weights_out[i] = exp2_lut_256[lut_idx + 128];  // [0, 32767]
        sum += weights_out[i];
    }
    *weight_sum_out = sum;
}
```

#### Task 4.4: Implement Stage 3 - Integer P×V
```cpp
// INT16 weights × INT16 V → INT32 context
void compute_pv_int32(
    const int16_t* weights,  // [kv_len]
    const int16_t* V,        // [kv_len × head_dim]
    int kv_len,
    int head_dim,
    int32_t* context_out)    // [head_dim]
{
    std::memset(context_out, 0, head_dim * sizeof(int32_t));
    for (int kv = 0; kv < kv_len; ++kv) {
        int16_t w = weights[kv];
        if (w == 0) continue;
        for (int d = 0; d < head_dim; ++d) {
            context_out[d] += static_cast<int32_t>(w) * static_cast<int32_t>(V[kv * head_dim + d]);
        }
    }
}
```

#### Task 4.5: Implement Stage 4 - Integer Wo Projection
```cpp
// INT32 context → INT16 → VPDPWSSD → INT32 → Q16_1
void wo_projection_int32(
    const int32_t* context,      // [input_dim]
    const WoPackedWeights* Wo,   // INT8 packed weights
    int input_dim,
    int d_model,
    int32_t weight_sum,
    float v_head_scale,
    float q_head_scale,
    float k_head_scale,
    Q16_1Block* output)
{
    // Step 1: Requantize context INT32 → INT16
    int16_t* ctx_int16 = ...;
    float ctx_scale;
    requantize_int32_to_int16_uniform(context, input_dim, ctx_int16, &ctx_scale);
    
    // Step 2: VPDPWSSD GEMV (INT16 × INT16 → INT32)
    int32_t* proj_int32 = ...;
    for (int n = 0; n < d_model; ++n) {
        int32_t acc = 0;
        for (int k = 0; k < input_dim; ++k) {
            int16_t w = sign_extend_int8(Wo->get(k, n));
            acc += static_cast<int32_t>(ctx_int16[k]) * static_cast<int32_t>(w);
        }
        proj_int32[n] = acc;
    }
    
    // Step 3: Compute combined output scale
    // output = (context_int32 / weight_sum × V_scale) × ctx_scale × Wo_scale
    float combined_scale = ctx_scale * v_head_scale / static_cast<float>(weight_sum)
                         / 32767.0f  // weight normalization
                         * ...;      // Wo block scales
    
    // Step 4: Requantize INT32 → Q16_1
    requantize_int32_to_q16_1(proj_int32, d_model, combined_scale, output);
}
```

#### Task 4.6: Full Kernel Integration
```cpp
bool q16_integer_attention_decode(const Q16IntegerAttentionParams& params) {
    // Validate all inputs are normalized
    LLAMINAR_ASSERT(params.q_head_scale > 0, "Q must be normalized");
    LLAMINAR_ASSERT(params.k_head_scale > 0, "K must be normalized");
    LLAMINAR_ASSERT(params.v_head_scale > 0, "V must be normalized");
    
    for (int h = 0; h < params.num_heads; ++h) {
        // Stage 1: Q×K^T
        compute_qk_scores_int32(Q_head, K_head, kv_len, head_dim, scores);
        
        // Stage 2: Softmax
        exp2_fixed_softmax_int32_to_int16(scores, kv_len, weights, &weight_sum);
        
        // Stage 3: P×V
        compute_pv_int32(weights, V_head, kv_len, head_dim, context);
        
        // Concatenate context for Wo
        concat_context(context, full_context, h, head_dim);
    }
    
    // Stage 4: Wo projection
    wo_projection_int32(full_context, Wo, input_dim, d_model,
                        weight_sum, v_head_scale, q_head_scale, k_head_scale,
                        projection);
    
    // Stage 5: Residual add
    simd::q16_1_add_q16_1(residual_in, projection, residual_out, d_model);
    
    return true;
}
```

### Phase 5: Integration and Testing

#### Task 5.1: Create Q16_INTEGER_V2 Backend
- [ ] Add `Q16_INTEGER_V2` to `FusedAttentionBackend` enum
- [ ] Wire to new `Q16IntegerAttentionRef` kernel
- [ ] Update `FusedAttentionWoStage` dispatch
- [ ] Add block size configuration to backend params

#### Task 5.2: Update GraphOrchestrator for Block Sizes + Normalization
- [ ] Detect model architecture (MLA vs Standard)
- [ ] Select appropriate block sizes based on head dimensions
- [ ] Enable Q normalization when using Q16_INTEGER_V2
- [ ] Pass head scales to attention kernel params
- [ ] Configure KV cache normalization with appropriate block sizes

#### Task 5.3: Parity Tests
- [ ] Create `Test__Q16IntegerAttention_vs_FP32.cpp`
- [ ] Target: ≥0.95 cosine similarity for ATTENTION_CONTEXT
- [ ] Compare against both FP32 baseline and current Q16FusedAttentionRef
- [ ] Test with 64-block and 128-block configurations

#### Task 5.4: Layer-by-Layer Validation
- [ ] Run layer-by-layer comparison similar to existing HybridQ16 test
- [ ] Verify no layer shows catastrophic degradation (>10% cosine drop)
- [ ] Specifically validate layer 22/23 (problem layers in v1)

### Phase 6: MLA Architecture Support (DeepSeek V3, Kimi K2)

#### Task 6.1: MLA-Specific Attention Kernel
```cpp
// MLA has split Q/K: nope (128-dim) + rope (64-dim) = 192 total
struct MLAAttentionParams {
    // Q/K split structure
    const int16_t* Q_nope;      // [n_heads × 128]
    const int16_t* Q_rope;      // [n_heads × 64]
    const int16_t* K_nope;      // [kv_len × n_kv_heads × 128]
    const int16_t* K_rope;      // [kv_len × n_kv_heads × 64]
    float q_nope_scale, q_rope_scale;
    float k_nope_scale, k_rope_scale;
    
    // V is standard 128-dim
    const int16_t* V;           // [kv_len × n_kv_heads × 128]
    float v_head_scale;         // Single scale (128-block optimal!)
    
    // ... rest of params
};

// MLA dot product naturally splits
void compute_mla_qk_scores_int32(const MLAAttentionParams& params, ...) {
    for (int kv = 0; kv < kv_len; ++kv) {
        // Nope dot (2 × 64-blocks → can normalize to 1 scale)
        int32_t nope_score = dot_int32(Q_nope, K_nope[kv], 128);
        
        // Rope dot (1 × 64-block → already optimal!)
        int32_t rope_score = dot_int32(Q_rope, K_rope[kv], 64);
        
        // Combine with only 2 scale pairs (not 3!)
        scores[kv] = nope_score + rope_score;
    }
}
```

#### Task 6.2: MLA KV Cache Layout
- [ ] Store K_nope and K_rope separately in cache
- [ ] K_nope: 128-dim as 2×64-blocks (normalize to head scale)
- [ ] K_rope: 64-dim as 1×64-block (already optimal!)
- [ ] V: 128-dim as 1×128-block (optimal!)

#### Task 6.3: MLA Unit Tests
- [ ] Test: nope/rope split produces same scores as combined
- [ ] Test: MLA-specific normalization works correctly
- [ ] Test: V path achieves 1-block-per-head optimal path

### Phase 7: Performance Optimization (Future)

#### Task 7.1: SIMD Vectorization
- [ ] Vectorize `compute_qk_scores_int32` with VPDPWSSD
- [ ] Vectorize `compute_pv_int32` with VPDPWSSD
- [ ] Vectorize Wo projection
- [ ] Specialized versions for 64-block vs 128-block

#### Task 7.2: JIT Implementation
- [ ] Port scalar reference to Xbyak JIT
- [ ] Reuse existing register guard system
- [ ] Target: 2-3× speedup over scalar
- [ ] Block-size-specialized codegen

---

## Key Differences from v1

| Aspect | v1 (Failed) | v2 (Proposed) |
|--------|-------------|---------------|
| **Block size** | Fixed 32 | Model-aware (64, 128) |
| **Scale handling** | Per-block, tracked at runtime | Per-head, normalized upfront |
| **Optimal path** | Never | 1-block-per-head skips normalization |
| **MLA support** | None | Native nope/rope split |
| **Softmax** | `std::exp()` FP32 | Exp2FixedSoftmax LUT |
| **V scale tracking** | Per-element FP64 | Single head scale |
| **Implementation** | Modified existing file | Fresh implementation |
| **FP32 in compute** | Everywhere | Never (scales only) |
| **Validation** | Post-hoc discovery | Built-in assertions |

---

## Risk Mitigation

### Risk 1: Normalization Precision Loss
**Concern**: Requantizing to head scale may lose precision  
**Mitigation**: 
- Head scale is max of block scales, so ratios are ≤1.0
- INT16 has enough headroom for typical value distributions
- Monitor clipping rate in normalization

### Risk 2: KV Cache Scale Evolution
**Concern**: As tokens accumulate, scale may need updating  
**Mitigation**:
- Start with fixed conservative scale
- Add renormalization support in v2.1 if needed
- Profile actual scale distributions on real prompts

### Risk 3: Softmax LUT Precision
**Concern**: 256-entry LUT may be too coarse  
**Mitigation**:
- Adaptive shift maps actual score range to LUT
- exp2 is well-behaved for softmax (no exp overflow issues)
- Can increase to 512 or 1024 entries if needed

---

## Success Criteria

1. **Precision**: ≥0.95 cosine similarity vs FP32 on ATTENTION_CONTEXT
2. **Integer Purity**: Zero FP32 operations in data path (verified by code review)
3. **No Layer Collapse**: All layers within 5% cosine of average
4. **Documentation Accuracy**: Comments match actual implementation

---

## File Manifest

### New Files
```
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h
tests/v2/integration/Test__Q16IntegerAttention_vs_FP32.cpp
tests/v2/unit/Test__Q16Normalization.cpp
tests/v2/unit/Test__Q16BlockFormats.cpp
```

### Modified Files
```
src/v2/execution/RuntimeConfig.h                  # Add Q16_INTEGER_V2 enum
src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.h/cpp  # Q normalization + block size
src/v2/execution/compute_stages/stages/KVCacheUpdateStage.h/cpp # K/V normalization + block size
src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp # Dispatch
src/v2/tensors/UnifiedKVCache.h/cpp              # Head scale storage + variable block size
src/v2/loaders/GGUFLoader.cpp                     # Block size selection based on model
```

### Unchanged (Reference Only)
```
src/v2/kernels/cpu/attention/q16_1/ref/Q16FusedAttentionRef.cpp  # Keep as cautionary tale
```

---

## Timeline Estimate

| Phase | Tasks | Estimated Effort |
|-------|-------|------------------|
| Phase 0 | Variable Block Size Infrastructure | 3-4 hours |
| Phase 1 | Per-Head Scale Metadata | 2-3 hours |
| Phase 2 | Q Normalization | 3-4 hours |
| Phase 3 | K/V Normalization | 4-5 hours |
| Phase 4 | Integer Attention Kernel | 6-8 hours |
| Phase 5 | Integration & Testing | 4-5 hours |
| Phase 6 | MLA Architecture Support | 4-6 hours |
| Phase 7 | Performance Optimization | 6-8 hours (future) |
| **Total (core)** | Phases 0-5 | **~22-29 hours** |
| **Total (with MLA)** | Phases 0-6 | **~26-35 hours** |

---

## Appendix A: Model Block Size Reference

| Model | Architecture | Q/K head_dim | V head_dim | Q/K Block | V Block | Blocks/Head |
|-------|-------------|--------------|------------|-----------|---------|-------------|
| Qwen2.5-0.5B | Standard | 64 | 64 | 64 | 64 | **1** (optimal) |
| GPT-2 | Standard | 64 | 64 | 64 | 64 | **1** (optimal) |
| Qwen3-8B | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| Llama-3-8B | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| MiniMax-M1 | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| Mistral-7B | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| DeepSeek V3 | MLA | 192 | 128 | **192** | 128 | **1** (optimal) |
| Kimi K2 | MLA | 192 | 128 | **192** | 128 | **1** (optimal) |

**Key insight**: With model-aware block sizes, **all model families get optimal 1-block-per-head path**:
- Standard 64-dim models → 64-block
- Standard 128-dim models → 128-block  
- MLA 192-dim Q/K → 192-block, 128-dim V → 128-block

No normalization needed for any model when using the correct block size!
// For i=0   (input=-128): LUT[0] = ~0 (negligible weight)
static const int16_t exp2_lut_256[256] = {
    0, 0, 0, 0, 0, 0, 0, 0,     // -128 to -121: essentially zero
    // ... gradual increase ...
    32767                        // 0: full weight
};
```

The adaptive shift computation:
```cpp
int compute_adaptive_shift(int32_t score_range) {
    // Target: map score_range to ~128 LUT entries (7 bits)
    // shift = log2(score_range) - 7, minimum 0
    if (score_range <= 128) return 0;
    int shift = 0;
    while ((score_range >> shift) > 128) shift++;
    return shift;
}
```

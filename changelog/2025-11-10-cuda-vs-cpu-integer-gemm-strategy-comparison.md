# CUDA vs CPU Integer GEMM Strategy Comparison

**Date**: November 10, 2025  
**Context**: Clarifying strategy alignment with llama.cpp CUDA implementation

---

## Executive Summary

**Your Intuition is PARTIALLY CORRECT** - but with critical differences:

| Aspect | CUDA (llama.cpp) | CPU (Llaminar) | Match? |
|--------|------------------|----------------|---------|
| **Weights stay quantized** | ✅ YES | ✅ YES | ✅ **SAME** |
| **Activations quantized to INT8** | ❌ **NO** | ✅ YES | ❌ **DIFFERENT** |
| **Integer dot product** | ✅ YES (DP4A) | ✅ YES (DPBUSD) | ✅ **SAME** |
| **On-the-fly decode** | ✅ YES | ✅ YES | ✅ **SAME** |

**Key Difference**: CUDA uses **INT8 activations from Q8_1 pre-quantized format**, CPU uses **FP32 activations dynamically quantized to Q8_0**.

---

## Strategy Breakdown

### 1. Weight Handling (BOTH IDENTICAL ✅)

#### CUDA Implementation (mmq.cuh, lines 2310-2372)

```cuda
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_iq4_nl(
    const char * __restrict__ x, int * __restrict__ x_tile, ...) {
    
    const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;
    
    // Decode IQ4_NL nibbles to INT8 using lookup table
    const int aux_q4 = get_int_b2(bxi->qs, kqsx);
    const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);  // <-- LUT decode
    
    // Store decoded INT8 values in shared memory tile
    x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0]      = v.x;
    x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + QI4_NL] = v.y;
    
    // Store FP16 scale factor
    x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(bxi->d);
}
```

**Key Points**:
- ✅ IQ4_NL weights remain in compressed 4-bit format in global memory
- ✅ Decode happens **on-the-fly** into shared memory tiles using lookup table
- ✅ Result is INT8 values (from LUT: -127 to +113)
- ✅ Scale factor extracted separately

#### CPU Implementation (IntegerGemm.cpp, lines 243-250)

```cpp
// Decode weight blocks into INT8 vectors on-the-fly
__m256i b_vec_lo = decode_iq4nl_block_to_vec(b_block0);
__m256i b_vec_hi = decode_iq4nl_block_to_vec(b_block1);

static inline __m256i decode_iq4nl_block_to_vec(const IQ4_NLBlock &block) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
    const __m128i mask = _mm_set1_epi8(0x0F);
    // Extract nibbles, use shuffle for LUT lookup
    return _mm256_shuffle_epi8(table256, indices);  // <-- SIMD LUT decode
}
```

**Key Points**:
- ✅ IQ4_NL weights remain in compressed 4-bit format in memory
- ✅ Decode happens **on-the-fly** using AVX2 `vpshufb` (SIMD LUT)
- ✅ Result is INT8 values (same LUT: -127 to +113)
- ✅ Scale factor extracted separately

**Verdict**: ✅ **IDENTICAL STRATEGY** - Both decode weights on-the-fly, never materialize full FP32 weight matrix.

---

### 2. Activation Handling (DIFFERENT ❌)

#### CUDA Implementation (mmq.cuh - Relies on Q8_1 Pre-Quantized Activations)

**CRITICAL**: CUDA kernels assume activations are **already quantized to Q8_1 format** before GEMM.

```cuda
// From vecdotq.cuh, line 1176
static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, ...) {
    
    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;  // <-- Already INT8!
    
    int sumi = 0;
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);  // <-- DP4A (INT8×INT8)
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);  // <-- Scales
    return d * sumi;
}
```

**Key Points**:
- ❌ Activations are **NOT** FP32 - they're pre-quantized to `block_q8_1` (INT8 + FP16 scale)
- ✅ No dynamic quantization inside GEMM kernel
- ✅ Uses DP4A instruction: `int32_t += int8_t × int8_t` (4 pairs at once)

#### CPU Implementation (IntegerGemm.cpp, lines 200-220)

```cpp
// Pre-quantize entire A matrix ONCE (not per-row)
std::vector<Q8_0Block> a_q8_all(M * k_blocks);

for (int m = 0; m < M; ++m) {
    const float *a_row = A + m * lda;  // <-- Input is FP32!
    Q8_0Block *a_q8_row = &a_q8_all[m * k_blocks];
    quantize_fp32_to_q8_0(a_row, a_q8_row, K);  // <-- Dynamic quantization
}

// Then GEMM loop uses pre-quantized INT8 activations
__m256i a_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block0.qs));
__m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a_unsigned, b_vec);
```

**Key Points**:
- ❌ Activations start as **FP32** (not pre-quantized)
- ✅ Dynamic quantization to Q8_0 happens **once** before GEMM loop
- ✅ Uses DPBUSD instruction: `int32_t += uint8_t × int8_t` (4 pairs at once)

**Verdict**: ❌ **DIFFERENT STRATEGY** 
- CUDA: Expects activations **already quantized** (offline or in earlier stage)
- CPU: Quantizes activations **on-demand** from FP32 input

---

### 3. Integer Dot Product (BOTH IDENTICAL ✅)

#### CUDA: DP4A Instruction

```cuda
__device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif __CUDA_ARCH__ >= CC_TURING
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c));
#else
    // Scalar fallback
    const int8_t *a_ptr = (const int8_t *)&a;
    const int8_t *b_ptr = (const int8_t *)&b;
    c += a_ptr[0] * b_ptr[0];
    c += a_ptr[1] * b_ptr[1];
    c += a_ptr[2] * b_ptr[2];
    c += a_ptr[3] * b_ptr[3];
#endif
    return c;
}
```

**Operation**: `c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]` (4 INT8 pairs → 1 INT32)

#### CPU: DPBUSD Instruction

```cpp
__m512i acc = _mm512_dpbusd_epi32(
    _mm512_setzero_si512(),  // Initial accumulator
    a_unsigned,               // 64× UINT8 (unsigned activations)
    b_vec                     // 64× INT8 (signed weights)
);
```

**Operation**: `acc[i] += a[4*i+0]*b[4*i+0] + ... + a[4*i+3]*b[4*i+3]` (16 INT8 quads → 16 INT32)

**Verdict**: ✅ **SAME CONCEPT, DIFFERENT SCALE**
- CUDA DP4A: 4 INT8 pairs per instruction
- CPU DPBUSD: 16× INT8 quads per instruction (64 elements total)

---

### 4. Unsigned vs Signed INT8 (Important Detail)

#### CUDA: Signed × Signed

```cuda
sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
// v.x = INT8 signed weight (from LUT: -127 to +113)
// q8[l] = INT8 signed activation (from Q8_1: -127 to +127)
```

#### CPU: Unsigned × Signed (with Bias Correction)

```cpp
const __m512i bias128 = _mm512_set1_epi8((char)0x80);  // +128

// Convert activations from signed to unsigned (VNNI requirement)
__m512i a_unsigned = _mm512_add_epi8(a_vec, bias128);  // Add 128

// Compute correction terms per block (128 * Σ(B_signed))
int32_t b_sum0 = sum_int8_vec(b_vec_lo);
int32_t dot0 = horizontal_sum_epi32(acc_lo) - (128 * b_sum0);  // Bias correction
```

**Why the difference?**
- CUDA DP4A: Supports signed × signed natively
- AVX512-VNNI DPBUSD: Requires **unsigned × signed** operands

**Mathematical equivalence**:
```
(a + 128) * b = a*b + 128*b
∴ a*b = (a + 128)*b - 128*b
      = DPBUSD(a+128, b) - 128*Σ(b)  <-- Our correction term
```

**Verdict**: ✅ **EQUIVALENT (with bias correction)** - Both compute INT8×INT8 dot products, CPU needs algebraic adjustment for VNNI.

---

## Summary Table

| Feature | CUDA (llama.cpp) | CPU (Llaminar) | Strategy Match |
|---------|------------------|----------------|----------------|
| **Weight Format** | IQ4_NL (4-bit) | IQ4_NL (4-bit) | ✅ Identical |
| **Weight Materialization** | Never (stay compressed) | Never (stay compressed) | ✅ Identical |
| **Weight Decode** | On-the-fly LUT (shared mem) | On-the-fly LUT (SIMD) | ✅ Identical |
| **Activation Input** | Q8_1 (pre-quantized) | FP32 (full precision) | ❌ Different |
| **Activation Quantization** | Offline/earlier stage | Dynamic (in-kernel) | ❌ Different |
| **Activation Format** | INT8 signed | INT8 signed → unsigned | ⚠️ Equivalent |
| **Integer Instruction** | DP4A (4 pairs) | DPBUSD (64 elements) | ✅ Equivalent |
| **Dot Product** | INT8 × INT8 → INT32 | UINT8 × INT8 → INT32 | ✅ Equivalent |
| **Bias Correction** | Not needed (signed×signed) | Required (unsigned×signed) | ⚠️ Implementation detail |
| **Scale Application** | Post-accumulation (FP32) | Post-accumulation (FP64) | ✅ Equivalent |

---

## Key Insights

### What's SAME ✅

1. **Weights never dequantized** - IQ4_NL blocks remain compressed in memory
2. **On-the-fly decode** - Lookup table applied during GEMM, not pre-materialized
3. **Integer arithmetic** - Dot products computed in INT8/INT32 domain
4. **Block-wise scales** - FP16/FP32 scale factors applied after accumulation

### What's DIFFERENT ❌

1. **Activation source**:
   - CUDA: Expects **pre-quantized Q8_1** activations
   - CPU: Accepts **FP32** activations, quantizes dynamically

2. **Quantization timing**:
   - CUDA: Quantization happened **before** GEMM kernel (offline or separate pass)
   - CPU: Quantization happens **inside** GEMM function (one-time upfront cost)

3. **Performance implications**:
   - CUDA: Lower memory bandwidth (Q8_1 = 1 byte/element vs FP32 = 4 bytes/element)
   - CPU: Higher memory bandwidth, but compensated by cache efficiency

---

## Why the Difference?

### CUDA Context (llama.cpp)

**Activation Pipeline**:
```
Input Tokens → Embedding (FP16) → Attention (FP16) → FFN (FP16)
                    ↓                    ↓                 ↓
                Quantize to Q8_1  Quantize to Q8_1  Quantize to Q8_1
                    ↓                    ↓                 ↓
             IQ4_NL×Q8_1 GEMM    IQ4_NL×Q8_1 GEMM  IQ4_NL×Q8_1 GEMM
```

**Rationale**:
- GPU memory bandwidth is **precious** (VRAM is limited, bandwidth-constrained)
- Pre-quantizing activations to Q8_1 saves **4× memory bandwidth** (FP32 → INT8)
- Quantization overhead amortized across multiple GEMM operations per layer
- Shared memory tiles can hold 4× more data (critical for GPU occupancy)

### CPU Context (Llaminar)

**Activation Pipeline**:
```
Input Tokens → Embedding (FP32) → Attention (FP32) → FFN (FP32)
                                         ↓                 ↓
                                   FP32→Q8_0         FP32→Q8_0
                                         ↓                 ↓
                                 IQ4_NL×Q8_0 GEMM  IQ4_NL×Q8_0 GEMM
```

**Rationale**:
- CPU has **larger caches** (L1/L2/L3 hierarchy, less bandwidth-constrained)
- FP32 activations stay in cache across pipeline stages (no intermediate stores)
- Quantization overhead is **one-time** per GEMM (not per-layer)
- Simpler interface: User provides FP32, kernel handles quantization internally

---

## Performance Trade-offs

### CUDA Approach (Pre-Quantized Activations)

**Advantages**:
- ✅ **4× lower memory bandwidth** for activations
- ✅ **4× more activations** fit in shared memory (better occupancy)
- ✅ **Amortized quantization** cost across multiple GEMMs

**Disadvantages**:
- ❌ Requires **separate quantization kernel** launch
- ❌ Intermediate Q8_1 storage in VRAM (extra allocation)
- ❌ More complex pipeline (quantize → GEMM vs direct GEMM)

### CPU Approach (Dynamic Quantization)

**Advantages**:
- ✅ **Simpler API** - Single function call, FP32 in/out
- ✅ **No intermediate storage** - Q8_0 buffer is stack/local
- ✅ **Cache-friendly** - FP32 activations stay in L2/L3 across layers

**Disadvantages**:
- ❌ **4× higher memory bandwidth** for reading activations
- ❌ **Quantization overhead** per GEMM call
- ❌ **Larger L2 footprint** (FP32 activations + Q8_0 temp buffer)

---

## Numerical Equivalence

Despite different quantization timing, both approaches should produce **numerically similar** results:

### CUDA: Q8_1 Quantization (Symmetric + Zero-Point)

```c
// From llama.cpp ggml-quants.c
block_q8_1 {
    half d;         // Scale factor
    half s;         // Sum of elements (for zero-point correction)
    int8_t qs[32];  // Quantized values
}

q_i = round(x_i / d) - round(mean(x) / d)  // Zero-point quantization
```

### CPU: Q8_0 Quantization (Asymmetric)

```cpp
// Our implementation (IntegerGemm.cpp, lines 86-150)
struct Q8_0Block {
    uint16_t d;     // FP16 scale factor
    int8_t qs[32];  // Quantized values
};

float scale = max(abs(x)) / 127.0f;
q_i = round(x_i / scale);  // Asymmetric quantization
```

**Key Difference**: Q8_1 uses **zero-point** (mean subtraction), Q8_0 is **asymmetric** (no mean).

**Impact**: Negligible for weights (which are centered), minor for activations (which may have non-zero mean).

---

## Conclusion

### Your Intuition: **2/3 Correct** ✅✅❌

1. ✅ **"We never dequantize weights"** - **CORRECT**: Both keep IQ4_NL compressed, decode on-the-fly
2. ❌ **"We quantize activations to INT8 upfront"** - **PARTIALLY CORRECT**:
   - CUDA: Activations are **already INT8** (Q8_1) before entering GEMM
   - CPU: Activations are **FP32**, we quantize to Q8_0 **inside** GEMM function
3. ✅ **"We perform INT8 GEMM with VNNI"** - **CORRECT**: Both use integer dot products (DP4A vs DPBUSD)

### Final Verdict

**Our strategy is ALMOST IDENTICAL to CUDA, with one key difference**:

| Strategy Component | CUDA | CPU | Match? |
|--------------------|------|-----|--------|
| Keep weights compressed | ✅ | ✅ | ✅ YES |
| Decode weights on-the-fly | ✅ | ✅ | ✅ YES |
| Integer dot product | ✅ | ✅ | ✅ YES |
| **Activation quantization** | **Offline** | **Inline** | ❌ **NO** |

**Recommendation**: For production, consider **optional pre-quantized activation path** to match CUDA's bandwidth efficiency:

```cpp
// Current (FP32 input)
bool gemm_int8_iq4nl_vnni(const float *A, ...);

// Future (Q8_0 input - skip quantization)
bool gemm_int8_iq4nl_vnni_q8(const Q8_0Block *A_q8, ...);
```

This would give users the choice:
- **Simplicity**: FP32 input (current) - easier API, cache-friendly
- **Performance**: Q8_0 input (future) - lower bandwidth, matches CUDA strategy


# Typed Residual Optimization Project Plan

**Author**: David Sanftenberg  
**Date**: December 2, 2025  
**Status**: In Progress (Phase 1 Complete, Phase 2 Started)

## Executive Summary

This document outlines the implementation plan for **typed residuals** and **typed KV cache** in Llaminar V2 - memory bandwidth optimizations that store intermediate tensors in reduced-precision formats (Q8_1, FP16, BF16) instead of FP32.

**Key insight**: The optimization only yields benefits if we **fuse** the output quantization directly into kernel epilogues, so we never write FP32 to DRAM and then read it back for quantization. The data must go `Registers → Quantize → DRAM` in one pass.

### Scope

| Component | Current | Target Formats | Memory Savings | Config |
|-----------|---------|----------------|----------------|--------|
| **Residual Buffers** | FP32 | FP32, BF16, FP16, Q8_1 | Up to 3.5× | `ActivationPrecision` enum |
| **KV Cache** | FP32 | FP32, BF16, FP16, Q8_1 | Up to 3.5× | `ActivationPrecision` enum |

**Design Simplification**: All buffer formats (residuals, activations, KV cache) use a single `ActivationPrecision` enum. Kernels are treated as black boxes - they accept inputs in the configured precision and produce outputs in the same precision. This unified approach simplifies configuration and eliminates the need for separate format enums.

### Memory Impact (Qwen2.5-7B, 4K context)

| Component | FP32 Size | Q8_1 Size | BF16/FP16 Size |
|-----------|-----------|-----------|----------------|
| KV Cache (32 layers × 4K × 4096) | ~2 GB | ~600 MB | ~1 GB |
| Residuals (per forward) | ~168 MB | ~48 MB | ~84 MB |

## Current Architecture

### Existing Infrastructure

1. **ActivationPrecision Enum** (`PipelineConfig.h:95`)
   ```cpp
   enum class ActivationPrecision { FP32, BF16, FP16, INT32 };  // INT32 to be replaced with Q8_1
   ```
   - Already plumbed through pipeline configuration
   - Used to create activation tensors via `createActivationTensor()`
   - NOT currently used for residuals (all residuals are FP32)
   - **NOTE**: INT32 was intended for accumulation but is unused; will be replaced with Q8_1

2. **Templated Kernels** (already implemented)
   - `CpuAttentionKernelT<TensorT>` - FP32, BF16, FP16 specializations
   - `CPURMSNormKernelT<TensorT>` - templated with precision-specific paths
   - `CPURMSNormTypedKernel<ActivationPrecision>` - **NEW** typed kernel with fused precision conversion
   - `CPUSwiGLUKernelT<TensorT>` - templated
   - `CPURoPEKernelT<TensorT>` - templated

3. **ActivationTraits** (`primitives/ActivationTraits.h`)
   ```cpp
   template<typename TensorT> struct ActivationTraits {
       using ElementType = ...;
       static void apply_softmax(...);
       static std::unique_ptr<ITensorGemm> create_activation_gemm();
       static std::shared_ptr<TensorBase> allocate_workspace(...);
   };
   ```
   - Specializations exist for FP32Tensor, BF16Tensor, FP16Tensor, Q8_1Tensor
   - INT32Tensor specialization will be removed (unused)
   - Provides compile-time dispatch for precision-specific operations

4. **Q8_1Block Structure** (`BlockStructures.h:68`)
   ```cpp
   struct Q8_1Block {
       uint16_t d;      // FP16 scale factor
       int16_t sum_qs;  // Pre-computed sum for VNNI compensation
       int8_t qs[32];   // 32 quantized int8 values
   };
   ```
   - Already used for activation quantization in GEMM kernels
   - 36 bytes per 32 elements = 1.125 bytes/element vs 4 bytes/element for FP32 (3.5x compression)

### Current Data Flow (FP32 Residuals)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Per-Layer Flow (Attention Block)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  current_hidden (FP32)                                                      │
│         │                                                                   │
│         ├──────────────────────────────────┐                                │
│         │                                  │                                │
│         ▼                                  ▼                                │
│  save_residual()                      RMSNorm                               │
│  (memcpy FP32→FP32)                       │                                 │
│         │                                  ▼                                │
│         │                             Q/K/V GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│  residual buffer (FP32)               Attention                             │
│         │                                  │                                │
│         │                                  ▼                                │
│         │                            Output GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│         └────────────────────────────► add_residual()                       │
│                                       (FP32 + FP32)                         │
│                                            │                                │
│                                            ▼                                │
│                                    current_hidden (FP32)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Memory Traffic per layer (d_model=896, seq_len=512):
- save_residual: 896 × 512 × 4 = 1.75 MB write
- add_residual:  896 × 512 × 4 = 1.75 MB read (residual)
                 896 × 512 × 4 = 1.75 MB read (input)
                 896 × 512 × 4 = 1.75 MB write (output)
Total: 7 MB per layer × 24 layers = 168 MB per forward pass (just residuals!)
```

### Proposed Data Flow (Q8_1 Residuals)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Per-Layer Flow (Attention Block) - Q8_1 Residuals                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  current_hidden (FP32)                                                      │
│         │                                                                   │
│         ├──────────────────────────────────┐                                │
│         │                                  │                                │
│         ▼                                  ▼                                │
│  quantize_to_residual()               RMSNorm                               │
│  (FP32→Q8_1 in registers)                 │                                 │
│         │                                  ▼                                │
│         │                             Q/K/V GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│  residual buffer (Q8_1)               Attention                             │
│         │                                  │                                │
│         │                                  ▼                                │
│         │                            Output GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│         └────────────────────────────► add_residual_q8()                    │
│                                       (dequant Q8_1 + FP32)                 │
│                                            │                                │
│                                            ▼                                │
│                                    current_hidden (FP32)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Memory Traffic per layer (d_model=896, seq_len=512):
- save_residual: 896 × 512 × 1.125 = 0.5 MB write (Q8_1)
- add_residual:  896 × 512 × 1.125 = 0.5 MB read (residual Q8_1)
                 896 × 512 × 4 = 1.75 MB read (input FP32)
                 896 × 512 × 4 = 1.75 MB write (output FP32)
Total: 4.5 MB per layer × 24 layers = 108 MB per forward pass

Bandwidth Reduction: 168 MB → 108 MB = 36% reduction in residual traffic
```

## Implementation Progress

### Phase 1: Infrastructure & Traits ✅ COMPLETE (Dec 2, 2025)

- [x] `ResidualTraits<ActivationPrecision>` template with FP32, BF16, FP16, Q8_1 specializations
- [x] `quantize()`, `dequantize()`, `fused_add()`, `allocate()` primitives
- [x] SIMD-optimized implementations (AVX512/AVX2) for all precision types
- [x] Unit tests for ResidualTraits (14 tests passing)
- [x] Updated PipelineConfig.h to use unified `ActivationPrecision`
- [x] Removed redundant `ResidualFormat` and `KVCacheFormat` enums

### Phase 2: Typed Kernel Interfaces 🔄 IN PROGRESS (Started Dec 4, 2025)

- [x] `CPURMSNormTypedKernel<ActivationPrecision>` - typed RMSNorm with fused precision conversion
  - `apply_typed()` - accepts typed input/output, computes in FP32, fused dequant/requant
  - `apply_with_residual_add()` - fused residual addition + RMSNorm + requant
  - Specializations: FP32, BF16, FP16, Q8_1
  - Unit tests: 18 tests passing
- [ ] Typed kernel integration with pipeline Op classes (RMSNormOp)
- [ ] Pipeline instantiation based on `ActivationPrecision` config

### Phase 3: Fused Residual Addition 📋 NOT STARTED

- [ ] RMSNorm with fused residual input (dequant + add + normalize in one pass)
- [ ] Update pipeline to use fused residual patterns

### Phase 4: Pipeline Integration 📋 NOT STARTED

- [ ] `ActivationBuffers` with typed residual support
- [ ] Config-driven kernel instantiation

### Phase 5: KV Cache Integration 📋 NOT STARTED

- [ ] Typed KV cache storage and retrieval

### Phase 6: Typed KV Cache 📋 NOT STARTED

- [ ] KV cache precision follows `ActivationPrecision` config
- [ ] Modified K/V storage: store in configured precision after projection
- [ ] Modified K/V retrieval: dequant before attention computation
- [ ] Performance validation: measure cache hit latency impact

---

## Implementation Phases (Detailed Design)

### Phase 1: Infrastructure & Traits (2-3 days)

#### 1.1 Modify ActivationPrecision Enum (Replace INT32 with Q8_1)

```cpp
// PipelineConfig.h
enum class ActivationPrecision {
    FP32,   // 32-bit float (default)
    BF16,   // bfloat16
    FP16,   // IEEE float16
    Q8_1    // Block-quantized int8 with scale (36 bytes per 32 elements)
            // REPLACES INT32 which was unused
};
```

**Rationale for removing INT32**:
- INT32 was intended for quantized GEMM accumulation but was never implemented
- No existing code paths use `ActivationPrecision::INT32`
- Q8_1 is more useful: provides 3.5x compression for residual/activation storage
- Keeps enum size small (4 values) for switch statement efficiency
```

#### 1.2 Unified Buffer Format (No Separate ResidualFormat)

**Design Decision**: After architectural review, we determined that a separate `ResidualFormat` enum is unnecessary. All buffer formats (activations, residuals, KV cache) use the single `ActivationPrecision` enum.

**Rationale**:
- Kernels are black boxes: they handle internal precision conversions transparently
- Config layer only needs to specify DRAM storage format
- Simpler mental model: one enum controls all intermediate buffer formats
- Reduces configuration complexity and potential for mismatched formats

```cpp
// PipelineConfig.h - Simplified
struct PipelineConfig {
    ActivationPrecision activation_precision = ActivationPrecision::FP32;
    // No separate residual_format - residuals use activation_precision
    // ...
};
```

#### 1.3 Create ResidualTraits Template

```cpp
// src/v2/kernels/cpu/primitives/ResidualTraits.h

template<ActivationPrecision Precision>
struct ResidualTraits;

template<>
struct ResidualTraits<ActivationPrecision::FP32> {
    using StorageType = float;
    static constexpr size_t bytes_per_element = 4;
    static constexpr bool needs_quantization = false;
    
    static void save(const float* src, void* dst, size_t n) {
        std::memcpy(dst, src, n * sizeof(float));
    }
    
    static void add_to_output(const void* residual, const float* input, 
                              float* output, size_t n) {
        const float* res = static_cast<const float*>(residual);
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i) {
            output[i] = res[i] + input[i];
        }
    }
};

template<>
struct ResidualTraits<ActivationPrecision::Q8_1> {
    using StorageType = Q8_1Block;
    static constexpr size_t bytes_per_element = 36.0f / 32.0f;  // 1.125
    static constexpr bool needs_quantization = true;
    static constexpr size_t block_size = 32;
    
    static size_t storage_size(size_t n_elements) {
        size_t n_blocks = (n_elements + 31) / 32;
        return n_blocks * sizeof(Q8_1Block);
    }
    
    static void save(const float* src, void* dst, size_t n);
    static void add_to_output(const void* residual, const float* input,
                              float* output, size_t n);
};

template<>
struct ResidualTraits<ActivationPrecision::BF16> {
    using StorageType = uint16_t;  // BF16 stored as uint16
    static constexpr size_t bytes_per_element = 2;
    static constexpr bool needs_quantization = true;  // FP32→BF16 truncation
    
    static void save(const float* src, void* dst, size_t n);
    static void add_to_output(const void* residual, const float* input,
                              float* output, size_t n);
};

template<>
struct ResidualTraits<ActivationPrecision::FP16> {
    using StorageType = uint16_t;  // FP16 stored as uint16
    static constexpr size_t bytes_per_element = 2;
    static constexpr bool needs_quantization = true;
    
    static void save(const float* src, void* dst, size_t n);
    static void add_to_output(const void* residual, const float* input,
                              float* output, size_t n);
};
```

#### 1.4 Create Typed Residual Buffer Class

```cpp
// src/v2/tensors/ResidualBuffer.h

class IResidualBuffer {
public:
    virtual ~IResidualBuffer() = default;
    
    // Save FP32 activation to residual (quantizes if needed)
    virtual void save(const float* src, size_t n_elements) = 0;
    
    // Add residual to FP32 input, write to FP32 output
    virtual void add_to(const float* input, float* output, size_t n_elements) = 0;
    
    // Get raw storage for advanced use
    virtual void* raw_data() = 0;
    virtual const void* raw_data() const = 0;
    virtual size_t storage_bytes() const = 0;
    virtual ActivationPrecision format() const = 0;
};

template<ActivationPrecision Precision>
class TypedResidualBuffer : public IResidualBuffer {
    using Traits = ResidualTraits<Precision>;
    using StorageType = typename Traits::StorageType;
    
    std::vector<uint8_t> storage_;
    size_t capacity_elements_;
    
public:
    explicit TypedResidualBuffer(size_t max_elements);
    
    void save(const float* src, size_t n) override {
        Traits::save(src, storage_.data(), n);
    }
    
    void add_to(const float* input, float* output, size_t n) override {
        Traits::add_to_output(storage_.data(), input, output, n);
    }
    
    // ... other methods
};

// Factory function
std::unique_ptr<IResidualBuffer> createResidualBuffer(
    ActivationPrecision precision, size_t max_elements);
```

### Phase 2: Fused Kernel Epilogues (3-5 days)

This is the **critical optimization**: output quantization must happen in kernel epilogues while data is still in registers/L1 cache.

#### 2.1 GEMM Output with Fused Residual Save

The key insight is that `QuantisedGemmKernel` already computes results tile-by-tile. We need to add an optional "write as Q8_1" path in the epilogue.

```cpp
// GemmFusedOps extension
struct GemmFusedOps {
    // ... existing fields ...
    
    // NEW: Residual save fusion
    enum class OutputFormat {
        FP32,      // Default: write FP32 to C
        Q8_1,      // Quantize to Q8_1 while writing
        BF16,      // Convert to BF16 while writing  
        FP16       // Convert to FP16 while writing
    };
    
    OutputFormat output_format = OutputFormat::FP32;
    void* residual_dst = nullptr;  // Destination for fused residual save
    
    static GemmFusedOps with_residual_save(OutputFormat fmt, void* dst) {
        GemmFusedOps ops;
        ops.output_format = fmt;
        ops.residual_dst = dst;
        return ops;
    }
};
```

#### 2.2 Kernel Microkernel Epilogue Modifications

The actual kernel epilogues (in the VNNI microkernel) need modification:

```cpp
// In QuantisedGemmKernel - after accumulation, before store
void kernel_epilogue_q8_1(float* accum, Q8_1Block* dst, int n_cols) {
    // accum contains the FP32 results for this tile
    // We need to quantize and write to dst
    
    // Find max for quantization scale
    float max_abs = 0.0f;
    for (int i = 0; i < n_cols; ++i) {
        max_abs = std::max(max_abs, std::abs(accum[i]));
    }
    
    float d = max_abs / 127.0f;
    if (d < 1e-10f) d = 1e-10f;
    float id = 1.0f / d;
    
    // Quantize in groups of 32
    for (int blk = 0; blk < n_cols; blk += 32) {
        Q8_1Block& block = dst[blk / 32];
        block.d = fp32_to_fp16(d);  // Or per-block scale if needed
        
        int32_t sum_qs = 0;
        for (int j = 0; j < 32 && blk + j < n_cols; ++j) {
            int8_t q = static_cast<int8_t>(std::round(accum[blk + j] * id));
            block.qs[j] = q;
            sum_qs += q;
        }
        block.sum_qs = sum_qs;
    }
}
```

#### 2.3 Attention Output Projection with Fused Residual

Modify the attention output projection (`wo` GEMM) to optionally write Q8_1:

```cpp
// In Qwen2Pipeline::attention_block()

// Option A: Separate residual buffer (current approach, but typed)
if (activation_precision_ == ActivationPrecision::Q8_1) {
    // Project wo with fused Q8_1 residual save
    TRY_OP(project_with_residual_save(
        buffers.attn_output.get(), layer.wo.get(), buffers.attn_proj.get(),
        effective_seq_len, d_model_, n_heads_ * head_dim_,
        buffers.residual.get(), ActivationPrecision::Q8_1,  // Fused save to Q8_1
        layer_prefix + "_ATTENTION_OUTPUT"));
}
else {
    // Standard FP32 path
    TRY_OP(project(...));
    TRY_OP(save_residual(...));
}
```

#### 2.4 FFN Down Projection with Fused Residual

Same pattern for FFN down projection:

```cpp
// In Qwen2Pipeline::ffn_block()

if (activation_precision_ == ActivationPrecision::Q8_1) {
    TRY_OP(project_with_residual_save(
        buffers.up.get(), layer.down_proj.get(), buffers.ffn_output.get(),
        effective_seq_len, d_model_, d_ff_,
        buffers.residual.get(), ActivationPrecision::Q8_1,
        layer_prefix + "_FFN_DOWN"));
}
```

### Phase 3: Fused Residual Addition (2-3 days)

The residual addition should be fused into the **consumer** of the output, typically RMSNorm of the next operation.

#### 3.1 RMSNorm with Fused Residual Input

```cpp
// New interface for RMSNorm with Q8_1 residual fusion
class ITensorRMSNormFused {
public:
    // Apply RMSNorm with fused residual addition
    // output = RMSNorm(dequant(residual) + input)
    virtual bool apply_with_residual(
        const void* residual,        // Q8_1/BF16/FP16 residual
        ActivationPrecision residual_precision,
        const float* input,          // FP32 input to add
        const float* weight,         // RMSNorm gamma
        float* output,               // FP32 output
        int rows, int cols,
        float epsilon = 1e-6f) = 0;
};
```

#### 3.2 Fused Kernel Implementation

```cpp
// CPURMSNormKernelT with residual fusion
template<typename TensorT>
bool CPURMSNormKernelT<TensorT>::apply_with_residual_q8(
    const Q8_1Block* residual,
    const float* input,
    const float* weight,
    float* output,
    int rows, int cols,
    float epsilon)
{
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        // 1. Dequantize residual + add input into scratch
        std::array<float, 4096> scratch;  // Or thread-local buffer
        
        int n_blocks = (cols + 31) / 32;
        for (int blk = 0; blk < n_blocks; ++blk) {
            const Q8_1Block& block = residual[row * n_blocks + blk];
            float d = fp16_to_fp32(block.d);
            
            for (int j = 0; j < 32 && blk * 32 + j < cols; ++j) {
                int idx = blk * 32 + j;
                float dequant = block.qs[j] * d;
                scratch[idx] = dequant + input[row * cols + idx];
            }
        }
        
        // 2. Compute RMS over fused input
        float sum_sq = 0.0f;
        for (int i = 0; i < cols; ++i) {
            sum_sq += scratch[i] * scratch[i];
        }
        float rms = std::sqrt(sum_sq / cols + epsilon);
        float inv_rms = 1.0f / rms;
        
        // 3. Normalize and scale
        for (int i = 0; i < cols; ++i) {
            output[row * cols + i] = scratch[i] * inv_rms * weight[i];
        }
    }
    return true;
}
```

### Phase 4: Pipeline Integration (2-3 days)

#### 4.1 ActivationBuffers with Typed Residual

```cpp
// PipelineBase.h
struct ActivationBuffers {
    // Residual now uses typed buffer instead of TensorBase
    std::unique_ptr<IResidualBuffer> residual;
    
    // Other activation buffers (still FP32/BF16/FP16 based on activation_precision)
    std::shared_ptr<TensorBase> normalized;
    std::shared_ptr<TensorBase> Q;
    // ...
};
```

#### 4.2 Modified Qwen2Pipeline Methods

```cpp
// Qwen2Pipeline.cpp

ActivationBuffers Qwen2Pipeline::createBuffersForDevice(int device_idx, int max_seq_len) {
    ActivationBuffers buffers;
    int effective_max = batch_size_ * max_seq_len;
    buffers.max_seq_len = effective_max;
    
    // Create typed residual buffer based on activation precision
    ActivationPrecision precision = config_.activation_precision;
    
    buffers.residual = createResidualBuffer(
        precision, 
        static_cast<size_t>(effective_max) * d_model_);
    
    // Rest of activation buffers unchanged...
}

bool Qwen2Pipeline::attention_block(...) {
    // ...
    
    // Save residual (uses typed buffer's quantization)
    buffers.residual->save(input_hidden->data(), effective_seq_len * d_model_);
    
    // ... attention computation ...
    
    // Add residual (uses typed buffer's dequant + add)
    buffers.residual->add_to(
        buffers.attn_proj->data(),
        current_hidden_->mutable_data(),
        effective_seq_len * d_model_);
    
    return true;
}
```

### Phase 5: SIMD Optimization (3-4 days)

#### 5.1 AVX512 Q8_1 Quantization

```cpp
// SIMDHelpers.h - Q8_1 quantization with AVX512

inline void quantize_fp32_to_q8_1_avx512(
    const float* src, Q8_1Block* dst, size_t n_blocks)
{
    for (size_t blk = 0; blk < n_blocks; ++blk) {
        const float* src_blk = src + blk * 32;
        Q8_1Block& out = dst[blk];
        
        // Load 32 floats
        __m512 v0 = _mm512_loadu_ps(src_blk);
        __m512 v1 = _mm512_loadu_ps(src_blk + 16);
        
        // Find max absolute value
        __m512 abs0 = _mm512_abs_ps(v0);
        __m512 abs1 = _mm512_abs_ps(v1);
        __m512 max_abs = _mm512_max_ps(abs0, abs1);
        float max_val = _mm512_reduce_max_ps(max_abs);
        
        // Compute scale
        float d = max_val / 127.0f;
        if (d < 1e-10f) d = 1e-10f;
        float id = 1.0f / d;
        
        out.d = fp32_to_fp16(d);
        
        // Quantize: round(val * id)
        __m512 scale_vec = _mm512_set1_ps(id);
        __m512 q0 = _mm512_roundscale_ps(_mm512_mul_ps(v0, scale_vec), 
                                          _MM_FROUND_TO_NEAREST_INT);
        __m512 q1 = _mm512_roundscale_ps(_mm512_mul_ps(v1, scale_vec),
                                          _MM_FROUND_TO_NEAREST_INT);
        
        // Convert to int8 and pack
        __m512i i0 = _mm512_cvtps_epi32(q0);
        __m512i i1 = _mm512_cvtps_epi32(q1);
        
        // Pack to int8 (AVX512BW)
        __m256i packed = _mm512_cvtsepi32_epi8(_mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_cvtsepi32_epi8(i0)),
            _mm512_cvtsepi32_epi8(i1), 1));
        
        _mm256_storeu_si256((__m256i*)out.qs, packed);
        
        // Compute sum_qs using horizontal add
        __m512i sum = _mm512_add_epi32(i0, i1);
        out.sum_qs = _mm512_reduce_add_epi32(sum);
    }
}
```

#### 5.2 AVX512 Q8_1 Dequant + Add

```cpp
inline void dequant_q8_1_add_fp32_avx512(
    const Q8_1Block* residual, 
    const float* input,
    float* output,
    size_t n_blocks)
{
    for (size_t blk = 0; blk < n_blocks; ++blk) {
        const Q8_1Block& block = residual[blk];
        const float* in_ptr = input + blk * 32;
        float* out_ptr = output + blk * 32;
        
        float d = fp16_to_fp32(block.d);
        __m512 scale = _mm512_set1_ps(d);
        
        // Load int8 values
        __m256i qs = _mm256_loadu_si256((__m256i*)block.qs);
        
        // Extend to int32
        __m512i i0 = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(qs));
        __m512i i1 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(qs, 1));
        
        // Convert to float and scale
        __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(i0), scale);
        __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(i1), scale);
        
        // Load input and add
        __m512 in0 = _mm512_loadu_ps(in_ptr);
        __m512 in1 = _mm512_loadu_ps(in_ptr + 16);
        
        __m512 out0 = _mm512_add_ps(f0, in0);
        __m512 out1 = _mm512_add_ps(f1, in1);
        
        // Store
        _mm512_storeu_ps(out_ptr, out0);
        _mm512_storeu_ps(out_ptr + 16, out1);
    }
}
```

### Phase 6: Typed KV Cache (4-6 days)

The KV cache stores K/V tensors for all past tokens across all layers. For long-context inference, this dominates memory usage:

**Memory Formula**: `2 × n_layers × max_seq_len × n_kv_heads × head_dim × bytes_per_element`

| Model | Layers | KV Heads | Head Dim | 4K FP32 | 4K Q8_1 | 32K FP32 | 32K Q8_1 |
|-------|--------|----------|----------|---------|---------|----------|----------|
| Qwen2.5-0.5B | 24 | 2 | 64 | 24 MB | 7 MB | 192 MB | 55 MB |
| Qwen2.5-7B | 28 | 4 | 128 | 224 MB | 64 MB | 1.8 GB | 512 MB |
| Qwen2.5-14B | 40 | 8 | 128 | 640 MB | 183 MB | 5 GB | 1.4 GB |
| Qwen2.5-72B | 80 | 8 | 128 | 1.3 GB | 366 MB | 10 GB | 2.9 GB |

#### 6.1 KV Cache Uses ActivationPrecision (No Separate Enum)

**Design Decision**: KV cache format follows `ActivationPrecision` - no separate `KVCacheFormat` enum needed.

**Rationale**: K and V tensors are outputs from projection GEMMs, which produce activation tensors. If `ActivationPrecision::BF16`, then K/V projections produce BF16 tensors, which get stored in the KV cache as BF16. There's no sensible use case for "FP32 activations but Q8_1 KV cache" - the K/V values come from activation tensors!

Similarly, residuals now also use `ActivationPrecision` for the same reason - they are activation values being stored, and the kernel black-box model means we only need to specify the DRAM storage format.

```cpp
// KV cache format is derived from activation_precision
// NO new enum needed - just use ActivationPrecision

// In KVCache constructor:
KVCache(const MPIContext& mpi_ctx, int n_layers, int max_seq_len,
        int n_kv_heads, int head_dim, int device_idx,
        ActivationPrecision precision = ActivationPrecision::FP32);
```

#### 6.2 Create Typed KV Cache Entry

```cpp
// src/v2/tensors/KVCache.h

/**
 * @brief Storage format traits for KV cache (reuses ActivationPrecision)
 */
template<ActivationPrecision Precision>
struct KVCacheTraits;

template<>
struct KVCacheTraits<ActivationPrecision::FP32> {
    using StorageType = float;
    static constexpr size_t bytes_per_element = 4;
    static constexpr bool needs_conversion = false;
};

template<>
struct KVCacheTraits<ActivationPrecision::BF16> {
    using StorageType = uint16_t;
    static constexpr size_t bytes_per_element = 2;
    static constexpr bool needs_conversion = true;
};

template<>
struct KVCacheTraits<ActivationPrecision::FP16> {
    using StorageType = uint16_t;
    static constexpr size_t bytes_per_element = 2;
    static constexpr bool needs_conversion = true;
};

template<>
struct KVCacheTraits<ActivationPrecision::Q8_1> {
    using StorageType = Q8_1Block;
    static constexpr size_t bytes_per_element = 36.0f / 32.0f;  // 1.125
    static constexpr bool needs_conversion = true;
    static constexpr size_t block_size = 32;
};

/**
 * @brief Typed per-layer KV cache entry
 *
 * Stores K/V in the same format as activations.
 */
struct TypedKVCacheEntry
{
    void* K_storage = nullptr;  // Raw storage (format-dependent)
    void* V_storage = nullptr;
    size_t storage_bytes_k = 0;
    size_t storage_bytes_v = 0;
    int cached_tokens = 0;
    int device_idx = -1;
    ActivationPrecision precision = ActivationPrecision::FP32;
    
    ~TypedKVCacheEntry();
};
```

#### 6.3 KV Cache Interface Updates

```cpp
// KVCache.h - Updated interface

class KVCache {
public:
    /**
     * @brief Construct KV cache with specified storage precision
     *
     * @param precision Storage format matches activation precision
     */
    KVCache(const MPIContext& mpi_ctx, int n_layers, int max_seq_len,
            int n_kv_heads, int head_dim, int device_idx,
            ActivationPrecision precision = ActivationPrecision::FP32);
    
    /**
     * @brief Append K/V tensors (format matches activation precision)
     *
     * Input format depends on activation_precision setting.
     * - FP32: float* input
     * - BF16: uint16_t* input  
     * - Q8_1: Q8_1Block* input
     */
    bool append_kv(int layer, const void* new_k, const void* new_v, int new_seq_len);
    
    /**
     * @brief Get raw K/V storage for direct consumption by typed kernels
     */
    const void* get_k_raw(int layer) const;
    const void* get_v_raw(int layer) const;
    
    /**
     * @brief Get storage precision
     */
    ActivationPrecision precision() const { return precision_; }
    
    /**
     * @brief Get memory usage in bytes
     */
    size_t memory_usage_bytes() const;

private:
    ActivationPrecision precision_;
    // ...
};
```

#### 6.4 Fused K/V Projection with KV Cache Write

The key optimization is fusing K/V projection output with cache append:

```cpp
// In attention kernel or orchestrator

// K/V projection outputs are already in the correct activation format
// For BF16 activations, K_proj is BF16, so append directly:
kv_cache_->append_kv(layer_idx, K_proj->data(), V_proj->data(), effective_seq_len);

// No format conversion needed - K/V tensors match KV cache format
```

#### 6.5 Attention Kernel with Typed KV

Attention kernels need to handle typed K/V inputs:

```cpp
// GQAAttention or similar

class IAttentionKernel {
public:
    // Attention with typed K/V cache (format matches activation precision)
    virtual bool compute_with_typed_kv(
        const void* Q,               // Query (activation format)
        const void* K_cache,         // K cache (same format)
        const void* V_cache,         // V cache (same format)
        ActivationPrecision precision,
        int cached_seq_len,
        void* output,                // Output (activation format)
        const AttentionConfig& config
    ) = 0;
};
```

For Q8_1 activations, the attention kernel can either:
1. **Dequantize on-the-fly** during Q·K^T computation (bandwidth-efficient)
2. **Pre-dequantize** to temporary FP32 buffer (simpler implementation)

Option 1 is preferred for long contexts:

```cpp
// Q·K^T with on-the-fly Q8_1 dequantization
// K is stored as Q8_1 blocks: [cached_seq, n_kv_heads * head_dim]
// Need to compute: Q[q, :] · K[k, :]^T for all (q, k) pairs

void qk_dot_q8_cache(
    const float* Q,          // [seq_q, head_dim]
    const Q8_1Block* K_q8,   // [cached_seq, kv_dim / 32] blocks
    float* scores,           // [seq_q, cached_seq]
    int seq_q, int cached_seq, int head_dim)
{
    // For each query position
    for (int q = 0; q < seq_q; ++q) {
        const float* Q_row = Q + q * head_dim;
        
        // For each cached key position
        for (int k = 0; k < cached_seq; ++k) {
            float dot = 0.0f;
            
            // Accumulate over head_dim (32 elements at a time)
            int n_blocks = (head_dim + 31) / 32;
            for (int blk = 0; blk < n_blocks; ++blk) {
                const Q8_1Block& block = K_q8[k * n_blocks + blk];
                float d = fp16_to_fp32(block.d);
                
                // Dequant + dot in one pass
                for (int j = 0; j < 32 && blk * 32 + j < head_dim; ++j) {
                    float K_val = block.qs[j] * d;
                    dot += Q_row[blk * 32 + j] * K_val;
                }
            }
            
            scores[q * cached_seq + k] = dot;
        }
    }
}
```

#### 6.6 Memory Layout Considerations

For Q8_1 format, the layout must be attention-friendly:

```
Option A: Per-token blocks (contiguous K/V per position)
K[token=0]: [block_0, block_1, ..., block_N]  (N = head_dim / 32)
K[token=1]: [block_0, block_1, ..., block_N]
...

Option B: Per-dimension blocks (contiguous dimension across tokens)
K[dim_block=0]: [token_0, token_1, ..., token_T]
K[dim_block=1]: [token_0, token_1, ..., token_T]
...
```

Option A (per-token) is preferred because:
- Cache-friendly for sequential key access during Q·K^T
- Append is simple (just add blocks at end)
- Natural for position-based eviction

### Phase 7: Testing & Validation (2-3 days)

#### 6.1 Unit Tests

```cpp
// tests/v2/unit/Test__ResidualBuffer.cpp

TEST(ResidualBuffer, Q8_1_RoundTrip) {
    // Test quantize → dequant preserves values within tolerance
    std::vector<float> original(1024);
    // Fill with random values
    
    auto buffer = createResidualBuffer(ActivationPrecision::Q8_1, 1024);
    buffer->save(original.data(), 1024);
    
    std::vector<float> zeros(1024, 0.0f);
    std::vector<float> recovered(1024);
    buffer->add_to(zeros.data(), recovered.data(), 1024);
    
    // Check max error
    float max_err = 0.0f;
    for (size_t i = 0; i < 1024; ++i) {
        max_err = std::max(max_err, std::abs(original[i] - recovered[i]));
    }
    
    // Q8_1 should have <1% error for typical activation ranges
    float max_abs = *std::max_element(original.begin(), original.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    EXPECT_LT(max_err / max_abs, 0.01f);
}
```

#### 6.2 E2E Parity Tests

```cpp
// tests/v2/e2e/Test__TypedResidualParity.cpp

TEST(TypedResidualParity, Q8_1_vs_FP32_Logits) {
    // Load same model twice with different activation precisions
    PipelineConfig config_fp32;
    config_fp32.activation_precision = ActivationPrecision::FP32;
    
    PipelineConfig config_q8;
    config_q8.activation_precision = ActivationPrecision::Q8_1;
    
    auto pipeline_fp32 = createPipeline(config_fp32);
    auto pipeline_q8 = createPipeline(config_q8);
    
    // Run same input through both
    std::vector<int> tokens = {785, 3974, 13876};  // "The quick brown"
    
    pipeline_fp32->forward(tokens.data(), tokens.size());
    pipeline_q8->forward(tokens.data(), tokens.size());
    
    // Compare logits
    const float* logits_fp32 = pipeline_fp32->logits();
    const float* logits_q8 = pipeline_q8->logits();
    
    // Top-1 should match
    int top1_fp32 = std::max_element(logits_fp32, logits_fp32 + vocab_size) - logits_fp32;
    int top1_q8 = std::max_element(logits_q8, logits_q8 + vocab_size) - logits_q8;
    EXPECT_EQ(top1_fp32, top1_q8);
    
    // Cosine similarity should be high
    float cos_sim = cosine_similarity(logits_fp32, logits_q8, vocab_size);
    EXPECT_GT(cos_sim, 0.999f);
}
```

#### 6.3 Performance Benchmarks

```cpp
// tests/v2/performance/Perf__TypedResidual.cpp

TEST(TypedResidualPerf, BandwidthComparison) {
    const int seq_len = 512;
    const int d_model = 896;
    const int n_iterations = 1000;
    
    // Benchmark FP32 residual
    auto buffer_fp32 = createResidualBuffer(ActivationPrecision::FP32, seq_len * d_model);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iterations; ++i) {
        buffer_fp32->save(input.data(), seq_len * d_model);
        buffer_fp32->add_to(input.data(), output.data(), seq_len * d_model);
    }
    auto fp32_time = elapsed(start);
    
    // Benchmark Q8_1 residual
    auto buffer_q8 = createResidualBuffer(ActivationPrecision::Q8_1, seq_len * d_model);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iterations; ++i) {
        buffer_q8->save(input.data(), seq_len * d_model);
        buffer_q8->add_to(input.data(), output.data(), seq_len * d_model);
    }
    auto q8_time = elapsed(start);
    
    // Report
    std::cout << "FP32 residual: " << fp32_time << " ms\n";
    std::cout << "Q8_1 residual: " << q8_time << " ms\n";
    std::cout << "Speedup: " << fp32_time / q8_time << "x\n";
}
```

### Phase 8: Documentation & CLI (1 day)

#### 8.1 CLI Arguments

```bash
# ArgParser.h
--activation-precision <prec> Activation/residual/KV cache precision: fp32, bf16, fp16, q8_1 (default: fp32)
                              Controls all intermediate buffer formats (unified design)

# Example usage
./run_llaminar.sh -- -m model.gguf --activation-precision bf16 --benchmark
# This gives: BF16 activations, BF16 residuals, BF16 KV cache (all unified)
```

#### 8.2 Documentation Update

Update `.github/copilot-instructions.md` with:
- Updated `--activation-precision` CLI option (now affects KV cache too)
- New `--residual-format` CLI option  
- Performance characteristics of each format
- Accuracy/performance tradeoffs
- Memory usage calculator for different configurations

## File List

### New Files

| File | Purpose |
|------|---------|
| `src/v2/kernels/cpu/primitives/ResidualTraits.h` | Compile-time traits for residual formats |
| `src/v2/kernels/cpu/primitives/ResidualTraits.cpp` | Vectorized quantize/dequant implementations |
| `src/v2/tensors/ResidualBuffer.h` | Typed residual buffer interface + implementations |
| `src/v2/tensors/ResidualBuffer.cpp` | Non-inline implementations |
| `tests/v2/unit/Test__ResidualTraits.cpp` | Unit tests for residual traits (COMPLETE ✓) |
| `tests/v2/unit/Test__ResidualBuffer.cpp` | Unit tests for residual buffer operations |
| `tests/v2/unit/Test__TypedKVCache.cpp` | Unit tests for typed KV cache |
| `tests/v2/e2e/Test__TypedResidualParity.cpp` | E2E parity tests |
| `tests/v2/e2e/Test__TypedKVCacheParity.cpp` | E2E parity tests for KV cache |
| `tests/v2/performance/Perf__TypedResidual.cpp` | Performance benchmarks |
| `tests/v2/performance/Perf__TypedKVCache.cpp` | KV cache performance benchmarks |

### Modified Files

| File | Changes |
|------|---------|
| `src/v2/pipelines/PipelineConfig.h` | Replace INT32 with Q8_1 in ActivationPrecision (COMPLETE ✓) |
| `src/v2/pipelines/PipelineBase.h` | Add TensorFactory member (COMPLETE ✓) |
| `src/v2/pipelines/PipelineBase.cpp` | Initialize tensor_factory_ conditionally (COMPLETE ✓) |
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | Use typed residual buffer, typed KV cache |
| `src/v2/tensors/KVCache.h` | Add ActivationPrecision support, typed storage |
| `src/v2/tensors/KVCache.cpp` | Implement typed storage based on ActivationPrecision |
| `src/v2/tensors/TensorFactory.h` | Add createActivation(), createResidual() (COMPLETE ✓) |
| `src/v2/tensors/TensorFactory.cpp` | Implement factory methods (COMPLETE ✓) |
| `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` | Add fused residual save in epilogue |
| `src/v2/kernels/cpu/ops/CPURMSNormKernelT.h` | Add `apply_with_residual()` fused method |
| `src/v2/kernels/cpu/attention/GQAAttention.cpp` | Support typed K/V cache |
| `src/v2/tensors/SIMDHelpers.h` | Add AVX512/AVX2 Q8_1 quant/dequant intrinsics |
| `.github/copilot-instructions.md` | Document new features |

## Timeline Estimate

| Phase | Duration | Dependencies | Status |
|-------|----------|--------------|--------|
| Phase 1: Infrastructure & Traits | 2-3 days | None | ✅ COMPLETE |
| Phase 2: Fused Kernel Epilogues | 3-5 days | Phase 1 | Pending |
| Phase 3: Fused Residual Addition | 2-3 days | Phase 1 | Pending |
| Phase 4: Pipeline Integration (Residuals) | 2-3 days | Phases 2, 3 | Pending |
| Phase 5: SIMD Optimization | 3-4 days | Phase 4 | Pending |
| Phase 6: Typed KV Cache | 4-6 days | Phase 1 | Pending |
| Phase 7: Testing & Validation | 2-3 days | Phases 5, 6 | Pending |
| Phase 8: Documentation & CLI | 1 day | Phase 7 | Pending |

**Total**: ~19-29 days (4-6 weeks)

## Risk Assessment

### High Risk
- **Accuracy degradation (residuals)**: Q8_1 has limited precision. May cause accuracy issues in deeper layers where residuals accumulate quantization noise.
  - Mitigation: E2E parity tests, configurable per-layer residual format

- **Accuracy degradation (KV cache)**: Quantized K/V affects attention scores directly.
  - Mitigation: BF16/FP16 may be safer than Q8_1 for KV cache; test thoroughly

### Medium Risk
- **Performance not as expected**: Memory bandwidth savings may be offset by quantization compute overhead
  - Mitigation: Profile both paths, ensure quantization is SIMD-optimized, fuse operations

- **KV cache append overhead**: Q8_1 quantization during decode adds latency
  - Mitigation: Use BF16/FP16 for KV if Q8_1 decode latency is unacceptable

### Low Risk
- **Configuration simplicity**: Single `ActivationPrecision` enum controls all buffer formats
  - Benefit: Eliminates confusion about separate format enums

## Success Criteria

### Residual Typed Storage
1. **Correctness**: Q8_1 residuals produce same top-1 token as FP32 on test prompts
2. **Performance**: ≥20% reduction in residual memory traffic on benchmark
3. **Stability**: No accuracy degradation on 1000-token generation tests

### KV Cache Typed Storage
4. **Correctness**: Typed KV cache produces same top-1 token as FP32 on test prompts
5. **Performance**: ≥30% reduction in KV cache memory usage for long contexts
6. **Scalability**: Enable 32K+ context on memory-constrained systems

### General
7. **Compatibility**: Existing flags continue to work unchanged
8. **Documentation**: Clear guidance on when to use each format

## Appendix A: Memory Bandwidth Calculations

### Per-Layer Memory Traffic (Qwen2.5-0.5B, d_model=896, seq_len=512)

| Operation | FP32 Traffic | Q8_1 Traffic | Savings |
|-----------|-------------|--------------|---------|
| save_residual (attn) | 1.75 MB | 0.5 MB | 71% |
| add_residual (attn) | 3.5 MB | 2.25 MB | 36% |
| save_residual (ffn) | 1.75 MB | 0.5 MB | 71% |
| add_residual (ffn) | 3.5 MB | 2.25 MB | 36% |
| **Total per layer** | **10.5 MB** | **5.5 MB** | **48%** |

### Full Forward Pass (24 layers)

| Format | Total Residual Traffic | vs FP32 |
|--------|----------------------|---------|
| FP32 | 252 MB | baseline |
| Q8_1 | 132 MB | -48% |
| BF16/FP16 | 126 MB | -50% |

## Appendix B: Quantization Error Analysis

### Q8_1 Block Properties
- Scale: FP16 (16 bits)
- Values: INT8 × 32
- Range: -127 to +127 scaled by FP16 scale

### Expected Error
- Relative error: ≤ 1/127 ≈ 0.8% per quantization
- Accumulated over 24 layers: ~10-20% drift in residual stream
- Acceptable because: attention mechanism re-normalizes at each layer (RMSNorm)

### When to Avoid Q8_1 Residuals
- Very deep models (>100 layers)
- Tasks requiring high numerical precision
- When using INT8 weight quantization (compounds error)

## Appendix C: KV Cache Memory Calculations

### KV Cache Formula

```
KV Cache Size = 2 × n_layers × max_seq_len × n_kv_heads × head_dim × bytes_per_element
              = 2 × n_layers × max_seq_len × kv_dim × bytes_per_element

where kv_dim = n_kv_heads × head_dim
```

### KV Cache Size by Model and Format

| Model | Layers | KV Dim | 4K FP32 | 4K BF16 | 4K Q8_1 | 32K FP32 | 32K Q8_1 |
|-------|--------|--------|---------|---------|---------|----------|----------|
| Qwen2.5-0.5B | 24 | 128 | 24 MB | 12 MB | 7 MB | 192 MB | 55 MB |
| Qwen2.5-3B | 36 | 512 | 144 MB | 72 MB | 41 MB | 1.15 GB | 330 MB |
| Qwen2.5-7B | 28 | 512 | 112 MB | 56 MB | 32 MB | 896 MB | 256 MB |
| Qwen2.5-14B | 40 | 1024 | 320 MB | 160 MB | 91 MB | 2.5 GB | 730 MB |
| Qwen2.5-32B | 64 | 1024 | 512 MB | 256 MB | 146 MB | 4 GB | 1.1 GB |
| Qwen2.5-72B | 80 | 1024 | 640 MB | 320 MB | 183 MB | 5 GB | 1.4 GB |

### Per-Token KV Memory

| Format | Bytes per Token per Layer | Compression |
|--------|--------------------------|-------------|
| FP32 | 8 × kv_dim | 1× (baseline) |
| BF16/FP16 | 4 × kv_dim | 2× |
| Q8_1 | ~2.25 × kv_dim | ~3.5× |

### KV Cache Bandwidth During Attention

For each decode step, the full KV cache is read during Q·K^T and Score·V:

```
Bandwidth per decode step = 2 × cached_seq × kv_dim × bytes_per_element × n_layers
```

| Context | Layers | KV Dim | FP32 Read | Q8_1 Read | Savings |
|---------|--------|--------|-----------|-----------|---------|
| 4K | 28 | 512 | 112 MB | 32 MB | 71% |
| 16K | 28 | 512 | 448 MB | 128 MB | 71% |
| 32K | 28 | 512 | 896 MB | 256 MB | 71% |

This bandwidth is **per decode step**, making typed KV cache critical for long-context inference.

### Recommended Formats by Use Case

| Use Case | Residual Format | KV Cache Format | Notes |
|----------|-----------------|-----------------|-------|
| Maximum accuracy | FP32 | FP32 | Baseline, most memory |
| Balanced (recommended) | BF16 | BF16 | Good accuracy, 2× compression |
| Long context (16K+) | BF16 | Q8_1 | Memory-efficient for large contexts |
| Extreme compression | Q8_1 | Q8_1 | Test accuracy carefully |
| Intel AMX optimized | BF16 | BF16 | Native BF16 compute |

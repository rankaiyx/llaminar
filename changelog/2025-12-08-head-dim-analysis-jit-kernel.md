# Head Dimension Analysis for Q8_1 Fused Attention JIT Kernel

**Date**: December 8, 2025  
**Branch**: feature/typed-residuals  
**Author**: David Sanftenberg

## Overview

Analyzed head dimension requirements across target model architectures to determine JIT kernel coverage for the `QuantisedAttentionJit_Q8_1_Fused` kernel.

## Current JIT Kernel Support

The JIT kernel in `CPUAttentionKernelTyped.h` currently caches optimized instances for:
- `head_dim=64`
- `head_dim=128`

Other head_dim values fall back to `fused_q8_1_attention_reference()`.

## Model Architecture Analysis

### Qwen 2.5 Family ✅ (All Covered)

| Model | hidden_size | num_heads | head_dim | JIT Support |
|-------|-------------|-----------|----------|-------------|
| 0.5B | 896 | 14 | **64** | ✅ |
| 1.5B | 1536 | 12 | **128** | ✅ |
| 3B | 2048 | 16 | **128** | ✅ |
| 7B | 3584 | 28 | **128** | ✅ |
| 14B | 5120 | 40 | **128** | ✅ |
| 32B | 5120 | 40 | **128** | ✅ |
| 72B | 8192 | 64 | **128** | ✅ |

### Qwen3 Dense Family ✅ (All Covered)

| Model | hidden_size | num_heads | head_dim (explicit) | JIT Support |
|-------|-------------|-----------|---------------------|-------------|
| 0.6B | 1024 | 16 | **64** | ✅ |
| 1.7B | 2048 | 16 | **128** | ✅ |
| 8B | 4096 | 32 | **128** | ✅ |
| 14B | 5120 | 40 | **128** | ✅ |
| 32B | 5120 | 64 | **128** | ✅ |

Note: Qwen3 explicitly declares `head_dim` in config.json (unlike Qwen 2.5 which computed it).

### Qwen3-MoE Family ✅ (All Covered)

| Model | hidden_size | num_heads | head_dim (explicit) | JIT Support |
|-------|-------------|-----------|---------------------|-------------|
| 30B-A3B | 2048 | 32 | **128** | ✅ |
| 235B-A22B | 4096 | 64 | **128** | ✅ |

### DeepSeek V3 / R1 ❌ (NOT Covered - MLA Architecture)

DeepSeek V3/R1 uses **Multi-head Latent Attention (MLA)**, which is architecturally different:

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size | 7168 | |
| num_attention_heads | 128 | |
| num_key_value_heads | 128 | |
| **qk_nope_head_dim** | **128** | Non-positional component |
| **qk_rope_head_dim** | **64** | Rotary positional component |
| **v_head_dim** | **128** | Value head dimension |
| kv_lora_rank | 512 | KV latent compression |
| q_lora_rank | 1536 | Query latent compression |

**Key Differences from Standard Attention:**
1. Q/K head dimension = 128 (nope) + 64 (rope) = **192 total**
2. V head dimension = **128** (asymmetric with Q/K)
3. Uses latent space projections via LoRA-style compression
4. Standard fused attention kernel cannot directly support this

**Future Work Required:**
- Separate MLA pipeline implementation
- MLA-specific attention kernel (handles asymmetric Q/K vs V dimensions)
- Latent space projection handling

## Conclusion

**No changes needed to current JIT kernel caching.** The existing head_dim=64 and head_dim=128 support covers:
- ✅ All Qwen 2.5 models (0.5B - 72B)
- ✅ All Qwen3 dense models (0.6B - 32B)
- ✅ All Qwen3-MoE models (30B-A3B, 235B-A22B)

DeepSeek V3/R1 support will require a dedicated MLA implementation (tracked separately).

## Files Referenced

- `src/v2/kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h` - JIT kernel
- `src/v2/kernels/cpu/attention/CPUAttentionKernelTyped.h` - head_dim caching (lines 285-307)

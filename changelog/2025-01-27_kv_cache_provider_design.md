# KVCacheProvider Design Document

**Author**: David Sanftenberg  
**Date**: 2025-01-27  
**Status**: Design Proposal

## Problem Statement

The current architecture has a critical cache transfer issue:

1. **Prefill Path**: `PrefillProvider::execute()` populates KV cache internally but doesn't expose it
2. **Decode Path**: `MPIAttentionKernel` reads from `QwenPipeline::k_cache_[]` and `v_cache_[]`
3. **The Gap**: Cache never transfers from PrefillProvider → QwenPipeline
4. **Impact**: Decode receives empty cache → zero attention scores → token divergence

## Design Goals

1. **MPI Compatible**: Support per-rank cache sharding (KV head parallelism)
2. **Architecture Fit**: Integrate cleanly with existing PrefillProvider pattern
3. **Zero Copy**: Avoid unnecessary memory allocations where possible
4. **Backward Compatible**: Don't break existing decode path or tests
5. **Type Safe**: Ensure proper cache ownership and lifetime management

## Proposed Design

### Option A: KVCacheProvider Interface (RECOMMENDED)

A lightweight interface that PrefillProvider implementations can use to populate cache during execution.

```cpp
/**
 * @file kv_cache_provider.h
 * @brief Interface for KV cache management in distributed prefill execution
 * @author David Sanftenberg
 */

#pragma once

#include "tensors/tensor_base.h"
#include <memory>
#include <vector>

namespace llaminar {

/**
 * @brief Interface for providing KV cache during prefill execution
 *
 * This interface allows PrefillProvider implementations to populate and expose
 * KV cache in a format compatible with the decode path. Each MPI rank owns a
 * subset of KV heads (head parallelism).
 *
 * Cache Layout Per Rank:
 * - K cache: [n_layers][seq_len, local_kv_head_dim]
 * - V cache: [n_layers][seq_len, local_kv_head_dim]
 * - local_kv_head_dim = (n_head_kv / world_size) * head_dim
 *
 * Lifetime:
 * - Cache is populated during PrefillProvider::execute()
 * - Pipeline retrieves cache after prefill completes
 * - Pipeline stores in QwenPipeline::k_cache_ / v_cache_ for decode
 *
 * Thread Safety:
 * - Not thread-safe: single-threaded access assumed
 * - MPI-safe: each rank owns independent cache partition
 */
class KVCacheProvider {
public:
    virtual ~KVCacheProvider() = default;

    /**
     * @brief Get K cache for all layers
     * @return Vector of K cache tensors, one per layer
     * @note Each tensor shape: [seq_len, local_kv_head_dim]
     */
    virtual const std::vector<std::shared_ptr<TensorBase>>& getKCache() const = 0;

    /**
     * @brief Get V cache for all layers
     * @return Vector of V cache tensors, one per layer
     * @note Each tensor shape: [seq_len, local_kv_head_dim]
     */
    virtual const std::vector<std::shared_ptr<TensorBase>>& getVCache() const = 0;

    /**
     * @brief Set K cache for a specific layer (used by provider during execution)
     * @param layer_idx Layer index (0-based)
     * @param k_cache K cache tensor for this layer
     */
    virtual void setKCache(int layer_idx, std::shared_ptr<TensorBase> k_cache) = 0;

    /**
     * @brief Set V cache for a specific layer (used by provider during execution)
     * @param layer_idx Layer index (0-based)
     * @param v_cache V cache tensor for this layer
     */
    virtual void setVCache(int layer_idx, std::shared_ptr<TensorBase> v_cache) = 0;

    /**
     * @brief Reserve cache capacity for given sequence length and layer count
     * @param n_layers Number of transformer layers
     * @param seq_len Sequence length (tokens in prefill)
     * @param kv_head_dim Dimensionality per KV head for this rank
     */
    virtual void reserve(int n_layers, int seq_len, int kv_head_dim) = 0;

    /**
     * @brief Clear all cached tensors (for reset/cleanup)
     */
    virtual void clear() = 0;

    /**
     * @brief Get number of layers with populated cache
     * @return Count of layers with valid cache
     */
    virtual int size() const = 0;

    /**
     * @brief Check if cache is populated for given layer
     * @param layer_idx Layer index to check
     * @return true if cache exists for this layer
     */
    virtual bool hasCache(int layer_idx) const = 0;
};

/**
 * @brief Simple vector-based implementation of KVCacheProvider
 *
 * Stores cache as vectors of shared pointers. Suitable for most use cases.
 */
class SimpleKVCacheProvider : public KVCacheProvider {
public:
    SimpleKVCacheProvider() = default;

    const std::vector<std::shared_ptr<TensorBase>>& getKCache() const override {
        return k_cache_;
    }

    const std::vector<std::shared_ptr<TensorBase>>& getVCache() const override {
        return v_cache_;
    }

    void setKCache(int layer_idx, std::shared_ptr<TensorBase> k_cache) override {
        if (layer_idx >= static_cast<int>(k_cache_.size())) {
            k_cache_.resize(layer_idx + 1);
        }
        k_cache_[layer_idx] = std::move(k_cache);
    }

    void setVCache(int layer_idx, std::shared_ptr<TensorBase> v_cache) override {
        if (layer_idx >= static_cast<int>(v_cache_.size())) {
            v_cache_.resize(layer_idx + 1);
        }
        v_cache_[layer_idx] = std::move(v_cache);
    }

    void reserve(int n_layers, int /*seq_len*/, int /*kv_head_dim*/) override {
        k_cache_.reserve(n_layers);
        v_cache_.reserve(n_layers);
    }

    void clear() override {
        k_cache_.clear();
        v_cache_.clear();
    }

    int size() const override {
        return static_cast<int>(std::min(k_cache_.size(), v_cache_.size()));
    }

    bool hasCache(int layer_idx) const override {
        return layer_idx >= 0 &&
               layer_idx < static_cast<int>(k_cache_.size()) &&
               k_cache_[layer_idx] != nullptr &&
               layer_idx < static_cast<int>(v_cache_.size()) &&
               v_cache_[layer_idx] != nullptr;
    }

private:
    std::vector<std::shared_ptr<TensorBase>> k_cache_;
    std::vector<std::shared_ptr<TensorBase>> v_cache_;
};

} // namespace llaminar
```

### Integration Changes

#### 1. PrefillProvider Modification

```cpp
// In prefill_provider.h - Add cache provider parameter
class PrefillProvider {
public:
    virtual bool execute(
        const std::vector<int>& tokens,
        const IModelWeights& weights,
        std::shared_ptr<TensorBase>& output,
        StageContext& ctx,
        PrefillMetrics& metrics,
        KVCacheProvider* cache_provider = nullptr  // NEW: Optional cache output
    ) = 0;
    
    // ... rest of interface unchanged
};
```

#### 2. Implementation in Concrete Providers

```cpp
// In openblas_prefill_provider.cpp (example)
bool OpenBLASPrefillProvider::execute(
    const std::vector<int>& tokens,
    const IModelWeights& weights,
    std::shared_ptr<TensorBase>& output,
    StageContext& ctx,
    PrefillMetrics& metrics,
    KVCacheProvider* cache_provider)
{
    // ... existing embedding, layer loop setup ...
    
    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        // ... attention computation that produces k_cache, v_cache ...
        
        // NEW: Populate cache provider if given
        if (cache_provider) {
            cache_provider->setKCache(layer_idx, k_cache);
            cache_provider->setVCache(layer_idx, v_cache);
        }
        
        // ... rest of layer processing ...
    }
    
    return true;
}
```

#### 3. QwenPipeline Integration

```cpp
// In qwen_pipeline.cpp - prefill() method
bool QwenPipeline::prefill(const std::vector<int>& tokens,
                           const IModelWeights& weights_iface,
                           StageContext& ctx)
{
    // ... existing setup ...
    
    auto provider = PrefillProviderFactory::create(config_, mpi_ctx_, 
                                                   static_cast<int>(tokens.size()));
    
    // NEW: Create cache provider
    SimpleKVCacheProvider cache_provider;
    const int n_layers = config_.getLayerConfig().n_layers;
    const int kv_head_dim = (config_.getLayerConfig().n_head_kv / getWorldSize()) * 
                            config_.getLayerConfig().head_dim;
    cache_provider.reserve(n_layers, tokens.size(), kv_head_dim);
    
    // Execute prefill with cache capture
    std::shared_ptr<TensorBase> output;
    PrefillMetrics metrics;
    bool success = provider->execute(tokens, weights_iface, output, ctx, metrics, 
                                     &cache_provider);  // NEW: Pass cache provider
    
    if (!success) {
        LOG_ERROR("prefill: Provider execution failed");
        return false;
    }
    
    // NEW: Transfer cache to pipeline storage
    if (use_kv_cache_) {
        const auto& k_caches = cache_provider.getKCache();
        const auto& v_caches = cache_provider.getVCache();
        
        // Initialize cache vectors if needed
        if (k_cache_.empty()) {
            k_cache_.resize(n_layers);
            v_cache_.resize(n_layers);
        }
        
        // Copy cache references (shared_ptr assignment, no data copy)
        for (int i = 0; i < n_layers; ++i) {
            if (cache_provider.hasCache(i)) {
                k_cache_[i] = k_caches[i];
                v_cache_[i] = v_caches[i];
            }
        }
        
        // Update n_past_
        n_past_ = static_cast<int>(tokens.size());
    }
    
    // ... existing metrics logging and return ...
}
```

### Alternative: Option B - Direct Cache Return

Instead of a separate interface, extend PrefillProvider to directly return cache:

```cpp
struct PrefillResult {
    std::shared_ptr<TensorBase> output;           // Logits
    std::vector<std::shared_ptr<TensorBase>> k_cache;  // K cache per layer
    std::vector<std::shared_ptr<TensorBase>> v_cache;  // V cache per layer
    PrefillMetrics metrics;
};

class PrefillProvider {
public:
    virtual bool execute(
        const std::vector<int>& tokens,
        const IModelWeights& weights,
        PrefillResult& result,  // Changed: return struct instead of separate params
        StageContext& ctx
    ) = 0;
};
```

**Pros**: Simpler, no extra interface
**Cons**: Breaks existing API, requires more refactoring

## Recommendation

**Use Option A (KVCacheProvider interface)** for these reasons:

1. **Backward Compatible**: Optional parameter doesn't break existing code
2. **Flexible**: Providers can choose whether to populate cache
3. **Clear Ownership**: Cache provider manages lifetime, pipeline consumes
4. **Extensible**: Easy to add cache metadata or alternate implementations
5. **MPI Agnostic**: Interface doesn't assume MPI details, works per-rank

## Implementation Plan

1. ✅ **Design Review** (this document)
2. **Create kv_cache_provider.h** with interface and SimpleKVCacheProvider
3. **Update PrefillProvider base class** to accept optional cache_provider param
4. **Modify OpenBLASPrefillProvider** to populate cache when provider given
5. **Modify COSMAPrefillProvider** to populate cache when provider given
6. **Update QwenPipeline::prefill()** to create cache provider and transfer results
7. **Test with ParityFramework.TrueIncrementalDecodeVsPyTorch**
8. **Verify K cache non-zero during decode**
9. **Validate attention scores distribution**
10. **Remove debug logging**

## Testing Strategy

### Unit Tests
- `KVCacheProviderTest.BasicOperations` - set/get/clear/reserve
- `KVCacheProviderTest.MultiLayer` - multiple layers with different sizes
- `KVCacheProviderTest.MPI` - per-rank cache sharding (2 ranks)

### Integration Tests
- `PrefillProviderTest.CachePopulation` - verify cache populated after prefill
- `QwenPipelineTest.CacheTransfer` - verify pipeline receives cache from provider
- `AttentionTest.NonZeroCache` - verify decode path sees populated cache

### End-to-End
- `ParityFramework.TrueIncrementalDecodeVsPyTorch` - full generation test
- Verify attention scores: should be [0.19, 0.20, 0.35, 0.14, 0.08, 0.05]
- Verify tokens match PyTorch reference

## Edge Cases

1. **No Cache Provider**: Provider receives `nullptr` → skip cache population (current behavior)
2. **Partial Cache**: Some layers populated, others not → pipeline checks `hasCache()`
3. **MPI Mismatch**: Cache shape validation in pipeline before use
4. **Memory Pressure**: Cache provider uses same allocation as decode path (zero copy)

## Performance Impact

- **Prefill**: Minimal overhead (pointer assignment only)
- **Cache Transfer**: Zero copy (shared_ptr assignment)
- **Decode**: No change (reads same k_cache_/v_cache_ as before)

## Future Extensions

1. **Cache Compression**: Interface can support quantized cache formats
2. **Paged Cache**: Support for block-sparse or paged attention
3. **Multi-Query**: Easy to extend for different attention variants
4. **CPU/GPU**: Interface works for accelerator backends

## Questions for Review

1. Should cache provider be mandatory or optional?
   - **Proposal**: Optional for backward compatibility, but mandatory for AbstractPipeline
2. Should we validate cache shapes in pipeline?
   - **Proposal**: Yes, add assertions in debug builds
3. How to handle cache growth during decode?
   - **Proposal**: Pipeline manages growth, provider only handles prefill

## References

- `changelog/2025-01-27_kv_cache_architecture_issue.md` - Problem analysis
- `src/prefill_provider.h` - Current provider interface
- `src/qwen_pipeline.cpp` - Current prefill/decode implementation
- `src/kernels/MPIAttentionKernel.cpp` - Cache consumption in decode

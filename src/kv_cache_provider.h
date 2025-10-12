/**
 * @file kv_cache_provider.h
 * @brief Interface for KV cache management in distributed prefill execution
 * @author David Sanftenberg
 * @date 2025-01-27
 *
 * This file provides an interface for PrefillProvider implementations to populate
 * and expose KV cache in a format compatible with the decode path. Each MPI rank
 * owns a subset of KV heads (head parallelism).
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
 */

#pragma once

#include "tensors/tensor_base.h"
#include <memory>
#include <vector>

namespace llaminar
{

    /**
     * @brief Interface for providing KV cache during prefill execution
     *
     * This interface allows PrefillProvider implementations to populate and expose
     * KV cache in a format compatible with the decode path. Each MPI rank owns a
     * subset of KV heads (head parallelism).
     *
     * Thread Safety:
     * - Not thread-safe: single-threaded access assumed
     * - MPI-safe: each rank owns independent cache partition
     *
     * Example Usage:
     * @code
     *   SimpleKVCacheProvider cache_provider;
     *   cache_provider.reserve(n_layers, seq_len, local_kv_head_dim);
     *
     *   // Provider populates during execution
     *   provider->execute(tokens, weights, output, ctx, metrics, &cache_provider);
     *
     *   // Pipeline retrieves cache
     *   const auto& k_caches = cache_provider.getKCache();
     *   const auto& v_caches = cache_provider.getVCache();
     *   for (int i = 0; i < n_layers; ++i) {
     *       k_cache_[i] = k_caches[i];
     *       v_cache_[i] = v_caches[i];
     *   }
     * @endcode
     */
    class KVCacheProvider
    {
    public:
        virtual ~KVCacheProvider() = default;

        /**
         * @brief Get K cache for all layers
         * @return Vector of K cache tensors, one per layer
         * @note Each tensor shape: [seq_len, local_kv_head_dim]
         */
        virtual const std::vector<std::shared_ptr<TensorBase>> &getKCache() const = 0;

        /**
         * @brief Get V cache for all layers
         * @return Vector of V cache tensors, one per layer
         * @note Each tensor shape: [seq_len, local_kv_head_dim]
         */
        virtual const std::vector<std::shared_ptr<TensorBase>> &getVCache() const = 0;

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
         * @note This is optional optimization to avoid reallocations during execute
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
     * Memory is managed by the shared_ptr reference counting - no explicit
     * deallocation needed.
     */
    class SimpleKVCacheProvider : public KVCacheProvider
    {
    public:
        SimpleKVCacheProvider() = default;

        const std::vector<std::shared_ptr<TensorBase>> &getKCache() const override
        {
            return k_cache_;
        }

        const std::vector<std::shared_ptr<TensorBase>> &getVCache() const override
        {
            return v_cache_;
        }

        void setKCache(int layer_idx, std::shared_ptr<TensorBase> k_cache) override
        {
            if (layer_idx >= static_cast<int>(k_cache_.size()))
            {
                k_cache_.resize(layer_idx + 1);
            }
            k_cache_[layer_idx] = std::move(k_cache);
        }

        void setVCache(int layer_idx, std::shared_ptr<TensorBase> v_cache) override
        {
            if (layer_idx >= static_cast<int>(v_cache_.size()))
            {
                v_cache_.resize(layer_idx + 1);
            }
            v_cache_[layer_idx] = std::move(v_cache);
        }

        void reserve(int n_layers, int /*seq_len*/, int /*kv_head_dim*/) override
        {
            k_cache_.reserve(n_layers);
            v_cache_.reserve(n_layers);
        }

        void clear() override
        {
            k_cache_.clear();
            v_cache_.clear();
        }

        int size() const override
        {
            return static_cast<int>(std::min(k_cache_.size(), v_cache_.size()));
        }

        bool hasCache(int layer_idx) const override
        {
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

/**
 * @file KVCache.h
 * @brief Typed Key-Value cache for autoregressive decode (single sequence)
 * @author David Sanftenberg
 * @date October 25, 2025
 *
 * Phase 3d: KV cache for incremental decode
 * Stores K/V tensors per layer to avoid recomputing past context.
 *
 * December 2025: Refactored to typed template supporting FP32, BF16, FP16, Q8_1.
 * The cache precision now matches the pipeline's ActivationPrecision by default,
 * avoiding repeated quantization/dequantization cycles.
 */

#pragma once

#include "Tensors.h"
#include "TensorFactory.h"
#include "../utils/MPIContext.h"
#include "../pipelines/PipelineConfig.h"
#include <vector>
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // Precision-to-Tensor Type Mapping
    // =========================================================================

    namespace detail
    {
        /**
         * @brief Map ActivationPrecision to tensor type for KV cache
         */
        template <ActivationPrecision P>
        struct KVCacheTensor;

        template <>
        struct KVCacheTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };

        template <>
        struct KVCacheTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };

        template <>
        struct KVCacheTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };

        template <>
        struct KVCacheTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
        };
    } // namespace detail

    // =========================================================================
    // KVCacheEntry Template
    // =========================================================================

    /**
     * @brief Per-layer KV cache entry with typed tensors
     */
    template <ActivationPrecision Precision>
    struct KVCacheEntry
    {
        using TensorT = typename detail::KVCacheTensor<Precision>::Type;

        std::shared_ptr<TensorT> K; // [past_seq_len, n_kv_heads * head_dim]
        std::shared_ptr<TensorT> V; // [past_seq_len, n_kv_heads * head_dim]
        int cached_tokens = 0;      // Number of tokens currently cached
        int device_idx = -1;        // Device where this layer's cache resides
    };

    // =========================================================================
    // IKVCache Interface (for polymorphic use)
    // =========================================================================

    /**
     * @brief Abstract interface for type-erased KV cache access
     *
     * Enables polymorphic use when precision is determined at runtime.
     */
    class IKVCache
    {
    public:
        virtual ~IKVCache() = default;

        virtual ActivationPrecision precision() const = 0;
        virtual TensorBase *get_k_base(int layer) = 0;
        virtual const TensorBase *get_k_base(int layer) const = 0;
        virtual TensorBase *get_v_base(int layer) = 0;
        virtual const TensorBase *get_v_base(int layer) const = 0;

        /**
         * @brief Get K cache as shared_ptr (polymorphic access)
         * @note Returns shared_ptr to TensorBase; cast to concrete type if needed
         */
        virtual std::shared_ptr<TensorBase> get_k(int layer) const = 0;

        /**
         * @brief Get V cache as shared_ptr (polymorphic access)
         * @note Returns shared_ptr to TensorBase; cast to concrete type if needed
         */
        virtual std::shared_ptr<TensorBase> get_v(int layer) const = 0;

        virtual int get_cached_tokens(int layer) const = 0;

        /**
         * @brief Append K/V to cache (full tensor version - uses tensor shape[0] for token count)
         */
        virtual bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v) = 0;

        /**
         * @brief Append K/V to cache (partial tensor version - explicit token count)
         *
         * Use when the input tensors are larger than the data to be cached.
         * Only the first `num_tokens` rows will be copied.
         *
         * @param layer Layer index
         * @param new_k K tensor [>=num_tokens, kv_dim]
         * @param new_v V tensor [>=num_tokens, kv_dim]
         * @param num_tokens Number of tokens to actually append
         * @return true on success
         */
        virtual bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) = 0;

        virtual void clear() = 0;
        virtual void clear_layer(int layer) = 0;
        virtual void evict_oldest(int tokens_to_evict) = 0;
        virtual int num_layers() const = 0;
        virtual int max_seq_len() const = 0;
        virtual int get_layer_device(int layer) const = 0;
        virtual int get_total_evicted() const = 0;
        virtual void reset_eviction_counter() = 0;
    };

    // =========================================================================
    // KVCache Template Class
    // =========================================================================

    /**
     * @brief Typed KV cache for single-sequence autoregressive decode
     *
     * Stores K/V tensors for each layer to enable incremental decode:
     * - Prefill: Store K/V for all prompt tokens
     * - Decode: Append K/V for new token, use full cached context
     *
     * Template parameter Precision determines the tensor type:
     * - FP32: Standard 32-bit float (default, highest accuracy)
     * - BF16: Brain float16 (Intel AMX optimization)
     * - FP16: Half precision (ARM/GPU optimization)
     * - Q8_1: Block-quantized int8 (memory bandwidth optimization)
     *
     * Design:
     * - Pre-allocates K/V buffers up to max_seq_len
     * - Supports efficient append without reallocation
     * - Thread-safe per-layer access (no concurrent layer access needed)
     */
    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class KVCache : public IKVCache
    {
    public:
        using TensorT = typename detail::KVCacheTensor<Precision>::Type;
        using EntryT = KVCacheEntry<Precision>;

        /**
         * @brief Construct KV cache for all layers
         *
         * @param mpi_ctx MPI context for NUMA-aware allocation
         * @param n_layers Number of transformer layers
         * @param max_seq_len Maximum sequence length (cache capacity)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param device_idx Default device for all layers (-1 = CPU)
         */
        KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len,
                int n_kv_heads, int head_dim, int device_idx = -1);

        /**
         * @brief Construct KV cache with per-layer attention device placement
         *
         * @param mpi_ctx MPI context for NUMA-aware allocation
         * @param n_layers Number of transformer layers
         * @param max_seq_len Maximum sequence length (cache capacity)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param attention_devices Device where attention is computed per layer
         */
        KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len,
                int n_kv_heads, int head_dim, const std::vector<int> &attention_devices);

        // IKVCache interface implementation
        ActivationPrecision precision() const override { return Precision; }

        TensorBase *get_k_base(int layer) override;
        const TensorBase *get_k_base(int layer) const override;
        TensorBase *get_v_base(int layer) override;
        const TensorBase *get_v_base(int layer) const override;

        // Polymorphic accessors (returns base type)
        std::shared_ptr<TensorBase> get_k(int layer) const override;
        std::shared_ptr<TensorBase> get_v(int layer) const override;

        int get_cached_tokens(int layer) const override;
        bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v) override;
        bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) override;

        void clear() override;
        void clear_layer(int layer) override;
        void evict_oldest(int tokens_to_evict) override;

        int num_layers() const override { return n_layers_; }
        int max_seq_len() const override { return max_seq_len_; }
        int get_layer_device(int layer) const override;
        int get_total_evicted() const override { return total_evicted_; }
        void reset_eviction_counter() override { total_evicted_ = 0; }

        // =====================================================================
        // Typed accessors (for direct use when precision is known at compile time)
        // =====================================================================

        /**
         * @brief Get typed K cache for specific layer (compile-time precision)
         *
         * @param layer Layer index (0-based)
         * @return K tensor [cached_tokens, n_kv_heads * head_dim], or nullptr if empty
         */
        std::shared_ptr<TensorT> get_k_typed(int layer) const;

        /**
         * @brief Get typed V cache for specific layer (compile-time precision)
         *
         * @param layer Layer index (0-based)
         * @return V tensor [cached_tokens, n_kv_heads * head_dim], or nullptr if empty
         */
        std::shared_ptr<TensorT> get_v_typed(int layer) const;

        /**
         * @brief Append new K/V to cache (typed version - zero-copy for matching precision)
         *
         * @param layer Layer index
         * @param new_k New K tensor [new_seq_len, n_kv_heads * head_dim]
         * @param new_v New V tensor [new_seq_len, n_kv_heads * head_dim]
         * @return true on success, false if capacity exceeded
         */
        bool append_kv_typed(int layer, const TensorT *new_k, const TensorT *new_v);

        /**
         * @brief Get attention device for a specific layer (alias)
         */
        int get_attention_device(int layer) const
        {
            return get_layer_device(layer);
        }

    private:
        int n_layers_;
        int max_seq_len_;
        int n_kv_heads_;
        int head_dim_;
        int device_idx_;
        int total_evicted_ = 0;

        std::vector<EntryT> cache_;

        // Helper to allocate typed tensor
        std::shared_ptr<TensorT> allocate_tensor(TensorFactory &factory, size_t rows, size_t cols, int device_idx);

        // Helper to copy data for append operation (specialized per precision)
        void copy_append_data(TensorT *dst, const TensorT *src, int offset_tokens, int new_tokens, int kv_dim);

        // Helper to shift data for eviction (specialized per precision)
        void shift_evict_data(TensorT *tensor, int tokens_to_evict, int tokens_to_keep, int kv_dim);

        // Internal implementation for append (handles both full-tensor and partial-tensor cases)
        bool append_kv_impl(int layer, const TensorT *new_k, const TensorT *new_v, int num_tokens);
    };

    // =========================================================================
    // Type Aliases for Common Precisions
    // =========================================================================

    using KVCacheFP32 = KVCache<ActivationPrecision::FP32>;
    using KVCacheBF16 = KVCache<ActivationPrecision::BF16>;
    using KVCacheFP16 = KVCache<ActivationPrecision::FP16>;
    using KVCacheQ8_1 = KVCache<ActivationPrecision::Q8_1>;

    // =========================================================================
    // Factory Functions for Runtime Precision Selection
    // =========================================================================

    /**
     * @brief Create KV cache with specified activation precision (single device)
     *
     * @param precision Activation precision for cache storage
     * @param mpi_ctx MPI context for NUMA-aware allocation
     * @param n_layers Number of transformer layers
     * @param max_seq_len Maximum sequence length
     * @param n_kv_heads Number of KV heads
     * @param head_dim Dimension per head
     * @param device_idx Device index (-1 = CPU)
     * @return Unique pointer to IKVCache implementation
     */
    std::unique_ptr<IKVCache> createKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int max_seq_len,
        int n_kv_heads, int head_dim,
        int device_idx = -1);

    /**
     * @brief Create KV cache with per-layer device placement
     */
    std::unique_ptr<IKVCache> createKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices);

} // namespace llaminar2

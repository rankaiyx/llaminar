/**
 * @file CPUKVCache.h
 * @brief Shared CPU KV cache interface and precision→tensor mappings.
 */

#pragma once

#include "../IKVCache.h" // Unified KVCache interface
#include "../../tensors/Tensors.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/TensorLayout.h"
#include "../../backends/DeviceId.h"
#include "../../utils/MPIContext.h"
#include "../../execution/config/RuntimeConfig.h"
#include <vector>
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // KV Cache Layout Mode
    // =========================================================================

    /**
     * @brief Memory layout mode for KV cache storage
     *
     * Controls how K/V tensors are organized in memory:
     *
     * POSITION_MAJOR (default):
     *   Storage: [position][n_kv_heads][head_dim]
     *   Block indexing: block[p * n_kv_heads + h]
     *   Best for: Sequential cache append (new positions at end)
     *   Used by: FP32/BF16/FP16 attention backends
     *
     * HEAD_MAJOR:
     *   Storage: [n_kv_heads][position][head_dim]
     *   Block indexing: block[h * kv_len + p]
     *   Best for: Per-head attention computation (head-contiguous access)
     *   Used by: Q16_INTEGER attention kernel
     *
     * @note Choosing HEAD_MAJOR for Q16_1 caches eliminates the
     *       transpose workaround in FusedAttentionWoStage.
     */
    enum class KVCacheLayoutMode : uint8_t
    {
        POSITION_MAJOR, ///< [position][n_kv_heads][head_dim] - cache-append friendly
        HEAD_MAJOR      ///< [n_kv_heads][position][head_dim] - attention-compute friendly
    };

    // =========================================================================
    // ICPUKVCache Interface
    // =========================================================================

    /**
     * @brief Abstract interface for type-erased unified KV cache access
     *
     * Enables polymorphic use when precision is determined at runtime.
     * Supports both single-sequence (batch_size=1) and batched modes.
     *
     * Inherits from IKVCache to provide unified CPU/GPU interface.
     */
    class ICPUKVCache : public IKVCache
    {
    public:
        virtual ~ICPUKVCache() = default;

        // =====================================================================
        // IKVCache Interface (public API)
        // =====================================================================

        // Metadata (IKVCache)
        virtual ActivationPrecision precision() const = 0;
        int n_layers() const override { return num_layers(); }
        virtual int max_seq_len() const = 0;
        virtual TensorLayout kv_layout() const = 0;

        // Per-sequence token tracking (IKVCache)
        virtual int get_cached_tokens(int layer, int seq_idx = 0) const = 0;

        // Sharding (Tensor Parallelism) info (IKVCache)
        virtual bool is_sharded() const = 0;      ///< True if using local KV heads
        virtual int local_n_kv_heads() const = 0; ///< Local KV heads (this rank)
        virtual int local_kv_dim() const = 0;     ///< local_n_kv_heads * head_dim
        virtual int kv_head_start() const = 0;    ///< Starting KV head index

        // =====================================================================
        // CPU-Specific Methods (exposed for testing and internal use)
        // =====================================================================

        virtual int num_layers() const = 0;
        virtual int batch_size() const = 0;
        virtual KVCacheLayoutMode layout_mode() const = 0;
        virtual int n_kv_heads() const = 0; ///< Total KV heads (across all ranks)

    public:
        // =================================================================
        // Unified KV Access (preferred interface)
        // =================================================================

        /**
         * @brief Get both K and V cache tensors for attention computation
         *
         * This is the preferred interface - fetches both K and V in a single call.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @param out_k Output: pointer to K cache tensor
         * @param out_v Output: pointer to V cache tensor
         * @param out_kv_len Output: number of cached tokens (optional)
         * @return true on success
         */
        virtual bool get_kv(int layer, int seq_idx,
                            ITensor **out_k, ITensor **out_v,
                            int *out_kv_len = nullptr) = 0;

        virtual bool get_kv(int layer, int seq_idx,
                            const ITensor **out_k, const ITensor **out_v,
                            int *out_kv_len = nullptr) const = 0;

        // Convenience overloads for seq_idx=0
        bool get_kv(int layer, ITensor **out_k, ITensor **out_v, int *out_kv_len = nullptr)
        {
            return get_kv(layer, 0, out_k, out_v, out_kv_len);
        }

        bool get_kv(int layer, const ITensor **out_k, const ITensor **out_v, int *out_kv_len = nullptr) const
        {
            return get_kv(layer, 0, out_k, out_v, out_kv_len);
        }

        // =================================================================
        // Legacy tensor access (deprecated - use get_kv() instead)
        // =================================================================

        /// @deprecated Use get_kv() instead
        ITensor *get_k(int layer, int seq_idx = 0) override = 0;
        /// @deprecated Use get_kv() instead
        const ITensor *get_k(int layer, int seq_idx = 0) const override = 0;
        /// @deprecated Use get_kv() instead
        ITensor *get_v(int layer, int seq_idx = 0) override = 0;
        /// @deprecated Use get_kv() instead
        const ITensor *get_v(int layer, int seq_idx = 0) const override = 0;

        // =================================================================
        // IKVCache interface bridging
        // =================================================================

        // Bridge IKVCache::append(ITensor*) to ICPUKVCache::append_kv(TensorBase*)
        // dynamic_cast required due to virtual inheritance in ITensor hierarchy
        bool append(int layer, int seq_idx, const ITensor *K, const ITensor *V, int num_tokens) override
        {
            return append_kv(layer, seq_idx,
                             dynamic_cast<const TensorBase *>(K),
                             dynamic_cast<const TensorBase *>(V),
                             num_tokens);
        }

        // Bring in IKVCache convenience overloads
        using IKVCache::append;

        // IKVCache::gather_kv_batched bridge (uses ITensor*, delegates to TensorBase* version)
        int gather_kv_batched(
            int layer,
            int num_sequences,
            ITensor *out_k,
            ITensor *out_v,
            std::vector<int> &out_kv_lens) override
        {
            // dynamic_cast required due to virtual inheritance in ITensor hierarchy
            return gather_kv_batched(layer, num_sequences,
                                     dynamic_cast<TensorBase *>(out_k),
                                     dynamic_cast<TensorBase *>(out_v),
                                     out_kv_lens);
        }

        // Cache management (IKVCache)
        void clear() override = 0;
        void clear_sequence(int layer, int seq_idx) override = 0;
        void clear_layer(int layer) override = 0;

        // Bring in IKVCache::clear_sequence(seq_idx) default implementation
        using IKVCache::clear_sequence;

        // =================================================================
        // CPU-Specific Methods (for testing and internal use)
        // =================================================================

        /**
         * @brief Append K/V to cache for a sequence (TensorBase version)
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0 for single-sequence mode)
         * @param new_k K tensor to append
         * @param new_v V tensor to append
         * @return true on success, false if capacity exceeded
         */
        virtual bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) = 0;

        /**
         * @brief Append K/V with explicit token count (partial tensor)
         */
        virtual bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) = 0;

        // Convenience for single-sequence mode (seq_idx = 0)
        bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v)
        {
            return append_kv(layer, 0, new_k, new_v);
        }

        bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v, int num_tokens)
        {
            return append_kv(layer, 0, new_k, new_v, num_tokens);
        }

        /**
         * @brief Gather K/V from multiple cache slots into batched output tensors (TensorBase version)
         *
         * @param layer Layer index
         * @param num_sequences Number of sequences to gather (typically batch_size)
         * @param out_k Output K tensor [num_sequences * max_kv_len, kv_dim]
         * @param out_v Output V tensor [num_sequences * max_kv_len, kv_dim]
         * @param out_kv_lens Per-sequence kv_lens (output, size = num_sequences)
         * @return Maximum kv_len across all sequences, or -1 on error
         */
        virtual int gather_kv_batched(
            int layer,
            int num_sequences,
            TensorBase *out_k,
            TensorBase *out_v,
            std::vector<int> &out_kv_lens) = 0;

        // Eviction operations
        virtual void evict_oldest(int tokens_to_evict) = 0;
        virtual void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) = 0;

        // Device placement
        virtual DeviceId get_layer_device(int layer) const = 0;

        // Eviction tracking
        virtual int get_total_evicted() const = 0;
        virtual void reset_eviction_counter() = 0;
    };

    // =========================================================================
    // Forward Declaration
    // =========================================================================

    namespace detail
    {
        /**
         * @brief Map ActivationPrecision to tensor type for KV cache
         */
        template <ActivationPrecision P>
        struct CPUKVCacheTensor;

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::Q16_1>
        {
            using Type = Q16_1Tensor;
        };
    } // namespace detail

} // namespace llaminar2

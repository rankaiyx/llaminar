/**
 * @file KVCache.cpp
 * @brief Typed KV cache implementation
 * @author David Sanftenberg
 */

#include "KVCache.h"
#include "../utils/Logger.h"
#include "SIMDHelpers.h"
#include <cstring>
#include <map>

namespace llaminar2
{

    // =========================================================================
    // Helper: Tensor Allocation (specialized per precision)
    // Note: TensorFactory returns unique_ptr, we convert to shared_ptr.
    // device_idx is for tracking, TensorFactory uses MPI context for NUMA.
    // =========================================================================

    template <>
    std::shared_ptr<FP32Tensor> KVCache<ActivationPrecision::FP32>::allocate_tensor(
        TensorFactory &factory, size_t rows, size_t cols, int device_idx)
    {
        (void)device_idx; // Used for tracking, factory handles NUMA via MPI context
        return factory.createFP32({rows, cols}, device_idx);
    }

    template <>
    std::shared_ptr<BF16Tensor> KVCache<ActivationPrecision::BF16>::allocate_tensor(
        TensorFactory &factory, size_t rows, size_t cols, int /* device_idx */)
    {
        return factory.createBF16({rows, cols});
    }

    template <>
    std::shared_ptr<FP16Tensor> KVCache<ActivationPrecision::FP16>::allocate_tensor(
        TensorFactory &factory, size_t rows, size_t cols, int /* device_idx */)
    {
        return factory.createFP16({rows, cols});
    }

    template <>
    std::shared_ptr<Q8_1Tensor> KVCache<ActivationPrecision::Q8_1>::allocate_tensor(
        TensorFactory &factory, size_t rows, size_t cols, int /* device_idx */)
    {
        return factory.createQ8_1({rows, cols});
    }

    // =========================================================================
    // Helper: Copy data for append (specialized per precision)
    // =========================================================================

    template <>
    void KVCache<ActivationPrecision::FP32>::copy_append_data(
        FP32Tensor *dst, const FP32Tensor *src, int offset_tokens, int new_tokens, int kv_dim)
    {
        float *dst_data = dst->mutable_data();
        const float *src_data = src->data();
        size_t offset = static_cast<size_t>(offset_tokens) * kv_dim;
        size_t copy_size = static_cast<size_t>(new_tokens) * kv_dim * sizeof(float);
        std::memcpy(dst_data + offset, src_data, copy_size);
    }

    template <>
    void KVCache<ActivationPrecision::BF16>::copy_append_data(
        BF16Tensor *dst, const BF16Tensor *src, int offset_tokens, int new_tokens, int kv_dim)
    {
        uint16_t *dst_data = dst->mutable_bf16_data();
        const uint16_t *src_data = src->bf16_data();
        size_t offset = static_cast<size_t>(offset_tokens) * kv_dim;
        size_t copy_size = static_cast<size_t>(new_tokens) * kv_dim * sizeof(uint16_t);
        std::memcpy(dst_data + offset, src_data, copy_size);
    }

    template <>
    void KVCache<ActivationPrecision::FP16>::copy_append_data(
        FP16Tensor *dst, const FP16Tensor *src, int offset_tokens, int new_tokens, int kv_dim)
    {
        uint16_t *dst_data = dst->mutable_fp16_data();
        const uint16_t *src_data = src->fp16_data();
        size_t offset = static_cast<size_t>(offset_tokens) * kv_dim;
        size_t copy_size = static_cast<size_t>(new_tokens) * kv_dim * sizeof(uint16_t);
        std::memcpy(dst_data + offset, src_data, copy_size);
    }

    template <>
    void KVCache<ActivationPrecision::Q8_1>::copy_append_data(
        Q8_1Tensor *dst, const Q8_1Tensor *src, int offset_tokens, int new_tokens, int kv_dim)
    {
        // Q8_1: Copy blocks directly
        Q8_1Block *dst_blocks = dst->mutable_q8_1_blocks();
        const Q8_1Block *src_blocks = src->q8_1_blocks();

        // Calculate blocks per row
        size_t blocks_per_row = (kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        size_t offset_blocks = static_cast<size_t>(offset_tokens) * blocks_per_row;
        size_t copy_blocks = static_cast<size_t>(new_tokens) * blocks_per_row;

        std::memcpy(dst_blocks + offset_blocks, src_blocks, copy_blocks * sizeof(Q8_1Block));
    }

    // =========================================================================
    // Helper: Shift data for eviction (specialized per precision)
    // =========================================================================

    template <>
    void KVCache<ActivationPrecision::FP32>::shift_evict_data(
        FP32Tensor *tensor, int tokens_to_evict, int tokens_to_keep, int kv_dim)
    {
        float *data = tensor->mutable_data();
        size_t shift_offset = static_cast<size_t>(tokens_to_evict) * kv_dim;
        size_t keep_size = static_cast<size_t>(tokens_to_keep) * kv_dim * sizeof(float);
        std::memmove(data, data + shift_offset, keep_size);
    }

    template <>
    void KVCache<ActivationPrecision::BF16>::shift_evict_data(
        BF16Tensor *tensor, int tokens_to_evict, int tokens_to_keep, int kv_dim)
    {
        uint16_t *data = tensor->mutable_bf16_data();
        size_t shift_offset = static_cast<size_t>(tokens_to_evict) * kv_dim;
        size_t keep_size = static_cast<size_t>(tokens_to_keep) * kv_dim * sizeof(uint16_t);
        std::memmove(data, data + shift_offset, keep_size);
    }

    template <>
    void KVCache<ActivationPrecision::FP16>::shift_evict_data(
        FP16Tensor *tensor, int tokens_to_evict, int tokens_to_keep, int kv_dim)
    {
        uint16_t *data = tensor->mutable_fp16_data();
        size_t shift_offset = static_cast<size_t>(tokens_to_evict) * kv_dim;
        size_t keep_size = static_cast<size_t>(tokens_to_keep) * kv_dim * sizeof(uint16_t);
        std::memmove(data, data + shift_offset, keep_size);
    }

    template <>
    void KVCache<ActivationPrecision::Q8_1>::shift_evict_data(
        Q8_1Tensor *tensor, int tokens_to_evict, int tokens_to_keep, int kv_dim)
    {
        Q8_1Block *blocks = tensor->mutable_q8_1_blocks();
        size_t blocks_per_row = (kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        size_t shift_blocks = static_cast<size_t>(tokens_to_evict) * blocks_per_row;
        size_t keep_blocks = static_cast<size_t>(tokens_to_keep) * blocks_per_row;
        std::memmove(blocks, blocks + shift_blocks, keep_blocks * sizeof(Q8_1Block));
    }

    // =========================================================================
    // Constructor: Single device
    // =========================================================================

    template <ActivationPrecision Precision>
    KVCache<Precision>::KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len,
                                int n_kv_heads, int head_dim, int device_idx)
        : n_layers_(n_layers), max_seq_len_(max_seq_len), n_kv_heads_(n_kv_heads),
          head_dim_(head_dim), device_idx_(device_idx)
    {
        cache_.resize(n_layers);

        TensorFactory factory(mpi_ctx);
        size_t kv_dim = static_cast<size_t>(n_kv_heads) * head_dim;

        for (int i = 0; i < n_layers; ++i)
        {
            cache_[i].K = allocate_tensor(factory, static_cast<size_t>(max_seq_len), kv_dim, device_idx);
            cache_[i].V = allocate_tensor(factory, static_cast<size_t>(max_seq_len), kv_dim, device_idx);
            cache_[i].cached_tokens = 0;
            cache_[i].device_idx = device_idx;
        }

        // Calculate memory usage based on precision
        size_t bytes_per_element = 4; // FP32 default
        if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
        {
            bytes_per_element = 2;
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            // Q8_1: 36 bytes per 32 elements
            size_t total_elements = 2 * n_layers * max_seq_len * kv_dim;
            size_t total_blocks = (total_elements + 31) / 32;
            size_t total_bytes = total_blocks * sizeof(Q8_1Block);
            LOG_INFO("[KVCache<Q8_1>] Allocated cache for " << n_layers << " layers, max_seq_len=" << max_seq_len
                                                            << ", kv_dim=" << kv_dim << " on device " << device_idx
                                                            << " (" << (total_bytes / 1024 / 1024) << " MB)");
            return;
        }

        size_t total_bytes = 2 * n_layers * max_seq_len * kv_dim * bytes_per_element;
        const char *precision_name = (Precision == ActivationPrecision::FP32)   ? "FP32"
                                     : (Precision == ActivationPrecision::BF16) ? "BF16"
                                     : (Precision == ActivationPrecision::FP16) ? "FP16"
                                                                                : "Q8_1";
        LOG_INFO("[KVCache<" << precision_name << ">] Allocated cache for " << n_layers << " layers, max_seq_len=" << max_seq_len
                             << ", kv_dim=" << kv_dim << " on device " << device_idx
                             << " (" << (total_bytes / 1024 / 1024) << " MB)");
    }

    // =========================================================================
    // Constructor: Per-layer device placement
    // =========================================================================

    template <ActivationPrecision Precision>
    KVCache<Precision>::KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len,
                                int n_kv_heads, int head_dim, const std::vector<int> &layer_devices)
        : n_layers_(n_layers), max_seq_len_(max_seq_len), n_kv_heads_(n_kv_heads),
          head_dim_(head_dim), device_idx_(-2) // -2 = heterogeneous
    {
        if (static_cast<int>(layer_devices.size()) != n_layers)
        {
            LOG_ERROR("[KVCache] layer_devices size (" << layer_devices.size() << ") must match n_layers (" << n_layers << ")");
            throw std::invalid_argument("layer_devices size mismatch");
        }

        cache_.resize(n_layers);
        size_t kv_dim = static_cast<size_t>(n_kv_heads) * head_dim;

        TensorFactory factory(mpi_ctx);
        std::map<int, size_t> device_bytes;

        // Calculate bytes per element
        size_t bytes_per_element = 4;
        if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
        {
            bytes_per_element = 2;
        }

        for (int i = 0; i < n_layers; ++i)
        {
            int layer_device = layer_devices[i];
            cache_[i].K = allocate_tensor(factory, static_cast<size_t>(max_seq_len), kv_dim, layer_device);
            cache_[i].V = allocate_tensor(factory, static_cast<size_t>(max_seq_len), kv_dim, layer_device);
            cache_[i].cached_tokens = 0;
            cache_[i].device_idx = layer_device;

            if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                size_t elements = 2 * max_seq_len * kv_dim;
                size_t blocks = (elements + 31) / 32;
                device_bytes[layer_device] += blocks * sizeof(Q8_1Block);
            }
            else
            {
                device_bytes[layer_device] += 2 * max_seq_len * kv_dim * bytes_per_element;
            }
        }

        const char *precision_name = (Precision == ActivationPrecision::FP32)   ? "FP32"
                                     : (Precision == ActivationPrecision::BF16) ? "BF16"
                                     : (Precision == ActivationPrecision::FP16) ? "FP16"
                                                                                : "Q8_1";
        LOG_INFO("[KVCache<" << precision_name << ">] Heterogeneous allocation for " << n_layers << " layers:");
        for (const auto &[device, bytes] : device_bytes)
        {
            double mb = bytes / (1024.0 * 1024.0);
            std::string device_name = (device == -1) ? "CPU" : "GPU " + std::to_string(device);
            LOG_INFO("  " << device_name << ": " << mb << " MB");
        }
    }

    // =========================================================================
    // Typed accessors (compile-time precision)
    // =========================================================================

    template <ActivationPrecision Precision>
    std::shared_ptr<typename KVCache<Precision>::TensorT> KVCache<Precision>::get_k_typed(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("[KVCache] Invalid layer index: " << layer);
            return nullptr;
        }
        if (cache_[layer].cached_tokens == 0)
        {
            return nullptr;
        }
        return cache_[layer].K;
    }

    template <ActivationPrecision Precision>
    std::shared_ptr<typename KVCache<Precision>::TensorT> KVCache<Precision>::get_v_typed(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("[KVCache] Invalid layer index: " << layer);
            return nullptr;
        }
        if (cache_[layer].cached_tokens == 0)
        {
            return nullptr;
        }
        return cache_[layer].V;
    }

    // =========================================================================
    // Polymorphic accessors (IKVCache interface - returns shared_ptr<TensorBase>)
    // =========================================================================

    template <ActivationPrecision Precision>
    std::shared_ptr<TensorBase> KVCache<Precision>::get_k(int layer) const
    {
        return get_k_typed(layer); // shared_ptr<TensorT> converts to shared_ptr<TensorBase>
    }

    template <ActivationPrecision Precision>
    std::shared_ptr<TensorBase> KVCache<Precision>::get_v(int layer) const
    {
        return get_v_typed(layer); // shared_ptr<TensorT> converts to shared_ptr<TensorBase>
    }

    // =========================================================================
    // TensorBase accessors (for IKVCache interface)
    // =========================================================================

    template <ActivationPrecision Precision>
    TensorBase *KVCache<Precision>::get_k_base(int layer)
    {
        if (layer < 0 || layer >= n_layers_ || cache_[layer].cached_tokens == 0)
        {
            return nullptr;
        }
        return cache_[layer].K.get();
    }

    template <ActivationPrecision Precision>
    const TensorBase *KVCache<Precision>::get_k_base(int layer) const
    {
        if (layer < 0 || layer >= n_layers_ || cache_[layer].cached_tokens == 0)
        {
            return nullptr;
        }
        return cache_[layer].K.get();
    }

    template <ActivationPrecision Precision>
    TensorBase *KVCache<Precision>::get_v_base(int layer)
    {
        if (layer < 0 || layer >= n_layers_ || cache_[layer].cached_tokens == 0)
        {
            return nullptr;
        }
        return cache_[layer].V.get();
    }

    template <ActivationPrecision Precision>
    const TensorBase *KVCache<Precision>::get_v_base(int layer) const
    {
        if (layer < 0 || layer >= n_layers_ || cache_[layer].cached_tokens == 0)
        {
            return nullptr;
        }
        return cache_[layer].V.get();
    }

    template <ActivationPrecision Precision>
    int KVCache<Precision>::get_cached_tokens(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return 0;
        }
        return cache_[layer].cached_tokens;
    }

    template <ActivationPrecision Precision>
    int KVCache<Precision>::get_layer_device(int layer) const
    {
        if (layer >= 0 && layer < n_layers_)
        {
            return cache_[layer].device_idx;
        }
        return -1;
    }

    // =========================================================================
    // Append K/V (internal implementation)
    // =========================================================================

    template <ActivationPrecision Precision>
    bool KVCache<Precision>::append_kv_impl(int layer, const TensorT *new_k, const TensorT *new_v, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("[KVCache] Invalid layer index: " << layer);
            return false;
        }

        if (!new_k || !new_v)
        {
            LOG_ERROR("[KVCache] Null K or V tensor");
            return false;
        }

        const auto &new_k_shape = new_k->shape();
        const auto &new_v_shape = new_v->shape();

        if (new_k_shape.size() != 2 || new_v_shape.size() != 2)
        {
            LOG_ERROR("[KVCache] K/V must be 2D tensors");
            return false;
        }

        int kv_dim = static_cast<int>(new_k_shape[1]);

        // Validate num_tokens fits in the source tensors
        if (num_tokens > static_cast<int>(new_k_shape[0]) || num_tokens > static_cast<int>(new_v_shape[0]))
        {
            LOG_ERROR("[KVCache] num_tokens (" << num_tokens << ") exceeds tensor size (K: "
                                               << new_k_shape[0] << ", V: " << new_v_shape[0] << ")");
            return false;
        }

        if (new_k_shape[1] != new_v_shape[1])
        {
            LOG_ERROR("[KVCache] K/V dimension mismatch");
            return false;
        }

        int current_cached = cache_[layer].cached_tokens;

        // Auto-evict oldest tokens if cache would overflow
        if (current_cached + num_tokens > max_seq_len_)
        {
            int tokens_to_evict = (current_cached + num_tokens) - max_seq_len_;
            if (layer == 0)
            {
                if (total_evicted_ == 0)
                {
                    LOG_DEBUG("[KVCache] Cache capacity reached (" << max_seq_len_ << " tokens). "
                                                                   << "Starting sliding window eviction.");
                }
                LOG_TRACE("[KVCache] Cache full, evicting " << tokens_to_evict << " oldest tokens");
                evict_oldest(tokens_to_evict);
            }
            current_cached = cache_[layer].cached_tokens;
        }

        // Copy new K/V into cache buffers at offset
        copy_append_data(cache_[layer].K.get(), new_k, current_cached, num_tokens, kv_dim);
        copy_append_data(cache_[layer].V.get(), new_v, current_cached, num_tokens, kv_dim);

        cache_[layer].cached_tokens += num_tokens;

        LOG_TRACE("[KVCache] Layer " << layer << " appended " << num_tokens << " tokens (total: " << cache_[layer].cached_tokens << ")");

        return true;
    }

    // =========================================================================
    // Append K/V (typed version - full tensor)
    // =========================================================================

    template <ActivationPrecision Precision>
    bool KVCache<Precision>::append_kv_typed(int layer, const TensorT *new_k, const TensorT *new_v)
    {
        if (!new_k)
            return false;
        int num_tokens = static_cast<int>(new_k->shape()[0]);
        return append_kv_impl(layer, new_k, new_v, num_tokens);
    }

    // =========================================================================
    // Append K/V (polymorphic version via TensorBase* - full tensor)
    // =========================================================================

    template <ActivationPrecision Precision>
    bool KVCache<Precision>::append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v)
    {
        // Try to dynamic_cast to the correct tensor type
        const TensorT *typed_k = dynamic_cast<const TensorT *>(new_k);
        const TensorT *typed_v = dynamic_cast<const TensorT *>(new_v);

        if (!typed_k || !typed_v)
        {
            LOG_ERROR("[KVCache] Input tensors must match cache precision ("
                      << (Precision == ActivationPrecision::FP32   ? "FP32"
                          : Precision == ActivationPrecision::BF16 ? "BF16"
                          : Precision == ActivationPrecision::FP16 ? "FP16"
                                                                   : "Q8_1")
                      << ")");
            return false;
        }

        return append_kv_typed(layer, typed_k, typed_v);
    }

    // =========================================================================
    // Append K/V (polymorphic version with explicit num_tokens)
    // =========================================================================

    template <ActivationPrecision Precision>
    bool KVCache<Precision>::append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v, int num_tokens)
    {
        // Try to dynamic_cast to the correct tensor type
        const TensorT *typed_k = dynamic_cast<const TensorT *>(new_k);
        const TensorT *typed_v = dynamic_cast<const TensorT *>(new_v);

        if (!typed_k || !typed_v)
        {
            LOG_ERROR("[KVCache] Input tensors must match cache precision ("
                      << (Precision == ActivationPrecision::FP32   ? "FP32"
                          : Precision == ActivationPrecision::BF16 ? "BF16"
                          : Precision == ActivationPrecision::FP16 ? "FP16"
                                                                   : "Q8_1")
                      << ")");
            return false;
        }

        return append_kv_impl(layer, typed_k, typed_v, num_tokens);
    }

    // =========================================================================
    // Clear
    // =========================================================================

    template <ActivationPrecision Precision>
    void KVCache<Precision>::clear()
    {
        for (int i = 0; i < n_layers_; ++i)
        {
            cache_[i].cached_tokens = 0;
        }
        total_evicted_ = 0;
        LOG_DEBUG("[KVCache] Cleared all layers");
    }

    template <ActivationPrecision Precision>
    void KVCache<Precision>::clear_layer(int layer)
    {
        if (layer >= 0 && layer < n_layers_)
        {
            cache_[layer].cached_tokens = 0;
            LOG_DEBUG("[KVCache] Cleared layer " << layer);
        }
    }

    // =========================================================================
    // Evict oldest tokens
    // =========================================================================

    template <ActivationPrecision Precision>
    void KVCache<Precision>::evict_oldest(int tokens_to_evict)
    {
        if (tokens_to_evict <= 0)
        {
            return;
        }

        int kv_dim = n_kv_heads_ * head_dim_;
        total_evicted_ += tokens_to_evict;

#pragma omp parallel for
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            int current_cached = cache_[layer].cached_tokens;
            if (tokens_to_evict >= current_cached)
            {
                cache_[layer].cached_tokens = 0;
                continue;
            }

            int tokens_to_keep = current_cached - tokens_to_evict;
            shift_evict_data(cache_[layer].K.get(), tokens_to_evict, tokens_to_keep, kv_dim);
            shift_evict_data(cache_[layer].V.get(), tokens_to_evict, tokens_to_keep, kv_dim);
            cache_[layer].cached_tokens = tokens_to_keep;
        }

        LOG_TRACE("[KVCache] Evicted " << tokens_to_evict << " oldest tokens from all layers (total_evicted_=" << total_evicted_ << ")");
    }

    // =========================================================================
    // Explicit template instantiations
    // =========================================================================

    template class KVCache<ActivationPrecision::FP32>;
    template class KVCache<ActivationPrecision::BF16>;
    template class KVCache<ActivationPrecision::FP16>;
    template class KVCache<ActivationPrecision::Q8_1>;

    // =========================================================================
    // Factory functions
    // =========================================================================

    std::unique_ptr<IKVCache> createKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int max_seq_len,
        int n_kv_heads, int head_dim,
        int device_idx)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<KVCache<ActivationPrecision::FP32>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, device_idx);
        case ActivationPrecision::BF16:
            return std::make_unique<KVCache<ActivationPrecision::BF16>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, device_idx);
        case ActivationPrecision::FP16:
            return std::make_unique<KVCache<ActivationPrecision::FP16>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, device_idx);
        case ActivationPrecision::Q8_1:
            return std::make_unique<KVCache<ActivationPrecision::Q8_1>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, device_idx);
        default:
            LOG_ERROR("[createKVCache] Unknown precision: " << static_cast<int>(precision));
            return nullptr;
        }
    }

    std::unique_ptr<IKVCache> createKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<KVCache<ActivationPrecision::FP32>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, attention_devices);
        case ActivationPrecision::BF16:
            return std::make_unique<KVCache<ActivationPrecision::BF16>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, attention_devices);
        case ActivationPrecision::FP16:
            return std::make_unique<KVCache<ActivationPrecision::FP16>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, attention_devices);
        case ActivationPrecision::Q8_1:
            return std::make_unique<KVCache<ActivationPrecision::Q8_1>>(
                mpi_ctx, n_layers, max_seq_len, n_kv_heads, head_dim, attention_devices);
        default:
            LOG_ERROR("[createKVCache] Unknown precision: " << static_cast<int>(precision));
            return nullptr;
        }
    }

} // namespace llaminar2

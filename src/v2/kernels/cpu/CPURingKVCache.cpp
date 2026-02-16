#include "CPURingKVCache.h"

#include <algorithm>
#include <cstring>

namespace llaminar2
{

    namespace
    {
        DeviceId deviceIdFromLegacyIndex(int legacy_idx)
        {
            if (legacy_idx <= 0)
            {
                return DeviceId::cpu();
            }
            return DeviceId::cuda(legacy_idx - 1);
        }
    }

    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, DeviceId device,
        KVCacheLayoutMode layout_mode)
        : CPURingKVCache(mpi_ctx, n_layers, batch_size, max_seq_len,
                         n_kv_heads, n_kv_heads, 0, head_dim,
                         device, layout_mode)
    {
    }

    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
        : CPURingKVCache(mpi_ctx, n_layers, batch_size, max_seq_len,
                         n_kv_heads, n_kv_heads, 0, head_dim,
                         DeviceId::cpu(), layout_mode)
    {
        if (attention_devices.size() >= static_cast<size_t>(n_layers_))
        {
            for (int i = 0; i < n_layers_; ++i)
            {
                layer_devices_[i] = deviceIdFromLegacyIndex(attention_devices[i]);
            }
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                initialize_layer(layer, layer_devices_[layer]);
            }
        }
    }

    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device,
        KVCacheLayoutMode layout_mode)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len), n_kv_heads_(n_kv_heads), local_n_kv_heads_(local_n_kv_heads), kv_head_start_(kv_head_start), head_dim_(head_dim), kv_dim_(local_n_kv_heads * head_dim), is_sharded_(local_n_kv_heads != n_kv_heads), layout_mode_(layout_mode)
    {
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);
        entries_.resize(n_layers_);
        layer_devices_.resize(n_layers_, device);

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, layer_devices_[layer]);
        }
    }

    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
        : CPURingKVCache(mpi_ctx, n_layers, batch_size, max_seq_len,
                         n_kv_heads, local_n_kv_heads, kv_head_start,
                         head_dim, DeviceId::cpu(), layout_mode)
    {
        if (attention_devices.size() >= static_cast<size_t>(n_layers_))
        {
            for (int i = 0; i < n_layers_; ++i)
            {
                layer_devices_[i] = deviceIdFromLegacyIndex(attention_devices[i]);
            }
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                initialize_layer(layer, layer_devices_[layer]);
            }
        }
    }

    template <ActivationPrecision Precision>
    std::shared_ptr<typename CPURingKVCache<Precision>::TensorT> CPURingKVCache<Precision>::allocate_tensor(
        size_t rows, size_t cols, DeviceId device)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            return tensor_factory_->createFP32({rows, cols}, device);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            return tensor_factory_->createBF16({rows, cols});
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            return tensor_factory_->createFP16({rows, cols});
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            return tensor_factory_->createQ8_1({rows, cols}, device);
        }
        else
        {
            return tensor_factory_->createQ16_1({rows, cols}, device);
        }
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::initialize_layer(int layer, DeviceId device)
    {
        entries_[layer].resize(batch_size_);
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            auto &entry = entries_[layer][seq_idx];
            if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
            {
                entry.K = allocate_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
                entry.V = allocate_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
            }
            else
            {
                entry.K = allocate_tensor(max_seq_len_, kv_dim_, device);
                entry.V = allocate_tensor(max_seq_len_, kv_dim_, device);
            }
            entry.head = 0;
            entry.size = 0;
        }
    }

    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::get_cached_tokens(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].size;
    }

    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::get_kv(int layer, int seq_idx,
                                           ITensor **out_k, ITensor **out_v,
                                           int *out_kv_len)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        auto &entry = entries_[layer][seq_idx];
        if (out_k)
            *out_k = entry.K.get();
        if (out_v)
            *out_v = entry.V.get();
        if (out_kv_len)
            *out_kv_len = entry.size;
        return true;
    }

    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::get_kv(int layer, int seq_idx,
                                           const ITensor **out_k, const ITensor **out_v,
                                           int *out_kv_len) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        const auto &entry = entries_[layer][seq_idx];
        if (out_k)
            *out_k = entry.K.get();
        if (out_v)
            *out_v = entry.V.get();
        if (out_kv_len)
            *out_kv_len = entry.size;
        return true;
    }

    template <ActivationPrecision Precision>
    ITensor *CPURingKVCache<Precision>::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CPURingKVCache<Precision>::get_k(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision Precision>
    ITensor *CPURingKVCache<Precision>::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].V.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CPURingKVCache<Precision>::get_v(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].V.get();
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::clear()
    {
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
            {
                entries_[layer][seq_idx].head = 0;
                entries_[layer][seq_idx].size = 0;
            }
        }
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return;
        }
        entries_[layer][seq_idx].head = 0;
        entries_[layer][seq_idx].size = 0;
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            entries_[layer][seq_idx].head = 0;
            entries_[layer][seq_idx].size = 0;
        }
    }

    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v)
    {
        if (!new_k || !new_v)
        {
            return false;
        }
        return append_kv(layer, seq_idx, new_k, new_v, static_cast<int>(new_k->shape()[0]));
    }

    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_ || !new_k || !new_v || num_tokens <= 0)
        {
            return false;
        }

        const TensorT *typed_k = dynamic_cast<const TensorT *>(new_k);
        const TensorT *typed_v = dynamic_cast<const TensorT *>(new_v);
        if (!typed_k || !typed_v)
        {
            return false;
        }

        return append_kv_impl(layer, seq_idx, typed_k, typed_v, num_tokens);
    }

    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_kv_impl(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens)
    {
        auto &entry = entries_[layer][seq_idx];

        TensorT *dst_k = entry.K.get();
        TensorT *dst_v = entry.V.get();
        if (!dst_k || !dst_v)
        {
            return false;
        }

        const size_t src_rows_k = new_k->rows();
        const size_t src_rows_v = new_v->rows();
        const int rows_to_take = std::min({num_tokens, static_cast<int>(src_rows_k), static_cast<int>(src_rows_v)});
        if (rows_to_take <= 0)
        {
            return true;
        }

        const int src_start = std::max(0, rows_to_take - max_seq_len_);
        const int tokens_to_write = rows_to_take - src_start;

        auto copy_position_major_token = [&](int src_row, int dst_row)
        {
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src_k = new_k->typed_data();
                const float *src_v = new_v->typed_data();
                float *dst_k_data = dst_k->mutable_typed_data();
                float *dst_v_data = dst_v->mutable_typed_data();
                if (!src_k || !src_v || !dst_k_data || !dst_v_data)
                {
                    return false;
                }

                const float *src_k_row = src_k + static_cast<size_t>(src_row) * kv_dim_;
                const float *src_v_row = src_v + static_cast<size_t>(src_row) * kv_dim_;
                float *dst_k_row = dst_k_data + static_cast<size_t>(dst_row) * kv_dim_;
                float *dst_v_row = dst_v_data + static_cast<size_t>(dst_row) * kv_dim_;
                std::memcpy(dst_k_row, src_k_row, static_cast<size_t>(kv_dim_) * sizeof(float));
                std::memcpy(dst_v_row, src_v_row, static_cast<size_t>(kv_dim_) * sizeof(float));
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src_k = new_k->typed_data();
                const uint16_t *src_v = new_v->typed_data();
                uint16_t *dst_k_data = dst_k->mutable_typed_data();
                uint16_t *dst_v_data = dst_v->mutable_typed_data();
                if (!src_k || !src_v || !dst_k_data || !dst_v_data)
                {
                    return false;
                }

                const uint16_t *src_k_row = src_k + static_cast<size_t>(src_row) * kv_dim_;
                const uint16_t *src_v_row = src_v + static_cast<size_t>(src_row) * kv_dim_;
                uint16_t *dst_k_row = dst_k_data + static_cast<size_t>(dst_row) * kv_dim_;
                uint16_t *dst_v_row = dst_v_data + static_cast<size_t>(dst_row) * kv_dim_;
                std::memcpy(dst_k_row, src_k_row, static_cast<size_t>(kv_dim_) * sizeof(uint16_t));
                std::memcpy(dst_v_row, src_v_row, static_cast<size_t>(kv_dim_) * sizeof(uint16_t));
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const Q8_1Block *src_k = new_k->typed_data();
                const Q8_1Block *src_v = new_v->typed_data();
                Q8_1Block *dst_k_data = dst_k->mutable_typed_data();
                Q8_1Block *dst_v_data = dst_v->mutable_typed_data();
                if (!src_k || !src_v || !dst_k_data || !dst_v_data)
                {
                    return false;
                }

                const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const Q8_1Block *src_k_row = src_k + static_cast<size_t>(src_row) * blocks_per_row;
                const Q8_1Block *src_v_row = src_v + static_cast<size_t>(src_row) * blocks_per_row;
                Q8_1Block *dst_k_row = dst_k_data + static_cast<size_t>(dst_row) * blocks_per_row;
                Q8_1Block *dst_v_row = dst_v_data + static_cast<size_t>(dst_row) * blocks_per_row;
                std::memcpy(dst_k_row, src_k_row, blocks_per_row * sizeof(Q8_1Block));
                std::memcpy(dst_v_row, src_v_row, blocks_per_row * sizeof(Q8_1Block));
                return true;
            }
            else
            {
                const Q16BlockSize bs = dst_k->q16_block_size();
                const size_t block_bytes = q16_block_size_bytes(bs);
                const size_t block_elements = q16_block_size_elements(bs);
                const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;

                const uint8_t *src_k_bytes = reinterpret_cast<const uint8_t *>(new_k->raw_data());
                const uint8_t *src_v_bytes = reinterpret_cast<const uint8_t *>(new_v->raw_data());
                uint8_t *dst_k_bytes = reinterpret_cast<uint8_t *>(dst_k->raw_mutable_data());
                uint8_t *dst_v_bytes = reinterpret_cast<uint8_t *>(dst_v->raw_mutable_data());
                if (!src_k_bytes || !src_v_bytes || !dst_k_bytes || !dst_v_bytes)
                {
                    return false;
                }

                const uint8_t *src_k_row = src_k_bytes + static_cast<size_t>(src_row) * blocks_per_row * block_bytes;
                const uint8_t *src_v_row = src_v_bytes + static_cast<size_t>(src_row) * blocks_per_row * block_bytes;
                uint8_t *dst_k_row = dst_k_bytes + static_cast<size_t>(dst_row) * blocks_per_row * block_bytes;
                uint8_t *dst_v_row = dst_v_bytes + static_cast<size_t>(dst_row) * blocks_per_row * block_bytes;
                std::memcpy(dst_k_row, src_k_row, blocks_per_row * block_bytes);
                std::memcpy(dst_v_row, src_v_row, blocks_per_row * block_bytes);
                return true;
            }
        };

        auto copy_head_major_token = [&](int src_row, int dst_pos)
        {
            if (local_n_kv_heads_ <= 0 || head_dim_ <= 0)
            {
                return false;
            }

            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src_k_all = new_k->typed_data();
                const float *src_v_all = new_v->typed_data();
                float *dst_k_ptr = dst_k->mutable_typed_data();
                float *dst_v_ptr = dst_v->mutable_typed_data();
                if (!src_k_all || !src_v_all || !dst_k_ptr || !dst_v_ptr)
                {
                    return false;
                }
                const float *src_k = src_k_all + static_cast<size_t>(src_row) * kv_dim_;
                const float *src_v = src_v_all + static_cast<size_t>(src_row) * kv_dim_;
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const float *src_k_head = src_k + static_cast<size_t>(h) * head_dim_;
                    const float *src_v_head = src_v + static_cast<size_t>(h) * head_dim_;
                    float *dst_k_head = dst_k_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    float *dst_v_head = dst_v_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    std::memcpy(dst_k_head, src_k_head, static_cast<size_t>(head_dim_) * sizeof(float));
                    std::memcpy(dst_v_head, src_v_head, static_cast<size_t>(head_dim_) * sizeof(float));
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src_k_all = new_k->typed_data();
                const uint16_t *src_v_all = new_v->typed_data();
                uint16_t *dst_k_ptr = dst_k->mutable_typed_data();
                uint16_t *dst_v_ptr = dst_v->mutable_typed_data();
                if (!src_k_all || !src_v_all || !dst_k_ptr || !dst_v_ptr)
                {
                    return false;
                }
                const uint16_t *src_k = src_k_all + static_cast<size_t>(src_row) * kv_dim_;
                const uint16_t *src_v = src_v_all + static_cast<size_t>(src_row) * kv_dim_;
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const uint16_t *src_k_head = src_k + static_cast<size_t>(h) * head_dim_;
                    const uint16_t *src_v_head = src_v + static_cast<size_t>(h) * head_dim_;
                    uint16_t *dst_k_head = dst_k_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    uint16_t *dst_v_head = dst_v_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    std::memcpy(dst_k_head, src_k_head, static_cast<size_t>(head_dim_) * sizeof(uint16_t));
                    std::memcpy(dst_v_head, src_v_head, static_cast<size_t>(head_dim_) * sizeof(uint16_t));
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const auto *src_k_all = new_k->typed_data();
                const auto *src_v_all = new_v->typed_data();
                auto *dst_k_blocks = dst_k->mutable_typed_data();
                auto *dst_v_blocks = dst_v->mutable_typed_data();
                if (!src_k_all || !src_v_all || !dst_k_blocks || !dst_v_blocks)
                {
                    return false;
                }

                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const size_t src_blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const Q8_1Block *src_k_blocks = src_k_all + static_cast<size_t>(src_row) * src_blocks_per_row;
                const Q8_1Block *src_v_blocks = src_v_all + static_cast<size_t>(src_row) * src_blocks_per_row;

                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const Q8_1Block *src_k_head = src_k_blocks + static_cast<size_t>(h) * blocks_per_head;
                    const Q8_1Block *src_v_head = src_v_blocks + static_cast<size_t>(h) * blocks_per_head;
                    (void)src_blocks_per_row;
                    Q8_1Block *dst_k_head = dst_k_blocks + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head;
                    Q8_1Block *dst_v_head = dst_v_blocks + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head;
                    std::memcpy(dst_k_head, src_k_head, blocks_per_head * sizeof(Q8_1Block));
                    std::memcpy(dst_v_head, src_v_head, blocks_per_head * sizeof(Q8_1Block));
                }
                return true;
            }
            else
            {
                const Q16BlockSize bs = dst_k->q16_block_size();
                const size_t block_bytes = q16_block_size_bytes(bs);
                const size_t block_elements = q16_block_size_elements(bs);
                const uint8_t *src_k_bytes = reinterpret_cast<const uint8_t *>(new_k->raw_data());
                const uint8_t *src_v_bytes = reinterpret_cast<const uint8_t *>(new_v->raw_data());
                uint8_t *dst_k_bytes = reinterpret_cast<uint8_t *>(dst_k->raw_mutable_data());
                uint8_t *dst_v_bytes = reinterpret_cast<uint8_t *>(dst_v->raw_mutable_data());
                if (!src_k_bytes || !src_v_bytes || !dst_k_bytes || !dst_v_bytes)
                {
                    return false;
                }

                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + block_elements - 1) / block_elements;
                const size_t src_blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;
                const uint8_t *src_k_row = src_k_bytes + static_cast<size_t>(src_row) * src_blocks_per_row * block_bytes;
                const uint8_t *src_v_row = src_v_bytes + static_cast<size_t>(src_row) * src_blocks_per_row * block_bytes;
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const uint8_t *src_k_head = src_k_row + static_cast<size_t>(h) * blocks_per_head * block_bytes;
                    const uint8_t *src_v_head = src_v_row + static_cast<size_t>(h) * blocks_per_head * block_bytes;
                    uint8_t *dst_k_head = dst_k_bytes + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head * block_bytes;
                    uint8_t *dst_v_head = dst_v_bytes + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head * block_bytes;
                    std::memcpy(dst_k_head, src_k_head, blocks_per_head * block_bytes);
                    std::memcpy(dst_v_head, src_v_head, blocks_per_head * block_bytes);
                }
                return true;
            }
        };

        for (int i = 0; i < tokens_to_write; ++i)
        {
            const int src_row = src_start + i;
            int dst_pos = 0;
            if (entry.size < max_seq_len_)
            {
                dst_pos = (entry.head + entry.size) % max_seq_len_;
                ++entry.size;
            }
            else
            {
                dst_pos = entry.head;
                entry.head = (entry.head + 1) % max_seq_len_;
            }

            const bool ok = (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
                                ? copy_head_major_token(src_row, dst_pos)
                                : copy_position_major_token(src_row, dst_pos);
            if (!ok)
            {
                return false;
            }
        }

        return true;
    }

    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_one_tensor(TensorT *dst, const TensorT *src, EntryT &entry, int num_tokens)
    {
        if (!dst || !src || max_seq_len_ <= 0)
        {
            return false;
        }

        const size_t src_rows = src->rows();
        const size_t dst_rows = dst->rows();
        if (dst_rows < static_cast<size_t>(max_seq_len_) || src_rows == 0)
        {
            return false;
        }

        const int rows_to_take = std::min(num_tokens, static_cast<int>(src_rows));
        if (rows_to_take <= 0)
        {
            return true;
        }

        const int src_start = std::max(0, rows_to_take - max_seq_len_);
        const int tokens_to_write = rows_to_take - src_start;

        const size_t src_row_bytes = src->size_bytes() / std::max<size_t>(1, src->rows());
        const size_t dst_row_bytes = dst->size_bytes() / std::max<size_t>(1, dst->rows());
        const size_t row_bytes = std::min(src_row_bytes, dst_row_bytes);

        const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(src->raw_data());
        uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst->raw_mutable_data());
        if (!src_bytes || !dst_bytes || row_bytes == 0)
        {
            return false;
        }

        for (int i = 0; i < tokens_to_write; ++i)
        {
            const int src_row = src_start + i;
            int dst_row = 0;
            if (entry.size < max_seq_len_)
            {
                dst_row = (entry.head + entry.size) % max_seq_len_;
                ++entry.size;
            }
            else
            {
                dst_row = entry.head;
                entry.head = (entry.head + 1) % max_seq_len_;
            }

            std::memcpy(dst_bytes + static_cast<size_t>(dst_row) * dst_row_bytes,
                        src_bytes + static_cast<size_t>(src_row) * src_row_bytes,
                        row_bytes);
        }

        return true;
    }

    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::gather_kv_batched(
        int layer,
        int num_sequences,
        TensorBase *out_k,
        TensorBase *out_v,
        std::vector<int> &out_kv_lens)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return -1;
        }
        if (num_sequences <= 0 || num_sequences > batch_size_)
        {
            return -1;
        }

        TensorT *typed_k = dynamic_cast<TensorT *>(out_k);
        TensorT *typed_v = dynamic_cast<TensorT *>(out_v);
        if (!typed_k || !typed_v)
        {
            return -1;
        }

        out_kv_lens.resize(num_sequences);
        int max_kv_len = 0;
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
        {
            const int kv_len = entries_[layer][seq_idx].size;
            out_kv_lens[seq_idx] = kv_len;
            max_kv_len = std::max(max_kv_len, kv_len);
        }

        if (max_kv_len == 0)
        {
            return 0;
        }

        const size_t expected_rows = static_cast<size_t>(num_sequences) * static_cast<size_t>(max_kv_len);
        const size_t expected_cols = static_cast<size_t>(kv_dim_);
        if (typed_k->shape().size() < 2 || typed_v->shape().size() < 2)
        {
            return -1;
        }
        if (typed_k->shape()[0] < expected_rows || typed_k->shape()[1] != expected_cols)
        {
            return -1;
        }
        if (typed_v->shape()[0] < expected_rows || typed_v->shape()[1] != expected_cols)
        {
            return -1;
        }

        auto gather_tensor = [&](const TensorT *src_tensor, TensorT *dst_tensor, const EntryT &entry, int seq_idx, int kv_len) -> bool
        {
            if (!src_tensor || !dst_tensor || kv_len <= 0)
            {
                return true;
            }

            const int head = entry.head;

            if (layout_mode_ == KVCacheLayoutMode::POSITION_MAJOR)
            {
                if constexpr (Precision == ActivationPrecision::FP32)
                {
                    const float *src = src_tensor->typed_data();
                    float *dst = dst_tensor->mutable_typed_data();
                    if (!src || !dst)
                    {
                        return false;
                    }

                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const float *src_row = src + static_cast<size_t>(phys) * kv_dim_;
                        float *dst_row_ptr = dst + dst_row * kv_dim_;
                        std::memcpy(dst_row_ptr, src_row, static_cast<size_t>(kv_dim_) * sizeof(float));
                    }
                    return true;
                }
                else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
                {
                    const uint16_t *src = src_tensor->typed_data();
                    uint16_t *dst = dst_tensor->mutable_typed_data();
                    if (!src || !dst)
                    {
                        return false;
                    }

                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const uint16_t *src_row = src + static_cast<size_t>(phys) * kv_dim_;
                        uint16_t *dst_row_ptr = dst + dst_row * kv_dim_;
                        std::memcpy(dst_row_ptr, src_row, static_cast<size_t>(kv_dim_) * sizeof(uint16_t));
                    }
                    return true;
                }
                else if constexpr (Precision == ActivationPrecision::Q8_1)
                {
                    const Q8_1Block *src = src_tensor->typed_data();
                    Q8_1Block *dst = dst_tensor->mutable_typed_data();
                    if (!src || !dst)
                    {
                        return false;
                    }

                    const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const Q8_1Block *src_row = src + static_cast<size_t>(phys) * blocks_per_row;
                        Q8_1Block *dst_row_ptr = dst + dst_row * blocks_per_row;
                        std::memcpy(dst_row_ptr, src_row, blocks_per_row * sizeof(Q8_1Block));
                    }
                    return true;
                }
                else
                {
                    const Q16BlockSize bs = src_tensor->q16_block_size();
                    const size_t block_bytes = q16_block_size_bytes(bs);
                    const size_t block_elements = q16_block_size_elements(bs);
                    const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;

                    const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(src_tensor->raw_data());
                    uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst_tensor->raw_mutable_data());
                    if (!src_bytes || !dst_bytes)
                    {
                        return false;
                    }

                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const uint8_t *src_row = src_bytes + static_cast<size_t>(phys) * blocks_per_row * block_bytes;
                        uint8_t *dst_row_ptr = dst_bytes + dst_row * blocks_per_row * block_bytes;
                        std::memcpy(dst_row_ptr, src_row, blocks_per_row * block_bytes);
                    }
                    return true;
                }
            }

            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src = reinterpret_cast<const float *>(src_tensor->raw_data());
                float *dst = reinterpret_cast<float *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }
                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    float *dst_row_ptr = dst + dst_row * static_cast<size_t>(kv_dim_);
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const float *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * head_dim_;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * head_dim_, src_head, static_cast<size_t>(head_dim_) * sizeof(float));
                    }
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src = reinterpret_cast<const uint16_t *>(src_tensor->raw_data());
                uint16_t *dst = reinterpret_cast<uint16_t *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }
                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    uint16_t *dst_row_ptr = dst + dst_row * static_cast<size_t>(kv_dim_);
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const uint16_t *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * head_dim_;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * head_dim_, src_head, static_cast<size_t>(head_dim_) * sizeof(uint16_t));
                    }
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const Q8_1Block *src = reinterpret_cast<const Q8_1Block *>(src_tensor->raw_data());
                Q8_1Block *dst = reinterpret_cast<Q8_1Block *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }
                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const size_t dst_blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                if (dst_blocks_per_row < static_cast<size_t>(local_n_kv_heads_) * blocks_per_head)
                {
                    return false;
                }

                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    Q8_1Block *dst_row_ptr = dst + dst_row * dst_blocks_per_row;
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const Q8_1Block *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * blocks_per_head;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * blocks_per_head,
                                    src_head,
                                    blocks_per_head * sizeof(Q8_1Block));
                    }
                }
                return true;
            }
            else
            {
                const Q16BlockSize bs = src_tensor->q16_block_size();
                const size_t block_bytes = q16_block_size_bytes(bs);
                const size_t block_elements = q16_block_size_elements(bs);
                const uint8_t *src = reinterpret_cast<const uint8_t *>(src_tensor->raw_data());
                uint8_t *dst = reinterpret_cast<uint8_t *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }

                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + block_elements - 1) / block_elements;
                const size_t dst_blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;
                if (dst_blocks_per_row < static_cast<size_t>(local_n_kv_heads_) * blocks_per_head)
                {
                    return false;
                }

                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    uint8_t *dst_row_ptr = dst + dst_row * dst_blocks_per_row * block_bytes;
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const uint8_t *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * blocks_per_head * block_bytes;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * blocks_per_head * block_bytes,
                                    src_head,
                                    blocks_per_head * block_bytes);
                    }
                }
                return true;
            }
        };

        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
        {
            const auto &entry = entries_[layer][seq_idx];
            const int kv_len = out_kv_lens[seq_idx];
            if (kv_len <= 0)
            {
                continue;
            }

            if (!gather_tensor(entry.K.get(), typed_k, entry, seq_idx, kv_len))
            {
                return -1;
            }
            if (!gather_tensor(entry.V.get(), typed_v, entry, seq_idx, kv_len))
            {
                return -1;
            }
        }

        return max_kv_len;
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::evict_oldest(int tokens_to_evict)
    {
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            evict_oldest_from_sequence(seq_idx, tokens_to_evict);
        }
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::evict_oldest_from_sequence(int seq_idx, int tokens_to_evict)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_ || tokens_to_evict <= 0)
        {
            return;
        }

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            auto &entry = entries_[layer][seq_idx];
            const int evict = std::min(tokens_to_evict, entry.size);
            if (evict <= 0)
            {
                continue;
            }

            entry.head = (entry.head + evict) % max_seq_len_;
            entry.size -= evict;
            total_evicted_ += evict;
        }
    }

    template <ActivationPrecision Precision>
    DeviceId CPURingKVCache<Precision>::get_layer_device(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return DeviceId::cpu();
        }
        return layer_devices_[layer];
    }

    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::ring_head(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].head;
    }

    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::ring_size(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].size;
    }

    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        DeviceId device,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        default:
            LOG_ERROR("createCPURingKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        default:
            LOG_ERROR("createCPURingKVCache(attention_devices): unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                         n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        default:
            LOG_ERROR("createShardedCPURingKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                         n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        default:
            LOG_ERROR("createShardedCPURingKVCache(attention_devices): unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    template class CPURingKVCache<ActivationPrecision::FP32>;
    template class CPURingKVCache<ActivationPrecision::BF16>;
    template class CPURingKVCache<ActivationPrecision::FP16>;
    template class CPURingKVCache<ActivationPrecision::Q8_1>;
    template class CPURingKVCache<ActivationPrecision::Q16_1>;

} // namespace llaminar2

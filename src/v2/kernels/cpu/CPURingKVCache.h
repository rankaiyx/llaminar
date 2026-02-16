/**
 * @file CPURingKVCache.h
 * @brief Phase 1 CPU ring-buffer KV cache scaffold
 */

#pragma once

#include "CPUKVCache.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    template <ActivationPrecision Precision>
    struct CPURingKVCacheEntry
    {
        using TensorT = typename detail::CPUKVCacheTensor<Precision>::Type;

        std::shared_ptr<TensorT> K;
        std::shared_ptr<TensorT> V;
        int head = 0;
        int size = 0;
    };

    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class CPURingKVCache : public ICPUKVCache
    {
    public:
        using TensorT = typename detail::CPUKVCacheTensor<Precision>::Type;
        using EntryT = CPURingKVCacheEntry<Precision>;

        CPURingKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim, DeviceId device = DeviceId::cpu(),
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        CPURingKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim, const std::vector<int> &attention_devices,
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        CPURingKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                       int head_dim, DeviceId device = DeviceId::cpu(),
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        CPURingKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                       int head_dim, const std::vector<int> &attention_devices,
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        ActivationPrecision precision() const override { return Precision; }
        int max_seq_len() const override { return max_seq_len_; }
        int n_layers() const override { return n_layers_; }

        TensorLayout kv_layout() const override
        {
            return (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
                       ? TensorLayout::KV_HEAD_POS_DIM
                       : TensorLayout::KV_POS_HEAD_DIM;
        }

        int get_cached_tokens(int layer, int seq_idx = 0) const override;

        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override;

        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override;

        using ICPUKVCache::get_kv;

        ITensor *get_k(int layer, int seq_idx = 0) override;
        const ITensor *get_k(int layer, int seq_idx = 0) const override;
        ITensor *get_v(int layer, int seq_idx = 0) override;
        const ITensor *get_v(int layer, int seq_idx = 0) const override;

        void clear() override;
        void clear_sequence(int layer, int seq_idx) override;
        void clear_layer(int layer) override;
        using ICPUKVCache::clear_sequence;

        bool is_sharded() const override { return is_sharded_; }
        int local_n_kv_heads() const override { return local_n_kv_heads_; }
        int kv_head_start() const override { return kv_head_start_; }
        int local_kv_dim() const override { return kv_dim_; }

        int num_layers() const override { return n_layers_; }
        int batch_size() const override { return batch_size_; }
        KVCacheLayoutMode layout_mode() const override { return layout_mode_; }
        int n_kv_heads() const override { return n_kv_heads_; }

        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) override;
        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) override;
        using ICPUKVCache::append_kv;

        int gather_kv_batched(int layer, int num_sequences, TensorBase *out_k, TensorBase *out_v,
                              std::vector<int> &out_kv_lens) override;

        void evict_oldest(int tokens_to_evict) override;
        void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) override;

        DeviceId get_layer_device(int layer) const override;
        int get_total_evicted() const override { return total_evicted_; }
        void reset_eviction_counter() override { total_evicted_ = 0; }

        int ring_head(int layer, int seq_idx = 0) const;
        int ring_size(int layer, int seq_idx = 0) const;

    private:
        int n_layers_;
        int batch_size_;
        int max_seq_len_;
        int n_kv_heads_;
        int local_n_kv_heads_;
        int kv_head_start_;
        int head_dim_;
        int kv_dim_;
        bool is_sharded_;
        int total_evicted_ = 0;
        KVCacheLayoutMode layout_mode_;

        std::vector<std::vector<EntryT>> entries_;
        std::vector<DeviceId> layer_devices_;
        std::unique_ptr<TensorFactory> tensor_factory_;

        std::shared_ptr<TensorT> allocate_tensor(size_t rows, size_t cols, DeviceId device);
        void initialize_layer(int layer, DeviceId device);
        bool append_kv_impl(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens);
        bool append_one_tensor(TensorT *dst, const TensorT *src, EntryT &entry, int num_tokens);
    };

    using CPURingKVCacheFP32 = CPURingKVCache<ActivationPrecision::FP32>;
    using CPURingKVCacheBF16 = CPURingKVCache<ActivationPrecision::BF16>;
    using CPURingKVCacheFP16 = CPURingKVCache<ActivationPrecision::FP16>;
    using CPURingKVCacheQ8_1 = CPURingKVCache<ActivationPrecision::Q8_1>;
    using CPURingKVCacheQ16_1 = CPURingKVCache<ActivationPrecision::Q16_1>;

    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        DeviceId device = DeviceId::cpu(),
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device = DeviceId::cpu(),
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

} // namespace llaminar2

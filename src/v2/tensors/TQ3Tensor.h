/**
 * @file TQ3Tensor.h
 * @brief TurboQuant 3-bit tensor for KV cache storage
 *
 * Same design as TQ4Tensor but with a 3-bit total budget split as
 * 2-bit MSE codebook indices plus a 1-bit QJL residual sketch.
 *
 * @see TQ3Block for the per-head block structure.
 * @see TQ4Tensor.h for detailed design documentation.
 */

#pragma once

#include "TensorClasses.h"

#include <vector>
#include <memory>
#include <cstring>

namespace llaminar2
{

    class TurboQuantContext;

    /**
     * @brief TurboQuant 3-bit tensor for KV cache storage.
     *
     * Same pattern as TQ4Tensor. Uses 3-bit Lloyd-Max codebook for higher compression
     * at the cost of slightly more quantization error.
     */
    class TQ3Tensor : public TypedTensorBase<TQ3Tensor, uint8_t>,
                      public TensorBase
    {
    public:
        using value_type = uint8_t;
        static constexpr int static_type_id() { return TensorTypeId::TQ3; }

        const uint8_t *data_impl() const { return raw_blocks_.data(); }
        uint8_t *mutable_data_impl() { return raw_blocks_.data(); }

        TQ3Tensor(const std::vector<size_t> &shape, int head_dim, DeviceId device = DeviceId::cpu());

        ~TQ3Tensor() override = default;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::TQ3; }
        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        const float *fp32_data() const override { return data(); }
        float *mutable_data() override;

        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return raw_blocks_.data(); }
        void *raw_mutable_data() final { return raw_blocks_.data(); }

    protected:
        size_t byte_size() const override { return raw_blocks_.size(); }
        void *raw_host_data_ptr() override { return raw_blocks_.data(); }
        const void *raw_host_data_ptr() const override { return raw_blocks_.data(); }

    public:

        bool copyFrom(const TensorBase *src) override;

        void release_raw_data() override;
        bool is_raw_data_released() const override { return raw_data_released_; }

        void to_fp32(float *dst) const override;
        void to_bf16(uint16_t *) const override { throw std::runtime_error("TQ3Tensor: to_bf16 not supported"); }
        void to_fp16(uint16_t *) const override { throw std::runtime_error("TQ3Tensor: to_fp16 not supported"); }
        void to_int8_blocked(int8_t *, float *, size_t) const override { throw std::runtime_error("TQ3Tensor: to_int8_blocked not supported"); }
        bool to_int8_perchannel(int8_t *, float *, float *) const override { return false; }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        std::unique_ptr<ITensorGemm> createGemm() override { return nullptr; }

        std::shared_ptr<TensorBase> create_view(const std::vector<size_t> &, size_t) override
        {
            throw std::runtime_error("TQ3Tensor: create_view not supported");
        }

        int head_dim() const { return head_dim_; }
        size_t block_bytes() const { return block_bytes_; }
        size_t blocks_per_row() const { return blocks_per_row_; }
        size_t total_blocks() const { return rows() * blocks_per_row_; }

        void set_turboquant_context(const TurboQuantContext *turboquant_ctx) { turboquant_ctx_ = turboquant_ctx; }
        const TurboQuantContext *turboquant_context() const { return turboquant_ctx_; }

        static std::shared_ptr<TQ3Tensor> quantize_from_fp32(
            const float *src,
            const std::vector<size_t> &shape,
            int head_dim,
            const TurboQuantContext &turboquant_ctx);

        bool copyFrom_fp32_rows(const float *src_data, size_t num_rows, const TurboQuantContext &turboquant_ctx);

        void dequantize_to_fp32(float *dst, const TurboQuantContext &turboquant_ctx) const;

    private:
        std::vector<size_t> shape_;
        int head_dim_;
        size_t block_bytes_;
        size_t blocks_per_row_;
        DeviceId device_;
        std::vector<uint8_t> raw_blocks_;
        bool raw_data_released_ = false;

        mutable std::vector<float> dequant_cache_;
        mutable bool dequant_cache_valid_ = false;

        const TurboQuantContext *turboquant_ctx_ = nullptr;

        void invalidate_dequant_cache() { dequant_cache_valid_ = false; }
    };

} // namespace llaminar2

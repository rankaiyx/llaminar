/**
 * @file LocalTPWeightSharder.cpp
 * @brief Implementation of weight sharding for LOCAL tensor parallelism
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ILocalTPWeightSharder.h"
#include "../tensors/TensorClasses.h"
#include "../tensors/TensorSlice.h"
#include "../tensors/TensorFactory.h"
#include "../utils/Logger.h"
#include <stdexcept>

namespace llaminar2
{

    /**
     * @brief Concrete implementation of ILocalTPWeightSharder
     *
     * Uses TensorSlice to create shards without copying data when possible.
     * Supports proportional sharding via ILocalTPContext weights.
     */
    class LocalTPWeightSharder : public ILocalTPWeightSharder
    {
    public:
        LocalTPWeightSharder() = default;
        ~LocalTPWeightSharder() override = default;

        // =====================================================================
        // Multi-Device Sharding
        // =====================================================================

        std::vector<std::unique_ptr<TensorBase>> shardColumnParallel(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx) override
        {
            if (!full_weight)
            {
                throw std::invalid_argument("LocalTPWeightSharder::shardColumnParallel: null weight");
            }

            std::vector<std::unique_ptr<TensorBase>> shards;
            shards.reserve(tp_ctx.degree());

            for (int i = 0; i < tp_ctx.degree(); ++i)
            {
                const auto &device = tp_ctx.deviceAt(i);
                auto shard = getColumnShard(full_weight, tp_ctx, device);
                shards.push_back(std::move(shard));
            }

            return shards;
        }

        std::vector<std::unique_ptr<TensorBase>> shardRowParallel(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx) override
        {
            if (!full_weight)
            {
                throw std::invalid_argument("LocalTPWeightSharder::shardRowParallel: null weight");
            }

            std::vector<std::unique_ptr<TensorBase>> shards;
            shards.reserve(tp_ctx.degree());

            for (int i = 0; i < tp_ctx.degree(); ++i)
            {
                const auto &device = tp_ctx.deviceAt(i);
                auto shard = getRowShard(full_weight, tp_ctx, device);
                shards.push_back(std::move(shard));
            }

            return shards;
        }

        // =====================================================================
        // Single-Device Sharding
        // =====================================================================

        std::unique_ptr<TensorBase> getColumnShard(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) override
        {
            if (!full_weight)
            {
                throw std::invalid_argument("LocalTPWeightSharder::getColumnShard: null weight");
            }

            int idx = tp_ctx.indexForDevice(device);
            if (idx < 0)
            {
                throw std::invalid_argument("LocalTPWeightSharder::getColumnShard: device not in context");
            }

            // Get column range from LOCAL TP context
            auto [col_start, col_end] = tp_ctx.colRangeForDevice(device, static_cast<int>(full_weight->cols()));

            // Create slice metadata for column-parallel
            SliceMetadata meta;
            meta.mode = SliceMode::COLUMN_PARALLEL;
            meta.original_rows = full_weight->rows();
            meta.original_cols = full_weight->cols();
            meta.slice_start = col_start;
            meta.slice_end = col_end;
            meta.rank = idx;
            meta.world_size = tp_ctx.degree();
            meta.inner_is_presliced = false; // Will slice during kernel creation

            // Create a TensorSlice wrapping the full weight
            // Note: We need to create a shared_ptr from the const raw pointer.
            // In practice, weights should be owned by WeightManager as shared_ptr.
            // This creates a non-owning slice wrapper.

            // For now, we create a copy-based slice for correctness.
            // TODO: Optimize to use view when WeightManager provides shared_ptr
            auto slice = createColumnSliceCopy(full_weight, col_start, col_end - col_start);

            LOG_DEBUG("LocalTPWeightSharder::getColumnShard: device " << idx
                                                                      << " cols [" << col_start << ", " << col_end << ")"
                                                                      << " of " << full_weight->cols());

            return slice;
        }

        std::unique_ptr<TensorBase> getRowShard(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) override
        {
            if (!full_weight)
            {
                throw std::invalid_argument("LocalTPWeightSharder::getRowShard: null weight");
            }

            int idx = tp_ctx.indexForDevice(device);
            if (idx < 0)
            {
                throw std::invalid_argument("LocalTPWeightSharder::getRowShard: device not in context");
            }

            // Get row range from LOCAL TP context
            auto [row_start, row_end] = tp_ctx.rowRangeForDevice(device, static_cast<int>(full_weight->rows()));

            // Create slice metadata for row-parallel
            SliceMetadata meta;
            meta.mode = SliceMode::ROW_PARALLEL;
            meta.original_rows = full_weight->rows();
            meta.original_cols = full_weight->cols();
            meta.slice_start = row_start;
            meta.slice_end = row_end;
            meta.rank = idx;
            meta.world_size = tp_ctx.degree();
            meta.inner_is_presliced = false;

            // Create a copy-based slice for correctness
            auto slice = createRowSliceCopy(full_weight, row_start, row_end - row_start);

            LOG_DEBUG("LocalTPWeightSharder::getRowShard: device " << idx
                                                                   << " rows [" << row_start << ", " << row_end << ")"
                                                                   << " of " << full_weight->rows());

            return slice;
        }

        // =====================================================================
        // Query Methods
        // =====================================================================

        int columnCountForDevice(
            int total_cols,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) const override
        {
            auto [col_start, col_end] = tp_ctx.colRangeForDevice(device, total_cols);
            return col_end - col_start;
        }

        int rowCountForDevice(
            int total_rows,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) const override
        {
            auto [row_start, row_end] = tp_ctx.rowRangeForDevice(device, total_rows);
            return row_end - row_start;
        }

    private:
        /**
         * @brief Create a row slice by copying data
         *
         * For FP32 tensors, copies rows [start, start+count).
         * For quantized tensors, dequantizes to FP32 first.
         */
        std::unique_ptr<TensorBase> createRowSliceCopy(
            const TensorBase *tensor, int row_start, int row_count)
        {
            const size_t cols = tensor->cols();

            // Handle FP32 tensors directly (most common for activations)
            if (auto *fp32 = dynamic_cast<const FP32Tensor *>(tensor))
            {
                std::vector<size_t> shape = {static_cast<size_t>(row_count), cols};
                auto slice = std::make_unique<FP32Tensor>(shape);

                const float *src = fp32->data() + row_start * cols;
                float *dst = slice->mutable_data();
                std::memcpy(dst, src, row_count * cols * sizeof(float));

                return slice;
            }

            // For quantized weight tensors, dequantize to FP32 first
            LOG_WARN("LocalTPWeightSharder: falling back to FP32 slice for non-FP32 tensor");

            // Dequantize full tensor to FP32
            std::vector<float> full_fp32(tensor->numel());
            tensor->to_fp32(full_fp32.data());

            // Create slice
            std::vector<size_t> shape = {static_cast<size_t>(row_count), cols};
            auto slice = std::make_unique<FP32Tensor>(shape);

            // Copy rows
            float *dst = slice->mutable_data();
            const float *src = full_fp32.data() + row_start * cols;
            std::memcpy(dst, src, row_count * cols * sizeof(float));

            return slice;
        }

        /**
         * @brief Create a column slice by copying data
         *
         * For FP32 tensors, copies columns [start, start+count) from each row.
         * This requires strided access since data is row-major.
         */
        std::unique_ptr<TensorBase> createColumnSliceCopy(
            const TensorBase *tensor, int col_start, int col_count)
        {
            const size_t rows = tensor->rows();
            const size_t full_cols = tensor->cols();

            // Handle FP32 tensors
            if (auto *fp32 = dynamic_cast<const FP32Tensor *>(tensor))
            {
                std::vector<size_t> shape = {rows, static_cast<size_t>(col_count)};
                auto slice = std::make_unique<FP32Tensor>(shape);

                const float *src = fp32->data();
                float *dst = slice->mutable_data();

                for (size_t r = 0; r < rows; ++r)
                {
                    std::memcpy(dst + r * col_count,
                                src + r * full_cols + col_start,
                                col_count * sizeof(float));
                }

                return slice;
            }

            // For quantized weight tensors, dequantize then slice
            LOG_WARN("LocalTPWeightSharder: falling back to FP32 column slice for non-FP32 tensor");

            // Dequantize full tensor to FP32
            std::vector<float> full_fp32(tensor->numel());
            tensor->to_fp32(full_fp32.data());

            // Create slice
            std::vector<size_t> shape = {rows, static_cast<size_t>(col_count)};
            auto slice = std::make_unique<FP32Tensor>(shape);
            float *dst = slice->mutable_data();

            for (size_t r = 0; r < rows; ++r)
            {
                std::memcpy(dst + r * col_count,
                            full_fp32.data() + r * full_cols + col_start,
                            col_count * sizeof(float));
            }

            return slice;
        }
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<ILocalTPWeightSharder> createLocalTPWeightSharder()
    {
        return std::make_unique<LocalTPWeightSharder>();
    }

} // namespace llaminar2

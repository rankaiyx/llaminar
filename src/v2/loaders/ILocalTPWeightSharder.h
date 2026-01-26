/**
 * @file ILocalTPWeightSharder.h
 * @brief Interface for sharding weights across LOCAL TP devices
 *
 * Provides weight sharding strategies for LOCAL tensor parallelism:
 * - Column-parallel: Split output dimension (Q, K, V projections, FFN up/gate)
 * - Row-parallel: Split input dimension (output projections, FFN down)
 *
 * Supports proportional sharding for heterogeneous GPU configurations
 * where devices have different compute capabilities.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../collective/ILocalTPContext.h"
#include "../tensors/TensorClasses.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Interface for sharding weights across LOCAL TP devices
     *
     * Handles weight partitioning for tensor parallelism within a single MPI rank.
     * Two primary sharding patterns:
     *
     * Column-parallel (split output dimension):
     *   - Q, K, V projections: [hidden_dim, n_heads * head_dim] → split heads
     *   - FFN gate/up: [hidden_dim, ffn_dim] → split ffn_dim
     *   - Activation: X @ W → Y (each device has subset of Y columns)
     *   - Requires all-gather after to reconstruct full activation
     *
     * Row-parallel (split input dimension):
     *   - Output projection: [n_heads * head_dim, hidden_dim] → split input
     *   - FFN down: [ffn_dim, hidden_dim] → split ffn_dim
     *   - Activation: X @ W → Y (partial Y, needs all-reduce to sum)
     *   - Requires all-reduce after to sum partial results
     *
     * Thread safety: All methods are thread-safe.
     */
    class ILocalTPWeightSharder
    {
    public:
        virtual ~ILocalTPWeightSharder() = default;

        // =====================================================================
        // Multi-Device Sharding
        // =====================================================================

        /**
         * @brief Shard a weight tensor for column-parallel operation
         *
         * Splits weight along output dimension (columns in row-major layout).
         * Returns one shard per device in tp_ctx.
         *
         * Example: [hidden, n_heads * head_dim] with 2 devices
         *   → Device 0: [hidden, n_heads/2 * head_dim]
         *   → Device 1: [hidden, n_heads/2 * head_dim]
         *
         * For proportional TP (e.g., 73%/27%):
         *   → Device 0: [hidden, 0.73 * n_heads * head_dim]
         *   → Device 1: [hidden, 0.27 * n_heads * head_dim]
         *
         * @param full_weight Complete weight tensor (not modified)
         * @param tp_ctx LOCAL TP context with devices and weights
         * @return Vector of shards, one per device (in device order)
         */
        virtual std::vector<std::unique_ptr<TensorBase>> shardColumnParallel(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx) = 0;

        /**
         * @brief Shard a weight tensor for row-parallel operation
         *
         * Splits weight along input dimension (rows in row-major layout).
         * Returns one shard per device in tp_ctx.
         *
         * Example: [ffn_dim, hidden] with 2 devices
         *   → Device 0: [ffn_dim/2, hidden]
         *   → Device 1: [ffn_dim/2, hidden]
         *
         * @param full_weight Complete weight tensor (not modified)
         * @param tp_ctx LOCAL TP context with devices and weights
         * @return Vector of shards, one per device (in device order)
         */
        virtual std::vector<std::unique_ptr<TensorBase>> shardRowParallel(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx) = 0;

        // =====================================================================
        // Single-Device Sharding
        // =====================================================================

        /**
         * @brief Get column shard for a specific device
         *
         * Convenience method when you only need one device's shard.
         *
         * @param full_weight Complete weight tensor
         * @param tp_ctx LOCAL TP context
         * @param device Device to get shard for
         * @return Column shard for the specified device
         */
        virtual std::unique_ptr<TensorBase> getColumnShard(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) = 0;

        /**
         * @brief Get row shard for a specific device
         *
         * Convenience method when you only need one device's shard.
         *
         * @param full_weight Complete weight tensor
         * @param tp_ctx LOCAL TP context
         * @param device Device to get shard for
         * @return Row shard for the specified device
         */
        virtual std::unique_ptr<TensorBase> getRowShard(
            const TensorBase *full_weight,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) = 0;

        // =====================================================================
        // Query Methods
        // =====================================================================

        /**
         * @brief Get column count for a device's shard
         *
         * @param total_cols Total columns in full weight
         * @param tp_ctx LOCAL TP context
         * @param device Device to query
         * @return Number of columns in this device's column-parallel shard
         */
        virtual int columnCountForDevice(
            int total_cols,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) const = 0;

        /**
         * @brief Get row count for a device's shard
         *
         * @param total_rows Total rows in full weight
         * @param tp_ctx LOCAL TP context
         * @param device Device to query
         * @return Number of rows in this device's row-parallel shard
         */
        virtual int rowCountForDevice(
            int total_rows,
            const ILocalTPContext &tp_ctx,
            const GlobalDeviceAddress &device) const = 0;
    };

    /**
     * @brief Factory function to create a LocalTPWeightSharder
     *
     * @return Unique pointer to ILocalTPWeightSharder implementation
     */
    std::unique_ptr<ILocalTPWeightSharder> createLocalTPWeightSharder();

} // namespace llaminar2

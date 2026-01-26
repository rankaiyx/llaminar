/**
 * @file ILocalTPContext.h
 * @brief Interface for LOCAL tensor parallelism operations
 *
 * LOCAL TP = multiple devices within a single MPI rank, decoupled from MPI world_size.
 * This enables tensor parallelism across GPUs owned by one rank, using high-bandwidth
 * backends like NCCL, RCCL, or PCIeBAR instead of cross-node MPI.
 *
 * Key concepts:
 * - LOCAL TP degree can be different from MPI world_size
 * - Supports proportional work distribution via weights (e.g., NVIDIA 73%, AMD 27%)
 * - Backend selection based on device types (NCCL for CUDA-only, PCIeBAR for mixed)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/GlobalDeviceAddress.h"
#include "../config/OrchestrationConfig.h"
#include "../tensors/ITensor.h"
#include <memory>
#include <vector>
#include <utility>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Interface for LOCAL tensor parallelism operations
     *
     * LOCAL TP = multiple devices within a single MPI rank.
     * This decouples TP degree from MPI world_size, enabling:
     * - Single-rank multi-GPU execution
     * - Heterogeneous GPU TP (CUDA + ROCm on same rank)
     * - Proportional work distribution for mixed-capability GPUs
     *
     * Thread safety: All methods are thread-safe. Collective operations
     * must be called from all participating threads/devices.
     */
    class ILocalTPContext
    {
    public:
        virtual ~ILocalTPContext() = default;

        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Get devices participating in LOCAL TP
         * @return Vector of GlobalDeviceAddress for all devices in this context
         */
        virtual const std::vector<GlobalDeviceAddress> &devices() const = 0;

        /**
         * @brief Get weights for proportional TP
         *
         * Weights determine work distribution. A device with weight 0.73 gets
         * 73% of the heads/columns in column-parallel operations.
         *
         * @return Vector of weights (sum to 1.0), same length as devices()
         */
        virtual const std::vector<float> &weights() const = 0;

        /**
         * @brief Get backend type for collective operations
         *
         * AUTO resolution:
         * - All CUDA devices → NCCL
         * - All ROCm devices → RCCL
         * - Mixed or CPU involved → PCIeBAR or HOST
         *
         * @return CollectiveBackendType
         */
        virtual CollectiveBackendType backend() const = 0;

        /**
         * @brief Get total TP degree (number of devices)
         * @return Number of devices participating in LOCAL TP
         */
        virtual int degree() const = 0;

        // =====================================================================
        // Collective Operations
        // =====================================================================

        /**
         * @brief All-reduce across LOCAL devices (in-place)
         *
         * Performs sum reduction across all devices, leaving result on all devices.
         * This is the core operation after row-parallel GEMM (e.g., Wo projection).
         *
         * @param tensor Tensor to all-reduce (modified in-place)
         * @return true on success, false on error
         */
        virtual bool allreduce(TensorBase *tensor) = 0;

        /**
         * @brief All-reduce across LOCAL devices (out-of-place)
         *
         * @param input Source tensor (read-only)
         * @param output Destination tensor (must be pre-allocated)
         * @return true on success, false on error
         */
        virtual bool allreduce(const TensorBase *input, TensorBase *output) = 0;

        /**
         * @brief All-gather: gather shards from all devices
         *
         * Each device contributes its local shard, result is full tensor on all devices.
         * Used after column-parallel operations to reconstruct full activations.
         *
         * @param local_shard This device's shard (may differ in size due to proportional TP)
         * @param global_tensor Pre-allocated output for full tensor
         * @return true on success, false on error
         */
        virtual bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) = 0;

        /**
         * @brief Gather shards from multiple devices into a single output tensor
         *
         * This is the orchestrator-friendly variant of allgather. Instead of requiring
         * each device to call allgather independently (SPMD pattern), this method
         * accepts all shards at once from a single control thread.
         *
         * Used by MultiDeviceOrchestrator to gather partial logits from column-parallel
         * LM head execution across LOCAL TP devices.
         *
         * @param shards Vector of shard tensors (one per device, in device order)
         * @param output Pre-allocated output tensor for concatenated result
         * @return true on success, false on error
         *
         * @note The shards are concatenated in device order (device 0's shard first,
         *       then device 1's shard, etc.). For column-parallel LM head, this means
         *       output[0:vocab_local_0] = shard[0], output[vocab_local_0:vocab_local_0+vocab_local_1] = shard[1], etc.
         */
        virtual bool gatherFromDevices(
            const std::vector<const TensorBase *> &shards,
            TensorBase *output) = 0;

        /**
         * @brief Reduce-scatter: reduce then scatter result slices
         *
         * Reduces input across all devices, then scatters result so each device
         * gets a different portion. Useful for fused reduce-scatter operations.
         *
         * @param input Full tensor to reduce
         * @param output_shard This device's portion of the result
         * @return true on success, false on error
         */
        virtual bool reduceScatter(const TensorBase *input, TensorBase *output_shard) = 0;

        // =====================================================================
        // Synchronization
        // =====================================================================

        /**
         * @brief Synchronize all LOCAL devices
         *
         * Blocks until all devices have completed their pending operations.
         * Call after async collective operations to ensure completion.
         */
        virtual void synchronize() = 0;

        // =====================================================================
        // Device Management
        // =====================================================================

        /**
         * @brief Get index for a device (0-based)
         * @param device Device to look up
         * @return Index in devices() vector, or -1 if not found
         */
        virtual int indexForDevice(const GlobalDeviceAddress &device) const = 0;

        /**
         * @brief Get device by index
         * @param index 0-based index
         * @return GlobalDeviceAddress at that index
         * @throws std::out_of_range if index is invalid
         */
        virtual const GlobalDeviceAddress &deviceAt(int index) const = 0;

        /**
         * @brief Get work fraction for a device
         * @param device Device to look up
         * @return Weight for this device (0.0-1.0)
         */
        virtual float weightForDevice(const GlobalDeviceAddress &device) const = 0;

        // =====================================================================
        // Weight Sharding Utilities
        // =====================================================================

        /**
         * @brief Get head count for a device (for attention sharding)
         *
         * Distributes heads proportionally according to weights.
         * Example: 28 heads with weights [0.73, 0.27] → [20, 8] heads
         *
         * @param device Device to get head count for
         * @param total_heads Total attention heads in the model
         * @return Number of heads this device should process
         */
        virtual int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const = 0;

        /**
         * @brief Get row range for a device (for row-parallel matrix sharding)
         *
         * Used for row-parallel operations (e.g., down projection).
         * Returns [start, end) range - end is exclusive.
         *
         * @param device Device to get range for
         * @param total_rows Total rows in the matrix
         * @return Pair of (start_row, end_row) - end is exclusive
         */
        virtual std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const = 0;

        /**
         * @brief Get column range for a device (for column-parallel matrix sharding)
         *
         * Used for column-parallel operations (e.g., Q/K/V projections, FFN up/gate).
         * Returns [start, end) range - end is exclusive.
         *
         * @param device Device to get range for
         * @param total_cols Total columns in the matrix
         * @return Pair of (start_col, end_col) - end is exclusive
         */
        virtual std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const = 0;
    };

    /**
     * @brief Factory function to create a LocalTPContext
     *
     * @param devices Devices participating in LOCAL TP
     * @param weights Work distribution weights (empty for equal distribution)
     * @param backend Backend type (AUTO to select based on device types)
     * @return Unique pointer to ILocalTPContext implementation
     */
    std::unique_ptr<ILocalTPContext> createLocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend);

} // namespace llaminar2

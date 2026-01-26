/**
 * @file LocalTPContext.h
 * @brief Implementation of LOCAL tensor parallelism context
 *
 * Provides concrete implementation of ILocalTPContext for managing
 * tensor parallelism across multiple devices within a single MPI rank.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "ILocalTPContext.h"
#include "DeviceGroup.h"
#include "ICollectiveBackend.h"
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>

namespace llaminar2
{

    /**
     * @brief Concrete implementation of LOCAL tensor parallelism context
     *
     * Manages device list, weight distribution, and collective operations
     * for LOCAL TP (multiple devices within a single MPI rank).
     *
     * Thread safety: All public methods are thread-safe.
     */
    class LocalTPContext : public ILocalTPContext
    {
    public:
        /**
         * @brief Construct a LocalTPContext
         *
         * @param devices Devices participating in LOCAL TP (must be non-empty)
         * @param weights Work distribution weights (empty for equal distribution)
         * @param backend Backend type for collectives (AUTO to detect from devices)
         * @throws std::invalid_argument if devices is empty or weights mismatch
         */
        LocalTPContext(
            std::vector<GlobalDeviceAddress> devices,
            std::vector<float> weights,
            CollectiveBackendType backend);

        ~LocalTPContext() override = default;

        // Disable copy (has mutex)
        LocalTPContext(const LocalTPContext &) = delete;
        LocalTPContext &operator=(const LocalTPContext &) = delete;

        // Enable move
        LocalTPContext(LocalTPContext &&) = default;
        LocalTPContext &operator=(LocalTPContext &&) = default;

        // =====================================================================
        // Configuration (ILocalTPContext)
        // =====================================================================

        const std::vector<GlobalDeviceAddress> &devices() const override;
        const std::vector<float> &weights() const override;
        CollectiveBackendType backend() const override;
        int degree() const override;

        // =====================================================================
        // Collective Operations (ILocalTPContext)
        // =====================================================================

        bool allreduce(TensorBase *tensor) override;
        bool allreduce(const TensorBase *input, TensorBase *output) override;
        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override;
        bool gatherFromDevices(
            const std::vector<const TensorBase *> &shards,
            TensorBase *output) override;
        bool reduceScatter(const TensorBase *input, TensorBase *output_shard) override;

        // =====================================================================
        // Synchronization (ILocalTPContext)
        // =====================================================================

        void synchronize() override;

        // =====================================================================
        // Device Management (ILocalTPContext)
        // =====================================================================

        int indexForDevice(const GlobalDeviceAddress &device) const override;
        const GlobalDeviceAddress &deviceAt(int index) const override;
        float weightForDevice(const GlobalDeviceAddress &device) const override;

        // =====================================================================
        // Weight Sharding Utilities (ILocalTPContext)
        // =====================================================================

        int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override;
        std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const override;
        std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const override;

    private:
        std::vector<GlobalDeviceAddress> devices_;
        std::vector<float> weights_; ///< Normalized weights (sum to 1.0)
        CollectiveBackendType backend_;
        std::unordered_map<GlobalDeviceAddress, int> device_to_index_;

        mutable std::mutex mutex_; ///< Protects collective operations

        /// Backend implementation for collective operations
        std::unique_ptr<ICollectiveBackend> backend_impl_;

        /// Device group for backend initialization
        DeviceGroup device_group_;

        /// Track if backend was successfully initialized
        bool backend_initialized_ = false;

        // =====================================================================
        // PCIeBAR Buffer Registration State
        // =====================================================================
        // For PCIeBAR backend, we must allocate ROCm buffers in the BAR region
        // so the correct offsets are used during cross-vendor allreduce.

        /// Cached buffer size for PCIeBAR allreduce (to detect size changes)
        size_t pciebar_buffer_size_ = 0;

        /// Collective ID for PCIeBAR registered allreduce
        std::string pciebar_collective_id_;

        /// Whether PCIeBAR buffers have been registered
        bool pciebar_buffers_registered_ = false;

        // =====================================================================
        // PCIeBAR Barrier Synchronization State
        // =====================================================================
        // For PCIeBAR backend with heterogeneous GPUs (CUDA + ROCm), threads from
        // different devices call allreduce() concurrently. We need a rendezvous
        // barrier so all devices have contributed their data before the PCIeBAR
        // transfer happens (NCCL-style collective semantics).

        /// Mutex for barrier synchronization (separate from mutex_ to avoid deadlock)
        mutable std::mutex barrier_mutex_;

        /// Condition variable for barrier wait/notify
        std::condition_variable barrier_cv_;

        /// Number of threads that have arrived at the barrier
        std::atomic<int> barrier_count_{0};

        /// Generation counter to prevent spurious wakeups and ensure barrier reusability
        std::atomic<uint64_t> barrier_generation_{0};

        /// Tensor being reduced (set by first arrival, used by executor)
        TensorBase *barrier_tensor_{nullptr};

        /// Result of allreduce (set by executor, read by all waiters)
        bool barrier_result_{false};

        /**
         * @brief Initialize the collective backend
         *
         * Creates the appropriate backend based on backend_ type and initializes it.
         * Called at the end of constructor after devices_ and backend_ are set.
         *
         * @return true if backend was successfully initialized
         */
        bool initializeBackend();

        /**
         * @brief Ensure PCIeBAR buffers are allocated and registered
         *
         * For PCIeBAR backend, ROCm buffers must be allocated within the BAR region
         * to get correct BAR offsets. This method:
         * 1. Allocates ROCm buffer in BAR region (if not already done for this size)
         * 2. Registers both CUDA and ROCm buffers with the backend
         * 3. Returns the collective_id to use with allreduceRegistered()
         *
         * @param tensor Tensor to prepare for allreduce
         * @return true if buffers are ready, false on failure
         */
        bool ensurePCIeBarBuffersRegistered(TensorBase *tensor);

        /**
         * @brief Get device pointers for all devices participating in collective
         *
         * For multi-GPU collectives, we need a buffer pointer for each device.
         * This helper extracts device pointers from a tensor that may have
         * multiple device buffers (one per device in the TP group).
         *
         * @param tensor Tensor with data on all devices
         * @return Vector of device pointers (one per device in devices_)
         */
        std::vector<void *> getDeviceBuffers(TensorBase *tensor);

        /**
         * @brief Convert our data type to CollectiveDataType
         * @param tensor Tensor to get dtype from
         * @return CollectiveDataType for the tensor
         */
        CollectiveDataType tensorDTypeToCollective(const TensorBase *tensor) const;

        /**
         * @brief Internal allreduce implementation (assumes lock is already held)
         *
         * Used by out-of-place allreduce after copying input to output.
         *
         * @param tensor Tensor to allreduce in-place
         * @return true on success
         */
        bool allreduceImpl(TensorBase *tensor);

        /**
         * @brief Allreduce with barrier synchronization for PCIeBAR backend
         *
         * Implements NCCL-style collective semantics where all devices must
         * call allreduce before any data transfer happens. This is necessary
         * for heterogeneous GPU setups where CUDA and ROCm threads run
         * independently and may be at different pipeline stages.
         *
         * The barrier works as follows:
         * 1. First arrivals wait at the barrier
         * 2. Last arrival executes the actual PCIeBAR transfer
         * 3. All devices are released with the same result
         *
         * @param tensor Tensor to allreduce in-place
         * @return true on success (same result for all participants)
         */
        bool allreduceWithBarrier(TensorBase *tensor);

        /**
         * @brief Execute the actual PCIeBAR allreduce operation
         *
         * Called by the last arrival in allreduceWithBarrier(). All other
         * threads are waiting, so we have exclusive access to the barrier_tensor_.
         *
         * @param tensor Tensor to allreduce in-place
         * @return true on success
         */
        bool executePCIeBarAllreduce(TensorBase *tensor);

        /**
         * @brief Normalize weights to sum to 1.0
         * @param weights Input weights (may not sum to 1.0)
         * @return Normalized weights
         */
        static std::vector<float> normalizeWeights(const std::vector<float> &weights);

        /**
         * @brief Compute cumulative counts for range calculations
         *
         * Given total count and weights, computes cumulative counts for
         * proportional distribution. Used by rowRangeForDevice/colRangeForDevice.
         *
         * @param total Total count to distribute
         * @param norm_weights Normalized weights (must sum to 1.0)
         * @return Cumulative counts (length = weights.size() + 1, starts at 0, ends at total)
         */
        static std::vector<int> computeCumulativeCounts(int total, const std::vector<float> &norm_weights);

        /**
         * @brief Auto-detect backend from device types
         *
         * - All CUDA devices → NCCL
         * - All ROCm devices → RCCL
         * - Mixed GPU types → PCIE_BAR
         * - CPU involved → HOST
         *
         * @param devices Device list to analyze
         * @return Detected backend type
         */
        static CollectiveBackendType autoDetectBackend(const std::vector<GlobalDeviceAddress> &devices);

        /**
         * @brief Build device-to-index lookup map
         */
        void buildDeviceIndex();
    };

} // namespace llaminar2

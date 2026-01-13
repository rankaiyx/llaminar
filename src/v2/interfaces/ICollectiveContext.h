/**
 * @file ICollectiveContext.h
 * @brief Interface for collective context operations
 *
 * Abstracts collective communication context to enable:
 * 1. Unit testing without MPI/NCCL runtime
 * 2. Mock collective operations for isolated stage testing
 * 3. Deterministic testing of distributed algorithms
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../collective/ICollectiveBackend.h"
#include "../backends/DeviceId.h"
#include <memory>
#include <vector>

namespace llaminar2 {

// Forward declarations
class ITensor;
class IBackendRouter;
class MPIContext;

/**
 * @brief Abstract interface for collective context operations
 *
 * This interface abstracts the CollectiveContext to enable:
 * - Unit testing of pipeline stages without real MPI/NCCL runtime
 * - Mock collective operations for isolated testing
 * - Injection of test doubles for failure injection and verification
 *
 * Implementations:
 * - CollectiveContext: Real backend-routing implementation
 * - MockCollectiveContext: Test implementation with configurable behavior
 */
class ICollectiveContext {
public:
    virtual ~ICollectiveContext() = default;

    // =========================================================================
    // High-Level Collective Operations
    // These are called by GraphExecutor when executing collective stages
    // =========================================================================

    /**
     * @brief Execute an AllReduce operation
     *
     * Automatically selects the best backend based on tensor location.
     *
     * @param buffer In-place buffer to reduce
     * @param count Number of elements (0 = use buffer->numel())
     * @param tensor_device Device where tensor resides
     * @param op Reduction operation (default: SUM)
     * @return true on success
     */
    virtual bool executeAllreduce(
        ITensor* buffer,
        size_t count,
        DeviceId tensor_device,
        CollectiveOp op = CollectiveOp::ALLREDUCE_SUM) = 0;

    /**
     * @brief Execute an AllGather operation
     *
     * @param local_input Local slice [seq_len, local_dim]
     * @param full_output Full output [seq_len, full_dim]
     * @param actual_seq_len Actual sequence length (0 = use tensor rows)
     * @param tensor_device Device where tensors reside
     * @return true on success
     */
    virtual bool executeAllgather(
        ITensor* local_input,
        ITensor* full_output,
        size_t actual_seq_len,
        DeviceId tensor_device) = 0;

    /**
     * @brief Execute a Broadcast operation
     *
     * @param buffer Buffer to broadcast (in-place)
     * @param count Number of elements
     * @param root_rank Rank that holds the source data
     * @param tensor_device Device where buffer resides
     * @return true on success
     */
    virtual bool executeBroadcast(
        ITensor* buffer,
        size_t count,
        int root_rank,
        DeviceId tensor_device) = 0;

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * @brief Does this configuration require collectives?
     * @return true if world_size > 1
     */
    virtual bool requiresCollectives() const = 0;

    /**
     * @brief Get the world size (number of ranks)
     * @return World size
     */
    virtual int worldSize() const = 0;

    /**
     * @brief Get the local rank
     * @return Rank (0-indexed)
     */
    virtual int rank() const = 0;

    /**
     * @brief Get devices on this rank
     * @return Vector of device IDs
     */
    virtual const std::vector<DeviceId>& localDevices() const = 0;

    /**
     * @brief Check if a specific backend is available
     * @param type Backend type to check
     * @return true if available
     */
    virtual bool isBackendAvailable(CollectiveBackendType type) const = 0;

    // =========================================================================
    // Advanced: Direct Access (for testing/debugging)
    // =========================================================================

    /**
     * @brief Get the backend router interface (internal use)
     * @return Pointer to the router (may be nullptr for mocks)
     */
    virtual IBackendRouter* router() = 0;

    /**
     * @brief Get the MPI context (may be nullptr)
     * @return Pointer to the MPI context
     */
    virtual MPIContext* mpiContext() = 0;
};

} // namespace llaminar2

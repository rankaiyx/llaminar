/**
 * @file RCCLBackend.cpp
 * @brief RCCL-based collective backend implementation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "RCCLBackend.h"
#include "../../utils/Logger.h"

#ifdef HAVE_RCCL
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <thread>
#include <atomic>
#endif

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    RCCLBackend::RCCLBackend(std::shared_ptr<MPIContext> mpi_ctx)
        : mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_DEBUG("RCCLBackend: Created" << (mpi_ctx_ ? " with MPI context (world_size=" + std::to_string(mpi_ctx_->world_size()) + ")" : ""));
    }

    RCCLBackend::~RCCLBackend()
    {
        if (initialized_)
        {
            shutdown();
        }
    }

    // =========================================================================
    // Availability Check
    // =========================================================================

    bool RCCLBackend::isAvailable() const
    {
#ifdef HAVE_RCCL
        int rocm_device_count = 0;
        hipError_t err = hipGetDeviceCount(&rocm_device_count);
        return (err == hipSuccess && rocm_device_count > 0);
#else
        return false;
#endif
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool RCCLBackend::initialize(const DeviceGroup &group)
    {
#ifdef HAVE_RCCL
        if (initialized_)
        {
            LOG_WARN("RCCLBackend::initialize: Already initialized, shutting down first");
            shutdown();
        }

        // Validate all devices are ROCm
        for (const auto &device : group.devices)
        {
            if (device.type != DeviceType::ROCm)
            {
                last_error_ = "RCCLBackend only supports ROCm devices";
                LOG_ERROR(last_error_);
                return false;
            }
        }

        // Determine if this is multi-process (MPI) or single-process
        const bool is_multi_process = mpi_ctx_ && mpi_ctx_->world_size() > 1;

        if (is_multi_process)
        {
            // Multi-process mode: use MPI world size/rank
            num_ranks_ = mpi_ctx_->world_size();
            local_rank_ = mpi_ctx_->rank();
        }
        else
        {
            // Single-process mode: use DeviceGroup size/rank
            num_ranks_ = static_cast<int>(group.size());
            local_rank_ = group.local_rank;
        }

        if (num_ranks_ < 1)
        {
            last_error_ = "RCCLBackend requires at least 1 device";
            LOG_ERROR(last_error_);
            return false;
        }

        // Set the HIP device for this rank
        DeviceId local_device = group.localDevice();
        hipError_t hip_err = hipSetDevice(local_device.ordinal);
        if (hip_err != hipSuccess)
        {
            last_error_ = "Failed to set HIP device " + std::to_string(local_device.ordinal) +
                          ": " + hipGetErrorString(hip_err);
            LOG_ERROR(last_error_);
            return false;
        }

        // Create HIP stream
        hip_err = hipStreamCreate(&stream_);
        if (hip_err != hipSuccess)
        {
            last_error_ = "Failed to create HIP stream: " + std::string(hipGetErrorString(hip_err));
            LOG_ERROR(last_error_);
            return false;
        }

        // Create RCCL communicator
        if (is_multi_process)
        {
            // Multi-process: coordinate via MPI
            // Rank 0 generates unique ID, broadcasts to all ranks
            ncclUniqueId id;

            if (mpi_ctx_->rank() == 0)
            {
                ncclResult_t rccl_err = ncclGetUniqueId(&id);
                if (rccl_err != ncclSuccess)
                {
                    last_error_ = "ncclGetUniqueId failed: " + std::string(ncclGetErrorString(rccl_err));
                    LOG_ERROR(last_error_);
                    hipStreamDestroy(stream_);
                    stream_ = nullptr;
                    return false;
                }
            }

            // Broadcast the unique ID from rank 0 to all other ranks
            int mpi_err = MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_ctx_->comm());
            if (mpi_err != MPI_SUCCESS)
            {
                last_error_ = "MPI_Bcast of ncclUniqueId failed";
                LOG_ERROR(last_error_);
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            // All ranks initialize their communicator with the shared unique ID
            ncclResult_t rccl_err = ncclCommInitRank(&comm_, num_ranks_, id, local_rank_);
            if (rccl_err != ncclSuccess)
            {
                last_error_ = "ncclCommInitRank failed (multi-process): " + std::string(ncclGetErrorString(rccl_err));
                LOG_ERROR(last_error_);
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            LOG_INFO("RCCLBackend: Initialized multi-process with " << num_ranks_
                                                                    << " MPI ranks, local_rank=" << local_rank_
                                                                    << ", device=" << local_device.ordinal);
        }
        else if (num_ranks_ == 1)
        {
            // Single GPU - create a trivial communicator
            ncclResult_t rccl_err = ncclCommInitAll(&comm_, 1, nullptr);
            if (rccl_err != ncclSuccess)
            {
                last_error_ = "ncclCommInitAll failed: " + std::string(ncclGetErrorString(rccl_err));
                LOG_ERROR(last_error_);
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }
            LOG_INFO("RCCLBackend: Initialized single-GPU mode");
        }
        else
        {
            // Multi-GPU single process (no MPI context)
            // Use threaded ncclCommInitRank - each GPU gets its own thread
            // This tests the same code path used in multi-process scenarios
            //
            // ncclCommInitRank requires ALL ranks to call it concurrently,
            // so we spawn N threads, each calling ncclCommInitRank for its rank.

            ncclUniqueId id;
            ncclResult_t rccl_err = ncclGetUniqueId(&id);
            if (rccl_err != ncclSuccess)
            {
                last_error_ = "ncclGetUniqueId failed: " + std::string(ncclGetErrorString(rccl_err));
                LOG_ERROR(last_error_);
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            // Allocate arrays for per-GPU resources
            all_comms_.resize(num_ranks_, nullptr);
            all_streams_.resize(num_ranks_, nullptr);
            device_ordinals_.resize(num_ranks_);
            std::atomic<int> error_count{0};
            std::vector<std::string> thread_errors(num_ranks_);

            // Build device list from group
            for (int i = 0; i < num_ranks_; ++i)
            {
                device_ordinals_[i] = group.devices[i].ordinal;
            }

            // Launch threads - each thread initializes one GPU's communicator and stream
            std::vector<std::thread> threads;
            threads.reserve(num_ranks_);

            for (int rank = 0; rank < num_ranks_; ++rank)
            {
                threads.emplace_back([this, rank, &id, &error_count, &thread_errors]()
                                     {
                    // Set device for this thread
                    hipError_t hip_err = hipSetDevice(device_ordinals_[rank]);
                    if (hip_err != hipSuccess) {
                        thread_errors[rank] = "hipSetDevice failed for rank " + std::to_string(rank) +
                                              ": " + hipGetErrorString(hip_err);
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }

                    // Create stream for this GPU
                    hip_err = hipStreamCreate(&all_streams_[rank]);
                    if (hip_err != hipSuccess) {
                        thread_errors[rank] = "hipStreamCreate failed for rank " + std::to_string(rank) +
                                              ": " + hipGetErrorString(hip_err);
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }

                    // Initialize communicator for this rank
                    // All threads call this concurrently with the same unique ID
                    ncclResult_t err = ncclCommInitRank(&all_comms_[rank], num_ranks_, id, rank);
                    if (err != ncclSuccess) {
                        thread_errors[rank] = "ncclCommInitRank failed for rank " + std::to_string(rank) +
                                              ": " + ncclGetErrorString(err);
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        // Clean up the stream we created
                        hipStreamDestroy(all_streams_[rank]);
                        all_streams_[rank] = nullptr;
                        return;
                    } });
            }

            // Wait for all threads to complete
            for (auto &t : threads)
            {
                t.join();
            }

            // Check for errors
            if (error_count.load() > 0)
            {
                // Collect error messages
                std::string combined_errors;
                for (int i = 0; i < num_ranks_; ++i)
                {
                    if (!thread_errors[i].empty())
                    {
                        if (!combined_errors.empty())
                            combined_errors += "; ";
                        combined_errors += thread_errors[i];
                    }
                }
                last_error_ = "Threaded ncclCommInitRank failed: " + combined_errors;
                LOG_ERROR(last_error_);

                // Cleanup any successfully created resources
                for (int i = 0; i < num_ranks_; ++i)
                {
                    if (all_comms_[i])
                    {
                        ncclCommDestroy(all_comms_[i]);
                        all_comms_[i] = nullptr;
                    }
                    if (all_streams_[i])
                    {
                        hipStreamDestroy(all_streams_[i]);
                        all_streams_[i] = nullptr;
                    }
                }
                all_comms_.clear();
                all_streams_.clear();
                device_ordinals_.clear();
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            // Store the communicator and stream for local_rank
            comm_ = all_comms_[local_rank_];
            // Keep using the main stream_ for single-buffer APIs
            // The all_streams_ array is for multi-buffer APIs

            is_multi_gpu_single_process_ = true;

            LOG_INFO("RCCLBackend: Initialized multi-GPU single-process (threaded ncclCommInitRank) with "
                     << num_ranks_ << " GPU(s), local_rank=" << local_rank_);
        }

        initialized_ = true;
        return true;
#else
        last_error_ = "RCCL not available (HAVE_RCCL not defined)";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    void RCCLBackend::shutdown()
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            return;
        }

        // If we used multi-GPU single-process mode, destroy all resources
        if (!all_comms_.empty())
        {
            for (size_t i = 0; i < all_comms_.size(); ++i)
            {
                if (all_comms_[i])
                {
                    ncclCommDestroy(all_comms_[i]);
                }
                if (i < all_streams_.size() && all_streams_[i])
                {
                    hipStreamDestroy(all_streams_[i]);
                }
            }
            all_comms_.clear();
            all_streams_.clear();
            device_ordinals_.clear();
            comm_ = nullptr; // comm_ was pointing into all_comms_
        }
        else if (comm_)
        {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }

        if (stream_)
        {
            hipStreamDestroy(stream_);
            stream_ = nullptr;
        }

        is_multi_gpu_single_process_ = false;

        initialized_ = false;
        LOG_DEBUG("RCCLBackend: Shutdown complete");
#endif
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool RCCLBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclAllReduce(
            buffer,
            buffer, // In-place: send and recv are the same
            count,
            toRcclDataType(dtype),
            toRcclRedOp(op),
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclAllReduce failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclAllGather(
            send_buf,
            recv_buf,
            send_count,
            toRcclDataType(dtype),
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclAllGather failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        // RCCL does not have a native allgatherv. We emulate it using ncclSend/ncclRecv.
        // For now, we use a simpler approach: regular allgather with max count, then extract.
        // This is less efficient but works correctly.

        // Find max recv count to use as uniform count
        int max_count = 0;
        for (int c : recv_counts)
        {
            max_count = std::max(max_count, c);
        }

        // If all counts are equal, use regular allgather
        bool all_equal = true;
        for (int c : recv_counts)
        {
            if (c != recv_counts[0])
            {
                all_equal = false;
                break;
            }
        }

        if (all_equal)
        {
            return allgather(send_buf, recv_buf, send_count, dtype);
        }

        // Variable counts - RCCL doesn't support this natively
        // Fall back to point-to-point sends/recvs
        ncclDataType_t rccl_dtype = toRcclDataType(dtype);

        ncclResult_t err = ncclGroupStart();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupStart failed";
            return false;
        }

        size_t dtype_size = 0;
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            dtype_size = 4;
            break;
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16:
            dtype_size = 2;
            break;
        case CollectiveDataType::INT32:
            dtype_size = 4;
            break;
        case CollectiveDataType::INT8:
            dtype_size = 1;
            break;
        }

        // Each rank sends to all others and receives from all others
        for (int peer = 0; peer < num_ranks_; ++peer)
        {
            // Send my data to peer
            err = ncclSend(send_buf, send_count, rccl_dtype, peer, comm_, stream_);
            if (err != ncclSuccess)
            {
                ncclGroupEnd();
                last_error_ = "ncclSend failed in allgatherv";
                return false;
            }

            // Receive from peer at their offset
            char *recv_ptr = static_cast<char *>(recv_buf) + displacements[peer] * dtype_size;
            err = ncclRecv(recv_ptr, recv_counts[peer], rccl_dtype, peer, comm_, stream_);
            if (err != ncclSuccess)
            {
                ncclGroupEnd();
                last_error_ = "ncclRecv failed in allgatherv";
                return false;
            }
        }

        err = ncclGroupEnd();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupEnd failed";
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclReduceScatter(
            send_buf,
            recv_buf,
            recv_count,
            toRcclDataType(dtype),
            toRcclRedOp(op),
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclReduceScatter failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclBroadcast(
            buffer,
            buffer,
            count,
            toRcclDataType(dtype),
            root,
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclBroadcast failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Multi-GPU Single-Process Collective Operations
    // =========================================================================

    bool RCCLBackend::isMultiGpuSingleProcess() const
    {
        return is_multi_gpu_single_process_;
    }

    bool RCCLBackend::allreduceMulti(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allreduceMulti requires multi-GPU single-process mode";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match GPU count (" + std::to_string(num_ranks_) + ")";
            return false;
        }

        // Use ncclGroupStart/End to issue all operations atomically
        ncclResult_t err = ncclGroupStart();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupStart failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        // Issue AllReduce on each GPU with its buffer, communicator, and stream
        for (int i = 0; i < num_ranks_; ++i)
        {
            // Set device context (required for RCCL to know which GPU)
            hipSetDevice(device_ordinals_[i]);

            err = ncclAllReduce(
                buffers[i],
                buffers[i], // In-place
                count,
                toRcclDataType(dtype),
                toRcclRedOp(op),
                all_comms_[i],
                all_streams_[i]);

            if (err != ncclSuccess)
            {
                ncclGroupEnd(); // Try to clean up
                last_error_ = "ncclAllReduce failed on GPU " + std::to_string(i) +
                              ": " + std::string(ncclGetErrorString(err));
                LOG_ERROR(last_error_);
                return false;
            }
        }

        err = ncclGroupEnd();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupEnd failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::allgatherMulti(
        const std::vector<const void *> &send_bufs,
        const std::vector<void *> &recv_bufs,
        size_t send_count,
        CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allgatherMulti requires multi-GPU single-process mode";
            return false;
        }

        if (send_bufs.size() != static_cast<size_t>(num_ranks_) ||
            recv_bufs.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count doesn't match GPU count";
            return false;
        }

        ncclResult_t err = ncclGroupStart();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupStart failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        for (int i = 0; i < num_ranks_; ++i)
        {
            hipSetDevice(device_ordinals_[i]);

            err = ncclAllGather(
                send_bufs[i],
                recv_bufs[i],
                send_count,
                toRcclDataType(dtype),
                all_comms_[i],
                all_streams_[i]);

            if (err != ncclSuccess)
            {
                ncclGroupEnd();
                last_error_ = "ncclAllGather failed on GPU " + std::to_string(i) +
                              ": " + std::string(ncclGetErrorString(err));
                LOG_ERROR(last_error_);
                return false;
            }
        }

        err = ncclGroupEnd();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupEnd failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)send_bufs;
        (void)recv_bufs;
        (void)send_count;
        (void)dtype;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::broadcastMulti(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        int root)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "broadcastMulti requires multi-GPU single-process mode";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count doesn't match GPU count";
            return false;
        }

        if (root < 0 || root >= num_ranks_)
        {
            last_error_ = "Invalid root rank: " + std::to_string(root);
            return false;
        }

        ncclResult_t err = ncclGroupStart();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupStart failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        for (int i = 0; i < num_ranks_; ++i)
        {
            hipSetDevice(device_ordinals_[i]);

            err = ncclBroadcast(
                buffers[i],
                buffers[i],
                count,
                toRcclDataType(dtype),
                root,
                all_comms_[i],
                all_streams_[i]);

            if (err != ncclSuccess)
            {
                ncclGroupEnd();
                last_error_ = "ncclBroadcast failed on GPU " + std::to_string(i) +
                              ": " + std::string(ncclGetErrorString(err));
                LOG_ERROR(last_error_);
                return false;
            }
        }

        err = ncclGroupEnd();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupEnd failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)root;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    bool RCCLBackend::synchronize()
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            return true;
        }

        // In multi-GPU single-process mode, synchronize ALL streams
        if (is_multi_gpu_single_process_)
        {
            for (int i = 0; i < num_ranks_; ++i)
            {
                hipSetDevice(device_ordinals_[i]);
                hipError_t err = hipStreamSynchronize(all_streams_[i]);
                if (err != hipSuccess)
                {
                    last_error_ = "hipStreamSynchronize failed on GPU " + std::to_string(i);
                    LOG_ERROR(last_error_);
                    return false;
                }
            }
            return true;
        }

        // Single-GPU or MPI mode: just sync the main stream
        hipError_t err = hipStreamSynchronize(stream_);
        if (err != hipSuccess)
        {
            last_error_ = "hipStreamSynchronize failed";
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        return true;
#endif
    }

    // =========================================================================
    // Type Conversion Helpers
    // =========================================================================

#ifdef HAVE_RCCL
    ncclDataType_t RCCLBackend::toRcclDataType(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return ncclFloat32;
        case CollectiveDataType::FLOAT16:
            return ncclFloat16;
        case CollectiveDataType::BFLOAT16:
            return ncclBfloat16;
        case CollectiveDataType::INT32:
            return ncclInt32;
        case CollectiveDataType::INT8:
            return ncclInt8;
        default:
            LOG_WARN("RCCLBackend: Unknown dtype, defaulting to float32");
            return ncclFloat32;
        }
    }

    ncclRedOp_t RCCLBackend::toRcclRedOp(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
            return ncclSum;
        case CollectiveOp::ALLREDUCE_MAX:
            return ncclMax;
        case CollectiveOp::ALLREDUCE_MIN:
            return ncclMin;
        default:
            LOG_WARN("RCCLBackend: Unknown op, defaulting to SUM");
            return ncclSum;
        }
    }
#endif

} // namespace llaminar2

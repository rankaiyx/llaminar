/**
 * @file NCCLBackend.cpp
 * @brief NCCL-based collective backend implementation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "NCCLBackend.h"
#include "../../utils/Logger.h"

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#endif

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    NCCLBackend::NCCLBackend()
    {
        LOG_DEBUG("NCCLBackend: Created");
    }

    NCCLBackend::~NCCLBackend()
    {
        if (initialized_)
        {
            shutdown();
        }
    }

    // =========================================================================
    // Availability Check
    // =========================================================================

    bool NCCLBackend::isAvailable() const
    {
#ifdef HAVE_NCCL
        int cuda_device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
        return (err == cudaSuccess && cuda_device_count > 0);
#else
        return false;
#endif
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool NCCLBackend::initialize(const DeviceGroup &group)
    {
#ifdef HAVE_NCCL
        if (initialized_)
        {
            LOG_WARN("NCCLBackend::initialize: Already initialized, shutting down first");
            shutdown();
        }

        // Validate all devices are CUDA
        for (const auto &device : group.devices)
        {
            if (device.type != DeviceType::CUDA)
            {
                last_error_ = "NCCLBackend only supports CUDA devices";
                LOG_ERROR(last_error_);
                return false;
            }
        }

        num_ranks_ = static_cast<int>(group.size());
        local_rank_ = group.local_rank;

        if (num_ranks_ < 1)
        {
            last_error_ = "NCCLBackend requires at least 1 device";
            LOG_ERROR(last_error_);
            return false;
        }

        // Set the CUDA device for this rank
        DeviceId local_device = group.localDevice();
        cudaError_t cuda_err = cudaSetDevice(local_device.ordinal);
        if (cuda_err != cudaSuccess)
        {
            last_error_ = "Failed to set CUDA device " + std::to_string(local_device.ordinal);
            LOG_ERROR(last_error_);
            return false;
        }

        // Create CUDA stream
        cuda_err = cudaStreamCreate(&stream_);
        if (cuda_err != cudaSuccess)
        {
            last_error_ = "Failed to create CUDA stream";
            LOG_ERROR(last_error_);
            return false;
        }

        // Create NCCL communicator
        // For single-process multi-GPU, we use ncclCommInitAll
        // For multi-process, we'd use ncclCommInitRank with unique ID
        if (num_ranks_ == 1)
        {
            // Single GPU - create a trivial communicator
            ncclResult_t nccl_err = ncclCommInitAll(&comm_, 1, nullptr);
            if (nccl_err != ncclSuccess)
            {
                last_error_ = "ncclCommInitAll failed: " + std::string(ncclGetErrorString(nccl_err));
                LOG_ERROR(last_error_);
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }
        }
        else
        {
            // Multi-GPU single process
            // TODO: For multi-process, need to broadcast unique ID via MPI
            ncclUniqueId id;
            ncclResult_t nccl_err = ncclGetUniqueId(&id);
            if (nccl_err != ncclSuccess)
            {
                last_error_ = "ncclGetUniqueId failed";
                LOG_ERROR(last_error_);
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            nccl_err = ncclCommInitRank(&comm_, num_ranks_, id, local_rank_);
            if (nccl_err != ncclSuccess)
            {
                last_error_ = "ncclCommInitRank failed: " + std::string(ncclGetErrorString(nccl_err));
                LOG_ERROR(last_error_);
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }
        }

        initialized_ = true;
        LOG_INFO("NCCLBackend: Initialized with " << num_ranks_ << " GPU(s), local_rank=" << local_rank_);
        return true;
#else
        last_error_ = "NCCL not available (HAVE_NCCL not defined)";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    void NCCLBackend::shutdown()
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            return;
        }

        if (comm_)
        {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }

        if (stream_)
        {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        initialized_ = false;
        LOG_DEBUG("NCCLBackend: Shutdown complete");
#endif
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool NCCLBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclAllReduce(
            buffer,
            buffer, // In-place: send and recv are the same
            count,
            toNcclDataType(dtype),
            toNcclRedOp(op),
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
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclAllGather(
            send_buf,
            recv_buf,
            send_count,
            toNcclDataType(dtype),
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
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclReduceScatter(
            send_buf,
            recv_buf,
            recv_count,
            toNcclDataType(dtype),
            toNcclRedOp(op),
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
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclBroadcast(
            buffer,
            buffer,
            count,
            toNcclDataType(dtype),
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
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    bool NCCLBackend::synchronize()
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            return true;
        }

        cudaError_t err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess)
        {
            last_error_ = "cudaStreamSynchronize failed";
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

#ifdef HAVE_NCCL
    ncclDataType_t NCCLBackend::toNcclDataType(CollectiveDataType dtype)
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
            LOG_WARN("NCCLBackend: Unknown dtype, defaulting to float32");
            return ncclFloat32;
        }
    }

    ncclRedOp_t NCCLBackend::toNcclRedOp(CollectiveOp op)
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
            LOG_WARN("NCCLBackend: Unknown op, defaulting to SUM");
            return ncclSum;
        }
    }
#endif

} // namespace llaminar2

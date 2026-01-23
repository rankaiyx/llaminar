/**
 * @file NCCLBackendCUDA.cu
 * @brief CUDA and NCCL-specific helper functions for NCCLBackend
 *
 * Isolated CUDA runtime and NCCL API calls in separate compilation unit to avoid
 * conflicts with HIP headers when building with both CUDA and ROCm support.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <string>
#include <cstring>

namespace llaminar2
{
    namespace nccl_backend_detail
    {

        // =========================================================================
        // Device Management
        // =========================================================================

        bool cudaSetDeviceOrdinal(int device_ordinal)
        {
            cudaError_t err = cudaSetDevice(device_ordinal);
            return (err == cudaSuccess);
        }

        bool cudaGetDeviceCountWrapper(int *count)
        {
            cudaError_t err = cudaGetDeviceCount(count);
            return (err == cudaSuccess);
        }

        // =========================================================================
        // Stream Management
        // =========================================================================

        bool cudaCreateStream(void **stream_ptr)
        {
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err == cudaSuccess)
            {
                *stream_ptr = static_cast<void *>(stream);
                return true;
            }
            *stream_ptr = nullptr;
            return false;
        }

        bool cudaDestroyStream(void *stream)
        {
            if (stream)
            {
                cudaError_t err = cudaStreamDestroy(static_cast<cudaStream_t>(stream));
                return (err == cudaSuccess);
            }
            return true;
        }

        bool cudaSynchronizeStream(void *stream)
        {
            if (!stream)
            {
                return false;
            }
            cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            return (err == cudaSuccess);
        }

        // =========================================================================
        // Error Handling
        // =========================================================================

        std::string cudaGetLastErrorString()
        {
            cudaError_t err = cudaGetLastError();
            return std::string(cudaGetErrorString(err));
        }

        // =========================================================================
        // NCCL Unique ID Management
        // =========================================================================

        // Size of ncclUniqueId for serialization
        size_t ncclUniqueIdSize()
        {
            return sizeof(ncclUniqueId);
        }

        bool ncclGetUniqueIdWrapper(void *id_out)
        {
            ncclUniqueId *id = static_cast<ncclUniqueId *>(id_out);
            ncclResult_t r = ncclGetUniqueId(id);
            return (r == ncclSuccess);
        }

        // =========================================================================
        // NCCL Communicator Management
        // =========================================================================

        bool ncclCommInitRankWrapper(void **comm_out, int nranks, void *unique_id, int rank, std::string &error_out)
        {
            ncclComm_t comm;
            ncclResult_t r = ncclCommInitRank(&comm, nranks, *static_cast<ncclUniqueId *>(unique_id), rank);
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                *comm_out = nullptr;
                return false;
            }
            *comm_out = static_cast<void *>(comm);
            return true;
        }

        bool ncclCommInitAllWrapper(void **comms_out, int ndevs, const int *devlist, std::string &error_out)
        {
            ncclComm_t *comms = new ncclComm_t[ndevs];
            ncclResult_t r = ncclCommInitAll(comms, ndevs, devlist);
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                delete[] comms;
                *comms_out = nullptr;
                return false;
            }
            // For single device, just return the first comm
            *comms_out = static_cast<void *>(comms[0]);
            delete[] comms;
            return true;
        }

        void ncclCommDestroyWrapper(void *comm)
        {
            if (comm)
            {
                ncclCommDestroy(static_cast<ncclComm_t>(comm));
            }
        }

        // =========================================================================
        // NCCL Data Type Conversion
        // =========================================================================

        // Internal conversion from our enum values to NCCL types
        // We use integers to avoid exposing NCCL types in the header
        ncclDataType_t toNcclDataType(int dtype_int)
        {
            // These values must match CollectiveDataType enum
            switch (dtype_int)
            {
            case 0: // FLOAT32
                return ncclFloat;
            case 1: // FLOAT16
                return ncclHalf;
            case 2: // BFLOAT16
                return ncclBfloat16;
            case 3: // INT32
                return ncclInt32;
            case 4: // INT8
                return ncclInt8;
            default:
                return ncclFloat;
            }
        }

        ncclRedOp_t toNcclRedOp(int op_int)
        {
            // These values must match CollectiveOp enum
            switch (op_int)
            {
            case 0: // SUM
                return ncclSum;
            case 1: // PROD
                return ncclProd;
            case 2: // MIN
                return ncclMin;
            case 3: // MAX
                return ncclMax;
            default:
                return ncclSum;
            }
        }

        // =========================================================================
        // NCCL Collective Operations
        // =========================================================================

        bool ncclAllReduceWrapper(void *sendbuff, void *recvbuff, size_t count,
                                  int dtype_int, int op_int, void *comm, void *stream,
                                  std::string &error_out)
        {
            ncclResult_t r = ncclAllReduce(sendbuff, recvbuff, count,
                                           toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                           static_cast<ncclComm_t>(comm),
                                           static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclAllGatherWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                  int dtype_int, void *comm, void *stream,
                                  std::string &error_out)
        {
            ncclResult_t r = ncclAllGather(sendbuff, recvbuff, sendcount,
                                           toNcclDataType(dtype_int),
                                           static_cast<ncclComm_t>(comm),
                                           static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclBroadcastWrapper(void *buff, size_t count, int dtype_int, int root,
                                  void *comm, void *stream, std::string &error_out)
        {
            ncclResult_t r = ncclBroadcast(buff, buff, count, toNcclDataType(dtype_int), root,
                                           static_cast<ncclComm_t>(comm),
                                           static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclReduceScatterWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                      int dtype_int, int op_int, void *comm, void *stream,
                                      std::string &error_out)
        {
            ncclResult_t r = ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                               toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                               static_cast<ncclComm_t>(comm),
                                               static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // =========================================================================
        // NCCL Group Operations (for multi-GPU single process)
        // =========================================================================

        bool ncclGroupStartWrapper(std::string &error_out)
        {
            ncclResult_t r = ncclGroupStart();
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclGroupEndWrapper(std::string &error_out)
        {
            ncclResult_t r = ncclGroupEnd();
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // Batched operations within a group
        bool ncclAllReduceInGroupWrapper(void *sendbuff, void *recvbuff, size_t count,
                                         int dtype_int, int op_int, void *comm, void *stream,
                                         std::string &error_out)
        {
            ncclResult_t r = ncclAllReduce(sendbuff, recvbuff, count,
                                           toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                           static_cast<ncclComm_t>(comm),
                                           static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclAllGatherInGroupWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                         int dtype_int, void *comm, void *stream,
                                         std::string &error_out)
        {
            ncclResult_t r = ncclAllGather(sendbuff, recvbuff, sendcount,
                                           toNcclDataType(dtype_int),
                                           static_cast<ncclComm_t>(comm),
                                           static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclBroadcastInGroupWrapper(void *buff, size_t count, int dtype_int, int root,
                                         void *comm, void *stream, std::string &error_out)
        {
            ncclResult_t r = ncclBroadcast(buff, buff, count, toNcclDataType(dtype_int), root,
                                           static_cast<ncclComm_t>(comm),
                                           static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // =========================================================================
        // Point-to-Point Operations (for allgatherv emulation)
        // =========================================================================

        bool ncclSendWrapper(const void *sendbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out)
        {
            ncclResult_t r = ncclSend(sendbuff, count, toNcclDataType(dtype_int), peer,
                                      static_cast<ncclComm_t>(comm),
                                      static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclRecvWrapper(void *recvbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out)
        {
            ncclResult_t r = ncclRecv(recvbuff, count, toNcclDataType(dtype_int), peer,
                                      static_cast<ncclComm_t>(comm),
                                      static_cast<cudaStream_t>(stream));
            if (r != ncclSuccess)
            {
                error_out = ncclGetErrorString(r);
                return false;
            }
            return true;
        }

    } // namespace nccl_backend_detail
} // namespace llaminar2

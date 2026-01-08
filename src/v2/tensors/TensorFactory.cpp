/**
 * @file TensorFactory.cpp
 * @brief Factory implementation for creating tensors with NUMA-aware allocation
 * @author David Sanftenberg
 */

#include "TensorFactory.h"
#include "BlockStructures.h"
#include "../utils/Logger.h"
#include <stdexcept>
#include <sstream>

#ifdef HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#ifdef HAVE_CUDA
#include "cuda/CUDATypedTensor.h"
#endif

namespace llaminar2
{
    TensorFactory::TensorFactory(const MPIContext &mpi_ctx)
        : mpi_rank_(mpi_ctx.rank()), numa_node_(-1)
    {
        // Determine NUMA node for this MPI rank
        numa_node_ = getNumaNodeForRank(mpi_rank_);

        // Bind current thread to NUMA node if available
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }
    }

    std::unique_ptr<FP32Tensor> TensorFactory::createFP32(const std::vector<size_t> &shape, DeviceId device)
    {
        if (device.is_gpu())
        {
            // NOTE: createFP32 returns unique_ptr<FP32Tensor> (CPU-specific).
            // For GPU tensors, use create(TensorType::FP32, shape, device) which returns unique_ptr<ITensor>
            // and can create CUDAFp32Tensor. However, callers using shared_ptr<TensorBase> (like GraphOrchestrator)
            // need refactoring to use shared_ptr<ITensor> before GPU tensors can be used.
            LOG_ERROR("TensorFactory::createFP32: Returns CPU tensor type. For GPU, use create(TensorType::FP32, ...) -> ITensor");
            throw std::runtime_error("createFP32() returns CPU-specific FP32Tensor. Use create(TensorType::FP32, device) for GPU.");
        }

        // Ensure we're on the correct NUMA node for CPU tensors
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // CPU tensor: use legacy device_idx for now (will be updated when CPUTensorBase migrates to DeviceId)
        return std::make_unique<FP32Tensor>(shape, device.toLegacyIndex());
    }

    std::unique_ptr<FP16Tensor> TensorFactory::createFP16(const std::vector<size_t> &shape, DeviceId device)
    {
        if (device.is_gpu())
        {
            LOG_ERROR("TensorFactory::createFP16: GPU tensor creation not yet implemented for device " << device.to_string());
            throw std::runtime_error("GPU tensor creation not yet implemented");
        }

        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<FP16Tensor>(shape);
    }

    std::unique_ptr<FP16Tensor> TensorFactory::createFP16(const std::vector<size_t> &shape,
                                                          const std::vector<uint16_t> &fp16_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<FP16Tensor>(shape, fp16_data);
    }

    std::unique_ptr<BF16Tensor> TensorFactory::createBF16(const std::vector<size_t> &shape, DeviceId device)
    {
        if (device.is_gpu())
        {
            LOG_ERROR("TensorFactory::createBF16: GPU tensor creation not yet implemented for device " << device.to_string());
            throw std::runtime_error("GPU tensor creation not yet implemented");
        }

        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<BF16Tensor>(shape);
    }

    std::unique_ptr<BF16Tensor> TensorFactory::createBF16(const std::vector<size_t> &shape,
                                                          const std::vector<uint16_t> &bf16_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<BF16Tensor>(shape, bf16_data);
    }

    std::unique_ptr<INT32Tensor> TensorFactory::createINT32(const std::vector<size_t> &shape, DeviceId device)
    {
        if (device.is_gpu())
        {
            LOG_ERROR("TensorFactory::createINT32: GPU tensor creation not yet implemented for device " << device.to_string());
            throw std::runtime_error("GPU tensor creation not yet implemented");
        }

        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<INT32Tensor>(shape);
    }

    std::unique_ptr<Q8_1Tensor> TensorFactory::createQ8_1(const std::vector<size_t> &shape, DeviceId device)
    {
        if (device.is_gpu())
        {
            LOG_ERROR("TensorFactory::createQ8_1: GPU tensor creation not yet implemented for device " << device.to_string());
            throw std::runtime_error("GPU tensor creation not yet implemented");
        }

        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor (no raw_data, allocates internally)
        // This creates a Q8_1Tensor that supports mutable_data() and quantize_from_cache()
        return std::make_unique<Q8_1Tensor>(shape, device.toLegacyIndex());
    }

    std::unique_ptr<Q8_1Tensor> TensorFactory::createQ8_1(const std::vector<size_t> &shape,
                                                          const std::vector<uint8_t> &raw_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<Q8_1Tensor>(shape, raw_data);
    }

    std::unique_ptr<Q16_1Tensor> TensorFactory::createQ16_1(const std::vector<size_t> &shape,
                                                            DeviceId device)
    {
        if (device.is_gpu())
        {
            LOG_ERROR("TensorFactory::createQ16_1: GPU tensor creation not yet implemented for device " << device.to_string());
            throw std::runtime_error("GPU tensor creation not yet implemented");
        }

        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor
        // Q16_1Tensor(shape, device_idx) creates an empty tensor for residual accumulation
        return std::make_unique<Q16_1Tensor>(shape, device.toLegacyIndex());
    }

    std::unique_ptr<Q16_1Tensor> TensorFactory::createQ16_1(const std::vector<size_t> &shape,
                                                            Q16BlockSize block_size,
                                                            DeviceId device)
    {
        if (device.is_gpu())
        {
            LOG_ERROR("TensorFactory::createQ16_1: GPU tensor creation not yet implemented for device " << device.to_string());
            throw std::runtime_error("GPU tensor creation not yet implemented");
        }

        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor with custom block size
        return std::make_unique<Q16_1Tensor>(shape, block_size, device.toLegacyIndex());
    }

    std::unique_ptr<ITensor> TensorFactory::create(TensorType type,
                                                   const std::vector<size_t> &shape,
                                                   DeviceId device)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda())
        {
            // Create CUDA tensor based on type
            switch (type)
            {
            case TensorType::FP32:
                return std::make_unique<CUDAFp32Tensor>(shape, device.ordinal);
            case TensorType::FP16:
                return std::make_unique<CUDAFP16Tensor>(shape, device.ordinal);
            case TensorType::BF16:
                return std::make_unique<CUDABF16Tensor>(shape, device.ordinal);
            case TensorType::INT8:
                return std::make_unique<CUDAINT8Tensor>(shape, device.ordinal);
            case TensorType::INT32:
                return std::make_unique<CUDAINT32Tensor>(shape, device.ordinal);
            case TensorType::Q8_1:
                return std::make_unique<CUDAQ8_1Tensor>(shape, device.ordinal);
            case TensorType::Q16_1:
                // Default Q16_1 uses 32-element blocks
                return std::make_unique<CUDAQ16_1Tensor>(shape, device.ordinal);
            default:
                LOG_ERROR("TensorFactory::create: CUDA tensor type "
                          << static_cast<int>(type) << " not yet supported");
                throw std::runtime_error("CUDA tensor type not yet supported");
            }
        }
#endif

        if (device.is_gpu())
        {
            // ROCm or other GPU backend - not yet implemented
            LOG_ERROR("TensorFactory::create: GPU tensor creation not yet implemented for device " << device.to_string());
            throw std::runtime_error("GPU tensor creation not yet implemented");
        }

        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        switch (type)
        {
        case TensorType::FP32:
            return createFP32(shape, device);
        case TensorType::FP16:
            return createFP16(shape, device);
        case TensorType::BF16:
            return createBF16(shape, device);
        case TensorType::INT32:
            return createINT32(shape, device);
        case TensorType::Q8_1:
            return createQ8_1(shape, device);
        case TensorType::Q16_1:
            return createQ16_1(shape, device);
        default:
            LOG_ERROR("TensorFactory::create: unsupported type " << static_cast<int>(type));
            std::ostringstream oss;
            oss << "TensorFactory::create: unsupported type " << static_cast<int>(type)
                << " - use createQuantized for weight tensors";
            throw std::runtime_error(oss.str());
        }
    }

    std::unique_ptr<ITensor> TensorFactory::createActivation(const std::vector<size_t> &shape,
                                                             ActivationPrecision precision,
                                                             DeviceId device)
    {
        // Map ActivationPrecision to TensorType for device-agnostic creation
        TensorType type;
        switch (precision)
        {
        case ActivationPrecision::FP32:
            type = TensorType::FP32;
            break;
        case ActivationPrecision::BF16:
            type = TensorType::BF16;
            break;
        case ActivationPrecision::FP16:
            type = TensorType::FP16;
            break;
        case ActivationPrecision::Q8_1:
            type = TensorType::Q8_1;
            break;
        case ActivationPrecision::Q16_1:
        case ActivationPrecision::HybridQ16:
            // Q16_1/HybridQ16 default block size - use head_dim overload for explicit size
            type = TensorType::Q16_1;
            break;
        default:
            LOG_ERROR("TensorFactory::createActivation: unknown precision, defaulting to FP32");
            type = TensorType::FP32;
            break;
        }

        // Delegate to device-agnostic create() method
        return create(type, shape, device);
    }

    std::unique_ptr<ITensor> TensorFactory::createActivation(const std::vector<size_t> &shape,
                                                             ActivationPrecision precision,
                                                             int head_dim,
                                                             DeviceId device)
    {
        // For Q16_1 and HybridQ16, use optimal block size based on head_dim
        if (precision == ActivationPrecision::Q16_1 ||
            precision == ActivationPrecision::HybridQ16)
        {
            Q16BlockSize block_size = optimal_q16_block_size(head_dim);
            LOG_DEBUG("TensorFactory::createActivation: Q16 with head_dim=" << head_dim
                                                                            << " -> block_size=" << static_cast<int>(block_size));

#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                // Create CUDA Q16_1 tensor with appropriate block size
                switch (block_size)
                {
                case Q16BlockSize::BLOCK_32:
                    return std::make_unique<CUDAQ16_1Tensor>(shape, device.ordinal);
                case Q16BlockSize::BLOCK_64:
                    return std::make_unique<CUDAQ16_1_64Tensor>(shape, device.ordinal);
                case Q16BlockSize::BLOCK_128:
                    return std::make_unique<CUDAQ16_1_128Tensor>(shape, device.ordinal);
                default:
                    LOG_ERROR("TensorFactory::createActivation: Unsupported Q16 block size "
                              << static_cast<int>(block_size) << " on CUDA");
                    throw std::runtime_error("Unsupported Q16 block size on CUDA");
                }
            }
#endif
            if (device.is_gpu())
            {
                LOG_ERROR("TensorFactory::createActivation: Q16_1 with head_dim not supported on non-CUDA GPU");
                throw std::runtime_error("Q16_1 with head_dim not supported on non-CUDA GPU");
            }
            return createQ16_1(shape, block_size, device);
        }

        // All other precisions: delegate to base overload
        return createActivation(shape, precision, device);
    }

    std::unique_ptr<CPUTensorBase> TensorFactory::createQuantized(TensorType type,
                                                                  const std::vector<size_t> &shape,
                                                                  const std::vector<uint8_t> &raw_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        switch (type)
        {
        case TensorType::IQ4_NL:
            return std::make_unique<IQ4_NLTensor>(shape, raw_data);
        case TensorType::Q8_0:
            return std::make_unique<Q8_0Tensor>(shape, raw_data);
        case TensorType::Q4_0:
            return std::make_unique<Q4_0Tensor>(shape, raw_data);
        case TensorType::Q4_1:
            return std::make_unique<Q4_1Tensor>(shape, raw_data);
        case TensorType::Q5_0:
            return std::make_unique<Q5_0Tensor>(shape, raw_data);
        case TensorType::Q5_1:
            return std::make_unique<Q5_1Tensor>(shape, raw_data);
        case TensorType::Q6_K:
            return std::make_unique<Q6_KTensor>(shape, raw_data);
        case TensorType::Q2_K:
            return std::make_unique<Q2_KTensor>(shape, raw_data);
        case TensorType::Q5_K:
            return std::make_unique<Q5_KTensor>(shape, raw_data);
        case TensorType::Q3_K:
            return std::make_unique<Q3_KTensor>(shape, raw_data);
        case TensorType::Q4_K:
            return std::make_unique<Q4_KTensor>(shape, raw_data);
        case TensorType::Q8_K:
            return std::make_unique<Q8_KTensor>(shape, raw_data);
        case TensorType::IQ4_XS:
            return std::make_unique<IQ4_XSTensor>(shape, raw_data);
        case TensorType::IQ2_XXS:
            return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
        case TensorType::IQ2_XS:
            return std::make_unique<IQ2_XSTensor>(shape, raw_data);
        case TensorType::IQ3_XXS:
            return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
        case TensorType::IQ2_S:
            return std::make_unique<IQ2_STensor>(shape, raw_data);
        case TensorType::IQ3_S:
            return std::make_unique<IQ3_STensor>(shape, raw_data);
        case TensorType::IQ1_S:
            return std::make_unique<IQ1_STensor>(shape, raw_data);
        case TensorType::IQ1_M:
            return std::make_unique<IQ1_MTensor>(shape, raw_data);
        default:
            LOG_ERROR("TensorFactory::createQuantized: unsupported type " << static_cast<int>(type));
            std::ostringstream oss;
            oss << "TensorFactory::createQuantized: unsupported type " << static_cast<int>(type);
            throw std::runtime_error(oss.str());
        }
    }

    bool TensorFactory::isNumaAvailable()
    {
#ifdef HAVE_NUMA
        if (numa_available() < 0)
        {
            return false;
        }

        // Check if system has multiple NUMA nodes
        int max_node = numa_max_node();
        return max_node > 0;
#else
        return false;
#endif
    }

    void TensorFactory::bindToNumaNode()
    {
#ifdef HAVE_NUMA
        if (numa_node_ >= 0 && numa_available() >= 0)
        {
            // Bind current thread to NUMA node
            struct bitmask *mask = numa_allocate_nodemask();
            numa_bitmask_clearall(mask);
            numa_bitmask_setbit(mask, numa_node_);

            // Set memory binding policy for allocations
            numa_set_membind(mask);

            // Also bind CPU affinity (optional, helps with first-touch)
            // numa_run_on_node_mask(mask);

            numa_free_nodemask(mask);
        }
#endif
    }

    int TensorFactory::getNumaNodeForRank(int rank)
    {
#ifdef HAVE_NUMA
        if (numa_available() < 0)
        {
            return -1;
        }

        int max_node = numa_max_node();
        if (max_node <= 0)
        {
            return -1; // Only one NUMA node
        }

        // Simple round-robin mapping: rank % (max_node + 1)
        // More sophisticated mapping could query actual CPU topology
        return rank % (max_node + 1);
#else
        (void)rank; // Suppress unused parameter warning
        return -1;
#endif
    }

#ifdef HAVE_CUDA
    std::unique_ptr<ITensor> TensorFactory::createCUDATensor(TensorType type,
                                                             const std::vector<size_t> &shape,
                                                             int cuda_ordinal)
    {
        switch (type)
        {
        case TensorType::FP32:
            return std::make_unique<CUDAFp32Tensor>(shape, cuda_ordinal);
        default:
            LOG_ERROR("TensorFactory::createCUDATensor: unsupported CUDA tensor type "
                      << static_cast<int>(type));
            throw std::runtime_error("Unsupported CUDA tensor type");
        }
    }
#endif

} // namespace llaminar2

/**
 * @file WeightPreloader.cpp
 * @brief Implementation of weight pre-packing before graph execution
 */

#include "WeightPreloader.h"
#include "../kernels/KernelFactory.h"
#include "../utils/Logger.h"
#include "../backends/ComputeBackend.h"
#include "../backends/DeviceId.h"
#include "../tensors/TensorClasses.h"
#include <cstring>

#ifdef HAVE_ROCM
#include "../kernels/rocm/ROCmQuantisedGemmKernel.h"
#endif

#ifdef HAVE_CUDA
#include "../kernels/cuda/CUDAQuantisedGemmKernel.h"
#endif

namespace llaminar2
{

    namespace
    {

        /**
         * @brief Create a quantized tensor from raw bytes (generic for all tensor types)
         *
         * This mirrors TensorFactory::createQuantized() but doesn't require an instance.
         * Handles all 27+ quantized tensor types in a single function.
         *
         * @param type Tensor type enum
         * @param shape Tensor dimensions
         * @param raw_data Raw bytes (moved into tensor)
         * @return New tensor of the specified type, or nullptr on failure
         */
        std::unique_ptr<TensorBase> createQuantizedTensorFromRawData(
            TensorType type,
            const std::vector<size_t> &shape,
            std::vector<uint8_t> raw_data)
        {
            switch (type)
            {
            case TensorType::Q4_0:
                return std::make_unique<Q4_0Tensor>(shape, raw_data);
            case TensorType::Q8_0:
                return std::make_unique<Q8_0Tensor>(shape, raw_data);
            case TensorType::Q8_1:
                return std::make_unique<Q8_1Tensor>(shape, raw_data);
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
            case TensorType::IQ4_NL:
                return std::make_unique<IQ4_NLTensor>(shape, raw_data);
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
            case TensorType::BF16:
                // BF16 uses uint16_t data, need to reinterpret
                {
                    std::vector<uint16_t> bf16_data(raw_data.size() / 2);
                    std::memcpy(bf16_data.data(), raw_data.data(), raw_data.size());
                    return std::make_unique<BF16Tensor>(shape, bf16_data);
                }
            case TensorType::FP16:
                // FP16 uses uint16_t data, need to reinterpret
                {
                    std::vector<uint16_t> fp16_data(raw_data.size() / 2);
                    std::memcpy(fp16_data.data(), raw_data.data(), raw_data.size());
                    return std::make_unique<FP16Tensor>(shape, fp16_data);
                }
            default:
                LOG_ERROR("[WeightPreloader] Unsupported tensor type for cloning: "
                          << static_cast<int>(type));
                return nullptr;
            }
        }

    } // anonymous namespace

    WeightPreloader::WeightPreloader(
        std::shared_ptr<WeightManager> weight_manager,
        std::shared_ptr<WeightPlacementMap> placement_map)
        : weight_manager_(std::move(weight_manager)), placement_map_(std::move(placement_map))
    {
        if (!weight_manager_)
        {
            throw std::invalid_argument("WeightPreloader: weight_manager cannot be null");
        }
    }

    bool WeightPreloader::preloadAll(
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        // Get all cached weight names from the weight manager
        // Note: We iterate the cache, so weights must already be loaded
        auto &cache = weight_manager_->cache_;
        if (cache.empty())
        {
            LOG_WARN("[WeightPreloader] No weights loaded in cache - nothing to preload");
            return true;
        }

        std::vector<std::string> weight_names;
        weight_names.reserve(cache.size());
        for (const auto &[name, tensor] : cache)
        {
            weight_names.push_back(name);
        }

        return preload(weight_names, progress_callback, release_raw_data);
    }

    bool WeightPreloader::preload(
        const std::vector<std::string> &weight_names,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        size_t total = weight_names.size();
        size_t current = 0;
        bool all_success = true;

        LOG_DEBUG("[WeightPreloader] Preloading " << total << " weights...");

        for (const auto &name : weight_names)
        {
            current++;

            // Get the tensor from cache
            auto tensor = weight_manager_->getWeight(name);
            if (!tensor)
            {
                LOG_WARN("[WeightPreloader] Weight not found: " << name);
                continue;
            }

            // Determine target device (now returns full DeviceId with ordinal)
            DeviceId target = getTargetDevice(name);

            // Report progress
            if (progress_callback)
            {
                progress_callback(current, total, name);
            }

            // Skip non-GEMM weights (embeddings, norms, biases) for GEMM packing
            // But still upload them to GPU for kernel access!
            if (!weight_manager_->isGemmWeight(name))
            {
                if (target.is_gpu())
                {
                    // Non-GEMM weights still need to be on GPU for kernels like RMSNorm
                    // Use ensureOnDevice to upload without GEMM packing
                    if (tensor->ensureOnDevice(target))
                    {
                        LOG_TRACE("[WeightPreloader] Uploaded non-GEMM weight to " << target.to_string()
                                                                                   << ": " << name << " ["
                                                                                   << tensor->shape()[0] << "x"
                                                                                   << (tensor->shape().size() > 1 ? tensor->shape()[1] : 1) << "]");
                        num_gpu_packed_++; // Track as uploaded even though not GEMM-packed
                    }
                    else
                    {
                        LOG_WARN("[WeightPreloader] Failed to upload non-GEMM weight: " << name);
                    }
                }
                else
                {
                    LOG_TRACE("[WeightPreloader] Skipping non-GEMM weight (CPU target): " << name);
                }
                continue;
            }

            // Pack the weight
            if (!packWeight(tensor.get(), target, release_raw_data))
            {
                LOG_ERROR("[WeightPreloader] Failed to pack weight: " << name);
                all_success = false;
            }
        }

        LOG_DEBUG("[WeightPreloader] Preloading complete: "
                  << num_cpu_packed_ << " CPU, " << num_gpu_packed_ << " GPU");

        return all_success;
    }

    bool WeightPreloader::preloadWithOverrideDevice(
        const std::vector<std::string> &weight_names,
        DeviceId override_device,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        size_t total = weight_names.size();
        size_t current = 0;
        bool all_success = true;

        LOG_DEBUG("[WeightPreloader] Preloading " << total << " weights to "
                                                  << override_device.to_string() << "...");

        for (const auto &name : weight_names)
        {
            current++;

            // Get the tensor from cache
            auto tensor = weight_manager_->getWeight(name);
            if (!tensor)
            {
                LOG_WARN("[WeightPreloader] Weight not found: " << name);
                continue;
            }

            // Report progress
            if (progress_callback)
            {
                progress_callback(current, total, name);
            }

            // Pack the weight to the override device
            if (!packWeight(tensor.get(), override_device, release_raw_data))
            {
                LOG_ERROR("[WeightPreloader] Failed to pack weight: " << name);
                all_success = false;
            }
        }

        LOG_DEBUG("[WeightPreloader] Preloading complete: "
                  << num_cpu_packed_ << " CPU, " << num_gpu_packed_ << " GPU");

        return all_success;
    }

    bool WeightPreloader::preloadForDevice(
        DeviceType target_device,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        auto &cache = weight_manager_->cache_;

        std::vector<std::string> matching_names;
        for (const auto &[name, tensor] : cache)
        {
            // When no placement map exists, load ALL GEMM weights to the target device
            // This is the common case for single-device inference
            if (!placement_map_)
            {
                // Only include GEMM weights (skip embeddings, norms, etc.)
                if (weight_manager_->isGemmWeight(name))
                {
                    matching_names.push_back(name);
                }
            }
            else if (getTargetDevice(name).type == target_device)
            {
                // Filter by device type when placement map exists
                matching_names.push_back(name);
            }
        }

        // Override target device in the preload call when no placement map exists
        // This ensures weights are packed for the actual target device, not CPU
        // Convert DeviceType to DeviceId with ordinal 0 (legacy path)
        DeviceId target_id = (target_device == DeviceType::CPU) ? DeviceId::cpu() : (target_device == DeviceType::CUDA) ? DeviceId::cuda(0)
                                                                                                                        : DeviceId::rocm(0);

        bool gemm_success = true;
        if (!placement_map_ && !matching_names.empty())
        {
            LOG_DEBUG("[WeightPreloader] No placement map - preloading all "
                      << matching_names.size() << " GEMM weights to target device");
            gemm_success = preloadWithOverrideDevice(matching_names, target_id, progress_callback, release_raw_data);
        }
        else
        {
            gemm_success = preload(matching_names, progress_callback, release_raw_data);
        }

        // Also upload non-GEMM weights (norms, embeddings) to GPU
        // This eliminates lazy upload overhead during inference
        if (target_id.is_gpu())
        {
            uploadNonGemmWeights(target_id);
        }

        return gemm_success;
    }

    bool WeightPreloader::preloadForDevice(
        DeviceId target_device,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        // Use the KernelFactory's thread-local mechanism to pass the device ordinal
        // to kernel creation calls within this scope
        using namespace llaminar::v2::kernels;

        DeviceType device_type = target_device.type;

        // Helper lambda for GPU preloading - shared by ROCm and CUDA paths
        auto preload_for_gpu = [&](auto &guard) -> bool
        {
            // Now all kernel creation within this call will use the correct ordinal
            auto &cache = weight_manager_->cache_;
            std::vector<std::string> matching_names;
            for (const auto &[name, tensor] : cache)
            {
                if (!placement_map_)
                {
                    if (weight_manager_->isGemmWeight(name))
                    {
                        matching_names.push_back(name);
                    }
                }
                else if (getTargetDevice(name).type == device_type)
                {
                    // Filter by device type when placement map exists
                    matching_names.push_back(name);
                }
            }

            bool gemm_success = true;
            if (!placement_map_ && !matching_names.empty())
            {
                LOG_DEBUG("[WeightPreloader] No placement map - preloading all "
                          << matching_names.size() << " GEMM weights to device "
                          << target_device.to_string());
                gemm_success = preloadWithOverrideDevice(matching_names, target_device, progress_callback, release_raw_data);
            }
            else
            {
                gemm_success = preload(matching_names, progress_callback, release_raw_data);
            }

            // Also upload non-GEMM weights (norms, embeddings) to GPU
            uploadNonGemmWeights(target_device);

            return gemm_success;
        };

        // For ROCm devices, set the thread-local ordinal so kernels are created
        // on the correct device in multi-GPU setups
        if (target_device.is_rocm())
        {
            LOG_DEBUG("[WeightPreloader] Setting ROCm ordinal " << target_device.ordinal
                                                                << " for weight preloading");
            // Create guard - will be active for entire preload operation
            KernelFactory::ROCmOrdinalGuard guard(target_device.ordinal);
            return preload_for_gpu(guard);
        }
        // For CUDA devices, set the thread-local ordinal similarly
        else if (target_device.is_cuda())
        {
            LOG_DEBUG("[WeightPreloader] Setting CUDA ordinal " << target_device.ordinal
                                                                << " for weight preloading");
            // Create guard - will be active for entire preload operation
            KernelFactory::CUDAOrdinalGuard guard(target_device.ordinal);
            return preload_for_gpu(guard);
        }
        else
        {
            // For CPU, just delegate to DeviceType version
            return preloadForDevice(device_type, progress_callback, release_raw_data);
        }
    }

    DeviceId WeightPreloader::getTargetDevice(const std::string &weight_name) const
    {
        if (!placement_map_)
        {
            return DeviceId::cpu(); // Default to CPU
        }

        DeviceId device_id = placement_map_->getDeviceForWeight(weight_name);
        if (!device_id.is_valid())
        {
            return DeviceId::cpu();
        }

        return device_id;
    }

    bool WeightPreloader::packWeight(
        TensorBase *tensor,
        DeviceId target_device,
        bool release_raw_data)
    {
        if (!tensor)
        {
            return false;
        }

        // Use device-targeted kernel creation API
        // This ensures the kernel is created for the correct device from the start
        using namespace llaminar::v2::kernels;

        // Create kernel for target device (this also packs weights appropriately)
        // Use the DeviceId overload to preserve device ordinal for multi-GPU
        auto *kernel = KernelFactory::getOrCreateGemm(tensor, target_device);
        bool success = (kernel != nullptr);

        if (success)
        {
            if (target_device.is_cpu())
            {
                num_cpu_packed_++;

                // For CPU, we can release raw data since packed weights are in CPU cache
                if (release_raw_data)
                {
                    tensor->release_raw_data();
                    LOG_TRACE("[WeightPreloader] Released raw data for: " << tensor->shape()[0]
                                                                          << "x" << tensor->shape()[1]);
                }
            }
            else
            {
                num_gpu_packed_++;

#ifdef HAVE_ROCM
                // For ROCm, force upload to device NOW to avoid hipMalloc during inference
                if (target_device.is_rocm())
                {
                    auto *rocm_kernel = dynamic_cast<llaminar2::rocm::ROCmQuantisedGemmKernel *>(kernel);
                    if (rocm_kernel)
                    {
                        LOG_TRACE("[WeightPreloader] Calling ensureWeightsConverted for ROCm tensor="
                                  << (void *)tensor << " kernel=" << (void *)kernel
                                  << " shape=" << tensor->shape()[0] << "x" << tensor->shape()[1]);
                        rocm_kernel->ensureWeightsConverted();
                        LOG_TRACE("[WeightPreloader] Uploaded ROCm weights: "
                                  << tensor->shape()[0] << "x" << tensor->shape()[1]);
                    }
                    else
                    {
                        LOG_WARN("[WeightPreloader] Cast to ROCmQuantisedGemmKernel failed!");
                    }
                }
#endif

#ifdef HAVE_CUDA
                // For CUDA, force upload to device NOW to avoid cudaMalloc during inference
                if (target_device.is_cuda())
                {
                    auto *cuda_kernel = dynamic_cast<llaminar2::cuda::CUDAQuantisedGemmKernel *>(kernel);
                    if (cuda_kernel)
                    {
                        LOG_TRACE("[WeightPreloader] Calling ensureWeightsConverted for CUDA tensor="
                                  << (void *)tensor << " kernel=" << (void *)kernel
                                  << " shape=" << tensor->shape()[0] << "x" << tensor->shape()[1]);
                        cuda_kernel->ensureWeightsConverted();
                        LOG_TRACE("[WeightPreloader] Uploaded CUDA weights: "
                                  << tensor->shape()[0] << "x" << tensor->shape()[1]);
                    }
                    else
                    {
                        LOG_WARN("[WeightPreloader] Cast to CUDAQuantisedGemmKernel failed!");
                    }
                }
#endif

                // For GPU, do NOT release raw data!
                // The tensor coherence system (ensureOnDevice) still needs the host data
                // to upload to GPU buffers during stage execution.
                LOG_TRACE("[WeightPreloader] GPU weight packed (keeping raw data): "
                          << tensor->shape()[0] << "x" << tensor->shape()[1]);
            }
        }

        return success;
    }

    size_t WeightPreloader::uploadNonGemmWeights(DeviceId target_device)
    {
        if (!target_device.is_gpu())
        {
            LOG_DEBUG("[WeightPreloader] uploadNonGemmWeights skipped for CPU target");
            return 0;
        }

        auto &cache = weight_manager_->cache_;
        size_t uploaded_count = 0;

        LOG_DEBUG("[WeightPreloader] Uploading non-GEMM weights to " << target_device.to_string() << "...");

        for (const auto &[name, tensor] : cache)
        {
            // Only process non-GEMM weights (norms, embeddings, biases)
            if (weight_manager_->isGemmWeight(name))
            {
                continue;
            }

            // NOTE: We no longer skip embedding tensors!
            // In multi-GPU scenarios, each device needs its own copy of the embedding table.
            // The embedding kernel will use the device-specific copy.

            if (!tensor)
            {
                LOG_WARN("[WeightPreloader] Non-GEMM weight is null: " << name);
                continue;
            }

            // Get or create device-specific tensor (clones for multi-GPU)
            // This ensures each device has its own tensor copy to avoid
            // race conditions when TensorBase::ensureOnDevice() is called
            TensorBase *device_tensor = getOrCreateDeviceTensor(name, tensor.get(), target_device);
            if (!device_tensor)
            {
                LOG_WARN("[WeightPreloader] Failed to get device tensor: " << name);
                continue;
            }

            // Set debug name so transfers can be traced
            device_tensor->setDebugName(name);

            // Upload to GPU using ensureOnDevice (no GEMM packing needed)
            if (device_tensor->ensureOnDevice(target_device))
            {
                size_t rows = device_tensor->shape()[0];
                size_t cols = device_tensor->shape().size() > 1 ? device_tensor->shape()[1] : 1;
                size_t bytes = rows * cols * sizeof(float);
                LOG_TRACE("[WeightPreloader] Uploaded non-GEMM weight: " << name
                                                                         << " [" << rows << "x" << cols << "] = " << bytes << " bytes");
                uploaded_count++;
            }
            else
            {
                LOG_WARN("[WeightPreloader] Failed to upload non-GEMM weight: " << name);
            }
        }

        LOG_INFO("[WeightPreloader] Uploaded " << uploaded_count << " non-GEMM weights to "
                                               << target_device.to_string());

        return uploaded_count;
    }

    TensorBase *WeightPreloader::getOrCreateDeviceTensor(
        const std::string &original_name,
        TensorBase *original,
        DeviceId target_device)
    {
        if (!original)
        {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(per_device_mutex_);

        // Track first device - original tensors stay on this device
        if (!first_device_.has_value())
        {
            first_device_ = target_device;
            LOG_DEBUG("[WeightPreloader] First device for non-GEMM weights: " << target_device.to_string());
        }

        // If this is the first device, use original tensor directly
        if (first_device_.value() == target_device)
        {
            return original;
        }

        // For subsequent devices, check if we already have a clone
        std::string cache_key = target_device.to_string() + ":" + original_name;

        auto it = per_device_tensors_.find(cache_key);
        if (it != per_device_tensors_.end())
        {
            return it->second.get();
        }

        // Need to create a clone for this device
        //
        // DESIGN NOTE: We use a generic approach via TensorFactory::createQuantized()
        // which handles ALL tensor types (FP32, Q4_0, Q8_0, Q2_K, IQ4_NL, etc.)
        // without hardcoding special cases. This works because:
        // 1. raw_data() returns the underlying bytes for any tensor type
        // 2. size_bytes() returns the correct byte count
        // 3. createQuantized() reconstructs the tensor from (type, shape, bytes)
        //
        // For embeddings specifically, ideally we'd keep them quantized on GPU
        // and dequantize on-the-fly in the kernel (like QuantisedGemmKernel does).
        // But for now, the embedding kernels dequantize to FP32, so the GPU copy
        // is FP32 regardless. This is a future optimization opportunity.
        //
        std::unique_ptr<TensorBase> clone;

        // Generic path: copy raw bytes and recreate tensor of same type
        TensorType tensor_type = original->native_type();
        size_t byte_count = original->size_bytes();
        const void *raw_ptr = original->raw_data();

        if (!raw_ptr || byte_count == 0)
        {
            LOG_WARN("[WeightPreloader] Cannot clone tensor with no data: " << original_name);
            return original;
        }

        // Copy raw bytes
        std::vector<uint8_t> raw_copy(byte_count);
        std::memcpy(raw_copy.data(), raw_ptr, byte_count);

        // Special case for FP32 (most common for norms/biases) - uses different constructor
        if (tensor_type == TensorType::FP32)
        {
            auto fp32_clone = std::make_unique<FP32Tensor>(original->shape());
            std::memcpy(fp32_clone->mutable_data(), raw_ptr, byte_count);
            clone = std::move(fp32_clone);
        }
        else
        {
            // All quantized types: use the common (shape, raw_data) constructor pattern
            // TensorFactory::createQuantized handles all 27+ quantized tensor types
            try
            {
                // We don't have a TensorFactory instance, but we can create tensors directly
                // using the same switch logic that createQuantized uses
                clone = createQuantizedTensorFromRawData(tensor_type, original->shape(), std::move(raw_copy));
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[WeightPreloader] Failed to clone tensor type "
                         << static_cast<int>(tensor_type)
                         << " for device " << target_device.to_string() << ": " << e.what());
                return original;
            }
        }

        if (!clone)
        {
            LOG_WARN("[WeightPreloader] Failed to create clone for tensor type "
                     << static_cast<int>(tensor_type));
            return original;
        }

        clone->setDebugName(original_name + "@" + target_device.to_string());

        LOG_DEBUG("[WeightPreloader] Cloned tensor for " << target_device.to_string()
                                                         << ": " << original_name << " [" << original->shape()[0] << "x"
                                                         << (original->shape().size() > 1 ? original->shape()[1] : 1) << "]"
                                                         << " type=" << static_cast<int>(original->native_type()));

        TensorBase *result = clone.get();
        per_device_tensors_[cache_key] = std::move(clone);
        return result;
    }

    TensorBase *WeightPreloader::getDeviceTensorFor(TensorBase *original, DeviceId target_device)
    {
        if (!original)
        {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(per_device_mutex_);

        // If this is the first device, original tensor is used directly
        if (first_device_.has_value() && first_device_.value() == target_device)
        {
            return original;
        }

        // Look up by debug name - tensors set their debug name during preloading
        std::string original_name = original->debugName();
        if (original_name.empty())
        {
            // No debug name set, can't look up clone
            LOG_WARN("[WeightPreloader] getDeviceTensorFor: tensor has no debug name, using original");
            return original;
        }

        // Check if we have a clone for this device
        std::string cache_key = target_device.to_string() + ":" + original_name;
        auto it = per_device_tensors_.find(cache_key);
        if (it != per_device_tensors_.end())
        {
            LOG_TRACE("[WeightPreloader] getDeviceTensorFor: found clone for " << cache_key);
            return it->second.get();
        }

        // No clone exists - either it's the first device or tensor wasn't preloaded
        LOG_TRACE("[WeightPreloader] getDeviceTensorFor: no clone for " << cache_key << ", using original");
        return original;
    }

} // namespace llaminar2

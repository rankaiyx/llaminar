/**
 * @file WeightPreloader.cpp
 * @brief Implementation of weight pre-packing before graph execution
 */

#include "WeightPreloader.h"
#include "../kernels/KernelFactory.h"
#include "../utils/Logger.h"
#include "../backends/ComputeBackend.h"

namespace llaminar2
{

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
        auto &cache = weight_manager_->cache();
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

        LOG_INFO("[WeightPreloader] Preloading " << total << " weights...");

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

            // Determine target device
            DeviceType target = getTargetDevice(name);

            // Report progress
            if (progress_callback)
            {
                progress_callback(current, total, name);
            }

            // Skip non-GEMM weights (embeddings, norms, biases)
            // These don't need packing
            if (!weight_manager_->isGemmWeight(name))
            {
                LOG_TRACE("[WeightPreloader] Skipping non-GEMM weight: " << name);
                continue;
            }

            // Pack the weight
            if (!packWeight(tensor.get(), target, release_raw_data))
            {
                LOG_ERROR("[WeightPreloader] Failed to pack weight: " << name);
                all_success = false;
            }
        }

        LOG_INFO("[WeightPreloader] Preloading complete: "
                 << num_cpu_packed_ << " CPU, " << num_gpu_packed_ << " GPU");

        return all_success;
    }

    bool WeightPreloader::preloadForDevice(
        DeviceType target_device,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        auto &cache = weight_manager_->cache();

        std::vector<std::string> matching_names;
        for (const auto &[name, tensor] : cache)
        {
            if (getTargetDevice(name) == target_device)
            {
                matching_names.push_back(name);
            }
        }

        return preload(matching_names, progress_callback, release_raw_data);
    }

    DeviceType WeightPreloader::getTargetDevice(const std::string &weight_name) const
    {
        if (!placement_map_)
        {
            return DeviceType::CPU; // Default to CPU
        }

        int device_idx = placement_map_->getDeviceForWeight(weight_name);
        if (device_idx < 0)
        {
            return DeviceType::CPU;
        }

        // Device index 0 typically means CPU, >0 means GPU
        // Check with DeviceManager to be sure
        const auto &dm = DeviceManager::instance();
        if (static_cast<size_t>(device_idx) < dm.devices().size())
        {
            const auto &device = dm.devices()[device_idx];
            // ComputeBackendType maps to DeviceType
            if (device.type == ComputeBackendType::GPU_CUDA)
            {
                return DeviceType::CUDA;
            }
            else if (device.type == ComputeBackendType::GPU_ROCM)
            {
                return DeviceType::ROCm;
            }
        }

        return DeviceType::CPU;
    }

    bool WeightPreloader::packWeight(
        TensorBase *tensor,
        DeviceType target_device,
        bool release_raw_data)
    {
        if (!tensor)
        {
            return false;
        }

        // Use the unified packing API
        using namespace llaminar::v2::kernels;

        bool success = KernelFactory::ensurePackedWeightsInTensorCache(tensor, target_device);

        if (success)
        {
            if (target_device == DeviceType::CPU)
            {
                num_cpu_packed_++;
            }
            else
            {
                num_gpu_packed_++;
            }

            // Release raw GGUF data to save memory
            if (release_raw_data)
            {
                tensor->release_raw_data();
                LOG_TRACE("[WeightPreloader] Released raw data for: " << tensor->shape()[0]
                                                                      << "x" << tensor->shape()[1]);
            }
        }

        return success;
    }

} // namespace llaminar2

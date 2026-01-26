/**
 * @file LocalTPContext.cpp
 * @brief Implementation of LOCAL tensor parallelism context
 * @author David Sanftenberg
 * @date January 2026
 */

#include "LocalTPContext.h"
#include "backends/HostBackend.h"
#include "../tensors/TensorClasses.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

// Conditionally include GPU-specific backends
#ifdef HAVE_CUDA
#include "backends/NCCLBackend.h"
#endif

#ifdef HAVE_ROCM
#include "backends/RCCLBackend.h"
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/PCIeBARBackend.h"
#endif

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    LocalTPContext::LocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend)
        : devices_(std::move(devices))
    {
        // Validate devices
        if (devices_.empty())
        {
            throw std::invalid_argument("LocalTPContext: devices cannot be empty");
        }

        // Handle weights
        if (weights.empty())
        {
            // Equal distribution
            weights_.resize(devices_.size(), 1.0f / static_cast<float>(devices_.size()));
        }
        else if (weights.size() != devices_.size())
        {
            throw std::invalid_argument(
                "LocalTPContext: weights count (" + std::to_string(weights.size()) +
                ") must match device count (" + std::to_string(devices_.size()) + ")");
        }
        else
        {
            weights_ = normalizeWeights(weights);
        }

        // Handle backend
        if (backend == CollectiveBackendType::AUTO)
        {
            backend_ = autoDetectBackend(devices_);
        }
        else
        {
            backend_ = backend;
        }

        // Build lookup index
        buildDeviceIndex();

        LOG_DEBUG("LocalTPContext created: degree=" << degree()
                                                    << ", backend=" << collectiveBackendTypeToString(backend_));

        // Initialize backend for multi-device scenarios
        if (degree() > 1)
        {
            if (!initializeBackend())
            {
                LOG_WARN("LocalTPContext: Failed to initialize backend "
                         << collectiveBackendTypeToString(backend_)
                         << ", collectives will be no-ops");
            }
        }
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    const std::vector<GlobalDeviceAddress> &LocalTPContext::devices() const
    {
        return devices_;
    }

    const std::vector<float> &LocalTPContext::weights() const
    {
        return weights_;
    }

    CollectiveBackendType LocalTPContext::backend() const
    {
        return backend_;
    }

    int LocalTPContext::degree() const
    {
        return static_cast<int>(devices_.size());
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool LocalTPContext::allreduce(TensorBase *tensor)
    {
        if (!tensor)
        {
            LOG_ERROR("LocalTPContext::allreduce: null tensor");
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex_);

        // Single device - no-op
        if (degree() == 1)
        {
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allreduce: Backend not initialized, skipping");
            return true; // Return true to allow pipeline to continue
        }

        // For LOCAL TP, we use the Multi-GPU API if available
        // Each device has its own buffer that participates in the collective
        if (backend_impl_->isMultiGpuSingleProcess())
        {
            // Get device buffers for all devices
            auto buffers = getDeviceBuffers(tensor);
            if (buffers.size() != static_cast<size_t>(degree()))
            {
                LOG_ERROR("LocalTPContext::allreduce: Failed to get device buffers");
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            LOG_DEBUG("LocalTPContext::allreduce: Multi-GPU allreduce with "
                      << degree() << " devices, " << count << " elements");

            bool result = backend_impl_->allreduceMulti(
                buffers, count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (!result)
            {
                LOG_ERROR("LocalTPContext::allreduce: Backend allreduceMulti failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
        else
        {
            // Fallback: single-buffer API (tensor must be on local device)
            // Ensure tensor is on the local device
            DeviceId local_device = devices_[0].toLocalDeviceId();
            if (!tensor->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allreduce: Failed to ensure tensor on device");
                return false;
            }

            void *buffer = tensor->gpu_data_ptr();
            if (!buffer)
            {
                LOG_ERROR("LocalTPContext::allreduce: No device buffer available");
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            // ================================================================
            // PCIeBAR Backend: Use barrier-synchronized allreduce
            // ================================================================
            // For heterogeneous GPU setups (CUDA + ROCm), multiple threads call
            // allreduce() concurrently from different devices. We need NCCL-style
            // collective semantics where all devices must arrive at the allreduce
            // before any data transfer happens. Use barrier-synchronized path.
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            if (backend_ == CollectiveBackendType::PCIE_BAR && degree() > 1)
            {
                // Release the main mutex before entering barrier (to avoid deadlock)
                // The barrier has its own mutex for synchronization
                lock.unlock();
                return allreduceWithBarrier(tensor);
            }
#endif

            LOG_DEBUG("LocalTPContext::allreduce: Single-buffer allreduce with "
                      << count << " elements");

            bool result = backend_impl_->allreduce(
                buffer, count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (result)
            {
                tensor->mark_device_dirty();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allreduce: Backend allreduce failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
    }

    bool LocalTPContext::allreduce(const TensorBase *input, TensorBase *output)
    {
        if (!input || !output)
        {
            LOG_ERROR("LocalTPContext::allreduce: null input or output tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy
        if (degree() == 1)
        {
            // Copy input to output
            const float *src = input->data();
            float *dst = output->mutable_data();
            size_t count = std::min(input->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allreduce (out-of-place): Backend not initialized, skipping");
            // Fall back to copy
            const float *src = input->data();
            float *dst = output->mutable_data();
            size_t count = std::min(input->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // For out-of-place allreduce:
        // 1. Copy input to output
        // 2. Perform in-place allreduce on output
        LOG_DEBUG("LocalTPContext::allreduce (out-of-place): copying input to output first");

        // Copy on host first
        const float *src = input->data();
        float *dst = output->mutable_data();
        size_t count = std::min(input->numel(), output->numel());
        std::memcpy(dst, src, count * sizeof(float));

        // Now delegate to in-place allreduce (need to cast away const for the API)
        // The mutex is already held, so we call directly without re-locking
        return allreduceImpl(output);
    }

    // Private implementation that assumes lock is already held
    bool LocalTPContext::allreduceImpl(TensorBase *tensor)
    {
        if (!tensor)
        {
            return false;
        }

        if (degree() == 1)
        {
            return true;
        }

        if (!backend_initialized_ || !backend_impl_)
        {
            return true;
        }

        if (backend_impl_->isMultiGpuSingleProcess())
        {
            auto buffers = getDeviceBuffers(tensor);
            if (buffers.size() != static_cast<size_t>(degree()))
            {
                LOG_ERROR("LocalTPContext::allreduceImpl: Failed to get device buffers");
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            return backend_impl_->allreduceMulti(
                buffers, count, dtype, CollectiveOp::ALLREDUCE_SUM);
        }
        else
        {
            DeviceId local_device = devices_[0].toLocalDeviceId();
            if (!tensor->ensureOnDevice(local_device))
            {
                return false;
            }

            void *buffer = tensor->gpu_data_ptr();
            if (!buffer)
            {
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            // PCIeBAR Backend: Use registered allreduce path (same logic as main allreduce)
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            if (backend_ == CollectiveBackendType::PCIE_BAR)
            {
                if (!ensurePCIeBarBuffersRegistered(tensor))
                {
                    return false;
                }

                auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
                if (pcie_backend)
                {
                    // Copy tensor data to the registered BAR buffer on ROCm side
                    DeviceId rocm_device;
                    for (const auto &device : devices_)
                    {
                        DeviceId local_id = device.toLocalDeviceId();
                        if (local_id.is_rocm())
                        {
                            rocm_device = local_id;
                            break;
                        }
                    }

                    auto rocm_buf_opt = pcie_backend->getBuffer(pciebar_collective_id_, rocm_device);
                    if (rocm_buf_opt.has_value() && rocm_buf_opt->ptr)
                    {
                        const float *host_data = tensor->data();
                        std::memcpy(rocm_buf_opt->ptr, host_data, count * sizeof(float));
                    }
                }

                bool result = backend_impl_->allreduceRegistered(
                    pciebar_collective_id_, count, dtype, CollectiveOp::ALLREDUCE_SUM);

                if (result)
                {
                    tensor->mark_device_dirty();
                }
                return result;
            }
#endif

            bool result = backend_impl_->allreduce(
                buffer, count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (result)
            {
                tensor->mark_device_dirty();
            }
            return result;
        }
    }

    // =========================================================================
    // PCIeBAR Barrier-Synchronized Allreduce
    // =========================================================================

    bool LocalTPContext::allreduceWithBarrier(TensorBase *tensor)
    {
        const int num_participants = degree();

        std::unique_lock<std::mutex> lock(barrier_mutex_);

        // Capture current generation to detect spurious wakeups
        uint64_t my_generation = barrier_generation_.load();

        // Increment arrival count
        int arrival_order = barrier_count_.fetch_add(1);

        if (arrival_order == 0)
        {
            // First arrival: store tensor reference for the executor
            barrier_tensor_ = tensor;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrier: First arrival (device thread), "
                      << "waiting for " << (num_participants - 1) << " more devices");
        }
        else
        {
            LOG_DEBUG("LocalTPContext::allreduceWithBarrier: Device arrival #" << (arrival_order + 1)
                                                                               << " of " << num_participants);
        }

        if (arrival_order + 1 < num_participants)
        {
            // Not the last arrival: wait for completion
            barrier_cv_.wait(lock, [this, my_generation]()
                             { return barrier_generation_.load() > my_generation; });

            // Woke up - get the shared result
            bool result = barrier_result_;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrier: Waiter released with result=" << result);
            return result;
        }

        // =====================================================================
        // LAST ARRIVAL: Execute the actual allreduce
        // =====================================================================
        // All other threads are waiting, so we have exclusive access to barrier_tensor_

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: All " << num_participants
                                                               << " devices arrived, executing PCIeBAR allreduce");

        // The actual PCIeBAR transfer
        bool success = executePCIeBarAllreduce(barrier_tensor_);

        // Store result and signal completion
        barrier_result_ = success;
        barrier_tensor_ = nullptr;
        barrier_count_.store(0);
        barrier_generation_.fetch_add(1);

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: PCIeBAR allreduce completed with result="
                  << success << ", releasing waiters (generation=" << barrier_generation_.load() << ")");

        lock.unlock();
        barrier_cv_.notify_all();

        return success;
    }

    bool LocalTPContext::executePCIeBarAllreduce(TensorBase *tensor)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!tensor)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: null tensor");
            return false;
        }

        // Lock the main mutex for backend operations
        std::lock_guard<std::mutex> lock(mutex_);

        // Ensure buffers are allocated in BAR region and registered
        if (!ensurePCIeBarBuffersRegistered(tensor))
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to register PCIeBAR buffers");
            return false;
        }

        size_t count = tensor->numel();
        CollectiveDataType dtype = tensorDTypeToCollective(tensor);

        LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Using PCIeBAR registered allreduce for "
                  << count << " elements (collective_id=" << pciebar_collective_id_ << ")");

        // Copy tensor data to the registered BAR buffer on ROCm side
        auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
        if (pcie_backend)
        {
            // Get the registered ROCm buffer to copy data into it
            DeviceId rocm_device;
            for (const auto &device : devices_)
            {
                DeviceId local_id = device.toLocalDeviceId();
                if (local_id.is_rocm())
                {
                    rocm_device = local_id;
                    break;
                }
            }

            auto rocm_buf_opt = pcie_backend->getBuffer(pciebar_collective_id_, rocm_device);
            if (rocm_buf_opt.has_value() && rocm_buf_opt->ptr)
            {
                // Copy tensor data (on host) to the BAR-mapped ROCm buffer
                // This makes the data available for the PCIeBAR transfer
                const float *host_data = tensor->data();
                size_t bytes = count * sizeof(float);
                std::memcpy(rocm_buf_opt->ptr, host_data, bytes);

                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Copied " << bytes
                                                                             << " bytes to BAR buffer at offset " << rocm_buf_opt->bar_offset);
            }
        }

        bool result = backend_impl_->allreduceRegistered(
            pciebar_collective_id_, count, dtype, CollectiveOp::ALLREDUCE_SUM);

        if (result)
        {
            // After allreduce, the result is in both CUDA and ROCm buffers.
            // Copy from CUDA buffer back to tensor to update host state
            tensor->mark_device_dirty();
        }
        else
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: PCIeBAR allreduceRegistered failed: "
                      << backend_impl_->lastError());
        }

        return result;
#else
        // Shouldn't be called without CUDA+ROCm
        LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: PCIeBAR requires both CUDA and ROCm");
        return false;
#endif
    }

    bool LocalTPContext::allgather(const TensorBase *local_shard, TensorBase *global_tensor)
    {
        if (!local_shard || !global_tensor)
        {
            LOG_ERROR("LocalTPContext::allgather: null tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy
        if (degree() == 1)
        {
            const float *src = local_shard->data();
            float *dst = global_tensor->mutable_data();
            size_t count = std::min(local_shard->numel(), global_tensor->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allgather: Backend not initialized, skipping");
            // Fall back to copy of local shard
            const float *src = local_shard->data();
            float *dst = global_tensor->mutable_data();
            size_t count = std::min(local_shard->numel(), global_tensor->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // For LOCAL TP with Multi-GPU:
        // Each device sends its shard, receives all shards concatenated
        if (backend_impl_->isMultiGpuSingleProcess())
        {
            // For multi-GPU allgather, we need send buffers (one per device)
            // and recv buffers (one per device, each gets the full gathered result)

            // Note: In LOCAL TP, the local_shard and global_tensor parameters
            // represent the buffers for the "local" device. For true multi-GPU,
            // we would need separate buffers per device. For now, we use the
            // single-buffer API.
            DeviceId local_device = devices_[0].toLocalDeviceId();

            // Ensure local shard is on device (const-cast needed for ensureOnDevice)
            auto *mutable_shard = const_cast<TensorBase *>(local_shard);
            if (!mutable_shard->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to ensure local_shard on device");
                return false;
            }

            // Ensure global tensor is allocated on device
            if (!global_tensor->allocateOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to allocate global_tensor on device");
                return false;
            }

            const void *send_buf = mutable_shard->gpu_data_ptr();
            void *recv_buf = global_tensor->gpu_data_ptr();

            if (!send_buf || !recv_buf)
            {
                LOG_ERROR("LocalTPContext::allgather: No device buffers available");
                return false;
            }

            size_t send_count = local_shard->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(local_shard);

            LOG_DEBUG("LocalTPContext::allgather: allgather with "
                      << degree() << " devices, " << send_count << " elements per device");

            bool result = backend_impl_->allgather(
                send_buf, recv_buf, send_count, dtype);

            if (result)
            {
                global_tensor->mark_device_dirty();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allgather: Backend allgather failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
        else
        {
            // Fallback to single-buffer allgather
            DeviceId local_device = devices_[0].toLocalDeviceId();

            auto *mutable_shard = const_cast<TensorBase *>(local_shard);
            if (!mutable_shard->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to ensure local_shard on device");
                return false;
            }

            if (!global_tensor->allocateOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to allocate global_tensor on device");
                return false;
            }

            const void *send_buf = mutable_shard->gpu_data_ptr();
            void *recv_buf = global_tensor->gpu_data_ptr();

            if (!send_buf || !recv_buf)
            {
                LOG_ERROR("LocalTPContext::allgather: No device buffers available");
                return false;
            }

            size_t send_count = local_shard->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(local_shard);

            bool result = backend_impl_->allgather(
                send_buf, recv_buf, send_count, dtype);

            if (result)
            {
                global_tensor->mark_device_dirty();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allgather: Backend allgather failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
    }

    bool LocalTPContext::gatherFromDevices(
        const std::vector<const TensorBase *> &shards,
        TensorBase *output)
    {
        if (shards.empty() || !output)
        {
            LOG_ERROR("LocalTPContext::gatherFromDevices: empty shards or null output");
            return false;
        }

        // Validate shard count matches device count
        if (static_cast<int>(shards.size()) != degree())
        {
            LOG_ERROR("LocalTPContext::gatherFromDevices: shard count (" << shards.size()
                                                                         << ") doesn't match device count (" << degree() << ")");
            return false;
        }

        // Verify all shards are non-null
        for (size_t i = 0; i < shards.size(); ++i)
        {
            if (!shards[i])
            {
                LOG_ERROR("LocalTPContext::gatherFromDevices: shard[" << i << "] is null");
                return false;
            }
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy the shard to output
        if (degree() == 1)
        {
            const float *src = shards[0]->data();
            float *dst = output->mutable_data();
            size_t count = std::min(shards[0]->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Multi-device: concatenate all shards into output
        // For now, use CPU-side gather (works with any backend)
        // TODO: For GPU backends with allgatherMulti support, use device-side gather

        float *dst = output->mutable_data();
        size_t offset = 0;
        size_t output_capacity = output->numel();

        for (size_t i = 0; i < shards.size(); ++i)
        {
            const float *src = shards[i]->data();
            size_t shard_size = shards[i]->numel();

            // Check bounds
            if (offset + shard_size > output_capacity)
            {
                LOG_ERROR("LocalTPContext::gatherFromDevices: output buffer too small. "
                          << "Need " << (offset + shard_size) << ", have " << output_capacity);
                return false;
            }

            std::memcpy(dst + offset, src, shard_size * sizeof(float));
            offset += shard_size;
        }

        LOG_DEBUG("LocalTPContext::gatherFromDevices: gathered " << shards.size()
                                                                 << " shards, total " << offset << " elements");

        return true;
    }

    bool LocalTPContext::reduceScatter(const TensorBase *input, TensorBase *output_shard)
    {
        if (!input || !output_shard)
        {
            LOG_ERROR("LocalTPContext::reduceScatter: null tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy the appropriate shard
        if (degree() == 1)
        {
            const float *src = input->data();
            float *dst = output_shard->mutable_data();
            size_t count = std::min(input->numel(), output_shard->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::reduceScatter: Backend not initialized, skipping");
            // Fall back to copy of first shard
            const float *src = input->data();
            float *dst = output_shard->mutable_data();
            size_t count = output_shard->numel();
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // ReduceScatter: reduce across devices, each device gets a slice
        DeviceId local_device = devices_[0].toLocalDeviceId();

        // Ensure input is on device
        auto *mutable_input = const_cast<TensorBase *>(input);
        if (!mutable_input->ensureOnDevice(local_device))
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Failed to ensure input on device");
            return false;
        }

        // Ensure output is allocated on device
        if (!output_shard->allocateOnDevice(local_device))
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Failed to allocate output_shard on device");
            return false;
        }

        const void *send_buf = mutable_input->gpu_data_ptr();
        void *recv_buf = output_shard->gpu_data_ptr();

        if (!send_buf || !recv_buf)
        {
            LOG_ERROR("LocalTPContext::reduceScatter: No device buffers available");
            return false;
        }

        size_t recv_count = output_shard->numel();
        CollectiveDataType dtype = tensorDTypeToCollective(input);

        LOG_DEBUG("LocalTPContext::reduceScatter: reduceScatter with "
                  << degree() << " devices, " << recv_count << " elements per device");

        bool result = backend_impl_->reduceScatter(
            send_buf, recv_buf, recv_count, dtype, CollectiveOp::ALLREDUCE_SUM);

        if (result)
        {
            output_shard->mark_device_dirty();
        }
        else
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Backend reduceScatter failed: "
                      << backend_impl_->lastError());
        }
        return result;
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    void LocalTPContext::synchronize()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - no-op
        if (degree() == 1)
        {
            return;
        }

        // Synchronize via backend
        if (backend_initialized_ && backend_impl_)
        {
            LOG_DEBUG("LocalTPContext::synchronize: Synchronizing backend "
                      << collectiveBackendTypeToString(backend_));
            if (!backend_impl_->synchronize())
            {
                LOG_WARN("LocalTPContext::synchronize: Backend synchronize failed: "
                         << backend_impl_->lastError());
            }
        }
    }

    // =========================================================================
    // Device Management
    // =========================================================================

    int LocalTPContext::indexForDevice(const GlobalDeviceAddress &device) const
    {
        auto it = device_to_index_.find(device);
        if (it != device_to_index_.end())
        {
            return it->second;
        }
        return -1;
    }

    const GlobalDeviceAddress &LocalTPContext::deviceAt(int index) const
    {
        if (index < 0 || index >= static_cast<int>(devices_.size()))
        {
            throw std::out_of_range(
                "LocalTPContext::deviceAt: index " + std::to_string(index) +
                " out of range [0, " + std::to_string(devices_.size()) + ")");
        }
        return devices_[index];
    }

    float LocalTPContext::weightForDevice(const GlobalDeviceAddress &device) const
    {
        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::weightForDevice: device not found");
            return 0.0f;
        }
        return weights_[idx];
    }

    // =========================================================================
    // Weight Sharding Utilities
    // =========================================================================

    int LocalTPContext::headsForDevice(const GlobalDeviceAddress &device, int total_heads) const
    {
        if (total_heads <= 0)
        {
            return 0;
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::headsForDevice: device not found");
            return 0;
        }

        // Use cumulative counts to ensure exact distribution
        auto cumulative = computeCumulativeCounts(total_heads, weights_);
        return cumulative[idx + 1] - cumulative[idx];
    }

    std::pair<int, int> LocalTPContext::rowRangeForDevice(
        const GlobalDeviceAddress &device, int total_rows) const
    {
        if (total_rows <= 0)
        {
            return {0, 0};
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::rowRangeForDevice: device not found");
            return {0, 0};
        }

        auto cumulative = computeCumulativeCounts(total_rows, weights_);
        return {cumulative[idx], cumulative[idx + 1]};
    }

    std::pair<int, int> LocalTPContext::colRangeForDevice(
        const GlobalDeviceAddress &device, int total_cols) const
    {
        if (total_cols <= 0)
        {
            return {0, 0};
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::colRangeForDevice: device not found");
            return {0, 0};
        }

        auto cumulative = computeCumulativeCounts(total_cols, weights_);
        return {cumulative[idx], cumulative[idx + 1]};
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    std::vector<float> LocalTPContext::normalizeWeights(const std::vector<float> &weights)
    {
        if (weights.empty())
        {
            return {};
        }

        // Check for non-positive weights
        for (float w : weights)
        {
            if (w <= 0.0f)
            {
                throw std::invalid_argument("LocalTPContext: weights must be positive");
            }
        }

        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= 0.0f)
        {
            throw std::invalid_argument("LocalTPContext: weight sum must be positive");
        }

        std::vector<float> normalized(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            normalized[i] = weights[i] / sum;
        }

        return normalized;
    }

    std::vector<int> LocalTPContext::computeCumulativeCounts(
        int total, const std::vector<float> &norm_weights)
    {
        std::vector<int> cumulative(norm_weights.size() + 1);
        cumulative[0] = 0;

        // Distribute proportionally with rounding to ensure exact total
        int remaining = total;
        float remaining_weight = 1.0f;

        for (size_t i = 0; i < norm_weights.size(); ++i)
        {
            if (i == norm_weights.size() - 1)
            {
                // Last device gets the remainder to ensure exact total
                cumulative[i + 1] = total;
            }
            else
            {
                // Proportional distribution with proper rounding
                float proportion = norm_weights[i] / remaining_weight;
                int count = static_cast<int>(std::round(proportion * remaining));

                // Ensure at least 1 if there's work remaining
                if (count == 0 && remaining > 0)
                {
                    count = 1;
                }
                // Don't exceed remaining
                count = std::min(count, remaining);

                cumulative[i + 1] = cumulative[i] + count;
                remaining -= count;
                remaining_weight -= norm_weights[i];
            }
        }

        return cumulative;
    }

    CollectiveBackendType LocalTPContext::autoDetectBackend(
        const std::vector<GlobalDeviceAddress> &devices)
    {
        bool has_cuda = false;
        bool has_rocm = false;
        bool has_cpu = false;

        for (const auto &dev : devices)
        {
            if (dev.isCUDA())
                has_cuda = true;
            else if (dev.isROCm())
                has_rocm = true;
            else if (dev.isCPU())
                has_cpu = true;
        }

        // If CPU is involved, use host-staged backend
        if (has_cpu)
        {
            return CollectiveBackendType::HOST;
        }

        // Mixed GPU types - use PCIe BAR
        if (has_cuda && has_rocm)
        {
            return CollectiveBackendType::PCIE_BAR;
        }

        // All CUDA - use NCCL
        if (has_cuda && !has_rocm)
        {
            return CollectiveBackendType::NCCL;
        }

        // All ROCm - use RCCL
        if (has_rocm && !has_cuda)
        {
            return CollectiveBackendType::RCCL;
        }

        // Default to HOST if nothing detected (shouldn't happen)
        return CollectiveBackendType::HOST;
    }

    void LocalTPContext::buildDeviceIndex()
    {
        device_to_index_.clear();
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            device_to_index_[devices_[i]] = static_cast<int>(i);
        }
    }

    // =========================================================================
    // Backend Initialization and Helper Methods
    // =========================================================================

    bool LocalTPContext::initializeBackend()
    {
        // Build device group from devices_
        DeviceGroupBuilder builder;
        builder.setName("LocalTP_" + std::to_string(degree()) + "_devices");
        builder.setScope(CollectiveScope::LOCAL);
        builder.setLocalRank(0); // In LOCAL TP, we manage all devices from rank 0

        for (const auto &device : devices_)
        {
            builder.addDevice(device.toLocalDeviceId());
        }

        device_group_ = builder.build();

        // Create appropriate backend based on type
        switch (backend_)
        {
        case CollectiveBackendType::NCCL:
#ifdef HAVE_CUDA
            LOG_DEBUG("LocalTPContext: Creating NCCL backend");
            backend_impl_ = std::make_unique<NCCLBackend>();
#else
            LOG_WARN("LocalTPContext: NCCL requested but CUDA not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::RCCL:
#ifdef HAVE_ROCM
            LOG_DEBUG("LocalTPContext: Creating RCCL backend");
            backend_impl_ = std::make_unique<RCCLBackend>();
#else
            LOG_WARN("LocalTPContext: RCCL requested but ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            LOG_DEBUG("LocalTPContext: Creating PCIe BAR backend");
            backend_impl_ = std::make_unique<PCIeBARBackend>();
#else
            LOG_WARN("LocalTPContext: PCIE_BAR requested but both CUDA and ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::HOST:
        case CollectiveBackendType::AUTO:
        default:
            LOG_DEBUG("LocalTPContext: Creating HOST backend");
            backend_impl_ = std::make_unique<HostBackend>();
            break;
        }

        // Check if backend is available
        if (!backend_impl_->isAvailable())
        {
            LOG_WARN("LocalTPContext: Backend " << collectiveBackendTypeToString(backend_)
                                                << " not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
        }

        // Initialize the backend with device group
        if (!backend_impl_->initialize(device_group_))
        {
            LOG_ERROR("LocalTPContext: Failed to initialize backend: "
                      << backend_impl_->lastError());
            backend_impl_.reset();
            backend_initialized_ = false;
            return false;
        }

        backend_initialized_ = true;
        LOG_INFO("LocalTPContext: Backend " << backend_impl_->name()
                                            << " initialized for " << degree() << " devices");
        return true;
    }

    bool LocalTPContext::ensurePCIeBarBuffersRegistered(TensorBase *tensor)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!tensor)
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: null tensor");
            return false;
        }

        // Only needed for PCIeBAR backend
        if (backend_ != CollectiveBackendType::PCIE_BAR)
        {
            return true;
        }

        auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
        if (!pcie_backend)
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Backend is not PCIeBARBackend");
            return false;
        }

        size_t buffer_bytes = tensor->numel() * sizeof(float);

        // If buffers are already registered for this size, we're done
        if (pciebar_buffers_registered_ && buffer_bytes <= pciebar_buffer_size_)
        {
            LOG_DEBUG("ensurePCIeBarBuffersRegistered: Reusing registered buffers for collective "
                      << pciebar_collective_id_);
            return true;
        }

        // If size changed, we need to re-register (unregister old first)
        if (pciebar_buffers_registered_ && buffer_bytes > pciebar_buffer_size_)
        {
            LOG_WARN("ensurePCIeBarBuffersRegistered: Buffer size increased from "
                     << pciebar_buffer_size_ << " to " << buffer_bytes
                     << ", need to re-register. This may leak BAR memory.");
            // Note: We can't actually reclaim BAR allocations (bump allocator),
            // but we can unregister the old collective_id
            for (const auto &device : devices_)
            {
                pcie_backend->unregisterBuffer(pciebar_collective_id_, device.toLocalDeviceId());
            }
            pciebar_buffers_registered_ = false;
        }

        // Generate a unique collective_id for this LocalTPContext
        pciebar_collective_id_ = "LocalTP_allreduce_" + std::to_string(reinterpret_cast<uintptr_t>(this));
        pciebar_buffer_size_ = buffer_bytes;

        LOG_DEBUG("ensurePCIeBarBuffersRegistered: Registering buffers for " << pciebar_collective_id_
                                                                             << " (size=" << buffer_bytes << " bytes)");

        // For 2-device LOCAL TP (CUDA + ROCm):
        // 1. CUDA buffer: use tensor's existing GPU buffer (allocated by standard path)
        // 2. ROCm buffer: must be allocated in BAR region to get correct offset

        DeviceId cuda_device;
        DeviceId rocm_device;
        void *cuda_ptr = nullptr;

        // Identify CUDA and ROCm devices
        for (const auto &device : devices_)
        {
            DeviceId local_id = device.toLocalDeviceId();
            if (local_id.is_cuda())
            {
                cuda_device = local_id;
                // Get CUDA buffer from tensor
                if (!tensor->ensureOnDevice(cuda_device))
                {
                    LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to ensure tensor on CUDA device");
                    return false;
                }
                cuda_ptr = tensor->gpu_data_ptr();
                if (!cuda_ptr)
                {
                    LOG_ERROR("ensurePCIeBarBuffersRegistered: No CUDA buffer available");
                    return false;
                }
            }
            else if (local_id.is_rocm())
            {
                rocm_device = local_id;
            }
        }

        if (!cuda_device.is_valid() || !rocm_device.is_valid())
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Need exactly one CUDA and one ROCm device");
            return false;
        }

        // Allocate ROCm buffer in BAR region
        auto bar_alloc = pcie_backend->allocateInBarRegion(buffer_bytes);
        if (!bar_alloc.has_value())
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to allocate " << buffer_bytes
                                                                            << " bytes in BAR region");
            return false;
        }

        void *rocm_bar_ptr = bar_alloc->first;
        size_t rocm_bar_offset = bar_alloc->second;

        LOG_DEBUG("ensurePCIeBarBuffersRegistered: Allocated BAR buffer at offset " << rocm_bar_offset);

        // Register CUDA buffer
        if (!pcie_backend->registerBuffer(pciebar_collective_id_, cuda_device, cuda_ptr, buffer_bytes))
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to register CUDA buffer");
            return false;
        }

        // Register ROCm buffer (with BAR offset)
        if (!pcie_backend->registerBuffer(pciebar_collective_id_, rocm_device, rocm_bar_ptr, buffer_bytes))
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to register ROCm buffer");
            pcie_backend->unregisterBuffer(pciebar_collective_id_, cuda_device);
            return false;
        }

        pciebar_buffers_registered_ = true;
        LOG_INFO("ensurePCIeBarBuffersRegistered: Successfully registered buffers for PCIeBAR allreduce"
                 << " (collective_id=" << pciebar_collective_id_
                 << ", size=" << buffer_bytes << " bytes"
                 << ", BAR offset=" << rocm_bar_offset << ")");

        return true;
#else
        (void)tensor;
        return true; // No-op when PCIeBAR not available
#endif
    }

    std::vector<void *> LocalTPContext::getDeviceBuffers(TensorBase *tensor)
    {
        std::vector<void *> buffers;
        buffers.reserve(devices_.size());

        // For LOCAL TP, we need to get the GPU buffer for each device
        // Current implementation assumes tensor has data on all devices
        // or we need to replicate it

        for (size_t i = 0; i < devices_.size(); ++i)
        {
            DeviceId device_id = devices_[i].toLocalDeviceId();

            // Ensure tensor data is on this device
            if (!tensor->ensureOnDevice(device_id))
            {
                LOG_ERROR("LocalTPContext::getDeviceBuffers: Failed to ensure tensor on device "
                          << device_id.toString());
                return {}; // Return empty vector to indicate failure
            }

            void *ptr = tensor->gpu_data_ptr();
            if (!ptr)
            {
                LOG_ERROR("LocalTPContext::getDeviceBuffers: No GPU buffer for device "
                          << device_id.toString());
                return {};
            }

            buffers.push_back(ptr);
        }

        return buffers;
    }

    CollectiveDataType LocalTPContext::tensorDTypeToCollective(const TensorBase *tensor) const
    {
        if (!tensor)
        {
            return CollectiveDataType::FLOAT32;
        }

        // Get tensor type and map to CollectiveDataType
        // Most common case is FP32 for activation tensors
        TensorType tt = tensor->native_type();
        switch (tt)
        {
        case TensorType::FP32:
            return CollectiveDataType::FLOAT32;
        case TensorType::FP16:
            return CollectiveDataType::FLOAT16;
        case TensorType::BF16:
            return CollectiveDataType::BFLOAT16;
        case TensorType::INT32:
            return CollectiveDataType::INT32;
        case TensorType::INT8:
        case TensorType::Q8_0:
        case TensorType::Q8_1:
            return CollectiveDataType::INT8;
        default:
            // For quantized types that don't have a direct mapping, use FLOAT32
            // (collectives are typically on dequantized activations)
            return CollectiveDataType::FLOAT32;
        }
    }

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<ILocalTPContext> createLocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend)
    {
        return std::make_unique<LocalTPContext>(
            std::move(devices),
            std::move(weights),
            backend);
    }

} // namespace llaminar2

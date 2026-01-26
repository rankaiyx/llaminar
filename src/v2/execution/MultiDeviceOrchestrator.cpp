/**
 * @file MultiDeviceOrchestrator.cpp
 * @brief Multi-device orchestrator implementation for LOCAL tensor parallelism
 * @author David Sanftenberg
 * @date January 2026
 *
 * Implements coordination of multiple DeviceGraphOrchestrator instances for LOCAL
 * tensor parallelism across multiple devices within a single MPI rank.
 *
 * Key features:
 * - Parallel forward pass execution across devices via std::async
 * - AllGather for combining partial logits from column-parallel LM head
 * - Unified snapshot/profiling API across all device runners
 */

#include "MultiDeviceOrchestrator.h"
#include "DeviceGraphOrchestrator.h"
#include "InferenceRunnerFactory.h"
#include "../collective/ILocalTPContext.h"
#include "../config/TensorParallelConfig.h"
#include "../interfaces/IModelContext.h"
#include "../tensors/TensorClasses.h"
#include "../tensors/TensorFactory.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <future>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Config Implementation
    // =========================================================================

    bool MultiDeviceOrchestrator::Config::validate() const
    {
        // Must have at least one device
        if (devices.empty())
        {
            LOG_ERROR("MultiDeviceOrchestrator::Config: No devices specified");
            return false;
        }

        // If weights are provided, must match device count
        if (!weights.empty() && weights.size() != devices.size())
        {
            LOG_ERROR("MultiDeviceOrchestrator::Config: Weights count (" << weights.size()
                                                                         << ") doesn't match device count (" << devices.size() << ")");
            return false;
        }

        // If weights are provided, must sum to approximately 1.0
        if (!weights.empty())
        {
            float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.01f)
            {
                LOG_ERROR("MultiDeviceOrchestrator::Config: Weights sum to " << sum << ", expected 1.0");
                return false;
            }
        }

        return true;
    }

    std::vector<float> MultiDeviceOrchestrator::Config::getNormalizedWeights() const
    {
        if (weights.empty() || weights.size() != devices.size())
        {
            // Equal distribution
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        // Normalize to ensure sum is exactly 1.0
        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= 0.0f)
        {
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        std::vector<float> normalized(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            normalized[i] = weights[i] / sum;
        }
        return normalized;
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    std::unique_ptr<MultiDeviceOrchestrator> MultiDeviceOrchestrator::createForTest(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
    {
        // Use the private constructor
        return std::unique_ptr<MultiDeviceOrchestrator>(
            new MultiDeviceOrchestrator(
                std::move(model_ctx),
                std::move(device_runners),
                std::move(tp_ctx),
                config));
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    MultiDeviceOrchestrator::MultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)), config_(config)
    {
        if (!config_.validate())
        {
            throw std::invalid_argument("Invalid MultiDeviceOrchestrator configuration");
        }

        LOG_INFO("MultiDeviceOrchestrator: Creating with " << config_.devices.size()
                                                           << " devices, backend=" << static_cast<int>(config_.backend));

        // Create LOCAL TP context from config
        tp_ctx_ = createLocalTPContext(
            config_.devices,
            config_.getNormalizedWeights(),
            config_.backend);

        if (!tp_ctx_)
        {
            throw std::runtime_error("Failed to create LOCAL TP context");
        }

        // Validate TP degree
        if (tp_ctx_->degree() < 2)
        {
            LOG_WARN("MultiDeviceOrchestrator: TP degree is " << tp_ctx_->degree()
                                                              << ", multi-device orchestration may not be beneficial");
        }

        // Initialize device runners
        initializeDeviceRunners();

        LOG_INFO("MultiDeviceOrchestrator: Initialized with " << device_runners_.size() << " device runners");
    }

    MultiDeviceOrchestrator::MultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)), tp_ctx_(std::move(tp_ctx)), config_(config)
    {
        if (!tp_ctx_)
        {
            throw std::invalid_argument("tp_ctx cannot be null");
        }

        // Validate TP degree
        if (tp_ctx_->degree() < 2)
        {
            LOG_WARN("MultiDeviceOrchestrator: TP degree is " << tp_ctx_->degree()
                                                              << ", multi-device orchestration may not be beneficial");
        }

        LOG_INFO("MultiDeviceOrchestrator: Creating with pre-existing TP context, "
                 << tp_ctx_->degree() << " devices");

        // Initialize device runners
        initializeDeviceRunners();

        LOG_INFO("MultiDeviceOrchestrator: Initialized with " << device_runners_.size() << " device runners");
    }

    // Private constructor for createForTest
    MultiDeviceOrchestrator::MultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)),
          tp_ctx_(std::move(tp_ctx)),
          device_runners_(std::move(device_runners)),
          config_(config)
    {
        LOG_DEBUG("MultiDeviceOrchestrator: Created via createForTest with "
                  << device_runners_.size() << " injected device runners");
    }

    MultiDeviceOrchestrator::~MultiDeviceOrchestrator() = default;

    // Move operations
    MultiDeviceOrchestrator::MultiDeviceOrchestrator(MultiDeviceOrchestrator &&) noexcept = default;
    MultiDeviceOrchestrator &MultiDeviceOrchestrator::operator=(MultiDeviceOrchestrator &&) noexcept = default;

    // =========================================================================
    // Private Methods
    // =========================================================================

    void MultiDeviceOrchestrator::initializeDeviceRunners()
    {
        if (!tp_ctx_)
        {
            throw std::runtime_error("Cannot initialize device runners: tp_ctx_ is null");
        }

        const auto &devices = tp_ctx_->devices();
        device_runners_.reserve(devices.size());

        LOG_DEBUG("MultiDeviceOrchestrator: Initializing " << devices.size() << " device runners");

        // =====================================================================
        // BUILD TENSORPARALLELCONFIG FOR LOCAL TP WEIGHT SHARDING
        // =====================================================================
        // This enables WeightManager to slice weights by DeviceId instead of
        // falling back to REPLICATED mode for world_size==1.
        // =====================================================================
        {
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr && tp_ctx_)
            {
                // Get model dimensions
                int n_heads = model_ctx_->headCount();
                int n_kv_heads = model_ctx_->headCountKV();
                int d_ff = model_ctx_->feedForwardLength();
                int vocab_size = model_ctx_->vocabSize();

                // Estimate d_ff if not available (common SwiGLU ratio)
                if (d_ff <= 0)
                {
                    d_ff = model_ctx_->embeddingLength() * 4;
                    LOG_WARN("MultiDeviceOrchestrator: feedForwardLength() unavailable, using estimate: " << d_ff);
                }

                auto tp_config = std::make_shared<TensorParallelConfig>(
                    TensorParallelConfig::fromLocalTPContext(
                        *tp_ctx_, n_heads, n_kv_heads, d_ff, vocab_size));

                weight_mgr->setTensorParallelConfig(tp_config);

                LOG_INFO("MultiDeviceOrchestrator: Set TensorParallelConfig for LOCAL TP ("
                         << tp_ctx_->degree() << " devices, "
                         << "heads=" << n_heads << ", kv_heads=" << n_kv_heads
                         << ", d_ff=" << d_ff << ", vocab=" << vocab_size << ")");
            }
        }

        // =====================================================================
        // PRE-LOAD WEIGHTS FOR ALL DEVICES
        // =====================================================================
        // This is critical for multi-device operation:
        // - Creates device-specific clones of shared tensors (embedding, norms)
        // - Uploads each clone to its target device BEFORE parallel execution
        // - Avoids race condition where multiple devices try to upload same tensor
        //
        // The WeightManager now handles all device-aware weight management centrally.
        // =====================================================================
        {
            // Collect device IDs for preloading
            std::vector<DeviceId> device_ids;
            device_ids.reserve(devices.size());
            for (const auto &device_addr : devices)
            {
                device_ids.push_back(device_addr.toLocalDeviceId());
            }

            // Pre-load all weights for all devices
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr)
            {
                LOG_INFO("MultiDeviceOrchestrator: Pre-loading weights for " << device_ids.size() << " devices");
                if (!weight_mgr->preloadForDevices(device_ids))
                {
                    LOG_WARN("MultiDeviceOrchestrator: Weight preloading failed, may encounter race conditions");
                }
            }
        }

        for (int device_idx = 0; device_idx < static_cast<int>(devices.size()); ++device_idx)
        {
            const auto &device_addr = devices[device_idx];

            // Convert GlobalDeviceAddress to DeviceId
            DeviceId device_id = device_addr.toLocalDeviceId();

            LOG_DEBUG("MultiDeviceOrchestrator: Creating runner for device " << device_idx
                                                                             << " (" << device_id.toString() << ")");

            // Build InferenceRunnerConfig for LOCAL TP
            InferenceRunnerConfig runner_config;
            runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
            runner_config.batch_size = config_.batch_size;
            runner_config.activation_precision = config_.activation_precision;
            runner_config.kv_cache_scale = config_.kv_cache_scale;
            runner_config.use_mapped_memory = config_.use_mapped_memory;

            // Set LOCAL TP parameters
            runner_config.local_tp_ctx = tp_ctx_.get();
            runner_config.local_tp_device_index = device_idx;

            // Create the inference runner
            auto runner = createTestableInferenceRunner(model_ctx_, device_id, runner_config);

            if (!runner)
            {
                throw std::runtime_error("Failed to create inference runner for device " +
                                         std::to_string(device_idx));
            }

            // Cast to DeviceGraphOrchestrator
            auto *device_orchestrator = dynamic_cast<DeviceGraphOrchestrator *>(runner.get());
            if (!device_orchestrator)
            {
                throw std::runtime_error("Inference runner is not a DeviceGraphOrchestrator for device " +
                                         std::to_string(device_idx));
            }

            // Transfer ownership
            runner.release();
            device_runners_.push_back(std::unique_ptr<DeviceGraphOrchestrator>(device_orchestrator));

            LOG_DEBUG("MultiDeviceOrchestrator: Successfully created runner for device " << device_idx);
        }

        // Allocate combined logits buffer if we have vocab size info
        // For column-parallel LM head, this needs to be [batch_size * max_seq_len, vocab_size]
        // to hold the gathered logits from all devices
        if (model_ctx_ && device_runners_.size() > 0)
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                // Calculate max tokens = batch_size * max_seq_len
                size_t max_tokens = static_cast<size_t>(config_.batch_size) *
                                    static_cast<size_t>(config_.max_seq_len);
                combined_logits_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{max_tokens, static_cast<size_t>(vocab)});
                LOG_DEBUG("MultiDeviceOrchestrator: Allocated combined logits buffer ["
                          << max_tokens << ", " << vocab << "]");
            }
        }
    }

    bool MultiDeviceOrchestrator::gatherLogits(size_t seq_len)
    {
        if (!combined_logits_ || device_runners_.empty())
        {
            return false;
        }

        // Single device or no TP context - just copy from primary device
        if (!tp_ctx_ || device_runners_.size() == 1)
        {
            const float *primary_logits = device_runners_[0]->logits();
            if (primary_logits)
            {
                int vocab = vocab_size();
                // For decode, seq_len=1 so we copy vocab elements
                // For prefill, seq_len * vocab elements
                size_t copy_size = seq_len * static_cast<size_t>(vocab);
                std::memcpy(combined_logits_->mutable_data(), primary_logits,
                            copy_size * sizeof(float));
                last_gathered_logits_size_ = copy_size;
            }
            return true;
        }

        // Check if column-parallel LM head is enabled by checking if any device
        // has logits_local allocated
        bool has_column_parallel_lm_head = false;
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                const auto &state = runner->inferenceState();
                if (state.logits_local)
                {
                    has_column_parallel_lm_head = true;
                    break;
                }
            }
        }

        if (!has_column_parallel_lm_head)
        {
            // LM head is replicated, not sharded - use primary device's full logits
            const float *primary_logits = device_runners_[0]->logits();
            if (primary_logits)
            {
                int vocab = vocab_size();
                size_t copy_size = seq_len * static_cast<size_t>(vocab);
                std::memcpy(combined_logits_->mutable_data(), primary_logits,
                            copy_size * sizeof(float));
                last_gathered_logits_size_ = copy_size;
            }
            return true;
        }

        // Column-parallel LM head: each device has logits_local [seq_len, vocab_local]
        // We need to gather along the vocab dimension (axis=1), producing [seq_len, vocab_total]
        //
        // For each row (token position), concatenate vocab_local slices from all devices:
        //   output[row, 0:vocab_local_0] = device_0_logits[row, :]
        //   output[row, vocab_local_0:vocab_local_0+vocab_local_1] = device_1_logits[row, :]
        //   etc.

        // Collect info from all devices
        std::vector<const float *> device_data;
        std::vector<size_t> vocab_locals;

        for (const auto &runner : device_runners_)
        {
            if (!runner)
            {
                LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: null device runner");
                return false;
            }

            const auto &state = runner->inferenceState();
            if (!state.logits_local)
            {
                LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: device missing logits_local");
                return false;
            }

            const auto &shape = state.logits_local->shape();
            if (shape.size() < 2)
            {
                LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: logits_local must be 2D");
                return false;
            }

            // Use the passed-in seq_len instead of reading from tensor shape
            // (tensor shape is pre-allocated for max_seq_len, but for decode we only want 1 row)
            size_t this_vocab_local = shape[1];

            device_data.push_back(state.logits_local->data());
            vocab_locals.push_back(this_vocab_local);
        }

        // Calculate total vocab and validate output buffer
        size_t total_vocab = 0;
        for (size_t vl : vocab_locals)
        {
            total_vocab += vl;
        }

        size_t expected_output_size = seq_len * total_vocab;
        if (combined_logits_->numel() < expected_output_size)
        {
            LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: output buffer too small. "
                      << "Need " << expected_output_size << ", have " << combined_logits_->numel());
            return false;
        }

        // Perform row-by-row column gathering
        float *output = combined_logits_->mutable_data();

        // Debug: Log first elements from each device for decode debugging
        if (seq_len == 1 && device_data.size() >= 2)
        {
            LOG_DEBUG("MultiDeviceOrchestrator::gatherLogits: DECODE DEBUG" << " device0_first=" << device_data[0][0] << " device0_max_idx=" << [&]()
                      {
                          size_t max_idx = 0;
                          float max_val = device_data[0][0];
                          for (size_t i = 1; i < vocab_locals[0]; ++i) {
                              if (device_data[0][i] > max_val) {
                                  max_val = device_data[0][i];
                                  max_idx = i;
                              }
                          }
                          return std::to_string(max_idx) + " (val=" + std::to_string(max_val) + ")"; }() << " device1_first=" << device_data[1][0] << " device1_max_idx=" << [&]()
                      {
                          size_t max_idx = 0;
                          float max_val = device_data[1][0];
                          for (size_t i = 1; i < vocab_locals[1]; ++i) {
                              if (device_data[1][i] > max_val) {
                                  max_val = device_data[1][i];
                                  max_idx = i;
                              }
                          }
                          return std::to_string(max_idx) + " (val=" + std::to_string(max_val) + ")"; }());
        }

        for (size_t row = 0; row < seq_len; ++row)
        {
            size_t col_offset = 0;
            for (size_t dev = 0; dev < device_data.size(); ++dev)
            {
                const float *src = device_data[dev] + row * vocab_locals[dev];
                float *dst = output + row * total_vocab + col_offset;
                std::memcpy(dst, src, vocab_locals[dev] * sizeof(float));
                col_offset += vocab_locals[dev];
            }
        }

        // Store the actual gathered size for getSnapshot()
        last_gathered_logits_size_ = expected_output_size;

        LOG_DEBUG("MultiDeviceOrchestrator::gatherLogits: gathered column-parallel logits "
                  << "[" << seq_len << ", " << total_vocab << "] from " << device_data.size() << " devices");

        return true;
    }

    void MultiDeviceOrchestrator::aggregateStats() const
    {
        if (!stats_dirty_ || device_runners_.empty())
        {
            return;
        }

        // Reset or create aggregated stats
        if (!aggregated_stats_)
        {
            aggregated_stats_ = std::make_unique<GraphExecutorStats>();
        }
        aggregated_stats_->reset();

        // Aggregate from all device runners
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                const auto *stats = runner->executorStats();
                if (stats)
                {
                    aggregated_stats_->total_stages_executed += stats->total_stages_executed;
                    aggregated_stats_->total_flops += stats->total_flops;
                    aggregated_stats_->total_time_ms += stats->total_time_ms;
                    aggregated_stats_->total_execute_ms += stats->total_execute_ms;

                    // Merge stage times
                    for (const auto &[stage_name, time_ms] : stats->stage_times_ms)
                    {
                        aggregated_stats_->stage_times_ms[stage_name] += time_ms;
                    }
                }
            }
        }

        // Average the times (since devices run in parallel)
        if (!device_runners_.empty())
        {
            size_t count = device_runners_.size();
            aggregated_stats_->total_time_ms /= static_cast<double>(count);
            aggregated_stats_->total_execute_ms /= static_cast<double>(count);
            for (auto &[stage_name, time_ms] : aggregated_stats_->stage_times_ms)
            {
                time_ms /= static_cast<double>(count);
            }
        }

        stats_dirty_ = false;
    }

    // =========================================================================
    // IInferenceRunner Interface Implementation
    // =========================================================================

    bool MultiDeviceOrchestrator::forward(const int *tokens, int seq_len)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("MultiDeviceOrchestrator::forward: No device runners available");
            return false;
        }

        LOG_DEBUG("MultiDeviceOrchestrator::forward: seq_len=" << seq_len
                                                               << ", devices=" << device_runners_.size());

        // Launch parallel forward passes on all devices
        std::vector<std::future<bool>> futures;
        futures.reserve(device_runners_.size());

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            auto &runner = device_runners_[i];
            if (runner)
            {
                // Cast to IInferenceRunner* to disambiguate the forward() call
                // DeviceGraphOrchestrator has both forward(tokens, seq_len) -> bool
                // and forward(tokens, seq_len, batch_size=1) -> const float*
                IInferenceRunner *runner_iface = runner.get();
                futures.push_back(std::async(std::launch::async,
                                             [runner_iface, tokens, seq_len]() -> bool
                                             {
                                                 return runner_iface->forward(tokens, seq_len);
                                             }));
            }
        }

        // Wait for all to complete and check results
        // IMPORTANT: Store the FIRST exception so we can re-throw it with the real error message.
        // When one device throws (e.g., VerificationFailure with NaN/Inf), it can cause CUDA
        // context destruction, which then makes other devices fail with misleading "context is
        // destroyed" errors. We want to surface the original root cause exception.
        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;

        for (size_t i = 0; i < futures.size(); ++i)
        {
            try
            {
                bool success = futures[i].get();
                if (!success)
                {
                    LOG_ERROR("MultiDeviceOrchestrator::forward: Device " << i << " forward failed");
                    all_success = false;
                }
            }
            catch (const std::exception &e)
            {
                all_success = false;

                // Check if this is a secondary "context destroyed" error vs the real root cause
                std::string error_msg = e.what();
                bool is_context_destroyed = (error_msg.find("context is destroyed") != std::string::npos ||
                                             error_msg.find("context destroyed") != std::string::npos ||
                                             error_msg.find("error 709") != std::string::npos);

                if (!first_exception)
                {
                    // This is the first exception - store it
                    first_exception = std::current_exception();
                    first_exception_device = i;
                    LOG_ERROR("MultiDeviceOrchestrator::forward: Device " << i
                                                                          << " threw PRIMARY exception: " << error_msg);
                }
                else if (!is_context_destroyed)
                {
                    // This is a substantive error (not just context cleanup failure)
                    // Replace the stored exception if the first one was a context error
                    try
                    {
                        std::rethrow_exception(first_exception);
                    }
                    catch (const std::exception &first_e)
                    {
                        std::string first_msg = first_e.what();
                        bool first_is_context_destroyed =
                            (first_msg.find("context is destroyed") != std::string::npos ||
                             first_msg.find("context destroyed") != std::string::npos ||
                             first_msg.find("error 709") != std::string::npos);

                        if (first_is_context_destroyed)
                        {
                            // Replace context error with the real error
                            LOG_WARN("MultiDeviceOrchestrator::forward: Replacing secondary context error "
                                     "with primary error from device "
                                     << i);
                            first_exception = std::current_exception();
                            first_exception_device = i;
                        }
                    }
                    LOG_ERROR("MultiDeviceOrchestrator::forward: Device " << i
                                                                          << " threw exception: " << error_msg);
                }
                else
                {
                    // Secondary context error - log but don't replace the primary exception
                    LOG_WARN("MultiDeviceOrchestrator::forward: Device " << i
                                                                         << " threw SECONDARY exception (likely due to primary failure): "
                                                                         << error_msg);
                }
            }
        }

        // If we captured an exception, re-throw it with full context
        if (first_exception)
        {
            LOG_ERROR("MultiDeviceOrchestrator::forward: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (all_success)
        {
            // Gather logits from all devices
            // Pass seq_len so gatherLogits knows how many rows to gather
            // (logits_local buffer is pre-allocated for max_seq_len)
            if (!gatherLogits(static_cast<size_t>(seq_len)))
            {
                LOG_ERROR("MultiDeviceOrchestrator::forward: Failed to gather logits");
                all_success = false;
            }

            // Update position tracking
            current_position_ += seq_len;
            current_padded_seq_len_ = seq_len;
            stats_dirty_ = true;
        }

        return all_success;
    }

    const float *MultiDeviceOrchestrator::logits() const
    {
        // Return combined logits if available
        if (combined_logits_ && device_runners_.size() > 1)
        {
            return combined_logits_->data();
        }

        // For single device, return primary device's logits
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->logits();
        }

        return nullptr;
    }

    bool MultiDeviceOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("MultiDeviceOrchestrator::forward_batch: No device runners available");
            return false;
        }

        LOG_DEBUG("MultiDeviceOrchestrator::forward_batch: batch_size=" << token_batches.size()
                                                                        << ", devices=" << device_runners_.size());

        // Launch parallel batch forward passes on all devices
        std::vector<std::future<bool>> futures;
        futures.reserve(device_runners_.size());

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            auto &runner = device_runners_[i];
            if (runner)
            {
                futures.push_back(std::async(std::launch::async,
                                             [&runner, &token_batches]()
                                             {
                                                 return runner->forward_batch(token_batches);
                                             }));
            }
        }

        // Wait for all to complete - using same exception capture pattern as forward()
        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;

        for (size_t i = 0; i < futures.size(); ++i)
        {
            try
            {
                bool success = futures[i].get();
                if (!success)
                {
                    LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Device " << i << " forward_batch failed");
                    all_success = false;
                }
            }
            catch (const std::exception &e)
            {
                all_success = false;
                std::string error_msg = e.what();
                bool is_context_destroyed = (error_msg.find("context is destroyed") != std::string::npos ||
                                             error_msg.find("error 709") != std::string::npos);

                if (!first_exception)
                {
                    first_exception = std::current_exception();
                    first_exception_device = i;
                    LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Device " << i
                                                                                << " threw PRIMARY exception: " << error_msg);
                }
                else if (is_context_destroyed)
                {
                    LOG_WARN("MultiDeviceOrchestrator::forward_batch: Device " << i
                                                                               << " threw SECONDARY exception (context destroyed): " << error_msg);
                }
                else
                {
                    LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Device " << i
                                                                                << " threw exception: " << error_msg);
                }
            }
        }

        if (first_exception)
        {
            LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (all_success)
        {
            // Update batch tracking from primary device
            if (!device_runners_.empty() && device_runners_[0])
            {
                current_batch_size_ = device_runners_[0]->batch_size();
                current_padded_seq_len_ = device_runners_[0]->padded_seq_len();
                current_sequence_lengths_ = device_runners_[0]->sequence_lengths();
            }
            stats_dirty_ = true;
        }

        return all_success;
    }

    const float *MultiDeviceOrchestrator::getLogits(int seq_idx) const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getLogits(seq_idx);
        }
        return nullptr;
    }

    int MultiDeviceOrchestrator::batch_size() const
    {
        return current_batch_size_;
    }

    int MultiDeviceOrchestrator::padded_seq_len() const
    {
        return current_padded_seq_len_;
    }

    const std::vector<int> &MultiDeviceOrchestrator::sequence_lengths() const
    {
        return current_sequence_lengths_;
    }

    int MultiDeviceOrchestrator::vocab_size() const
    {
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->vocab_size();
        }
        if (model_ctx_)
        {
            return static_cast<int>(model_ctx_->vocabSize());
        }
        return 0;
    }

    void MultiDeviceOrchestrator::clear_cache()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::clear_cache: Clearing cache on all "
                  << device_runners_.size() << " devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->clear_cache();
            }
        }

        current_position_ = 0;
        stats_dirty_ = true;
    }

    int MultiDeviceOrchestrator::get_position() const
    {
        // Return position from primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->get_position();
        }
        return current_position_;
    }

    ExecutionPath MultiDeviceOrchestrator::executionPath() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->executionPath();
        }
        return ExecutionPath::GRAPH;
    }

    const char *MultiDeviceOrchestrator::architecture() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->architecture();
        }
        return "Unknown";
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void MultiDeviceOrchestrator::enableSnapshotCapture(const std::string &output_dir)
    {
        LOG_DEBUG("MultiDeviceOrchestrator::enableSnapshotCapture: Enabling on all devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->enableSnapshotCapture(output_dir);
            }
        }
    }

    void MultiDeviceOrchestrator::disableSnapshotCapture()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::disableSnapshotCapture: Disabling on all devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->disableSnapshotCapture();
            }
        }
    }

    void MultiDeviceOrchestrator::clearSnapshots()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::clearSnapshots: Clearing on all devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->clearSnapshots();
            }
        }
    }

    const float *MultiDeviceOrchestrator::getSnapshot(const std::string &key, size_t &out_size) const
    {
        // For LM_HEAD with multi-device TP, return the gathered combined_logits
        // This is necessary because each device only has logits_local with vocab_local entries,
        // but tests expect the full vocab_size logits.
        if (key == "LM_HEAD" && device_runners_.size() > 1 && combined_logits_ && tp_ctx_)
        {
            // Check if we have column-parallel LM head
            bool has_column_parallel_lm_head = false;
            for (const auto &runner : device_runners_)
            {
                if (runner)
                {
                    const auto &state = runner->inferenceState();
                    if (state.logits_local)
                    {
                        has_column_parallel_lm_head = true;
                        break;
                    }
                }
            }

            if (has_column_parallel_lm_head)
            {
                // Return the combined logits which have full vocab_size
                // Use the actual gathered size from last gatherLogits() call,
                // NOT the buffer capacity (which is pre-allocated for max_seq_len)
                out_size = last_gathered_logits_size_;
                const float *ptr = combined_logits_->data();
                LOG_DEBUG("MultiDeviceOrchestrator::getSnapshot LM_HEAD returning combined_logits with "
                          << out_size << " elements (column-parallel gathering), ptr=" << (void *)ptr
                          << " first_element=" << (ptr ? ptr[0] : -999999.0f));
                return ptr;
            }
        }

        // Default: get from primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getSnapshot(key, out_size);
        }
        out_size = 0;
        return nullptr;
    }

    std::vector<std::string> MultiDeviceOrchestrator::getSnapshotKeys() const
    {
        // Merge keys from all devices (use set to deduplicate)
        std::set<std::string> all_keys;

        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                auto keys = runner->getSnapshotKeys();
                all_keys.insert(keys.begin(), keys.end());
            }
        }

        return std::vector<std::string>(all_keys.begin(), all_keys.end());
    }

    // =========================================================================
    // Profiling API
    // =========================================================================

    const GraphExecutorStats *MultiDeviceOrchestrator::executorStats() const
    {
        aggregateStats();
        return aggregated_stats_.get();
    }

    void MultiDeviceOrchestrator::resetExecutorStats()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::resetExecutorStats: Resetting on all devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->resetExecutorStats();
            }
        }

        if (aggregated_stats_)
        {
            aggregated_stats_->reset();
        }
        stats_dirty_ = true;
    }

    // =========================================================================
    // Orchestration API
    // =========================================================================

    bool MultiDeviceOrchestrator::hasPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->hasPlacementPlan();
        }
        return false;
    }

    const PlacementPlan &MultiDeviceOrchestrator::getPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getPlacementPlan();
        }
        throw std::runtime_error("No placement plan available: no device runners");
    }

    // =========================================================================
    // IMultiDeviceOrchestrator Interface Implementation
    // =========================================================================

    int MultiDeviceOrchestrator::device_count() const
    {
        return static_cast<int>(device_runners_.size());
    }

    IInferenceRunner *MultiDeviceOrchestrator::deviceRunner(int device_idx)
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    const IInferenceRunner *MultiDeviceOrchestrator::deviceRunner(int device_idx) const
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    ILocalTPContext *MultiDeviceOrchestrator::localTPContext()
    {
        return tp_ctx_.get();
    }

    const ILocalTPContext *MultiDeviceOrchestrator::localTPContext() const
    {
        return tp_ctx_.get();
    }

    bool MultiDeviceOrchestrator::allDevicesReady() const
    {
        if (device_runners_.empty())
        {
            return false;
        }

        for (const auto &runner : device_runners_)
        {
            if (!runner)
            {
                return false;
            }
            // Check if runner is ready (has vocab_size > 0 indicates initialization)
            if (runner->vocab_size() <= 0)
            {
                return false;
            }
        }

        return true;
    }

    void MultiDeviceOrchestrator::synchronizeDevices()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::synchronizeDevices: Synchronizing all devices");

        if (tp_ctx_)
        {
            tp_ctx_->synchronize();
        }
    }

} // namespace llaminar2

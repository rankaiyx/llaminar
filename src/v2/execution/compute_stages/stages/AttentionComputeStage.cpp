/**
 * @file AttentionComputeStage.cpp
 * @brief Implementation of AttentionComputeStage
 */

#include "AttentionComputeStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../tensors/TQ4Tensor.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../kernels/cpu/turboquant/TurboQuantDequantize.h"
#include <limits>
#include <fstream>
#include <filesystem>

namespace llaminar2
{

    // =============================================================================
    // AttentionComputeStage Implementation
    // =============================================================================

    AttentionComputeStage::AttentionComputeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        // Pre-allocate TQ4 dequant buffers at construction time (model load)
        // to avoid heap allocations during the decode hot path.
        if (params_.turboquant_ctx && params_.kv_cache)
        {
            const int kv_dim = params_.n_kv_heads * params_.head_dim;
            const int max_len = params_.kv_cache->max_seq_len();
            const size_t buf_size = static_cast<size_t>(max_len) * static_cast<size_t>(kv_dim);

            // V always needs FP32 dequant buffer
            tq_decode_v_fp32_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{buf_size});

            if (params_.head_dim != 128)
            {
                // K and V both need FP32 dequant
                tq_decode_k_fp32_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{buf_size});
            }
        }
    }

    // =============================================================================
    // Kernel Caching (for Workspace Binding)
    // =============================================================================

    ITensorAttention *AttentionComputeStage::getOrCreateKernel()
    {
        // Need Q tensor to determine kernel type
        if (!params_.Q)
        {
            LOG_ERROR("[AttentionComputeStage::getOrCreateKernel] Q tensor not set");
            return nullptr;
        }

        auto *kernel = getOrRefreshKernelByTensorType(
            cached_kernel_,
            cached_kernel_tensor_type_,
            params_.Q,
            [&]()
            {
                return llaminar::v2::kernels::KernelFactory::getOrCreateAttention(params_.Q, params_.device_id);
            });
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage::getOrCreateKernel] Failed to create attention kernel for "
                      << params_.Q->dtype_name());
            return nullptr;
        }

        LOG_DEBUG("[AttentionComputeStage::getOrCreateKernel] Created and cached attention kernel for "
                  << params_.Q->dtype_name() << " on " << params_.device_id.to_string());

        return kernel;
    }

    IWorkspaceConsumer *AttentionComputeStage::getKernelAsWorkspaceConsumer()
    {
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            return nullptr;
        }
        return dynamic_cast<IWorkspaceConsumer *>(kernel);
    }

    bool AttentionComputeStage::execute(IDeviceContext *ctx)
    {
        // Dynamic kv_len: query from KV cache at execution time if available
        // This enables declarative graph construction where the stage runs after
        // KVCacheAppendStage has already appended tokens
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            effective_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            if (effective_kv_len == 0)
            {
                effective_kv_len = params_.seq_len; // Prefill case
            }
            LOG_TRACE("[AttentionComputeStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }

        // Read K/V from cache at execution time when requested.
        // This allows GPU prefill to use the FP16 tensors in the KV cache
        // (populated by KVCacheAppendStage) instead of the Q8_1 projection
        // buffers, eliminating the Q8_1→FP32→FP16 triple conversion.
        ITensor *effective_K = params_.K;
        ITensor *effective_V = params_.V;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            // Always override K/V from KV cache when:
            // 1. read_kv_from_cache is set (GPU optimization for type conversion), OR
            // 2. We're in decode mode (effective_kv_len > seq_len) - the graph may have
            //    been built during prefill (cached_tokens=0) with K/V wired to activation
            //    scratch buffers. During decode, those buffers only hold the current token's
            //    projection. The full KV history lives in the cache (populated by
            //    KVCacheAppendStage which runs before this stage).
            if (params_.read_kv_from_cache || effective_kv_len > params_.seq_len)
            {
                ITensor *cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                ITensor *cache_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                if (cache_k && cache_v)
                {
                    effective_K = cache_k;
                    effective_V = cache_v;
                    LOG_TRACE("[AttentionComputeStage] Using cache K/V ("
                              << cache_k->dtype_name() << ") for layer " << params_.layer_idx);
                }
                else
                {
                    LOG_TRACE("[AttentionComputeStage] Cache K/V not available yet, using wired tensors");
                }
            }
        }

        // Detect attention mode if auto-detection enabled
        AttentionMode mode = params_.attention_mode;
        if (params_.auto_detect_mode)
        {
            mode = detect_attention_mode(params_.batch_size, params_.seq_len, effective_kv_len);
        }

        // =====================================================================
        // CPU DECODE: FP16 KV handling
        //
        // The attention kernel has two paths for FP16 KV on CPU decode:
        //
        // 1. FP16-native path (AVX-512 + F16C): The kernel reads FP16 K/V
        //    directly and converts to FP32 on the fly in the dot product and
        //    V accumulation inner loops. This halves KV memory bandwidth and
        //    eliminates persistent FP32 buffers that cause DRAM bank
        //    disturbance degrading subsequent GEMM at long context (>300 tok).
        //
        // 2. FP32 buffer path (fallback): Persistent FP32 buffers with
        //    incremental conversion. Used when AVX-512/F16C not available.
        //
        // The kernel's compute_tensor() detects FP16 K/V tensors and routes
        // to path 1 automatically. We only need to avoid creating the FP32
        // buffers when the hardware supports the native path.
        // =====================================================================
        const bool is_decode_mode = (mode == AttentionMode::DECODE ||
                                     (params_.seq_len < effective_kv_len && params_.batch_size == 1));

        // Check if FP16-native decode path is available (AVX-512 + F16C)
        const bool fp16_native_available =
#if defined(__AVX512F__) && defined(__F16C__)
            true;
#else
            false;
#endif

        if (is_decode_mode && gpuStream() == nullptr)
        {
            auto *K_fp16 = dynamic_cast<FP16Tensor *>(effective_K);
            auto *V_fp16 = dynamic_cast<FP16Tensor *>(effective_V);

            if (K_fp16 && V_fp16 && !fp16_native_available)
            {
                // Fallback: maintain persistent FP32 buffers for platforms
                // without AVX-512 F16C support
                const int kv_dim = params_.n_kv_heads * params_.head_dim;
                const int max_len = params_.kv_cache ? params_.kv_cache->max_seq_len() : effective_kv_len;

                // Lazy init: allocate persistent FP32 buffers once at max capacity
                if (!decode_k_fp32_)
                {
                    decode_k_fp32_ = std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_len) * static_cast<size_t>(kv_dim)});
                    decode_v_fp32_ = std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_len) * static_cast<size_t>(kv_dim)});
                    decode_kv_fp32_rows_ = 0;
                }

                // Detect KV cache clear (kv_len decreased since last call)
                if (effective_kv_len < decode_kv_fp32_rows_)
                {
                    decode_kv_fp32_rows_ = 0;
                }

                // Convert only newly appended rows (typically 1 per decode step)
                if (decode_kv_fp32_rows_ < effective_kv_len)
                {
                    const size_t from = static_cast<size_t>(decode_kv_fp32_rows_);
                    const size_t n_new = (static_cast<size_t>(effective_kv_len) - from) * static_cast<size_t>(kv_dim);
                    const size_t offset = from * static_cast<size_t>(kv_dim);

                    simd::convert_fp16_to_fp32(
                        K_fp16->typed_data() + offset,
                        decode_k_fp32_->mutable_data() + offset,
                        n_new);
                    simd::convert_fp16_to_fp32(
                        V_fp16->typed_data() + offset,
                        decode_v_fp32_->mutable_data() + offset,
                        n_new);

                    decode_kv_fp32_rows_ = effective_kv_len;
                }

                effective_K = decode_k_fp32_.get();
                effective_V = decode_v_fp32_.get();
            }
            // When fp16_native_available && K/V are FP16: pass FP16 tensors
            // directly to the kernel. compute_tensor() will detect FP16 K/V
            // and use compute_decode_fp16kv() for zero-copy FP16 attention.
        }

        // =====================================================================
        // CPU DECODE: TQ4 KV dequantization
        //
        // TurboQuant KV cache: both K and V are dequantized to FP32.
        // Scalar-full mode uses 4-bit MSE centroids for reconstruction.
        // =====================================================================
        if (is_decode_mode && gpuStream() == nullptr)
        {
            auto *K_tq4 = dynamic_cast<TQ4Tensor *>(effective_K);
            auto *V_tq4 = dynamic_cast<TQ4Tensor *>(effective_V);

            if (K_tq4 && V_tq4)
            {
                if (!params_.turboquant_ctx)
                {
                    LOG_ERROR("[AttentionComputeStage] TQ4 KV cache requires turboquant_ctx in params");
                    return false;
                }
                const auto &layer_turboquant_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);
                const auto *turboquant_ctx = &layer_turboquant_ctx;

                // Set the shared TurboQuant context on cache tensors so dequant is possible
                K_tq4->set_turboquant_context(turboquant_ctx);
                V_tq4->set_turboquant_context(turboquant_ctx);

                const int kv_dim = params_.n_kv_heads * params_.head_dim;
                const int max_len = params_.kv_cache
                                        ? params_.kv_cache->max_seq_len()
                                        : effective_kv_len;

                // K+V dequant buffers (pre-allocated in constructor, fallback here for safety)
                if (!tq_decode_k_fp32_)
                {
                    tq_decode_k_fp32_ = std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_len) * static_cast<size_t>(kv_dim)});
                    tq_decode_v_fp32_ = std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_len) * static_cast<size_t>(kv_dim)});
                    tq_decode_fp32_rows_ = 0;
                }

                if (effective_kv_len < tq_decode_fp32_rows_)
                    tq_decode_fp32_rows_ = 0;

                if (tq_decode_fp32_rows_ < effective_kv_len)
                {
                    turboquant_dequantize_kv_rows(
                        K_tq4->typed_data(), V_tq4->typed_data(),
                        *turboquant_ctx,
                        tq_decode_k_fp32_->mutable_data(),
                        tq_decode_v_fp32_->mutable_data(),
                        tq_decode_fp32_rows_, effective_kv_len,
                        params_.head_dim, params_.n_kv_heads,
                        K_tq4->blocks_per_row() * K_tq4->block_bytes(),
                        V_tq4->blocks_per_row() * V_tq4->block_bytes(),
                        K_tq4->block_bytes(), V_tq4->block_bytes());
                    tq_decode_fp32_rows_ = effective_kv_len;
                }

                effective_K = tq_decode_k_fp32_.get();
                effective_V = tq_decode_v_fp32_.get();
            }
        }

        LOG_DEBUG("[AttentionComputeStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << effective_kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " position_offset=" << params_.position_offset
                                                            << " mode=" << attention_mode_name(mode)
                                                            << " Q_type=" << (params_.Q ? params_.Q->dtype_name() : "null")
                                                            << " K_type=" << (effective_K ? effective_K->dtype_name() : "null")
                                                            << " V_type=" << (effective_V ? effective_V->dtype_name() : "null")
                                                            << " output=" << (void *)params_.output);

        // Validate inputs
        if (!ensureRequiredPointers("AttentionComputeStage", {
                                                                 {"Q", params_.Q},
                                                                 {"K", effective_K},
                                                                 {"V", effective_V},
                                                                 {"output", params_.output},
                                                             }))
        {
            return false;
        }

        if (params_.seq_len <= 0 || effective_kv_len <= 0 ||
            params_.n_heads <= 0 || params_.n_kv_heads <= 0 || params_.head_dim <= 0)
        {
            LOG_ERROR("[AttentionComputeStage] Invalid dimensions");
            return false;
        }

        if (params_.n_heads % params_.n_kv_heads != 0)
        {
            LOG_ERROR("[AttentionComputeStage] n_heads (" << params_.n_heads
                                                          << ") must be divisible by n_kv_heads (" << params_.n_kv_heads << ")");
            return false;
        }

        // Use cached kernel (enables workspace binding for GPU kernels)
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage] Failed to get attention kernel");
            return false;
        }
        bindStageStream(kernel);

        // Get device index using proper ordinal for GPU devices (0-based), not legacy index
        int device_idx = params_.device_id.toKernelDeviceIndex();

        // Build proper causal mask for decode mode
        // Key insight: For single-token decode (seq_len=1), the query at position
        // (kv_len-1) can attend to ALL positions, so the mask is all-zeros.
        // Skip mask construction entirely and pass nullptr + causal=false.
        // Only build an explicit mask for multi-token decode (seq_len > 1 but < kv_len).
        std::unique_ptr<FP32Tensor> decode_mask;
        ITensor *mask_to_use = params_.workspace_mask;

        if (params_.causal && is_decode_mode)
        {
            if (gpuStream() == nullptr && params_.seq_len > 1)
            {
                // Multi-token decode on CPU: some tokens need causal masking
                const int base_pos = (params_.position_offset > 0)
                                         ? params_.position_offset
                                         : (effective_kv_len - params_.seq_len);

                decode_mask = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(params_.seq_len * effective_kv_len)});
                float *mask_data = decode_mask->mutable_data();

                for (int q = 0; q < params_.seq_len; ++q)
                {
                    const int q_pos = base_pos + q;
                    for (int k = 0; k < effective_kv_len; ++k)
                    {
                        mask_data[q * effective_kv_len + k] = (k <= q_pos)
                                                                  ? 0.0f
                                                                  : -std::numeric_limits<float>::infinity();
                    }
                }

                mask_to_use = decode_mask.get();
            }
            else
            {
                // Single-token decode (seq_len=1) or GPU stream: mask is all-zeros,
                // equivalent to nullptr + causal=false (attend to all positions).
                mask_to_use = nullptr;
            }
        }

        // Dispatch to kernel's compute method
        // IMPORTANT: For decode with explicit mask, we pass causal=false to avoid
        // double-masking (kernel would apply "n > m" on top of our mask)
        const bool kernel_causal = params_.causal && !is_decode_mode;

        // Device-agnostic unified path using compute_tensor()
        // The kernel factory creates the appropriate kernel (CPU or GPU) based on dev_type,
        // and compute_tensor() handles type dispatch internally.
        // Since compute_tensor() now takes ITensor*, we can pass Q/K/V directly without casting.
        // This allows GPU tensor wrappers (like GpuTensorView from CUDA KV cache) to work.

        if (!ensureRequiredPointers("AttentionComputeStage", {
                                                                 {"Q", params_.Q},
                                                                 {"K", params_.K},
                                                                 {"V", params_.V},
                                                                 {"output", params_.output},
                                                             }))
        {
            return false;
        }

        // Device coherence is now handled automatically by DeviceGraphExecutor at stage boundaries
        // based on the stage's coherencePolicy() (FULL by default)

        LOG_DEBUG("[AttentionComputeStage] Executing kernel: Q_type=" << params_.Q->dtype_name()
                                                                      << " device=" << params_.device_id.to_string()
                                                                      << " device_idx=" << device_idx);

        // =====================================================================
        // DEBUG: Dump effective K/V to binary files for Python analysis
        // Enable with LLAMINAR_DUMP_EFFECTIVE_KV=1
        // Dumps layer 0 data (or all layers with LLAMINAR_DUMP_EFFECTIVE_KV_ALL=1)
        // =====================================================================
        {
            static const bool dump_enabled = (std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV") &&
                                              std::atoi(std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV")) != 0);
            static const bool dump_all = (std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV_ALL") &&
                                          std::atoi(std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV_ALL")) != 0);

            if (dump_enabled && (dump_all || params_.layer_idx == 0) && is_decode_mode)
            {
                static int dump_iteration = 0;
                const std::string dump_dir = "/tmp/effective_kv_dump/layer" +
                                             std::to_string(params_.layer_idx) +
                                             "_iter" + std::to_string(dump_iteration);
                std::filesystem::create_directories(dump_dir);

                // Write metadata
                {
                    std::ofstream meta(dump_dir + "/meta.txt");
                    meta << "layer=" << params_.layer_idx << "\n"
                         << "iteration=" << dump_iteration << "\n"
                         << "seq_len=" << params_.seq_len << "\n"
                         << "kv_len=" << effective_kv_len << "\n"
                         << "n_heads=" << params_.n_heads << "\n"
                         << "n_kv_heads=" << params_.n_kv_heads << "\n"
                         << "head_dim=" << params_.head_dim << "\n"
                         << "batch_size=" << params_.batch_size << "\n"
                         << "mode=" << attention_mode_name(mode) << "\n"
                         << "Q_type=" << (params_.Q ? params_.Q->dtype_name() : "null") << "\n"
                         << "K_type=" << (effective_K ? effective_K->dtype_name() : "null") << "\n"
                         << "V_type=" << (effective_V ? effective_V->dtype_name() : "null") << "\n"
                         << "K_is_dequanted=" << (effective_K == tq_decode_k_fp32_.get() ? 1 : 0) << "\n"
                         << "V_is_dequanted=" << (effective_V == tq_decode_v_fp32_.get() ? 1 : 0) << "\n"
                         << "K_ptr=" << (void *)effective_K << "\n"
                         << "V_ptr=" << (void *)effective_V << "\n";
                }

                // Dump Q (FP32)
                if (auto *q_fp32 = dynamic_cast<FP32Tensor *>(params_.Q))
                {
                    const size_t q_elems = static_cast<size_t>(params_.seq_len) *
                                           params_.n_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/Q.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(q_fp32->data()),
                            q_elems * sizeof(float));
                }

                // Dump effective K (should be FP32 after dequant)
                if (auto *k_fp32 = dynamic_cast<FP32Tensor *>(effective_K))
                {
                    const size_t k_elems = static_cast<size_t>(effective_kv_len) *
                                           params_.n_kv_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/K_effective.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(k_fp32->data()),
                            k_elems * sizeof(float));
                }

                // Dump effective V (should be FP32 after dequant)
                if (auto *v_fp32 = dynamic_cast<FP32Tensor *>(effective_V))
                {
                    const size_t v_elems = static_cast<size_t>(effective_kv_len) *
                                           params_.n_kv_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/V_effective.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(v_fp32->data()),
                            v_elems * sizeof(float));
                }

                // Dump raw TQ4 K cache if available
                if (params_.kv_cache && params_.layer_idx >= 0)
                {
                    ITensor *cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                    if (auto *k_tq4 = dynamic_cast<TQ4Tensor *>(cache_k))
                    {
                        const size_t raw_bytes = static_cast<size_t>(effective_kv_len) *
                                                 k_tq4->blocks_per_row() * k_tq4->block_bytes();
                        std::ofstream f(dump_dir + "/K_cache_tq4.bin", std::ios::binary);
                        f.write(reinterpret_cast<const char *>(k_tq4->typed_data()),
                                raw_bytes);
                        std::ofstream m(dump_dir + "/K_cache_meta.txt");
                        m << "blocks_per_row=" << k_tq4->blocks_per_row() << "\n"
                          << "block_bytes=" << k_tq4->block_bytes() << "\n"
                          << "head_dim=" << k_tq4->head_dim() << "\n"
                          << "rows=" << effective_kv_len << "\n";
                    }
                }

                LOG_INFO("[AttentionComputeStage] Dumped effective K/V to " << dump_dir);
                if (params_.layer_idx == 0)
                    dump_iteration++;
            }
        }

        bool success = kernel->compute_tensor(
            params_.Q, effective_K, effective_V, params_.output,
            params_.batch_size,
            params_.seq_len,
            effective_kv_len,
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            kernel_causal, // Pass false for decode (we built the mask explicitly)
            params_.window_size,
            params_.workspace_scores,
            mask_to_use, // Use our decode mask if we built one
            params_.mpi_ctx,
            device_idx);

        // Device coherence (mark_device_dirty) is now handled automatically by DeviceGraphExecutor
        // at stage boundaries based on the stage's coherencePolicy() (FULL by default)

        if (!success)
        {
            LOG_ERROR("[AttentionComputeStage] Kernel compute_tensor() failed");
            return false;
        }

        LOG_DEBUG("[AttentionComputeStage] Execute complete (mode=" << attention_mode_name(mode) << ")");
        return true;
    }

    size_t AttentionComputeStage::estimatedFlops() const
    {
        // Attention FLOPs:
        // Q @ K^T: 2 * batch * n_heads * seq_len * kv_len * head_dim
        // softmax: ~4 * batch * n_heads * seq_len * kv_len
        // scores @ V: 2 * batch * n_heads * seq_len * kv_len * head_dim
        const size_t qk_flops = 2ULL * params_.batch_size * params_.n_heads *
                                params_.seq_len * params_.kv_len * params_.head_dim;
        const size_t softmax_flops = 4ULL * params_.batch_size * params_.n_heads *
                                     params_.seq_len * params_.kv_len;
        const size_t sv_flops = qk_flops;
        return qk_flops + softmax_flops + sv_flops;
    }

    size_t AttentionComputeStage::estimatedMemoryBytes() const
    {
        // Workspace for attention scores: n_heads * seq_len * kv_len
        return static_cast<size_t>(params_.n_heads) * params_.seq_len * params_.kv_len * sizeof(float);
    }

    bool AttentionComputeStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageDumpInfo AttentionComputeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input: Q, K, V tensors
        // Q shape: [batch_size * seq_len, n_heads * head_dim]
        // K/V shape: [batch_size * kv_len, n_kv_heads * head_dim]
        const size_t total_q_tokens = static_cast<size_t>(params_.batch_size * params_.seq_len);
        const size_t total_kv_tokens = static_cast<size_t>(params_.batch_size * params_.kv_len);

        if (params_.Q)
        {
            info.addInput("Q", params_.Q, total_q_tokens, params_.n_heads * params_.head_dim);
        }
        if (params_.K)
        {
            info.addInput("K", params_.K, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
        }
        if (params_.V)
        {
            info.addInput("V", params_.V, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
        }

        // Output: attention context
        // Output shape: [batch_size * seq_len, n_heads * head_dim]
        if (params_.output)
        {
            LOG_DEBUG("[AttentionComputeStage::getDumpInfo] output=" << (void *)params_.output
                                                                     << " type=" << params_.output->dtype_name()
                                                                     << " batch_size=" << params_.batch_size
                                                                     << " seq_len=" << params_.seq_len
                                                                     << " total_tokens=" << total_q_tokens
                                                                     << " n_heads*head_dim=" << (params_.n_heads * params_.head_dim));
            // Use ITensor* overload to enable coherence tracking
            info.addOutput("output", params_.output, total_q_tokens, params_.n_heads * params_.head_dim);
        }
        else
        {
            LOG_DEBUG("[AttentionComputeStage::getDumpInfo] output is NULL");
        }

        // Scalars capture all necessary info for debugging
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);
        info.addScalarInt("window_size", params_.window_size);
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        // Add attention mode info (as int - PREFILL=0, DECODE=1, BATCHED_DECODE=2, CHUNKED_PREFILL=3)
        AttentionMode mode = params_.auto_detect_mode
                                 ? detect_attention_mode(params_.batch_size, params_.seq_len, params_.kv_len)
                                 : params_.attention_mode;
        info.addScalarInt("attention_mode", static_cast<int>(mode));
        info.addScalarBool("auto_detect_mode", params_.auto_detect_mode);

        return info;
    }

    StageBufferRequirements AttentionComputeStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q (query)
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {q_rows, q_cols}, buf_type);
        }

        // Input: K (key - may have different kv_len than Q's seq_len)
        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {k_rows, k_cols}, buf_type);
        }

        // Input: V (value)
        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {v_rows, v_cols}, buf_type);
        }

        // Output: attention output
        if (params_.output)
        {
            const size_t out_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {out_rows, out_cols}, buf_type);
        }

        // Scratch: workspace buffers (if pre-allocated)
        if (params_.workspace_scores)
        {
            reqs.addScratch("workspace_scores", params_.workspace_scores->shape(),
                            toBufferTensorType(params_.workspace_scores->native_type()));
        }
        if (params_.workspace_context)
        {
            reqs.addScratch("workspace_context", params_.workspace_context->shape(),
                            toBufferTensorType(params_.workspace_context->native_type()));
        }

        return reqs;
    }

    StageBufferContract AttentionComputeStage::bufferContract() const
    {
        if (!params_.q_buffer_id || !params_.output_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
                            .addInput(*params_.q_buffer_id)
                            .addOutput(*params_.output_buffer_id);

        if (params_.workspace_scores_buffer_id)
            contract.addInOut(*params_.workspace_scores_buffer_id);
        if (params_.workspace_context_buffer_id)
            contract.addInOut(*params_.workspace_context_buffer_id);

        return contract;
    }

} // namespace llaminar2

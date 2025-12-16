/**
 * @file ComputeStage.cpp
 * @brief Implementation of compute stage abstractions
 * @author David Sanftenberg
 * @date December 2025
 */

#include "ComputeStage.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../utils/OpenMPUtils.h"
#include "../kernels/KernelFactory.h"
#include "../kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "../tensors/UnifiedKVCache.h"
#include "../tensors/SIMDHelpers.h"
#include "../tensors/Tensors.h" // For TensorBase::rows(), cols(), dtype_name()
#include "../pipelines/attention/MpiAttentionOrchestrator.h"
#include "../pipelines/MPIStrategy.h"
#include "../utils/MPIContext.h"

#include <cstring>
#include <cmath>
#include <sstream>
#include <omp.h>

// Note: In production, GEMM would delegate to KernelFactory
// For now, we use placeholder implementation to avoid cross-namespace dependencies

#include <mpi.h>

namespace llaminar2
{

    // =============================================================================
    // Stage Type Names
    // =============================================================================

    const char *computeStageTypeName(ComputeStageType type)
    {
        switch (type)
        {
        case ComputeStageType::GEMM:
            return "GEMM";
        case ComputeStageType::GEMM_BIAS:
            return "GEMM_BIAS";
        case ComputeStageType::GEMM_FUSED_QKV:
            return "GEMM_FUSED_QKV";
        case ComputeStageType::RMS_NORM:
            return "RMS_NORM";
        case ComputeStageType::LAYER_NORM:
            return "LAYER_NORM";
        case ComputeStageType::SWIGLU:
            return "SWIGLU";
        case ComputeStageType::GELU:
            return "GELU";
        case ComputeStageType::SILU:
            return "SILU";
        case ComputeStageType::ROPE:
            return "ROPE";
        case ComputeStageType::ATTENTION:
            return "ATTENTION";
        case ComputeStageType::ATTENTION_QK:
            return "ATTENTION_QK";
        case ComputeStageType::ATTENTION_SOFTMAX:
            return "ATTENTION_SOFTMAX";
        case ComputeStageType::ATTENTION_V:
            return "ATTENTION_V";
        case ComputeStageType::ADD_RESIDUAL:
            return "ADD_RESIDUAL";
        case ComputeStageType::SCALE:
            return "SCALE";
        case ComputeStageType::MOE_ROUTER:
            return "MOE_ROUTER";
        case ComputeStageType::MOE_EXPERT_FFN:
            return "MOE_EXPERT_FFN";
        case ComputeStageType::MOE_COMBINE:
            return "MOE_COMBINE";
        case ComputeStageType::ALLREDUCE:
            return "ALLREDUCE";
        case ComputeStageType::ALLGATHER:
            return "ALLGATHER";
        case ComputeStageType::COPY:
            return "COPY";
        case ComputeStageType::QUANTIZE:
            return "QUANTIZE";
        case ComputeStageType::DEQUANTIZE:
            return "DEQUANTIZE";
        default:
            return "UNKNOWN";
        }
    }

    // =============================================================================
    // StageDumpInfo Implementation
    // =============================================================================

    StageDumpInfo &StageDumpInfo::addWeight(const char *name, const TensorBase *tensor)
    {
        if (tensor)
        {
            weights.push_back({name, tensor, nullptr, 0,
                               tensor->rows(), tensor->cols(),
                               tensor->dtype_name()});
        }
        return *this;
    }

    // =============================================================================
    // QuantizeStage Implementation
    // =============================================================================

    QuantizeStage::QuantizeStage(Params params) : params_(std::move(params)) {}

    bool QuantizeStage::execute(IDeviceContext *ctx)
    {
        (void)ctx; // CPU quantization doesn't need device context

        if (!params_.input || !params_.output)
        {
            LOG_ERROR("[QuantizeStage] Null input or output buffer");
            return false;
        }

        if (params_.m <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[QuantizeStage] Invalid dimensions: m=" << params_.m << " k=" << params_.k);
            return false;
        }

        LOG_DEBUG("[QuantizeStage] Quantizing FP32->Q8_1: " << params_.m << "x" << params_.k);

        // Use the same quantization as FusedGEMM/QuantisedGemmKernel
        int k_blocks = (params_.k + 31) / 32;
        Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(params_.output);
        const bool k_aligned = (params_.k % 32 == 0);

        auto do_quantize = [&]()
        {
            int quant_thresh = debugEnv().gemm.gemm_quant_parallel_threshold;
            if (quant_thresh == 0)
                quant_thresh = omp_get_max_threads();

            if (params_.m < quant_thresh)
            {
#pragma omp for collapse(2) schedule(static)
                for (int i = 0; i < params_.m; ++i)
                {
                    for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                    {
                        const float *a_row = params_.input + i * params_.k + k_blk * 32;
                        Q8_1Block *row_blocks = all_blocks + i * k_blocks;
                        int valid_elements = std::min(32, params_.k - k_blk * 32);
                        simd::quantize_single_block(a_row, row_blocks[k_blk], valid_elements);
                    }
                }
            }
            else
            {
#pragma omp for schedule(static)
                for (int i = 0; i < params_.m; ++i)
                {
                    const float *a_row = params_.input + i * params_.k;
                    Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                    if (k_aligned)
                    {
                        for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                        {
                            simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                        }
                    }
                    else
                    {
                        for (int k_blk = 0; k_blk < k_blocks - 1; ++k_blk)
                        {
                            simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                        }
                        if (k_blocks > 0)
                        {
                            int last_k_blk = k_blocks - 1;
                            int valid_elements = params_.k - last_k_blk * 32;
                            simd::quantize_single_block(a_row + last_k_blk * 32, row_blocks[last_k_blk], valid_elements);
                        }
                    }
                }
            }
        };

        OMP_WORKSHARE_REGION(do_quantize);

        return true;
    }

    size_t QuantizeStage::estimatedFlops() const
    {
        // Quantization: ~3 ops per element (find max, scale, round)
        return static_cast<size_t>(3) * params_.m * params_.k;
    }

    size_t QuantizeStage::estimatedMemoryBytes() const
    {
        // Read: m * k floats, Write: m * ceil(k/32) Q8_1Blocks
        size_t read_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);
        size_t write_bytes = get_quantized_buffer_size(params_.m, params_.k);
        return read_bytes + write_bytes;
    }

    bool QuantizeStage::supportsBackend(ComputeBackendType backend) const
    {
        // CPU only for now
        return backend == ComputeBackendType::CPU_OPENBLAS ||
               backend == ComputeBackendType::CPU_MKL;
    }

    size_t QuantizeStage::get_quantized_buffer_size(int m, int k)
    {
        int k_blocks = (k + 31) / 32;
        return static_cast<size_t>(m) * k_blocks * sizeof(Q8_1Block);
    }

    StageDumpInfo QuantizeStage::getDumpInfo() const
    {
        StageDumpInfo info;
        int k_blocks = (params_.k + 31) / 32;

        // Input: FP32 activations
        info.addInput("input", params_.input, params_.m, params_.k);

        // Output: Q8_1 blocks (rows × blocks_per_row blocks)
        info.outputs.push_back({"output_q8_1",
                                params_.output,
                                static_cast<size_t>(params_.m),
                                static_cast<size_t>(k_blocks),
                                "Q8_1",
                                sizeof(Q8_1Block)});

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        info.addScalarInt("k_blocks", k_blocks);

        return info;
    }

    // =============================================================================
    // GEMMStage Implementation
    // =============================================================================

    GEMMStage::GEMMStage(Params params) : params_(std::move(params)) {}

    bool GEMMStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[GEMMStage] Null device context");
            return false;
        }

        // Validate: need either FP32 input (A) or pre-quantized input (A_q8_1)
        if (!params_.A && !params_.A_q8_1)
        {
            LOG_ERROR("[GEMMStage] No input provided: both A and A_q8_1 are null");
            return false;
        }

        if (params_.A && params_.A_q8_1)
        {
            LOG_WARN("[GEMMStage] Both A and A_q8_1 provided, using A_q8_1 (pre-quantized)");
        }

        if (!params_.B || !params_.C)
        {
            LOG_ERROR("[GEMMStage] Invalid parameters: B=" << params_.B << " C=" << params_.C);
            return false;
        }

        if (params_.m <= 0 || params_.n <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[GEMMStage] Invalid dimensions: m=" << params_.m
                                                           << " n=" << params_.n << " k=" << params_.k);
            return false;
        }

        LOG_DEBUG("[GEMMStage] Execute GEMM: " << params_.m << "x" << params_.n << "x" << params_.k
                                               << " mode=" << (params_.A_q8_1 ? "Q8_1" : "FP32")
                                               << " weight ptr=" << static_cast<const void *>(params_.B)
                                               << " weight shape=[" << (params_.B ? params_.B->shape()[0] : 0) << ","
                                               << (params_.B ? params_.B->shape()[1] : 0) << "]");

        // Get cached kernel from KernelFactory (handles weight packing once)
        auto *gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.B);
        if (!gemm)
        {
            LOG_ERROR("[GEMMStage] Failed to get GEMM kernel for weight tensor");
            return false;
        }

        // Cast to QuantisedGemmKernel for full API access
        auto *qgemm = dynamic_cast<gemm_v4::QuantisedGemmKernel *>(gemm);

        // Dimension validation
        if (qgemm)
        {
            int kernel_n = qgemm->get_n();
            int kernel_k = qgemm->get_k();
            LOG_DEBUG("[GEMMStage] Kernel dimensions: N=" << kernel_n << " K=" << kernel_k
                                                          << ", params: n=" << params_.n << " k=" << params_.k);
            if (kernel_n != params_.n || kernel_k != params_.k)
            {
                LOG_ERROR("[GEMMStage] DIMENSION MISMATCH! kernel N=" << kernel_n << " vs params n=" << params_.n
                                                                      << ", kernel K=" << kernel_k << " vs params k=" << params_.k);
                return false;
            }
        }

        if (qgemm)
        {
            bool success;

            if (params_.A_q8_1)
            {
                // =========================================================
                // Q8_1 MODE: Use pre-quantized activations (like FusedGEMM)
                // =========================================================
                LOG_DEBUG("[GEMMStage] Using pre-quantized Q8_1 path");

                // Build fused ops configuration
                GemmFusedOps fused_ops = params_.do_swiglu
                                             ? GemmFusedOps::swiglu(params_.gate_input)
                                             : GemmFusedOps::none();

                success = qgemm->multiply_with_precomputed_q8_1(
                    params_.A_q8_1,
                    static_cast<float *>(params_.C),
                    params_.m, params_.n, params_.k,
                    params_.bias, // Fused bias
                    false,        // No accumulation
                    params_.alpha, params_.beta,
                    params_.mpi_ctx, params_.device_idx,
                    fused_ops);
            }
            else
            {
                // =========================================================
                // FP32 MODE: Quantize internally then GEMM
                // =========================================================
                LOG_DEBUG("[GEMMStage] Using FP32 path with internal quantization");

                // Use multiply_fused to support bias and swiglu
                success = qgemm->multiply_fused(
                    static_cast<const float *>(params_.A),
                    static_cast<float *>(params_.C),
                    params_.m, params_.n, params_.k,
                    params_.bias,           // Fused bias
                    nullptr,                // No attention mask
                    false,                  // No softmax
                    nullptr, nullptr,       // No softmax buffers
                    (params_.beta != 0.0f), // Accumulate if beta != 0
                    params_.alpha, params_.beta,
                    params_.mpi_ctx, params_.device_idx,
                    params_.gate_input,
                    params_.do_swiglu);
            }

            return success;
        }

        // FP32 GEMM fallback (for non-quantized weights)
        if (params_.A_q8_1)
        {
            LOG_ERROR("[GEMMStage] Q8_1 input requires QuantisedGemmKernel, but got non-quantized kernel");
            return false;
        }

        return gemm->multiply(
            static_cast<const float *>(params_.A),
            static_cast<float *>(params_.C),
            params_.m, params_.n, params_.k,
            params_.alpha, params_.beta);
    }

    size_t GEMMStage::estimatedFlops() const
    {
        // GEMM: 2 * M * N * K (multiply + add)
        return static_cast<size_t>(2) * params_.m * params_.n * params_.k;
    }

    size_t GEMMStage::estimatedMemoryBytes() const
    {
        // A: m * k reads, B: k * n reads, C: m * n writes (+ reads if beta != 0)
        size_t a_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);
        size_t c_bytes = static_cast<size_t>(params_.m) * params_.n * sizeof(float);

        // B may be quantized, so we estimate based on tensor
        // For now, assume FP32 - tensor introspection would be better
        size_t b_bytes = static_cast<size_t>(params_.k) * params_.n * sizeof(float);

        return a_bytes + b_bytes + c_bytes;
    }

    bool GEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // CPU always supported
        // GPU support depends on whether we have CUDA/ROCm kernels
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            // TODO: Enable when GPU GEMM kernels are implemented
            return false;
        default:
            return false;
        }
    }

    StageDumpInfo GEMMStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input A: either FP32 or Q8_1
        if (params_.A_q8_1)
        {
            int k_blocks = (params_.k + 31) / 32;
            info.addInputQ8_1("A_q8_1", params_.A_q8_1, params_.m, k_blocks);
        }
        else if (params_.A)
        {
            info.addInput("A", static_cast<const float *>(params_.A), params_.m, params_.k);
        }

        // Weight tensor B
        info.addWeight("B", params_.B);

        // Output C
        info.addOutput("C", static_cast<const float *>(params_.C), params_.m, params_.n);

        // Optional inputs
        if (params_.bias)
        {
            info.addInput("bias", params_.bias, 1, params_.n);
        }
        if (params_.gate_input)
        {
            info.addInput("gate_input", params_.gate_input, params_.m, params_.n);
        }

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("n", params_.n);
        info.addScalarInt("k", params_.k);
        info.addScalar("alpha", params_.alpha);
        info.addScalar("beta", params_.beta);
        info.addScalarBool("transpose_B", params_.transpose_B);
        info.addScalarBool("do_swiglu", params_.do_swiglu);
        info.addScalarInt("device_idx", params_.device_idx);

        return info;
    }

    // =============================================================================
    // RMSNormStage Implementation
    // =============================================================================

    RMSNormStage::RMSNormStage(Params params) : params_(std::move(params)) {}

    bool RMSNormStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[RMSNormStage] Execute: seq_len=" << params_.seq_len
                                                     << " hidden_dim=" << params_.hidden_dim << " eps=" << params_.eps);

        if (!ctx)
        {
            LOG_ERROR("[RMSNormStage] Null device context");
            return false;
        }

        const float *input = static_cast<const float *>(params_.input);
        float *output = static_cast<float *>(params_.output);
        const float *gamma = params_.gamma;
        const int seq_len = params_.seq_len;
        const int hidden_dim = params_.hidden_dim;
        const float eps = params_.eps;

        // Debug: log first few values
        LOG_DEBUG("[RMSNormStage] input[0:4]="
                  << input[0] << "," << input[1] << "," << input[2] << "," << input[3]);

        // Execute RMSNorm using parallel iteration
        // Supports both in-place (input == output) and out-of-place operation
        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t i_)
                    {
        int i = static_cast<int>(i_);
        const float* in_row = input + i * hidden_dim;
        float* out_row = output + i * hidden_dim;
        
        // Compute RMS from input
        float sum_sq = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            sum_sq += in_row[j] * in_row[j];
        }
        float rms = std::sqrt(sum_sq / hidden_dim + eps);
        float inv_rms = 1.0f / rms;
        
        // Normalize and scale, write to output
        for (int j = 0; j < hidden_dim; ++j) {
            out_row[j] = in_row[j] * inv_rms * gamma[j];
        } });

        // Debug: log output values
        LOG_DEBUG("[RMSNormStage] output[0:4]="
                  << output[0] << "," << output[1] << "," << output[2] << "," << output[3]);

        return true;
    }

    size_t RMSNormStage::estimatedFlops() const
    {
        // Per row: hidden_dim squares + hidden_dim adds + sqrt + div + hidden_dim muls
        // Approximately 4 * hidden_dim FLOPs per row
        return static_cast<size_t>(4) * params_.seq_len * params_.hidden_dim;
    }

    size_t RMSNormStage::estimatedMemoryBytes() const
    {
        // Read input + gamma, write output (in-place, so same buffer)
        size_t input_bytes = static_cast<size_t>(params_.seq_len) * params_.hidden_dim * sizeof(float);
        size_t gamma_bytes = static_cast<size_t>(params_.hidden_dim) * sizeof(float);
        return 2 * input_bytes + gamma_bytes; // Read + write + gamma
    }

    bool RMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo RMSNormStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input (may be same as output for in-place)
        info.addInput("input", static_cast<const float *>(params_.input), params_.seq_len, params_.hidden_dim);

        // Gamma weights
        info.addInput("gamma", params_.gamma, 1, params_.hidden_dim);

        // Output
        info.addOutput("output", static_cast<const float *>(params_.output), params_.seq_len, params_.hidden_dim);

        // Scalar params
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("hidden_dim", params_.hidden_dim);
        info.addScalar("eps", params_.eps);

        return info;
    }

    // =============================================================================
    // RoPEStage Implementation
    // =============================================================================

    RoPEStage::RoPEStage(Params params) : params_(std::move(params)) {}

    bool RoPEStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[RoPEStage] Execute: seq_len=" << params_.seq_len
                                                  << " n_heads=" << params_.n_heads << " head_dim=" << params_.head_dim
                                                  << " pos_offset=" << params_.pos_offset);

        if (!ctx)
        {
            LOG_ERROR("[RoPEStage] Null device context");
            return false;
        }

        float *tensor = static_cast<float *>(params_.tensor);
        const int seq_len = params_.seq_len;
        const int n_heads = params_.n_heads;
        const int head_dim = params_.head_dim;
        const int pos_offset = params_.pos_offset;
        const float theta_base = params_.theta_base;

        // Execute RoPE using parallel iteration over positions
        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t pos_)
                    {
        int pos = static_cast<int>(pos_);
        int actual_pos = pos + pos_offset;
        
        for (int h = 0; h < n_heads; ++h) {
            float* head_ptr = tensor + pos * n_heads * head_dim + h * head_dim;
            
            // Apply rotary embedding to pairs of elements
            for (int i = 0; i < head_dim / 2; ++i) {
                float freq = 1.0f / std::pow(theta_base, 
                    static_cast<float>(2 * i) / head_dim);
                float angle = actual_pos * freq;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                float x0 = head_ptr[2 * i];
                float x1 = head_ptr[2 * i + 1];
                
                head_ptr[2 * i]     = x0 * cos_val - x1 * sin_val;
                head_ptr[2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        } });
        return true;
    }

    size_t RoPEStage::estimatedFlops() const
    {
        // Per position per head: head_dim/2 rotations, each ~10 FLOPs (sin, cos, 4 muls, 2 adds)
        return static_cast<size_t>(10) * params_.seq_len * params_.n_heads * (params_.head_dim / 2);
    }

    size_t RoPEStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(2) * params_.seq_len * params_.n_heads *
               params_.head_dim * sizeof(float); // Read + write
    }

    bool RoPEStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo RoPEStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input/output tensor (in-place operation)
        info.addInput("tensor", static_cast<const float *>(params_.tensor),
                      params_.seq_len, params_.n_heads * params_.head_dim);
        info.addOutput("tensor", static_cast<const float *>(params_.tensor),
                       params_.seq_len, params_.n_heads * params_.head_dim);

        // Scalar params
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarInt("pos_offset", params_.pos_offset);
        info.addScalar("theta_base", params_.theta_base);

        return info;
    }

    // =============================================================================
    // AttentionStage Implementation
    // =============================================================================

    AttentionStage::AttentionStage(Params params) : params_(std::move(params)) {}

    bool AttentionStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[AttentionStage] Execute: seq_len=" << params_.seq_len << " kv_len=" << params_.kv_len
                                                       << " n_heads=" << params_.n_heads << " n_kv_heads=" << params_.n_kv_heads
                                                       << " head_dim=" << params_.head_dim << " causal=" << params_.causal);

        if (!ctx)
        {
            LOG_ERROR("[AttentionStage] Null device context");
            return false;
        }

        // This is a simplified implementation - production would use optimized kernels
        const float *Q = static_cast<const float *>(params_.Q);
        const float *K = static_cast<const float *>(params_.K);
        const float *V = static_cast<const float *>(params_.V);
        float *output = static_cast<float *>(params_.output);

        // Debug: dump first few values of Q/K/V
        LOG_DEBUG("[AttentionStage] Q[0:4] = " << Q[0] << ", " << Q[1] << ", " << Q[2] << ", " << Q[3]);
        LOG_DEBUG("[AttentionStage] K[0:4] = " << K[0] << ", " << K[1] << ", " << K[2] << ", " << K[3]);
        LOG_DEBUG("[AttentionStage] V[0:4] = " << V[0] << ", " << V[1] << ", " << V[2] << ", " << V[3]);

        const int seq_len = params_.seq_len;
        const int kv_len = params_.kv_len;
        const int n_heads = params_.n_heads;
        const int n_kv_heads = params_.n_kv_heads;
        const int head_dim = params_.head_dim;
        const int heads_per_kv = n_heads / n_kv_heads;
        const float scale = params_.scale;

        // Get workspace from context for attention scores
        size_t scores_size = static_cast<size_t>(seq_len) * kv_len * sizeof(float);
        void *workspace = ctx->getWorkspace(scores_size * n_heads);
        float *scores_buf = static_cast<float *>(workspace);

        // Process each query head
        ctx->runFor(0, static_cast<size_t>(n_heads), [=](size_t h_)
                    {
        int h = static_cast<int>(h_);
        int kv_h = h / heads_per_kv;  // GQA: map query head to KV head
        float* scores = scores_buf + h * seq_len * kv_len;
        
        // Q * K^T
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            const float* q_vec = Q + q_pos * n_heads * head_dim + h * head_dim;
            
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                // Apply causal mask
                if (params_.causal && k_pos > q_pos) {
                    scores[q_pos * kv_len + k_pos] = -INFINITY;
                    continue;
                }
                
                const float* k_vec = K + k_pos * n_kv_heads * head_dim + kv_h * head_dim;
                
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[q_pos * kv_len + k_pos] = dot * scale;
            }
        }
        
        // Softmax
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            float* row = scores + q_pos * kv_len;
            
            // Find max
            float max_val = row[0];
            for (int k_pos = 1; k_pos < kv_len; ++k_pos) {
                if (row[k_pos] > max_val) max_val = row[k_pos];
            }
            
            // Exp and sum
            float sum = 0.0f;
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                row[k_pos] = std::exp(row[k_pos] - max_val);
                sum += row[k_pos];
            }
            
            // Normalize
            float inv_sum = 1.0f / sum;
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                row[k_pos] *= inv_sum;
            }
        }
        
        // Scores * V
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            float* out_vec = output + q_pos * n_heads * head_dim + h * head_dim;
            const float* score_row = scores + q_pos * kv_len;
            
            std::memset(out_vec, 0, head_dim * sizeof(float));
            
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                const float* v_vec = V + k_pos * n_kv_heads * head_dim + kv_h * head_dim;
                float s = score_row[k_pos];
                
                for (int d = 0; d < head_dim; ++d) {
                    out_vec[d] += s * v_vec[d];
                }
            }
        } });

        // Debug: dump first few values of output
        LOG_DEBUG("[AttentionStage] output[0:4] = " << output[0] << ", " << output[1] << ", " << output[2] << ", " << output[3]);

        return true;
    }

    size_t AttentionStage::estimatedFlops() const
    {
        // QK: 2 * seq_len * kv_len * head_dim (per head)
        // Softmax: ~5 * seq_len * kv_len (per head)
        // V: 2 * seq_len * kv_len * head_dim (per head)
        size_t qk_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * params_.kv_len;
        size_t v_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads;
    }

    size_t AttentionStage::estimatedMemoryBytes() const
    {
        size_t q_bytes = static_cast<size_t>(params_.seq_len) * params_.n_heads *
                         params_.head_dim * sizeof(float);
        size_t kv_bytes = static_cast<size_t>(params_.kv_len) * params_.n_kv_heads *
                          params_.head_dim * sizeof(float);
        size_t out_bytes = q_bytes;
        return q_bytes + 2 * kv_bytes + out_bytes; // Q + K + V + output
    }

    bool AttentionStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo AttentionStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input tensors
        info.addInput("Q", static_cast<const float *>(params_.Q),
                      params_.seq_len, params_.n_heads * params_.head_dim);
        info.addInput("K", static_cast<const float *>(params_.K),
                      params_.kv_len, params_.n_kv_heads * params_.head_dim);
        info.addInput("V", static_cast<const float *>(params_.V),
                      params_.kv_len, params_.n_kv_heads * params_.head_dim);

        // Output
        info.addOutput("output", static_cast<const float *>(params_.output),
                       params_.seq_len, params_.n_heads * params_.head_dim);

        // Scalar params
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);
        info.addScalar("scale", params_.scale);

        return info;
    }

    // =============================================================================
    // SwiGLUStage Implementation
    // =============================================================================

    SwiGLUStage::SwiGLUStage(Params params) : params_(std::move(params)) {}

    bool SwiGLUStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[SwiGLUStage] Execute: seq_len=" << params_.seq_len
                                                    << " intermediate_dim=" << params_.intermediate_dim);

        if (!ctx)
        {
            LOG_ERROR("[SwiGLUStage] Null device context");
            return false;
        }

        const float *gate = static_cast<const float *>(params_.gate);
        const float *up = static_cast<const float *>(params_.up);
        float *output = static_cast<float *>(params_.output);
        const int seq_len = params_.seq_len;
        const int intermediate_dim = params_.intermediate_dim;

        // SwiGLU: silu(gate) * up
        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t i_)
                    {
        int i = static_cast<int>(i_);
        for (int j = 0; j < intermediate_dim; ++j) {
            int idx = i * intermediate_dim + j;
            float g = gate[idx];
            // SiLU: x * sigmoid(x)
            float silu = g / (1.0f + std::exp(-g));
            output[idx] = silu * up[idx];
        } });
        return true;
    }

    size_t SwiGLUStage::estimatedFlops() const
    {
        // Per element: exp, div, mul, mul (~6 FLOPs)
        return static_cast<size_t>(6) * params_.seq_len * params_.intermediate_dim;
    }

    size_t SwiGLUStage::estimatedMemoryBytes() const
    {
        size_t bytes = static_cast<size_t>(params_.seq_len) * params_.intermediate_dim * sizeof(float);
        return 3 * bytes; // gate + up + output
    }

    bool SwiGLUStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo SwiGLUStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input tensors
        info.addInput("gate", static_cast<const float *>(params_.gate),
                      params_.seq_len, params_.intermediate_dim);
        info.addInput("up", static_cast<const float *>(params_.up),
                      params_.seq_len, params_.intermediate_dim);

        // Output
        info.addOutput("output", static_cast<const float *>(params_.output),
                       params_.seq_len, params_.intermediate_dim);

        // Scalar params
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("intermediate_dim", params_.intermediate_dim);

        return info;
    }

    // =============================================================================
    // ResidualAddStage Implementation (Precision-Aware)
    // =============================================================================

    ResidualAddStage::ResidualAddStage(Params params) : params_(std::move(params)) {}

    bool ResidualAddStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[ResidualAddStage] Execute: num_elements=" << params_.num_elements
                                                              << " precision=" << static_cast<int>(params_.precision));

        if (!ctx)
        {
            LOG_ERROR("[ResidualAddStage] Null device context");
            return false;
        }

        // Dispatch based on precision
        switch (params_.precision)
        {
        case ActivationPrecision::FP32:
            return executeFP32(ctx);
        case ActivationPrecision::BF16:
            return executeBF16(ctx);
        case ActivationPrecision::FP16:
            return executeFP16(ctx);
        case ActivationPrecision::Q8_1:
            return executeQ8_1(ctx);
        default:
            LOG_ERROR("[ResidualAddStage] Unknown precision: " << static_cast<int>(params_.precision));
            return false;
        }
    }

    bool ResidualAddStage::executeFP32(IDeviceContext *ctx)
    {
        const float *input = static_cast<const float *>(params_.input);
        const float *residual = static_cast<const float *>(params_.residual);
        float *output = static_cast<float *>(params_.output);
        const size_t n = params_.num_elements;

        LOG_DEBUG("[ResidualAddStage::FP32] input[0:4]="
                  << input[0] << "," << input[1] << "," << input[2] << "," << input[3]
                  << " residual[0:4]="
                  << residual[0] << "," << residual[1] << "," << residual[2] << "," << residual[3]);

        ctx->runFor(0, n, [=](size_t i)
                    { output[i] = input[i] + residual[i]; });

        LOG_DEBUG("[ResidualAddStage::FP32] output[0:4]="
                  << output[0] << "," << output[1] << "," << output[2] << "," << output[3]);

        return true;
    }

    bool ResidualAddStage::executeBF16(IDeviceContext *ctx)
    {
        const uint16_t *input = static_cast<const uint16_t *>(params_.input);
        const uint16_t *residual = static_cast<const uint16_t *>(params_.residual);
        uint16_t *output = static_cast<uint16_t *>(params_.output);
        const size_t n = params_.num_elements;

        LOG_DEBUG("[ResidualAddStage::BF16] Converting and adding " << n << " elements");

        ctx->runFor(0, n, [=](size_t i)
                    {
            float in_f = simd::bf16_to_fp32(input[i]);
            float res_f = simd::bf16_to_fp32(residual[i]);
            output[i] = simd::fp32_to_bf16(in_f + res_f); });

        return true;
    }

    bool ResidualAddStage::executeFP16(IDeviceContext *ctx)
    {
        const uint16_t *input = static_cast<const uint16_t *>(params_.input);
        const uint16_t *residual = static_cast<const uint16_t *>(params_.residual);
        uint16_t *output = static_cast<uint16_t *>(params_.output);
        const size_t n = params_.num_elements;

        LOG_DEBUG("[ResidualAddStage::FP16] Converting and adding " << n << " elements");

        ctx->runFor(0, n, [=](size_t i)
                    {
            float in_f = simd::fp16_to_fp32(input[i]);
            float res_f = simd::fp16_to_fp32(residual[i]);
            output[i] = simd::fp32_to_fp16(in_f + res_f); });

        return true;
    }

    bool ResidualAddStage::executeQ8_1(IDeviceContext *ctx)
    {
        (void)ctx; // Q8_1 add is synchronous, doesn't use ctx parallelism

        const Q8_1Block *input = static_cast<const Q8_1Block *>(params_.input);
        const Q8_1Block *residual = static_cast<const Q8_1Block *>(params_.residual);
        Q8_1Block *output = static_cast<Q8_1Block *>(params_.output);

        // Q8_1 blocks: 32 elements per block
        // num_elements is the total FP32 element count
        const size_t n = params_.num_elements;

        LOG_DEBUG("[ResidualAddStage::Q8_1] Native Q8_1 addition for " << n << " elements");

        // Use SIMD-optimized Q8_1 addition
        // This performs: dequant(input) + dequant(residual) -> requant(output)
        // All in registers with fused scale computation
        simd::q8_1_add_q8_1(input, residual, output, n);

        LOG_DEBUG("[ResidualAddStage::Q8_1] Completed native Q8_1 addition");

        return true;
    }

    size_t ResidualAddStage::estimatedFlops() const
    {
        // For Q8_1: dequant (1 mul/elem) + add + requant (1 mul/elem) = ~3 ops/elem
        // For FP32/BF16/FP16: 1 add per element
        if (params_.precision == ActivationPrecision::Q8_1)
        {
            return params_.num_elements * 3;
        }
        return params_.num_elements;
    }

    size_t ResidualAddStage::estimatedMemoryBytes() const
    {
        // Memory depends on precision
        size_t bytes_per_element;
        switch (params_.precision)
        {
        case ActivationPrecision::Q8_1:
            // Q8_1Block: 36 bytes per 32 elements = 1.125 bytes/element
            bytes_per_element = 1; // Approximate
            break;
        case ActivationPrecision::BF16:
        case ActivationPrecision::FP16:
            bytes_per_element = 2;
            break;
        case ActivationPrecision::FP32:
        default:
            bytes_per_element = 4;
            break;
        }
        return 3 * params_.num_elements * bytes_per_element; // input + residual + output
    }

    bool ResidualAddStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo ResidualAddStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Determine dtype string based on precision
        const char *dtype = "FP32";
        size_t elem_size = sizeof(float);
        switch (params_.precision)
        {
        case ActivationPrecision::Q8_1:
            dtype = "Q8_1";
            elem_size = sizeof(Q8_1Block);
            break;
        case ActivationPrecision::BF16:
            dtype = "BF16";
            elem_size = 2;
            break;
        case ActivationPrecision::FP16:
            dtype = "FP16";
            elem_size = 2;
            break;
        default:
            break;
        }

        int rows = params_.rows > 0 ? params_.rows : 1;
        int cols = params_.cols > 0 ? params_.cols : static_cast<int>(params_.num_elements);

        // Input tensors
        info.inputs.push_back({"input", params_.input,
                               static_cast<size_t>(rows), static_cast<size_t>(cols),
                               dtype, elem_size});
        info.inputs.push_back({"residual", params_.residual,
                               static_cast<size_t>(rows), static_cast<size_t>(cols),
                               dtype, elem_size});

        // Output
        info.outputs.push_back({"output", params_.output,
                                static_cast<size_t>(rows), static_cast<size_t>(cols),
                                dtype, elem_size});

        // Scalar params
        info.addScalarInt("num_elements", static_cast<int>(params_.num_elements));
        info.addScalarInt("rows", rows);
        info.addScalarInt("cols", cols);
        info.addScalarInt("precision", static_cast<int>(params_.precision));

        return info;
    }

    // =============================================================================
    // AllreduceStage Implementation
    // =============================================================================

    AllreduceStage::AllreduceStage(Params params) : params_(std::move(params)) {}

    bool AllreduceStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;
        LOG_DEBUG("[AllreduceStage] Execute: buffer=" << params_.buffer
                                                      << " count=" << params_.count << " has_comm=" << (params_.mpi_comm != nullptr));
        if (!params_.mpi_comm)
        {
            LOG_ERROR("[AllreduceStage] Null MPI communicator");
            return false;
        }

        MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm);

        LOG_DEBUG("[AllreduceStage] Calling MPI_Allreduce with count=" << params_.count);
        int result = MPI_Allreduce(
            MPI_IN_PLACE,
            params_.buffer,
            static_cast<int>(params_.count),
            MPI_FLOAT,
            MPI_SUM,
            comm);

        LOG_DEBUG("[AllreduceStage] MPI_Allreduce returned " << result);
        return result == MPI_SUCCESS;
    }

    bool AllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // Allreduce is backend-agnostic (works with any device that has MPI support)
        (void)backend;
        return true;
    }

    // =============================================================================
    // MoE Stages Implementation
    // =============================================================================

    MoERouterStage::MoERouterStage(Params params) : params_(std::move(params)) {}

    bool MoERouterStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoERouterStage] Null device context");
            return false;
        }

        // Router is a simple matmul: hidden @ gate_weights
        // This computes logits for each expert
        const float *hidden = static_cast<const float *>(params_.hidden);
        const float *gate_weights = static_cast<const float *>(params_.gate_weights);
        float *logits = params_.router_logits;

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;

        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t t_)
                    {
        int t = static_cast<int>(t_);
        const float* h = hidden + t * d_model;
        float* out = logits + t * num_experts;
        
        for (int e = 0; e < num_experts; ++e) {
            const float* w = gate_weights + e * d_model;
            float dot = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                dot += h[d] * w[d];
            }
            out[e] = dot;
        } });
        return true;
    }

    size_t MoERouterStage::estimatedFlops() const
    {
        // seq_len * d_model * num_experts (dot products)
        return static_cast<size_t>(2) * params_.seq_len * params_.d_model * params_.num_experts;
    }

    bool MoERouterStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // -----------------------------------------------------------------------------

    MoEExpertStage::MoEExpertStage(Params params) : params_(std::move(params)) {}

    bool MoEExpertStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertStage] Null device context");
            return false;
        }

        if (!params_.token_indices || params_.token_indices->empty())
        {
            // No tokens routed to this expert - nothing to do
            return true;
        }

        // This is a placeholder - real implementation would use the actual expert weights
        // For now, we just demonstrate the structure
        LOG_DEBUG("[MoEExpertStage] Processing expert " << params_.expert_id
                                                        << " with " << params_.token_indices->size() << " tokens");

        // In real implementation:
        // 1. Gather tokens from input based on token_indices
        // 2. Apply gate projection
        // 3. Apply up projection
        // 4. SwiGLU activation
        // 5. Apply down projection
        // 6. Scatter results back

        return true;
    }

    std::string MoEExpertStage::name() const
    {
        std::ostringstream oss;
        oss << "MOE_EXPERT_" << params_.expert_id;
        return oss.str();
    }

    size_t MoEExpertStage::estimatedFlops() const
    {
        if (!params_.token_indices)
            return 0;
        size_t num_tokens = params_.token_indices->size();
        // FFN: gate + up + down projections
        // gate: num_tokens * d_model * intermediate_dim
        // up: num_tokens * d_model * intermediate_dim
        // down: num_tokens * intermediate_dim * d_model
        return static_cast<size_t>(6) * num_tokens * params_.d_model * params_.intermediate_dim;
    }

    bool MoEExpertStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // -----------------------------------------------------------------------------

    MoECombineStage::MoECombineStage(Params params) : params_(std::move(params)) {}

    bool MoECombineStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoECombineStage] Null device context");
            return false;
        }

        // Placeholder - combines expert outputs weighted by router scores
        LOG_DEBUG("[MoECombineStage] Combining "
                  << (params_.expert_outputs ? params_.expert_outputs->size() : 0)
                  << " expert outputs");

        // In real implementation:
        // For each token:
        //   output[t] = sum over k experts: weight[t][k] * expert_output[k][t]

        return true;
    }

    bool MoECombineStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // AttentionWithKVCacheStage Implementation
    // =============================================================================

    AttentionWithKVCacheStage::AttentionWithKVCacheStage(Params params)
        : params_(std::move(params)) {}

    AttentionWithKVCacheStage::Mode AttentionWithKVCacheStage::effectiveMode() const
    {
        if (params_.mode != Mode::AUTO)
        {
            return params_.mode;
        }

        // Auto-detect based on seq_len and cache state
        if (params_.batch_size > 1 && params_.sequence_lengths != nullptr)
        {
            return Mode::BATCHED;
        }

        // Single sequence: check if decode (single token) vs prefill
        if (params_.seq_len == 1)
        {
            // Single token with cache -> decode mode
            if (params_.kv_cache != nullptr)
            {
                int cached = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
                if (cached > 0)
                {
                    return Mode::DECODE;
                }
            }
        }

        return Mode::PREFILL;
    }

    MpiAttentionConfig AttentionWithKVCacheStage::buildAttentionConfig() const
    {
        MpiAttentionConfig config;
        config.n_heads = params_.n_heads;
        config.n_kv_heads = params_.n_kv_heads;
        config.head_dim = params_.head_dim;
        config.causal = params_.causal;
        config.window_size = params_.window_size;
        config.precision = ActivationPrecision::FP32; // TODO: Make configurable
        config.mpi_ctx = params_.mpi_ctx;
        config.mpi_strategy = static_cast<MPIStrategy>(params_.mpi_strategy);
        config.verbose_logging = false;

        // Wire workspace buffers if provided
        if (params_.workspace_scores)
        {
            config.workspace_scores = std::shared_ptr<TensorBase>(
                params_.workspace_scores, [](TensorBase *) {}); // Non-owning
        }
        if (params_.workspace_context)
        {
            config.workspace_context = std::shared_ptr<TensorBase>(
                params_.workspace_context, [](TensorBase *) {});
        }
        if (params_.workspace_mask)
        {
            config.workspace_mask = std::shared_ptr<TensorBase>(
                params_.workspace_mask, [](TensorBase *) {});
        }

        return config;
    }

    bool AttentionWithKVCacheStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Null device context");
            return false;
        }

        if (!params_.Q || !params_.K || !params_.V || !params_.output)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Invalid tensors: Q=" << params_.Q
                                                                        << " K=" << params_.K << " V=" << params_.V << " output=" << params_.output);
            return false;
        }

        Mode mode = effectiveMode();
        LOG_DEBUG("[AttentionWithKVCacheStage] Execute mode=" << static_cast<int>(mode)
                                                              << " seq_len=" << params_.seq_len << " batch=" << params_.batch_size
                                                              << " layer=" << params_.layer_idx << " heads=" << params_.n_heads
                                                              << " kv_heads=" << params_.n_kv_heads << " head_dim=" << params_.head_dim);

        switch (mode)
        {
        case Mode::PREFILL:
            return executePrefill(ctx);
        case Mode::DECODE:
            return executeDecode(ctx);
        case Mode::BATCHED:
            return executeBatched(ctx);
        case Mode::AUTO:
            // Should never reach here
            LOG_ERROR("[AttentionWithKVCacheStage] AUTO mode not resolved");
            return false;
        }

        return false;
    }

    bool AttentionWithKVCacheStage::executePrefill(IDeviceContext *ctx)
    {
        (void)ctx; // Device context for future GPU support

        LOG_DEBUG("[AttentionWithKVCacheStage::executePrefill] layer=" << params_.layer_idx
                                                                       << " seq_len=" << params_.seq_len);

        // Step 1: Append new K/V to cache (if cache provided)
        if (params_.kv_cache)
        {
            bool append_ok = params_.kv_cache->append_kv(
                params_.layer_idx, 0, params_.K, params_.V, params_.seq_len);
            if (!append_ok)
            {
                LOG_ERROR("[AttentionWithKVCacheStage] Failed to append K/V to cache");
                return false;
            }
            LOG_DEBUG("[AttentionWithKVCacheStage] Appended " << params_.seq_len
                                                              << " tokens to layer " << params_.layer_idx << " cache");
        }

        // Step 2: Get full K/V from cache (includes just-appended tokens)
        TensorBase *K_full = params_.K;
        TensorBase *V_full = params_.V;
        int kv_len = params_.seq_len;

        if (params_.kv_cache)
        {
            // Use cached K/V (will include all tokens including just-appended)
            K_full = params_.kv_cache->get_k_base(params_.layer_idx, 0);
            V_full = params_.kv_cache->get_v_base(params_.layer_idx, 0);
            kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            LOG_DEBUG("[AttentionWithKVCacheStage] Using cached K/V with " << kv_len << " tokens");
        }

        // Step 3: Build attention config and dispatch to MpiAttentionOrchestrator
        MpiAttentionConfig config = buildAttentionConfig();

        // Dispatch to MPI-aware attention
        bool success = MpiAttentionOrchestrator::compute_mpi(
            params_.Q, K_full, V_full, params_.output,
            config,
            params_.batch_size,
            params_.sequence_lengths);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] MpiAttentionOrchestrator::compute_mpi failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executePrefill] Complete");
        return true;
    }

    bool AttentionWithKVCacheStage::executeDecode(IDeviceContext *ctx)
    {
        (void)ctx;

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] layer=" << params_.layer_idx
                                                                      << " position=" << params_.position_offset);

        if (!params_.kv_cache)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Decode mode requires KV cache");
            return false;
        }

        // Step 1: Append single token K/V to cache
        bool append_ok = params_.kv_cache->append_kv(
            params_.layer_idx, 0, params_.K, params_.V, 1);
        if (!append_ok)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Failed to append decode token to cache");
            return false;
        }

        // Step 2: Get full cached K/V
        TensorBase *K_cached = params_.kv_cache->get_k_base(params_.layer_idx, 0);
        TensorBase *V_cached = params_.kv_cache->get_v_base(params_.layer_idx, 0);
        int kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] Attending to " << kv_len << " cached tokens");

        // Step 3: Dispatch attention (Q is single token, K/V is full cache)
        MpiAttentionConfig config = buildAttentionConfig();

        // Note: For decode, Q has seq_len=1 but K/V has seq_len=kv_len
        // MpiAttentionOrchestrator handles asymmetric Q/KV lengths

        bool success = MpiAttentionOrchestrator::compute_mpi(
            params_.Q, K_cached, V_cached, params_.output,
            config,
            params_.batch_size,
            nullptr); // No padding mask for decode

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] MpiAttentionOrchestrator failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] Complete");
        return true;
    }

    bool AttentionWithKVCacheStage::executeBatched(IDeviceContext *ctx)
    {
        (void)ctx;

        LOG_DEBUG("[AttentionWithKVCacheStage::executeBatched] layer=" << params_.layer_idx
                                                                       << " batch_size=" << params_.batch_size << " seq_len=" << params_.seq_len);

        // For batched mode with different sequence lengths, we need to:
        // 1. Append K/V per sequence to cache
        // 2. Build combined K/V tensors with padding
        // 3. Apply padding-aware attention mask

        if (params_.kv_cache)
        {
            // Append each sequence's K/V to cache
            const int d_kv = params_.n_kv_heads * params_.head_dim;
            for (int b = 0; b < params_.batch_size; ++b)
            {
                int actual_len = params_.sequence_lengths ? (*params_.sequence_lengths)[b] : params_.seq_len;

                // Create view of K/V for this batch
                // Note: This assumes K/V are [batch * seq_len, n_kv_heads * head_dim]
                // TODO: Implement proper batch slicing

                bool append_ok = params_.kv_cache->append_kv(
                    params_.layer_idx, b, params_.K, params_.V, actual_len);
                if (!append_ok)
                {
                    LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] Failed to append batch " << b);
                    return false;
                }
            }
        }

        // Build attention config
        MpiAttentionConfig config = buildAttentionConfig();

        // Dispatch batched attention
        bool success = MpiAttentionOrchestrator::compute_mpi(
            params_.Q, params_.K, params_.V, params_.output,
            config,
            params_.batch_size,
            params_.sequence_lengths);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] MpiAttentionOrchestrator failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executeBatched] Complete");
        return true;
    }

    size_t AttentionWithKVCacheStage::estimatedFlops() const
    {
        // QK: 2 * seq_len * kv_len * head_dim (per head)
        // For prefill: kv_len ≈ seq_len
        // For decode: kv_len = cached_tokens, seq_len = 1
        int estimated_kv_len = params_.seq_len; // Conservative estimate
        if (params_.kv_cache && params_.seq_len == 1)
        {
            estimated_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
        }

        size_t qk_flops = 2ULL * params_.seq_len * estimated_kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * estimated_kv_len;
        size_t v_flops = 2ULL * params_.seq_len * estimated_kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads * params_.batch_size;
    }

    size_t AttentionWithKVCacheStage::estimatedMemoryBytes() const
    {
        // Q + K + V + output (all at FP32)
        size_t qkv_bytes = static_cast<size_t>(params_.batch_size) * params_.seq_len *
                           (params_.n_heads + 2 * params_.n_kv_heads) * params_.head_dim * sizeof(float);
        return qkv_bytes;
    }

    bool AttentionWithKVCacheStage::supportsBackend(ComputeBackendType backend) const
    {
        // MpiAttentionOrchestrator currently only supports CPU
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo AttentionWithKVCacheStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Q/K/V inputs
        if (params_.Q)
        {
            info.addInput("Q", params_.Q->data(),
                          params_.batch_size * params_.seq_len,
                          params_.n_heads * params_.head_dim);
        }

        // Output
        if (params_.output)
        {
            info.addOutput("output", params_.output->data(),
                           params_.batch_size * params_.seq_len,
                           params_.n_heads * params_.head_dim);
        }

        // Scalar params
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);

        return info;
    }

    // =============================================================================
    // KVCacheAppendStage Implementation
    // =============================================================================

    KVCacheAppendStage::KVCacheAppendStage(Params params)
        : params_(std::move(params)) {}

    bool KVCacheAppendStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.kv_cache)
        {
            LOG_ERROR("[KVCacheAppendStage] No KV cache provided");
            return false;
        }

        if (!params_.K || !params_.V)
        {
            LOG_ERROR("[KVCacheAppendStage] Invalid K/V tensors");
            return false;
        }

        int num_tokens = params_.num_tokens;
        if (num_tokens <= 0)
        {
            // Infer from tensor shape
            num_tokens = static_cast<int>(params_.K->shape()[0]);
        }

        LOG_DEBUG("[KVCacheAppendStage] Appending " << num_tokens
                                                    << " tokens to layer " << params_.layer_idx << " seq " << params_.seq_idx);

        bool success = params_.kv_cache->append_kv(
            params_.layer_idx, params_.seq_idx,
            params_.K, params_.V, num_tokens);

        if (!success)
        {
            LOG_ERROR("[KVCacheAppendStage] append_kv failed");
            return false;
        }

        return true;
    }

    // =============================================================================
    // ComputeStageFactory Implementation
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createQuantize(
        const QuantizeStage::Params &params,
        ComputeBackendType target_backend)
    {
        // CPU only for now - GPU quantization would use different kernel
        (void)target_backend;
        return std::make_unique<QuantizeStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGEMM(
        const GEMMStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<GEMMStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUGEMMStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU GEMM not compiled in, falling back to CPU");
            return std::make_unique<GEMMStage>(params);
#endif
        default:
            LOG_ERROR("[ComputeStageFactory] Unknown backend for GEMM");
            return nullptr;
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRMSNorm(
        const RMSNormStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<RMSNormStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPURMSNormStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU RMSNorm not compiled in, using CPU");
            return std::make_unique<RMSNormStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for RMSNorm, using CPU");
            return std::make_unique<RMSNormStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRoPE(
        const RoPEStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<RoPEStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPURoPEStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU RoPE not compiled in, using CPU");
            return std::make_unique<RoPEStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for RoPE, using CPU");
            return std::make_unique<RoPEStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttention(
        const AttentionStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<AttentionStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUAttentionStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU Attention not compiled in, using CPU");
            return std::make_unique<AttentionStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for Attention, using CPU");
            return std::make_unique<AttentionStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createSwiGLU(
        const SwiGLUStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<SwiGLUStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUSwiGLUStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU SwiGLU not compiled in, using CPU");
            return std::make_unique<SwiGLUStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for SwiGLU, using CPU");
            return std::make_unique<SwiGLUStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createResidualAdd(
        const ResidualAddStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<ResidualAddStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUResidualAddStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU ResidualAdd not compiled in, using CPU");
            return std::make_unique<ResidualAddStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for ResidualAdd, using CPU");
            return std::make_unique<ResidualAddStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoERouter(
        const MoERouterStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<MoERouterStage>(params);
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for MoERouter, using CPU");
            return std::make_unique<MoERouterStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpert(
        const MoEExpertStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<MoEExpertStage>(params);
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for MoEExpert, using CPU");
            return std::make_unique<MoEExpertStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoECombine(
        const MoECombineStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<MoECombineStage>(params);
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for MoECombine, using CPU");
            return std::make_unique<MoECombineStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAllreduce(
        const AllreduceStage::Params &params,
        ComputeBackendType /* target_backend */)
    {
        // Allreduce is backend-agnostic (uses MPI directly)
        return std::make_unique<AllreduceStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttentionWithKVCache(
        const AttentionWithKVCacheStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<AttentionWithKVCacheStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            // TODO: GPU attention implementation
            LOG_WARN("[ComputeStageFactory] GPU AttentionWithKVCache not yet implemented, using CPU");
            return std::make_unique<AttentionWithKVCacheStage>(params);
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for AttentionWithKVCache, using CPU");
            return std::make_unique<AttentionWithKVCacheStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createKVCacheAppend(
        const KVCacheAppendStage::Params &params,
        ComputeBackendType /* target_backend */)
    {
        // KV cache append is backend-agnostic (pure memory operations)
        return std::make_unique<KVCacheAppendStage>(params);
    }

    // =============================================================================
    // GPU Compute Stage Implementations
    // =============================================================================

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

#ifdef HAVE_CUDA
#include "../backends/cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "../backends/rocm/ROCmBackend.h"
#endif

    // Helper to get GPU device ID from context
    static int getGPUDeviceId(IDeviceContext *ctx)
    {
        auto *gpu_ctx = dynamic_cast<IGPUDeviceContext *>(ctx);
        return gpu_ctx ? gpu_ctx->gpuDeviceId() : 0;
    }

    // -----------------------------------------------------------------------------
    // GPUGEMMStage
    // -----------------------------------------------------------------------------

    GPUGEMMStage::GPUGEMMStage(GEMMStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUGEMMStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUGEMMStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUGEMMStage] Execute GEMM on GPU " << device_id
                                                        << ": " << params_.m << "x" << params_.n << "x" << params_.k);

        // TODO: Delegate to backend-specific GEMM
        // For quantized weights, use backend->gemmIQ4NL()
        // For FP32/FP16, use cuBLAS/rocBLAS via backend

        // Placeholder: mark as successful (actual kernels TBD)
        return true;
    }

    size_t GPUGEMMStage::estimatedFlops() const
    {
        return static_cast<size_t>(2) * params_.m * params_.n * params_.k;
    }

    size_t GPUGEMMStage::estimatedMemoryBytes() const
    {
        size_t a_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);
        size_t b_bytes = static_cast<size_t>(params_.k) * params_.n * sizeof(float);
        size_t c_bytes = static_cast<size_t>(params_.m) * params_.n * sizeof(float);
        return a_bytes + b_bytes + c_bytes;
    }

    bool GPUGEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPURMSNormStage
    // -----------------------------------------------------------------------------

    GPURMSNormStage::GPURMSNormStage(RMSNormStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPURMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPURMSNormStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPURMSNormStage] Execute RMSNorm on GPU " << device_id);

        // TODO: Launch custom RMSNorm kernel with warp-level reduction
        // Kernel signature: rmsnorm_kernel<<<blocks, threads>>>(input, gamma, seq_len, hidden_dim, eps)

        return true;
    }

    size_t GPURMSNormStage::estimatedFlops() const
    {
        return static_cast<size_t>(4) * params_.seq_len * params_.hidden_dim;
    }

    size_t GPURMSNormStage::estimatedMemoryBytes() const
    {
        size_t input_bytes = static_cast<size_t>(params_.seq_len) * params_.hidden_dim * sizeof(float);
        size_t gamma_bytes = static_cast<size_t>(params_.hidden_dim) * sizeof(float);
        return 2 * input_bytes + gamma_bytes;
    }

    bool GPURMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPUSwiGLUStage
    // -----------------------------------------------------------------------------

    GPUSwiGLUStage::GPUSwiGLUStage(SwiGLUStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUSwiGLUStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUSwiGLUStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUSwiGLUStage] Execute SwiGLU on GPU " << device_id);

        // TODO: Launch fused SwiGLU kernel
        // output[i] = silu(gate[i]) * up[i]
        // where silu(x) = x * sigmoid(x)

        return true;
    }

    size_t GPUSwiGLUStage::estimatedFlops() const
    {
        // silu: ~5 ops, mul: 1 op = 6 per element
        return static_cast<size_t>(6) * params_.seq_len * params_.intermediate_dim;
    }

    size_t GPUSwiGLUStage::estimatedMemoryBytes() const
    {
        size_t elem_bytes = static_cast<size_t>(params_.seq_len) * params_.intermediate_dim * sizeof(float);
        return 3 * elem_bytes; // gate + up + output
    }

    bool GPUSwiGLUStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPUResidualAddStage
    // -----------------------------------------------------------------------------

    GPUResidualAddStage::GPUResidualAddStage(ResidualAddStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUResidualAddStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUResidualAddStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUResidualAddStage] Execute ResidualAdd on GPU " << device_id);

        // TODO: Launch simple element-wise addition kernel
        // output[i] = input[i] + residual[i]

        return true;
    }

    size_t GPUResidualAddStage::estimatedFlops() const
    {
        return params_.num_elements; // One add per element
    }

    size_t GPUResidualAddStage::estimatedMemoryBytes() const
    {
        return 3 * params_.num_elements * sizeof(float); // input + residual + output
    }

    bool GPUResidualAddStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPURoPEStage
    // -----------------------------------------------------------------------------

    GPURoPEStage::GPURoPEStage(RoPEStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPURoPEStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPURoPEStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPURoPEStage] Execute RoPE on GPU " << device_id);

        // TODO: Launch RoPE kernel
        // Apply rotary embeddings: cos/sin precomputed per position/dimension

        return true;
    }

    size_t GPURoPEStage::estimatedFlops() const
    {
        return static_cast<size_t>(10) * params_.seq_len * params_.n_heads * (params_.head_dim / 2);
    }

    size_t GPURoPEStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(2) * params_.seq_len * params_.n_heads *
               params_.head_dim * sizeof(float);
    }

    bool GPURoPEStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPUAttentionStage
    // -----------------------------------------------------------------------------

    GPUAttentionStage::GPUAttentionStage(AttentionStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUAttentionStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUAttentionStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUAttentionStage] Execute Attention on GPU " << device_id
                                                                  << " seq=" << params_.seq_len << " kv=" << params_.kv_len);

        // TODO: Implement Flash Attention or standard attention
        // 1. Q * K^T with scaling
        // 2. Causal mask application
        // 3. Online softmax
        // 4. Attention @ V

        return true;
    }

    size_t GPUAttentionStage::estimatedFlops() const
    {
        size_t qk_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * params_.kv_len;
        size_t v_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads;
    }

    size_t GPUAttentionStage::estimatedMemoryBytes() const
    {
        size_t qkv_bytes = static_cast<size_t>(params_.seq_len + 2 * params_.kv_len) *
                           params_.n_heads * params_.head_dim * sizeof(float);
        size_t scores_bytes = static_cast<size_t>(params_.seq_len) * params_.kv_len *
                              params_.n_heads * sizeof(float);
        return qkv_bytes + scores_bytes;
    }

    bool GPUAttentionStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

#endif // HAVE_CUDA || HAVE_ROCM

} // namespace llaminar2

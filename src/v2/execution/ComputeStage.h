/**
 * @file ComputeStage.h
 * @brief Compute stage abstraction for device-agnostic kernel dispatch
 * @author David Sanftenberg
 * @date December 2025
 *
 * ComputeStage represents a single parallelizable operation that can execute
 * on any device (CPU, GPU). Stages are the unit of work for layer-level
 * parallelism and enable clean separation of serial setup from parallel compute.
 *
 * Key benefits:
 * 1. Device-agnostic interface - same API for CPU and GPU kernels
 * 2. Composable - build compute graphs from atomic operations
 * 3. Introspectable - stages know their FLOP counts, memory needs
 * 4. MoE-ready - expert FFN stages for parallel expert execution
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "DeviceContext.h"
#include "../pipelines/PipelineConfig.h" // For ActivationPrecision
#include "../tensors/BlockStructures.h"  // For Q8_1Block in StageDumpInfo

namespace llaminar2
{

    // Forward declarations (needed for StageDumpInfo before full TensorBase definition)
    class TensorBase;

    /**
     * @brief Detailed dump info for stage debugging
     *
     * Contains all input, output, and parameter buffers needed to fully
     * reproduce and debug a stage execution. Used by StageDumper for
     * first-class debugging support.
     */
    struct StageDumpInfo
    {
        // Input buffers (read during execute)
        struct InputBuffer
        {
            const char *name = nullptr; ///< Buffer name (e.g., "A", "input", "gate")
            const void *data = nullptr; ///< Pointer to data
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";          ///< "FP32", "Q8_1", "IQ4_NL", etc.
            size_t element_size = sizeof(float); ///< Bytes per element (or block)
        };

        // Output buffers (written during execute)
        struct OutputBuffer
        {
            const char *name = nullptr; ///< Buffer name (e.g., "C", "output")
            const void *data = nullptr; ///< Pointer to data
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";
            size_t element_size = sizeof(float);
        };

        // Weight/parameter buffers (read-only during execute)
        struct WeightBuffer
        {
            const char *name = nullptr;         ///< Weight name (e.g., "B", "gamma", "wq")
            const TensorBase *tensor = nullptr; ///< Original quantized tensor
            const void *raw_data = nullptr;     ///< Raw quantized data
            size_t raw_size = 0;                ///< Raw data size in bytes
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = nullptr; ///< Quantization type
        };

        // Scalar parameters
        struct ScalarParam
        {
            const char *name = nullptr;
            double value = 0;
            const char *dtype = "float"; ///< "int", "float", "bool"
        };

        std::vector<InputBuffer> inputs;
        std::vector<OutputBuffer> outputs;
        std::vector<WeightBuffer> weights;
        std::vector<ScalarParam> scalars;

        // Convenience methods for building dump info
        StageDumpInfo &addInput(const char *name, const float *data, size_t rows, size_t cols)
        {
            inputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        StageDumpInfo &addInputQ8_1(const char *name, const void *data, size_t rows, size_t cols)
        {
            // For Q8_1, cols should be in blocks (cols/32 blocks per row)
            inputs.push_back({name, data, rows, cols, "Q8_1", sizeof(Q8_1Block)});
            return *this;
        }

        StageDumpInfo &addOutput(const char *name, const float *data, size_t rows, size_t cols)
        {
            outputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        /**
         * @brief Add weight tensor with explicit dimensions (avoids header dependency)
         */
        StageDumpInfo &addWeight(const char *name, const TensorBase *tensor,
                                 size_t rows, size_t cols, const char *dtype)
        {
            weights.push_back({name, tensor, nullptr, 0, rows, cols, dtype});
            return *this;
        }

        /**
         * @brief Add weight tensor (implementation in ComputeStage.cpp to avoid header deps)
         */
        StageDumpInfo &addWeight(const char *name, const TensorBase *tensor);

        StageDumpInfo &addScalar(const char *name, double value, const char *dtype = "float")
        {
            scalars.push_back({name, value, dtype});
            return *this;
        }

        StageDumpInfo &addScalarInt(const char *name, int value)
        {
            scalars.push_back({name, static_cast<double>(value), "int"});
            return *this;
        }

        StageDumpInfo &addScalarBool(const char *name, bool value)
        {
            scalars.push_back({name, value ? 1.0 : 0.0, "bool"});
            return *this;
        }
    };

    // Forward declarations (TensorBase already declared above)
    class IKVCache;
    class IUnifiedKVCache;
    class MPIContext;
    struct MpiAttentionConfig;

    /**
     * @brief Types of compute operations
     */
    enum class ComputeStageType
    {
        // Matrix operations
        GEMM,           ///< General matrix multiplication
        GEMM_BIAS,      ///< GEMM with bias addition
        GEMM_FUSED_QKV, ///< Fused Q/K/V projection

        // Normalization
        RMS_NORM,   ///< RMS normalization
        LAYER_NORM, ///< Layer normalization (future)

        // Activations
        SWIGLU, ///< SwiGLU activation (FFN)
        GELU,   ///< GELU activation (future)
        SILU,   ///< SiLU activation

        // Attention
        ROPE,              ///< Rotary position encoding
        ATTENTION,         ///< Full attention (Q*K^T, softmax, *V)
        ATTENTION_QK,      ///< Q*K^T only
        ATTENTION_SOFTMAX, ///< Softmax only
        ATTENTION_V,       ///< Softmax @ V only

        // Element-wise
        ADD_RESIDUAL, ///< Element-wise addition (residual connection)
        SCALE,        ///< Element-wise scaling

        // MoE specific
        MOE_ROUTER,     ///< Expert routing (softmax + top-k)
        MOE_EXPERT_FFN, ///< Single expert FFN execution
        MOE_COMBINE,    ///< Combine expert outputs with weights

        // Collective
        ALLREDUCE, ///< MPI allreduce (sum)
        ALLGATHER, ///< MPI allgather

        // Utility
        COPY,       ///< Memory copy
        QUANTIZE,   ///< Quantization (FP32 → Q8_1, etc.)
        DEQUANTIZE, ///< Dequantization
    };

    /**
     * @brief Convert stage type to string for logging
     */
    const char *computeStageTypeName(ComputeStageType type);

    /**
     * @brief Base class for all compute stages
     *
     * Derived classes implement device-specific kernels while maintaining
     * a common interface for orchestration.
     */
    class IComputeStage
    {
    public:
        virtual ~IComputeStage() = default;

        // =========================================================================
        // Execution
        // =========================================================================

        /**
         * @brief Execute this stage on the given device context
         *
         * The stage must be compatible with the device type (CPU stages on CPU, etc.)
         * GPU stages may enqueue work asynchronously - call ctx->synchronize() if
         * you need completion.
         *
         * @param ctx Device context to execute on
         * @return true on success, false on error
         */
        virtual bool execute(IDeviceContext *ctx) = 0;

        // =========================================================================
        // Introspection
        // =========================================================================

        /**
         * @brief Get the operation type
         */
        virtual ComputeStageType type() const = 0;

        /**
         * @brief Human-readable name (for profiling/logging)
         */
        virtual std::string name() const
        {
            return computeStageTypeName(type());
        }

        /**
         * @brief Estimated floating-point operations
         *
         * Used for load balancing and performance modeling.
         * Returns 0 if not applicable (e.g., memory ops).
         */
        virtual size_t estimatedFlops() const { return 0; }

        /**
         * @brief Estimated memory traffic in bytes
         *
         * Includes reads and writes. Used for bandwidth estimation.
         */
        virtual size_t estimatedMemoryBytes() const { return 0; }

        /**
         * @brief Does this stage require MPI synchronization after?
         *
         * True for stages like row-parallel GEMM that need allreduce.
         */
        virtual bool requiresAllreduce() const { return false; }

        /**
         * @brief Can this stage execute on the given backend?
         */
        virtual bool supportsBackend(ComputeBackendType backend) const = 0;

        /**
         * @brief Get detailed dump info for debugging/snapshotting
         *
         * Returns comprehensive information about all inputs, outputs, weights,
         * and parameters needed to reproduce this stage execution.
         * Used by StageDumper for first-class debugging support.
         *
         * @return Dump info with all buffers and parameters
         */
        virtual StageDumpInfo getDumpInfo() const
        {
            return StageDumpInfo{}; // Default: empty info
        }
    };

    // =============================================================================
    // Concrete Stage Implementations (CPU)
    // =============================================================================

    /**
     * @brief Quantize FP32 activations to Q8_1 for shared use across multiple GEMMs
     *
     * This stage enables the same pattern as FusedGEMM:
     * 1. Quantize activations once (QuantizeStage)
     * 2. Execute multiple GEMMs with the shared Q8_1 buffer (GEMMStage)
     *
     * This avoids redundant quantization when the same activations feed multiple
     * projections (e.g., Q/K/V attention projections, gate/up FFN projections).
     */
    class QuantizeStage : public IComputeStage
    {
    public:
        struct Params
        {
            const float *input; ///< FP32 input [m × k]
            void *output;       ///< Q8_1 output buffer (sized via get_quantized_buffer_size)
            int m;              ///< Number of rows (sequence length)
            int k;              ///< Number of columns (hidden dimension)
        };

        explicit QuantizeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::QUANTIZE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

        /**
         * @brief Calculate required buffer size for Q8_1 output
         */
        static size_t get_quantized_buffer_size(int m, int k);

    private:
        Params params_;
    };

    /**
     * @brief GEMM stage: C = alpha * A * B + beta * C
     *
     * Supports two input modes (matching FusedGEMM functionality):
     * 1. **FP32 mode**: A is FP32 activations → quantize internally then GEMM
     * 2. **Q8_1 mode**: A_q8_1 is pre-quantized → use directly (no redundant quantization)
     *
     * Q8_1 mode enables efficient multi-GEMM patterns:
     * @code
     * // Quantize once, use for Q/K/V projections
     * auto quant_stage = QuantizeStage({normalized, q8_buffer, m, k});
     * auto q_gemm = GEMMStage({.A_q8_1 = q8_buffer, .B = wq, ...});
     * auto k_gemm = GEMMStage({.A_q8_1 = q8_buffer, .B = wk, ...});
     * auto v_gemm = GEMMStage({.A_q8_1 = q8_buffer, .B = wv, ...});
     * @endcode
     */
    class GEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Primary fields (backward compatible order for positional init)
            const void *A = nullptr;       ///< FP32 activation matrix (m × k)
            const TensorBase *B = nullptr; ///< Weight tensor (k × n, may be quantized)
            void *C = nullptr;             ///< Output matrix (m × n)
            int m = 0, n = 0, k = 0;       ///< Matrix dimensions
            float alpha = 1.0f;            ///< Scale factor for A*B
            float beta = 0.0f;             ///< Scale factor for existing C
            bool transpose_B = false;      ///< Whether B is transposed (n × k)

            // Extended fields (use designated initializers)
            const void *A_q8_1 = nullptr;      ///< Pre-quantized Q8_1 activations (from QuantizeStage)
            const float *bias = nullptr;       ///< Fused bias [n] (nullptr if none)
            const float *gate_input = nullptr; ///< SwiGLU gate input [m, n] (nullptr if not SwiGLU)
            bool do_swiglu = false;            ///< Whether to apply SwiGLU fusion

            // MPI context for tensor-parallel execution (optional)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit GEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

    /**
     * @brief RMS normalization stage
     */
    class RMSNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *input;  ///< Input tensor (read-only)
            void *output;       ///< Output tensor (can be same as input for in-place)
            const float *gamma; ///< Scale weights
            int seq_len;        ///< Sequence length
            int hidden_dim;     ///< Hidden dimension
            float eps;          ///< Epsilon for numerical stability
        };

        explicit RMSNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

    /**
     * @brief Rotary position encoding stage
     */
    class RoPEStage : public IComputeStage
    {
    public:
        struct Params
        {
            void *tensor;     ///< Q or K tensor to apply RoPE
            int seq_len;      ///< Sequence length
            int n_heads;      ///< Number of heads
            int head_dim;     ///< Dimension per head
            int pos_offset;   ///< Position offset (for KV cache)
            float theta_base; ///< RoPE theta base (default 10000.0)
        };

        explicit RoPEStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

    /**
     * @brief Full attention stage (Q*K^T, softmax, *V)
     *
     * @deprecated Use AttentionWithKVCacheStage for production. This naive
     * implementation is kept for testing and simple use cases.
     */
    class AttentionStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *Q;  ///< Query tensor [seq_len, n_heads * head_dim]
            const void *K;  ///< Key tensor [kv_len, n_kv_heads * head_dim]
            const void *V;  ///< Value tensor [kv_len, n_kv_heads * head_dim]
            void *output;   ///< Output tensor [seq_len, n_heads * head_dim]
            int seq_len;    ///< Query sequence length
            int kv_len;     ///< Key/value sequence length
            int n_heads;    ///< Number of query heads
            int n_kv_heads; ///< Number of KV heads (GQA)
            int head_dim;   ///< Dimension per head
            bool causal;    ///< Apply causal mask
            float scale;    ///< Attention scale (1/sqrt(head_dim))
        };

        explicit AttentionStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

    // Forward declarations for attention infrastructure
    class IUnifiedKVCache;
    class MPIContext;
    class MpiAttentionConfig;

    /**
     * @brief Production attention stage with KV cache and MPI support
     *
     * This is the full-featured attention implementation that integrates with:
     * - **KV Cache**: Automatic append during prefill, retrieval during decode
     * - **MPI Parallelism**: Tensor-parallel attention via MpiAttentionOrchestrator
     * - **Batched Execution**: Multiple sequences with padding masks
     * - **Asymmetric Lengths**: Different Q and KV lengths (prefill vs decode)
     * - **GQA/MHA/MQA**: All attention variants supported
     *
     * Execution Modes:
     * 1. **Prefill**: Process all tokens, append to KV cache, full attention
     * 2. **Decode**: Single token query, use cached K/V, efficient O(n) per token
     * 3. **Batched**: Multiple sequences, padding-aware masking
     *
     * Usage in LayerExecutor:
     * @code
     * AttentionWithKVCacheStage::Params params{
     *     .Q = buffers.Q.get(),
     *     .K = buffers.K.get(),
     *     .V = buffers.V.get(),
     *     .output = buffers.attn_output.get(),
     *     .kv_cache = kv_cache_.get(),
     *     .layer_idx = layer_idx,
     *     // ... configuration
     * };
     * graph.addNode("attention", ComputeStageFactory::createAttentionWithKVCache(params), device);
     * @endcode
     *
     * MoE Compatibility:
     * For MoE models, attention is typically shared across experts (not expert-specific).
     * The KV cache is shared, and this stage can be used directly in MoE pipelines.
     * Expert parallelism affects FFN routing, not attention computation.
     *
     * @see MpiAttentionOrchestrator for the underlying attention computation
     * @see KVCacheAppendStage for explicit cache operations (advanced pipelining)
     */
    class AttentionWithKVCacheStage : public IComputeStage
    {
    public:
        /**
         * @brief Execution mode for attention
         */
        enum class Mode
        {
            AUTO,    ///< Detect from seq_len and KV cache state
            PREFILL, ///< Process multiple tokens, populate cache
            DECODE,  ///< Single token, use cached K/V
            BATCHED  ///< Multiple sequences with different lengths
        };

        struct Params
        {
            // Input/output tensors (already projected)
            TensorBase *Q = nullptr;      ///< Query [batch_size * seq_len, n_heads * head_dim]
            TensorBase *K = nullptr;      ///< Key [batch_size * seq_len, n_kv_heads * head_dim]
            TensorBase *V = nullptr;      ///< Value [batch_size * seq_len, n_kv_heads * head_dim]
            TensorBase *output = nullptr; ///< Output [batch_size * seq_len, n_heads * head_dim]

            // KV cache integration
            IUnifiedKVCache *kv_cache = nullptr; ///< KV cache (nullptr = no caching)
            int layer_idx = 0;                   ///< Layer index for cache access

            // Execution mode
            Mode mode = Mode::AUTO;

            // Attention configuration
            int batch_size = 1;
            int seq_len = 0;      ///< Query sequence length
            int n_heads = 0;      ///< Number of query heads
            int n_kv_heads = 0;   ///< Number of KV heads (GQA: n_kv_heads < n_heads)
            int head_dim = 0;     ///< Dimension per head
            bool causal = true;   ///< Apply causal mask
            int window_size = -1; ///< Sliding window attention (-1 = full)

            // MPI configuration (for tensor-parallel execution)
            std::shared_ptr<MPIContext> mpi_ctx;
            int mpi_strategy = 0; ///< MPIStrategy cast to int (avoid header dep)

            // Workspace buffers (pre-allocated by pipeline for zero hot-path allocation)
            // If nullptr, stage will allocate internally (slower)
            TensorBase *workspace_scores = nullptr;  ///< [n_heads * max_seq, max_seq]
            TensorBase *workspace_context = nullptr; ///< [max_seq, head_dim]
            TensorBase *workspace_mask = nullptr;    ///< [max_seq, max_seq]

            // Per-sequence lengths for batched attention (nullptr = all same length)
            const std::vector<int> *sequence_lengths = nullptr;

            // Position offset for decode mode (current position in sequence)
            int position_offset = 0;
        };

        explicit AttentionWithKVCacheStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

        /**
         * @brief Get the actual execution mode that will be used
         *
         * If mode is AUTO, this detects based on seq_len and cache state.
         */
        Mode effectiveMode() const;

    private:
        Params params_;

        /// Prefill: Process all tokens, populate KV cache
        bool executePrefill(IDeviceContext *ctx);

        /// Decode: Single token against cached K/V
        bool executeDecode(IDeviceContext *ctx);

        /// Batched: Multiple sequences with padding
        bool executeBatched(IDeviceContext *ctx);

        /// Build MpiAttentionConfig from params
        MpiAttentionConfig buildAttentionConfig() const;
    };

    /**
     * @brief Explicit KV cache append stage
     *
     * Separates cache operations from attention computation, enabling:
     * - Pipelined execution: Append on one device while attending on another
     * - Explicit control: Manual cache management for advanced use cases
     * - Cross-device caches: Cache on GPU while computing on CPU
     *
     * For most use cases, prefer AttentionWithKVCacheStage which handles
     * cache operations internally. Use this stage when you need fine-grained
     * control over cache timing.
     *
     * @see AttentionWithKVCacheStage for integrated attention + cache
     */
    class KVCacheAppendStage : public IComputeStage
    {
    public:
        struct Params
        {
            const TensorBase *K = nullptr; ///< Key to append [seq_len, n_kv_heads * head_dim]
            const TensorBase *V = nullptr; ///< Value to append [seq_len, n_kv_heads * head_dim]
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int seq_idx = 0;    ///< Sequence index (for batched caches)
            int num_tokens = 0; ///< Number of tokens to append (0 = full tensor)
        };

        explicit KVCacheAppendStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::COPY; } // Cache ops are data movement
        bool supportsBackend(ComputeBackendType backend) const override { return true; }

    private:
        Params params_;
    };

    /**
     * @brief SwiGLU activation stage
     */
    class SwiGLUStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *gate; ///< Gate tensor [seq_len, intermediate_dim]
            const void *up;   ///< Up tensor [seq_len, intermediate_dim]
            void *output;     ///< Output tensor [seq_len, intermediate_dim]
            int seq_len;
            int intermediate_dim;
        };

        explicit SwiGLUStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

    /**
     * @brief Residual addition stage: output = input + residual
     *
     * Precision-aware implementation:
     * - FP32: Simple float addition
     * - BF16/FP16: Convert to FP32, add, convert back
     * - Q8_1: Native block addition via simd::q8_1_add_q8_1()
     *
     * The Q8_1 path performs fused dequant-add-requant in registers,
     * avoiding memory traffic for intermediate FP32 values.
     */
    class ResidualAddStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *input;                                         ///< Input tensor data (projection output)
            const void *residual;                                      ///< Residual tensor data (previous hidden state)
            void *output;                                              ///< Output tensor data (can be same as residual for in-place)
            size_t num_elements;                                       ///< Total elements (rows * cols)
            int rows = 0;                                              ///< Number of rows (seq_len, for snapshot)
            int cols = 0;                                              ///< Number of columns (d_model, for snapshot)
            ActivationPrecision precision = ActivationPrecision::FP32; ///< Activation precision
        };

        explicit ResidualAddStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;

        // Precision-specific implementations
        bool executeFP32(IDeviceContext *ctx);
        bool executeBF16(IDeviceContext *ctx);
        bool executeFP16(IDeviceContext *ctx);
        bool executeQ8_1(IDeviceContext *ctx);
    };

    /**
     * @brief MPI Allreduce stage
     */
    class AllreduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            void *buffer;   ///< Buffer to allreduce (in-place)
            size_t count;   ///< Number of elements
            void *mpi_comm; ///< MPI communicator (cast to MPI_Comm)
        };

        explicit AllreduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    // =============================================================================
    // MoE Stages
    // =============================================================================

    /**
     * @brief MoE router stage: compute expert selection
     */
    class MoERouterStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *hidden;       ///< Hidden states [seq_len, d_model]
            const void *gate_weights; ///< Router weights [d_model, num_experts]
            float *router_logits;     ///< Output: router logits [seq_len, num_experts]
            int seq_len;
            int d_model;
            int num_experts;
        };

        explicit MoERouterStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief Single expert FFN execution
     *
     * This stage handles tokens routed to one specific expert.
     * Multiple MoEExpertStages can execute in parallel on different devices.
     */
    class MoEExpertStage : public IComputeStage
    {
    public:
        struct Params
        {
            int expert_id;                         ///< Which expert this is
            const void *input;                     ///< Input tokens for this expert
            void *output;                          ///< Output buffer
            const TensorBase *gate_w;              ///< Expert gate weights
            const TensorBase *up_w;                ///< Expert up weights
            const TensorBase *down_w;              ///< Expert down weights
            const std::vector<int> *token_indices; ///< Which tokens to process
            int d_model;
            int intermediate_dim;
        };

        explicit MoEExpertStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override;
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief Combine expert outputs with router weights
     */
    class MoECombineStage : public IComputeStage
    {
    public:
        struct Params
        {
            const std::vector<const void *> *expert_outputs; ///< Outputs from each expert
            const std::vector<float> *expert_weights;        ///< Router weights per token-expert
            const std::vector<int> *token_expert_map;        ///< Which experts each token used
            void *output;                                    ///< Combined output [seq_len, d_model]
            int seq_len;
            int d_model;
            int top_k; ///< Experts per token
        };

        explicit MoECombineStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_COMBINE; }
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    // =============================================================================
    // Stage Factory
    // =============================================================================

    /**
     * @brief Factory for creating device-appropriate compute stages
     */
    class ComputeStageFactory
    {
    public:
        /**
         * @brief Create a quantization stage (FP32 → Q8_1)
         *
         * Used to pre-quantize activations for reuse across multiple GEMMs.
         * The output buffer can then be passed to GEMMStage via A_q8_1.
         */
        static std::unique_ptr<IComputeStage> createQuantize(
            const QuantizeStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a GEMM stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createGEMM(
            const GEMMStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create an RMSNorm stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createRMSNorm(
            const RMSNormStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a RoPE stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createRoPE(
            const RoPEStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create an attention stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createAttention(
            const AttentionStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a SwiGLU stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createSwiGLU(
            const SwiGLUStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a residual add stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createResidualAdd(
            const ResidualAddStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a MoE router stage for expert selection
         */
        static std::unique_ptr<IComputeStage> createMoERouter(
            const MoERouterStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create an expert FFN stage for MoE
         */
        static std::unique_ptr<IComputeStage> createMoEExpert(
            const MoEExpertStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a MoE combine stage for weighted expert output combination
         */
        static std::unique_ptr<IComputeStage> createMoECombine(
            const MoECombineStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create an Allreduce stage for MPI collective sum
         *
         * Used after row-parallel GEMM to combine partial results across ranks.
         */
        static std::unique_ptr<IComputeStage> createAllreduce(
            const AllreduceStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a production attention stage with KV cache and MPI support
         *
         * This is the recommended factory for attention in production pipelines.
         * For simple testing, use createAttention() instead.
         *
         * @param params Attention parameters including KV cache and MPI config
         * @param target_backend Target compute backend
         * @return AttentionWithKVCacheStage instance
         */
        static std::unique_ptr<IComputeStage> createAttentionWithKVCache(
            const AttentionWithKVCacheStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a KV cache append stage for explicit cache management
         *
         * For advanced use cases where cache operations need to be pipelined
         * separately from attention computation.
         */
        static std::unique_ptr<IComputeStage> createKVCacheAppend(
            const KVCacheAppendStage::Params &params,
            ComputeBackendType target_backend);
    };

    // =============================================================================
    // GPU Compute Stages (CUDA + ROCm)
    // =============================================================================

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

    /**
     * @brief GPU GEMM stage using IBackend
     *
     * Delegates to CUDABackend/ROCmBackend for actual computation.
     * Supports quantized formats via backend's gemmIQ4NL or falls back to cuBLAS/rocBLAS.
     */
    class GPUGEMMStage : public IComputeStage
    {
    public:
        explicit GPUGEMMStage(GEMMStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        GEMMStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU RMSNorm stage
     *
     * Custom CUDA/HIP kernel with warp-level reduction.
     */
    class GPURMSNormStage : public IComputeStage
    {
    public:
        explicit GPURMSNormStage(RMSNormStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        RMSNormStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU SwiGLU stage
     *
     * Fused silu(gate) * up kernel.
     */
    class GPUSwiGLUStage : public IComputeStage
    {
    public:
        explicit GPUSwiGLUStage(SwiGLUStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        SwiGLUStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU Residual Add stage
     *
     * Simple element-wise addition kernel.
     */
    class GPUResidualAddStage : public IComputeStage
    {
    public:
        explicit GPUResidualAddStage(ResidualAddStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        ResidualAddStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU RoPE stage
     *
     * Rotary position embedding kernel.
     */
    class GPURoPEStage : public IComputeStage
    {
    public:
        explicit GPURoPEStage(RoPEStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        RoPEStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU Attention stage
     *
     * Flash attention or standard attention implementation.
     */
    class GPUAttentionStage : public IComputeStage
    {
    public:
        explicit GPUAttentionStage(AttentionStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        AttentionStage::Params params_;
        ComputeBackendType backend_;
    };

#endif // HAVE_CUDA || HAVE_ROCM

} // namespace llaminar2

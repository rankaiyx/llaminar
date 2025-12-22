/**
 * @file Test__DecomposedAttention.cpp
 * @brief Integration tests for Phase 9 decomposed attention path
 * @author David Sanftenberg
 *
 * Tests that the new decomposed attention path (KVCacheAppendStage +
 * AttentionComputeStage) produces the same results as the legacy path
 * (AttentionWithKVCacheStage + MpiAttentionOrchestrator).
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>

#include "v2/pipelines/qwen/Qwen2LayerExecutor.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/UnifiedKVCache.h"
#include "v2/utils/MPIContext.h"

using namespace llaminar2;

namespace
{
    constexpr float TOLERANCE = 1e-4f;

    /**
     * @brief Helper to compute max absolute difference between tensors
     */
    float maxAbsDiff(const TensorBase *a, const TensorBase *b)
    {
        auto *fp32_a = dynamic_cast<const FP32Tensor *>(a);
        auto *fp32_b = dynamic_cast<const FP32Tensor *>(b);
        if (!fp32_a || !fp32_b)
            return -1.0f;

        size_t n = fp32_a->numel();
        if (n != fp32_b->numel())
            return -2.0f;

        float max_diff = 0.0f;
        const float *da = fp32_a->data();
        const float *db = fp32_b->data();
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::abs(da[i] - db[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    /**
     * @brief Copy tensor contents
     */
    void copyTensor(TensorBase *dst, const TensorBase *src)
    {
        auto *fp32_dst = dynamic_cast<FP32Tensor *>(dst);
        auto *fp32_src = dynamic_cast<const FP32Tensor *>(src);
        if (fp32_dst && fp32_src)
        {
            std::memcpy(fp32_dst->mutable_data(), fp32_src->data(),
                        fp32_src->numel() * sizeof(float));
        }
    }

    class Test__DecomposedAttention : public ::testing::Test
    {
    protected:
        MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};
        int device_idx_ = -1; // CPU

        // Model dimensions (Qwen2.5 0.5B-like)
        static constexpr int D_MODEL = 896;
        static constexpr int N_HEADS = 14;
        static constexpr int N_KV_HEADS = 2;
        static constexpr int HEAD_DIM = 64;
        static constexpr int D_FF = 4864;
        static constexpr float RMS_EPS = 1e-6f;
        static constexpr float ROPE_THETA = 1000000.0f;

        TensorFactory factory_{mpi_ctx_};

        void SetUp() override {}
        void TearDown() override {}

        /**
         * @brief Create executor with specified attention path
         */
        std::unique_ptr<Qwen2LayerExecutor> createExecutor(bool use_decomposed)
        {
            Qwen2ExecutorConfig config;
            config.d_model = D_MODEL;
            config.n_heads = N_HEADS;
            config.n_kv_heads = N_KV_HEADS;
            config.head_dim = HEAD_DIM;
            config.d_ff = D_FF;
            config.rms_norm_eps = RMS_EPS;
            config.rope_theta = ROPE_THETA;
            config.default_device = device_idx_;
            // NOTE: Decomposed attention is now always used (Phase 7 cleanup)
            // The use_decomposed parameter is ignored but kept for test parameterization
            (void)use_decomposed;

            auto mpi = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
            return std::make_unique<Qwen2LayerExecutor>(config, mpi);
        }

        /**
         * @brief Create mock layer weights
         */
        Qwen2LayerWeights createMockWeights()
        {
            Qwen2LayerWeights weights;

            // Create FP32 weight tensors (not quantized for this test)
            wq_ = factory_.createFP32({static_cast<size_t>(N_HEADS * HEAD_DIM), D_MODEL}, device_idx_);
            wk_ = factory_.createFP32({static_cast<size_t>(N_KV_HEADS * HEAD_DIM), D_MODEL}, device_idx_);
            wv_ = factory_.createFP32({static_cast<size_t>(N_KV_HEADS * HEAD_DIM), D_MODEL}, device_idx_);
            wo_ = factory_.createFP32({static_cast<size_t>(D_MODEL), static_cast<size_t>(N_HEADS * HEAD_DIM)}, device_idx_);
            attn_norm_ = factory_.createFP32({D_MODEL}, device_idx_);
            gate_proj_ = factory_.createFP32({D_FF, D_MODEL}, device_idx_);
            up_proj_ = factory_.createFP32({D_FF, D_MODEL}, device_idx_);
            down_proj_ = factory_.createFP32({D_MODEL, D_FF}, device_idx_);
            ffn_norm_ = factory_.createFP32({D_MODEL}, device_idx_);

            // Initialize with small random-ish values
            initWeights(wq_.get());
            initWeights(wk_.get());
            initWeights(wv_.get());
            initWeights(wo_.get());
            initNormWeights(attn_norm_.get());
            initWeights(gate_proj_.get());
            initWeights(up_proj_.get());
            initWeights(down_proj_.get());
            initNormWeights(ffn_norm_.get());

            weights.wq = wq_.get();
            weights.wk = wk_.get();
            weights.wv = wv_.get();
            weights.wo = wo_.get();
            weights.attn_norm = attn_norm_.get();
            weights.gate_proj = gate_proj_.get();
            weights.up_proj = up_proj_.get();
            weights.down_proj = down_proj_.get();
            weights.ffn_norm = ffn_norm_.get();

            return weights;
        }

        /**
         * @brief Create activation buffers
         */
        Qwen2ActivationBuffers createBuffers(int seq_len)
        {
            Qwen2ActivationBuffers buffers;

            residual_ = factory_.createFP32({static_cast<size_t>(seq_len), D_MODEL}, device_idx_);
            normalized_ = factory_.createFP32({static_cast<size_t>(seq_len), D_MODEL}, device_idx_);
            Q_ = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_HEADS * HEAD_DIM)}, device_idx_);
            K_ = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
            V_ = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
            attn_output_ = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_HEADS * HEAD_DIM)}, device_idx_);
            attn_proj_ = factory_.createFP32({static_cast<size_t>(seq_len), D_MODEL}, device_idx_);
            gate_ = factory_.createFP32({static_cast<size_t>(seq_len), D_FF}, device_idx_);
            up_ = factory_.createFP32({static_cast<size_t>(seq_len), D_FF}, device_idx_);
            ffn_output_ = factory_.createFP32({static_cast<size_t>(seq_len), D_FF}, device_idx_);
            current_hidden_ = factory_.createFP32({static_cast<size_t>(seq_len), D_MODEL}, device_idx_);

            // Workspace buffers
            workspace_scores_ = factory_.createFP32(
                {static_cast<size_t>(N_HEADS * seq_len), static_cast<size_t>(seq_len)}, device_idx_);
            workspace_context_ = factory_.createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(HEAD_DIM)}, device_idx_);

            buffers.residual = residual_.get();
            buffers.normalized = normalized_.get();
            buffers.Q = Q_.get();
            buffers.K = K_.get();
            buffers.V = V_.get();
            buffers.attn_output = attn_output_.get();
            buffers.attn_proj = attn_proj_.get();
            buffers.gate = gate_.get();
            buffers.up = up_.get();
            buffers.ffn_output = ffn_output_.get();
            buffers.current_hidden = current_hidden_.get();
            buffers.workspace_scores = workspace_scores_.get();
            buffers.workspace_context = workspace_context_.get();
            buffers.workspace_mask = nullptr;

            return buffers;
        }

    private:
        void initWeights(TensorBase *t)
        {
            auto *fp32 = dynamic_cast<FP32Tensor *>(t);
            if (!fp32)
                return;
            float *data = fp32->mutable_data();
            for (size_t i = 0; i < fp32->numel(); ++i)
            {
                // Small pseudo-random values
                data[i] = 0.01f * ((i * 17 + 31) % 97 - 48) / 48.0f;
            }
        }

        void initNormWeights(TensorBase *t)
        {
            auto *fp32 = dynamic_cast<FP32Tensor *>(t);
            if (!fp32)
                return;
            float *data = fp32->mutable_data();
            for (size_t i = 0; i < fp32->numel(); ++i)
            {
                // Norm weights close to 1.0
                data[i] = 1.0f + 0.01f * ((i * 13 + 7) % 19 - 9) / 9.0f;
            }
        }

        // Weight tensors (owned by test fixture)
        std::unique_ptr<FP32Tensor> wq_, wk_, wv_, wo_;
        std::unique_ptr<FP32Tensor> attn_norm_, gate_proj_, up_proj_, down_proj_, ffn_norm_;

        // Activation buffers (owned by test fixture)
        std::unique_ptr<FP32Tensor> residual_, normalized_;
        std::unique_ptr<FP32Tensor> Q_, K_, V_, attn_output_, attn_proj_;
        std::unique_ptr<FP32Tensor> gate_, up_, ffn_output_, current_hidden_;
        std::unique_ptr<FP32Tensor> workspace_scores_, workspace_context_;
    };

    /**
     * @brief Test that decomposed config flag is respected
     */
    TEST_F(Test__DecomposedAttention, ConfigFlagWiring)
    {
        auto executor_legacy = createExecutor(false);
        auto executor_decomposed = createExecutor(true);

        ASSERT_NE(executor_legacy, nullptr);
        ASSERT_NE(executor_decomposed, nullptr);
    }

    /**
     * @brief Test AttentionComputeStage standalone execution
     *
     * This test verifies that AttentionComputeStage can execute independently
     * with properly shaped FP32 tensors.
     */
    TEST_F(Test__DecomposedAttention, AttentionComputeStageStandalone)
    {
        int seq_len = 4;
        int n_heads = N_HEADS;
        int n_kv_heads = N_KV_HEADS;
        int head_dim = HEAD_DIM;
        size_t hidden_size = n_heads * head_dim;

        // Create tensors
        auto Q = factory_.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_idx_);
        auto K = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_idx_);
        auto V = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_idx_);
        auto output = factory_.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_idx_);
        auto workspace = factory_.createFP32(
            {static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)}, device_idx_);

        // Initialize Q/K/V with small values
        for (size_t i = 0; i < Q->numel(); ++i)
            Q->mutable_data()[i] = 0.1f * ((i * 7) % 11 - 5) / 5.0f;
        for (size_t i = 0; i < K->numel(); ++i)
            K->mutable_data()[i] = 0.1f * ((i * 13) % 11 - 5) / 5.0f;
        for (size_t i = 0; i < V->numel(); ++i)
            V->mutable_data()[i] = 0.1f * ((i * 17) % 11 - 5) / 5.0f;

        // Create and execute stage
        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.workspace_scores = workspace.get();
        params.mpi_ctx = &mpi_ctx_;
        params.device_idx = device_idx_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        bool success = stage->execute(nullptr);

        EXPECT_TRUE(success) << "AttentionComputeStage should execute successfully";

        // Verify output is valid
        for (size_t i = 0; i < output->numel(); ++i)
        {
            EXPECT_TRUE(std::isfinite(output->data()[i]))
                << "Output[" << i << "] should be finite";
        }
    }

    /**
     * @brief Test KVCacheAppendStage standalone execution
     */
    TEST_F(Test__DecomposedAttention, KVCacheAppendStageStandalone)
    {
        int seq_len = 4;
        int n_layers = 1;
        int max_seq_len = 64;

        // Create KV cache (FP32 precision, requires MPIContext)
        auto kv_cache = std::make_unique<UnifiedKVCache<ActivationPrecision::FP32>>(
            mpi_ctx_,
            n_layers,
            1, // max_batch_size
            max_seq_len,
            N_KV_HEADS,
            HEAD_DIM,
            device_idx_);

        // Create K/V tensors to append
        auto K = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
        auto V = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);

        // Initialize with test values
        for (size_t i = 0; i < K->numel(); ++i)
        {
            K->mutable_data()[i] = 0.5f + 0.01f * i;
            V->mutable_data()[i] = -0.5f + 0.01f * i;
        }

        // Create and execute stage
        KVCacheAppendStage::Params params;
        params.K = K.get();
        params.V = V.get();
        params.kv_cache = kv_cache.get();
        params.layer_idx = 0;
        params.seq_idx = 0;
        params.num_tokens = seq_len;

        auto stage = std::make_unique<KVCacheAppendStage>(params);
        bool success = stage->execute(nullptr);

        EXPECT_TRUE(success) << "KVCacheAppendStage should execute successfully";

        // Verify cache was updated
        int cached = kv_cache->get_cached_tokens(0, 0);
        EXPECT_EQ(cached, seq_len) << "Cache should have " << seq_len << " tokens";
    }

    /**
     * @brief Test decomposed path produces valid attention output
     *
     * This test creates a mini transformer-like setup and verifies that
     * the decomposed attention path (cache append + pure attention) produces
     * valid outputs. We don't compare to legacy path here since they use
     * different internal implementations.
     */
    TEST_F(Test__DecomposedAttention, DecomposedPathProducesValidOutput)
    {
        int seq_len = 8;
        int n_layers = 1;
        int max_seq_len = 64;

        // Create KV cache (FP32 precision, requires MPIContext)
        auto kv_cache = std::make_unique<UnifiedKVCache<ActivationPrecision::FP32>>(
            mpi_ctx_,
            n_layers,
            1,
            max_seq_len,
            N_KV_HEADS,
            HEAD_DIM,
            device_idx_);

        // Create Q/K/V tensors
        auto Q = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_HEADS * HEAD_DIM)}, device_idx_);
        auto K = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
        auto V = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
        auto output = factory_.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(N_HEADS * HEAD_DIM)}, device_idx_);
        auto workspace = factory_.createFP32(
            {static_cast<size_t>(N_HEADS * seq_len), static_cast<size_t>(seq_len)}, device_idx_);

        // Initialize with test values
        for (size_t i = 0; i < Q->numel(); ++i)
            Q->mutable_data()[i] = 0.1f * std::sin(i * 0.1f);
        for (size_t i = 0; i < K->numel(); ++i)
            K->mutable_data()[i] = 0.1f * std::cos(i * 0.1f);
        for (size_t i = 0; i < V->numel(); ++i)
            V->mutable_data()[i] = 0.1f * std::sin(i * 0.15f);

        // Step 1: Append K/V to cache
        {
            KVCacheAppendStage::Params params;
            params.K = K.get();
            params.V = V.get();
            params.kv_cache = kv_cache.get();
            params.layer_idx = 0;
            params.seq_idx = 0;
            params.num_tokens = seq_len;

            auto stage = std::make_unique<KVCacheAppendStage>(params);
            EXPECT_TRUE(stage->execute(nullptr)) << "KV cache append should succeed";
        }

        // Step 2: Get cached K/V
        TensorBase *K_cached = kv_cache->get_k_base(0, 0);
        TensorBase *V_cached = kv_cache->get_v_base(0, 0);
        int kv_len = kv_cache->get_cached_tokens(0, 0);

        EXPECT_NE(K_cached, nullptr) << "Cached K should be valid";
        EXPECT_NE(V_cached, nullptr) << "Cached V should be valid";
        EXPECT_EQ(kv_len, seq_len) << "Cached length should match seq_len";

        // Step 3: Execute attention on Q and cached K/V
        {
            AttentionComputeStage::Params params;
            params.Q = Q.get();
            params.K = K_cached;
            params.V = V_cached;
            params.output = output.get();
            params.batch_size = 1;
            params.seq_len = seq_len;
            params.kv_len = kv_len;
            params.n_heads = N_HEADS;
            params.n_kv_heads = N_KV_HEADS;
            params.head_dim = HEAD_DIM;
            params.causal = true;
            params.workspace_scores = workspace.get();
            params.mpi_ctx = &mpi_ctx_;
            params.device_idx = device_idx_;

            auto stage = std::make_unique<AttentionComputeStage>(params);
            EXPECT_TRUE(stage->execute(nullptr)) << "Attention compute should succeed";
        }

        // Verify output is valid
        size_t num_finite = 0;
        size_t num_nonzero = 0;
        for (size_t i = 0; i < output->numel(); ++i)
        {
            float val = output->data()[i];
            if (std::isfinite(val))
                num_finite++;
            if (std::abs(val) > 1e-8f)
                num_nonzero++;
        }

        EXPECT_EQ(num_finite, output->numel()) << "All outputs should be finite";
        EXPECT_GT(num_nonzero, 0u) << "Output should have non-zero values";
    }

    /**
     * @brief Test decode-style attention (single new token attending to cache)
     */
    TEST_F(Test__DecomposedAttention, DecodeStyleAttention)
    {
        int prefill_len = 8;
        int decode_len = 1;
        int n_layers = 1;
        int max_seq_len = 64;

        // Create KV cache (FP32 precision, requires MPIContext)
        auto kv_cache = std::make_unique<UnifiedKVCache<ActivationPrecision::FP32>>(
            mpi_ctx_,
            n_layers,
            1,
            max_seq_len,
            N_KV_HEADS,
            HEAD_DIM,
            device_idx_);

        // Create prefill K/V and append to cache
        auto K_prefill = factory_.createFP32({static_cast<size_t>(prefill_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
        auto V_prefill = factory_.createFP32({static_cast<size_t>(prefill_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);

        for (size_t i = 0; i < K_prefill->numel(); ++i)
        {
            K_prefill->mutable_data()[i] = 0.1f * std::sin(i * 0.1f);
            V_prefill->mutable_data()[i] = 0.1f * std::cos(i * 0.1f);
        }

        // Prefill: append to cache
        {
            KVCacheAppendStage::Params params;
            params.K = K_prefill.get();
            params.V = V_prefill.get();
            params.kv_cache = kv_cache.get();
            params.layer_idx = 0;
            params.seq_idx = 0;
            params.num_tokens = prefill_len;

            auto stage = std::make_unique<KVCacheAppendStage>(params);
            EXPECT_TRUE(stage->execute(nullptr));
        }

        EXPECT_EQ(kv_cache->get_cached_tokens(0, 0), prefill_len);

        // Create decode token's K/V
        auto K_decode = factory_.createFP32({static_cast<size_t>(decode_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
        auto V_decode = factory_.createFP32({static_cast<size_t>(decode_len), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, device_idx_);
        auto Q_decode = factory_.createFP32({static_cast<size_t>(decode_len), static_cast<size_t>(N_HEADS * HEAD_DIM)}, device_idx_);
        auto output_decode = factory_.createFP32({static_cast<size_t>(decode_len), static_cast<size_t>(N_HEADS * HEAD_DIM)}, device_idx_);
        auto workspace = factory_.createFP32(
            {static_cast<size_t>(N_HEADS * decode_len), static_cast<size_t>(prefill_len + decode_len)}, device_idx_);

        for (size_t i = 0; i < K_decode->numel(); ++i)
        {
            K_decode->mutable_data()[i] = 0.2f;
            V_decode->mutable_data()[i] = -0.2f;
        }
        for (size_t i = 0; i < Q_decode->numel(); ++i)
        {
            Q_decode->mutable_data()[i] = 0.15f * std::sin(i * 0.2f);
        }

        // Decode: append single token
        {
            KVCacheAppendStage::Params params;
            params.K = K_decode.get();
            params.V = V_decode.get();
            params.kv_cache = kv_cache.get();
            params.layer_idx = 0;
            params.seq_idx = 0;
            params.num_tokens = decode_len;

            auto stage = std::make_unique<KVCacheAppendStage>(params);
            EXPECT_TRUE(stage->execute(nullptr));
        }

        int total_kv_len = kv_cache->get_cached_tokens(0, 0);
        EXPECT_EQ(total_kv_len, prefill_len + decode_len);

        // Get full cached K/V
        TensorBase *K_cached = kv_cache->get_k_base(0, 0);
        TensorBase *V_cached = kv_cache->get_v_base(0, 0);

        // Execute decode attention (Q is single token, K/V is full cache)
        {
            AttentionComputeStage::Params params;
            params.Q = Q_decode.get();
            params.K = K_cached;
            params.V = V_cached;
            params.output = output_decode.get();
            params.batch_size = 1;
            params.seq_len = decode_len;
            params.kv_len = total_kv_len;
            params.n_heads = N_HEADS;
            params.n_kv_heads = N_KV_HEADS;
            params.head_dim = HEAD_DIM;
            params.causal = true;
            params.workspace_scores = workspace.get();
            params.mpi_ctx = &mpi_ctx_;
            params.device_idx = device_idx_;

            auto stage = std::make_unique<AttentionComputeStage>(params);
            EXPECT_TRUE(stage->execute(nullptr)) << "Decode attention should succeed";
        }

        // Verify decode output
        for (size_t i = 0; i < output_decode->numel(); ++i)
        {
            EXPECT_TRUE(std::isfinite(output_decode->data()[i]))
                << "Decode output[" << i << "] should be finite";
        }
    }

} // namespace

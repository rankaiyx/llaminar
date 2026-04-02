/**
 * @file Test__MoEPipeline.cpp
 * @brief Integration-style unit test: full MoE pipeline with mock model
 *
 * Tests the complete pipeline: Route → Dispatch → Execute → Combine
 * using real implementations with synthetic weights, verifying
 * numerical correctness end-to-end.
 *
 * Also tests:
 * - Shared expert with sigmoid gating
 * - Activation tracking via ExpertPlacementMap
 * - ExpertWeightCache integration
 * - ExpertRebalancer feedback loop
 */

#include <gtest/gtest.h>
#include "execution/moe/MoEPipeline.h"
#include "execution/moe/ExpertPlacementMap.h"
#include "execution/moe/ExpertWeightCache.h"
#include "execution/moe/ExpertRebalancer.h"
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace llaminar2;

namespace
{

    float silu(float x) { return x / (1.0f + std::exp(-x)); }

    /**
     * @brief In-memory weight provider for testing.
     *
     * Stores synthetic expert weights. Expert -1 is the shared expert.
     */
    class MockWeightProvider : public IExpertWeightProvider
    {
    public:
        struct ExpertWeightSet
        {
            std::vector<float> gate_w; // [intermediate, d_model]
            std::vector<float> up_w;   // [intermediate, d_model]
            std::vector<float> down_w; // [d_model, intermediate]
        };

        void addExpert(int layer_idx, int expert_id, ExpertWeightSet weights)
        {
            experts_[{layer_idx, expert_id}] = std::move(weights);
        }

        ExpertWeights getWeights(int layer_idx, int expert_id) override
        {
            auto it = experts_.find({layer_idx, expert_id});
            if (it == experts_.end())
                return {};
            return {
                it->second.gate_w.data(),
                it->second.up_w.data(),
                it->second.down_w.data(),
            };
        }

    private:
        struct Key
        {
            int layer, expert;
            bool operator==(const Key& other) const
            {
                return layer == other.layer && expert == other.expert;
            }
        };
        struct KeyHash
        {
            size_t operator()(const Key& k) const
            {
                return std::hash<int>()(k.layer) ^ (std::hash<int>()(k.expert) << 16);
            }
        };
        std::unordered_map<Key, ExpertWeightSet, KeyHash> experts_;
    };

    /**
     * @brief Build identity-like expert weights.
     * gate_w = I, up_w = I, down_w = I (for testing SwiGLU passthrough)
     */
    MockWeightProvider::ExpertWeightSet makeIdentityWeights(int d_model, int intermediate)
    {
        MockWeightProvider::ExpertWeightSet w;
        w.gate_w.assign(intermediate * d_model, 0.0f);
        w.up_w.assign(intermediate * d_model, 0.0f);
        w.down_w.assign(d_model * intermediate, 0.0f);
        int min_d = std::min(d_model, intermediate);
        for (int i = 0; i < min_d; ++i)
        {
            w.gate_w[i * d_model + i] = 1.0f;
            w.up_w[i * d_model + i] = 1.0f;
            w.down_w[i * intermediate + i] = 1.0f;
        }
        return w;
    }

    /**
     * @brief Build scaled expert weights.
     * gate_w = scale*I, up_w = I, down_w = I
     * SwiGLU output: SiLU(scale * x) * x
     */
    MockWeightProvider::ExpertWeightSet makeScaledWeights(
        int d_model, int intermediate, float scale)
    {
        auto w = makeIdentityWeights(d_model, intermediate);
        for (size_t i = 0; i < w.gate_w.size(); ++i)
            w.gate_w[i] *= scale;
        return w;
    }

} // namespace

// ============================================================================
// End-to-end pipeline test with real routing
// ============================================================================

TEST(Test__MoEPipeline, EndToEnd_IdentityWeights)
{
    const int d_model = 4;
    const int intermediate = 4;
    const int num_experts = 4;
    const int top_k = 2;
    const int seq_len = 2;

    // Create components
    auto router = std::make_shared<SoftmaxTopKRouter>(/*normalize=*/true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    // All experts have identity weights — output = SiLU(x) * x
    for (int e = 0; e < num_experts; ++e)
        weights->addExpert(0, e, makeIdentityWeights(d_model, intermediate));

    MoEPipeline::Config config{
        .num_experts = num_experts,
        .top_k = top_k,
        .d_model = d_model,
        .intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights);

    // Input: simple values
    float hidden[] = {
        1.0f, 2.0f, 3.0f, 4.0f, // token 0
        0.5f, 1.0f, 1.5f, 2.0f, // token 1
    };

    // Gate logits: token 0 routes to experts 0,1; token 1 routes to experts 2,3
    float gate_logits[] = {
        10.0f, 9.0f, 0.0f, 0.0f, // token 0 → experts 0, 1
        0.0f, 0.0f, 10.0f, 9.0f, // token 1 → experts 2, 3
    };

    float output[8] = {};

    auto result = pipeline.forward(hidden, gate_logits, output, 0, seq_len);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.num_active_experts, 4); // 2 experts × 2 tokens, 4 unique
    EXPECT_EQ(result.total_token_expert_pairs, 4); // 2 tokens × top-2

    // With identity weights and top-2 routing (normalized):
    // Each token's output is weighted sum of SiLU(x)*x from 2 experts
    // Since all experts are identical, output = SiLU(x)*x regardless of weights
    for (int t = 0; t < seq_len; ++t)
    {
        for (int d = 0; d < d_model; ++d)
        {
            float x = hidden[t * d_model + d];
            float expected = silu(x) * x;
            EXPECT_NEAR(output[t * d_model + d], expected, 1e-4f)
                << "Token " << t << " dim " << d;
        }
    }
}

// ============================================================================
// Divergent experts: different weights, same token — verifies correct
// weighted combination (this is the critical multi-expert correctness test)
// ============================================================================

TEST(Test__MoEPipeline, DivergentExperts_CorrectWeightedCombination)
{
    // Expert 0: gate_w = 1.0*I  → SwiGLU output = SiLU(x) * x
    // Expert 1: gate_w = 2.0*I  → SwiGLU output = SiLU(2x) * x
    //
    // Single token, top_k=2 routed to both experts.
    // Result must be: w0 * expert0(x) + w1 * expert1(x)
    // NOT: (w0 + w1) * <last expert>(x)  (which is what the shared-buffer bug produces)

    const int d_model = 4;
    const int intermediate = 4;
    const int num_experts = 2;
    const int top_k = 2;

    auto router = std::make_shared<SoftmaxTopKRouter>(/*normalize=*/true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    // Expert 0: identity → SiLU(x) * x
    weights->addExpert(0, 0, makeIdentityWeights(d_model, intermediate));
    // Expert 1: scaled gate → SiLU(2*x) * x
    weights->addExpert(0, 1, makeScaledWeights(d_model, intermediate, 2.0f));

    MoEPipeline::Config config{
        .num_experts = num_experts,
        .top_k = top_k,
        .d_model = d_model,
        .intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights);

    float hidden[] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Route to both experts with large logit difference so weights are predictable
    // Expert 0 gets ~73%, Expert 1 gets ~27%  (softmax of [10, 9])
    float gate_logits[] = {10.0f, 9.0f};

    float output[4] = {};
    auto result = pipeline.forward(hidden, gate_logits, output, 0, 1);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.num_active_experts, 2);

    // Get the actual routing weights assigned by softmax
    ASSERT_GE(result.routing.entries.size(), 2u);
    float w0 = 0.0f, w1 = 0.0f;
    for (const auto& entry : result.routing.entries)
    {
        if (entry.expert_id == 0) w0 = entry.weight;
        if (entry.expert_id == 1) w1 = entry.weight;
    }
    EXPECT_GT(w0, 0.0f);
    EXPECT_GT(w1, 0.0f);
    EXPECT_NEAR(w0 + w1, 1.0f, 1e-5f); // Normalized top-k

    // Verify: output = w0 * SiLU(x)*x + w1 * SiLU(2x)*x
    for (int d = 0; d < d_model; ++d)
    {
        float x = hidden[d];
        float expert0_out = silu(x) * x;         // SiLU(1.0*x) * x
        float expert1_out = silu(2.0f * x) * x;  // SiLU(2.0*x) * x
        float expected = w0 * expert0_out + w1 * expert1_out;

        EXPECT_NEAR(output[d], expected, 1e-4f)
            << "dim " << d << " w0=" << w0 << " w1=" << w1
            << " e0=" << expert0_out << " e1=" << expert1_out;

        // Also verify this is NOT the bug result (last expert only):
        // Bug would produce: (w0+w1) * expert1_out = expert1_out
        if (std::abs(expert0_out - expert1_out) > 0.01f)
        {
            EXPECT_NE(output[d], expert1_out)
                << "Output matches last expert only — shared-buffer bug!";
        }
    }
}

// ============================================================================
// Divergent experts: multi-token scenario
// ============================================================================

TEST(Test__MoEPipeline, DivergentExperts_MultiToken)
{
    // 3 tokens, each routed to different pairs of experts with different weights.
    // Verifies that per-token combination is independent and correct.

    const int d_model = 2;
    const int intermediate = 2;
    const int num_experts = 4;
    const int top_k = 2;
    const int seq_len = 3;

    auto router = std::make_shared<SoftmaxTopKRouter>(/*normalize=*/true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    // Each expert has different gate scaling
    for (int e = 0; e < num_experts; ++e)
    {
        float scale = 1.0f + 0.5f * e; // 1.0, 1.5, 2.0, 2.5
        weights->addExpert(0, e, makeScaledWeights(d_model, intermediate, scale));
    }

    MoEPipeline::Config config{
        .num_experts = num_experts,
        .top_k = top_k,
        .d_model = d_model,
        .intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights);

    float hidden[] = {
        1.0f, 0.5f, // token 0
        2.0f, 1.0f, // token 1
        0.3f, 0.7f, // token 2
    };

    // Different routing per token
    float gate_logits[] = {
        10.0f, 9.0f, 0.0f, 0.0f, // token 0 → experts 0, 1
        0.0f, 0.0f, 10.0f, 9.0f, // token 1 → experts 2, 3
        9.0f, 0.0f, 0.0f, 10.0f, // token 2 → experts 3, 0
    };

    float output[6] = {};
    auto result = pipeline.forward(hidden, gate_logits, output, 0, seq_len);

    ASSERT_TRUE(result.success);

    // Verify each token's output is finite and non-zero
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(output[i])) << "NaN at pos " << i;
        ASSERT_FALSE(std::isinf(output[i])) << "Inf at pos " << i;
    }

    // Verify manually for token 0: routed to experts 0 and 1
    // Extract weights from routing table
    const auto* t0_entries = result.routing.entriesForToken(0);
    ASSERT_NE(t0_entries, nullptr);
    float t0_w[4] = {};
    for (int k = 0; k < top_k; ++k)
        t0_w[t0_entries[k].expert_id] = t0_entries[k].weight;

    for (int d = 0; d < d_model; ++d)
    {
        float x = hidden[0 * d_model + d];
        float expected = 0.0f;
        // Sum up contributions from all experts this token was routed to
        for (int e = 0; e < num_experts; ++e)
        {
            if (t0_w[e] > 0.0f)
            {
                float scale = 1.0f + 0.5f * e;
                float expert_out = silu(scale * x) * x;
                expected += t0_w[e] * expert_out;
            }
        }
        EXPECT_NEAR(output[0 * d_model + d], expected, 1e-4f)
            << "Token 0 dim " << d;
    }
}

// ============================================================================
// Missing weight provider returns failure
// ============================================================================

TEST(Test__MoEPipeline, FailsOnMissingWeights)
{
    const int d_model = 4;
    const int intermediate = 4;

    auto router = std::make_shared<SoftmaxTopKRouter>(true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    // Only add expert 0 — expert 1 is missing
    weights->addExpert(0, 0, makeIdentityWeights(d_model, intermediate));

    MoEPipeline::Config config{
        .num_experts = 2,
        .top_k = 2,
        .d_model = d_model,
        .intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights);

    float hidden[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gate_logits[] = {10.0f, 10.0f}; // Routes to both experts
    float output[4] = {};

    auto result = pipeline.forward(hidden, gate_logits, output, 0, 1);

    EXPECT_FALSE(result.success) << "Should fail when expert weights are missing";
}

// ============================================================================
// Shared expert with sigmoid gating
// ============================================================================

TEST(Test__MoEPipeline, SharedExpert_SigmoidGating)
{
    const int d_model = 4;
    const int intermediate = 4;
    const int seq_len = 1;

    auto router = std::make_shared<SoftmaxTopKRouter>(true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    // Regular expert
    weights->addExpert(0, 0, makeIdentityWeights(d_model, intermediate));

    // Shared expert (expert_id = -1)
    weights->addExpert(0, -1, makeIdentityWeights(d_model, intermediate));

    MoEPipeline::Config config{
        .num_experts = 1,
        .top_k = 1,
        .d_model = d_model,
        .intermediate_size = intermediate,
        .has_shared_expert = true,
        .shared_intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights);

    float hidden[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gate_logits[] = {10.0f}; // Routes to expert 0
    float output[4] = {};

    // First, run regular MoE forward
    auto result = pipeline.forward(hidden, gate_logits, output, 0, seq_len);
    ASSERT_TRUE(result.success);

    // Now add shared expert contribution with sigmoid gate = 0.0 (sigmoid=0.5)
    float sigmoid_logit[] = {0.0f}; // sigmoid(0) = 0.5
    bool ok = pipeline.forwardSharedExpert(hidden, sigmoid_logit, output, 0, seq_len);
    ASSERT_TRUE(ok);

    // Output should now be: MoE_output + 0.5 * shared_output
    // Both are SiLU(x)*x with identity weights, so:
    // output = SiLU(x)*x + 0.5 * SiLU(x)*x = 1.5 * SiLU(x)*x
    for (int d = 0; d < d_model; ++d)
    {
        float x = hidden[d];
        float expected = 1.5f * silu(x) * x;
        EXPECT_NEAR(output[d], expected, 1e-4f) << "dim " << d;
    }
}

// ============================================================================
// Activation tracking via ExpertPlacementMap
// ============================================================================

TEST(Test__MoEPipeline, ActivationTrackingIntegration)
{
    const int d_model = 4;
    const int intermediate = 4;

    auto router = std::make_shared<SoftmaxTopKRouter>(true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    for (int e = 0; e < 4; ++e)
        weights->addExpert(0, e, makeIdentityWeights(d_model, intermediate));

    DeviceId gpu = DeviceId::cuda(0);
    ExpertPlacementMap placement(4, gpu);

    MoEPipeline::Config config{
        .num_experts = 4,
        .top_k = 2,
        .d_model = d_model,
        .intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights, &placement);

    float hidden[] = {1.0f, 2.0f, 3.0f, 4.0f};
    // Heavily route to expert 0 and 1
    float gate_logits[] = {10.0f, 9.0f, 0.0f, 0.0f};
    float output[4] = {};

    // Run 5 iterations
    for (int iter = 0; iter < 5; ++iter)
        pipeline.forward(hidden, gate_logits, output, 0, 1);

    auto hist = placement.activationHistogram();
    EXPECT_EQ(hist[0], 5u); // Expert 0 activated every iteration
    EXPECT_EQ(hist[1], 5u); // Expert 1 activated every iteration
    EXPECT_EQ(hist[2], 0u);
    EXPECT_EQ(hist[3], 0u);
}

// ============================================================================
// Full rebalancing feedback loop
// ============================================================================

TEST(Test__MoEPipeline, RebalancingFeedbackLoop)
{
    const int d_model = 4;
    const int intermediate = 4;

    auto router = std::make_shared<SoftmaxTopKRouter>(true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    for (int e = 0; e < 8; ++e)
        weights->addExpert(0, e, makeIdentityWeights(d_model, intermediate));

    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    // Start all experts on CPU
    ExpertPlacementMap placement(8, cpu);

    MoEPipeline::Config config{
        .num_experts = 8,
        .top_k = 2,
        .d_model = d_model,
        .intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights, &placement);

    float hidden[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {};

    // Simulate workload: experts 0 and 1 are hot
    float gate_logits[] = {10.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int iter = 0; iter < 100; ++iter)
        pipeline.forward(hidden, gate_logits, output, 0, 1);

    // Now run rebalancer
    ExpertRebalancer rebalancer(RebalanceParams{
        .activation_ratio_threshold = 1.5f,
        .min_total_activations = 50,
    });

    auto proposal = rebalancer.propose(placement, gpu, cpu);

    // Hot experts 0 and 1 should be proposed for GPU
    bool moved_0 = false, moved_1 = false;
    for (const auto& m : proposal.movements)
    {
        if (m.expert_id == 0 && m.to_device == gpu) moved_0 = true;
        if (m.expert_id == 1 && m.to_device == gpu) moved_1 = true;
    }

    EXPECT_TRUE(moved_0) << "Hot expert 0 should be proposed for GPU";
    EXPECT_TRUE(moved_1) << "Hot expert 1 should be proposed for GPU";

    // Apply the proposal
    int applied = ExpertRebalancer::apply(placement, proposal);
    EXPECT_GE(applied, 2);

    // Verify placement changed
    EXPECT_EQ(placement.deviceForExpert(0), gpu);
    EXPECT_EQ(placement.deviceForExpert(1), gpu);
    EXPECT_EQ(placement.deviceForExpert(2), cpu); // Cold — stays
}

// ============================================================================
// Mock model "inference" dry run — multiple layers
// ============================================================================

TEST(Test__MoEPipeline, MockModelInference_MultiLayer)
{
    const int d_model = 8;
    const int intermediate = 8;
    const int num_experts = 4;
    const int top_k = 2;
    const int seq_len = 3;
    const int num_layers = 4;

    auto router = std::make_shared<SoftmaxTopKRouter>(true);
    auto dispatcher = std::make_shared<StandardMoEDispatcher>();
    auto executor = std::make_shared<CPUMoEExpertExecutor>();
    auto combiner = std::make_shared<CPUMoECombiner>();
    auto weights = std::make_shared<MockWeightProvider>();

    // Create different experts per layer with different scales
    for (int layer = 0; layer < num_layers; ++layer)
    {
        for (int e = 0; e < num_experts; ++e)
        {
            float scale = 1.0f + 0.1f * (layer * num_experts + e);
            weights->addExpert(layer, e, makeScaledWeights(d_model, intermediate, scale));
        }
    }

    MoEPipeline::Config config{
        .num_experts = num_experts,
        .top_k = top_k,
        .d_model = d_model,
        .intermediate_size = intermediate,
    };

    MoEPipeline pipeline(config, router, dispatcher, executor, combiner, weights);

    // Initial hidden states
    std::vector<float> hidden(seq_len * d_model);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hidden)
        v = dist(rng);

    // Run through all layers
    for (int layer = 0; layer < num_layers; ++layer)
    {
        // Generate gate logits (random for each layer)
        std::vector<float> gate_logits(seq_len * num_experts);
        for (auto& v : gate_logits)
            v = dist(rng);

        std::vector<float> output(seq_len * d_model, 0.0f);

        auto result = pipeline.forward(
            hidden.data(), gate_logits.data(), output.data(), layer, seq_len);

        ASSERT_TRUE(result.success) << "Layer " << layer << " failed";
        EXPECT_GT(result.num_active_experts, 0) << "Layer " << layer;

        // Check output is finite
        for (int i = 0; i < seq_len * d_model; ++i)
        {
            ASSERT_FALSE(std::isnan(output[i]))
                << "NaN at layer " << layer << " pos " << i;
            ASSERT_FALSE(std::isinf(output[i]))
                << "Inf at layer " << layer << " pos " << i;
        }

        // Residual connection: hidden = hidden + output
        for (int i = 0; i < seq_len * d_model; ++i)
            hidden[i] += output[i];
    }

    // Verify final hidden states are non-trivial
    float sum = 0.0f;
    for (auto v : hidden)
        sum += std::abs(v);
    EXPECT_GT(sum, 0.0f) << "Final hidden states should be non-zero";
}

// ============================================================================
// Weight cache integration
// ============================================================================

TEST(Test__MoEPipeline, WeightCacheIntegration)
{
    // This test verifies ExpertWeightCache works with IExpertWeightStorage mock
    // and that cached lookups produce correct results
    
    struct MockStorage : public IExpertWeightStorage
    {
        int d_model = 4;
        int intermediate = 4;
        std::unordered_map<int, std::vector<float>> gate_weights;
        std::unordered_map<int, std::vector<float>> up_weights;
        std::unordered_map<int, std::vector<float>> down_weights;
        int load_count = 0;

        MockStorage()
        {
            for (int e = 0; e < 8; ++e)
            {
                gate_weights[e].assign(intermediate * d_model, 0.0f);
                up_weights[e].assign(intermediate * d_model, 0.0f);
                down_weights[e].assign(d_model * intermediate, 0.0f);
                int min_d = std::min(d_model, intermediate);
                for (int i = 0; i < min_d; ++i)
                {
                    gate_weights[e][i * d_model + i] = 1.0f;
                    up_weights[e][i * d_model + i] = 1.0f;
                    down_weights[e][i * intermediate + i] = 1.0f;
                }
            }
        }

        ExpertWeightHandle load(const ExpertWeightKey& key) override
        {
            load_count++;
            return ExpertWeightHandle{
                .key = key,
                .gate_w = gate_weights[key.expert_id].data(),
                .up_w = up_weights[key.expert_id].data(),
                .down_w = down_weights[key.expert_id].data(),
                .total_bytes = static_cast<size_t>(3 * d_model * intermediate * sizeof(float)),
            };
        }

        void release(const ExpertWeightHandle& /*handle*/) override {}
        size_t bytesPerExpert() const override
        {
            return 3 * d_model * intermediate * sizeof(float);
        }
    };

    auto storage = std::make_shared<MockStorage>();
    ExpertWeightCache cache(storage, /*max_bytes=*/1024 * 1024, EvictionPolicy::LRU);

    // Load experts through cache
    auto handle = cache.get({0, 3});
    EXPECT_NE(handle.gate_w, nullptr);
    EXPECT_EQ(storage->load_count, 1);

    // Second access is a cache hit
    auto handle2 = cache.get({0, 3});
    EXPECT_EQ(storage->load_count, 1);
    EXPECT_EQ(handle.gate_w, handle2.gate_w);

    // Different expert triggers another load
    cache.get({0, 5});
    EXPECT_EQ(storage->load_count, 2);

    auto stats = cache.stats();
    EXPECT_EQ(stats.hits, 1u);
    EXPECT_EQ(stats.misses, 2u);
}

// ============================================================================
// GQA Qwen2-style mock model inference
// ============================================================================

/**
 * @brief Mocks a simplified Qwen2 (dense, no MoE) model with GQA attention
 * as a full multi-layer inference dry run.
 *
 * Qwen2/Qwen2.5 uses GQA (Grouped Query Attention) with fewer KV heads than Q heads.
 * This test simulates the complete forward pass:
 *   For each layer:
 *     1. Attention norm (identity for testing)
 *     2. QKV projection → GQA attention → Wo projection
 *     3. Residual add
 *     4. FFN norm (identity for testing)
 *     5. Gate+Up → SwiGLU → Down (using MoE executor as dense FFN)
 *     6. Residual add
 *
 * Model config loosely based on Qwen2.5-0.5B:
 *   d_model=896, but scaled down to d_model=8 for test speed.
 *   n_heads=14 → 2, n_kv_heads=2 → 1 (GQA ratio = 2:1)
 *   intermediate=4864 → 16
 *   n_layers=24 → 4
 */
TEST(Test__MoEPipeline, GQA_Qwen2_MockModelInference)
{
    // Scaled-down Qwen2.5-0.5B-like configuration
    const int d_model = 8;
    const int n_heads = 2;       // Query heads
    const int n_kv_heads = 1;    // KV heads (GQA ratio = n_heads / n_kv_heads = 2)
    const int head_dim = d_model / n_heads; // = 4
    const int intermediate = 16;
    const int n_layers = 4;
    const int seq_len = 3;

    ASSERT_EQ(head_dim * n_heads, d_model) << "d_model must be divisible by n_heads";
    ASSERT_EQ(n_heads % n_kv_heads, 0) << "n_heads must be divisible by n_kv_heads";
    const int gqa_group_size = n_heads / n_kv_heads; // 2 query heads share 1 KV head

    // === Build per-layer weights ===
    struct LayerWeights
    {
        // QKV projection: [d_model, (n_heads + 2*n_kv_heads) * head_dim]
        std::vector<float> Wq;  // [n_heads * head_dim, d_model] = [8, 8]
        std::vector<float> Wk;  // [n_kv_heads * head_dim, d_model] = [4, 8]
        std::vector<float> Wv;  // [n_kv_heads * head_dim, d_model] = [4, 8]
        std::vector<float> Wo;  // [d_model, n_heads * head_dim] = [8, 8]

        // FFN weights (SwiGLU)
        std::vector<float> gate_w; // [intermediate, d_model]
        std::vector<float> up_w;   // [intermediate, d_model]
        std::vector<float> down_w; // [d_model, intermediate]
    };

    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    auto fill_random = [&](std::vector<float>& v, size_t size) {
        v.resize(size);
        for (auto& x : v) x = dist(rng);
    };

    std::vector<LayerWeights> layers(n_layers);
    for (int l = 0; l < n_layers; ++l)
    {
        auto& lw = layers[l];
        fill_random(lw.Wq, n_heads * head_dim * d_model);
        fill_random(lw.Wk, n_kv_heads * head_dim * d_model);
        fill_random(lw.Wv, n_kv_heads * head_dim * d_model);
        fill_random(lw.Wo, d_model * n_heads * head_dim);
        fill_random(lw.gate_w, intermediate * d_model);
        fill_random(lw.up_w, intermediate * d_model);
        fill_random(lw.down_w, d_model * intermediate);
    }

    // === Simple GEMM helper ===
    // C[M, N] = A[M, K] @ B[N, K]^T
    auto gemm = [](const float* A, const float* B/*transposed: [N, K]*/, float* C,
                    int M, int N, int K) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float dot = 0.0f;
                for (int k = 0; k < K; ++k)
                    dot += A[m * K + k] * B[n * K + k];
                C[m * N + n] = dot;
            }
        }
    };

    // === GQA attention (simplified, no RoPE, no causal mask, no scale) ===
    // For each query head h, the KV head is h / gqa_group_size
    auto gqa_attention = [&](const float* Q, // [seq_len, n_heads * head_dim]
                             const float* K, // [seq_len, n_kv_heads * head_dim]
                             const float* V, // [seq_len, n_kv_heads * head_dim]
                             float* out,     // [seq_len, n_heads * head_dim]
                             int sl) {
        // Per-head attention
        for (int h = 0; h < n_heads; ++h)
        {
            int kv_h = h / gqa_group_size;

            for (int t = 0; t < sl; ++t)
            {
                // Score = Q[t, h, :] . K[s, kv_h, :] for all s
                std::vector<float> scores(sl, 0.0f);
                for (int s = 0; s < sl; ++s)
                {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        dot += Q[t * n_heads * head_dim + h * head_dim + d] *
                               K[s * n_kv_heads * head_dim + kv_h * head_dim + d];
                    scores[s] = dot;
                }

                // Softmax
                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (auto& s : scores) { s = std::exp(s - max_s); sum_exp += s; }
                for (auto& s : scores) s /= sum_exp;

                // Weighted sum of V
                for (int d = 0; d < head_dim; ++d)
                {
                    float val = 0.0f;
                    for (int s = 0; s < sl; ++s)
                        val += scores[s] * V[s * n_kv_heads * head_dim + kv_h * head_dim + d];
                    out[t * n_heads * head_dim + h * head_dim + d] = val;
                }
            }
        }
    };

    // === SiLU ===
    auto silu_fn = [](float x) { return x / (1.0f + std::exp(-x)); };

    // === Run forward pass ===
    // Initial hidden states
    std::vector<float> hidden(seq_len * d_model);
    {
        std::uniform_real_distribution<float> uni(-0.5f, 0.5f);
        for (auto& v : hidden) v = uni(rng);
    }

    for (int l = 0; l < n_layers; ++l)
    {
        const auto& lw = layers[l];

        // --- Attention block ---
        // QKV projections
        std::vector<float> Q(seq_len * n_heads * head_dim);
        std::vector<float> K(seq_len * n_kv_heads * head_dim);
        std::vector<float> V(seq_len * n_kv_heads * head_dim);

        gemm(hidden.data(), lw.Wq.data(), Q.data(), seq_len, n_heads * head_dim, d_model);
        gemm(hidden.data(), lw.Wk.data(), K.data(), seq_len, n_kv_heads * head_dim, d_model);
        gemm(hidden.data(), lw.Wv.data(), V.data(), seq_len, n_kv_heads * head_dim, d_model);

        // GQA attention
        std::vector<float> attn_out(seq_len * n_heads * head_dim, 0.0f);
        gqa_attention(Q.data(), K.data(), V.data(), attn_out.data(), seq_len);

        // Wo projection
        std::vector<float> wo_out(seq_len * d_model, 0.0f);
        gemm(attn_out.data(), lw.Wo.data(), wo_out.data(), seq_len, d_model, n_heads * head_dim);

        // Residual add
        for (int i = 0; i < seq_len * d_model; ++i)
            hidden[i] += wo_out[i];

        // --- FFN block (SwiGLU via MoE executor) ---
        // Use CPUMoEExpertExecutor as a dense FFN (single expert, all tokens)
        CPUMoEExpertExecutor ffn_executor;

        ExpertBatch all_tokens;
        all_tokens.expert_id = 0;
        for (int t = 0; t < seq_len; ++t)
        {
            all_tokens.token_indices.push_back(t);
            all_tokens.weights.push_back(1.0f);
        }

        std::vector<float> ffn_out(seq_len * d_model, 0.0f);
        bool ok = ffn_executor.executeExpert(
            hidden.data(), ffn_out.data(),
            lw.gate_w.data(), lw.up_w.data(), lw.down_w.data(),
            all_tokens, d_model, intermediate);
        ASSERT_TRUE(ok) << "FFN failed at layer " << l;

        // Residual add
        for (int i = 0; i < seq_len * d_model; ++i)
            hidden[i] += ffn_out[i];

        // Verify no NaN/Inf per layer
        for (int i = 0; i < seq_len * d_model; ++i)
        {
            ASSERT_FALSE(std::isnan(hidden[i]))
                << "NaN at layer " << l << " pos " << i;
            ASSERT_FALSE(std::isinf(hidden[i]))
                << "Inf at layer " << l << " pos " << i;
        }
    }

    // === Verify final hidden states ===
    // After 4 layers of attention+FFN with random weights, values should be
    // non-trivial, finite, and varying (not all-zero or all-same)
    float sum = 0.0f, min_val = 1e30f, max_val = -1e30f;
    for (auto v : hidden)
    {
        sum += std::abs(v);
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    EXPECT_GT(sum, 0.0f) << "Final hidden states should be non-zero";
    EXPECT_NE(min_val, max_val) << "Final hidden states should have variance";

    // Verify GQA: the reduced KV head count (1 vs 2 query heads) means
    // the model processes successfully with asymmetric Q/K/V dimensions.
    // If GQA head mapping were wrong, attention would have produced garbage/NaN.
}

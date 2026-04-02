/**
 * @file Test__MoEExpertExecutor.cpp
 * @brief Unit tests for IMoEExpertExecutor and CPUMoEExpertExecutor
 *
 * Tests SwiGLU FFN: y = (SiLU(x @ gate_w^T) * (x @ up_w^T)) @ down_w^T
 */

#include <gtest/gtest.h>
#include "execution/moe/IMoEExpertExecutor.h"
#include "execution/moe/MoETypes.h"
#include <cmath>
#include <numeric>
#include <vector>

using namespace llaminar2;

namespace
{

    // SiLU(x) = x * sigmoid(x)
    float silu(float x) { return x / (1.0f + std::exp(-x)); }

    // Build identity-like weights for predictable testing
    // gate_w, up_w: [intermediate, d_model] — row-major
    // down_w: [d_model, intermediate] — row-major
    struct TestWeights
    {
        std::vector<float> gate_w;
        std::vector<float> up_w;
        std::vector<float> down_w;

        TestWeights(int d_model, int intermediate)
        {
            // Identity-like: gate_w[i][j] = (i==j) ? 1 : 0
            gate_w.assign(intermediate * d_model, 0.0f);
            up_w.assign(intermediate * d_model, 0.0f);
            down_w.assign(d_model * intermediate, 0.0f);

            int min_dim = std::min(d_model, intermediate);
            for (int i = 0; i < min_dim; ++i)
            {
                gate_w[i * d_model + i] = 1.0f;
                up_w[i * d_model + i] = 1.0f;
                down_w[i * intermediate + i] = 1.0f;
            }
        }
    };

} // namespace

TEST(Test__MoEExpertExecutor, SwiGLU_IdentityWeights)
{
    const int d_model = 4;
    const int intermediate = 4;

    CPUMoEExpertExecutor executor;
    TestWeights weights(d_model, intermediate);

    // Single token input: [1, 2, 3, 4]
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {};

    ExpertBatch batch;
    batch.expert_id = 0;
    batch.token_indices = {0};
    batch.weights = {1.0f};

    bool ok = executor.executeExpert(
        input, output,
        weights.gate_w.data(), weights.up_w.data(), weights.down_w.data(),
        batch, d_model, intermediate);

    ASSERT_TRUE(ok);

    // With identity weights:
    // gate = input, up = input
    // swiglu = SiLU(input) * input
    // output = swiglu @ I = swiglu
    for (int i = 0; i < d_model; ++i)
    {
        float expected = silu(input[i]) * input[i];
        EXPECT_NEAR(output[i], expected, 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(Test__MoEExpertExecutor, MultipleTokens)
{
    const int d_model = 2;
    const int intermediate = 2;

    CPUMoEExpertExecutor executor;
    TestWeights weights(d_model, intermediate);

    // 3 tokens, but only tokens 0 and 2 are in the batch
    float input[] = {
        1.0f,
        0.5f, // token 0
        9.0f,
        9.0f, // token 1 (not in batch)
        2.0f,
        1.0f, // token 2
    };
    float output[6] = {};

    ExpertBatch batch;
    batch.expert_id = 0;
    batch.token_indices = {0, 2};
    batch.weights = {1.0f, 1.0f};

    bool ok = executor.executeExpert(
        input, output,
        weights.gate_w.data(), weights.up_w.data(), weights.down_w.data(),
        batch, d_model, intermediate);

    ASSERT_TRUE(ok);

    // Token 0 output
    EXPECT_NEAR(output[0 * d_model + 0], silu(1.0f) * 1.0f, 1e-5f);
    EXPECT_NEAR(output[0 * d_model + 1], silu(0.5f) * 0.5f, 1e-5f);

    // Token 2 output (written at output[2*d_model])
    EXPECT_NEAR(output[2 * d_model + 0], silu(2.0f) * 2.0f, 1e-5f);
    EXPECT_NEAR(output[2 * d_model + 1], silu(1.0f) * 1.0f, 1e-5f);
}

TEST(Test__MoEExpertExecutor, EmptyBatch)
{
    CPUMoEExpertExecutor executor;

    float input[] = {1.0f, 2.0f};
    float output[2] = {99.0f, 99.0f};
    float w[4] = {};

    ExpertBatch empty_batch;
    empty_batch.expert_id = 0;

    bool ok = executor.executeExpert(
        input, output, w, w, w, empty_batch, 2, 2);

    ASSERT_TRUE(ok);
    // Output should be unchanged for empty batch
    EXPECT_FLOAT_EQ(output[0], 99.0f);
    EXPECT_FLOAT_EQ(output[1], 99.0f);
}

TEST(Test__MoEExpertExecutor, NonIdentityWeights)
{
    const int d_model = 2;
    const int intermediate = 2;

    CPUMoEExpertExecutor executor;

    // gate_w = [[2, 0], [0, 3]] — scales gate input
    float gate_w[] = {2.0f, 0.0f, 0.0f, 3.0f};
    // up_w = identity
    float up_w[] = {1.0f, 0.0f, 0.0f, 1.0f};
    // down_w = identity
    float down_w[] = {1.0f, 0.0f, 0.0f, 1.0f};

    float input[] = {1.0f, 1.0f}; // token 0
    float output[2] = {};

    ExpertBatch batch;
    batch.expert_id = 0;
    batch.token_indices = {0};
    batch.weights = {1.0f};

    bool ok = executor.executeExpert(
        input, output, gate_w, up_w, down_w,
        batch, d_model, intermediate);

    ASSERT_TRUE(ok);

    // gate = input @ gate_w^T = [1,1] @ [[2,0],[0,3]]^T = [2, 3]
    // up   = input @ up_w^T   = [1,1] @ [[1,0],[0,1]]^T = [1, 1]
    // swiglu = SiLU([2, 3]) * [1, 1] = [SiLU(2), SiLU(3)]
    // output = swiglu @ down_w^T = [SiLU(2), SiLU(3)]
    EXPECT_NEAR(output[0], silu(2.0f) * 1.0f, 1e-5f);
    EXPECT_NEAR(output[1], silu(3.0f) * 1.0f, 1e-5f);
}

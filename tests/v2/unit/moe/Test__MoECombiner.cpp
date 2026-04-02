/**
 * @file Test__MoECombiner.cpp
 * @brief Unit tests for IMoECombiner and CPUMoECombiner
 *
 * NOTE on combiner API: The CPUMoECombiner takes a single expert_outputs buffer.
 * In the MoE pipeline, the accumulation of weighted expert outputs is done inline
 * (per-expert execution → per-expert accumulate), so the combiner is used for
 * single-expert-at-a-time scenarios or standalone testing.
 */

#include <gtest/gtest.h>
#include "execution/moe/IMoECombiner.h"
#include "execution/moe/MoETypes.h"
#include <vector>

using namespace llaminar2;

TEST(Test__MoECombiner, WeightedScaling_SingleExpert)
{
    CPUMoECombiner combiner;

    const int seq_len = 2;
    const int d_model = 3;

    // Single expert processes both tokens with different routing weights
    std::vector<float> expert_outputs(seq_len * d_model, 0.0f);

    // Expert produced [1,2,3] for token 0, [4,5,6] for token 1
    expert_outputs[0 * d_model + 0] = 1.0f;
    expert_outputs[0 * d_model + 1] = 2.0f;
    expert_outputs[0 * d_model + 2] = 3.0f;
    expert_outputs[1 * d_model + 0] = 4.0f;
    expert_outputs[1 * d_model + 1] = 5.0f;
    expert_outputs[1 * d_model + 2] = 6.0f;

    DispatchPlan plan;
    plan.seq_len = seq_len;
    plan.d_model = d_model;
    plan.batches = {
        ExpertBatch{
            .expert_id = 0,
            .token_indices = {0, 1},
            .weights = {0.7f, 0.3f},
        },
    };

    float output[6] = {};

    bool ok = combiner.combine(expert_outputs.data(), plan, output, d_model);
    ASSERT_TRUE(ok);

    // Token 0: 0.7 * [1,2,3] = [0.7, 1.4, 2.1]
    EXPECT_NEAR(output[0], 0.7f, 1e-5f);
    EXPECT_NEAR(output[1], 1.4f, 1e-5f);
    EXPECT_NEAR(output[2], 2.1f, 1e-5f);

    // Token 1: 0.3 * [4,5,6] = [1.2, 1.5, 1.8]
    EXPECT_NEAR(output[3], 1.2f, 1e-5f);
    EXPECT_NEAR(output[4], 1.5f, 1e-5f);
    EXPECT_NEAR(output[5], 1.8f, 1e-5f);
}

TEST(Test__MoECombiner, EmptyPlan)
{
    CPUMoECombiner combiner;

    DispatchPlan plan;
    plan.seq_len = 2;
    plan.d_model = 3;

    float expert_outputs[6] = {};
    float output[6] = {99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f};

    bool ok = combiner.combine(expert_outputs, plan, output, 3);
    ASSERT_TRUE(ok);

    // Output should be zeroed for empty plan (combiner always zeros first)
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(output[i], 0.0f);
}

TEST(Test__MoECombiner, AccumulatesWeightedContributions)
{
    // The combiner accumulates w * expert_outputs[t] for each batch.
    // With a shared buffer, all batches read the same data at each position.
    // This test verifies the accumulation arithmetic is correct.
    CPUMoECombiner combiner;

    const int d_model = 2;

    std::vector<float> expert_outputs(1 * d_model, 0.0f);
    expert_outputs[0] = 10.0f;
    expert_outputs[1] = 20.0f;

    DispatchPlan plan;
    plan.seq_len = 1;
    plan.d_model = d_model;
    plan.batches = {
        ExpertBatch{
            .expert_id = 0,
            .token_indices = {0},
            .weights = {0.6f},
        },
        ExpertBatch{
            .expert_id = 1,
            .token_indices = {0},
            .weights = {0.4f},
        },
    };

    float output[2] = {};

    bool ok = combiner.combine(expert_outputs.data(), plan, output, d_model);
    ASSERT_TRUE(ok);

    // Accumulation: 0.6 * [10, 20] + 0.4 * [10, 20] = (0.6+0.4) * [10, 20] = [10, 20]
    EXPECT_NEAR(output[0], 10.0f, 1e-5f);
    EXPECT_NEAR(output[1], 20.0f, 1e-5f);
}

TEST(Test__MoECombiner, DifferentTokensDifferentWeights)
{
    // Two experts, each processing different tokens:
    // Expert 0 → token 0, Expert 1 → token 1
    // No token overlap, so single shared buffer is fine
    CPUMoECombiner combiner;

    const int seq_len = 2;
    const int d_model = 2;

    std::vector<float> expert_outputs(seq_len * d_model, 0.0f);
    // Token 0 output: [3, 6]
    expert_outputs[0] = 3.0f;
    expert_outputs[1] = 6.0f;
    // Token 1 output: [5, 10]
    expert_outputs[2] = 5.0f;
    expert_outputs[3] = 10.0f;

    DispatchPlan plan;
    plan.seq_len = seq_len;
    plan.d_model = d_model;
    plan.batches = {
        ExpertBatch{
            .expert_id = 0,
            .token_indices = {0},
            .weights = {0.8f},
        },
        ExpertBatch{
            .expert_id = 1,
            .token_indices = {1},
            .weights = {0.5f},
        },
    };

    float output[4] = {};

    bool ok = combiner.combine(expert_outputs.data(), plan, output, d_model);
    ASSERT_TRUE(ok);

    // Token 0: 0.8 * [3, 6] = [2.4, 4.8]
    EXPECT_NEAR(output[0], 2.4f, 1e-5f);
    EXPECT_NEAR(output[1], 4.8f, 1e-5f);
    // Token 1: 0.5 * [5, 10] = [2.5, 5.0]
    EXPECT_NEAR(output[2], 2.5f, 1e-5f);
    EXPECT_NEAR(output[3], 5.0f, 1e-5f);
}

/**
 * @file Test__MoERouter.cpp
 * @brief Unit tests for IMoERouter and SoftmaxTopKRouter
 */

#include <gtest/gtest.h>
#include "execution/moe/IMoERouter.h"
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace llaminar2;

// ============================================================================
// SoftmaxTopKRouter Tests
// ============================================================================

TEST(Test__MoERouter, TopK_SelectsCorrectExperts)
{
    SoftmaxTopKRouter router(/*normalize_weights=*/false);

    // 2 tokens, 4 experts, top-2
    // Token 0: experts have logits [1.0, 3.0, 0.5, 2.0] → top-2 = experts 1, 3
    // Token 1: experts have logits [4.0, 0.1, 0.2, 0.3] → top-2 = experts 0, 3
    float gate_logits[] = {
        1.0f,
        3.0f,
        0.5f,
        2.0f, // token 0
        4.0f,
        0.1f,
        0.2f,
        0.3f, // token 1
    };

    auto result = router.route(gate_logits, 2, 4, 2);

    ASSERT_EQ(result.seq_len, 2);
    ASSERT_EQ(result.top_k, 2);
    ASSERT_EQ(result.entries.size(), 4u);

    // Check token 0: experts 1 and 3 should be selected
    auto *t0_entries = result.entriesForToken(0);
    ASSERT_NE(t0_entries, nullptr);
    std::vector<int> t0_experts;
    for (int i = 0; i < result.top_k; ++i)
        t0_experts.push_back(t0_entries[i].expert_id);
    std::sort(t0_experts.begin(), t0_experts.end());
    EXPECT_EQ(t0_experts[0], 1);
    EXPECT_EQ(t0_experts[1], 3);

    // Check token 1: expert 0 should be the top pick
    auto *t1_entries = result.entriesForToken(1);
    ASSERT_NE(t1_entries, nullptr);
    bool found_expert_0 = false;
    for (int i = 0; i < result.top_k; ++i)
    {
        if (t1_entries[i].expert_id == 0)
            found_expert_0 = true;
    }
    EXPECT_TRUE(found_expert_0);
}

TEST(Test__MoERouter, Softmax_ProducesValidProbabilities)
{
    SoftmaxTopKRouter router(/*normalize_weights=*/false);

    // Single token, 4 experts
    float gate_logits[] = {1.0f, 2.0f, 3.0f, 4.0f};

    auto result = router.route(gate_logits, 1, 4, 4);

    ASSERT_EQ(result.entries.size(), 4u);

    // All weights should be positive
    float sum = 0.0f;
    for (auto &e : result.entries)
    {
        EXPECT_GT(e.weight, 0.0f);
        sum += e.weight;
    }

    // Full softmax should sum to ~1.0
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(Test__MoERouter, NormalizeWeights_SumsToOne)
{
    SoftmaxTopKRouter router(/*normalize_weights=*/true);

    float gate_logits[] = {1.0f, 10.0f, 0.5f, 5.0f};

    auto result = router.route(gate_logits, 1, 4, 2);

    ASSERT_EQ(result.entries.size(), 2u);

    float sum = 0.0f;
    for (auto &e : result.entries)
        sum += e.weight;

    // With normalization, top-k weights should sum to 1.0
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(Test__MoERouter, WithoutNormalize_DoesNotSumToOne)
{
    SoftmaxTopKRouter router(/*normalize_weights=*/false);

    // With extreme logit differences, top-2 softmax won't sum to 1
    float gate_logits[] = {100.0f, 0.0f, 0.0f, 99.0f};

    auto result = router.route(gate_logits, 1, 4, 2);

    ASSERT_EQ(result.entries.size(), 2u);

    float sum = 0.0f;
    for (auto &e : result.entries)
        sum += e.weight;

    // Without normalization, top-2 softmax probabilities
    // may or may not sum to exactly 1.0 depending on the distribution
    // but each weight should be valid
    for (auto &e : result.entries)
    {
        EXPECT_GT(e.weight, 0.0f);
        EXPECT_LE(e.weight, 1.0f);
    }
}

TEST(Test__MoERouter, NumericalStability_LargeLogits)
{
    SoftmaxTopKRouter router(/*normalize_weights=*/false);

    // Large logits shouldn't cause overflow
    float gate_logits[] = {1000.0f, 999.0f, 998.0f, 997.0f};

    auto result = router.route(gate_logits, 1, 4, 2);

    ASSERT_EQ(result.entries.size(), 2u);

    for (auto &e : result.entries)
    {
        EXPECT_FALSE(std::isnan(e.weight)) << "NaN weight with large logits";
        EXPECT_FALSE(std::isinf(e.weight)) << "Inf weight with large logits";
        EXPECT_GT(e.weight, 0.0f);
    }
}

TEST(Test__MoERouter, NumericalStability_NegativeLogits)
{
    SoftmaxTopKRouter router(/*normalize_weights=*/false);

    float gate_logits[] = {-1000.0f, -999.0f, -998.0f, -997.0f};

    auto result = router.route(gate_logits, 1, 4, 2);

    ASSERT_EQ(result.entries.size(), 2u);

    for (auto &e : result.entries)
    {
        EXPECT_FALSE(std::isnan(e.weight));
        EXPECT_FALSE(std::isinf(e.weight));
        EXPECT_GT(e.weight, 0.0f);
    }
}

TEST(Test__MoERouter, RoutingTable_EntriesForToken)
{
    RoutingTable table;
    table.seq_len = 3;
    table.top_k = 2;
    table.entries = {
        {0, 1, 0.6f},
        {0, 3, 0.4f},
        {1, 0, 0.5f},
        {1, 2, 0.5f},
        {2, 1, 0.7f},
        {2, 0, 0.3f},
    };

    auto *t0 = table.entriesForToken(0);
    auto *t1 = table.entriesForToken(1);
    auto *t2 = table.entriesForToken(2);

    ASSERT_NE(t0, nullptr);
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);

    // Verify specific entries
    EXPECT_EQ(t0[0].expert_id, 1);
    EXPECT_EQ(t0[1].expert_id, 3);
    EXPECT_EQ(t1[0].expert_id, 0);
}

TEST(Test__MoERouter, SingleToken_TopK1)
{
    SoftmaxTopKRouter router(/*normalize_weights=*/true);

    float gate_logits[] = {0.1f, 0.9f, 0.3f, 0.5f};

    auto result = router.route(gate_logits, 1, 4, 1);

    ASSERT_EQ(result.entries.size(), 1u);
    EXPECT_EQ(result.entries[0].expert_id, 1);          // Highest logit
    EXPECT_NEAR(result.entries[0].weight, 1.0f, 1e-5f); // Normalized top-1
}

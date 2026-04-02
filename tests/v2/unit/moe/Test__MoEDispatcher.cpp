/**
 * @file Test__MoEDispatcher.cpp
 * @brief Unit tests for IMoEDispatcher and StandardMoEDispatcher
 */

#include <gtest/gtest.h>
#include "execution/moe/IMoEDispatcher.h"
#include "execution/moe/MoETypes.h"
#include <algorithm>
#include <set>

using namespace llaminar2;

TEST(Test__MoEDispatcher, GroupsByExpert)
{
    StandardMoEDispatcher dispatcher;

    RoutingTable table;
    table.seq_len = 3;
    table.top_k = 2;
    table.entries = {
        {0, 1, 0.6f},
        {0, 3, 0.4f},
        {1, 1, 0.5f}, // expert 1 gets tokens 0 and 1
        {1, 2, 0.5f},
        {2, 3, 0.7f}, // expert 3 gets tokens 0 and 2
        {2, 0, 0.3f},
    };

    auto plan = dispatcher.dispatch(table, 64);

    EXPECT_EQ(plan.seq_len, 3);
    EXPECT_EQ(plan.d_model, 64);

    // Should have batches for experts 0, 1, 2, 3
    std::set<int> active_experts;
    for (auto &b : plan.batches)
        active_experts.insert(b.expert_id);

    EXPECT_TRUE(active_experts.count(0));
    EXPECT_TRUE(active_experts.count(1));
    EXPECT_TRUE(active_experts.count(2));
    EXPECT_TRUE(active_experts.count(3));

    // Expert 1 should have tokens 0 and 1
    auto *batch1 = plan.batchForExpert(1);
    ASSERT_NE(batch1, nullptr);
    EXPECT_EQ(batch1->numTokens(), 2);

    std::set<int> exp1_tokens(batch1->token_indices.begin(), batch1->token_indices.end());
    EXPECT_TRUE(exp1_tokens.count(0));
    EXPECT_TRUE(exp1_tokens.count(1));

    // Expert 3 should have tokens 0 and 2
    auto *batch3 = plan.batchForExpert(3);
    ASSERT_NE(batch3, nullptr);
    EXPECT_EQ(batch3->numTokens(), 2);
}

TEST(Test__MoEDispatcher, EmptyRoutingTable)
{
    StandardMoEDispatcher dispatcher;

    RoutingTable table;
    table.seq_len = 0;
    table.top_k = 2;

    auto plan = dispatcher.dispatch(table, 64);

    EXPECT_EQ(plan.numActiveExperts(), 0);
    EXPECT_TRUE(plan.batches.empty());
}

TEST(Test__MoEDispatcher, SingleTokenSingleExpert)
{
    StandardMoEDispatcher dispatcher;

    RoutingTable table;
    table.seq_len = 1;
    table.top_k = 1;
    table.entries = {{0, 5, 1.0f}};

    auto plan = dispatcher.dispatch(table, 32);

    ASSERT_EQ(plan.numActiveExperts(), 1);
    EXPECT_EQ(plan.batches[0].expert_id, 5);
    EXPECT_EQ(plan.batches[0].numTokens(), 1);
    EXPECT_EQ(plan.batches[0].token_indices[0], 0);
    EXPECT_NEAR(plan.batches[0].weights[0], 1.0f, 1e-6f);
}

TEST(Test__MoEDispatcher, WeightsArePreserved)
{
    StandardMoEDispatcher dispatcher;

    RoutingTable table;
    table.seq_len = 2;
    table.top_k = 1;
    table.entries = {
        {0, 0, 0.8f},
        {1, 0, 0.3f},
    };

    auto plan = dispatcher.dispatch(table, 64);

    auto *batch0 = plan.batchForExpert(0);
    ASSERT_NE(batch0, nullptr);
    EXPECT_EQ(batch0->numTokens(), 2);

    // Find weight for token 0
    for (int i = 0; i < batch0->numTokens(); ++i)
    {
        if (batch0->token_indices[i] == 0)
            EXPECT_NEAR(batch0->weights[i], 0.8f, 1e-6f);
        else if (batch0->token_indices[i] == 1)
            EXPECT_NEAR(batch0->weights[i], 0.3f, 1e-6f);
    }
}

TEST(Test__MoEDispatcher, DispatchPlan_BatchForExpert_ReturnsNull)
{
    DispatchPlan plan;
    plan.seq_len = 1;
    plan.d_model = 64;
    plan.batches.push_back(ExpertBatch{.expert_id = 2, .token_indices = {0}, .weights = {1.0f}});

    EXPECT_NE(plan.batchForExpert(2), nullptr);
    EXPECT_EQ(plan.batchForExpert(99), nullptr);
}

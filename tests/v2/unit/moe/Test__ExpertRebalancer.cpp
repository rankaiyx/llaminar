/**
 * @file Test__ExpertRebalancer.cpp
 * @brief Unit tests for ExpertRebalancer (histogram-based dynamic weight movement)
 */

#include <gtest/gtest.h>
#include "execution/moe/ExpertRebalancer.h"
#include <algorithm>

using namespace llaminar2;

TEST(Test__ExpertRebalancer, NoRebalance_InsufficientActivations)
{
    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    ExpertPlacementMap map(4, cpu);
    map.recordActivation(0); // Only 1 activation — below threshold

    ExpertRebalancer rebalancer(RebalanceParams{.min_total_activations = 100});
    auto proposal = rebalancer.propose(map, gpu, cpu);

    EXPECT_TRUE(proposal.empty());
}

TEST(Test__ExpertRebalancer, MovesHotExpertToGPU)
{
    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    // All experts on CPU
    ExpertPlacementMap map(4, cpu);

    // Expert 0 is very hot (80 activations), others are cold (10 each)
    for (int i = 0; i < 80; ++i)
        map.recordActivation(0);
    for (int i = 0; i < 10; ++i)
    {
        map.recordActivation(1);
        map.recordActivation(2);
        map.recordActivation(3);
    }

    RebalanceParams params;
    params.min_total_activations = 50;
    params.activation_ratio_threshold = 1.5f;

    ExpertRebalancer rebalancer(params);
    auto proposal = rebalancer.propose(map, gpu, cpu);

    EXPECT_FALSE(proposal.empty());

    // Expert 0 should be proposed for movement to GPU
    bool found_expert0 = false;
    for (const auto& m : proposal.movements)
    {
        if (m.expert_id == 0 && m.to_device == gpu)
            found_expert0 = true;
    }
    EXPECT_TRUE(found_expert0) << "Hot expert 0 should be moved to GPU";
}

TEST(Test__ExpertRebalancer, MovesColdExpertOffGPU)
{
    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    // All experts on GPU
    ExpertPlacementMap map(4, gpu);

    // Expert 3 is very cold (2 activations), others are hot (50 each)
    for (int i = 0; i < 50; ++i)
    {
        map.recordActivation(0);
        map.recordActivation(1);
        map.recordActivation(2);
    }
    map.recordActivation(3);
    map.recordActivation(3);

    RebalanceParams params;
    params.min_total_activations = 50;
    params.activation_ratio_threshold = 2.0f;

    ExpertRebalancer rebalancer(params);
    auto proposal = rebalancer.propose(map, gpu, cpu);

    // Expert 3 should be proposed for offloading to CPU
    bool found_expert3_offload = false;
    for (const auto& m : proposal.movements)
    {
        if (m.expert_id == 3 && m.to_device == cpu)
            found_expert3_offload = true;
    }
    EXPECT_TRUE(found_expert3_offload) << "Cold expert 3 should be offloaded to CPU";
}

TEST(Test__ExpertRebalancer, Apply_MovesExperts)
{
    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    ExpertPlacementMap map(4, cpu);

    RebalanceProposal proposal;
    proposal.movements = {
        ExpertMovement{.expert_id = 0, .from_device = cpu, .to_device = gpu, .activation_count = 100},
        ExpertMovement{.expert_id = 2, .from_device = cpu, .to_device = gpu, .activation_count = 80},
    };

    int applied = ExpertRebalancer::apply(map, proposal);
    EXPECT_EQ(applied, 2);

    EXPECT_EQ(map.deviceForExpert(0), gpu);
    EXPECT_EQ(map.deviceForExpert(1), cpu); // unchanged
    EXPECT_EQ(map.deviceForExpert(2), gpu);
    EXPECT_EQ(map.deviceForExpert(3), cpu); // unchanged
}

TEST(Test__ExpertRebalancer, MaxMovesPerCycle)
{
    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    // 16 experts all on CPU, all very hot
    ExpertPlacementMap map(16, cpu);
    for (int e = 0; e < 16; ++e)
    {
        for (int i = 0; i < 100; ++i)
            map.recordActivation(e);
    }

    RebalanceParams params;
    params.min_total_activations = 50;
    params.activation_ratio_threshold = 0.5f; // Very aggressive
    params.max_moves_per_cycle = 3;

    ExpertRebalancer rebalancer(params);
    auto proposal = rebalancer.propose(map, gpu, cpu);

    // Should be limited to 3 moves max
    EXPECT_LE(proposal.numMovements(), 3);
}

TEST(Test__ExpertRebalancer, NoMovement_WhenAlreadyOptimal)
{
    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    // All experts already on GPU with uniform activation (no cold experts)
    ExpertPlacementMap map(4, gpu);

    for (int e = 0; e < 4; ++e)
    {
        for (int i = 0; i < 100; ++i)
            map.recordActivation(e);
    }

    RebalanceParams params;
    params.min_total_activations = 50;
    params.activation_ratio_threshold = 2.0f;

    ExpertRebalancer rebalancer(params);
    auto proposal = rebalancer.propose(map, gpu, cpu);

    // With uniform activation on hot device, no expert is cold enough to offload
    // and no expert is on the cold device to promote. Proposal should be empty.
    EXPECT_TRUE(proposal.empty())
        << "Uniform activation on hot device should produce no movements, got "
        << proposal.numMovements();
}

TEST(Test__ExpertRebalancer, LocalityImprovementEstimate)
{
    DeviceId gpu = DeviceId::cuda(0);
    DeviceId cpu = DeviceId::cpu();

    ExpertPlacementMap map(4, cpu);

    // Expert 0 is extremely hot
    for (int i = 0; i < 200; ++i)
        map.recordActivation(0);
    for (int i = 0; i < 10; ++i)
    {
        map.recordActivation(1);
        map.recordActivation(2);
        map.recordActivation(3);
    }

    ExpertRebalancer rebalancer(RebalanceParams{
        .activation_ratio_threshold = 1.5f,
        .min_total_activations = 50,
    });
    auto proposal = rebalancer.propose(map, gpu, cpu);

    if (!proposal.empty())
    {
        EXPECT_GT(proposal.expected_locality_improvement, 0.0);
    }
}

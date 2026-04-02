/**
 * @file Test__MoEConfig.cpp
 * @brief Unit tests for MoEConfig
 */

#include <gtest/gtest.h>
#include "execution/moe/MoEConfig.h"

using namespace llaminar2;

TEST(Test__MoEConfig, DefaultIsInvalid)
{
    MoEConfig config;
    EXPECT_FALSE(config.isValid());
}

TEST(Test__MoEConfig, ValidConfig)
{
    MoEConfig config;
    config.num_experts = 64;
    config.top_k = 8;
    config.intermediate_size = 2560;

    EXPECT_TRUE(config.isValid());
}

TEST(Test__MoEConfig, IsMoELayer_EveryNLayers)
{
    MoEConfig config;
    config.num_experts = 64;
    config.top_k = 8;
    config.intermediate_size = 2560;
    config.moe_every_n_layers = 2;
    config.first_moe_layer = 1;

    EXPECT_FALSE(config.isMoELayer(0)); // Layer 0: not MoE, starts at 1
    EXPECT_TRUE(config.isMoELayer(1));  // Layer 1: first MoE layer
    EXPECT_FALSE(config.isMoELayer(2)); // Layer 2: dense
    EXPECT_TRUE(config.isMoELayer(3));  // Layer 3: MoE (1 + 2*1)
    EXPECT_FALSE(config.isMoELayer(4)); // Layer 4: dense
    EXPECT_TRUE(config.isMoELayer(5));  // Layer 5: MoE (1 + 2*2)
}

TEST(Test__MoEConfig, IsMoELayer_AllLayers)
{
    MoEConfig config;
    config.num_experts = 64;
    config.top_k = 8;
    config.intermediate_size = 2560;
    config.moe_every_n_layers = 1; // Every layer is MoE
    config.first_moe_layer = 0;

    for (int i = 0; i < 10; ++i)
        EXPECT_TRUE(config.isMoELayer(i));
}

TEST(Test__MoEConfig, SharedExpertConfig)
{
    MoEConfig config;
    config.num_experts = 64;
    config.top_k = 8;
    config.intermediate_size = 2560;
    config.has_shared_expert = true;
    config.shared_intermediate_size = 5120;
    config.shared_expert_gate = true;

    EXPECT_TRUE(config.isValid());
    EXPECT_TRUE(config.has_shared_expert);
    EXPECT_TRUE(config.shared_expert_gate);
    EXPECT_EQ(config.shared_intermediate_size, 5120);
}

/**
 * @file Test__ExpertPlacementMap.cpp
 * @brief Unit tests for ExpertPlacementMap: placement strategies, usage tracking, rebalancing
 */

#include <gtest/gtest.h>
#include "execution/moe/ExpertPlacementMap.h"
#include <algorithm>
#include <set>

using namespace llaminar2;

TEST(Test__ExpertPlacementMap, AllLocal_PlacesAllOnOneDevice)
{
    DeviceId gpu0 = DeviceId::cuda(0);
    ExpertPlacementMap map(8, gpu0);

    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(map.deviceForExpert(i), gpu0);

    EXPECT_EQ(map.expertsOnDevice(gpu0).size(), 8u);
}

TEST(Test__ExpertPlacementMap, RoundRobin_DistributesEvenly)
{
    std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
    ExpertPlacementMap map(8, devices, ExpertPlacementStrategy::ROUND_ROBIN);

    auto on_dev0 = map.expertsOnDevice(devices[0]);
    auto on_dev1 = map.expertsOnDevice(devices[1]);

    EXPECT_EQ(on_dev0.size(), 4u);
    EXPECT_EQ(on_dev1.size(), 4u);

    // Experts 0,2,4,6 on device 0; 1,3,5,7 on device 1
    EXPECT_EQ(on_dev0[0], 0);
    EXPECT_EQ(on_dev0[1], 2);
    EXPECT_EQ(on_dev1[0], 1);
    EXPECT_EQ(on_dev1[1], 3);
}

TEST(Test__ExpertPlacementMap, MoveExpert)
{
    DeviceId gpu0 = DeviceId::cuda(0);
    DeviceId gpu1 = DeviceId::cuda(1);

    ExpertPlacementMap map(4, gpu0);

    EXPECT_EQ(map.deviceForExpert(2), gpu0);
    map.moveExpert(2, gpu1);
    EXPECT_EQ(map.deviceForExpert(2), gpu1);
    EXPECT_EQ(map.expertsOnDevice(gpu0).size(), 3u);
    EXPECT_EQ(map.expertsOnDevice(gpu1).size(), 1u);
}

TEST(Test__ExpertPlacementMap, ActivationTracking)
{
    DeviceId gpu = DeviceId::cuda(0);
    ExpertPlacementMap map(4, gpu);

    map.recordActivation(0);
    map.recordActivation(0);
    map.recordActivation(0);
    map.recordActivation(1);
    map.recordActivation(3);
    map.recordActivation(3);

    auto hist = map.activationHistogram();
    ASSERT_EQ(hist.size(), 4u);
    EXPECT_EQ(hist[0], 3u);
    EXPECT_EQ(hist[1], 1u);
    EXPECT_EQ(hist[2], 0u);
    EXPECT_EQ(hist[3], 2u);
}

TEST(Test__ExpertPlacementMap, ResetActivationCounts)
{
    DeviceId gpu = DeviceId::cuda(0);
    ExpertPlacementMap map(4, gpu);

    map.recordActivation(0);
    map.recordActivation(1);
    map.resetActivationCounts();

    auto hist = map.activationHistogram();
    for (auto count : hist)
        EXPECT_EQ(count, 0u);
}

TEST(Test__ExpertPlacementMap, ApplyPlacement)
{
    DeviceId gpu0 = DeviceId::cuda(0);
    DeviceId gpu1 = DeviceId::cuda(1);

    ExpertPlacementMap map(4, gpu0);

    std::vector<DeviceId> new_placement = {gpu0, gpu1, gpu0, gpu1};
    map.applyPlacement(new_placement);

    EXPECT_EQ(map.deviceForExpert(0), gpu0);
    EXPECT_EQ(map.deviceForExpert(1), gpu1);
    EXPECT_EQ(map.deviceForExpert(2), gpu0);
    EXPECT_EQ(map.deviceForExpert(3), gpu1);
}

TEST(Test__ExpertPlacementMap, NumExperts)
{
    DeviceId gpu = DeviceId::cuda(0);
    ExpertPlacementMap map(64, gpu);

    EXPECT_EQ(map.numExperts(), 64);
}

/**
 * @file ExpertPlacementMap.cpp
 * @brief Implementation of ExpertPlacementMap
 */

#include "ExpertPlacementMap.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{

    ExpertPlacementMap::ExpertPlacementMap(int num_experts, DeviceId default_device)
        : num_experts_(num_experts), devices_({default_device}), expert_to_device_(num_experts, default_device), activation_counts_(num_experts, 0)
    {
    }

    ExpertPlacementMap::ExpertPlacementMap(
        int num_experts,
        const std::vector<DeviceId> &devices,
        ExpertPlacementStrategy strategy)
        : num_experts_(num_experts), devices_(devices), expert_to_device_(num_experts), activation_counts_(num_experts, 0)
    {
        if (devices_.empty())
            throw std::invalid_argument("ExpertPlacementMap: devices list must not be empty");

        switch (strategy)
        {
        case ExpertPlacementStrategy::ALL_LOCAL:
            std::fill(expert_to_device_.begin(), expert_to_device_.end(), devices_[0]);
            break;

        case ExpertPlacementStrategy::ROUND_ROBIN:
            for (int e = 0; e < num_experts; ++e)
                expert_to_device_[e] = devices_[e % devices_.size()];
            break;

        case ExpertPlacementStrategy::CAPACITY_AWARE:
            // Capacity-aware defaults to round-robin until we have memory queries
            for (int e = 0; e < num_experts; ++e)
                expert_to_device_[e] = devices_[e % devices_.size()];
            break;
        }
    }

    DeviceId ExpertPlacementMap::deviceForExpert(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= num_experts_)
            throw std::out_of_range("ExpertPlacementMap: expert_id out of range");
        return expert_to_device_[expert_id];
    }

    std::vector<int> ExpertPlacementMap::expertsOnDevice(DeviceId device) const
    {
        std::vector<int> result;
        for (int e = 0; e < num_experts_; ++e)
        {
            if (expert_to_device_[e] == device)
                result.push_back(e);
        }
        return result;
    }

    void ExpertPlacementMap::moveExpert(int expert_id, DeviceId new_device)
    {
        if (expert_id < 0 || expert_id >= num_experts_)
            throw std::out_of_range("ExpertPlacementMap: expert_id out of range");
        expert_to_device_[expert_id] = new_device;
    }

    void ExpertPlacementMap::applyPlacement(const std::vector<DeviceId> &expert_to_device)
    {
        if (static_cast<int>(expert_to_device.size()) != num_experts_)
            throw std::invalid_argument("ExpertPlacementMap: placement size mismatch");
        expert_to_device_ = expert_to_device;
    }

    void ExpertPlacementMap::recordActivation(int expert_id)
    {
        if (expert_id < 0 || expert_id >= num_experts_)
            return;
        std::lock_guard<std::mutex> lock(usage_mutex_);
        activation_counts_[expert_id]++;
    }

    uint64_t ExpertPlacementMap::activationCount(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= num_experts_)
            return 0;
        std::lock_guard<std::mutex> lock(usage_mutex_);
        return activation_counts_[expert_id];
    }

    std::vector<uint64_t> ExpertPlacementMap::activationHistogram() const
    {
        std::lock_guard<std::mutex> lock(usage_mutex_);
        return activation_counts_;
    }

    void ExpertPlacementMap::resetActivationCounts()
    {
        std::lock_guard<std::mutex> lock(usage_mutex_);
        std::fill(activation_counts_.begin(), activation_counts_.end(), 0);
    }

    std::string ExpertPlacementMap::summary() const
    {
        std::ostringstream oss;
        oss << "ExpertPlacementMap [" << num_experts_ << " experts, "
            << devices_.size() << " devices]\n";

        // Per-device expert count
        for (const auto &dev : devices_)
        {
            auto experts = expertsOnDevice(dev);
            oss << "  " << dev.to_string() << ": " << experts.size() << " experts";
            if (!experts.empty() && experts.size() <= 8)
            {
                oss << " [";
                for (size_t i = 0; i < experts.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << experts[i];
                }
                oss << "]";
            }
            oss << "\n";
        }
        return oss.str();
    }

} // namespace llaminar2

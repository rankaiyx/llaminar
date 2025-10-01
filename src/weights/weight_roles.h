#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include "utils/debug_env.h" // for WeightRole enum
#include "logger.h"

namespace llaminar
{

    // Heuristic classification of GGUF tensor names into WeightRole.
    // This will evolve; keep conservative (return Unknown when unsure).
    WeightRole classifyWeightRole(const std::string &tensor_name);

    // Simple parity capture registry (Phase 0). Activated only when
    // debugEnv().weight_slicing.validate is true. Stores original full
    // tensors by name + role for later comparison once sliced loading
    // paths are introduced.
    class WeightParityRegistry
    {
    public:
        static WeightParityRegistry &instance();

        void record(const std::string &name, WeightRole role, const std::vector<float> &data);

        struct Entry
        {
            WeightRole role;
            std::vector<float> data;
        };
        bool get(const std::string &name, Entry &out) const;

    private:
        mutable std::mutex mtx_;
        std::unordered_map<std::string, Entry> map_;
    };

} // namespace llaminar

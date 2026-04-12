/**
 * @file WeightSlicer.cpp
 * @brief Stateless weight slicing computations - implementation
 *
 * Extracted from WeightManager.cpp. Pure computation — no I/O, no caching.
 *
 * @author David Sanftenberg
 */

#include "WeightSlicer.h"
#include "../utils/Logger.h"
#include <sstream>
#include <stdexcept>

namespace llaminar2
{

    WeightSlicer::WeightSlicer(const ModelDimensions &dimensions,
                               const WeightShardingConfig &sharding_config,
                               std::shared_ptr<TensorParallelConfig> tp_config)
        : dimensions_(dimensions),
          sharding_config_(sharding_config),
          tp_config_(std::move(tp_config))
    {
    }

    // =========================================================================
    // Weight categorization
    // =========================================================================

    WeightSlicer::WeightCategory WeightSlicer::categorizeWeight(const std::string &name) const
    {
        if (name.find("attn_q.") != std::string::npos ||
            name.find("attn_k.") != std::string::npos ||
            name.find("attn_v.") != std::string::npos ||
            name.find("attn_qkv.") != std::string::npos)
        {
            return WeightCategory::ATTENTION_QKV;
        }

        if (name.find("attn_output.") != std::string::npos)
        {
            return WeightCategory::ATTENTION_WO;
        }

        if (name.find("ffn_gate.") != std::string::npos ||
            name.find("ffn_up.") != std::string::npos ||
            name.find("ffn_gate_up.") != std::string::npos)
        {
            return WeightCategory::FFN_GATE_UP;
        }

        if (name.find("ffn_down.") != std::string::npos)
        {
            return WeightCategory::FFN_DOWN;
        }

        if (name.find("output.weight") != std::string::npos)
        {
            return WeightCategory::LM_HEAD;
        }

        return WeightCategory::REPLICATE;
    }

    // =========================================================================
    // FusedQKV sub-block size computation
    // =========================================================================

    bool WeightSlicer::determineFusedQKVSubBlockSizes(
        size_t total_rows,
        size_t &q_rows, size_t &k_rows, size_t &v_rows,
        bool &replicate_qk) const
    {
        replicate_qk = false;

        if (dimensions_.isValid())
        {
            const size_t expected_q = static_cast<size_t>(dimensions_.n_heads) * dimensions_.head_dim;
            const size_t expected_kv = static_cast<size_t>(dimensions_.n_kv_heads) * dimensions_.head_dim;
            const size_t expected_qkv = expected_q + 2 * expected_kv;

            if (total_rows == expected_qkv)
            {
                // True GQA layout: [Q(n_heads*hd) | K(n_kv_heads*hd) | V(n_kv_heads*hd)]
                q_rows = expected_q;
                k_rows = expected_kv;
                v_rows = expected_kv;
                return true;
            }
            else if (total_rows % 3 == 0)
            {
                // 3 equal sub-blocks (some models store K/V at full n_heads size)
                q_rows = total_rows / 3;
                k_rows = total_rows / 3;
                v_rows = total_rows / 3;
                return true;
            }
            else if (dimensions_.hasGDN())
            {
                // GDN layout: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
                const size_t gdn_key_dim = static_cast<size_t>(dimensions_.gdn_n_k_heads) * dimensions_.gdn_d_state;
                const size_t gdn_value_dim = static_cast<size_t>(dimensions_.gdn_n_v_heads) * dimensions_.gdn_d_state;
                const size_t expected_gdn_qkv = 2 * gdn_key_dim + gdn_value_dim;

                if (total_rows == expected_gdn_qkv)
                {
                    q_rows = gdn_key_dim;
                    k_rows = gdn_key_dim;
                    v_rows = gdn_value_dim;

                    // GDN modular repeat: replicate Q and K, only shard V
                    replicate_qk = (dimensions_.gdn_n_v_heads > dimensions_.gdn_n_k_heads);
                    return true;
                }
            }
            // Falls through to simple equal row split (return false)
        }
        else
        {
            // No model dimensions: try 3 equal sub-blocks
            if (total_rows % 3 == 0)
            {
                q_rows = total_rows / 3;
                k_rows = total_rows / 3;
                v_rows = total_rows / 3;
                return true;
            }
        }

        return false;
    }

    SliceSpec WeightSlicer::computeSubBlockSlice(
        size_t block_rows, int rank, int world_size, bool replicate)
    {
        if (replicate)
        {
            return {0, block_rows}; // Full sub-block (replicated)
        }

        size_t rows_per_rank = block_rows / world_size;
        size_t start = rows_per_rank * rank;
        size_t count = (rank == world_size - 1)
                           ? (block_rows - start)
                           : rows_per_rank;
        return {start, count};
    }

    // =========================================================================
    // IWeightSlicer: Column-parallel slicing
    // =========================================================================

    SliceSpec WeightSlicer::computeColumnSlice(
        const std::string &name, size_t total_rows,
        int rank, int world_size) const
    {
        if (tp_config_)
        {
            return computeProportionalColumnSlice(name, total_rows, rank);
        }

        // Equal split
        size_t rows_per_rank = total_rows / world_size;
        size_t start = rows_per_rank * rank;
        size_t count = (rank == world_size - 1) ? (total_rows - start) : rows_per_rank;
        return {start, count};
    }

    // =========================================================================
    // IWeightSlicer: Row-parallel / input-parallel slicing
    // =========================================================================

    SliceSpec WeightSlicer::computeRowSlice(
        const std::string &name, size_t total_cols,
        int rank, int world_size) const
    {
        if (tp_config_)
        {
            return computeProportionalRowSlice(name, total_cols, rank);
        }

        // Equal split
        size_t cols_per_rank = total_cols / world_size;
        size_t start = cols_per_rank * rank;
        size_t count = (rank == world_size - 1) ? (total_cols - start) : cols_per_rank;
        return {start, count};
    }

    // =========================================================================
    // IWeightSlicer: FusedQKV sub-block slicing
    // =========================================================================

    std::optional<FusedQKVSliceResult> WeightSlicer::computeFusedQKVSlice(
        const std::string &name, size_t total_rows,
        int rank, int world_size) const
    {
        // Check if this weight is tagged as FusedQKVHeads
        if (sharding_config_.getDimensionType(name) != WeightDimensionType::FusedQKVHeads)
        {
            return std::nullopt;
        }

        size_t q_rows = 0, k_rows = 0, v_rows = 0;
        bool replicate_qk = false;

        if (!determineFusedQKVSubBlockSizes(total_rows, q_rows, k_rows, v_rows, replicate_qk))
        {
            return std::nullopt; // Fall through to simple equal row split
        }

        const size_t sub_block_sizes[3] = {q_rows, k_rows, v_rows};

        // Validate sub-block divisibility
        static constexpr const char *sub_names[3] = {"Q", "K", "V"};
        for (size_t s = 0; s < 3; s++)
        {
            if (replicate_qk && s < 2)
                continue;

            if (sub_block_sizes[s] % static_cast<size_t>(world_size) != 0)
            {
                std::ostringstream err;
                err << "[WeightSlicer] Cannot shard FusedQKV weight '" << name
                    << "': sub-block " << sub_names[s]
                    << " has " << sub_block_sizes[s]
                    << " rows, not divisible by TP degree " << world_size;
                throw std::invalid_argument(err.str());
            }

            if (sub_block_sizes[s] / static_cast<size_t>(world_size) == 0)
            {
                std::ostringstream err;
                err << "[WeightSlicer] Cannot shard FusedQKV weight '" << name
                    << "': sub-block " << sub_names[s]
                    << " has " << sub_block_sizes[s]
                    << " rows but TP degree is " << world_size
                    << " — each rank would get 0 rows";
                throw std::invalid_argument(err.str());
            }
        }

        FusedQKVSliceResult result;
        result.q_total = q_rows;
        result.k_total = k_rows;
        result.v_total = v_rows;
        result.replicate_qk = replicate_qk;

        result.q = computeSubBlockSlice(q_rows, rank, world_size, replicate_qk);
        result.k = computeSubBlockSlice(k_rows, rank, world_size, replicate_qk);
        result.v = computeSubBlockSlice(v_rows, rank, world_size, false /* V always sharded */);

        return result;
    }

    std::optional<FusedQKVSliceResult> WeightSlicer::computeFusedQKVSliceForAssignment(
        const std::string &name, size_t total_rows,
        const DeviceShardingAssignment &assignment) const
    {
        if (!tp_config_)
        {
            return std::nullopt;
        }

        // Check if this weight is tagged as FusedQKVHeads
        if (sharding_config_.getDimensionType(name) != WeightDimensionType::FusedQKVHeads)
        {
            return std::nullopt;
        }

        size_t q_rows = 0, k_rows = 0, v_rows = 0;
        bool replicate_qk = false;

        if (!determineFusedQKVSubBlockSizes(total_rows, q_rows, k_rows, v_rows, replicate_qk))
        {
            return std::nullopt;
        }

        int world_size = tp_config_->worldSize();
        int rank = assignment.local_rank;

        const size_t sub_block_sizes[3] = {q_rows, k_rows, v_rows};

        // Validate divisibility
        static constexpr const char *sub_names[3] = {"Q", "K", "V"};
        for (size_t s = 0; s < 3; s++)
        {
            if (replicate_qk && s < 2)
                continue;

            if (sub_block_sizes[s] % static_cast<size_t>(world_size) != 0)
            {
                std::ostringstream err;
                err << "[WeightSlicer] Cannot shard FusedQKV weight '" << name
                    << "': sub-block " << sub_names[s]
                    << " has " << sub_block_sizes[s]
                    << " rows, not divisible by TP degree " << world_size;
                throw std::invalid_argument(err.str());
            }
        }

        FusedQKVSliceResult result;
        result.q_total = q_rows;
        result.k_total = k_rows;
        result.v_total = v_rows;
        result.replicate_qk = replicate_qk;

        result.q = computeSubBlockSlice(q_rows, rank, world_size, replicate_qk);
        result.k = computeSubBlockSlice(k_rows, rank, world_size, replicate_qk);
        result.v = computeSubBlockSlice(v_rows, rank, world_size, false);

        return result;
    }

    // =========================================================================
    // IWeightSlicer: Slice for device assignment (LOCAL TP)
    // =========================================================================

    SliceSpec WeightSlicer::computeSliceForAssignment(
        const std::string &name, size_t total_size,
        const DeviceShardingAssignment &assignment) const
    {
        WeightDimensionType dim_type = sharding_config_.getDimensionType(name);

        switch (dim_type)
        {
        case WeightDimensionType::Heads:
        {
            if (!tp_config_)
            {
                LOG_ERROR("[WeightSlicer] No TensorParallelConfig for Heads dimension");
                return {0, total_size};
            }
            const int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
                return {0, total_size};
            const size_t head_dim = total_size / static_cast<size_t>(total_heads);
            return {static_cast<size_t>(assignment.head_start) * head_dim,
                    static_cast<size_t>(assignment.head_count) * head_dim};
        }

        case WeightDimensionType::KVHeads:
        {
            if (!tp_config_)
            {
                LOG_ERROR("[WeightSlicer] No TensorParallelConfig for KVHeads dimension");
                return {0, total_size};
            }
            const int total_kv_heads = tp_config_->totalKVHeads();
            if (total_kv_heads <= 0)
                return {0, total_size};
            const size_t head_dim = total_size / static_cast<size_t>(total_kv_heads);
            return {static_cast<size_t>(assignment.kv_head_start) * head_dim,
                    static_cast<size_t>(assignment.kv_head_count) * head_dim};
        }

        case WeightDimensionType::FFNHidden:
            return {static_cast<size_t>(assignment.d_ff_start),
                    static_cast<size_t>(assignment.d_ff_count)};

        case WeightDimensionType::Vocab:
            return {static_cast<size_t>(assignment.vocab_start),
                    static_cast<size_t>(assignment.vocab_count)};

        case WeightDimensionType::ProportionalHeads:
        {
            if (!tp_config_)
            {
                LOG_ERROR("[WeightSlicer] No TensorParallelConfig for ProportionalHeads dimension");
                return {0, total_size};
            }
            const int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
                return {0, total_size};
            const size_t start = total_size * static_cast<size_t>(assignment.head_start) / static_cast<size_t>(total_heads);
            const size_t end = total_size * static_cast<size_t>(assignment.head_start + assignment.head_count) / static_cast<size_t>(total_heads);
            return {start, end - start};
        }

        case WeightDimensionType::Bias1D:
        case WeightDimensionType::None:
        default:
            LOG_ERROR("[WeightSlicer] Cannot compute slice for dimension type "
                      << static_cast<int>(dim_type) << " on weight: " << name);
            return {0, total_size};
        }
    }

    // =========================================================================
    // Proportional slicing (TensorParallelConfig-based)
    // =========================================================================

    SliceSpec WeightSlicer::computeProportionalColumnSlice(
        const std::string &name, size_t total_rows, int rank) const
    {
        if (!tp_config_)
        {
            return {0, total_rows};
        }

        const auto &assignment = tp_config_->forRank(rank);
        WeightCategory category = categorizeWeight(name);

        switch (category)
        {
        case WeightCategory::ATTENTION_QKV:
        {
            const bool is_kv = (name.find("attn_k.") != std::string::npos ||
                                name.find("attn_v.") != std::string::npos);

            const int total_heads = is_kv
                                        ? tp_config_->totalKVHeads()
                                        : tp_config_->totalHeads();
            if (total_heads <= 0)
                return {0, total_rows};

            const size_t head_dim = total_rows / static_cast<size_t>(total_heads);

            if (is_kv)
            {
                return {static_cast<size_t>(assignment.kv_head_start) * head_dim,
                        static_cast<size_t>(assignment.kv_head_count) * head_dim};
            }
            else
            {
                return {static_cast<size_t>(assignment.head_start) * head_dim,
                        static_cast<size_t>(assignment.head_count) * head_dim};
            }
        }

        case WeightCategory::FFN_GATE_UP:
            return {static_cast<size_t>(assignment.d_ff_start),
                    static_cast<size_t>(assignment.d_ff_count)};

        case WeightCategory::LM_HEAD:
            return {static_cast<size_t>(assignment.vocab_start),
                    static_cast<size_t>(assignment.vocab_count)};

        default:
            return {0, total_rows};
        }
    }

    SliceSpec WeightSlicer::computeProportionalRowSlice(
        const std::string &name, size_t total_cols, int rank) const
    {
        if (!tp_config_)
        {
            return {0, total_cols};
        }

        const auto &assignment = tp_config_->forRank(rank);
        WeightCategory category = categorizeWeight(name);

        switch (category)
        {
        case WeightCategory::ATTENTION_WO:
        {
            int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
                return {0, total_cols};
            size_t head_dim = total_cols / static_cast<size_t>(total_heads);
            return {static_cast<size_t>(assignment.head_start) * head_dim,
                    static_cast<size_t>(assignment.head_count) * head_dim};
        }

        case WeightCategory::FFN_DOWN:
            return {static_cast<size_t>(assignment.d_ff_start),
                    static_cast<size_t>(assignment.d_ff_count)};

        default:
            return {0, total_cols};
        }
    }

} // namespace llaminar2

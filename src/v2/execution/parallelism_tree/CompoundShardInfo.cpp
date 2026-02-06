/**
 * @file CompoundShardInfo.cpp
 * @brief Implementation of CompoundShardInfo
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "CompoundShardInfo.h"
#include <sstream>
#include <stack>
#include <cassert>

namespace llaminar2
{

    // =========================================================================
    // CompoundShardInfo — Factory Methods
    // =========================================================================

    CompoundShardInfo CompoundShardInfo::fromLeaf(const ParallelismNode &leaf)
    {
        CompoundShardInfo info;
        info.layer_first = leaf.first_layer;
        info.layer_last = leaf.last_layer;
        info.tp_shard_index = 0;
        info.tp_total_shards = 1;
        info.work_fraction = 1.0f;
        info.has_embedding = leaf.has_embedding;
        info.has_lm_head = leaf.has_lm_head;
        return info;
    }

    namespace
    {
        /**
         * @brief Find the index of a child node within its parent
         * @param parent The parent node
         * @param child Pointer to the child node (from leafDevices())
         * @return Child index, or -1 if not found
         */
        int findChildIndex(const ParallelismNode &parent, const ParallelismNode *child)
        {
            for (size_t i = 0; i < parent.children.size(); ++i)
            {
                // Check if this child is the one we're looking for
                // We need to check if the child pointer is within this child's subtree
                auto leaves = parent.children[i].leafDevices();
                for (const auto *leaf : leaves)
                {
                    if (leaf == child)
                    {
                        return static_cast<int>(i);
                    }
                }
            }
            return -1;
        }

        /**
         * @brief Recursively find the path from a node to a target leaf
         * @param node Current node
         * @param target Target leaf pointer
         * @param path Output path (nodes from current down to parent of leaf)
         * @return true if target found in this subtree
         */
        bool findPathToLeaf(const ParallelismNode &node,
                            const ParallelismNode *target,
                            std::vector<const ParallelismNode *> &path)
        {
            if (&node == target)
            {
                return true;
            }

            if (node.isLeaf())
            {
                return false;
            }

            for (const auto &child : node.children)
            {
                if (findPathToLeaf(child, target, path))
                {
                    path.push_back(&node);
                    return true;
                }
            }

            return false;
        }

    } // anonymous namespace

    CompoundShardInfo CompoundShardInfo::fromTreePath(const ParallelismNode &leaf,
                                                      const ParallelismTree &tree)
    {
        CompoundShardInfo info;
        info.layer_first = leaf.first_layer;
        info.layer_last = leaf.last_layer;
        info.has_embedding = leaf.has_embedding;
        info.has_lm_head = leaf.has_lm_head;

        // Find path from root to leaf (path contains ancestors, leaf last)
        std::vector<const ParallelismNode *> path;
        if (!findPathToLeaf(tree.root, &leaf, path))
        {
            // Leaf not found in tree — return simple case
            info.tp_shard_index = 0;
            info.tp_total_shards = 1;
            info.work_fraction = 1.0f;
            return info;
        }

        // Reverse path so it goes from root to parent-of-leaf
        std::reverse(path.begin(), path.end());

        // Walk path and accumulate TP indices
        int total_shards = 1;
        int shard_index = 0;
        float work_fraction = 1.0f;

        // Current node being examined (starts as leaf)
        const ParallelismNode *current = &leaf;

        // Walk from leaf up to root
        for (const auto *ancestor : path)
        {
            if (ancestor->type == ParallelismNodeType::TENSOR_PARALLEL)
            {
                int child_idx = findChildIndex(*ancestor, &leaf);
                int num_children = static_cast<int>(ancestor->children.size());

                // Compute work fraction from tp_weights if available
                float this_fraction = 1.0f / num_children;
                if (!ancestor->tp_weights.empty() &&
                    static_cast<int>(ancestor->tp_weights.size()) == num_children &&
                    child_idx >= 0)
                {
                    // Sum weights and compute fraction
                    float total_weight = 0.0f;
                    for (float w : ancestor->tp_weights)
                        total_weight += w;
                    if (total_weight > 0.0f)
                    {
                        this_fraction = ancestor->tp_weights[child_idx] / total_weight;
                    }
                }

                // Accumulate: shard_index = outer_index * inner_degree + inner_index
                shard_index = shard_index * num_children + (child_idx >= 0 ? child_idx : 0);
                total_shards *= num_children;
                work_fraction *= this_fraction;
            }

            current = ancestor;
        }

        info.tp_shard_index = shard_index;
        info.tp_total_shards = total_shards;
        info.work_fraction = work_fraction;

        return info;
    }

    // =========================================================================
    // CompoundShardInfo — Validation
    // =========================================================================

    std::vector<std::string> CompoundShardInfo::validate() const
    {
        std::vector<std::string> errors;

        // Layer range validation (if assigned)
        if (layer_first >= 0 || layer_last >= 0)
        {
            if (layer_first < 0)
            {
                errors.push_back("layer_first is negative (" + std::to_string(layer_first) + ")");
            }
            if (layer_last < 0)
            {
                errors.push_back("layer_last is negative (" + std::to_string(layer_last) + ")");
            }
            if (layer_first > layer_last)
            {
                errors.push_back("layer_first (" + std::to_string(layer_first) +
                                 ") > layer_last (" + std::to_string(layer_last) + ")");
            }
        }

        // TP shard validation
        if (tp_total_shards < 1)
        {
            errors.push_back("tp_total_shards must be >= 1, got " + std::to_string(tp_total_shards));
        }

        if (tp_shard_index < 0)
        {
            errors.push_back("tp_shard_index must be >= 0, got " + std::to_string(tp_shard_index));
        }

        if (tp_shard_index >= tp_total_shards)
        {
            errors.push_back("tp_shard_index (" + std::to_string(tp_shard_index) +
                             ") must be < tp_total_shards (" + std::to_string(tp_total_shards) + ")");
        }

        // Work fraction validation
        if (work_fraction <= 0.0f)
        {
            errors.push_back("work_fraction must be > 0, got " + std::to_string(work_fraction));
        }

        if (work_fraction > 1.0f)
        {
            errors.push_back("work_fraction must be <= 1, got " + std::to_string(work_fraction));
        }

        return errors;
    }

    // =========================================================================
    // CompoundShardInfo — Serialization
    // =========================================================================

    std::string CompoundShardInfo::toString() const
    {
        std::ostringstream oss;
        oss << "CompoundShardInfo {\n";
        oss << "  layers: [" << layer_first << ", " << layer_last << "]";
        if (layer_first >= 0 && layer_last >= layer_first)
        {
            oss << " (" << layerCount() << " layers)";
        }
        oss << "\n";

        oss << "  tp_shard: " << tp_shard_index << " of " << tp_total_shards;
        if (isSharded())
        {
            oss << " (sharded)";
        }
        oss << "\n";

        oss << "  work_fraction: " << work_fraction << "\n";
        oss << "  has_embedding: " << (has_embedding ? "true" : "false") << "\n";
        oss << "  has_lm_head: " << (has_lm_head ? "true" : "false") << "\n";
        oss << "}";

        return oss.str();
    }

    // =========================================================================
    // CompoundShardInfo — Comparison
    // =========================================================================

    bool CompoundShardInfo::operator==(const CompoundShardInfo &other) const
    {
        // Use epsilon for float comparison
        constexpr float eps = 1e-6f;

        return layer_first == other.layer_first &&
               layer_last == other.layer_last &&
               tp_shard_index == other.tp_shard_index &&
               tp_total_shards == other.tp_total_shards &&
               std::abs(work_fraction - other.work_fraction) < eps &&
               has_embedding == other.has_embedding &&
               has_lm_head == other.has_lm_head;
    }

} // namespace llaminar2

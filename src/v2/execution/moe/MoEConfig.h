/**
 * @file MoEConfig.h
 * @brief MoE configuration extracted from model metadata
 *
 * Holds all MoE-specific parameters needed by the graph system.
 * Populated from GGUF model config during model loading.
 */

#pragma once

#include <cstdint>
#include <string>

namespace llaminar2
{

    /**
     * @brief MoE model configuration
     *
     * Immutable after construction. Extracted from GGUF metadata.
     */
    struct MoEConfig
    {
        int num_experts = 0;         ///< Total expert count (e.g. 256 for Qwen3.5)
        int top_k = 0;               ///< Experts activated per token (e.g. 8)
        int intermediate_size = 0;   ///< Per-expert FFN intermediate dim
        bool norm_topk_prob = false; ///< Normalize top-k routing weights

        // Shared expert configuration
        bool has_shared_expert = false;   ///< Has always-active shared expert
        int shared_intermediate_size = 0; ///< Shared expert FFN intermediate dim
        bool shared_expert_gate = false;  ///< Has sigmoid gating on shared expert output

        // MoE layer pattern
        int moe_every_n_layers = 1; ///< MoE on every Nth layer (1 = all layers)
        int first_moe_layer = 0;    ///< First layer with MoE FFN

        /// Returns true if this config describes a valid MoE model
        bool isValid() const { return num_experts > 0 && top_k > 0 && intermediate_size > 0; }

        /// Returns true if layer `idx` uses MoE FFN (vs dense)
        bool isMoELayer(int layer_idx) const
        {
            if (!isValid())
                return false;
            if (layer_idx < first_moe_layer)
                return false;
            if (moe_every_n_layers <= 0)
                return false;
            return ((layer_idx - first_moe_layer) % moe_every_n_layers) == 0;
        }
    };

} // namespace llaminar2

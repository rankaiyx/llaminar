#include "weights/WeightRoles.h"
#include <algorithm>

namespace llaminar
{

    static bool contains(const std::string &hay, const std::string &needle)
    {
        return hay.find(needle) != std::string::npos;
    }

    WeightRole classifyWeightRole(const std::string &tensor_name)
    {
        std::string n = tensor_name;
        std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c)
                       { return (char)std::tolower(c); });

        // Embedding (common patterns: tok_embeddings.weight, token_embd.weight, model.embed_tokens.weight)
        if (contains(n, "tok_emb") || contains(n, "token_emb") || contains(n, "embed_tokens"))
            return WeightRole::Embedding;

        // Attention QKV/O - handle various upstream styles
        // Patterns (examples): layers.0.attention.wq.weight, model.layers.1.attn_q.weight, blk.2.attention.w_k.weight
        if (contains(n, ".attention.wq") || contains(n, ".attn_q") || contains(n, ".attention.query"))
            return WeightRole::W_Q;
        if (contains(n, ".attention.wk") || contains(n, ".attn_k") || contains(n, ".attention.key"))
            return WeightRole::W_K;
        if (contains(n, ".attention.wv") || contains(n, ".attn_v") || contains(n, ".attention.value"))
            return WeightRole::W_V;
        if (contains(n, ".attention.wo") || contains(n, ".attn_o") || contains(n, ".attention.out"))
            return WeightRole::W_O;

        // MLP / FFN weights (variable naming across architectures)
        // LLaMA style: feed_forward.w1, feed_forward.w2, feed_forward.w3
        if (contains(n, ".feed_forward.w1") || contains(n, ".ffn.w1") || contains(n, ".mlp.w1") || contains(n, ".ffn_gate"))
            return WeightRole::W1;
        if (contains(n, ".feed_forward.w2") || contains(n, ".ffn.w2") || contains(n, ".mlp.w2") || contains(n, ".ffn_down"))
            return WeightRole::W2;
        if (contains(n, ".feed_forward.w3") || contains(n, ".ffn.w3") || contains(n, ".mlp.w3") || contains(n, ".ffn_up"))
            return WeightRole::W3;

        return WeightRole::Unknown;
    }

    WeightParityRegistry &WeightParityRegistry::instance()
    {
        static WeightParityRegistry inst;
        return inst;
    }

    void WeightParityRegistry::record(const std::string &name, WeightRole role, const std::vector<float> &data)
    {
        if (role == WeightRole::Unknown)
            return; // ignore unknown roles for now
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(name);
        if (it == map_.end())
        {
            map_[name] = Entry{role, data};
            // Increment capture counter (outside conditional for validate to reflect potential future use)
            weightSlicingCounters().captured++;
            if (debugEnv().weight_slicing.validate)
            {
                LOG_DEBUG("[WEIGHT_PARITY_CAPTURE] name=" << name << " role=" << weightRoleToString(role) << " elems=" << data.size());
            }
        }
    }

    bool WeightParityRegistry::get(const std::string &name, Entry &out) const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(name);
        if (it == map_.end())
            return false;
        out = it->second;
        return true;
    }

} // namespace llaminar

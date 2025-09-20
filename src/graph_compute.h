#pragma once

#include "common.h"
#include "tensors/tensor_base.h" // For modern TensorBase interface
#include "kernel_base.h"         // For KernelBase interface
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

// Forward declarations
class ComputeNode;

// Base class for all compute nodes in the graph
class ComputeNode : public std::enable_shared_from_this<ComputeNode>
{
public:
    ComputeNode(const std::string &name, const std::string &operation_type);
    virtual ~ComputeNode() = default;

    // Core functionality
    virtual bool execute() = 0;
    virtual bool validate() const = 0;

    // Graph connectivity
    void addInput(std::shared_ptr<ComputeNode> input);
    void addOutput(std::shared_ptr<ComputeNode> output);

    // Getters
    const std::string &getName() const { return name_; }
    const std::string &getOperationType() const { return operation_type_; }
    const std::vector<std::shared_ptr<ComputeNode>> &getInputs() const { return inputs_; }
    const std::vector<std::shared_ptr<ComputeNode>> &getOutputs() const { return outputs_; }

    // Execution state
    bool isExecuted() const { return executed_; }
    void markExecuted() { executed_ = true; }
    void reset() { executed_ = false; }

protected:
    std::string name_;
    std::string operation_type_;
    std::vector<std::shared_ptr<ComputeNode>> inputs_;
    std::vector<std::shared_ptr<ComputeNode>> outputs_;
    bool executed_;
};

// Compute graph class for managing execution flow
class ComputeGraph
{
public:
    ComputeGraph(const std::string &name = "DefaultGraph");
    ~ComputeGraph() = default;

    // Node management
    void addNode(std::shared_ptr<ComputeNode> node);
    std::shared_ptr<ComputeNode> getNode(const std::string &name) const;

    // Graph operations
    bool execute();
    bool validate() const;
    void reset();
    void clear();

    // Operator overloading for easy graph construction
    ComputeGraph &operator+(std::shared_ptr<ComputeNode> node);
    ComputeGraph &operator+=(std::shared_ptr<ComputeNode> node);

    // Graph analysis
    std::vector<std::shared_ptr<ComputeNode>> getExecutionOrder() const;
    void printGraph() const;
    size_t getNodeCount() const { return nodes_.size(); }

    // Getters
    const std::string &getName() const { return name_; }

private:
    std::string name_;
    std::vector<std::shared_ptr<ComputeNode>> nodes_;
    std::unordered_map<std::string, std::shared_ptr<ComputeNode>> node_map_;

    // Helper functions
    bool hasCycles() const;
    std::vector<std::shared_ptr<ComputeNode>> topologicalSort() const;
    void dfsVisit(std::shared_ptr<ComputeNode> node,
                  std::unordered_map<std::shared_ptr<ComputeNode>, int> &state,
                  std::vector<std::shared_ptr<ComputeNode>> &result) const;
};

// Generic kernel execution node - the new flexible architecture
class KernelNode : public ComputeNode
{
public:
    /**
     * @brief Create a kernel node with a specific kernel implementation
     * @param name Node name for identification
     * @param kernel Unique pointer to the kernel implementation
     */
    KernelNode(const std::string &name, std::unique_ptr<llaminar::KernelBase> kernel);

    bool execute() override;
    bool validate() const override;

    // Tensor management
    void setInputTensors(const std::vector<std::shared_ptr<llaminar::TensorBase>> &inputs);
    void setOutputTensors(const std::vector<std::shared_ptr<llaminar::TensorBase>> &outputs);
    void addInputTensor(std::shared_ptr<llaminar::TensorBase> input);
    void addOutputTensor(std::shared_ptr<llaminar::TensorBase> output);

    // Getters
    const std::vector<std::shared_ptr<llaminar::TensorBase>> &getInputTensors() const { return input_tensors_; }
    const std::vector<std::shared_ptr<llaminar::TensorBase>> &getOutputTensors() const { return output_tensors_; }
    const std::string &getKernelType() const;

private:
    std::unique_ptr<llaminar::KernelBase> kernel_;
    std::vector<std::shared_ptr<llaminar::TensorBase>> input_tensors_;
    std::vector<std::shared_ptr<llaminar::TensorBase>> output_tensors_;
};

// Specific node types

// Matrix multiplication node
class MatMulNode : public ComputeNode
{
public:
    MatMulNode(const std::string &name, int m, int n, int k);

    bool execute() override;
    bool validate() const override;

    // Matrix dimensions
    int getM() const { return m_; }
    int getN() const { return n_; }
    int getK() const { return k_; }

private:
    int m_, n_, k_;
};

// Input/Data node for feeding data into the graph
class DataNode : public ComputeNode
{
public:
    DataNode(const std::string &name, const std::vector<double> &data);

    bool execute() override;
    bool validate() const override;

    const std::vector<double> &getData() const { return data_; }
    void setData(const std::vector<double> &data) { data_ = data; }

private:
    std::vector<double> data_;
};

// Output node for extracting results from the graph
class OutputNode : public ComputeNode
{
public:
    OutputNode(const std::string &name);

    bool execute() override;
    bool validate() const override;

    const std::vector<double> &getResult() const { return result_; }

private:
    std::vector<double> result_;
};

// Transformer-specific compute nodes for LLM inference

// RMS normalization node for neural network layers

// Base class for transformer operations
class TransformerNode : public ComputeNode
{
public:
    TransformerNode(const std::string &name, const std::string &operation_type);

    void setInput(std::shared_ptr<llaminar::TensorBase> input) { input_tensor_ = input; }
    void setOutput(std::shared_ptr<llaminar::TensorBase> output) { output_tensor_ = output; }
    std::shared_ptr<llaminar::TensorBase> getInputTensor() const { return input_tensor_; }
    std::shared_ptr<llaminar::TensorBase> getOutputTensor() const { return output_tensor_; }

protected:
    std::shared_ptr<llaminar::TensorBase> input_tensor_;
    std::shared_ptr<llaminar::TensorBase> output_tensor_;
};

// RMS Normalization node
class RMSNormNode : public TransformerNode
{
public:
    RMSNormNode(const std::string &name, std::shared_ptr<llaminar::TensorBase> weight, float eps = 1e-6f);

    bool execute() override;
    bool validate() const override;

private:
    std::shared_ptr<llaminar::TensorBase> weight_;
    float eps_;
};

// Multi-head attention node
class AttentionNode : public TransformerNode
{
public:
    AttentionNode(const std::string &name,
                  std::shared_ptr<llaminar::TensorBase> q_weight,
                  std::shared_ptr<llaminar::TensorBase> k_weight,
                  std::shared_ptr<llaminar::TensorBase> v_weight,
                  std::shared_ptr<llaminar::TensorBase> out_weight,
                  std::shared_ptr<llaminar::TensorBase> k_cache,
                  std::shared_ptr<llaminar::TensorBase> v_cache,
                  int n_head, int n_head_kv, int n_past);

    bool execute() override;
    bool validate() const override;

    void setKVCache(std::shared_ptr<llaminar::TensorBase> k_cache, std::shared_ptr<llaminar::TensorBase> v_cache)
    {
        k_cache_ = k_cache;
        v_cache_ = v_cache;
    }

private:
    std::shared_ptr<llaminar::TensorBase> q_weight_;
    std::shared_ptr<llaminar::TensorBase> k_weight_;
    std::shared_ptr<llaminar::TensorBase> v_weight_;
    std::shared_ptr<llaminar::TensorBase> out_weight_;
    std::shared_ptr<llaminar::TensorBase> k_cache_, v_cache_;

    int n_head_, n_head_kv_, n_past_;
    int head_dim_;
    int kv_seq_len_ = 0; // Current KV cache length
};

// MLP (Feed Forward) node
class MLPNode : public TransformerNode
{
public:
    MLPNode(const std::string &name,
            std::shared_ptr<llaminar::TensorBase> gate_weight,
            std::shared_ptr<llaminar::TensorBase> up_weight,
            std::shared_ptr<llaminar::TensorBase> down_weight);

    bool execute() override;
    bool validate() const override;

private:
    std::shared_ptr<llaminar::TensorBase> gate_weight_;
    std::shared_ptr<llaminar::TensorBase> up_weight_;
    std::shared_ptr<llaminar::TensorBase> down_weight_;
};

// Complete transformer block node
class TransformerBlockNode : public TransformerNode
{
public:
    TransformerBlockNode(const std::string &name, int layer_idx,
                         std::shared_ptr<RMSNormNode> attn_norm,
                         std::shared_ptr<AttentionNode> attention,
                         std::shared_ptr<RMSNormNode> ffn_norm,
                         std::shared_ptr<MLPNode> mlp);

    bool execute() override;
    bool validate() const override;

private:
    int layer_idx_;
    std::shared_ptr<RMSNormNode> attn_norm_;
    std::shared_ptr<AttentionNode> attention_;
    std::shared_ptr<RMSNormNode> ffn_norm_;
    std::shared_ptr<MLPNode> mlp_;

    // Temporary tensors for residual connections
    std::shared_ptr<llaminar::TensorBase> attn_residual_;
    std::shared_ptr<llaminar::TensorBase> ffn_residual_;
};

// Embedding lookup node
class EmbeddingNode : public TransformerNode
{
public:
    EmbeddingNode(const std::string &name, std::shared_ptr<llaminar::TensorBase> embedding_weights);

    bool execute() override;
    bool validate() const override;

    void setTokenIds(const std::vector<int> &token_ids) { token_ids_ = token_ids; }

private:
    std::shared_ptr<llaminar::TensorBase> embedding_weights_;
    std::vector<int> token_ids_;
};

// Linear projection node (matrix multiplication + optional bias)
class LinearNode : public TransformerNode
{
public:
    LinearNode(const std::string &name,
               std::shared_ptr<llaminar::TensorBase> weight,
               std::shared_ptr<llaminar::TensorBase> bias = nullptr);

public:
    bool execute() override;
    bool validate() const override;

private:
    std::shared_ptr<llaminar::TensorBase> weight_;
    std::shared_ptr<llaminar::TensorBase> bias_;
};
#pragma once

#include "common.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

// Forward declarations
class ComputeNode;
class Tensor;

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
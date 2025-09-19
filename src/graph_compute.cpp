#include "graph_compute.h"
#include <iostream>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <stack>

// ComputeNode implementation
ComputeNode::ComputeNode(const std::string &name, const std::string &operation_type)
    : name_(name), operation_type_(operation_type), executed_(false) {}

void ComputeNode::addInput(std::shared_ptr<ComputeNode> input)
{
    inputs_.push_back(input);
    input->outputs_.push_back(shared_from_this());
}

void ComputeNode::addOutput(std::shared_ptr<ComputeNode> output)
{
    outputs_.push_back(output);
    output->inputs_.push_back(shared_from_this());
}

// ComputeGraph implementation
ComputeGraph::ComputeGraph(const std::string &name) : name_(name) {}

void ComputeGraph::addNode(std::shared_ptr<ComputeNode> node)
{
    if (!node)
        return;

    // Check if node already exists
    if (node_map_.find(node->getName()) != node_map_.end())
    {
        std::cerr << "Warning: Node '" << node->getName() << "' already exists in graph" << std::endl;
        return;
    }

    nodes_.push_back(node);
    node_map_[node->getName()] = node;
}

std::shared_ptr<ComputeNode> ComputeGraph::getNode(const std::string &name) const
{
    auto it = node_map_.find(name);
    return (it != node_map_.end()) ? it->second : nullptr;
}

bool ComputeGraph::execute()
{
    if (!validate())
    {
        std::cerr << "Graph validation failed before execution" << std::endl;
        return false;
    }

    // Get execution order through topological sort
    auto execution_order = getExecutionOrder();
    if (execution_order.empty() && !nodes_.empty())
    {
        std::cerr << "Failed to determine execution order (possible cycle detected)" << std::endl;
        return false;
    }

    // Execute nodes in order
    for (auto &node : execution_order)
    {
        std::cout << "Executing node: " << node->getName() << " (" << node->getOperationType() << ")" << std::endl;

        if (!node->execute())
        {
            std::cerr << "Failed to execute node: " << node->getName() << std::endl;
            return false;
        }

        node->markExecuted();
    }

    std::cout << "Graph execution completed successfully" << std::endl;
    return true;
}

bool ComputeGraph::validate() const
{
    // Check for cycles
    if (hasCycles())
    {
        std::cerr << "Graph validation failed: Cycle detected" << std::endl;
        return false;
    }

    // Validate each node
    for (const auto &node : nodes_)
    {
        if (!node->validate())
        {
            std::cerr << "Graph validation failed: Node '" << node->getName() << "' validation failed" << std::endl;
            return false;
        }
    }

    return true;
}

void ComputeGraph::reset()
{
    for (auto &node : nodes_)
    {
        node->reset();
    }
}

void ComputeGraph::clear()
{
    nodes_.clear();
    node_map_.clear();
}

ComputeGraph &ComputeGraph::operator+(std::shared_ptr<ComputeNode> node)
{
    addNode(node);
    return *this;
}

ComputeGraph &ComputeGraph::operator+=(std::shared_ptr<ComputeNode> node)
{
    addNode(node);
    return *this;
}

std::vector<std::shared_ptr<ComputeNode>> ComputeGraph::getExecutionOrder() const
{
    return topologicalSort();
}

void ComputeGraph::printGraph() const
{
    std::cout << "\n=== Compute Graph: " << name_ << " ===" << std::endl;
    std::cout << "Nodes: " << nodes_.size() << std::endl;

    for (const auto &node : nodes_)
    {
        std::cout << "  " << node->getName() << " (" << node->getOperationType() << ")";
        std::cout << " - Inputs: " << node->getInputs().size();
        std::cout << ", Outputs: " << node->getOutputs().size() << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

bool ComputeGraph::hasCycles() const
{
    std::unordered_map<std::shared_ptr<ComputeNode>, int> state; // 0=unvisited, 1=visiting, 2=visited

    for (const auto &node : nodes_)
    {
        if (state[node] == 0)
        {
            std::vector<std::shared_ptr<ComputeNode>> dummy;
            try
            {
                dfsVisit(node, state, dummy);
            }
            catch (const std::runtime_error &)
            {
                return true; // Cycle detected
            }
        }
    }
    return false;
}

std::vector<std::shared_ptr<ComputeNode>> ComputeGraph::topologicalSort() const
{
    std::unordered_map<std::shared_ptr<ComputeNode>, int> state; // 0=unvisited, 1=visiting, 2=visited
    std::vector<std::shared_ptr<ComputeNode>> result;

    for (const auto &node : nodes_)
    {
        if (state[node] == 0)
        {
            try
            {
                dfsVisit(node, state, result);
            }
            catch (const std::runtime_error &)
            {
                return {}; // Cycle detected, return empty vector
            }
        }
    }

    std::reverse(result.begin(), result.end());
    return result;
}

void ComputeGraph::dfsVisit(std::shared_ptr<ComputeNode> node,
                            std::unordered_map<std::shared_ptr<ComputeNode>, int> &state,
                            std::vector<std::shared_ptr<ComputeNode>> &result) const
{
    state[node] = 1; // Mark as visiting

    for (const auto &output : node->getOutputs())
    {
        if (state[output] == 1)
        {
            throw std::runtime_error("Cycle detected");
        }
        if (state[output] == 0)
        {
            dfsVisit(output, state, result);
        }
    }

    state[node] = 2; // Mark as visited
    result.push_back(node);
}

// MatMulNode implementation
MatMulNode::MatMulNode(const std::string &name, int m, int n, int k)
    : ComputeNode(name, "MatMul"), m_(m), n_(n), k_(k) {}

bool MatMulNode::execute()
{
    std::cout << "  MatMul: " << m_ << "x" << n_ << "x" << k_ << std::endl;
    // TODO: Call COSMA kernel through KernelManager
    return true;
}

bool MatMulNode::validate() const
{
    return m_ > 0 && n_ > 0 && k_ > 0;
}

// DataNode implementation
DataNode::DataNode(const std::string &name, const std::vector<double> &data)
    : ComputeNode(name, "Data"), data_(data) {}

bool DataNode::execute()
{
    std::cout << "  Data node: " << data_.size() << " elements" << std::endl;
    return true;
}

bool DataNode::validate() const
{
    return !data_.empty();
}

// OutputNode implementation
OutputNode::OutputNode(const std::string &name)
    : ComputeNode(name, "Output") {}

bool OutputNode::execute()
{
    std::cout << "  Output node: Collecting results" << std::endl;
    // TODO: Collect results from input nodes
    return true;
}

bool OutputNode::validate() const
{
    return !inputs_.empty(); // Must have at least one input
}
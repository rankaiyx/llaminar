#include "../src/graph_compute.h"
#include <iostream>
#include <memory>

// Test compute graph functionality
bool testComputeGraph()
{
    std::cout << "Testing compute graph functionality..." << std::endl;

    // Create a compute graph
    ComputeGraph graph("TestGraph");

    // Create some test nodes
    auto data_node = std::make_shared<DataNode>("input_data", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    auto matmul_node = std::make_shared<MatMulNode>("matrix_multiply", 2, 2, 2);
    auto output_node = std::make_shared<OutputNode>("output");

    // Add nodes to graph using operator overloading
    graph + data_node + matmul_node + output_node;

    // Set up connections
    matmul_node->addInput(data_node);
    output_node->addInput(matmul_node);

    // Print graph structure
    graph.printGraph();

    // Validate graph
    if (!graph.validate())
    {
        std::cerr << "Graph validation failed!" << std::endl;
        return false;
    }

    std::cout << "Graph validation passed" << std::endl;

    // Test execution order
    auto execution_order = graph.getExecutionOrder();
    std::cout << "Execution order:" << std::endl;
    for (const auto &node : execution_order)
    {
        std::cout << "  " << node->getName() << " (" << node->getOperationType() << ")" << std::endl;
    }

    // Execute graph
    bool success = graph.execute();
    if (!success)
    {
        std::cerr << "Graph execution failed!" << std::endl;
        return false;
    }

    std::cout << "Graph execution completed successfully" << std::endl;
    return true;
}

// Test cycle detection
bool testCycleDetection()
{
    std::cout << "\nTesting cycle detection..." << std::endl;

    ComputeGraph graph("CycleTestGraph");

    auto node1 = std::make_shared<DataNode>("node1", std::vector<double>{1.0});
    auto node2 = std::make_shared<MatMulNode>("node2", 1, 1, 1);
    auto node3 = std::make_shared<OutputNode>("node3");

    graph + node1 + node2 + node3;

    // Create a cycle: node1 -> node2 -> node3 -> node1
    node2->addInput(node1);
    node3->addInput(node2);
    node1->addInput(node3); // This creates a cycle

    // Graph should fail validation due to cycle
    if (graph.validate())
    {
        std::cerr << "Cycle detection failed - graph should not validate!" << std::endl;
        return false;
    }

    std::cout << "Cycle detection working correctly" << std::endl;
    return true;
}

int main()
{
    std::cout << "\n=== Compute Graph Test ===" << std::endl;
    std::cout << "Testing compute graph system functionality" << std::endl;

    bool all_tests_passed = true;

    // Test basic graph functionality
    if (!testComputeGraph())
    {
        all_tests_passed = false;
    }

    // Test cycle detection
    if (!testCycleDetection())
    {
        all_tests_passed = false;
    }

    if (all_tests_passed)
    {
        std::cout << "\n✓ COMPUTE GRAPH TEST SUCCESS: All tests passed" << std::endl;
    }
    else
    {
        std::cout << "\n✗ COMPUTE GRAPH TEST FAILURE: Some tests failed" << std::endl;
    }
    std::cout << "==========================" << std::endl;

    return all_tests_passed ? 0 : 1;
}
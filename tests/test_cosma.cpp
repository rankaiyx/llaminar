#include "../src/common.h"
#include "../src/argument_parser.h"
#include "../src/kernel_manager.h"
#include "../src/kernels/mul_mat.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Generate random matrix data
void fillMatrixRandom(std::vector<double> &matrix, size_t size, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 0; i < size; ++i)
    {
        matrix[i] = dis(gen);
    }
}

// Run COSMA kernel test
bool runCOSMAKernelTest(int m, int n, int k, int rank, int size)
{
    if (rank == 0)
    {
        std::cout << "Testing COSMA kernel: " << m << "x" << n << "x" << k
                  << " with " << size << " processes..." << std::endl;
    }

    // Create test matrices
    std::vector<double> data_A(m * k);
    std::vector<double> data_B(k * n);
    std::vector<double> data_C(m * n, 0.0);

    fillMatrixRandom(data_A, m * k, rank);
    fillMatrixRandom(data_B, k * n, rank + 100);

    // Create tensors
    auto tensor_A = std::make_shared<Tensor>(std::vector<int>{m, k}, data_A);
    auto tensor_B = std::make_shared<Tensor>(std::vector<int>{k, n}, data_B);
    auto tensor_C = std::make_shared<Tensor>(std::vector<int>{m, n}, data_C);

    // Prepare inputs and outputs
    std::vector<std::shared_ptr<Tensor>> inputs = {tensor_A, tensor_B};
    std::vector<std::shared_ptr<Tensor>> outputs = {tensor_C};

    // Execute kernel
    auto start = std::chrono::high_resolution_clock::now();
    bool success = KernelManager::getInstance().executeKernel("matrix_multiplication", inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();

    if (success)
    {
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Calculate performance metrics
        double ops = 2.0 * m * n * k; // FMA operations
        double gflops = (ops / 1e9) / (time_ms / 1000.0);
        double memory_gb = (m * k + k * n + m * n) * sizeof(double) / (1024.0 * 1024.0 * 1024.0);

        if (rank == 0)
        {
            std::cout << "  Time: " << time_ms << " ms" << std::endl;
            std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
            std::cout << "  Memory: " << memory_gb << " GB" << std::endl;
        }
    }

    return success;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse arguments
    ArgumentParser parser(argc, argv);
    LlaminarParams params;

    if (!parser.parse(params))
    {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        std::cout << "\n=== COSMA Kernel Test ===" << std::endl;
        std::cout << "Testing COSMA matrix multiplication kernels" << std::endl;
    }

    // Force kernel registration by creating a dummy instance
    // This ensures the static constructor is called
    auto dummy_kernel = std::make_shared<MatMulKernel>();

    // Test kernel registration
    if (rank == 0)
    {
        auto &kernel_manager = KernelManager::getInstance();
        auto registered_ops = kernel_manager.getRegisteredOperations();

        std::cout << "\nRegistered kernels:" << std::endl;
        for (const auto &op : registered_ops)
        {
            std::cout << "  " << op << std::endl;
        }

        if (!kernel_manager.isRegistered("matrix_multiplication"))
        {
            std::cerr << "Error: matrix_multiplication kernel not registered!" << std::endl;
            MPI_Finalize();
            return 1;
        }
    }

    // Run test cases
    std::vector<std::tuple<int, int, int>> test_cases = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512}};

    bool all_tests_passed = true;

    for (const auto &test_case : test_cases)
    {
        int m, n, k;
        std::tie(m, n, k) = test_case;

        bool success = runCOSMAKernelTest(m, n, k, rank, size);
        if (!success)
        {
            all_tests_passed = false;
            if (rank == 0)
            {
                std::cerr << "  ✗ Test failed for " << m << "x" << n << "x" << k << std::endl;
            }
        }
        else
        {
            if (rank == 0)
            {
                std::cout << "  ✓ Test passed for " << m << "x" << n << "x" << k << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Print performance report
    if (rank == 0)
    {
        KernelManager::getInstance().printPerformanceReport();

        if (all_tests_passed)
        {
            std::cout << "\n✓ COSMA KERNEL TEST SUCCESS: All tests passed" << std::endl;
        }
        else
        {
            std::cout << "\n✗ COSMA KERNEL TEST FAILURE: Some tests failed" << std::endl;
        }
        std::cout << "=========================" << std::endl;
    }

    MPI_Finalize();
    return all_tests_passed ? 0 : 1;
}
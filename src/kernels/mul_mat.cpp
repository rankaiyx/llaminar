#include "mul_mat.h"
#include <iostream>
#include <mpi.h>
#include <chrono>

// COSMA headers
#include <cosma/multiply.hpp>
#include <cosma/strategy.hpp>
#include <cosma/matrix.hpp>
#include <cosma/context.hpp>

MatMulKernel::MatMulKernel()
    : Kernel("MatMul", "matrix_multiplication"),
      strategy_("auto"), block_m_(256), block_n_(256), block_k_(256) {}

bool MatMulKernel::execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                           std::vector<std::shared_ptr<Tensor>> &outputs)
{
    if (!validate(inputs, outputs))
    {
        return false;
    }

    auto A = inputs[0];
    auto B = inputs[1];
    auto C = outputs[0];

    // Record execution start
    auto start = std::chrono::high_resolution_clock::now();

    bool success = executeCOSMA(*A, *B, *C);

    // Record execution time
    auto end = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end - start).count();
    recordExecutionTime(execution_time);

    if (success)
    {
        std::cout << "MatMul kernel executed successfully in "
                  << execution_time << " ms" << std::endl;
    }

    return success;
}

bool MatMulKernel::validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                            const std::vector<std::shared_ptr<Tensor>> &outputs) const
{
    // Check input count
    if (inputs.size() != 2)
    {
        std::cerr << "MatMul kernel requires exactly 2 inputs (A, B)" << std::endl;
        return false;
    }

    // Check output count
    if (outputs.size() != 1)
    {
        std::cerr << "MatMul kernel requires exactly 1 output (C)" << std::endl;
        return false;
    }

    auto A = inputs[0];
    auto B = inputs[1];
    auto C = outputs[0];

    // Check tensor rank (must be 2D matrices)
    if (A->getRank() != 2 || B->getRank() != 2 || C->getRank() != 2)
    {
        std::cerr << "MatMul kernel requires 2D tensors (matrices)" << std::endl;
        return false;
    }

    // Check matrix dimensions for compatibility (A: m×k, B: k×n, C: m×n)
    int m = A->getRows();
    int k_a = A->getCols();
    int k_b = B->getRows();
    int n = B->getCols();
    int m_c = C->getRows();
    int n_c = C->getCols();

    if (k_a != k_b)
    {
        std::cerr << "MatMul dimension mismatch: A cols (" << k_a
                  << ") != B rows (" << k_b << ")" << std::endl;
        return false;
    }

    if (m != m_c || n != n_c)
    {
        std::cerr << "MatMul dimension mismatch: C(" << m_c << "×" << n_c
                  << ") should be (" << m << "×" << n << ")" << std::endl;
        return false;
    }

    return true;
}

bool MatMulKernel::executeCOSMA(const Tensor &A, const Tensor &B, Tensor &C)
{
    try
    {
        // Get MPI context
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Matrix dimensions
        int m = A.getRows();
        int n = B.getCols();
        int k = A.getCols();

        if (rank == 0)
        {
            std::cout << "COSMA MatMul: " << m << "×" << k << " × " << k << "×" << n
                      << " → " << m << "×" << n << " with " << size << " processes" << std::endl;
        }

        // Create COSMA strategy
        cosma::Strategy strategy(m, n, k, size);

        // Create COSMA matrices
        cosma::CosmaMatrix<double> cosma_A('A', strategy, rank);
        cosma::CosmaMatrix<double> cosma_B('B', strategy, rank);
        cosma::CosmaMatrix<double> cosma_C('C', strategy, rank);

        // Copy data to COSMA matrices (local portions)
        // Note: In a real implementation, we'd need to properly distribute the data
        // For now, we'll do a simple copy assuming the data is already distributed
        const auto &data_A = A.getData();
        const auto &data_B = B.getData();

        // Get local matrix pointers
        auto local_A = cosma_A.matrix_pointer();
        auto local_B = cosma_B.matrix_pointer();
        auto local_C = cosma_C.matrix_pointer();

        // Fill local portions with data (simplified - assumes data is pre-distributed)
        size_t local_size_A = cosma_A.matrix_size();
        size_t local_size_B = cosma_B.matrix_size();
        if (local_size_A <= data_A.size())
        {
            std::copy(data_A.begin(), data_A.begin() + local_size_A, local_A);
        }
        if (local_size_B <= data_B.size())
        {
            std::copy(data_B.begin(), data_B.begin() + local_size_B, local_B);
        }

        // Perform COSMA multiplication: C = A × B
        cosma::multiply(cosma_A, cosma_B, cosma_C, strategy, MPI_COMM_WORLD, 1.0, 0.0);

        // Copy result back to output tensor
        auto &result_data = C.getData();
        size_t local_size_C = cosma_C.matrix_size();
        if (result_data.size() >= local_size_C)
        {
            std::copy(local_C, local_C + local_size_C, result_data.begin());
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "COSMA execution failed: " << e.what() << std::endl;
        return false;
    }
}

void MatMulKernel::initializeCOSMAContext()
{
    // Initialize COSMA context if needed
    // This could include setting up memory pools, communication patterns, etc.
    std::cout << "Initializing COSMA context for MatMul kernel" << std::endl;
}

// Register the kernel with the kernel manager
namespace
{
    struct MatMulKernelRegistrar
    {
        MatMulKernelRegistrar()
        {
            REGISTER_KERNEL("matrix_multiplication", MatMulKernel);
            REGISTER_KERNEL("matmul", MatMulKernel);
            REGISTER_KERNEL("mm", MatMulKernel);
        }
    };
    static MatMulKernelRegistrar registrar;
}
#include "kernels/MPILinearKernel.h"
#include "kernels/LinearKernel.h"
#include "tensors/tensor_factory.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <random>

using namespace llaminar;

class MPILinearKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI if not already done
        int flag;
        MPI_Initialized(&flag);
        if (!flag)
        {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        }

        // Set up random number generator for reproducible tests
        generator.seed(42);
    }

    void TearDown() override
    {
        // Note: Don't finalize MPI here as it might be used by other tests
    }

    void fillRandomData(std::shared_ptr<TensorBase> &tensor, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < tensor->data.size(); ++i)
        {
            tensor->data[i] = dist(generator);
        }
    }

    std::shared_ptr<TensorBase> createTensor(const std::vector<size_t> &shape)
    {
        auto tensor = llaminar::llaminar::TensorFactory::create_simple();

        // Convert size_t to int for shape
        tensor->shape.reserve(shape.size());
        for (const auto &dim : shape)
        {
            tensor->shape.push_back(static_cast<int>(dim));
        }

        size_t total_size = 1;
        for (const auto &dim : shape)
        {
            total_size *= dim;
        }

        tensor->data.resize(total_size, 0.0f);
        return tensor;
    }

    std::mt19937 generator;
};

TEST_F(MPILinearKernelTest, BasicFunctionality)
{
    MPILinearKernel mpi_kernel;

    // Test dimensions
    size_t seq_len = 4;
    size_t input_size = 6;
    size_t output_size = 8;

    // Create input tensors
    auto input = createTensor({seq_len, input_size});
    auto weight = createTensor({input_size, output_size});
    auto bias = createTensor({output_size});
    auto output = createTensor({seq_len, output_size});

    // Fill with test data
    fillRandomData(input);
    fillRandomData(weight);
    fillRandomData(bias);

    // Execute MPI kernel
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    EXPECT_TRUE(mpi_kernel.validate(inputs, outputs));
    EXPECT_TRUE(mpi_kernel.execute(inputs, outputs));

    // Check that output has expected values (non-zero for random input)
    bool has_nonzero = false;
    for (const auto &val : output->data)
    {
        if (std::abs(val) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(MPILinearKernelTest, WithoutBias)
{
    MPILinearKernel mpi_kernel;

    // Test dimensions
    size_t seq_len = 3;
    size_t input_size = 4;
    size_t output_size = 5;

    // Create input tensors (without bias)
    auto input = createTensor({seq_len, input_size});
    auto weight = createTensor({input_size, output_size});
    auto output = createTensor({seq_len, output_size});

    // Fill with test data
    fillRandomData(input);
    fillRandomData(weight);

    // Execute MPI kernel
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    EXPECT_TRUE(mpi_kernel.validate(inputs, outputs));
    EXPECT_TRUE(mpi_kernel.execute(inputs, outputs));
}

TEST_F(MPILinearKernelTest, ValidationTests)
{
    MPILinearKernel mpi_kernel;

    size_t seq_len = 2;
    size_t input_size = 3;
    size_t output_size = 4;

    auto input = createTensor({seq_len, input_size});
    auto weight = createTensor({input_size, output_size});
    auto output = createTensor({seq_len, output_size});

    // Test valid case
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_TRUE(mpi_kernel.validate(inputs, outputs));
    }

    // Test dimension mismatch
    {
        auto bad_weight = createTensor({input_size + 1, output_size});
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, bad_weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_FALSE(mpi_kernel.validate(inputs, outputs));
    }

    // Test wrong output size
    {
        auto bad_output = createTensor({seq_len, output_size + 1});
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {bad_output};
        EXPECT_FALSE(mpi_kernel.validate(inputs, outputs));
    }
}

TEST_F(MPILinearKernelTest, CompareWithSequential)
{
    // Only run this test if we have a small number of processes to avoid complexity
    MPILinearKernel mpi_kernel;
    if (mpi_kernel.getSize() > 4)
    {
        GTEST_SKIP() << "Skipping comparison test with large number of processes";
    }

    LinearKernel sequential_kernel;

    // Test dimensions - use smaller sizes for easier debugging
    size_t seq_len = 2;
    size_t input_size = 3;
    size_t output_size = 4;

    // Create tensors with deterministic data
    auto input = createTensor({seq_len, input_size});
    auto weight = createTensor({input_size, output_size});
    auto bias = createTensor({output_size});

    // Fill with simple test data for easier verification
    for (size_t i = 0; i < input->data.size(); ++i)
    {
        input->data[i] = static_cast<float>(i + 1);
    }
    auto weight_data = const_cast<float *>(weight->data());
    auto weight_size = weight->shape()[0] * weight->shape()[1];
    for (size_t i = 0; i < weight_size; ++i)
    {
        weight_data[i] = static_cast<float>((i % 3) + 1) * 0.1f;
    }
    auto bias_data = const_cast<float *>(bias->data());
    auto bias_size = bias->shape()[0];
    for (size_t i = 0; i < bias_size; ++i)
    {
        bias_data[i] = static_cast<float>(i + 1) * 0.01f;
    }

    // Execute sequential kernel
    auto sequential_output = createTensor({seq_len, output_size});
    std::vector<std::shared_ptr<TensorBase>> seq_inputs = {input, weight, bias};
    std::vector<std::shared_ptr<TensorBase>> seq_outputs = {sequential_output};

    EXPECT_TRUE(sequential_kernel.execute(seq_inputs, seq_outputs));

    // Execute MPI kernel
    auto mpi_output = createTensor({seq_len, output_size});
    std::vector<std::shared_ptr<TensorBase>> mpi_inputs = {input, weight, bias};
    std::vector<std::shared_ptr<TensorBase>> mpi_outputs = {mpi_output};

    EXPECT_TRUE(mpi_kernel.execute(mpi_inputs, mpi_outputs));

    // Compare results (allowing for small floating point differences)
    const float tolerance = 1e-5f;
    for (size_t i = 0; i < sequential_output->data.size(); ++i)
    {
        EXPECT_NEAR(sequential_output->data[i], mpi_output->data[i], tolerance)
            << "Mismatch at position " << i;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI for testing
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
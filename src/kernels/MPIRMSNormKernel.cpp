#include "MPIRMSNormKernel.h"
#include "../logger.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mpi.h>

namespace llaminar
{

    MPIRMSNormKernel::MPIRMSNormKernel(DistributionStrategy strategy)
        : strategy_(strategy), epsilon_(1e-6f)
    {
        LOG_DEBUG("MPIRMSNormKernel initialized with epsilon=" << epsilon_ << " on rank " << getRank());
    }

    bool MPIRMSNormKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                   std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIRMSNormKernel validation failed on rank " << getRank());
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto global_input = inputs[0];
        auto weight = inputs[1];
        auto global_output = outputs[0];

        // Extract dimensions
        size_t global_seq_len = static_cast<size_t>(global_input->shape()[0]);
        size_t hidden_size = static_cast<size_t>(global_input->shape()[1]);

        // Calculate local distribution
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        // Create local tensors
        auto local_input = createLocalTensor({static_cast<size_t>(local_seq_len), hidden_size});
        auto local_output = createLocalTensor({static_cast<size_t>(local_seq_len), hidden_size});

        // Distribute input data
        distributeInput(global_input, local_input, global_seq_len, hidden_size);

        // Compute distributed RMS normalization
        computeDistributedRMSNorm(local_input->data(), weight->data(),
                                  local_output->data(), local_seq_len,
                                  hidden_size, global_seq_len);

        // Gather results back to global output
        gatherOutput(local_output, global_output, global_seq_len, hidden_size);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        LOG_DEBUG("MPIRMSNormKernel executed successfully on rank " << getRank()
                                                                    << " in " << duration.count() << " microseconds");

        return true;
    }

    bool MPIRMSNormKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                    const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Basic validation
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            LOG_ERROR("MPIRMSNormKernel: Expected 2 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("MPIRMSNormKernel: Null tensor provided");
            return false;
        }

        // Check input is 2D [seq_len, hidden_size]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIRMSNormKernel: Input must be 2D [seq_len, hidden_size], got "
                      << input->shape().size() << "D");
            return false;
        }

        // Check weight is 1D [hidden_size]
        if (weight->shape().size() != 1)
        {
            LOG_ERROR("MPIRMSNormKernel: Weight must be 1D [hidden_size], got "
                      << weight->shape().size() << "D");
            return false;
        }

        // Check output is 2D [seq_len, hidden_size]
        if (output->shape().size() != 2)
        {
            LOG_ERROR("MPIRMSNormKernel: Output must be 2D [seq_len, hidden_size], got "
                      << output->shape().size() << "D");
            return false;
        }

        // Check dimension compatibility
        if (input->shape()[1] != weight->shape()[0])
        {
            LOG_ERROR("MPIRMSNormKernel: Dimension mismatch between input hidden_size ("
                      << input->shape()[1] << ") and weight (" << weight->shape()[0] << ")");
            return false;
        }

        // Check input and output have same shape
        if (input->shape()[0] != output->shape()[0] || input->shape()[1] != output->shape()[1])
        {
            LOG_ERROR("MPIRMSNormKernel: Input and output shape mismatch");
            return false;
        }

        return true;
    }

    void MPIRMSNormKernel::distributeInput(const std::shared_ptr<TensorBase> &global_input,
                                           std::shared_ptr<TensorBase> &local_input,
                                           size_t global_seq_len, size_t hidden_size)
    {
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        const float *global_data = global_input->data();
        float *local_data = local_input->data();

        // Copy the local portion of the input
        for (size_t i = 0; i < local_seq_len; ++i)
        {
            size_t global_row = seq_offset + i;
            std::memcpy(local_data + i * hidden_size,
                        global_data + global_row * hidden_size,
                        hidden_size * sizeof(float));
        }

        LOG_DEBUG("Distributed input: local size [" << local_seq_len << ", " << hidden_size
                                                    << "], offset " << seq_offset << " on rank " << getRank());
    }

    void MPIRMSNormKernel::gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                                        std::shared_ptr<TensorBase> &global_output,
                                        size_t global_seq_len, size_t hidden_size)
    {
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        // Prepare MPI gather parameters
        std::vector<int> recv_counts(getSize());
        std::vector<int> recv_offsets(getSize());

        for (int rank = 0; rank < getSize(); ++rank)
        {
            auto [rank_local_seq_len, rank_seq_offset] = getRowDistribution(global_seq_len, rank);
            recv_counts[rank] = static_cast<int>(rank_local_seq_len * hidden_size);
            recv_offsets[rank] = static_cast<int>(rank_seq_offset * hidden_size);
        }

        checkMPIError(MPI_Allgatherv(local_output->data(),
                                     static_cast<int>(local_seq_len * hidden_size), MPI_FLOAT,
                                     global_output->data(),
                                     recv_counts.data(), recv_offsets.data(), MPI_FLOAT,
                                     getComm()),
                      "MPI_Allgatherv in gatherOutput");

        LOG_DEBUG("Gathered output: [" << global_seq_len << ", " << hidden_size << "] on rank " << getRank());
    }

    void MPIRMSNormKernel::computeDistributedRMSNorm(const float *local_input, const float *weight,
                                                     float *local_output, size_t local_seq_len,
                                                     size_t hidden_size, size_t global_seq_len)
    {
        // Compute local contribution to global RMS
        float local_sum_sq = 0.0f;
        for (size_t i = 0; i < local_seq_len; ++i)
        {
            for (size_t j = 0; j < hidden_size; ++j)
            {
                float val = local_input[i * hidden_size + j];
                local_sum_sq += val * val;
            }
        }

        // Compute global RMS via MPI reduction
        float global_sum_sq = 0.0f;
        checkMPIError(MPI_Allreduce(&local_sum_sq, &global_sum_sq, 1, MPI_FLOAT, MPI_SUM, getComm()),
                      "MPI_Allreduce for RMS computation");

        float rms = std::sqrt(global_sum_sq / (global_seq_len * hidden_size) + epsilon_);

        // Apply normalization to local data
        for (size_t i = 0; i < local_seq_len; ++i)
        {
            for (size_t j = 0; j < hidden_size; ++j)
            {
                size_t idx = i * hidden_size + j;
                local_output[idx] = (local_input[idx] / rms) * weight[j];
            }
        }

        LOG_DEBUG("Computed distributed RMS normalization: rms=" << rms
                                                                 << ", local_seq_len=" << local_seq_len << " on rank " << getRank());
    }

    std::shared_ptr<TensorBase> MPIRMSNormKernel::createLocalTensor(const std::vector<size_t> &shape)
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());
        // Use TensorFactory to create a modern tensor
        return TensorFactory::create_simple(int_shape);
    }

} // namespace llaminar
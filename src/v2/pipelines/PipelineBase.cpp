/**
 * @file PipelineBase.cpp
 * @brief Base class implementation for transformer pipelines
 *
 * @author David Sanftenberg
 */

#include "PipelineBase.h"
#include <iostream>

namespace llaminar2
{

    PipelineBase::PipelineBase(const std::string &model_path,
                               std::shared_ptr<MPIContext> mpi_ctx,
                               int device_idx)
        : mpi_ctx_(mpi_ctx), device_idx_(device_idx), model_path_(model_path)
    {
        std::cout << "[PipelineBase] Initializing with model: " << model_path << "\n";

        if (mpi_ctx_)
        {
            std::cout << "[PipelineBase] MPI context provided, rank "
                      << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size() << "\n";
        }

        std::cout << "[PipelineBase] Device index: " << device_idx_
                  << (device_idx_ == -1 ? " (CPU)" : " (GPU)") << "\n";
    }

} // namespace llaminar2

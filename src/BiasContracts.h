/**
 * @file BiasContracts.h
 * @brief Lightweight bias dimension validation system
 *
 * This file provides a simple contract-based validation system for bias tensors.
 * Unlike the weight contract system (which handles complex GGUF parsing, multiple
 * slicing strategies, and 2D matrix validation), bias contracts focus solely on
 * dimension validation after manual pre-slicing.
 *
 * @author David Sanftenberg
 * @date 2025-10-12
 */

#pragma once

#include <string>
#include <memory>
#include <sstream>
#include "logger.h"
#include "tensors/tensor_base.h"

namespace llaminar
{

    /**
     * @brief Lightweight contract for validating bias tensor dimensions
     *
     * BiasContract provides dimension validation for 1D bias vectors that are
     * manually loaded and pre-sliced in the pipeline. This is intentionally
     * separate from WeightShapeContract because:
     *
     * 1. Biases are 1D, weights are 2D
     * 2. Biases are manually pre-sliced, weights use contract-based slicing helpers
     * 3. Bias validation is simple dimension checking, weight validation is complex
     *
     * This struct is NOT part of an inheritance hierarchy - it's a standalone
     * validation utility to keep things simple.
     */
    struct BiasContract
    {
        std::string tensor_name_pattern; ///< Pattern like "blk.{layer}.attn_q.bias"
        std::string description;         ///< Human-readable description
        int expected_full_dim;           ///< Total dimension in GGUF (before MPI slicing)
        int expected_local_dim;          ///< Expected dimension after MPI slicing
        int rank;                        ///< MPI rank for this validation
        int world_size;                  ///< MPI world size

        /**
         * @brief Construct a bias contract
         *
         * @param pattern Tensor name pattern (e.g., "blk.{layer}.attn_q.bias")
         * @param desc Human-readable description
         * @param full_dim Total dimension in GGUF
         * @param local_dim Expected local dimension after MPI slicing
         * @param mpi_rank MPI rank for validation
         * @param mpi_size MPI world size
         */
        BiasContract(const std::string &pattern,
                     const std::string &desc,
                     int full_dim,
                     int local_dim,
                     int mpi_rank,
                     int mpi_size)
            : tensor_name_pattern(pattern), description(desc), expected_full_dim(full_dim), expected_local_dim(local_dim), rank(mpi_rank), world_size(mpi_size)
        {
        }

        /**
         * @brief Validate a bias tensor's dimensions
         *
         * @param tensor The bias tensor to validate (after pre-slicing)
         * @param layer Layer number for error reporting
         * @param tensor_name Actual tensor name for error reporting
         *
         * @return true if dimensions match expectations, false otherwise
         */
        bool validate(const std::shared_ptr<TensorBase> &tensor,
                      int layer,
                      const std::string &tensor_name) const
        {
            if (!tensor)
            {
                LOG_ERROR("[BiasContract] Validation failed for " << tensor_name
                                                                  << " at layer " << layer << ": tensor is null");
                return false;
            }

            const size_t actual_size = tensor->size();

            // Check against expected local dimension
            if (actual_size != static_cast<size_t>(expected_local_dim))
            {
                LOG_ERROR("[BiasContract] Dimension mismatch for " << tensor_name
                                                                   << " at layer " << layer);
                LOG_ERROR("  Description: " << description);
                LOG_ERROR("  Expected local dim (rank " << rank << "/" << world_size
                                                        << "): " << expected_local_dim);
                LOG_ERROR("  Actual dim: " << actual_size);
                LOG_ERROR("  Expected full dim (in GGUF): " << expected_full_dim);
                return false;
            }

            LOG_DEBUG("[BiasContract] ✓ Validated " << tensor_name << " at layer " << layer
                                                    << ": dim=" << actual_size << " (expected " << expected_local_dim << ")");

            return true;
        }

        /**
         * @brief Validate a full (un-sliced) bias tensor before slicing
         *
         * This is useful to validate dimensions before performing MPI slicing.
         *
         * @param tensor The full bias tensor (before pre-slicing)
         * @param layer Layer number for error reporting
         * @param tensor_name Actual tensor name for error reporting
         *
         * @return true if dimensions match expected full size, false otherwise
         */
        bool validate_full(const std::shared_ptr<TensorBase> &tensor,
                           int layer,
                           const std::string &tensor_name) const
        {
            if (!tensor)
            {
                LOG_ERROR("[BiasContract] Full validation failed for " << tensor_name
                                                                       << " at layer " << layer << ": tensor is null");
                return false;
            }

            const size_t actual_size = tensor->size();

            if (actual_size != static_cast<size_t>(expected_full_dim))
            {
                LOG_ERROR("[BiasContract] Full dimension mismatch for " << tensor_name
                                                                        << " at layer " << layer);
                LOG_ERROR("  Description: " << description);
                LOG_ERROR("  Expected full dim (in GGUF): " << expected_full_dim);
                LOG_ERROR("  Actual dim: " << actual_size);
                return false;
            }

            LOG_DEBUG("[BiasContract] ✓ Validated full " << tensor_name << " at layer "
                                                         << layer << ": dim=" << actual_size);

            return true;
        }

        /**
         * @brief Get the expected slice range for this rank
         *
         * @return std::pair<int, int> (offset, length) for this rank's slice
         */
        std::pair<int, int> get_slice_range() const
        {
            // Use same distribution logic as QwenPipeline head distribution
            int per_rank = expected_full_dim / world_size;
            int remainder = expected_full_dim % world_size;

            int local_size = per_rank + (rank < remainder ? 1 : 0);
            int offset = rank * per_rank + std::min(rank, remainder);

            return {offset, local_size};
        }
    };

} // namespace llaminar

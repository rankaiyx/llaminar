/**
 * @file mpi_slicing_helper.h
 * @brief MPI-aware tensor loading with automatic dimension handling and slicing
 *
 * This file provides centralized logic for loading tensors with MPI slicing,
 * handling all complexity of:
 * - GGUF vs PyTorch dimension conventions
 * - Row vs column slicing based on weight contracts
 * - Dimension transposition when needed
 * - Offset calculation in original GGUF layout
 *
 * This eliminates duplicate slicing logic across the codebase and ensures
 * correct MPI distribution of weights.
 *
 * @author David Sanftenberg
 * @date 2025-01-12
 */

#pragma once

#include "model_loader.h"
#include "weight_contracts.h"
#include "tensors/tensor_base.h"
#include "tensors/tensor_factory.h"
#include <memory>
#include <string>
#include <vector>

namespace mpi_slicing
{

    /**
     * @brief Parameters for MPI-aware tensor slicing
     *
     * This structure contains all information needed to:
     * 1. Load the correct slice from GGUF file
     * 2. Optionally transpose the data
     * 3. Return tensor in expected PyTorch layout
     */
    struct SliceParams
    {
        int offset;                      ///< Starting index in GGUF layout
        int count;                       ///< Number of elements to load in GGUF layout
        bool is_column_slice;            ///< True for column slice, false for row slice
        std::vector<int> expected_shape; ///< Final shape after loading (PyTorch convention)
        bool needs_transpose;            ///< Whether to transpose data after loading

        SliceParams()
            : offset(0), count(0), is_column_slice(false), needs_transpose(false) {}
    };

    /**
     * @brief Calculate MPI slice parameters from weight contract
     *
     * Given a weight contract and MPI context, determines:
     * - Which slice this rank should load (offset, count)
     * - Whether it's a row or column slice
     * - Whether data needs transposition
     * - Expected final shape
     *
     * @param contract Weight shape contract defining slicing strategy
     * @param cfg Transformer layer configuration (defines n_head, d_model, etc.)
     * @param mpi_rank Current MPI rank
     * @param mpi_size Total number of MPI ranks
     * @return SliceParams containing all information needed for loading
     *
     * @note This function handles the complexity of dimension conventions:
     *       - GGUF stores weights as [in_features, out_features]
     *       - PyTorch expects [out_features, in_features]
     *       - Slicing in PyTorch convention may require row or column slicing in GGUF
     */
    SliceParams calculate_slice(const llaminar::WeightShapeContract &contract,
                                const TransformerLayerConfig &cfg,
                                int mpi_rank,
                                int mpi_size);

    /**
     * @brief Load tensor with automatic MPI slicing based on contract
     *
     * This is the PRIMARY FUNCTION for loading weights in an MPI context.
     * It handles all complexity transparently:
     *
     * 1. Calculates correct slice for this rank using calculate_slice()
     * 2. Loads the appropriate bytes from GGUF file (row or column slice)
     * 3. Transposes data if needed to match PyTorch convention
     * 4. Validates the result against the contract
     * 5. Returns correctly-shaped, correctly-sliced tensor
     *
     * @param loader ModelLoader instance for accessing GGUF file
     * @param contract Weight shape contract defining expected dimensions and slicing
     * @param cfg Transformer layer configuration
     * @param mpi_rank Current MPI rank
     * @param mpi_size Total number of MPI ranks
     * @param layer_index Layer index for layer-specific tensors (-1 for global tensors)
     * @return Shared pointer to loaded tensor in correct PyTorch layout
     *
     * @throws std::runtime_error if loading fails or validation fails
     *
     * Usage example:
     * @code
     * auto wq = mpi_slicing::load_with_contract(
     *     loader, Q_WEIGHT_CONTRACT, config, mpi_rank, mpi_size, layer);
     * // wq is now correctly sliced and in PyTorch layout, ready to use!
     * @endcode
     */
    std::shared_ptr<llaminar::TensorBase> load_with_contract(
        ModelLoader &loader,
        const llaminar::WeightShapeContract &contract,
        const TransformerLayerConfig &cfg,
        int mpi_rank,
        int mpi_size,
        int layer_index = -1);

} // namespace mpi_slicing

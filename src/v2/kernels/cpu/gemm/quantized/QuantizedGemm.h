/**
 * @file QuantizedGemm.h
 * @brief Factory function for creating auto-tuned quantized GEMM kernels
 *
 * Provides createQuantizedGemm() factory function that creates ITensorGemm
 * instances using the GemmAutoTuner infrastructure.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../../tensors/TensorKernels.h" // For ITensorGemm, ITensorGemmTileDataProvider
#include <memory>

namespace llaminar2
{
    /**
     * @brief Create auto-tuned quantized GEMM kernel
     *
     * Creates an ITensorGemm implementation that uses the GemmAutoTuner
     * to select optimal micro-kernel variants for the given tensor.
     *
     * @param decoder ITensorGemmTileDataProvider for quantized weight access
     * @return Unique pointer to auto-tuned GEMM kernel implementation
     */
    std::unique_ptr<ITensorGemm> createQuantizedGemm(const ITensorGemmTileDataProvider *decoder);

} // namespace llaminar2

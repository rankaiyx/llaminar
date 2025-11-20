/**
 * @file SwiGLUPrimitives.h
 * @brief SwiGLU primitives (scalar and vectorized)
 * @author GitHub Copilot
 */

#pragma once

namespace llaminar2::primitives
{
    /**
     * @brief Compute SwiGLU: output = gate * silu(up)
     *
     * @param gate Pointer to gate tensor data
     * @param up Pointer to up tensor data
     * @param output Pointer to output tensor data
     * @param size Number of elements
     */
    void compute_swiglu(const float *gate, const float *up, float *output, int size);

} // namespace llaminar2::primitives

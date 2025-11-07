/**
 * @file INT8SwiGLUKernel.h
 * @brief INT8 SwiGLU activation kernel
 * @author David Sanftenberg
 * @date 2025-11-06
 *
 * SwiGLU (Swish-Gated Linear Unit):
 *   SwiGLU(x, gate, up) = gate × SiLU(up)
 *   where SiLU(x) = x × sigmoid(x)
 *
 * This kernel implements SwiGLU in INT8/FP32 hybrid arithmetic:
 *   1. Dequantize gate and up projections from INT8 to FP32
 *   2. Compute SiLU(up) in FP32 for numerical stability
 *   3. Multiply gate × SiLU(up) in FP32
 *   4. Requantize output to INT8
 *
 * @example
 * @code
 *   // Create kernel
 *   INT8SwiGLUKernel swiglu;
 *
 *   // Prepare INT8 inputs with row scales
 *   std::vector<int8_t> gate_int8(batch * seq_len * d_ff);
 *   std::vector<float> gate_row_scales(batch * seq_len);
 *   std::vector<int8_t> up_int8(batch * seq_len * d_ff);
 *   std::vector<float> up_row_scales(batch * seq_len);
 *
 *   // Compute SwiGLU
 *   std::vector<int8_t> output_int8(batch * seq_len * d_ff);
 *   std::vector<float> output_row_scales(batch * seq_len);
 *
 *   swiglu.forward(
 *       gate_int8.data(), gate_row_scales.data(),
 *       up_int8.data(), up_row_scales.data(),
 *       output_int8.data(), output_row_scales.data(),
 *       batch, seq_len, d_ff);
 * @endcode
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace llaminar2
{

    /**
     * @brief INT8 SwiGLU (Swish-Gated Linear Unit) kernel
     *
     * Implements: output = gate × SiLU(up) where SiLU(x) = x × sigmoid(x)
     *
     * Computation flow:
     *   1. Dequantize: gate_fp32 = gate_int8 × gate_scale
     *   2. Dequantize: up_fp32 = up_int8 × up_scale
     *   3. Compute: silu_up = up_fp32 × sigmoid(up_fp32)
     *   4. Multiply: output_fp32 = gate_fp32 × silu_up
     *   5. Quantize: output_int8 = quantize(output_fp32)
     */
    class INT8SwiGLUKernel
    {
    public:
        /**
         * @brief Construct SwiGLU kernel
         *
         * @param device_idx Device index (-1 for CPU)
         */
        explicit INT8SwiGLUKernel(int device_idx = -1);

        /**
         * @brief Forward pass: INT8 gate/up → FP32 SwiGLU → INT8 output
         *
         * @param gate_int8 Gate projection INT8 [batch, seq_len, d_ff]
         * @param gate_row_scales Gate scales [batch * seq_len]
         * @param up_int8 Up projection INT8 [batch, seq_len, d_ff]
         * @param up_row_scales Up scales [batch * seq_len]
         * @param output_int8 Output INT8 [batch, seq_len, d_ff] (OUT)
         * @param output_row_scales Output scales [batch * seq_len] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         * @param d_ff Feed-forward dimension
         * @return true if successful, false otherwise
         */
        bool forward(
            const int8_t *gate_int8,
            const float *gate_row_scales,
            const int8_t *up_int8,
            const float *up_row_scales,
            int8_t *output_int8,
            float *output_row_scales,
            int batch,
            int seq_len,
            int d_ff);

        /**
         * @brief Get device index
         * @return Device index (-1 for CPU)
         */
        int device_idx() const { return device_idx_; }

    private:
        /**
         * @brief Compute sigmoid(x) with numerical stability
         *
         * Uses: sigmoid(x) = 1 / (1 + exp(-x))
         * For numerical stability:
         *   - If x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
         *   - If x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
         *
         * @param x Input value
         * @return sigmoid(x) in range [0, 1]
         */
        static float sigmoid(float x);

        /**
         * @brief Compute SiLU(x) = x × sigmoid(x)
         *
         * @param x Input value
         * @return SiLU(x)
         */
        static float silu(float x);

        int device_idx_; ///< Device index (-1 for CPU)

        // Temporary buffers (reused across forward calls)
        std::vector<float> gate_fp32_buffer_;   ///< [batch, seq_len, d_ff]
        std::vector<float> up_fp32_buffer_;     ///< [batch, seq_len, d_ff]
        std::vector<float> output_fp32_buffer_; ///< [batch, seq_len, d_ff]
    };

} // namespace llaminar2

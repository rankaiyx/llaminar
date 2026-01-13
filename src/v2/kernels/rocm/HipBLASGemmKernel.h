/**
 * @file HipBLASGemmKernel.h
 * @brief hipBLAS-based GEMM kernel for FP32/FP16/BF16 tensors
 *
 * **Purpose**: Provide GPU GEMM for non-quantized (floating-point) tensors using hipBLAS.
 *
 * **Design**:
 * - Standalone kernel class (does not inherit ITensorGemm to avoid MPI header dependency in HIP)
 * - Uses hipBLAS sgemm (FP32), hgemm (FP16), or emulated BF16 (via FP32 compute)
 * - Expects input/output matrices already on GPU device
 * - Caller responsible for device memory management
 *
 * **Usage**:
 * ```cpp
 * auto kernel = std::make_unique<HipBLASGemmKernel>(device_id);
 * kernel->execute(d_A, d_B, d_C, M, N, K, transA, transB);
 * ```
 *
 * **Note**: MI50 (gfx906) does NOT have native BF16 support. BF16 operations
 * are emulated by converting to FP32 for compute, then optionally back to BF16.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

#ifdef HAVE_ROCM
#include <hipblas/hipblas.h>
#endif

namespace llaminar2
{
    namespace rocm
    {

        /**
         * @brief hipBLAS-based GEMM kernel for floating-point tensors
         *
         * Supports:
         * - FP32: hipblasSgemm
         * - FP16: hipblasHgemm (on supported hardware)
         * - BF16: Emulated via FP32 compute (MI50 doesn't have native BF16)
         *
         * **Memory Requirements**:
         * - All pointers (A, B, C) must be in GPU device memory
         * - Caller manages device allocation/deallocation
         *
         * **Performance Notes**:
         * - MI50 (Vega 20, gfx906): 13.4 TFLOPS FP32, 26.5 TFLOPS FP16
         * - No Tensor Cores, but matrix FMA is well optimized
         * - BF16 requires conversion overhead (no hardware support)
         */
        class HipBLASGemmKernel
        {
        public:
            /**
             * @brief Supported precision modes
             */
            enum class Precision
            {
                FP32, ///< 32-bit float (hipblasSgemm)
                FP16, ///< 16-bit float (hipblasHgemm)
                BF16  ///< BFloat16 (emulated via FP32 - no native MI50 support)
            };

            /**
             * @brief Create hipBLAS GEMM kernel
             *
             * @param device_id HIP device ID (from hipGetDevice)
             * @param precision Floating-point precision to use
             *
             * @throws std::runtime_error if hipBLAS handle creation fails
             */
            explicit HipBLASGemmKernel(int device_id, Precision precision = Precision::FP32);

            /**
             * @brief Destructor - destroys hipBLAS handle
             */
            ~HipBLASGemmKernel();

            // Non-copyable
            HipBLASGemmKernel(const HipBLASGemmKernel &) = delete;
            HipBLASGemmKernel &operator=(const HipBLASGemmKernel &) = delete;

            // Movable
            HipBLASGemmKernel(HipBLASGemmKernel &&other) noexcept;
            HipBLASGemmKernel &operator=(HipBLASGemmKernel &&other) noexcept;

            // =================================================================
            // Device Query
            // =================================================================

            bool supports_device(int device_idx) const
            {
                return device_idx >= 0; // Supports any GPU device
            }

            // =================================================================
            // GEMM Operations
            // =================================================================

            /**
             * @brief GPU GEMM: C = alpha * A @ B + beta * C
             *
             * **Layout Convention**:
             * - A: Row-major [M × K]
             * - B: Row-major [N × K] if transpose_B=true, [K × N] otherwise
             * - C: Row-major [M × N]
             *
             * hipBLAS uses column-major, so we transpose the operation:
             * C^T = B^T @ A^T  (which gives C in row-major)
             *
             * @param d_A Device pointer to A [M × K]
             * @param d_B Device pointer to B [N × K] or [K × N]
             * @param d_C Device pointer to C [M × N]
             * @param M Rows in A and C
             * @param N Columns in C (rows in B if transposed)
             * @param K Columns in A (and B's inner dimension)
             * @param transA Transpose A (typically false)
             * @param transB Transpose B (typically true for weights)
             * @param alpha Scale for A@B
             * @param beta Scale for C
             *
             * @return true on success
             */
            bool execute(
                const float *d_A, const float *d_B, float *d_C,
                int M, int N, int K,
                bool transA = false, bool transB = false,
                float alpha = 1.0f, float beta = 0.0f);

            /**
             * @brief FP16 GEMM: C = alpha * A @ B + beta * C
             *
             * Uses hipblasHgemm for native FP16 computation.
             */
            bool execute_fp16(
                const void *d_A, const void *d_B, void *d_C,
                int M, int N, int K,
                bool transA = false, bool transB = false,
                float alpha = 1.0f, float beta = 0.0f);

            // Getters
            int device_id() const { return device_id_; }
            Precision precision() const { return precision_; }

        private:
#ifdef HAVE_ROCM
            hipblasHandle_t handle_ = nullptr;
#endif
            int device_id_ = 0;
            Precision precision_ = Precision::FP32;
        };

        /**
         * @brief Factory function for hipBLAS GEMM kernel
         */
        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(
            int device_id,
            HipBLASGemmKernel::Precision precision = HipBLASGemmKernel::Precision::FP32);

    } // namespace rocm
} // namespace llaminar2

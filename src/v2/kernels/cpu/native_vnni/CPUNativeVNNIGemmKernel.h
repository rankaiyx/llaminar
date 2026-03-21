/**
 * @file CPUNativeVNNIGemmKernel.h
 * @brief ITensorGemm implementation for CPU NativeVNNI GEMM/GEMV.
 *
 * This kernel keeps weights in their native quantized format (Q4_0, IQ4_NL, etc.)
 * and decodes blocks inline during computation using AVX-512 VNNI (vpdpbusd).
 *
 * ## Comparison with CPUQuantisedGemmKernel
 *
 * | Aspect | CPUQuantisedGemmKernel | CPUNativeVNNIGemmKernel |
 * |--------|--------------------|-----------------------|
 * | Weight packing | Decode to INT8 at pack time | Keep native bytes, decode at runtime |
 * | Weight memory | 1 byte/element | 0.5 byte/element (Q4_0) |
 * | GEMV bandwidth | 2× memory traffic | 1× memory traffic |
 * | Decode cost | Zero (pre-decoded) | Small (nibble unpack) |
 * | Best for | M>1 (compute-bound) | M=1 (memory-bound GEMV) |
 *
 * ## Supported Formats (Phase 1)
 *
 * - Q4_0: Simple symmetric 4-bit (16 byte payload / 32 elements)
 * - IQ4_NL: Non-linear 4-bit with LUT (16 byte payload / 32 elements)
 *
 * Additional formats can be added by implementing decode_native_block() cases
 * in CPUNativeVNNIGemv.h.
 */

#pragma once

#include "CPUNativeVNNIWeightPacker.h"
#include "CPUNativeVNNIGemv.h"
#include "tensors/TensorKernels.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"

namespace llaminar2::cpu::native_vnni
{

    class CPUNativeVNNIGemmKernel : public ITensorGemm
    {
    public:
        /**
         * @brief Construct from a quantized weight tensor.
         *
         * Packs weights into the CPU NativeVNNI layout at construction time.
         * The tensor must implement IINT8Unpackable and provide vnniFormatInfo().
         *
         * @param weights Source weight tensor [N, K]
         * @param row_start Start row for TP slicing (default 0)
         * @param row_end End row for TP slicing (default -1 = all)
         */
        explicit CPUNativeVNNIGemmKernel(const TensorBase *weights,
                                         int row_start = 0, int row_end = -1)
        {
            if (!packWeightsCPUNativeVNNI(weights, packed_, row_start, row_end))
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Failed to pack weights");
                valid_ = false;
                return;
            }
            valid_ = true;
            LOG_DEBUG("[CPUNativeVNNIGemmKernel] Packed "
                      << packed_.N << "×" << packed_.K
                      << " weights (codebook=" << (int)packed_.codebook_id
                      << ", payload=" << packed_.payload_bytes << " B/block"
                      << ", asymmetric=" << packed_.is_asymmetric << ")");
        }

        /**
         * @brief Construct from pre-packed weights (move).
         */
        explicit CPUNativeVNNIGemmKernel(CPUNativeVNNIPackedWeights &&packed)
            : packed_(std::move(packed)), valid_(true) {}

        ~CPUNativeVNNIGemmKernel() override = default;

        // -------------------------------------------------------------------
        // ITensorKernel interface
        // -------------------------------------------------------------------

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        // -------------------------------------------------------------------
        // ITensorGemm interface
        // -------------------------------------------------------------------

        /**
         * @brief C[m×n] = A[m×k] @ B_packed[n×k]^T
         *
         * For M=1: dispatches to optimized GEMV path.
         * For M>1: dispatches to row-by-row GEMV (phase 1; tiled GEMM is phase 2).
         */
        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)transpose_B;
            (void)mpi_ctx;
            (void)workspace;

            if (!valid_ || device_idx != -1)
                return false;

            if (n > packed_.N || k > packed_.K)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Dimension mismatch: "
                          << "requested n=" << n << " k=" << k
                          << " packed N=" << packed_.N << " K=" << packed_.K);
                return false;
            }

            // Handle beta scaling of existing C
            if (beta != 0.0f && beta != 1.0f)
            {
                for (int i = 0; i < m * n; ++i)
                    C[i] *= beta;
            }
            else if (beta == 0.0f)
            {
                // GEMM will overwrite C completely
            }

            if (m == 1)
            {
                // Optimized GEMV path
                if (beta == 0.0f && alpha == 1.0f)
                {
                    gemv_native_vnni(packed_, A, C);
                }
                else
                {
                    // General case: C = alpha * A@B + beta * C
                    // Use temp buffer for raw GEMV result, then blend
                    std::vector<float> temp(n);
                    gemv_native_vnni(packed_, A, temp.data());
                    for (int j = 0; j < n; ++j)
                        C[j] += alpha * temp[j];
                }
            }
            else
            {
                // M>1: row-by-row GEMV for now
                if (beta == 0.0f && alpha == 1.0f)
                {
                    gemm_native_vnni(packed_, A, C, m, n);
                }
                else
                {
                    std::vector<float> temp(n);
                    for (int row = 0; row < m; ++row)
                    {
                        gemv_native_vnni(packed_, A + row * k, temp.data());
                        for (int j = 0; j < n; ++j)
                            C[row * n + j] += alpha * temp[j];
                    }
                }
            }

            return true;
        }

        /**
         * @brief Tensor-aware multiply (delegates to raw multiply)
         */
        bool multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const TensorBase *bias = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr,
            int activation_row_offset = 0) override
        {
            (void)bias; // TODO: bias support

            const float *A_data = A->data() + activation_row_offset * k;
            float *C_data = C->mutable_data();

            return multiply(A_data, C_data, m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx, workspace);
        }

        /**
         * @brief Activation-activation multiply (not applicable for weight GEMM).
         */
        bool multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Not applicable for weight GEMM
        }

        /**
         * @brief Strided activation-activation multiply (not applicable for weight GEMM).
         */
        bool multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)A; (void)B; (void)C;
            (void)m; (void)n; (void)k;
            (void)lda; (void)ldb; (void)ldc;
            (void)transpose_B; (void)alpha; (void)beta;
            (void)mpi_ctx; (void)device_idx;
            return false; // Not applicable for weight GEMM
        }

        // -------------------------------------------------------------------
        // Accessors
        // -------------------------------------------------------------------

        bool isValid() const { return valid_; }

        const CPUNativeVNNIPackedWeights &packedWeights() const { return packed_; }

        uint8_t codebookId() const { return packed_.codebook_id; }

    private:
        CPUNativeVNNIPackedWeights packed_;
        bool valid_ = false;
    };

} // namespace llaminar2::cpu::native_vnni

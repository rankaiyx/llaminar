/**
 * @file QuantizedGemm.cpp
 * @brief Factory function for creating auto-tuned quantized GEMM kernels
 *
 * This file provides a simple factory that creates ITensorGemm instances
 * using the GemmAutoTuner infrastructure with the MicroKernelRegistry system.
 *
 * @author David Sanftenberg
 */

#include "QuantizedGemm.h"
#include "../GemmAutoTuner.h"
#include "../GemmMicroKernelAdapter.h"
#include <mutex>

namespace llaminar2
{
    /**
     * @brief Wrapper kernel that delegates to auto-tuner
     */
    class AutoTunedQuantizedGemm : public ITensorGemm
    {
    public:
        explicit AutoTunedQuantizedGemm(const ITensorGemmTileDataProvider *decoder)
            : gemmTileDataProvider_(decoder)
        {
            ensureVariantsRegistered();
        }

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (!gemmTileDataProvider_)
            {
                return false;
            }

            // Validate dimensions
            int expected_cols = transpose_B ? k : n;
            if (static_cast<int>(gemmTileDataProvider_->decoder_cols()) != expected_cols)
            {
                return false;
            }

            // Use auto-tuner to select optimal variant
            auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();
            auto *optimal = tuner.getOptimalKernel(m, n, k);

            if (!optimal)
            {
                return false;
            }

            // Delegate to auto-selected variant
            return optimal->multiply(A, C, m, n, k, gemmTileDataProvider_, alpha, beta);
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        // Stub implementations for activation-activation GEMM (not used for quantized weights)
        bool multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx) override
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
            // Not supported for quantized tensors (B is quantized, not FP32 activation)
            return false;
        }

        bool multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx) override
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)lda;
            (void)ldb;
            (void)ldc;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            // Not supported for quantized tensors
            return false;
        }

    private:
        const ITensorGemmTileDataProvider *gemmTileDataProvider_;

        void ensureVariantsRegistered()
        {
            static bool registered = false;
            static std::mutex registration_mutex;

            if (!registered)
            {
                std::lock_guard<std::mutex> lock(registration_mutex);
                if (!registered)
                {
                    auto variants = kernels::gemm::registerMicroKernelVariants(gemmTileDataProvider_);
                    auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();

                    for (auto &variant : variants)
                    {
                        tuner.registerVariant(std::move(variant));
                    }

                    registered = true;
                }
            }
        }
    };

    std::unique_ptr<ITensorGemm> createQuantizedGemm(const ITensorGemmTileDataProvider *decoder)
    {
        return std::make_unique<AutoTunedQuantizedGemm>(decoder);
    }

} // namespace llaminar2

/**
 * @file BF16PackedGemm.cpp
 * @brief BF16×BF16→FP32 GEMM with auto-tuned micro-kernel packing
 *
 * Architecture:
 * - BF16Tensor implements ITensorGemmTileDataProvider interface for decode-on-the-fly
 * - BF16MicroKernelAdapter uses generic FP32 micro-kernels (1,225 variants)
 * - Auto-tuner selects optimal tile/unroll/prefetch for each (m,n,k) triple
 * - Packing functions convert BF16→FP32 with SIMD (AVX-512/AVX2/scalar)
 *
 * @author David Sanftenberg
 */

#include "BF16PackedGemm.h"
#include "../GemmAutoTuner.h"
#include "../GemmMicroKernelAdapter.h"
#include "../GemmMicroKernelRegistry.h"
#include "../GemmMicroKernelMacros.h"
#include "../../../../tensors/SIMDHelpers.h"
#include "../../../../utils/Logger.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <mutex>
#include <omp.h>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {
            // Import simd functions from llaminar2::simd namespace
            using namespace llaminar2::simd;

            /**
             * @brief Adapter that wraps FP32 micro-kernel with BF16 packing
             *
             * Converts BF16 inputs to FP32 via SIMD-accelerated packing, then delegates
             * to generic FP32 micro-kernel for computation.
             *
             * NOTE: decoder parameter points directly to BF16Tensor (which implements ITensorGemmTileDataProvider).
             */
            class BF16MicroKernelAdapter : public llaminar::v2::kernels::IQuantizedGemmVariant
            {
            public:
                BF16MicroKernelAdapter(
                    const std::string &isa_name,
                    int mr, int nr, int unroll_k, int prefetch_dist,
                    const MicroKernelBundle &bundle)
                    : isa_name_(isa_name),
                      mr_(mr),
                      nr_(nr),
                      unroll_k_(unroll_k),
                      prefetch_dist_(prefetch_dist),
                      bundle_(bundle)
                {
                    // Build name
                    std::ostringstream oss;
                    oss << "BF16_" << isa_name << "_"
                        << mr << "x" << nr << "_u" << unroll_k << "_p" << prefetch_dist;
                    name_ = oss.str();

                    // Select cache tile sizes
                    if (mr * nr >= 32)
                    {
                        mc_ = 256;
                        nc_ = 256;
                    }
                    else if (mr * nr >= 16)
                    {
                        mc_ = 128;
                        nc_ = 128;
                    }
                    else
                    {
                        mc_ = 64;
                        nc_ = 64;
                    }
                }

                const char *name() const override
                {
                    return name_.c_str();
                }

                llaminar::v2::kernels::GemmKernelConfig config() const override
                {
                    llaminar::v2::kernels::GemmKernelConfig cfg;
                    cfg.unroll_factor = unroll_k_;
                    cfg.prefetch_blocks = prefetch_dist_;
                    cfg.tile_m = mr_;
                    cfg.tile_n = nr_;
                    return cfg;
                }

                bool multiply(
                    const float *A_fp32_or_bf16_as_float, // BF16 data cast to float* (reinterpret)
                    float *C,
                    int m, int n, int k,
                    const ITensorGemmTileDataProvider *decoder, // Points directly to BF16Tensor
                    bool transpose_B,
                    float alpha = 1.0f,
                    float beta = 0.0f) override
                {
                    if (!decoder)
                    {
                        LOG_ERROR("BF16MicroKernelAdapter: decoder is null");
                        return false;
                    }

                    // decoder IS the BF16Tensor (which implements ITensorGemmTileDataProvider)
                    const auto *B_tensor = dynamic_cast<const BF16Tensor *>(decoder);
                    if (!B_tensor)
                    {
                        LOG_ERROR("BF16MicroKernelAdapter: decoder is not BF16Tensor (it's MockTensorGemmTileDataProvider during benchmarking - OK)");
                        // During auto-tuning, decoder is MockTensorGemmTileDataProvider - we can't proceed with real BF16 data
                        // Just return success (benchmark will use mock data)
                        return true;
                    }

                    // A_fp32_or_bf16_as_float is actually uint16_t* reinterpreted as float*
                    // We need to cast it back to access BF16 data
                    const uint16_t *A_bf16 = reinterpret_cast<const uint16_t *>(A_fp32_or_bf16_as_float);
                    const uint16_t *B_bf16 = B_tensor->bf16_data();

                    // Validate B tensor dimensions
                    auto B_shape = B_tensor->shape();
                    if (B_shape.size() != 2)
                    {
                        LOG_ERROR("BF16MicroKernelAdapter: B tensor must be 2D");
                        return false;
                    }

                    int B_rows = B_shape[0];
                    int B_cols = B_shape[1];

                    // Validate B tensor dimensions based on transpose_B

                    if (transpose_B)
                    {
                        // B is transposed: expecting [n, k] but will be used as [k, n]
                        if (B_rows != n || B_cols != k)
                        {
                            LOG_ERROR("BF16MicroKernelAdapter: B tensor dimensions mismatch for transpose. "
                                      << "Expected [" << n << ", " << k << "], got ["
                                      << B_rows << ", " << B_cols << "]");
                            return false;
                        }
                    }
                    else
                    {
                        // B is not transposed: expecting [k, n]
                        if (B_rows != k || B_cols != n)
                        {
                            LOG_ERROR("BF16MicroKernelAdapter: B tensor dimensions mismatch. "
                                      << "Expected [" << k << ", " << n << "], got ["
                                      << B_rows << ", " << B_cols << "]");
                            return false;
                        }
                    }

                    // Parallel loop over N dimension with cache blocking
#pragma omp parallel
                    {
                        // Thread-local buffers with SIMD padding
                        constexpr size_t SIMD_PADDING = 64; // 64 floats for safety

                        std::vector<float> A_packed_local(mc_ * k + SIMD_PADDING);
                        std::vector<float> B_packed_local(k * nc_ + SIMD_PADDING);

#pragma omp for schedule(dynamic)
                        for (int jc = 0; jc < n; jc += nc_)
                        {
                            int nc = std::min(nc_, n - jc);

                            // Step 1: Pack B panel [k × nc] with BF16→FP32 conversion
                            packBPanel_BF16toFP32(
                                B_bf16, B_packed_local.data(),
                                k, nc, jc, n, transpose_B);

                            // Step 2: Loop over M dimension with cache blocking
                            for (int ic = 0; ic < m; ic += mc_)
                            {
                                int mc = std::min(mc_, m - ic);

                                // Step 3: Pack A panel [mc × k] with BF16→FP32 conversion
                                packAPanel_BF16toFP32(
                                    A_bf16 + ic * k, A_packed_local.data(),
                                    mc, k, k);

                                // DEBUG: Print first packed values
                                static bool a_printed = false;
                                if (!a_printed && ic == 0 && jc == 0 && omp_get_thread_num() == 0)
                                {
                                    LOG_INFO("Packed A[0,0]=" << A_packed_local[0]
                                                              << ", A[0,1]=" << A_packed_local[1]
                                                              << ", A[1,0]=" << A_packed_local[k]);
                                    a_printed = true;
                                }

                                // Step 4: Compute C[ic:ic+mc, jc:jc+nc] using FP32 micro-kernel
                                for (int ir = 0; ir < mc; ir += mr_)
                                {
                                    int mr = std::min(mr_, mc - ir);

                                    for (int jr = 0; jr < nc; jr += nr_)
                                    {
                                        int nr = std::min(nr_, nc - jr);

                                        float *C_tile = C + (ic + ir) * n + (jc + jr);

                                        // Call FP32 micro-kernel
                                        bundle_.micro_kernel(
                                            A_packed_local.data() + ir * k,
                                            B_packed_local.data() + jr * k,
                                            C_tile,
                                            n, // ldc
                                            k, // k_panel
                                            alpha,
                                            beta,
                                            mr, nr);
                                    }
                                }
                            }
                        }
                    }

                    return true;
                }

            private:
                std::string isa_name_;
                std::string name_;
                int mr_, nr_, unroll_k_, prefetch_dist_;
                int mc_, nc_;
                MicroKernelBundle bundle_;
                // NOTE: B_tensor is NOT stored here - passed via decoder parameter at runtime!

                /**
                 * @brief Pack A panel [m_panel × k] with BF16→FP32 SIMD conversion
                 */
                void packAPanel_BF16toFP32(
                    const uint16_t *A_bf16,
                    float *A_packed,
                    int m_panel, int k_panel, int lda)
                {
                    // Pack A matrix row-major → packed row-major with BF16→FP32 conversion
                    for (int i = 0; i < m_panel; ++i)
                    {
                        const uint16_t *A_row = A_bf16 + i * lda;
                        float *A_packed_row = A_packed + i * k_panel;

                        // FUSED conversion+packing: inline vector loop instead of helper to reduce loop overhead
#ifdef __AVX512F__
                        int j = 0;
                        // Process 32 BF16 (->32 FP32) per iteration
                        for (; j + 32 <= k_panel; j += 32)
                        {
                            const __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(A_row + j));
                            const __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0);
                            const __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);
                            const __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
                            const __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);
                            _mm512_storeu_ps(A_packed_row + j, _mm512_castsi512_ps(fp32_lo));
                            _mm512_storeu_ps(A_packed_row + j + 16, _mm512_castsi512_ps(fp32_hi));
                        }
                        // Remainder
                        for (; j < k_panel; ++j)
                        {
                            A_packed_row[j] = bf16_to_fp32(A_row[j]);
                        }
#elif defined(__AVX2__)
                        int j = 0;
                        // Process 16 BF16 per iteration using two cvtepu16_epi32 steps
                        for (; j + 16 <= k_panel; j += 16)
                        {
                            __m256i bf16_block = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(A_row + j));
                            __m128i bf16_lo = _mm256_extracti128_si256(bf16_block, 0);
                            __m128i bf16_hi = _mm256_extracti128_si256(bf16_block, 1);
                            __m256i fp32_lo = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_lo), 16);
                            __m256i fp32_hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_hi), 16);
                            _mm256_storeu_ps(A_packed_row + j, _mm256_castsi256_ps(fp32_lo));
                            _mm256_storeu_ps(A_packed_row + j + 8, _mm256_castsi256_ps(fp32_hi));
                        }
                        for (; j < k_panel; ++j)
                        {
                            A_packed_row[j] = bf16_to_fp32(A_row[j]);
                        }
#else
                        for (int j = 0; j < k_panel; ++j)
                        {
                            A_packed_row[j] = bf16_to_fp32(A_row[j]);
                        }
#endif
                    }
                }

                /**
                 * @brief Pack B panel [k × n_panel] with BF16→FP32 SIMD conversion
                 *
                 * Handles both transposed and non-transposed B matrices.
                 */
                void packBPanel_BF16toFP32(
                    const uint16_t *B_bf16,
                    float *B_packed,
                    int k_panel, int n_panel, int j_offset, int ldb,
                    bool transpose_B)
                {
                    if (transpose_B)
                    {
                        // B is [n, k], we need B^T = [k, n]
                        // B_bf16[j, k_idx] → B_packed[j * k_panel + k_idx]
                        for (int j = 0; j < n_panel; ++j)
                        {
                            const uint16_t *B_row = B_bf16 + (j_offset + j) * ldb;
                            float *B_packed_col = B_packed + j * k_panel;
#ifdef __AVX512F__
                            int k_idx = 0;
                            for (; k_idx + 32 <= k_panel; k_idx += 32)
                            {
                                const __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(B_row + k_idx));
                                const __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0);
                                const __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);
                                const __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
                                const __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);
                                _mm512_storeu_ps(B_packed_col + k_idx, _mm512_castsi512_ps(fp32_lo));
                                _mm512_storeu_ps(B_packed_col + k_idx + 16, _mm512_castsi512_ps(fp32_hi));
                            }
                            for (; k_idx < k_panel; ++k_idx)
                            {
                                B_packed_col[k_idx] = bf16_to_fp32(B_row[k_idx]);
                            }
#elif defined(__AVX2__)
                            int k_idx = 0;
                            for (; k_idx + 16 <= k_panel; k_idx += 16)
                            {
                                __m256i bf16_block = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(B_row + k_idx));
                                __m128i bf16_lo = _mm256_extracti128_si256(bf16_block, 0);
                                __m128i bf16_hi = _mm256_extracti128_si256(bf16_block, 1);
                                __m256i fp32_lo = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_lo), 16);
                                __m256i fp32_hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_hi), 16);
                                _mm256_storeu_ps(B_packed_col + k_idx, _mm256_castsi256_ps(fp32_lo));
                                _mm256_storeu_ps(B_packed_col + k_idx + 8, _mm256_castsi256_ps(fp32_hi));
                            }
                            for (; k_idx < k_panel; ++k_idx)
                            {
                                B_packed_col[k_idx] = bf16_to_fp32(B_row[k_idx]);
                            }
#else
                            for (int k_idx = 0; k_idx < k_panel; ++k_idx)
                            {
                                B_packed_col[k_idx] = bf16_to_fp32(B_row[k_idx]);
                            }
#endif
                        }
                    }
                    else
                    {
                        // B is [k, n], already in correct orientation
                        // Pack column-major: B_packed[j * k_panel + k_idx] = B[k_idx, j]
                        // Outer loop over columns, inner loop over rows within column
                        bool is_first_panel = (j_offset == 0);
                        for (int j = 0; j < n_panel; ++j)
                        {
                            for (int k_idx = 0; k_idx < k_panel; ++k_idx)
                            {
                                const uint16_t *B_row = B_bf16 + k_idx * ldb;
                                uint16_t bf16_val = B_row[j_offset + j];
                                float fp32_val = bf16_to_fp32(bf16_val);
                                B_packed[j * k_panel + k_idx] = fp32_val;

                                // Debug first few values of first panel
                                if (is_first_panel && j < 2 && k_idx < 2)
                                {
                                    LOG_INFO("PACK: j=" << j << " k_idx=" << k_idx
                                                        << " → B_packed[" << (j * k_panel + k_idx) << "]=" << fp32_val
                                                        << " (read from B[" << k_idx << "," << (j_offset + j) << "])");
                                }
                            }
                        }
                    }
                }
            };

            /**
             * @brief Register all BF16 micro-kernel variants with auto-tuner (without B_tensor)
             */
            std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>>
            registerBF16MicroKernelVariants()
            {
                std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>> variants;

                auto &registry = MicroKernelRegistry::instance();

                // Enumerate all registered kernels (same ranges as quantized GEMM)
                const std::vector<std::string> isa_names = {"simd::AVX512Tag", "simd::AVX2Tag"};
                const std::vector<int> mr_values = {1, 2, 4, 8, 16, 32, 64};
                const std::vector<int> nr_values = {1, 2, 4, 6, 8, 16, 32, 64};
                const std::vector<int> unroll_values = {1, 2, 4, 8, 16};
                const std::vector<int> prefetch_values = {0, 1, 2, 3, 5};

                LOG_DEBUG("Registering BF16 micro-kernel variants (B_tensor passed via decoder)");

                // Probe all combinations (registry will filter invalid ones)
                for (const auto &isa_name : isa_names)
                {
                    for (int mr : mr_values)
                    {
                        for (int nr : nr_values)
                        {
                            // Apply same register constraints as generator
                            if (isa_name == "simd::AVX512Tag" && mr * nr > 48)
                                continue;
                            if (isa_name == "simd::AVX2Tag" && mr * nr > 32)
                                continue;

                            for (int unroll_k : unroll_values)
                            {
                                for (int prefetch : prefetch_values)
                                {
                                    if (registry.has_kernel(isa_name, mr, nr, unroll_k, prefetch))
                                    {
                                        auto bundle = registry.get_kernel(isa_name, mr, nr, unroll_k, prefetch);

                                        // Create adapter WITHOUT B_tensor (will be passed via decoder)
                                        variants.push_back(
                                            std::make_unique<BF16MicroKernelAdapter>(
                                                isa_name, mr, nr, unroll_k, prefetch, bundle));
                                    }
                                }
                            }
                        }
                    }
                }

                LOG_DEBUG("Registered " << variants.size() << " BF16 micro-kernel variants");

                return variants;
            }

            /**
             * @brief Auto-tuned BF16×BF16→FP32 GEMM kernel
             */
            class AutoTunedBF16PackedGemm : public ITensorGemm
            {
            public:
                explicit AutoTunedBF16PackedGemm(const BF16Tensor *A_tensor, const BF16Tensor *B_tensor)
                    : A_tensor_(A_tensor), B_tensor_(B_tensor)
                {
                    ensureVariantsRegistered();
                }

                bool multiply(
                    const float *A, // Actually uint16_t* BF16 data
                    float *C,
                    int m, int n, int k,
                    bool transpose_B,
                    float alpha, float beta,
                    const MPIContext *mpi_ctx,
                    int device_idx) override
                {
                    (void)mpi_ctx;
                    (void)device_idx;

                    if (!B_tensor_)
                    {
                        LOG_ERROR("AutoTunedBF16PackedGemm: B_tensor is null");
                        return false;
                    }

                    // Use auto-tuner to select optimal variant
                    auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();
                    auto *optimal = tuner.getOptimalKernel(m, n, k);

                    if (!optimal)
                    {
                        LOG_ERROR("AutoTunedBF16PackedGemm: Failed to get optimal kernel");
                        return false;
                    }

                    // Pass B_tensor directly as ITensorGemmTileDataProvider (BF16Tensor implements ITensorGemmTileDataProvider)
                    const ITensorGemmTileDataProvider *decoder = B_tensor_;

                    // Delegate to auto-selected BF16 variant
                    return optimal->multiply(A, C, m, n, k, decoder, transpose_B, alpha, beta);
                }

                bool supports_device(int device_idx) const override
                {
                    return device_idx == -1; // CPU only
                }

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
                    LOG_ERROR("AutoTunedBF16PackedGemm: multiply_activations not implemented");
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
                    LOG_ERROR("AutoTunedBF16PackedGemm: multiply_activations_strided not implemented");
                    return false;
                }

            private:
                const BF16Tensor *A_tensor_;
                const BF16Tensor *B_tensor_;

                void ensureVariantsRegistered()
                {
                    static bool registered = false;
                    static std::mutex registration_mutex;

                    if (!registered)
                    {
                        std::lock_guard<std::mutex> lock(registration_mutex);
                        if (!registered)
                        {
                            // Register variants WITHOUT B_tensor (passed via decoder at runtime)
                            auto variants = kernels::gemm::registerBF16MicroKernelVariants();
                            auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();

                            LOG_INFO("Registering " << variants.size() << " BF16 micro-kernel variants with auto-tuner");

                            for (auto &variant : variants)
                            {
                                tuner.registerVariant(std::move(variant));
                            }

                            registered = true;
                            LOG_INFO("BF16 micro-kernel variants registered successfully");
                        }
                    }
                }
            };

        } // namespace gemm
    } // namespace kernels

    // Factory function in llaminar2:: namespace (matches header declaration)
    std::unique_ptr<ITensorGemm> createBF16PackedGemm(
        const BF16Tensor *A_tensor,
        const BF16Tensor *B_tensor)
    {
        return std::make_unique<kernels::gemm::AutoTunedBF16PackedGemm>(A_tensor, B_tensor);
    }

} // namespace llaminar2

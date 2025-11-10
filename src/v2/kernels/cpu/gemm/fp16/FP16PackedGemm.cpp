/**
 * @file FP16PackedGemm.cpp
 * @brief FP16×FP16→FP32 GEMM with full autotuner integration
 * @author David Sanftenberg
 *
 * Full autotuner implementation with cache-blocked, packed FP16 GEMM.
 * Follows INT8PackedGemm and BF16PackedGemm architecture:
 *   - Input: uint16_t (FP16) via ITensorGemmTileDataProvider interface
 *   - Output: FP32 (lossless conversion via hardware FP16→FP32)
 *   - Register constraint: MR × NR ≤ 32 (FP32 accumulators)
 *
 * Key design: FP16Tensor implements ITensorGemmTileDataProvider just like BF16Tensor
 * - FP16 tensors are element-wise FP16 values (not block-quantized)
 * - decode_block_at() converts entire row from FP16 to FP32
 * - block_size() returns row width (entire row = one "block")
 *
 * Performance expectations:
 *   - AVX512F: 2-3× faster than FP32 GEMM
 *   - AVX2+F16C: 1.5-2× faster than FP32 GEMM
 *   - Scalar: 0.5× FP32 speed (conversion overhead)
 */

#include "FP16PackedGemm.h"
#include "FP16GemmImpl.h"
#include "../GemmAutoTuner.h"
#include "../../../../utils/Logger.h"
#include "../../../../tensors/TensorKernels.h"
#include "../../../../tensors/Tensors.h" // For FP16Tensor
#include <memory>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <mutex>

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            namespace cpu
            {
                namespace fp16_impl
                {

                    /**
                     * @brief Micro-kernel adapter for FP16 GEMM with autotuner integration
                     *
                     * This adapter bridges the IQuantizedGemmVariant interface (autotuner)
                     * with FP16 GEMM kernels. Unlike INT8 which needs ITensorGemmTileDataProvider,
                     * FP16 data is passed directly as uint16_t arrays.
                     *
                     * Architecture:
                     *   1. Cache blocking: mc × nc × kc tiles for L2/L3 efficiency
                     *   2. Panel packing: Contiguous layouts for better prefetching
                     *   3. Micro-kernel: Calls gemm_fp16_auto() with optimal ISA
                     *   4. Register blocking: mr × nr tile sizes (≤32 for FP32 accumulators)
                     */
                    class FP16MicroKernelAdapter : public llaminar::v2::kernels::IQuantizedGemmVariant
                    {
                    public:
                        FP16MicroKernelAdapter(int mr, int nr, int unroll_k, int prefetch_dist)
                            : mr_(mr), nr_(nr), unroll_k_(unroll_k), prefetch_dist_(prefetch_dist)
                        {
                            // Build descriptive name: "FP16_8x6_u4_p2"
                            std::ostringstream oss;
                            oss << "FP16_" << mr << "x" << nr << "_u" << unroll_k << "_p" << prefetch_dist;
                            name_ = oss.str();

                            // Select cache tile sizes based on micro-kernel register footprint
                            // FP16 uses half the memory of FP32, allowing larger tiles
                            int register_footprint = mr * nr;

                            if (register_footprint <= 8)
                            {
                                // Small kernels: Large cache tiles for better reuse
                                mc_ = 512;
                                nc_ = 4096;
                                kc_ = 256;
                            }
                            else if (register_footprint <= 16)
                            {
                                // Medium kernels: Balanced tiles
                                mc_ = 384;
                                nc_ = 2048;
                                kc_ = 256;
                            }
                            else
                            {
                                // Large kernels: Conservative to avoid thrashing
                                mc_ = 256;
                                nc_ = 1024;
                                kc_ = 128;
                            }

                            // Ensure KC multiple of 16 for AVX512F (processes 16 FP16s per iteration)
                            kc_ = (kc_ + 15) & ~15;
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

                        /**
                         * @brief Execute FP16×FP16→FP32 GEMM with ITensorGemmTileDataProvider interface
                         *
                         * Standard multiply() implementation that extracts FP16 data from decoder.
                         * Follows the same pattern as INT8PackedGemm and BF16PackedGemm for consistency.
                         *
                         * The decoder can be:
                         *   - FP16Tensor (production): provides fp16_data()
                         *   - MockTensorGemmTileDataProvider (benchmarking): provides get_raw_block_at()
                         */
                        bool multiply(
                            const float *A_fp16_as_float, // FP16 data reinterpreted as float*
                            float *C,
                            int m, int n, int k,
                            const llaminar2::ITensorGemmTileDataProvider *decoder, // FP16Tensor or MockTensorGemmTileDataProvider
                            bool transpose_B,
                            float alpha = 1.0f,
                            float beta = 0.0f) override
                        {
                            if (!decoder)
                            {
                                LOG_ERROR("FP16MicroKernelAdapter: decoder cannot be nullptr");
                                return false;
                            }

                            // Get FP16 data pointers
                            const uint16_t *A_fp16 = reinterpret_cast<const uint16_t *>(A_fp16_as_float);
                            const uint16_t *B_fp16 = nullptr;

                            // Try to extract FP16Tensor first (production path)
                            const auto *fp16_tensor = dynamic_cast<const llaminar2::FP16Tensor *>(decoder);
                            if (fp16_tensor)
                            {
                                B_fp16 = fp16_tensor->fp16_data();
                            }
                            else
                            {
                                // Fallback: use ITensorGemmTileDataProvider interface (benchmarking path with MockTensorGemmTileDataProvider)
                                // MockTensorGemmTileDataProvider stores float data, but we reinterpret as uint16_t for FP16
                                B_fp16 = reinterpret_cast<const uint16_t *>(decoder->get_raw_block_at(0, 0));
                            }

                            const int lda = k; // A[m, k] row-major
                            const int ldb = transpose_B ? k : n;
                            const int ldc = n; // C[m, n] row-major

// Parallel cache-blocked execution
#pragma omp parallel
                            {
                                // Thread-local packing buffers (avoid false sharing)
                                std::vector<uint16_t> A_packed_local(mc_ * kc_);
                                std::vector<uint16_t> B_packed_local(kc_ * nc_);

#pragma omp for schedule(dynamic)
                                for (int jc = 0; jc < n; jc += nc_)
                                {
                                    int nc = std::min(nc_, n - jc);

                                    for (int pc = 0; pc < k; pc += kc_)
                                    {
                                        int kc = std::min(kc_, k - pc);

                                        // Pack B panel [kc × nc]
                                        packBPanel_FP16(
                                            B_fp16, B_packed_local.data(),
                                            kc, nc, jc, ldb, pc, transpose_B);

                                        // Loop over M dimension
                                        for (int ic = 0; ic < m; ic += mc_)
                                        {
                                            int mc = std::min(mc_, m - ic);

                                            // Pack A panel [mc × kc]
                                            packAPanel_FP16(
                                                A_fp16 + ic * lda + pc, A_packed_local.data(),
                                                mc, kc, lda);

                                            // Compute C[ic:ic+mc, jc:jc+nc] with micro-kernel tiling
                                            for (int ir = 0; ir < mc; ir += mr_)
                                            {
                                                int mr = std::min(mr_, mc - ir);

                                                for (int jr = 0; jr < nc; jr += nr_)
                                                {
                                                    int nr = std::min(nr_, nc - jr);

                                                    float *C_tile = C + (ic + ir) * ldc + (jc + jr);

                                                    // Call FP16 micro-kernel (auto-dispatch to best ISA)
                                                    // Alpha/beta handling: first K-panel uses alpha/beta,
                                                    // subsequent panels accumulate with alpha/1.0
                                                    float use_alpha = alpha;
                                                    float use_beta = (pc == 0) ? beta : 1.0f;

                                                    llaminar2::kernels::gemm::fp16_impl::gemm_fp16_auto(
                                                        A_packed_local.data() + ir * kc,
                                                        B_packed_local.data() + jr * kc,
                                                        C_tile,
                                                        mr, nr, kc,
                                                        ldc,
                                                        use_alpha, use_beta);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            return true;
                        }

                    private:
                        std::string name_;
                        int mr_, nr_, unroll_k_, prefetch_dist_;
                        int mc_, nc_, kc_;

                        /**
                         * @brief Pack A panel into contiguous buffer
                         *
                         * Input: A[m_panel, k_panel] row-major (uint16_t FP16)
                         * Output: A_packed[m_panel * k_panel] contiguous
                         */
                        void packAPanel_FP16(
                            const uint16_t *A_fp16,
                            uint16_t *A_packed,
                            int m_panel, int k_panel, int lda)
                        {
                            for (int i = 0; i < m_panel; ++i)
                            {
                                const uint16_t *A_row = A_fp16 + i * lda;
                                uint16_t *A_packed_row = A_packed + i * k_panel;

                                // Prefetch next row
                                if (i + 1 < m_panel)
                                {
                                    __builtin_prefetch(A_fp16 + (i + 1) * lda, 0, 3);
                                }

                                // Copy row (memcpy is highly optimized for contiguous data)
                                std::memcpy(A_packed_row, A_row, k_panel * sizeof(uint16_t));
                            }
                        }

                        /**
                         * @brief Pack B panel for optimal access patterns
                         *
                         * Handles both normal and transposed B layouts.
                         */
                        void packBPanel_FP16(
                            const uint16_t *B_fp16,
                            uint16_t *B_packed,
                            int k_panel, int n_panel,
                            int j_offset, int ldb, int k_offset,
                            bool transpose_B)
                        {
                            if (transpose_B)
                            {
                                // B is [n, k], we need B^T = [k, n]
                                // B_fp16[j, k_idx] → B_packed[j * k_panel + k_idx]
                                for (int j = 0; j < n_panel; ++j)
                                {
                                    const uint16_t *B_row = B_fp16 + (j_offset + j) * ldb + k_offset;
                                    uint16_t *B_packed_col = B_packed + j * k_panel;
                                    std::memcpy(B_packed_col, B_row, k_panel * sizeof(uint16_t));
                                }
                            }
                            else
                            {
                                // B is [k, n], pack column-major
                                for (int j = 0; j < n_panel; ++j)
                                {
                                    for (int k_idx = 0; k_idx < k_panel; ++k_idx)
                                    {
                                        const uint16_t *B_row = B_fp16 + (k_offset + k_idx) * ldb;
                                        B_packed[j * k_panel + k_idx] = B_row[j_offset + j];
                                    }
                                }
                            }
                        }
                    };

                    /**
                     * @brief Register all FP16 micro-kernel variants for autotuner
                     *
                     * Enumerates ~300-400 variants by varying:
                     *   - MR (micro-kernel rows): {1, 2, 4, 8, 16, 32}
                     *   - NR (micro-kernel cols): {1, 2, 4, 6, 8, 16, 32}
                     *   - UNROLL_K: {1, 2, 4, 8}
                     *   - PREFETCH_DIST: {0, 1, 2, 3}
                     *
                     * Constraint: MR × NR ≤ 32 (FP32 accumulator register file)
                     * This is stricter than INT8 (≤48) due to register size.
                     */
                    std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>>
                    registerFP16MicroKernelVariants()
                    {
                        std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>> variants;

                        const std::vector<int> mr_values = {1, 2, 4, 8, 16, 32};
                        const std::vector<int> nr_values = {1, 2, 4, 6, 8, 16, 32};
                        const std::vector<int> unroll_values = {1, 2, 4, 8};
                        const std::vector<int> prefetch_values = {0, 1, 2, 3};

                        LOG_DEBUG("Registering FP16 micro-kernel variants");

                        int variant_count = 0;
                        for (int mr : mr_values)
                        {
                            for (int nr : nr_values)
                            {
                                // FP32 accumulator register constraint (stricter than INT8)
                                if (mr * nr > 32)
                                    continue;

                                for (int unroll_k : unroll_values)
                                {
                                    for (int prefetch : prefetch_values)
                                    {
                                        variants.push_back(
                                            std::make_unique<FP16MicroKernelAdapter>(
                                                mr, nr, unroll_k, prefetch));
                                        variant_count++;
                                    }
                                }
                            }
                        }

                        LOG_DEBUG("Registered " << variant_count << " FP16 micro-kernel variants");
                        return variants;
                    }

                    /**
                     * @brief Auto-tuned FP16 GEMM kernel with variant selection
                     *
                     * This class implements ITensorGemm and integrates with GemmAutoTuner
                     * to automatically select optimal micro-kernel configuration per (m,n,k).
                     */
                    class AutoTunedFP16PackedGemm : public llaminar2::ITensorGemm
                    {
                    public:
                        AutoTunedFP16PackedGemm()
                        {
                            ensureVariantsRegistered();
                        }

                        bool multiply(
                            const float *A,
                            float *C,
                            int m, int n, int k,
                            bool transpose_B,
                            float alpha, float beta,
                            const llaminar2::MPIContext *mpi_ctx,
                            int device_idx) override
                        {
                            (void)mpi_ctx;
                            (void)device_idx;

                            // For weight-based GEMM, B would come from tensor
                            // Not yet implemented - requires weight tensor integration
                            LOG_ERROR("AutoTunedFP16PackedGemm::multiply (weight-based) not yet implemented");
                            LOG_ERROR("Use multiply_activations for activation-activation GEMM");
                            return false;
                        }

                        bool multiply_activations(
                            const float *A, // FP16 data as float*
                            const float *B, // FP16 data as float*
                            float *C,
                            int m, int n, int k,
                            bool transpose_B,
                            float alpha, float beta,
                            const llaminar2::MPIContext *mpi_ctx,
                            int device_idx) override
                        {
                            (void)mpi_ctx;

                            if (device_idx != -1)
                            {
                                LOG_ERROR("FP16 GEMM only supports CPU (device_idx=-1)");
                                return false;
                            }

                            // Select optimal variant for this workload
                            auto variant = selectVariant(m, n, k);
                            if (!variant)
                            {
                                LOG_ERROR("No suitable FP16 variant for (" << m << "×" << n << "×" << k << ")");
                                return false;
                            }

                            // Create temporary FP16Tensor wrapper for B matrix
                            // B shape depends on transpose_B: transpose_B ? [k, n] : [n, k]
                            std::vector<size_t> B_shape = transpose_B ? std::vector<size_t>{(size_t)k, (size_t)n}
                                                                      : std::vector<size_t>{(size_t)n, (size_t)k};

                            // Wrap B data (const_cast is safe - multiply() doesn't modify B)
                            const uint16_t *B_fp16 = reinterpret_cast<const uint16_t *>(B);
                            std::vector<uint16_t> B_data_copy(B_fp16, B_fp16 + (transpose_B ? k * n : n * k));
                            llaminar2::FP16Tensor B_tensor(B_shape, B_data_copy);

                            // Call standard multiply() with decoder
                            return variant->multiply(
                                A, C, m, n, k,
                                &B_tensor, // ITensorGemmTileDataProvider*
                                transpose_B,
                                alpha, beta);
                        }

                        bool multiply_activations_strided(
                            const float *A, const float *B, float *C,
                            int m, int n, int k,
                            int lda, int ldb, int ldc,
                            bool transpose_B,
                            float alpha, float beta,
                            const llaminar2::MPIContext *mpi_ctx,
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

                            LOG_ERROR("FP16 multiply_activations_strided not yet implemented");
                            return false;
                        }

                        bool supports_device(int device_idx) const override
                        {
                            return (device_idx == -1); // CPU only
                        }

                    private:
                        void ensureVariantsRegistered()
                        {
                            static std::once_flag flag;
                            std::call_once(flag, []()
                                           {
            auto variants = registerFP16MicroKernelVariants();
            auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();
            
            LOG_INFO("Registering " << variants.size() << " FP16 variants with autotuner");
            
            for (auto &variant : variants)
            {
                tuner.registerVariant(std::move(variant));
            }
            
            LOG_INFO("FP16 micro-kernel variants registered successfully"); });
                        }

                        llaminar::v2::kernels::IQuantizedGemmVariant *selectVariant(int m, int n, int k)
                        {
                            auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();
                            auto variant = tuner.getOptimalKernel(m, n, k);

                            if (variant)
                            {
                                LOG_DEBUG("Selected FP16 variant: " << variant->name()
                                                                    << " for (" << m << "×" << n << "×" << k << ")");
                            }

                            return variant;
                        }
                    };

                } // namespace fp16_impl

            } // namespace cpu
        } // namespace kernels
    } // namespace v2
} // namespace llaminar

// Public API in llaminar2 namespace (matches header declaration)
namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Check if FP16 GEMM is supported on current hardware
             *
             * FP16 is universally supported via:
             *   - AVX512F: Native _mm512_cvtph_ps conversion
             *   - AVX2+F16C: F16C _mm256_cvtph_ps extension
             *   - Scalar: Portable bit-manipulation fallback
             */
            bool isFP16GemmSupported()
            {
#if defined(__AVX512F__)
                LOG_DEBUG("FP16 GEMM: AVX512F native conversion available");
                return true;
#elif defined(__AVX2__) && defined(__F16C__)
                LOG_DEBUG("FP16 GEMM: AVX2+F16C conversion available");
                return true;
#else
                LOG_DEBUG("FP16 GEMM: Scalar fallback available");
                return true;
#endif
            }

            /**
             * @brief Create auto-tuned FP16 GEMM kernel
             *
             * Factory function returns ITensorGemm implementation with full
             * autotuner integration (~300-400 micro-kernel variants).
             */
            std::unique_ptr<ITensorGemm> createFP16PackedGemm(
                const TensorBase *A,
                const TensorBase *B)
            {
                (void)A;
                (void)B;

                LOG_INFO("Creating auto-tuned FP16 GEMM with micro-kernel variants");
                return std::make_unique<llaminar::v2::kernels::cpu::fp16_impl::AutoTunedFP16PackedGemm>();
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2

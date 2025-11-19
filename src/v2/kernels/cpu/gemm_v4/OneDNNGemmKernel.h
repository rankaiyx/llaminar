/**
 * @file OneDNNGemmKernel.h
 * @brief ITensorGemm implementation using OneDNN INT8 GEMM
 *
 * Provides a unified ITensorGemm interface using OneDNN's INT8 acceleration.
 * Includes OneDNN primitive wrappers (previously in OneDNNGemm.h) and the
 * ITensorGemm kernel implementation.
 *
 * @author David Sanftenberg
 * @date 2025-01-15
 */

#pragma once

#ifndef HAVE_ONEDNN
#error "OneDNN support is required to use gemm_v4"
#endif

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "OneDNNGemmAdapter.h"
#include "../../../utils/Logger.h"
#include "../primitives/SoftmaxPrimitives_New.h"

namespace llaminar2
{
    namespace gemm_v4
    {
#undef LLAMINAR_LIBMVEC_WIDTH
        namespace detail
        {
#if defined(__AVX512F__)
#define LLAMINAR_LIBMVEC_WIDTH 16
            constexpr int kLibmvecWidth = 16;
            using LibmVecPack = float __attribute__((vector_size(64)));
            extern "C" LibmVecPack _ZGVeN16v_expf(LibmVecPack) __attribute__((const));
            inline LibmVecPack libmvec_expf(LibmVecPack value)
            {
                return _ZGVeN16v_expf(value);
            }
#elif defined(__AVX__)
#define LLAMINAR_LIBMVEC_WIDTH 8
            constexpr int kLibmvecWidth = 8;
            using LibmVecPack = float __attribute__((vector_size(32)));
            extern "C" LibmVecPack _ZGVcN8v_expf(LibmVecPack) __attribute__((const));
            inline LibmVecPack libmvec_expf(LibmVecPack value)
            {
                return _ZGVcN8v_expf(value);
            }
#elif defined(__SSE2__)
#define LLAMINAR_LIBMVEC_WIDTH 4
            constexpr int kLibmvecWidth = 4;
            using LibmVecPack = float __attribute__((vector_size(16)));
            extern "C" LibmVecPack _ZGVbN4v_expf(LibmVecPack) __attribute__((const));
            inline LibmVecPack libmvec_expf(LibmVecPack value)
            {
                return _ZGVbN4v_expf(value);
            }
#else
#define LLAMINAR_LIBMVEC_WIDTH 0
            constexpr int kLibmvecWidth = 0;
#endif

            inline float apply_libmvec_exp(float *data, int length, float row_max)
            {
                float sum = 0.0f;
#if LLAMINAR_LIBMVEC_WIDTH > 0
                alignas(64) float input[LLAMINAR_LIBMVEC_WIDTH];
                int idx = 0;
                while (idx < length)
                {
                    const int lanes = std::min(LLAMINAR_LIBMVEC_WIDTH, length - idx);
                    for (int lane = 0; lane < lanes; ++lane)
                    {
                        input[lane] = data[idx + lane] - row_max;
                    }
                    for (int lane = lanes; lane < LLAMINAR_LIBMVEC_WIDTH; ++lane)
                    {
                        input[lane] = -std::numeric_limits<float>::infinity();
                    }

                    LibmVecPack vec_in;
                    std::memcpy(&vec_in, input, sizeof(vec_in));
                    LibmVecPack vec_out = libmvec_expf(vec_in);
                    std::memcpy(input, &vec_out, sizeof(vec_out));

                    for (int lane = 0; lane < lanes; ++lane)
                    {
                        const float exp_val = input[lane];
                        data[idx + lane] = exp_val;
                        sum += exp_val;
                    }

                    idx += lanes;
                }
#else
                for (int i = 0; i < length; ++i)
                {
                    const float exp_val = std::expf(data[i] - row_max);
                    data[i] = exp_val;
                    sum += exp_val;
                }
#endif
                return sum;
            }
        } // namespace detail

        // ========== OneDNN Primitive Wrappers (from OneDNNGemm.h) ==========

        /**
         * @brief Get singleton OneDNN CPU engine
         */
        inline dnnl::engine &onednn_engine()
        {
            static thread_local dnnl::engine engine_instance(dnnl::engine::kind::cpu, 0);
            return engine_instance;
        }

        /**
         * @brief Get singleton OneDNN execution stream
         */
        inline dnnl::stream &onednn_stream()
        {
            thread_local dnnl::stream stream_instance(onednn_engine());
            return stream_instance;
        }

        /**
         * @brief Execute INT8 matrix multiplication using OneDNN
         */
        inline bool run_onednn_int8_matmul(const int8_t *A,
                                           const int8_t *B,
                                           int32_t *C,
                                           int M,
                                           int N,
                                           int K)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::s8, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::s8, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::s32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<int8_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<int8_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN matmul failed: status=" << e.status
                                                          << " message=" << e.what());
                return false;
            }

            return true;
        }

        inline bool run_onednn_fp32_matmul(const float *A,
                                           const float *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            // LOG_DEBUG("Entering run_onednn_fp32_matmul M=" << M << " N=" << N << " K=" << K);

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed: status=" << e.status
                                                               << " message=" << e.what());
                return false;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed with std::exception: " << e.what());
                return false;
            }
            catch (...)
            {
                LOG_ERROR("OneDNN FP32 matmul failed with unknown exception");
                return false;
            }

            return true;
        }

        inline bool run_onednn_fp32_matmul_softmax(const float *A,
                                                   const float *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int softmax_axis)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            std::cerr << "run_onednn_fp32_matmul_softmax invoked with M=" << M
                      << " N=" << N << " K=" << K << " axis=" << softmax_axis << std::endl;

            const int ndims = 2;
            int axis = softmax_axis;
            if (axis < 0)
            {
                axis += ndims;
            }
            if (axis != ndims - 1)
            {
                // Current OneDNN softmax post-op implementation only supports the last dimension.
                std::cerr << "OneDNN matmul+softmax rejected axis " << softmax_axis
                          << " (normalized " << axis << ") for ndims=" << ndims << std::endl;
                return false;
            }

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;
                ops.append_softmax(axis, false);
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                std::cerr << "OneDNN matmul+softmax failed: status=" << e.status
                          << " message=" << e.what() << std::endl;
                return false;
            }

            std::cerr << "OneDNN matmul+softmax execution succeeded" << std::endl;
            return true;
        }

        inline bool apply_softmax_inplace(float *data,
                                          int rows,
                                          int cols,
                                          int axis)
        {
            if (!data || rows <= 0 || cols <= 0)
            {
                return false;
            }

            int normalized_axis = axis;
            if (normalized_axis < 0)
            {
                normalized_axis += 2; // Only 2D tensors supported here
            }

            auto exp_range = [](float value) -> float
            {
                return std::exp(value);
            };

            if (normalized_axis == 1)
            {
                for (int r = 0; r < rows; ++r)
                {
                    const size_t base = static_cast<size_t>(r) * static_cast<size_t>(cols);
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int c = 0; c < cols; ++c)
                    {
                        max_val = std::max(max_val, data[base + static_cast<size_t>(c)]);
                    }

                    float sum = detail::apply_libmvec_exp(data + base, cols, max_val);

                    const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                    for (int c = 0; c < cols; ++c)
                    {
                        data[base + static_cast<size_t>(c)] *= inv_sum;
                    }
                }
                return true;
            }

            if (normalized_axis == 0)
            {
                std::vector<float> column(static_cast<size_t>(rows));
                for (int c = 0; c < cols; ++c)
                {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int r = 0; r < rows; ++r)
                    {
                        const float val = data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
                        column[static_cast<size_t>(r)] = val;
                        max_val = std::max(max_val, val);
                    }

                    float sum = detail::apply_libmvec_exp(column.data(), rows, max_val);

                    const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                    for (int r = 0; r < rows; ++r)
                    {
                        data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                            column[static_cast<size_t>(r)] * inv_sum;
                    }
                }
                return true;
            }

            return false;
        }

        inline const float *prepare_rhs_for_matmul(const float *B,
                                                   int n,
                                                   int k,
                                                   bool transpose_B)
        {
            if (!transpose_B)
            {
                return B;
            }

            thread_local std::vector<float> B_transposed;
            B_transposed.resize(static_cast<size_t>(k) * static_cast<size_t>(n));

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < k; ++j)
                {
                    B_transposed[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
                        B[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)];
                }
            }

            return B_transposed.data();
        }

        // ========== Lightweight Activation View ==========

        /**
         * @brief Lightweight view wrapper for raw pointer activations
         *
         * Implements IActivationTensor interface without owning memory.
         * Used to avoid temporary allocations in ITensorGemm::multiply().
         */
        class ActivationView : public IActivationTensor
        {
        private:
            const float *data_;
            float *output_data_; // For from_int32_with_scales()
            std::vector<size_t> shape_;

        public:
            ActivationView(const float *data, size_t rows, size_t cols)
                : data_(data), output_data_(nullptr), shape_{rows, cols} {}

            ActivationView(float *data, size_t rows, size_t cols)
                : data_(data), output_data_(data), shape_{rows, cols} {}

            // IActivationTensor interface - only implement what we need
            ActivationPack to_int8_activation_pack(int rows, int cols) const override
            {
                ActivationPack pack;
                pack.rows = rows;
                pack.cols = cols;
                pack.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
                pack.row_scales.resize(static_cast<size_t>(rows));

                for (int r = 0; r < rows; ++r)
                {
                    const float *row_ptr = data_ + static_cast<size_t>(r) * static_cast<size_t>(cols);
                    float max_abs = 0.0f;
                    for (int c = 0; c < cols; ++c)
                    {
                        float val = std::abs(row_ptr[c]);
                        if (val > max_abs)
                            max_abs = val;
                    }

                    float scale = max_abs / 127.0f;
                    pack.row_scales[r] = scale;

                    if (scale > 0.0f)
                    {
                        float inv_scale = 127.0f / max_abs;
                        for (int c = 0; c < cols; ++c)
                        {
                            float val = row_ptr[c] * inv_scale;
                            pack.data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                                static_cast<int8_t>(std::round(std::min(127.0f, std::max(-127.0f, val))));
                        }
                    }
                    else
                    {
                        std::fill_n(pack.data.data() + static_cast<size_t>(r) * static_cast<size_t>(cols),
                                    cols, int8_t(0));
                    }
                }

                return pack;
            }

            bool from_int32_with_scales(const int32_t *accum, int rows, int cols,
                                        const float *row_scales, const float *col_scales,
                                        const float *bias = nullptr) override
            {
                if (!output_data_)
                {
                    return false;
                }

                for (int r = 0; r < rows; ++r)
                {
                    const float rscale = row_scales ? row_scales[r] : 1.0f;
                    for (int c = 0; c < cols; ++c)
                    {
                        const float cscale = col_scales ? col_scales[c] : 1.0f;
                        float val = static_cast<float>(accum[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)]) *
                                    rscale * cscale;
                        if (bias)
                        {
                            val += bias[c];
                        }
                        output_data_[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] = val;
                    }
                }

                return true;
            }

            // Unused methods (throw if called)
            std::unique_ptr<ITensorRoPE> createRoPE() override
            {
                throw std::runtime_error("ActivationView::createRoPE() not supported");
            }
            std::unique_ptr<ITensorSwiGLU> createSwiGLU() override
            {
                throw std::runtime_error("ActivationView::createSwiGLU() not supported");
            }
            std::unique_ptr<ITensorSoftmax> createSoftmax() override
            {
                throw std::runtime_error("ActivationView::createSoftmax() not supported");
            }
            std::unique_ptr<ITensorRMSNorm> createRMSNorm() override
            {
                throw std::runtime_error("ActivationView::createRMSNorm() not supported");
            }
            std::unique_ptr<ITensorAttention> createAttention() override
            {
                throw std::runtime_error("ActivationView::createAttention() not supported");
            }
            bool applyRMSNorm(const float *, int, int, float, const MPIContext *, int) override
            {
                throw std::runtime_error("ActivationView::applyRMSNorm() not supported");
            }
            bool applyRoPE(float *, const int *, int, int, int, int, float, bool, const MPIContext *, int) override
            {
                throw std::runtime_error("ActivationView::applyRoPE() not supported");
            }
        };

        struct Int8MatmulSoftmaxParams
        {
            int M;
            int N;
            int K;
            const float *row_scales = nullptr;
            const float *col_scales = nullptr;
            const float *bias = nullptr;
            bool causal = false;
            float softmax_scale = 1.0f;
            bool parallel_softmax = true;
        };

        inline bool run_onednn_int8_matmul_with_softmax(const int8_t *A,
                                                        const int8_t *B,
                                                        float *scores,
                                                        const Int8MatmulSoftmaxParams &params)
        {
            if (!A || !B || !scores)
            {
                return false;
            }

            const int M = params.M;
            const int N = params.N;
            const int K = params.K;
            if (M <= 0 || N <= 0 || K <= 0)
            {
                return false;
            }

            static thread_local std::vector<int32_t> accum_buffer;
            const size_t accum_elems = static_cast<size_t>(M) * static_cast<size_t>(N);
            if (accum_buffer.size() < accum_elems)
            {
                accum_buffer.resize(accum_elems);
            }

            if (!run_onednn_int8_matmul(A, B, accum_buffer.data(), M, N, K))
            {
                return false;
            }

            const int32_t *accum_ptr = accum_buffer.data();
            const float *row_scales = params.row_scales;
            const float *col_scales = params.col_scales;
            const float *bias = params.bias;
            const bool causal = params.causal;
            const float softmax_scale = params.softmax_scale;
            const bool parallel = params.parallel_softmax;

            auto process_row = [&](int row, std::vector<float> &scratch)
            {
                const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(N);
                const float row_scale = row_scales ? row_scales[row] : 1.0f;
                const int valid_cols = causal ? std::min(N, row + 1) : N;

                float row_max = -std::numeric_limits<float>::infinity();
                for (int col = 0; col < valid_cols; ++col)
                {
                    float value = static_cast<float>(accum_ptr[row_offset + static_cast<size_t>(col)]) * row_scale;
                    if (col_scales)
                    {
                        value *= col_scales[col];
                    }
                    if (bias)
                    {
                        value += bias[col];
                    }
                    value *= softmax_scale;

                    scratch[col] = value;
                    row_max = std::max(row_max, value);
                }
                for (int col = valid_cols; col < N; ++col)
                {
                    scratch[col] = -std::numeric_limits<float>::infinity();
                }

                if (!std::isfinite(row_max))
                {
                    std::fill(scores + row_offset, scores + row_offset + static_cast<size_t>(N), 0.0f);
                    return;
                }

                float sum = detail::apply_libmvec_exp(scratch.data(), valid_cols, row_max);

                const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                float *out_row = scores + row_offset;
                for (int col = 0; col < valid_cols; ++col)
                {
                    out_row[col] = scratch[col] * inv_sum;
                }
                for (int col = valid_cols; col < N; ++col)
                {
                    out_row[col] = 0.0f;
                }
            };

            auto run_serial = [&]()
            {
                std::vector<float> scratch(static_cast<size_t>(N));
                for (int row = 0; row < M; ++row)
                {
                    process_row(row, scratch);
                }
            };

#if defined(_OPENMP)
            if (parallel)
            {
#pragma omp parallel
                {
                    std::vector<float> scratch(static_cast<size_t>(N));
#pragma omp for schedule(static)
                    for (int row = 0; row < M; ++row)
                    {
                        process_row(row, scratch);
                    }
                }
            }
            else
#endif
            {
                run_serial();
            }

            return true;
        }

        // ========== ITensorGemm Implementation ==========

        /**
         * @brief OneDNN-based ITensorGemm kernel implementation
         *
         * **Usage**:
         * ```cpp
         * // In tensor's createGemm() method:
         * return std::make_unique<OneDNNGemmKernel>(this);
         * ```
         *
         * **Design**:
         * - Binds to a specific weight tensor at construction
         * - Works with weight tensors implementing TensorBase::to_int8_perchannel()
         * - Uses FP32 activations with per-row INT8 quantization
         * - Supports bias addition
         */
        class OneDNNGemmKernel : public ITensorGemm
        {
        public:
            /**
             * @brief Construct kernel bound to a weight tensor
             * @param weight_tensor Pointer to weight tensor (B matrix)
             */
            explicit OneDNNGemmKernel(const TensorBase *weight_tensor = nullptr)
                : weight_tensor_(weight_tensor) {}
            ~OneDNNGemmKernel() override = default;

            /**
             * @brief Check device support (CPU-only for OneDNN)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx == -1; // CPU only
            }

            /**
             * @brief Execute GEMM: C = alpha * A @ B^T + beta * C
             *
             * Uses OneDNN INT8 acceleration with per-row activation and per-column weight scaling.
             *
             * @param A Input activations [m, k] (FP32, row-major)
             * @param C Output matrix [m, n] (FP32, row-major)
             * @param m Number of rows in A and C
             * @param n Number of columns in B and C (before transpose)
             * @param k Number of columns in A and rows in B
             * @param transpose_B Whether B is stored transposed (must be true for weights)
             * @param alpha Scale factor (must be 1.0)
             * @param beta Scale factor (must be 0.0)
             * @param mpi_ctx MPI context (unused, for future distributed support)
             * @param device_idx Device index (must be -1 for CPU)
             *
             * @return true on success, false on error
             *
             * @note Weight tensor B must be bound to this kernel via constructor
             * @note Currently only supports alpha=1.0, beta=0.0, transpose_B=true
             */
            bool multiply_with_softmax(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                int softmax_axis = 1,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)mpi_ctx;

                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Only CPU execution supported (device_idx="
                              << device_idx << ")");
                    return false;
                }

                // We historically only supported the weight-bound path with B stored
                // transposed in the tensor. For simplicity and robustness in E2E
                // scenarios, treat a non-transposed B pointer here as a request to use
                // the provided activations as the RHS directly, falling back to the
                // generic activation-activation matmul + softmax path. This avoids
                // hard failure while preserving the existing optimised path when
                // transpose_B == true and a bound weight tensor is available.
                if (!transpose_B || !weight_tensor_)
                {
                    if (!transpose_B)
                    {
                        LOG_WARN("[OneDNNGemmKernel] multiply_with_softmax received non-transposed B; "
                                 "falling back to activation matmul + softmax");
                    }
                    else
                    {
                        LOG_WARN("[OneDNNGemmKernel] No bound weight tensor; falling back to "
                                 "activation matmul + softmax");
                    }

                    // Use activation-activation GEMM followed by explicit softmax.
                    if (!multiply_activations(A, B, C, m, n, k, /*transpose_B=*/transpose_B,
                                              /*alpha=*/1.0f, /*beta=*/0.0f,
                                              mpi_ctx, device_idx))
                    {
                        LOG_ERROR("[OneDNNGemmKernel] Fallback activation matmul failed in "
                                  "multiply_with_softmax");
                        return false;
                    }

                    int axis = softmax_axis;
                    if (axis < 0)
                    {
                        axis += 2;
                    }
                    if (axis != 0 && axis != 1)
                    {
                        LOG_ERROR("[OneDNNGemmKernel] Softmax axis must be 0, 1, or -1 (rows/cols)");
                        return false;
                    }

                    if (!apply_softmax_inplace(C, m, n, axis))
                    {
                        LOG_ERROR("[OneDNNGemmKernel] Softmax application failed in fallback path");
                        return false;
                    }

                    return true;
                }

                (void)B;

                if (m <= 0 || n <= 0 || k <= 0)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Invalid dimensions for fused multiply_with_softmax");
                    return false;
                }

                int axis = softmax_axis;
                if (axis < 0)
                {
                    axis += 2;
                }
                if (axis != 0 && axis != 1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Softmax axis must be 0, 1, or -1 (rows/cols)");
                    return false;
                }

                const auto &shape = weight_tensor_->shape();
                if (shape.size() != 2 || static_cast<int>(shape[0]) != n || static_cast<int>(shape[1]) != k)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Weight tensor shape mismatch for fused softmax path");
                    return false;
                }

                ActivationView activation_view(A, static_cast<size_t>(m), static_cast<size_t>(k));
                ActivationView output_view(C, static_cast<size_t>(m), static_cast<size_t>(n));

                if (!onednn_gemm_adapter(m, n, k,
                                         activation_view,
                                         *weight_tensor_,
                                         output_view,
                                         nullptr))
                {
                    LOG_ERROR("[OneDNNGemmKernel] OneDNN adapter matmul failed for fused softmax path");
                    return false;
                }

                if (!apply_softmax_inplace(C, m, n, axis))
                {
                    LOG_ERROR("[OneDNNGemmKernel] Softmax application failed after matmul");
                    return false;
                }

                return true;
            }

            /**
             * @brief Activation-activation GEMM using OneDNN FP32
             *
             * C = alpha * A @ B^T + beta * C (FP32 x FP32 -> FP32)
             *
             * Used for attention computation (Q @ K^T, scores @ V).
             */
            bool multiply_activations(const float *A, const float *B, float *C,
                                      int m, int n, int k,
                                      bool transpose_B = true,
                                      float alpha = 1.0f,
                                      float beta = 0.0f,
                                      const MPIContext *mpi_ctx = nullptr,
                                      int device_idx = -1) override
            {
                // Validate CPU-only execution
                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Only CPU execution supported (device_idx="
                              << device_idx << ")");
                    return false;
                }

                if (!A || !B || !C)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Null activation buffer in multiply_activations");
                    return false;
                }

                // For activation-activation, use FP32 matmul, then apply alpha/beta scaling
                const float *rhs_ptr = prepare_rhs_for_matmul(B, n, k, transpose_B);

                if (!run_onednn_fp32_matmul(A, rhs_ptr, C, m, n, k))
                {
                    return false;
                }

                // Fast path: default alpha=1, beta=0 (no-op)
                if (alpha == 1.0f && beta == 0.0f)
                {
                    return true;
                }

                const size_t total = static_cast<size_t>(m) * static_cast<size_t>(n);
                if (beta == 0.0f)
                {
                    for (size_t i = 0; i < total; ++i)
                    {
                        C[i] *= alpha;
                    }
                }
                else
                {
                    // General alpha/beta: C = alpha * C + beta * C_orig
                    // Since we don't have C_orig here, we only support beta==0 for now.
                    LOG_ERROR("[OneDNNGemmKernel] beta != 0.0f not yet supported in multiply_activations");
                    return false;
                }

                return true;
            }

            bool multiply_activations_strided(const float *A, const float *B, float *C,
                                              int m, int n, int k,
                                              int lda, int ldb, int ldc,
                                              bool transpose_B = true,
                                              float alpha = 1.0f,
                                              float beta = 0.0f,
                                              const MPIContext *mpi_ctx = nullptr,
                                              int device_idx = -1) override
            {
                (void)mpi_ctx;

                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] multiply_activations_strided only supports CPU (device_idx="
                              << device_idx << ")");
                    return false;
                }

                if (!A || !B || !C)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Null buffer in multiply_activations_strided");
                    return false;
                }

                if (m <= 0 || n <= 0 || k <= 0)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Invalid dimensions in multiply_activations_strided");
                    return false;
                }

                // Fast path: dense row-major layout for all matrices
                const bool a_row_major = (lda == k);
                const bool b_row_major = (ldb == (transpose_B ? k : n));
                const bool c_row_major = (ldc == n);

                if (a_row_major && b_row_major && c_row_major)
                {
                    // Underlying OneDNN matmul already supports arbitrary alpha/beta,
                    // so we can delegate directly here.
                    return multiply_activations(A,
                                                B,
                                                C,
                                                m,
                                                n,
                                                k,
                                                transpose_B,
                                                alpha,
                                                beta,
                                                mpi_ctx,
                                                device_idx);
                }

                // Fallback: copy to contiguous row-major A' (m×k) and B' (k×n) buffers,
                // perform GEMM, then scatter results back to C if ldc != n.
                std::vector<float> A_buf(static_cast<size_t>(m) * static_cast<size_t>(k));
                std::vector<float> B_buf(static_cast<size_t>(k) * static_cast<size_t>(n));

                // Pack A as row-major m×k from strided source (row-major with lda)
                for (int row = 0; row < m; ++row)
                {
                    const float *src = A + static_cast<size_t>(row) * static_cast<size_t>(lda);
                    float *dst = A_buf.data() + static_cast<size_t>(row) * static_cast<size_t>(k);
                    std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(k));
                }

                // Pack B into contiguous row-major k×n according to transpose_B
                // Logical layouts:
                //  - If transpose_B == false: B is k×n with leading dimension ldb (row-major),
                //    so source rows correspond to k and columns to n.
                //  - If transpose_B == true: original B is n×k, but GEMM wants k×n, so we
                //    treat the source as n rows of length k and transpose into k×n.
                if (!transpose_B)
                {
                    for (int ki = 0; ki < k; ++ki)
                    {
                        const float *src = B + static_cast<size_t>(ki) * static_cast<size_t>(ldb);
                        float *dst = B_buf.data() + static_cast<size_t>(ki) * static_cast<size_t>(n);
                        std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(n));
                    }
                }
                else
                {
                    for (int ki = 0; ki < k; ++ki)
                    {
                        for (int nj = 0; nj < n; ++nj)
                        {
                            const float *src_row = B + static_cast<size_t>(nj) * static_cast<size_t>(ldb);
                            B_buf[static_cast<size_t>(ki) * static_cast<size_t>(n) + static_cast<size_t>(nj)] =
                                src_row[static_cast<size_t>(ki)];
                        }
                    }
                }

                // We compute into a temporary contiguous buffer if C is strided.
                std::vector<float> C_tmp;
                float *C_target = nullptr;
                if (c_row_major)
                {
                    C_target = C;
                }
                else
                {
                    C_tmp.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
                    C_target = C_tmp.data();
                }

                // We packed B as k×n row-major, so we call GEMM with transpose_B=false
                if (!multiply_activations(A_buf.data(),
                                          B_buf.data(),
                                          C_target,
                                          m,
                                          n,
                                          k,
                                          false,
                                          alpha,
                                          beta,
                                          mpi_ctx,
                                          device_idx))
                {
                    return false;
                }

                if (!c_row_major)
                {
                    for (int row = 0; row < m; ++row)
                    {
                        const float *src = C_target + static_cast<size_t>(row) * static_cast<size_t>(n);
                        float *dst = C + static_cast<size_t>(row) * static_cast<size_t>(ldc);
                        std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(n));
                    }
                }

                return true;
            }

            bool multiply_activations_with_softmax(const float *A, const float *B, float *C,
                                                   int m, int n, int k,
                                                   bool transpose_B = true,
                                                   int softmax_axis = 1,
                                                   const MPIContext *mpi_ctx = nullptr,
                                                   int device_idx = -1) override
            {
                (void)mpi_ctx;

                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Fused matmul+softmax only available on CPU");
                    return false;
                }

                if (!A || !B || !C)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Null activation buffer in fused matmul+softmax");
                    return false;
                }

                if (m <= 0 || n <= 0 || k <= 0)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Invalid dimensions for fused matmul+softmax");
                    return false;
                }

                int axis = softmax_axis;
                if (axis < 0)
                {
                    axis += 2; // 2D tensor
                }
                if (axis != 0 && axis != 1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Softmax axis must be 0, 1, or -1 (rows/cols)");
                    return false;
                }

                const float *rhs_ptr = prepare_rhs_for_matmul(B, n, k, transpose_B);
                if (!run_onednn_fp32_matmul_softmax(A, rhs_ptr, C, m, n, k, axis))
                {
                    LOG_WARN("[OneDNNGemmKernel] Falling back to unfused matmul + softmax path");
                    if (!run_onednn_fp32_matmul(A, rhs_ptr, C, m, n, k))
                    {
                        return false;
                    }
                    if (!apply_softmax_inplace(C, m, n, axis))
                    {
                        LOG_ERROR("[OneDNNGemmKernel] Softmax fallback failed");
                        return false;
                    }
                }

                return true;
            }

        private:
            const TensorBase *weight_tensor_; ///< Bound weight tensor (B matrix)
        };

    } // namespace gemm_v4
} // namespace llaminar2

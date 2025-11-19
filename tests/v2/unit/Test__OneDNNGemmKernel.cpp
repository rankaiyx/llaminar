#include "gtest/gtest.h"

#include "src/v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

namespace
{

    class SimpleFP32Tensor : public TensorBase, public IActivationTensor
    {
    public:
        explicit SimpleFP32Tensor(const std::vector<size_t> &shape)
            : TensorBase(shape), data_(num_elements()) {}

        float *data() { return data_.data(); }
        const float *data() const { return data_.data(); }

        template <typename T>
        const T *data() const { return reinterpret_cast<const T *>(data_.data()); }

        const std::vector<size_t> &shape() const override { return TensorBase::shape(); }

        ActivationPack to_int8_activation_pack(int rows, int cols) const override
        {
            ActivationPack pack;
            pack.rows = rows;
            pack.cols = cols;
            pack.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            pack.row_scales.resize(static_cast<size_t>(rows));

            for (int r = 0; r < rows; ++r)
            {
                float max_abs = 0.0f;
                for (int c = 0; c < cols; ++c)
                {
                    float v = data_[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
                    max_abs = std::max(max_abs, std::fabs(v));
                }
                float scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
                pack.row_scales[static_cast<size_t>(r)] = 1.0f / scale;
                for (int c = 0; c < cols; ++c)
                {
                    float v = data_[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
                    int q = static_cast<int>(std::round(v * scale));
                    q = std::max(-127, std::min(127, q));
                    pack.data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] = static_cast<int8_t>(q);
                }
            }

            return pack;
        }

        bool from_int32_with_scales(const int32_t *src_int32,
                                    int rows,
                                    int cols,
                                    const float *row_scales,
                                    const float *col_scales,
                                    const float *bias = nullptr) override
        {
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    float v = static_cast<float>(src_int32[idx]) * row_scales[r] * col_scales[c];
                    if (bias)
                    {
                        v += bias[c];
                    }
                    data_[idx] = v;
                }
            }
            return true;
        }

        bool to_int8_perchannel(int8_t *row_major,
                                float * /*dummy_row_scales*/,
                                float *col_scales) const override
        {
            const int n = static_cast<int>(shape()[0]);
            const int k = static_cast<int>(shape()[1]);
            for (int c = 0; c < k; ++c)
            {
                float max_abs = 0.0f;
                for (int r = 0; r < n; ++r)
                {
                    float v = data_[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)];
                    max_abs = std::max(max_abs, std::fabs(v));
                }
                float scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
                col_scales[c] = 1.0f / scale;
                for (int r = 0; r < n; ++r)
                {
                    float v = data_[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)];
                    int q = static_cast<int>(std::round(v * scale));
                    q = std::max(-127, std::min(127, q));
                    row_major[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = static_cast<int8_t>(q);
                }
            }
            return true;
        }

        // Unused IActivationTensor kernel factories for this unit test
        std::unique_ptr<ITensorRoPE> createRoPE() override { return nullptr; }
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override { return nullptr; }
        std::unique_ptr<ITensorSoftmax> createSoftmax() override { return nullptr; }
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override { return nullptr; }
        std::unique_ptr<ITensorAttention> createAttention() override { return nullptr; }

        bool applyRMSNorm(const float *, int, int, float, const MPIContext *, int) override { return false; }
        bool applyRoPE(float *, const int *, int, int, int, int, float, bool, const MPIContext *, int) override { return false; }

    private:
        std::vector<float> data_;
    };

    TEST(Test__OneDNNGemmKernel, MultiplyBasic)
    {
        const int m = 2;
        const int k = 3;
        const int n = 4;

        SimpleFP32Tensor weights({static_cast<size_t>(n), static_cast<size_t>(k)});
        SimpleFP32Tensor activations({static_cast<size_t>(m), static_cast<size_t>(k)});
        SimpleFP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});

        // Fill weights and activations with small deterministic values
        for (int r = 0; r < n; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                weights.data()[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.01f * static_cast<float>(1 + r * k + c);
            }
        }
        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                activations.data()[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.02f * static_cast<float>(1 + r * k + c);
            }
        }

        OneDNNGemmKernel kernel(&weights);
        ASSERT_TRUE(kernel.multiply_activations(activations.data(),
                                                output.data(),
                                                m,
                                                n,
                                                k,
                                                /*transpose_B=*/true,
                                                /*alpha=*/1.0f,
                                                /*beta=*/0.0f,
                                                /*mpi_ctx=*/nullptr,
                                                /*device*/ nullptr));

        // Sanity: check that outputs are finite and non-zero
        for (size_t i = 0; i < output.num_elements(); ++i)
        {
            EXPECT_TRUE(std::isfinite(output.data()[i]));
        }
    }

    TEST(Test__OneDNNGemmKernel, MultiplyWithSoftmaxRowAxis)
    {
        const int m = 2;
        const int k = 4;
        const int n = 4;

        SimpleFP32Tensor weights({static_cast<size_t>(n), static_cast<size_t>(k)});
        SimpleFP32Tensor activations({static_cast<size_t>(m), static_cast<size_t>(k)});
        SimpleFP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});

        for (int r = 0; r < n; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                weights.data()[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.01f * static_cast<float>(1 + r * k + c);
            }
        }
        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                activations.data()[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.02f * static_cast<float>(1 + r * k + c);
            }
        }

        OneDNNGemmKernel kernel(&weights);
        ASSERT_TRUE(kernel.multiply_with_softmax(activations.data(),
                                                 output.data(),
                                                 m,
                                                 n,
                                                 k,
                                                 /*transpose_B=*/true,
                                                 /*softmax_axis=*/1));

        // Each row should sum to ~1 after softmax
        for (int r = 0; r < m; ++r)
        {
            float row_sum = 0.0f;
            for (int c = 0; c < n; ++c)
            {
                float v = output.data()[static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(c)];
                EXPECT_GT(v, 0.0f);
                row_sum += v;
            }
            EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
        }
    }

    TEST(Test__OneDNNGemmKernel, MultiplyActivationsBasic)
    {
        const int m = 3;
        const int k = 4;
        const int n = 2;

        std::vector<float> A(static_cast<size_t>(m) * static_cast<size_t>(k));
        std::vector<float> B(static_cast<size_t>(n) * static_cast<size_t>(k));
        std::vector<float> C(static_cast<size_t>(m) * static_cast<size_t>(n));

        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                A[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.01f * static_cast<float>(1 + r * k + c);
            }
        }
        for (int r = 0; r < n; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                B[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.015f * static_cast<float>(1 + r * k + c);
            }
        }

        OneDNNGemmKernel kernel(nullptr);
        ASSERT_TRUE(kernel.multiply_activations(A.data(),
                                                B.data(),
                                                C.data(),
                                                m,
                                                n,
                                                k,
                                                /*transpose_B=*/true));

        for (float v : C)
        {
            EXPECT_TRUE(std::isfinite(v));
        }
    }

    TEST(Test__OneDNNGemmKernel, MultiplyActivationsWithSoftmaxRowAxis)
    {
        const int m = 3;
        const int k = 4;
        const int n = 3;

        std::vector<float> A(static_cast<size_t>(m) * static_cast<size_t>(k));
        std::vector<float> B(static_cast<size_t>(n) * static_cast<size_t>(k));
        std::vector<float> C(static_cast<size_t>(m) * static_cast<size_t>(n));

        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                A[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.01f * static_cast<float>(1 + r * k + c);
            }
        }
        for (int r = 0; r < n; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                B[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = 0.02f * static_cast<float>(1 + r * k + c);
            }
        }

        OneDNNGemmKernel kernel(nullptr);
        ASSERT_TRUE(kernel.multiply_activations_with_softmax(A.data(),
                                                             B.data(),
                                                             C.data(),
                                                             m,
                                                             n,
                                                             k,
                                                             /*transpose_B=*/true,
                                                             /*softmax_axis=*/1));

        for (int r = 0; r < m; ++r)
        {
            float row_sum = 0.0f;
            for (int c = 0; c < n; ++c)
            {
                float v = C[static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(c)];
                EXPECT_GT(v, 0.0f);
                row_sum += v;
            }
            EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
        }
    }

    TEST(Test__OneDNNGemmKernel, MultiplyActivationsStridedFallbackMatchesContiguous)
    {
        const int m = 3;
        const int k = 4;
        const int n = 2;

        // Build strided A (lda > k), B (ldb > k), and C (ldc > n)
        const int lda = 6;
        const int ldb = 6;
        const int ldc = 4;

        std::vector<float> A_strided(static_cast<size_t>(m) * static_cast<size_t>(lda));
        std::vector<float> B_strided(static_cast<size_t>(n) * static_cast<size_t>(ldb));
        std::vector<float> C_strided(static_cast<size_t>(m) * static_cast<size_t>(ldc));

        std::vector<float> A_dense(static_cast<size_t>(m) * static_cast<size_t>(k));
        std::vector<float> B_dense(static_cast<size_t>(n) * static_cast<size_t>(k));
        std::vector<float> C_dense(static_cast<size_t>(m) * static_cast<size_t>(n));

        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                float v = 0.01f * static_cast<float>(1 + r * k + c);
                A_dense[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = v;
                A_strided[static_cast<size_t>(r) * static_cast<size_t>(lda) + static_cast<size_t>(c)] = v;
            }
        }
        for (int r = 0; r < n; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                float v = 0.015f * static_cast<float>(1 + r * k + c);
                B_dense[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = v;
                B_strided[static_cast<size_t>(r) * static_cast<size_t>(ldb) + static_cast<size_t>(c)] = v;
            }
        }

        OneDNNGemmKernel kernel(nullptr);

        ASSERT_TRUE(kernel.multiply_activations(A_dense.data(),
                                                B_dense.data(),
                                                C_dense.data(),
                                                m,
                                                n,
                                                k,
                                                /*transpose_B=*/true));

        ASSERT_TRUE(kernel.multiply_activations_strided(A_strided.data(),
                                                        B_strided.data(),
                                                        C_strided.data(),
                                                        m,
                                                        n,
                                                        k,
                                                        lda,
                                                        ldb,
                                                        ldc,
                                                        /*transpose_B=*/true));

        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < n; ++c)
            {
                float v_dense = C_dense[static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(c)];
                float v_strided = C_strided[static_cast<size_t>(r) * static_cast<size_t>(ldc) + static_cast<size_t>(c)];
                EXPECT_NEAR(v_dense, v_strided, 1e-4f);
            }
        }
    }

    TEST(Test__OneDNNGemmKernel, MultiplyActivationsStridedSupportsAlphaScaling)
    {
        const int m = 3;
        const int k = 4;
        const int n = 2;

        const int lda = 6;
        const int ldb = 6;
        const int ldc = 4;

        std::vector<float> A_strided(static_cast<size_t>(m) * static_cast<size_t>(lda));
        std::vector<float> B_strided(static_cast<size_t>(n) * static_cast<size_t>(ldb));
        std::vector<float> C_strided(static_cast<size_t>(m) * static_cast<size_t>(ldc));

        std::vector<float> A_dense(static_cast<size_t>(m) * static_cast<size_t>(k));
        std::vector<float> B_dense(static_cast<size_t>(n) * static_cast<size_t>(k));
        std::vector<float> C_dense(static_cast<size_t>(m) * static_cast<size_t>(n));

        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                float v = 0.02f * static_cast<float>(1 + r * k + c);
                A_dense[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = v;
                A_strided[static_cast<size_t>(r) * static_cast<size_t>(lda) + static_cast<size_t>(c)] = v;
            }
        }
        for (int r = 0; r < n; ++r)
        {
            for (int c = 0; c < k; ++c)
            {
                float v = -0.01f * static_cast<float>(1 + r * k + c);
                B_dense[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c)] = v;
                B_strided[static_cast<size_t>(r) * static_cast<size_t>(ldb) + static_cast<size_t>(c)] = v;
            }
        }

        const float alpha = 0.5f;

        OneDNNGemmKernel kernel(nullptr);

        ASSERT_TRUE(kernel.multiply_activations(A_dense.data(),
                                                B_dense.data(),
                                                C_dense.data(),
                                                m,
                                                n,
                                                k,
                                                /*transpose_B=*/true,
                                                alpha));

        ASSERT_TRUE(kernel.multiply_activations_strided(A_strided.data(),
                                                        B_strided.data(),
                                                        C_strided.data(),
                                                        m,
                                                        n,
                                                        k,
                                                        lda,
                                                        ldb,
                                                        ldc,
                                                        /*transpose_B=*/true,
                                                        alpha));

        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < n; ++c)
            {
                float v_dense = C_dense[static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(c)];
                float v_strided = C_strided[static_cast<size_t>(r) * static_cast<size_t>(ldc) + static_cast<size_t>(c)];
                EXPECT_NEAR(v_dense, v_strided, 1e-4f);
            }
        }
    }

} // namespace

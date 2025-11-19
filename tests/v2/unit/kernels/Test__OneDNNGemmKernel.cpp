#include <gtest/gtest.h>
#include "../../../../src/v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

TEST(OneDNNGemmKernel, StridedNonTransposedB)
{
    const int m = 2;
    const int k = 3;
    const int n = 4;

    float A[m * k] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f};

    const int ldb = n + 2;
    std::vector<float> B(static_cast<size_t>(k) * static_cast<size_t>(ldb), 0.0f);
    float B_dense[k * n] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f};

    for (int row = 0; row < k; ++row)
    {
        float *dst = B.data() + static_cast<size_t>(row) * static_cast<size_t>(ldb);
        std::memcpy(dst, B_dense + static_cast<size_t>(row) * static_cast<size_t>(n),
                    sizeof(float) * static_cast<size_t>(n));
    }

    std::vector<float> C(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);

    OneDNNGemmKernel kernel(nullptr);

    const int lda = k;
    const int ldc = n;

    bool ok = kernel.multiply_activations_strided(
        A,
        B.data(),
        C.data(),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        /*transpose_B=*/false,
        /*alpha=*/1.0f,
        /*beta=*/0.0f,
        nullptr,
        -1);

    ASSERT_TRUE(ok);

    float expected[m * n] = {
        1 * 1 + 2 * 5 + 3 * 9, 1 * 2 + 2 * 6 + 3 * 10,
        1 * 3 + 2 * 7 + 3 * 11, 1 * 4 + 2 * 8 + 3 * 12,
        4 * 1 + 5 * 5 + 6 * 9, 4 * 2 + 5 * 6 + 6 * 10,
        4 * 3 + 5 * 7 + 6 * 11, 4 * 4 + 5 * 8 + 6 * 12};

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(C[static_cast<size_t>(i)], expected[i], 1e-4f) << "Mismatch at index " << i;
    }
}

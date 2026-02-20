#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/rocm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_ONEDNN
#include "kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
    struct BenchConfig
    {
        std::string shape_name;
        int M;
        int N;
        int K;
    };

    struct BenchStats
    {
        double mean_ms = 0.0;
        double mean_tflops = 0.0;
        double cosine = 0.0;
    };

    enum class RatioFormat
    {
        Q4_0,
        Q4_1,
        Q5_0,
        Q5_1,
        Q6_K,
        Q4_K,
        Q5_K,
        Q3_K,
        Q2_K,
        IQ4_NL,
        IQ4_XS,
        IQ3_XXS,
        IQ3_S,
        IQ2_XXS,
        IQ2_XS,
        IQ2_S,
        IQ1_S,
        IQ1_M,
    };

    std::string formatName(RatioFormat format)
    {
        switch (format)
        {
        case RatioFormat::Q4_0:
            return "Q4_0";
        case RatioFormat::Q4_1:
            return "Q4_1";
        case RatioFormat::Q5_0:
            return "Q5_0";
        case RatioFormat::Q5_1:
            return "Q5_1";
        case RatioFormat::Q6_K:
            return "Q6_K";
        case RatioFormat::Q4_K:
            return "Q4_K";
        case RatioFormat::Q5_K:
            return "Q5_K";
        case RatioFormat::Q3_K:
            return "Q3_K";
        case RatioFormat::Q2_K:
            return "Q2_K";
        case RatioFormat::IQ4_NL:
            return "IQ4_NL";
        case RatioFormat::IQ4_XS:
            return "IQ4_XS";
        case RatioFormat::IQ3_XXS:
            return "IQ3_XXS";
        case RatioFormat::IQ3_S:
            return "IQ3_S";
        case RatioFormat::IQ2_XXS:
            return "IQ2_XXS";
        case RatioFormat::IQ2_XS:
            return "IQ2_XS";
        case RatioFormat::IQ2_S:
            return "IQ2_S";
        case RatioFormat::IQ1_S:
            return "IQ1_S";
        case RatioFormat::IQ1_M:
            return "IQ1_M";
        }
        return "Unknown";
    }

    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const std::string &value)
            : name_(name)
        {
            const char *existing = std::getenv(name_);
            had_original_ = (existing != nullptr);
            if (had_original_)
            {
                original_value_ = existing;
            }
            ::setenv(name_, value.c_str(), 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnvOverride()
        {
            if (had_original_)
            {
                ::setenv(name_, original_value_.c_str(), 1);
            }
            else
            {
                ::unsetenv(name_);
            }
            mutableDebugEnv().reload();
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        const char *name_;
        bool had_original_ = false;
        std::string original_value_;
    };

    class ROCmRatioVNNIPrefillPerfTest : public ::testing::Test
    {
    protected:
        bool has_rocm_device_ = false;

        void SetUp() override
        {
#ifdef HAVE_ROCM
            int count = 0;
            hipError_t err = hipGetDeviceCount(&count);
            has_rocm_device_ = (err == hipSuccess && count > 0);
#else
            has_rocm_device_ = false;
#endif
        }

        static double cosineSimilarity(const float *a, const float *b, size_t count)
        {
            double dot = 0.0;
            double norm_a = 0.0;
            double norm_b = 0.0;
            for (size_t i = 0; i < count; ++i)
            {
                dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
        }

        static void computeReference(const float *A, const float *B, float *C, int M, int N, int K)
        {
#ifdef HAVE_ONEDNN
            gemm_v4::run_onednn_fp32_matmul(A, B, C, M, N, K, true, 1.0f, 0.0f);
#else
            for (int m = 0; m < M; ++m)
            {
                for (int n = 0; n < N; ++n)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k)
                    {
                        acc += A[m * K + k] * B[n * K + k];
                    }
                    C[m * N + n] = acc;
                }
            }
#endif
        }

        static std::unique_ptr<TensorBase> createRatioWeights(RatioFormat format, int N, int K)
        {
            switch (format)
            {
            case RatioFormat::Q4_0:
                return TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q4_1:
                return TestTensorFactory::createQ4_1Random({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q5_0:
                return TestTensorFactory::createQ5_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q5_1:
                return TestTensorFactory::createQ5_1Random({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q6_K:
                return TestTensorFactory::createQ6_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q4_K:
                return TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q5_K:
                return TestTensorFactory::createQ5_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q3_K:
                return TestTensorFactory::createQ3_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::Q2_K:
                return TestTensorFactory::createQ2_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ4_NL:
                return TestTensorFactory::createIQ4_NLRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ4_XS:
                return TestTensorFactory::createIQ4_XSRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ3_XXS:
                return TestTensorFactory::createIQ3_XXSRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ3_S:
                return TestTensorFactory::createIQ3_SRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ2_XXS:
                return TestTensorFactory::createIQ2_XXSRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ2_XS:
                return TestTensorFactory::createIQ2_XSRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ2_S:
                return TestTensorFactory::createIQ2_SRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ1_S:
                return TestTensorFactory::createIQ1_SRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            case RatioFormat::IQ1_M:
                return TestTensorFactory::createIQ1_MRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
            }
            return nullptr;
        }

        BenchStats runFullPathPrefillBenchmark(const BenchConfig &cfg,
                                               TensorBase *weights,
                                               FP32Tensor *input,
                                               const std::vector<float> &ref,
                                               RatioFormat format,
                                               bool enable_native_prefill)
        {
            ScopedEnvOverride prefill_enabled(
                "LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL",
                enable_native_prefill ? "1" : "0");

            ROCmPackedWeights packed;
            EXPECT_TRUE(packWeightsToROCm(weights, packed));

            ROCmQuantisedGemmKernel kernel(&packed, 0);

            auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::rocm(0), 1024ULL * 1024ULL * 1024ULL);
            const auto reqs = kernel.getWorkspaceRequirements(cfg.M, cfg.N, cfg.K);
            EXPECT_TRUE(workspace->allocate(reqs));
            kernel.bindWorkspace(workspace.get());

            auto output = TestTensorFactory::createFP32({static_cast<size_t>(cfg.M), static_cast<size_t>(cfg.N)});

            constexpr int warmup_iters = 1;
            constexpr int bench_iters = 2;
            constexpr int trials = 1;

            for (int i = 0; i < warmup_iters; ++i)
            {
                if (!kernel.multiply_tensor(input, output.get(), cfg.M, cfg.N, cfg.K))
                {
                    ADD_FAILURE() << "multiply_tensor failed during warmup for mode="
                                  << (enable_native_prefill ? "native" : "ck") << " shape=" << cfg.shape_name
                                  << " format=" << formatName(format);
                    return {};
                }
            }
            (void)hipDeviceSynchronize();

            std::vector<double> samples_ms;
            for (int t = 0; t < trials; ++t)
            {
                for (int i = 0; i < bench_iters; ++i)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    if (!kernel.multiply_tensor(input, output.get(), cfg.M, cfg.N, cfg.K))
                    {
                        ADD_FAILURE() << "multiply_tensor failed during benchmark for mode="
                                      << (enable_native_prefill ? "native" : "ck") << " shape=" << cfg.shape_name
                                      << " format=" << formatName(format);
                        return {};
                    }
                    (void)hipDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                }
            }

            BenchStats stats;
            stats.mean_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / static_cast<double>(samples_ms.size());
            const double ops = 2.0 * static_cast<double>(cfg.M) * static_cast<double>(cfg.N) * static_cast<double>(cfg.K);
            stats.mean_tflops = (ops / (stats.mean_ms * 1e-3)) / 1e12;
            stats.cosine = cosineSimilarity(output->data(), ref.data(), ref.size());
            return stats;
        }

        void runFormatBenchmark(RatioFormat format)
        {
            if (!has_rocm_device_)
            {
                GTEST_SKIP() << "No ROCm device available";
            }

            const std::vector<BenchConfig> shapes = {
                {"7B AttnOut", 128, 3584, 3584},
                {"7B FFN Up", 128, 18944, 3584},
                {"7B FFN Gate", 128, 18944, 3584},
                {"7B FFN Down", 128, 3584, 18944},
                {"14B AttnOut", 128, 5120, 5120},
                {"14B FFN Up", 128, 13824, 5120},
                {"14B FFN Gate", 128, 13824, 5120},
                {"14B FFN Down", 128, 5120, 13824},
            };

            const double cosine_threshold = (format == RatioFormat::Q4_0 || format == RatioFormat::Q4_1) ? 0.98 : 0.99;

            std::cout << "\n[Perf] ROCm ratio-VNNI prefill full-path sweep\n"
                      << "[Perf] Path includes: activation quantization + prefill GEMM dispatch + scaling/epilogue\n"
                      << "[Perf] Format=" << formatName(format) << std::endl;

            for (const auto &shape : shapes)
            {
                auto weights = createRatioWeights(format, shape.N, shape.K);
                ASSERT_NE(weights, nullptr);

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(shape.M), static_cast<size_t>(shape.K)});
                std::vector<float> ref(static_cast<size_t>(shape.M) * static_cast<size_t>(shape.N));
                computeReference(input->data(), weights->fp32_data(), ref.data(), shape.M, shape.N, shape.K);

                BenchStats native_stats =
                    runFullPathPrefillBenchmark(shape, weights.get(), input.get(), ref, format, /*enable_native_prefill=*/true);
                BenchStats ck_stats =
                    runFullPathPrefillBenchmark(shape, weights.get(), input.get(), ref, format, /*enable_native_prefill=*/false);

                const double native_speedup_vs_ck = (native_stats.mean_ms > 0.0)
                                                        ? (ck_stats.mean_ms / native_stats.mean_ms)
                                                        : 0.0;

                std::cout << "  shape=" << std::left << std::setw(12) << shape.shape_name
                          << " M=" << std::setw(5) << shape.M
                          << " N=" << std::setw(6) << shape.N
                          << " K=" << std::setw(6) << shape.K
                          << " | native " << std::fixed << std::setprecision(3) << std::setw(8) << native_stats.mean_ms
                          << " ms (" << std::setprecision(3) << std::setw(7) << native_stats.mean_tflops << " TFLOPS"
                          << ", cos=" << std::setprecision(6) << native_stats.cosine << ")"
                          << " | ck " << std::setprecision(3) << std::setw(8) << ck_stats.mean_ms
                          << " ms (" << std::setprecision(3) << std::setw(7) << ck_stats.mean_tflops << " TFLOPS"
                          << ", cos=" << std::setprecision(6) << ck_stats.cosine << ")"
                          << " | native_vs_ck=" << std::setprecision(3) << native_speedup_vs_ck << "x"
                          << std::endl;

                EXPECT_GT(native_stats.cosine, cosine_threshold);
                EXPECT_GT(ck_stats.cosine, cosine_threshold);
            }
        }
    };

#define LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(TEST_NAME, FORMAT_ENUM) \
    TEST_F(ROCmRatioVNNIPrefillPerfTest, TEST_NAME)                     \
    {                                                                   \
        runFormatBenchmark(RatioFormat::FORMAT_ENUM);                   \
    }

    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q4_0, Q4_0)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q4_1, Q4_1)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q5_0, Q5_0)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q5_1, Q5_1)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q6_K, Q6_K)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q4_K, Q4_K)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q5_K, Q5_K)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q3_K, Q3_K)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_Q2_K, Q2_K)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ4_NL, IQ4_NL)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ4_XS, IQ4_XS)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ3_XXS, IQ3_XXS)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ3_S, IQ3_S)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ2_XXS, IQ2_XXS)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ2_XS, IQ2_XS)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ2_S, IQ2_S)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ1_S, IQ1_S)
    LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST(PrefillFullPath_IQ1_M, IQ1_M)

#undef LLAMINAR_DEFINE_RATIO_PREFILL_PERF_TEST
}

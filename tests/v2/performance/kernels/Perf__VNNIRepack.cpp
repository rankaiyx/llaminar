/**
 * @file Perf__VNNIRepack.cpp
 * @brief Micro-benchmark for VNNI weight repacking (deferred packing hot path).
 *
 * Measures repack cost in isolation by timing M>1 GEMM calls that trigger
 * ensureWorkspace() on each invocation.
 *
 * Matrix sizes match Qwen 2.5 7B Q8_0 projections.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <numeric>
#include <memory>
#include <vector>

#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace {

struct BenchResult {
    double avg_us;
    double min_us;
    int N, K;
};

BenchResult benchGEMM(const char *label, int M, int N, int K,
                      int warmup = 3, int iters = 8)
{
    auto weights = TestTensorFactory::createQ8_0Random({(size_t)N, (size_t)K});
    auto input = TestTensorFactory::createFP32Random({(size_t)M, (size_t)K}, -1.0f, 1.0f, 42);
    auto output = TestTensorFactory::createFP32Zeros({(size_t)M, (size_t)N});

    auto gemm = weights->createGemm();
    EXPECT_NE(gemm, nullptr);

    for (int i = 0; i < warmup; ++i)
        gemm->multiply_tensor(input.get(), output.get(), M, N, K);

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        gemm->multiply_tensor(input.get(), output.get(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / iters;
    double mn = *std::min_element(times.begin(), times.end());
    printf("  [%-20s] %4dx%-4d M=%-3d: avg=%8.1f us  min=%8.1f us\n",
           label, N, K, M, avg, mn);
    return {avg, mn, N, K};
}

} // namespace

TEST(Perf__VNNIRepack, Q8_0_IndividualProjections)
{
    printf("\n=== Q8_0 GEMM timing (includes repack) ===\n");
    printf("  Each call triggers ensureWorkspace() repack.\n\n");

    const int M = 128;
    benchGEMM("attn_q",    M, 3584, 3584);
    benchGEMM("attn_k",    M, 3584, 3584);
    benchGEMM("attn_v",    M, 3584, 3584);
    benchGEMM("ffn_gate",  M, 9216, 3584);
    benchGEMM("ffn_up",    M, 9216, 3584);
    benchGEMM("ffn_down",  M, 3584, 9216);
}

TEST(Perf__VNNIRepack, Q8_0_SequentialRepackStress)
{
    printf("\n=== Sequential repack stress (simulates layer execution) ===\n\n");

    const int M = 128;
    const int num_layers = 28;

    struct Proj {
        std::shared_ptr<TensorBase> weights;
        std::unique_ptr<ITensorGemm> gemm;
        std::unique_ptr<FP32Tensor> output;
        int N, K;
    };

    std::vector<Proj> projs;
    auto input = TestTensorFactory::createFP32Random({M, 9216}, -1.0f, 1.0f, 42);

    auto addProj = [&](int N, int K) {
        auto w = TestTensorFactory::createQ8_0Random({(size_t)N, (size_t)K});
        auto g = w->createGemm();
        auto o = TestTensorFactory::createFP32Zeros({(size_t)M, (size_t)N});
        Proj p;
        p.weights = std::move(w);
        p.gemm = std::move(g);
        p.output = std::move(o);
        p.N = N;
        p.K = K;
        projs.push_back(std::move(p));
    };

    addProj(3584, 3584); // Q
    addProj(3584, 3584); // K
    addProj(3584, 3584); // V
    addProj(9216, 3584); // gate
    addProj(9216, 3584); // up
    addProj(3584, 9216); // down

    // Warmup
    for (auto &p : projs)
        p.gemm->multiply_tensor(input.get(), p.output.get(), M, p.N, p.K);

    const int iters = 5;
    std::vector<double> layer_times;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : projs)
            p.gemm->multiply_tensor(input.get(), p.output.get(), M, p.N, p.K);
        auto t1 = std::chrono::high_resolution_clock::now();
        layer_times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    double layer_min = *std::min_element(layer_times.begin(), layer_times.end());
    double layer_avg = std::accumulate(layer_times.begin(), layer_times.end(), 0.0) / iters;

    printf("  Per-layer (6 projs): avg=%8.1f us  min=%8.1f us\n", layer_avg, layer_min);
    printf("  Full model (%d layers): est=%.1f ms\n", num_layers, num_layers * layer_min / 1000.0);
}

TEST(Perf__VNNIRepack, Q8_0_DecodePath)
{
    printf("\n=== M=1 decode path (VNNI repack + GEMV) ===\n\n");

    benchGEMM("attn_q M=1",   1, 3584, 3584, 5, 20);
    benchGEMM("ffn_gate M=1", 1, 9216, 3584, 5, 20);
    benchGEMM("ffn_down M=1", 1, 3584, 9216, 5, 20);
}

/**
 * @file Test__TurboQuantRoundtrip.cpp
 * @brief Unit tests for TurboQuant quantize → dequantize round-trip
 *
 * Tests the full pipeline:
 *   FP32 → normalize → rotate → quantize → dequantize → inverse rotate → rescale → FP32
 *
 * Validates:
 * - Roundtrip MSE for random vectors ≤ paper bounds
 * - Determinism: same input + seed → same output
 * - Zero vector handling
 * - Rotation matrix orthogonality
 * - Norm preservation
 * - TQ4 < TQ3 in MSE (more bits = better quality)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantRotation.h"
#include "kernels/cpu/turboquant/TurboQuantQuantize.h"
#include "kernels/cpu/turboquant/TurboQuantDequantize.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

using namespace llaminar2;

namespace
{

    // Helper: compute MSE between two vectors
    float compute_mse(const float *a, const float *b, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum / n;
    }

    // Helper: compute L2 norm
    float compute_norm(const float *x, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += x[i] * x[i];
        return std::sqrt(sum);
    }

    // Helper: compute cosine similarity
    float compute_cosine_similarity(const float *a, const float *b, int n)
    {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < n; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-30f || norm_b < 1e-30f) return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    float compute_dot(const float *a, const float *b, int n)
    {
        float dot = 0.0f;
        for (int i = 0; i < n; ++i)
            dot += a[i] * b[i];
        return dot;
    }

    std::vector<float> load_npy_f32(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open: " + path);

        char magic[6];
        file.read(magic, sizeof(magic));
        if (std::strncmp(magic, "\x93NUMPY", 6) != 0)
            throw std::runtime_error("Invalid NPY magic: " + path);

        uint8_t major = 0;
        uint8_t minor = 0;
        file.read(reinterpret_cast<char *>(&major), 1);
        file.read(reinterpret_cast<char *>(&minor), 1);

        uint32_t header_len = 0;
        if (major == 1) {
            uint16_t len16 = 0;
            file.read(reinterpret_cast<char *>(&len16), sizeof(len16));
            header_len = len16;
        } else if (major == 2 || major == 3) {
            file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
        } else {
            throw std::runtime_error("Unsupported NPY version in: " + path);
        }

        std::string header(header_len, '\0');
        file.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (header.find("'descr': '<f4'") == std::string::npos &&
            header.find("\"descr\": \"<f4\"") == std::string::npos)
            throw std::runtime_error("Expected little-endian float32 NPY: " + path);

        file.seekg(0, std::ios::end);
        const std::streamoff end = file.tellg();
        const std::streamoff data_offset = static_cast<std::streamoff>(6 + 2 + (major == 1 ? 2 : 4) + header_len);
        const std::streamoff data_bytes = end - data_offset;
        if (data_bytes < 0 || (data_bytes % static_cast<std::streamoff>(sizeof(float))) != 0)
            throw std::runtime_error("Invalid NPY payload size: " + path);

        std::vector<float> data(static_cast<size_t>(data_bytes / static_cast<std::streamoff>(sizeof(float))));
        file.seekg(data_offset, std::ios::beg);
        file.read(reinterpret_cast<char *>(data.data()), data_bytes);
        return data;
    }

    // Helper: generate random vector with given norm
    void generate_random_vector(float *out, int n, float target_norm, std::mt19937 &rng)
    {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < n; ++i)
            out[i] = dist(rng);
        float norm = compute_norm(out, n);
        if (norm > 1e-10f) {
            float scale = target_norm / norm;
            for (int i = 0; i < n; ++i)
                out[i] *= scale;
        }
    }

} // anonymous namespace

// ============================================================================
// Rotation Matrix Tests
// ============================================================================

TEST(Test__TurboQuantRoundtrip, RotationMatrix_Orthogonality_64)
{
    auto rot = generate_rotation_matrix(64);
    float err = verify_orthogonality(rot);
    EXPECT_LT(err, 1e-4f)
        << "64×64 rotation matrix not orthogonal: max error = " << err;
}

TEST(Test__TurboQuantRoundtrip, RotationMatrix_Orthogonality_128)
{
    auto rot = generate_rotation_matrix(128);
    float err = verify_orthogonality(rot);
    EXPECT_LT(err, 1e-3f)
        << "128×128 rotation matrix not orthogonal: max error = " << err;
}

TEST(Test__TurboQuantRoundtrip, RotationMatrix_Deterministic)
{
    auto rot1 = generate_rotation_matrix(64, 42);
    auto rot2 = generate_rotation_matrix(64, 42);
    ASSERT_EQ(rot1.matrix.size(), rot2.matrix.size());
    for (size_t i = 0; i < rot1.matrix.size(); ++i)
        EXPECT_EQ(rot1.matrix[i], rot2.matrix[i])
            << "Rotation not deterministic at index " << i;
}

TEST(Test__TurboQuantRoundtrip, RotationMatrix_DifferentSeeds)
{
    auto rot1 = generate_rotation_matrix(64, 42);
    auto rot2 = generate_rotation_matrix(64, 99);
    bool all_same = true;
    for (size_t i = 0; i < rot1.matrix.size(); ++i) {
        if (rot1.matrix[i] != rot2.matrix[i]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same)
        << "Different seeds should produce different rotation matrices";
}

TEST(Test__TurboQuantRoundtrip, Rotation_PreservesNorm)
{
    auto rot = generate_rotation_matrix(64);
    std::mt19937 rng(123);
    float input[64], output[64];
    generate_random_vector(input, 64, 2.5f, rng);

    apply_rotation(rot, input, output);

    float norm_in = compute_norm(input, 64);
    float norm_out = compute_norm(output, 64);
    EXPECT_NEAR(norm_in, norm_out, 1e-3f)
        << "Rotation should preserve L2 norm";
}

TEST(Test__TurboQuantRoundtrip, Rotation_InverseIsTranspose)
{
    auto rot = generate_rotation_matrix(64);
    std::mt19937 rng(456);
    float input[64], rotated[64], recovered[64];
    generate_random_vector(input, 64, 1.0f, rng);

    apply_rotation(rot, input, rotated);
    apply_rotation_transpose(rot, rotated, recovered);

    float mse = compute_mse(input, recovered, 64);
    EXPECT_LT(mse, 1e-6f)
        << "Π^T · Π · x should recover x, MSE = " << mse;
}

// ============================================================================
// TQ4 Roundtrip Tests (head_dim=64)
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_64_SingleVector)
{
    TurboQuantContext ctx(64);
    std::mt19937 rng(789);

    float input[64], output[64], scratch0[64], scratch1[64];
    generate_random_vector(input, 64, 1.5f, rng);

    TQ4Block_64 block;
    turboquant_quantize_tq4<64>(input, ctx, block, scratch0, scratch1);
    turboquant_dequantize_tq4<64>(block, ctx, output, scratch0);

    // Verify norm is stored correctly
    float expected_norm = compute_norm(input, 64);
    EXPECT_NEAR(block.norm, expected_norm, 1e-5f);

    // Verify cosine similarity (should be very high for 4-bit)
    float cosine = compute_cosine_similarity(input, output, 64);
    EXPECT_GT(cosine, 0.97f)
        << "TQ4-64 cosine similarity too low: " << cosine;

    float output_norm = compute_norm(output, 64);
    EXPECT_NEAR(output_norm, expected_norm, 0.01f)
        << "TQ4-64 prod reconstruction should stay close to the original norm";
}

TEST(Test__TurboQuantRoundtrip, TQ4_64_MSE_Within_Bounds)
{
    TurboQuantContext ctx(64);
    std::mt19937 rng(101);
    constexpr int NUM_VECTORS = 1000;

    float total_mse = 0.0f;
    float total_norm_sq = 0.0f;

    for (int v = 0; v < NUM_VECTORS; ++v) {
        float input[64], output[64], scratch0[64], scratch1[64];
        // Random vectors with varying norms
        generate_random_vector(input, 64, 0.5f + (v % 10) * 0.3f, rng);

        TQ4Block_64 block;
        turboquant_quantize_tq4<64>(input, ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq4<64>(block, ctx, output, scratch0);

        total_mse += compute_mse(input, output, 64) * 64;
        total_norm_sq += compute_norm(input, 64) * compute_norm(input, 64);
    }

    // Normalized MSE = MSE / E[||x||²]
    float normalized_mse = total_mse / total_norm_sq;
    EXPECT_LT(normalized_mse, 0.060f)
        << "TQ4-64 prod normalized MSE = " << normalized_mse << " exceeds the expected quality floor";
}

TEST(Test__TurboQuantRoundtrip, TQ4_64_Deterministic)
{
    TurboQuantContext ctx(64, 42, 42);
    std::mt19937 rng1(42), rng2(42);
    float input1[64], input2[64], output1[64], output2[64], scratch0[64], scratch1[64];
    generate_random_vector(input1, 64, 1.0f, rng1);
    generate_random_vector(input2, 64, 1.0f, rng2);

    TQ4Block_64 block1, block2;
    turboquant_quantize_tq4<64>(input1, ctx, block1, scratch0, scratch1);
    turboquant_quantize_tq4<64>(input2, ctx, block2, scratch0, scratch1);

    EXPECT_EQ(block1.norm, block2.norm);
    EXPECT_EQ(block1.residual_norm, block2.residual_norm);
    for (size_t i = 0; i < TQ4Block_64::MSE_BYTES; ++i)
        EXPECT_EQ(block1.mse_indices[i], block2.mse_indices[i]) << "MSE-byte mismatch at byte " << i;
    for (size_t i = 0; i < TQ4Block_64::QJL_BYTES; ++i)
        EXPECT_EQ(block1.qjl_signs[i], block2.qjl_signs[i]) << "QJL-byte mismatch at byte " << i;

    turboquant_dequantize_tq4<64>(block1, ctx, output1, scratch0);
    turboquant_dequantize_tq4<64>(block2, ctx, output2, scratch0);
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(output1[i], output2[i]) << "Dequant mismatch at element " << i;
}

TEST(Test__TurboQuantRoundtrip, TQ4_64_ZeroVector)
{
    TurboQuantContext ctx(64);
    float input[64] = {}, output[64], scratch0[64], scratch1[64];

    TQ4Block_64 block;
    turboquant_quantize_tq4<64>(input, ctx, block, scratch0, scratch1);
    EXPECT_NEAR(block.norm, 0.0f, 1e-30f);

    turboquant_dequantize_tq4<64>(block, ctx, output, scratch0);
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(output[i], 0.0f) << "Zero vector dequant non-zero at " << i;
}

// ============================================================================
// TQ4 Roundtrip Tests (head_dim=128)
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_128_MSE_Within_Bounds)
{
    TurboQuantContext ctx(128);
    std::mt19937 rng(202);
    constexpr int NUM_VECTORS = 500;

    float total_mse = 0.0f;
    float total_norm_sq = 0.0f;

    for (int v = 0; v < NUM_VECTORS; ++v) {
        float input[128], output[128], scratch0[128], scratch1[128];
        generate_random_vector(input, 128, 1.0f + (v % 5) * 0.5f, rng);

        TQ4Block_128 block;
        turboquant_quantize_tq4<128>(input, ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq4<128>(block, ctx, output, scratch0);

        total_mse += compute_mse(input, output, 128) * 128;
        total_norm_sq += compute_norm(input, 128) * compute_norm(input, 128);
    }

    float normalized_mse = total_mse / total_norm_sq;
    EXPECT_LT(normalized_mse, 0.060f)
        << "TQ4-128 prod normalized MSE = " << normalized_mse << " exceeds the expected quality floor";
}

// ============================================================================
// TQ3 Roundtrip Tests (head_dim=64)
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ3_64_SingleVector)
{
    TurboQuantContext ctx(64);
    std::mt19937 rng(303);

    float input[64], output[64], scratch0[64], scratch1[64];
    generate_random_vector(input, 64, 1.5f, rng);

    TQ3Block_64 block;
    turboquant_quantize_tq3<64>(input, ctx, block, scratch0, scratch1);
    turboquant_dequantize_tq3<64>(block, ctx, output, scratch0);

    float cosine = compute_cosine_similarity(input, output, 64);
    EXPECT_GT(cosine, 0.93f)
        << "TQ3-64 cosine similarity too low: " << cosine;

    float expected_norm = compute_norm(input, 64);
    float output_norm = compute_norm(output, 64);
    EXPECT_NEAR(output_norm, expected_norm, 0.03f)
        << "TQ3-64 prod reconstruction should stay close to the original norm";
}

TEST(Test__TurboQuantRoundtrip, TQ4_Qwen2DecodeStep0_LogitErrorRegression)
{
    const auto q_data = load_npy_f32("pytorch_qwen2_snapshots/decode_step0_layer0_Q_ROPE.npy");
    const auto k_data = load_npy_f32("pytorch_qwen2_snapshots/decode_step0_layer0_K_ROPE.npy");

    constexpr int head_dim = 64;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    ASSERT_EQ(q_data.size(), static_cast<size_t>(n_heads * head_dim));
    ASSERT_EQ(k_data.size(), static_cast<size_t>(n_kv_heads * head_dim));

    constexpr int q_heads_per_kv = n_heads / n_kv_heads;
    double total_tq4_abs_err = 0.0;
    double total_tq3_abs_err = 0.0;
    int total_logits = 0;

    TurboQuantContext ctx(head_dim, /*rotation_seed=*/31, /*projection_seed=*/131);

    for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        const float *k_head = k_data.data() + kv_head * head_dim;
        float k_tq4[head_dim];
        float k_tq3[head_dim];
        float scratch0[head_dim];
        float scratch1[head_dim];

        TQ4Block_64 block4;
        turboquant_quantize_tq4<64>(k_head, ctx, block4, scratch0, scratch1);
        turboquant_dequantize_tq4<64>(block4, ctx, k_tq4, scratch0);

        TQ3Block_64 block3;
        turboquant_quantize_tq3<64>(k_head, ctx, block3, scratch0, scratch1);
        turboquant_dequantize_tq3<64>(block3, ctx, k_tq3, scratch0);

        for (int q_head = kv_head * q_heads_per_kv; q_head < (kv_head + 1) * q_heads_per_kv; ++q_head) {
            const float *q_ptr = q_data.data() + q_head * head_dim;
            const float ref = compute_dot(q_ptr, k_head, head_dim) / std::sqrt(static_cast<float>(head_dim));
            const float tq4 = compute_dot(q_ptr, k_tq4, head_dim) / std::sqrt(static_cast<float>(head_dim));
            const float tq3 = compute_dot(q_ptr, k_tq3, head_dim) / std::sqrt(static_cast<float>(head_dim));
            total_tq4_abs_err += std::abs(static_cast<double>(tq4 - ref));
            total_tq3_abs_err += std::abs(static_cast<double>(tq3 - ref));
            ++total_logits;
        }
    }

    const double mean_tq4_abs_err = total_tq4_abs_err / static_cast<double>(total_logits);
    const double mean_tq3_abs_err = total_tq3_abs_err / static_cast<double>(total_logits);
    EXPECT_LT(mean_tq4_abs_err, mean_tq3_abs_err)
        << "TQ4 should preserve Qwen2 decode logits better than TQ3. tq4_err="
        << mean_tq4_abs_err << " tq3_err=" << mean_tq3_abs_err;
    EXPECT_TRUE(std::isfinite(mean_tq4_abs_err));
    EXPECT_TRUE(std::isfinite(mean_tq3_abs_err));
}

TEST(Test__TurboQuantRoundtrip, TQ3_64_MSE_Within_Bounds)
{
    TurboQuantContext ctx(64);
    std::mt19937 rng(404);
    constexpr int NUM_VECTORS = 1000;

    float total_mse = 0.0f;
    float total_norm_sq = 0.0f;

    for (int v = 0; v < NUM_VECTORS; ++v) {
        float input[64], output[64], scratch0[64], scratch1[64];
        generate_random_vector(input, 64, 0.5f + (v % 10) * 0.3f, rng);

        TQ3Block_64 block;
        turboquant_quantize_tq3<64>(input, ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq3<64>(block, ctx, output, scratch0);

        total_mse += compute_mse(input, output, 64) * 64;
        total_norm_sq += compute_norm(input, 64) * compute_norm(input, 64);
    }

    float normalized_mse = total_mse / total_norm_sq;
    EXPECT_LT(normalized_mse, 0.20f)
        << "TQ3-64 prod normalized MSE = " << normalized_mse << " exceeds the expected quality floor";
}

TEST(Test__TurboQuantRoundtrip, TQ3_64_ZeroVector)
{
    TurboQuantContext ctx(64);
    float input[64] = {}, output[64], scratch0[64], scratch1[64];

    TQ3Block_64 block;
    turboquant_quantize_tq3<64>(input, ctx, block, scratch0, scratch1);
    EXPECT_NEAR(block.norm, 0.0f, 1e-30f);

    turboquant_dequantize_tq3<64>(block, ctx, output, scratch0);
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(output[i], 0.0f);
}

// ============================================================================
// TQ3 Roundtrip Tests (head_dim=128)
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ3_128_MSE_Within_Bounds)
{
    TurboQuantContext ctx(128);
    std::mt19937 rng(505);
    constexpr int NUM_VECTORS = 500;

    float total_mse = 0.0f;
    float total_norm_sq = 0.0f;

    for (int v = 0; v < NUM_VECTORS; ++v) {
        float input[128], output[128], scratch0[128], scratch1[128];
        generate_random_vector(input, 128, 1.0f + (v % 5) * 0.5f, rng);

        TQ3Block_128 block;
        turboquant_quantize_tq3<128>(input, ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq3<128>(block, ctx, output, scratch0);

        total_mse += compute_mse(input, output, 128) * 128;
        total_norm_sq += compute_norm(input, 128) * compute_norm(input, 128);
    }

    float normalized_mse = total_mse / total_norm_sq;
    EXPECT_LT(normalized_mse, 0.20f)
        << "TQ3-128 prod normalized MSE = " << normalized_mse << " exceeds the expected quality floor";
}

// ============================================================================
// Comparative: TQ4 should be better than TQ3
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_BetterThanTQ3)
{
    TurboQuantContext ctx(128);
    std::mt19937 rng(606);
    constexpr int NUM_VECTORS = 500;

    float mse_tq4 = 0.0f, mse_tq3 = 0.0f;
    float total_norm_sq = 0.0f;

    for (int v = 0; v < NUM_VECTORS; ++v) {
        float input[128], out4[128], out3[128], scratch0[128], scratch1[128];
        generate_random_vector(input, 128, 1.0f, rng);

        TQ4Block_128 block4;
        turboquant_quantize_tq4<128>(input, ctx, block4, scratch0, scratch1);
        turboquant_dequantize_tq4<128>(block4, ctx, out4, scratch0);

        TQ3Block_128 block3;
        turboquant_quantize_tq3<128>(input, ctx, block3, scratch0, scratch1);
        turboquant_dequantize_tq3<128>(block3, ctx, out3, scratch0);

        mse_tq4 += compute_mse(input, out4, 128) * 128;
        mse_tq3 += compute_mse(input, out3, 128) * 128;
        total_norm_sq += compute_norm(input, 128) * compute_norm(input, 128);
    }

    float norm_mse_tq4 = mse_tq4 / total_norm_sq;
    float norm_mse_tq3 = mse_tq3 / total_norm_sq;
    EXPECT_LT(norm_mse_tq4, norm_mse_tq3)
        << "TQ4 (normalized MSE=" << norm_mse_tq4
        << ") should be better than TQ3 (normalized MSE=" << norm_mse_tq3 << ")";
}

// ============================================================================
// Batch quantize/dequantize
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_Batch_MatchesSingle)
{
    TurboQuantContext ctx(64);
    std::mt19937 rng(707);
    constexpr int N = 16;

    float input[N * 64];
    for (int i = 0; i < N * 64; ++i)
        input[i] = std::normal_distribution<float>(0.0f, 1.0f)(rng);

    // Batch quantize
    TQ4Block_64 blocks_batch[N];
    turboquant_quantize_tq4_batch<64>(input, ctx, blocks_batch, N);

    // Single quantize
    TQ4Block_64 blocks_single[N];
    float scratch0[64], scratch1[64];
    for (int v = 0; v < N; ++v)
        turboquant_quantize_tq4<64>(input + v * 64, ctx, blocks_single[v], scratch0, scratch1);

    // Compare
    for (int v = 0; v < N; ++v) {
        EXPECT_EQ(blocks_batch[v].norm, blocks_single[v].norm) << "Norm mismatch at v=" << v;
        EXPECT_EQ(blocks_batch[v].residual_norm, blocks_single[v].residual_norm) << "Residual norm mismatch at v=" << v;
        for (size_t i = 0; i < TQ4Block_64::MSE_BYTES; ++i)
            EXPECT_EQ(blocks_batch[v].mse_indices[i], blocks_single[v].mse_indices[i])
                << "MSE-byte mismatch at v=" << v << " byte=" << i;
        for (size_t i = 0; i < TQ4Block_64::QJL_BYTES; ++i)
            EXPECT_EQ(blocks_batch[v].qjl_signs[i], blocks_single[v].qjl_signs[i])
                << "QJL-byte mismatch at v=" << v << " byte=" << i;
    }
}

TEST(Test__TurboQuantRoundtrip, TQ3_Batch_MatchesSingle)
{
    TurboQuantContext ctx(64);
    std::mt19937 rng(808);
    constexpr int N = 16;

    float input[N * 64];
    for (int i = 0; i < N * 64; ++i)
        input[i] = std::normal_distribution<float>(0.0f, 1.0f)(rng);

    TQ3Block_64 blocks_batch[N];
    turboquant_quantize_tq3_batch<64>(input, ctx, blocks_batch, N);

    TQ3Block_64 blocks_single[N];
    float scratch0[64], scratch1[64];
    for (int v = 0; v < N; ++v)
        turboquant_quantize_tq3<64>(input + v * 64, ctx, blocks_single[v], scratch0, scratch1);

    for (int v = 0; v < N; ++v) {
        EXPECT_EQ(blocks_batch[v].norm, blocks_single[v].norm);
        EXPECT_EQ(blocks_batch[v].residual_norm, blocks_single[v].residual_norm);
        for (size_t i = 0; i < TQ3Block_64::MSE_BYTES; ++i)
            EXPECT_EQ(blocks_batch[v].mse_indices[i], blocks_single[v].mse_indices[i])
                << "MSE-byte mismatch at v=" << v << " byte=" << i;
        for (size_t i = 0; i < TQ3Block_64::QJL_BYTES; ++i)
            EXPECT_EQ(blocks_batch[v].qjl_signs[i], blocks_single[v].qjl_signs[i])
                << "QJL-byte mismatch at v=" << v << " byte=" << i;
    }
}

// ============================================================================
// Block size verification
// ============================================================================

TEST(Test__TurboQuantRoundtrip, BlockSizes)
{
    EXPECT_EQ(sizeof(TQ4Block_64), 40u);
    EXPECT_EQ(sizeof(TQ4Block_128), 72u);
    EXPECT_EQ(sizeof(TQ3Block_64), 32u);
    EXPECT_EQ(sizeof(TQ3Block_128), 56u);

    // Verify compression ratios vs FP32
    // FP32 for 64 elements = 256 bytes
    EXPECT_LT(sizeof(TQ4Block_64), 256u / 5);   // > 5× compression
    EXPECT_LT(sizeof(TQ3Block_64), 256u / 7);    // > 7× compression
    EXPECT_LT(sizeof(TQ4Block_128), 512u / 5);   // > 5× compression
    EXPECT_LT(sizeof(TQ3Block_128), 512u / 7);   // > 7× compression
}

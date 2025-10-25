/**
 * @file AttentionTestHarness.cpp
 * @brief Implementation of test harness for MPI attention testing (Phase 3a)
 * @author David Sanftenberg
 */

#include "AttentionTestHarness.h"
#include "MockPipeline.h"
#include "../utils/Logger.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace llaminar2
{
    namespace test
    {
        // ===== AttentionTestHarness Implementation =====

        AttentionTestHarness::AttentionTestHarness(
            int n_heads,
            int n_kv_heads,
            int head_dim,
            std::shared_ptr<MPIContext> mpi_ctx)
            : n_heads_(n_heads),
              n_kv_heads_(n_kv_heads),
              head_dim_(head_dim),
              mpi_ctx_(mpi_ctx)
        {
            // Create mock pipeline for testing
            mock_pipeline_ = std::make_unique<MockPipeline>(
                n_heads, n_kv_heads, head_dim, mpi_ctx);
        }

        bool AttentionTestHarness::runAttention(
            const float *Q,
            const float *K,
            const float *V,
            float *output,
            int seq_len,
            bool causal,
            bool use_mpi)
        {
            // Create FP32 tensors from raw data
            auto Q_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_heads_ * head_dim_)});
            auto K_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_kv_heads_ * head_dim_)});
            auto V_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_kv_heads_ * head_dim_)});
            auto output_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_heads_ * head_dim_)});

            // Copy input data to tensors
            std::memcpy(Q_tensor->mutable_data(), Q,
                        seq_len * n_heads_ * head_dim_ * sizeof(float));
            std::memcpy(K_tensor->mutable_data(), K,
                        seq_len * n_kv_heads_ * head_dim_ * sizeof(float));
            std::memcpy(V_tensor->mutable_data(), V,
                        seq_len * n_kv_heads_ * head_dim_ * sizeof(float));

            bool success = false;

            if (use_mpi && mpi_ctx_)
            {
                // MPI path: Use tensor-parallel attention
                success = mock_pipeline_->test_attention_gqa_tensor_parallel(
                    Q_tensor.get(),
                    K_tensor.get(),
                    V_tensor.get(),
                    output_tensor.get(),
                    n_heads_,
                    n_kv_heads_,
                    head_dim_,
                    causal,
                    -1); // window_size
            }
            else
            {
                // Single-rank path: Use standard attention
                success = mock_pipeline_->test_attention_gqa(
                    Q_tensor.get(),
                    K_tensor.get(),
                    V_tensor.get(),
                    output_tensor.get(),
                    n_heads_,
                    n_kv_heads_,
                    head_dim_,
                    causal,
                    -1); // window_size
            }

            // Copy output back to raw buffer
            if (success)
            {
                std::memcpy(output, output_tensor->data(),
                            seq_len * n_heads_ * head_dim_ * sizeof(float));
            }

            return success;
        }

        // ===== BaselineRunner Implementation =====

        BaselineRunner::BaselineRunner(int n_heads, int n_kv_heads, int head_dim)
            : n_heads_(n_heads),
              n_kv_heads_(n_kv_heads),
              head_dim_(head_dim)
        {
            // Create mock pipeline without MPI context (single-rank baseline)
            mock_pipeline_ = std::make_unique<MockPipeline>(
                n_heads, n_kv_heads, head_dim, nullptr);
        }

        bool BaselineRunner::runAttention(
            const float *Q,
            const float *K,
            const float *V,
            float *output,
            int seq_len,
            bool causal)
        {
            // Create FP32 tensors
            auto Q_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_heads_ * head_dim_)});
            auto K_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_kv_heads_ * head_dim_)});
            auto V_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_kv_heads_ * head_dim_)});
            auto output_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len),
                                    static_cast<size_t>(n_heads_ * head_dim_)});

            // Copy input data
            std::memcpy(Q_tensor->mutable_data(), Q,
                        seq_len * n_heads_ * head_dim_ * sizeof(float));
            std::memcpy(K_tensor->mutable_data(), K,
                        seq_len * n_kv_heads_ * head_dim_ * sizeof(float));
            std::memcpy(V_tensor->mutable_data(), V,
                        seq_len * n_kv_heads_ * head_dim_ * sizeof(float));

            // Call single-rank attention
            bool success = mock_pipeline_->test_attention_gqa(
                Q_tensor.get(),
                K_tensor.get(),
                V_tensor.get(),
                output_tensor.get(),
                n_heads_,
                n_kv_heads_,
                head_dim_,
                causal,
                -1); // window_size

            // Copy output back
            if (success)
            {
                std::memcpy(output, output_tensor->data(),
                            seq_len * n_heads_ * head_dim_ * sizeof(float));
            }

            return success;
        }

        // ===== Test Utilities Implementation =====

        namespace utils
        {
            ComparisonMetrics compareTensors(
                const float *a,
                const float *b,
                size_t count,
                float tolerance)
            {
                ComparisonMetrics metrics = {};
                metrics.total_elements = count;
                metrics.num_mismatches = 0;
                metrics.passed = true;

                double sum_abs_diff = 0.0;
                double sum_sq_diff = 0.0;
                double sum_sq_a = 0.0;
                metrics.max_abs_diff = 0.0;

                for (size_t i = 0; i < count; ++i)
                {
                    float diff = std::abs(a[i] - b[i]);
                    sum_abs_diff += diff;
                    sum_sq_diff += diff * diff;
                    sum_sq_a += a[i] * a[i];

                    if (diff > metrics.max_abs_diff)
                    {
                        metrics.max_abs_diff = diff;
                    }

                    if (diff > tolerance)
                    {
                        metrics.num_mismatches++;
                        metrics.passed = false;
                    }
                }

                metrics.mean_abs_diff = sum_abs_diff / count;
                metrics.rmse = std::sqrt(sum_sq_diff / count);

                // Relative L2 norm: ||a - b||_2 / ||a||_2
                double norm_a = std::sqrt(sum_sq_a);
                if (norm_a > 1e-10)
                {
                    metrics.rel_l2_norm = std::sqrt(sum_sq_diff) / norm_a;
                }
                else
                {
                    metrics.rel_l2_norm = 0.0;
                }

                return metrics;
            }

            void initializeTestData(float *data, size_t count, int seed)
            {
                std::mt19937 rng(seed);
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

                for (size_t i = 0; i < count; ++i)
                {
                    data[i] = dist(rng);
                }
            }

            void printMetrics(const ComparisonMetrics &metrics, const char *name)
            {
                std::cout << "=== Comparison Metrics";
                if (name && name[0])
                {
                    std::cout << ": " << name;
                }
                std::cout << " ===" << std::endl;

                std::cout << "  Max abs diff:   " << metrics.max_abs_diff << std::endl;
                std::cout << "  Mean abs diff:  " << metrics.mean_abs_diff << std::endl;
                std::cout << "  RMSE:           " << metrics.rmse << std::endl;
                std::cout << "  Rel L2 norm:    " << metrics.rel_l2_norm << std::endl;
                std::cout << "  Mismatches:     " << metrics.num_mismatches
                          << " / " << metrics.total_elements << std::endl;
                std::cout << "  Status:         " << (metrics.passed ? "PASSED" : "FAILED")
                          << std::endl;
            }

        } // namespace utils

    } // namespace test
} // namespace llaminar2

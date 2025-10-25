/**
 * @file AttentionTestHarness.h
 * @brief Test harness for MPI tensor-parallel attention (Phase 3a)
 * @author David Sanftenberg
 *
 * Provides a simplified interface for testing attention computation without
 * requiring full pipeline or ModelContext infrastructure. Enables correctness
 * validation and performance benchmarking of MPI tensor-parallel attention.
 */

#pragma once

#include <memory>
#include <vector>
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorFactory.h"

namespace llaminar2
{
    namespace test
    {
        // Forward declaration
        class MockPipeline;

        /**
         * @brief Test harness for MPI attention testing
         *
         * Wraps PipelineBase::attention_gqa_tensor_parallel() and
         * attention_gqa() to provide a simple test interface that
         * accepts raw float* pointers instead of TensorBase*.
         *
         * This allows testing without ModelContext or full pipeline.
         */
        class AttentionTestHarness
        {
        public:
            /**
             * @brief Constructor
             *
             * @param n_heads Number of query heads
             * @param n_kv_heads Number of key/value heads (for GQA)
             * @param head_dim Dimension per head
             * @param mpi_ctx MPI context (nullptr for single-rank)
             */
            AttentionTestHarness(
                int n_heads,
                int n_kv_heads,
                int head_dim,
                std::shared_ptr<MPIContext> mpi_ctx = nullptr);

            /**
             * @brief Run attention computation
             *
             * @param Q Query data [seq_len, n_heads * head_dim]
             * @param K Key data [seq_len, n_kv_heads * head_dim]
             * @param V Value data [seq_len, n_kv_heads * head_dim]
             * @param output Output buffer [seq_len, n_heads * head_dim] (pre-allocated)
             * @param seq_len Sequence length
             * @param causal Apply causal masking
             * @param use_mpi Use MPI tensor-parallel path (if mpi_ctx provided)
             * @return true on success, false on error
             */
            bool runAttention(
                const float *Q,
                const float *K,
                const float *V,
                float *output,
                int seq_len,
                bool causal = true,
                bool use_mpi = false);

            /**
             * @brief Get number of heads
             */
            int n_heads() const { return n_heads_; }

            /**
             * @brief Get number of KV heads
             */
            int n_kv_heads() const { return n_kv_heads_; }

            /**
             * @brief Get head dimension
             */
            int head_dim() const { return head_dim_; }

            /**
             * @brief Check if MPI context is available
             */
            bool has_mpi() const { return mpi_ctx_ != nullptr; }

        private:
            int n_heads_;
            int n_kv_heads_;
            int head_dim_;
            std::shared_ptr<MPIContext> mpi_ctx_;
            std::unique_ptr<MockPipeline> mock_pipeline_;
        };

        /**
         * @brief Baseline single-rank attention runner
         *
         * Always runs single-rank attention (no MPI) regardless of
         * MPI context. Used as ground truth for correctness comparison.
         */
        class BaselineRunner
        {
        public:
            /**
             * @brief Constructor
             *
             * @param n_heads Number of query heads
             * @param n_kv_heads Number of key/value heads
             * @param head_dim Dimension per head
             */
            BaselineRunner(int n_heads, int n_kv_heads, int head_dim);

            /**
             * @brief Run single-rank attention (baseline)
             *
             * @param Q Query data [seq_len, n_heads * head_dim]
             * @param K Key data [seq_len, n_kv_heads * head_dim]
             * @param V Value data [seq_len, n_kv_heads * head_dim]
             * @param output Output buffer [seq_len, n_heads * head_dim]
             * @param seq_len Sequence length
             * @param causal Apply causal masking
             * @return true on success
             */
            bool runAttention(
                const float *Q,
                const float *K,
                const float *V,
                float *output,
                int seq_len,
                bool causal = true);

        private:
            int n_heads_;
            int n_kv_heads_;
            int head_dim_;
            std::unique_ptr<MockPipeline> mock_pipeline_;
        };

        /**
         * @brief Test utilities for tensor comparison
         */
        namespace utils
        {
            /**
             * @brief Comparison metrics
             */
            struct ComparisonMetrics
            {
                double max_abs_diff;   ///< Maximum absolute difference
                double mean_abs_diff;  ///< Mean absolute difference
                double rmse;           ///< Root mean square error
                double rel_l2_norm;    ///< Relative L2 norm
                size_t num_mismatches; ///< Number of elements exceeding tolerance
                size_t total_elements; ///< Total number of elements compared
                bool passed;           ///< true if all diffs within tolerance
            };

            /**
             * @brief Compare two tensors and compute diff metrics
             *
             * @param a First tensor
             * @param b Second tensor
             * @param count Number of elements
             * @param tolerance Absolute tolerance for pass/fail
             * @return Comparison metrics
             */
            ComparisonMetrics compareTensors(
                const float *a,
                const float *b,
                size_t count,
                float tolerance = 1e-4f);

            /**
             * @brief Initialize test data deterministically
             *
             * Fills tensor with deterministic pattern for reproducibility.
             *
             * @param data Output buffer
             * @param count Number of elements
             * @param seed Random seed for pattern generation
             */
            void initializeTestData(float *data, size_t count, int seed = 42);

            /**
             * @brief Print comparison metrics
             *
             * @param metrics Metrics to print
             * @param name Optional name for the comparison
             */
            void printMetrics(const ComparisonMetrics &metrics, const char *name = "");

        } // namespace utils

    } // namespace test
} // namespace llaminar2

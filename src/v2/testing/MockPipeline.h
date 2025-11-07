/**
 * @file MockPipeline.h
 * @brief Minimal concrete pipeline for testing attention methods (Phase 3a)
 * @author David Sanftenberg
 *
 * This mock pipeline allows testing of PipelineBase attention methods without
 * requiring full model loading or ModelContext infrastructure. It exposes
 * protected attention methods via public forwarding functions.
 */

#pragma once

#include "../pipelines/PipelineBase.h"
#include "../utils/MPIContext.h"
#include <memory>
#include <vector>
#include <functional>

namespace llaminar2
{
    namespace test
    {
        /**
         * @brief Minimal concrete pipeline for testing attention methods
         *
         * This class provides the minimum implementation needed to instantiate
         * PipelineBase and call its protected attention methods. It's designed
         * solely for testing purposes and should not be used in production.
         *
         * Key features:
         * - No model loading required
         * - No weight initialization required
         * - Exposes attention methods via public forwarding
         * - Minimal memory footprint
         */
        class MockPipeline : public PipelineBase
        {
        public:
            /**
             * @brief Construct a mock pipeline for testing
             *
             * @param n_heads Number of attention heads
             * @param n_kv_heads Number of key/value heads (for GQA)
             * @param head_dim Dimension of each attention head
             * @param mpi_ctx MPI context (nullptr for single-rank testing)
             */
            MockPipeline(int n_heads,
                         int n_kv_heads,
                         int head_dim,
                         std::shared_ptr<MPIContext> mpi_ctx);

            /**
             * @brief Destructor
             */
            virtual ~MockPipeline() = default;

            // ===== Test-only public forwarding methods =====

            /**
             * @brief Test wrapper for attention_gqa (single-rank)
             *
             * Forwards to protected PipelineBase::attention_gqa()
             */
            bool test_attention_gqa(
                TensorBase *Q,
                TensorBase *K,
                TensorBase *V,
                TensorBase *output,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal,
                int window_size = -1);

            /**
             * @brief Test wrapper for attention_gqa_mpi (MPI dispatcher)
             *
             * Forwards to protected PipelineBase::attention_gqa_mpi()
             */
            bool test_attention_gqa_mpi(
                TensorBase *Q,
                TensorBase *K,
                TensorBase *V,
                TensorBase *output,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal,
                int window_size = -1);

            /**
             * @brief Test wrapper for attention_gqa_tensor_parallel (distributed)
             *
             * Forwards to protected PipelineBase::attention_gqa_tensor_parallel()
             */
            bool test_attention_gqa_tensor_parallel(
                TensorBase *Q,
                TensorBase *K,
                TensorBase *V,
                TensorBase *output,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal,
                int window_size = -1);

            // ===== Accessors =====

            int n_heads() const { return n_heads_; }
            int n_kv_heads() const { return n_kv_heads_; }
            int head_dim() const { return head_dim_; }

        protected:
            // ===== Minimal implementations of abstract PipelineBase methods =====

            /**
             * @brief Dummy forward implementation (not used in testing)
             */
            bool forward(const int *tokens, int seq_len) override
            {
                return false; // Not implemented for mock
            }

            /**
             * @brief Return mock architecture name
             */
            const char *architecture() const override
            {
                return "mock_test_pipeline";
            }

            /**
             * @brief Dummy weight names (not used in testing)
             * 
             * Returns a minimal set of weight names to ensure proper device
             * discovery. Without this, discoverActiveDevices() returns an empty
             * list which causes undefined behavior in multi-device mode.
             */
            std::vector<std::string> getAllWeightNames() const override
            {
                // Return at least one dummy weight so device discovery works
                return {"blk.0.attn_q.weight"};
            }

            /**
             * @brief Dummy buffer creation (not used in testing)
             */
            ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) override
            {
                return ActivationBuffers{}; // Empty buffers
            }

            /**
             * @brief Dummy transformer layer (not used in testing)
             */
            bool transformer_layer(int layer_idx, int seq_len) override
            {
                return false; // Not implemented for mock
            }

        private:
            int n_heads_;
            int n_kv_heads_;
            int head_dim_;
        };

    } // namespace test
} // namespace llaminar2

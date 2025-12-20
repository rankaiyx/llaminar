/**
 * @file InferenceRunner.h
 * @brief Factory for creating IInferenceRunner implementations
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file provides a factory function for creating inference runners.
 * The actual interface (IInferenceRunner) is defined in IInferenceRunner.h.
 *
 * Both PipelineBase and GraphOrchestrator directly implement IInferenceRunner,
 * so no wrapper classes are needed.
 *
 * Design Goals:
 * - Pipeline and GraphOrchestrator have NO dependencies on each other
 * - Execution path selection happens at construction time
 * - Both paths implement IInferenceRunner directly
 * - Main.cpp and ChatUI use IInferenceRunner exclusively
 *
 * @code
 * // Create runner - automatically selects path based on environment
 * auto runner = createInferenceRunner(model_ctx, mpi_ctx, device_idx, config);
 *
 * // Inference (works regardless of path)
 * runner->forward(tokens, seq_len);
 * const float* logits = runner->logits();
 * @endcode
 */

#pragma once

#include "IInferenceRunner.h"
#include "../loaders/ModelContext.h"
#include "../pipelines/PipelineConfig.h"
#include "../utils/MPIContext.h"
#include <memory>

namespace llaminar2
{
    // Forward declarations to avoid pulling in full headers
    class PipelineBase;
    class GraphOrchestrator;

    /**
     * @brief Configuration for inference runner creation
     */
    struct InferenceRunnerConfig
    {
        int max_seq_len = 4096;
        int batch_size = 1;

        // Explicit path selection (overrides environment)
        bool force_pipeline = false;
        bool force_graph = false;

        // Pipeline-specific settings (passed through when creating Pipeline)
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
    };

    /**
     * @brief Factory function to create appropriate inference runner
     *
     * Selection logic:
     * 1. If force_pipeline: use Pipeline path
     * 2. If force_graph: use Graph path
     * 3. Otherwise: use Graph path (default as of December 2025)
     *
     * @param model_ctx Model context with weights
     * @param mpi_ctx MPI context (nullptr for single-rank)
     * @param device_idx Target device
     * @param config Runner configuration
     * @return Unique pointer to IInferenceRunner, or nullptr on failure
     */
    std::unique_ptr<IInferenceRunner> createInferenceRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        const InferenceRunnerConfig &config = {});

} // namespace llaminar2

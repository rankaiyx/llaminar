/**
 * @file IInferenceRunner.h
 * @brief Interface for inference execution (Pipeline or Graph)
 * @author David Sanftenberg
 * @date December 2025
 *
 * Simple interface implemented by both PipelineBase and GraphOrchestrator,
 * enabling callers to use either execution path interchangeably.
 */

#pragma once

namespace llaminar2
{

    /**
     * @brief Execution path type
     */
    enum class ExecutionPath
    {
        PIPELINE, ///< Traditional imperative pipeline (Qwen2Pipeline)
        GRAPH     ///< Graph-based execution (GraphOrchestrator)
    };

    /**
     * @brief Interface for inference execution
     *
     * Both PipelineBase and GraphOrchestrator implement this interface,
     * enabling unified usage regardless of execution strategy.
     */
    class IInferenceRunner
    {
    public:
        virtual ~IInferenceRunner() = default;

        /**
         * @brief Run forward pass
         *
         * @param tokens Token IDs
         * @param seq_len Sequence length
         * @return true if forward succeeded
         */
        virtual bool forward(const int *tokens, int seq_len) = 0;

        /**
         * @brief Get logits from last forward pass
         *
         * @return Pointer to logits [vocab_size], or nullptr if unavailable
         */
        virtual const float *logits() const = 0;

        /**
         * @brief Get vocabulary size
         */
        virtual int vocab_size() const = 0;

        /**
         * @brief Clear KV cache (reset for new sequence)
         */
        virtual void clear_cache() = 0;

        /**
         * @brief Get current position in cache
         */
        virtual int get_position() const = 0;

        /**
         * @brief Get execution path being used
         */
        virtual ExecutionPath executionPath() const = 0;

        /**
         * @brief Get architecture name (e.g., "qwen2")
         */
        virtual const char *architecture() const = 0;
    };

} // namespace llaminar2

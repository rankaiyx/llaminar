#pragma once

#include "tokenizer_interface.h"
#include "../argument_parser.h"
#include "../mpi_transformer_pipeline.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace llaminar
{
    namespace chat
    {
        using ResponseCallback = std::function<void(const std::string &text, bool is_complete)>;

        class ResponseGenerator
        {
        public:
            ResponseGenerator(std::shared_ptr<TokenizerInterface> tokenizer,
                              std::shared_ptr<MPITransformerPipeline> pipeline,
                              const LlaminarParams &params);

            ResponseGenerator(std::shared_ptr<TokenizerInterface> tokenizer,
                              std::shared_ptr<MPITransformerPipeline> pipeline,
                              const LlaminarParams &params,
                              const MPITransformerPipeline::ModelWeights &weights);

            ~ResponseGenerator() = default;

            std::string generateResponse(const std::vector<int32_t> &prompt_tokens,
                                         ResponseCallback callback = nullptr);

            std::string generateStreamingResponse(const std::vector<int32_t> &prompt_tokens,
                                                  ResponseCallback callback);

            void setGenerationParams(float temperature, int32_t top_k,
                                     float top_p, int32_t max_tokens);

            void setStopConditions(const std::vector<int32_t> &stop_tokens,
                                   const std::vector<std::string> &stop_sequences);

        private:
            std::shared_ptr<TokenizerInterface> tokenizer_;
            std::shared_ptr<MPITransformerPipeline> pipeline_;
            const LlaminarParams &params_;
            MPITransformerPipeline::ModelWeights weights_;

            float temperature_;
            int32_t top_k_;
            float top_p_;
            int32_t max_tokens_;

            std::vector<int32_t> stop_tokens_;
            int32_t max_new_tokens_;
            std::vector<std::string> stop_sequences_;

            bool shouldStop(const std::vector<int32_t> &tokens,
                            const std::string &text) const;
            int32_t sampleToken(const std::vector<float> &logits);
            void applyTemperature(std::vector<float> &logits, float temperature);
            void applyTopK(std::vector<float> &logits, int32_t k);
            void applyTopP(std::vector<float> &logits, float p);
            std::vector<float> softmax(const std::vector<float> &logits);
            int32_t sampleFromProbs(const std::vector<float> &probs);
        };

    } // namespace chat
} // namespace llaminar

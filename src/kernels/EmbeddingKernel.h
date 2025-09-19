#pragma once

#include "../tensor.h"
#include "../kernel_base.h"
#include <string>
#include <vector>
#include <memory>

namespace llaminar
{

    /**
     * @brief Embedding lookup kernel for token embedding
     */
    class EmbeddingKernel : public KernelBase
    {
    public:
        EmbeddingKernel(size_t vocab_size, size_t embedding_dim);
        ~EmbeddingKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                     std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) const override;

        std::string getKernelType() const override { return "Embedding"; }
        size_t getExpectedInputCount() const override { return 2; }
        size_t getExpectedOutputCount() const override { return 1; }

    private:
        /**
         * @brief Core embedding lookup computation
         * @param token_ids Token ID array
         * @param embedding_table Embedding table data pointer
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param embedding_dim Embedding dimension
         */
        void computeEmbedding(const int *token_ids, const float *embedding_table,
                              float *output, size_t seq_len, size_t embedding_dim);

        size_t vocab_size_;
        size_t embedding_dim_;
    };

} // namespace llaminar
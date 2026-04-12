/**
 * @file PreparedEmbeddingWeights.cpp
 * @brief Implementation of PreparedEmbeddingWeights lifecycle
 *
 * Uses IBackend abstraction for GPU memory management — no raw CUDA/HIP calls.
 */

#include "PreparedEmbeddingWeights.h"
#include "../../backends/BackendManager.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    PreparedEmbeddingWeights::~PreparedEmbeddingWeights()
    {
        if (!device_data)
            return;

        IBackend *backend = getBackendFor(device_id);
        if (backend)
        {
            backend->free(device_data, device_id.ordinal);
        }
        else
        {
            LOG_WARN("[PreparedEmbeddingWeights] No backend for device " << device_id
                                                                         << " — leaking " << byte_size << " bytes");
        }

        device_data = nullptr;
    }

    PreparedEmbeddingWeights::PreparedEmbeddingWeights(PreparedEmbeddingWeights &&other) noexcept
        : device_data(other.device_data),
          byte_size(other.byte_size),
          blocks_per_row(other.blocks_per_row),
          vocab_size(other.vocab_size),
          vocab_offset(other.vocab_offset),
          total_vocab(other.total_vocab),
          d_model(other.d_model),
          device_id(other.device_id)
    {
        other.device_data = nullptr;
        other.byte_size = 0;
    }

    PreparedEmbeddingWeights &PreparedEmbeddingWeights::operator=(PreparedEmbeddingWeights &&other) noexcept
    {
        if (this != &other)
        {
            // Free existing allocation
            this->~PreparedEmbeddingWeights();
            device_data = other.device_data;
            byte_size = other.byte_size;
            blocks_per_row = other.blocks_per_row;
            vocab_size = other.vocab_size;
            vocab_offset = other.vocab_offset;
            total_vocab = other.total_vocab;
            d_model = other.d_model;
            device_id = other.device_id;
            other.device_data = nullptr;
            other.byte_size = 0;
        }
        return *this;
    }

} // namespace llaminar2

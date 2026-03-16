/**
 * @file CUDABatchGemmContext.h
 * @brief BatchedNativeVNNIContext for holding concatenated GEMM weights.
 *
 * Used by FusedGateUpGemmAdapter and FusedQKVGEMMStage to cache
 * concatenated NativeVNNI weights for batched GEMM dispatch.
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace llaminar2
{
    namespace cuda
    {
        // Forward-declare extern "C" free function (defined in CUDABatchGemmOps.cu)
        extern "C" void cudaBatchGemm_freeDevice(void *d_ptr);

        /**
         * @brief Holds concatenated NativeVNNI weights for batched GEMM dispatch.
         *
         * Built lazily on first prefill (M > 1) call, cached for all subsequent calls.
         * Includes a temporary output buffer for the combined [M, N_total] result.
         */
        struct BatchedNativeVNNIContext
        {
            uint8_t *d_payload = nullptr;   ///< Concatenated NativeVNNI payload
            uint16_t *d_scales = nullptr;   ///< Concatenated NativeVNNI scales
            uint16_t *d_mins = nullptr;     ///< Concatenated mins (asymmetric codebooks)
            uint32_t *d_emins = nullptr;    ///< Concatenated emins (IQ formats)
            float *d_output_temp = nullptr; ///< Temp output [M_alloc, N_total] FP32
            int n_total = 0;                ///< Sum of all projection N dims
            int n_offsets[3] = {};          ///< Starting N offset per projection
            int n_count = 0;                ///< Number of projections (2 or 3)
            int m_alloc = 0;                ///< Max M allocated for temp output
            uint8_t codebook_id = 0;
            uint32_t blocks_per_row = 0;
            bool valid = false;

            void reset()
            {
                if (d_payload)
                    cudaBatchGemm_freeDevice(d_payload);
                if (d_scales)
                    cudaBatchGemm_freeDevice(d_scales);
                if (d_mins)
                    cudaBatchGemm_freeDevice(d_mins);
                if (d_emins)
                    cudaBatchGemm_freeDevice(d_emins);
                if (d_output_temp)
                    cudaBatchGemm_freeDevice(d_output_temp);
                d_payload = nullptr;
                d_scales = nullptr;
                d_mins = nullptr;
                d_emins = nullptr;
                d_output_temp = nullptr;
                n_total = 0;
                n_count = 0;
                m_alloc = 0;
                valid = false;
            }

            ~BatchedNativeVNNIContext() { reset(); }

            BatchedNativeVNNIContext() = default;
            BatchedNativeVNNIContext(const BatchedNativeVNNIContext &) = delete;
            BatchedNativeVNNIContext &operator=(const BatchedNativeVNNIContext &) = delete;
            BatchedNativeVNNIContext(BatchedNativeVNNIContext &&o) noexcept { moveFrom(o); }
            BatchedNativeVNNIContext &operator=(BatchedNativeVNNIContext &&o) noexcept
            {
                if (this != &o)
                {
                    reset();
                    moveFrom(o);
                }
                return *this;
            }

        private:
            void moveFrom(BatchedNativeVNNIContext &o)
            {
                d_payload = o.d_payload;
                d_scales = o.d_scales;
                d_mins = o.d_mins;
                d_emins = o.d_emins;
                d_output_temp = o.d_output_temp;
                n_total = o.n_total;
                std::copy(std::begin(o.n_offsets), std::end(o.n_offsets), std::begin(n_offsets));
                n_count = o.n_count;
                m_alloc = o.m_alloc;
                codebook_id = o.codebook_id;
                blocks_per_row = o.blocks_per_row;
                valid = o.valid;
                o.d_payload = nullptr;
                o.d_scales = nullptr;
                o.d_mins = nullptr;
                o.d_emins = nullptr;
                o.d_output_temp = nullptr;
                o.valid = false;
            }
        };

    } // namespace cuda
} // namespace llaminar2

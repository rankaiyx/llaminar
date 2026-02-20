#pragma once

#include <cstdint>

namespace llaminar2
{
    namespace rocm
    {
        /**
         * @brief Versioned ratio-VNNI prefill ABI descriptor.
         *
         * Design intent:
         * - Keep the legacy fixed-parameter ABI available for compatibility.
         * - Introduce a single extensible descriptor so future ratio families
         *   (different block sizes, payload encodings, side channels) do not
         *   require repeated C ABI signature churn.
         *
         * ABI version policy:
         * - `abi_version == 2` for this descriptor layout.
         * - New descriptor versions should use a new version constant and keep
         *   older launchers supported behind compatibility wrappers.
         */
        struct ROCmRatioVNNIPrefillAbiV2Desc
        {
            uint16_t abi_version = 2;   ///< Descriptor ABI version.
            uint8_t bitwidth = 0;       ///< Quant bitwidth (Phase-1/2 currently 4).
            uint8_t codebook_id = 0;    ///< 0=linear, 4=IQ4 for current formats.
            uint8_t has_min = 0;        ///< Non-zero if payload includes explicit min channel.
            uint8_t block_size = 0;     ///< Logical elements represented per payload block.
            uint16_t payload_bytes = 0; ///< Bytes per (block, output-column) payload record.
            uint8_t ratio_bytes = 1;    ///< Bytes per ratio element (int8 today).
            uint8_t reserved0 = 0;      ///< Reserved for alignment/future flags.
            uint8_t min_bytes = 0;      ///< Bytes per min element (0=no min side-channel, int8=1 today).
            uint8_t reserved1 = 0;      ///< Reserved for alignment/future use.
            uint32_t flags = 0;         ///< Reserved feature flags.
            uint32_t blocks_per_row = 0;///< Number of quant blocks along K for one output row.
            uint32_t payload_stride_bytes = 0; ///< Byte stride between adjacent payload records.
            uint32_t ratio_stride_bytes = 0;   ///< Byte stride between adjacent ratio records.
            uint32_t min_stride_bytes = 0;     ///< Byte stride between adjacent min records.
        };

        static constexpr uint16_t ROCM_RATIO_VNNI_PREFILL_ABI_V2 = 2;
    } // namespace rocm
} // namespace llaminar2

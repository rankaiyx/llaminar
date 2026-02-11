/**
 * @file RoPEDeviceParams.h
 * @brief Device-side parameter buffer for graph-captured RoPE stages
 *
 * Like AttentionDeviceParams, this struct lives in a device buffer and is
 * updated via H2D memcpy before each graph replay. The kernel reads from
 * the device buffer instead of frozen scalar arguments, allowing pos_offset
 * to change between graph replays.
 *
 * Flow:
 * 1. During graph capture: kernel launch with device_params pointer is recorded
 * 2. Between replays: updateDynamicParams() updates pinned host copy
 * 3. During replay: captured H2D memcpy re-reads updated host value → kernel sees new pos_offset
 */

#pragma once

namespace llaminar2
{
    namespace rope
    {
        struct RoPEDeviceParams
        {
            int pos_offset = 0; ///< Position offset for decode (seq_len=1) or contiguous positions
        };
    } // namespace rope
} // namespace llaminar2

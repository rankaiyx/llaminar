/**
 * @file CPUFlashAttentionKernelT.cpp
 * @brief Explicit template instantiations for CPUFlashAttentionKernelT.
 *
 * Because `CPUFlashAttentionKernelT` is a class template whose full definition
 * lives in the header, the compiler needs to know which instantiations to
 * generate object code for.  This file explicitly instantiates the three
 * supported activation precisions so that:
 *
 *   1. The template code is compiled **exactly once** (in this translation unit)
 *      rather than in every file that includes the header.
 *   2. Link times are shorter and binary size is smaller.
 *   3. The `extern template` declarations in the header suppress implicit
 *      instantiation in other TUs, avoiding ODR issues.
 *
 * If you add a new `ActivationPrecision` variant (e.g. `Q8_1`), you must add
 * a corresponding `template class` line here **and** an `extern template` line
 * in the header.
 *
 * @see CPUFlashAttentionKernelT.h  The full template definition.
 */

#include "CPUFlashAttentionKernelT.h"

namespace llaminar2
{
    // --- Explicit instantiations for all supported precisions ---

    /** @brief FP32 instantiation — the fully-optimised flash-attention path. */
    template class CPUFlashAttentionKernelT<ActivationPrecision::FP32>;

    /** @brief BF16 instantiation — returns false for non-FP32 operations. */
    template class CPUFlashAttentionKernelT<ActivationPrecision::BF16>;

    /** @brief FP16 instantiation — returns false for non-FP32 operations. */
    template class CPUFlashAttentionKernelT<ActivationPrecision::FP16>;

} // namespace llaminar2

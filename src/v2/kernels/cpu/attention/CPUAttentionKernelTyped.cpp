/**
 * @file CPUAttentionKernelTyped.cpp
 * @brief Explicit instantiations of CPUAttentionKernelTyped template
 *
 * Explicit instantiations reduce compilation time and binary size.
 * All precision variants (FP32, BF16, FP16, Q8_1) compiled once here.
 *
 * @author David Sanftenberg
 */

#include "CPUAttentionKernelTyped.h"

namespace llaminar2
{
    // Explicit instantiations for all supported activation precisions
    // Compiled once, reused across translation units

    template class CPUAttentionKernelTyped<ActivationPrecision::FP32>;
    template class CPUAttentionKernelTyped<ActivationPrecision::BF16>;
    template class CPUAttentionKernelTyped<ActivationPrecision::FP16>;
    template class CPUAttentionKernelTyped<ActivationPrecision::Q8_1>;

} // namespace llaminar2

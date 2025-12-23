/**
 * @file CPUAttentionKernelT.cpp
 * @brief Explicit instantiations of CPUAttentionKernelT template
 *
 * Explicit instantiations reduce compilation time and binary size.
 * All precision variants (FP32, BF16, FP16, Q8_1) compiled once here.
 *
 * @author David Sanftenberg
 */

#include "CPUAttentionKernelT.h"

namespace llaminar2
{
    // Explicit instantiations for all supported activation precisions
    // Compiled once, reused across translation units

    template class CPUAttentionKernelT<ActivationPrecision::FP32>;
    template class CPUAttentionKernelT<ActivationPrecision::BF16>;
    template class CPUAttentionKernelT<ActivationPrecision::FP16>;
    template class CPUAttentionKernelT<ActivationPrecision::Q8_1>;

} // namespace llaminar2

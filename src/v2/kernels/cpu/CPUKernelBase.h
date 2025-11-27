/**
 * @file CPUKernelBase.h
 * @brief Base class for all CPU kernel implementations
 * @author David Sanftenberg
 * @date 2025-11-22
 */

#pragma once

namespace llaminar2
{
    /**
     * @brief Base class for all CPU kernel implementations
     *
     * Provides common functionality and interface for CPU-based kernels.
     * All CPU kernel classes should inherit from this base.
     */
    class CPUKernelBase
    {
    public:
        virtual ~CPUKernelBase() = default;

    protected:
        CPUKernelBase() = default;
    };

} // namespace llaminar2

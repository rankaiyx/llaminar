#pragma once

#include <vector>
#include <memory>
#include <string>
#include "tensors/tensor_base.h"

namespace llaminar
{
    /**
     * @brief Base interface for all computational kernels
     *
     * This interface provides a common API for all kernels to enable
     * polymorphic execution within the compute graph framework.
     */
    class KernelBase
    {
    public:
        virtual ~KernelBase() = default;

        /**
         * @brief Execute the kernel computation
         * @param inputs Vector of input tensors
         * @param outputs Vector of output tensors (should be pre-allocated)
         * @return true if execution succeeded, false otherwise
         */
        virtual bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                             std::vector<std::shared_ptr<TensorBase>> &outputs) = 0;

        /**
         * @brief Validate input and output tensor shapes and types
         * @param inputs Input tensors to validate
         * @param outputs Output tensors to validate
         * @return true if tensors are valid, false otherwise
         */
        virtual bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                              const std::vector<std::shared_ptr<TensorBase>> &outputs) const = 0;

        /**
         * @brief Get the kernel type name for debugging/logging
         * @return String identifying the kernel type
         */
        virtual std::string getKernelType() const = 0;

        /**
         * @brief Get expected number of input tensors
         * @return Number of input tensors this kernel expects
         */
        virtual size_t getExpectedInputCount() const = 0;

        /**
         * @brief Get expected number of output tensors
         * @return Number of output tensors this kernel produces
         */
        virtual size_t getExpectedOutputCount() const = 0;
    };

} // namespace llaminar
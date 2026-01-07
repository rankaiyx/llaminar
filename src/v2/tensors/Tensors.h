/**
 * @file Tensors.h
 * @brief CPU tensor classes
 *
 * This file provides the CPU tensor implementation classes.
 * After refactoring, the implementation lives in tensors/cpu/CPUTensors.h.
 *
 * The actual class name is `CPUTensorBase` (defined in cpu/CPUTensors.h).
 * A backward compatibility alias `TensorBase` is provided so that existing
 * code throughout the codebase continues to work without modification.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

// Include the actual CPU tensor implementation
// The main class is CPUTensorBase (defined in cpu/CPUTensors.h)
#include "cpu/CPUTensors.h"

namespace llaminar2
{
    /**
     * @brief Backward compatibility alias for CPUTensorBase
     *
     * The class was renamed from TensorBase to CPUTensorBase to clarify
     * it's the CPU implementation (vs future GPU tensor classes).
     *
     * This alias maintains backward compatibility with ~300+ usages of
     * TensorBase throughout the codebase outside tensors/cpu/.
     *
     * Both names refer to the same class:
     *   - CPUTensorBase: The actual class name (in cpu/CPUTensors.h)
     *   - TensorBase: Backward compatibility alias (this file)
     */
    using TensorBase = CPUTensorBase;

} // namespace llaminar2

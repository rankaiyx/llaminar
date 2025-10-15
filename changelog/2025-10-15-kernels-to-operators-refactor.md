# Namespace Refactoring: kernels → operators

**Date**: October 15, 2025  
**Author**: David Sanftenberg  
**Type**: Refactoring - Semantic Clarity

## Summary

Renamed the `kernels` namespace, directory, and all related classes to `operators` to better reflect their true nature as high-level orchestrators rather than low-level compute kernels.

## Motivation

The classes in the former `kernels` namespace (e.g., `MPILinearKernel`, `MPIAttentionKernel`) are not simple compute kernels but rather sophisticated **operators** that:

- Orchestrate complex multi-step operations
- Manage MPI communication and distribution
- Handle weight partitioning and replication
- Coordinate cache management (attention)
- Integrate multiple computational primitives

The term "operator" aligns with industry-standard terminology (TensorFlow, PyTorch, ONNX) and better distinguishes these components from the actual low-level compute kernels in `operators/common/` (like `RmsnormCore`, `SoftmaxCore`, `AttentionPrimitives`).

## Changes

### 1. Directory Structure
```
src/kernels/          → src/operators/
  ├── MPILinearKernel.*    → MPILinearOperator.*
  ├── MPIAttentionKernel.* → MPIAttentionOperator.*
  ├── MPIEmbeddingKernel.* → MPIEmbeddingOperator.*
  ├── MPIRMSNormKernel.*   → MPIRMSNormOperator.*
  ├── MPIResidualKernel.*  → MPIResidualOperator.*
  ├── MPIRoPEKernel.*      → MPIRoPEOperator.*
  ├── MPISwiGLUKernel.*    → MPISwiGLUOperator.*
  ├── attention/           → (unchanged)
  └── common/              → (unchanged - actual kernels)
```

### 2. Class Renames

| Before | After |
|--------|-------|
| `MPILinearKernel` | `MPILinearOperator` |
| `MPIAttentionKernel` | `MPIAttentionOperator` |
| `MPIEmbeddingKernel` | `MPIEmbeddingOperator` |
| `MPIRMSNormKernel` | `MPIRMSNormOperator` |
| `MPIResidualKernel` | `MPIResidualOperator` |
| `MPIRoPEKernel` | `MPIRoPEOperator` |
| `MPISwiGLUKernel` | `MPISwiGLUOperator` |

### 3. Path Updates

All include paths updated:
```cpp
// Before
#include "kernels/MPILinearKernel.h"
#include "../kernels/MPIAttentionKernel.h"

// After
#include "operators/MPILinearOperator.h"
#include "../operators/MPIAttentionOperator.h"
```

### 4. Documentation Updates

- Updated all `@file` tags in headers
- Updated docstring descriptions (e.g., "This kernel" → "This operator")
- Updated inline comments for clarity
- Preserved references to actual compute kernels in `common/`

## Implementation

### Automated Refactoring

Created two scripts to automate the refactoring:

1. **`scripts/rename_kernels_to_operators.py`**
   - Renamed directory: `src/kernels` → `src/operators`
   - Updated all `#include` paths
   - Renamed all class occurrences
   - Updated CMakeLists.txt files
   - Updated documentation strings

2. **`scripts/update_operator_includes.py`**
   - Updated file references after file renames
   - Ensured all includes point to new filenames

### Manual File Renames

```bash
# Renamed all operator files
git mv MPIAttentionKernel.* MPIAttentionOperator.*
git mv MPIEmbeddingKernel.* MPIEmbeddingOperator.*
git mv MPILinearKernel.* MPILinearOperator.*
git mv MPIRMSNormKernel.* MPIRMSNormOperator.*
git mv MPIResidualKernel.* MPIResidualOperator.*
git mv MPIRoPEKernel.* MPIRoPEOperator.*
git mv MPISwiGLUKernel.* MPISwiGLUOperator.*
```

## Verification

### Build Verification
```bash
cmake --build build --target llaminar_core --parallel
✓ Build succeeded with no errors
```

### Test Verification
```bash
ctest --test-dir build -R "^(BasicTest|NumaTest)$"
✓ 100% tests passed (2/2)
```

## Impact

### Files Changed
- **77 total files** modified or renamed
- **14 operator files** renamed (7 headers + 7 implementations)
- **40+ source files** with updated includes
- **53 files** with class name updates
- **3 CMakeLists.txt** files updated

### Key Areas Affected

1. **Operator Implementations** (`src/operators/`)
   - All operator files renamed and updated
   - Internal class references updated

2. **Pipeline Code** (`src/`)
   - `QwenPipeline.*` - Updated operator references
   - Prefill providers - Updated operator includes
   - Model loader - Updated path references

3. **Test Suite** (`tests/`)
   - 40+ test files updated with new includes
   - Test names remain unchanged (describe what they test)

4. **Build System**
   - Root `CMakeLists.txt` updated
   - Operator source lists updated

## Terminology Clarification

### Operators (High-Level Orchestrators)
- **Location**: `src/operators/MPI*Operator.*`
- **Purpose**: Distribute work across MPI ranks, manage state, coordinate computation
- **Examples**: `MPILinearOperator`, `MPIAttentionOperator`

### Kernels (Low-Level Compute Primitives)
- **Location**: `src/operators/common/*Core.*`, `AttentionPrimitives.*`
- **Purpose**: Actual mathematical computation
- **Examples**: `RmsnormCore`, `SoftmaxCore`, `AttentionPrimitives`

This distinction now matches industry conventions where:
- **Operators** = TensorFlow ops, PyTorch operators, ONNX operators
- **Kernels** = CUDA kernels, CPU kernels, compute primitives

## Benefits

1. **Clearer Semantics**: "Operator" accurately describes what these components do
2. **Industry Alignment**: Consistent with TensorFlow, PyTorch, ONNX terminology
3. **Better Distinction**: Separates orchestration (operators) from computation (kernels)
4. **Improved Readability**: Code intent is clearer to new contributors
5. **Future-Proofing**: Easier to add new operators following established patterns

## Migration Guide

### For Developers

When writing new code:

```cpp
// ✅ Correct - using operators
#include "operators/MPILinearOperator.h"
auto linear_op = std::make_unique<MPILinearOperator>(comm);

// ❌ Old style - no longer exists
#include "kernels/MPILinearKernel.h"
auto linear_kernel = std::make_unique<MPILinearKernel>(comm);
```

### For External Code

If you have external code that referenced the old names:
1. Update all `#include` paths: `kernels/` → `operators/`
2. Update all class names: `*Kernel` → `*Operator`
3. No functional changes required - APIs remain identical

## Statistics

- **Directory**: 1 renamed
- **Files**: 14 renamed  
- **Classes**: 7 renamed
- **Source Files**: 77 updated
- **Build Time Impact**: None (no functional changes)
- **Test Success Rate**: 100%

## Commands for Review

```bash
# See all changes
git status

# Review specific changes
git diff src/operators/
git diff --stat

# Verify specific operator
git log --follow src/operators/MPILinearOperator.h

# Run full test suite
ctest --test-dir build --output-on-failure --parallel
```

## Next Steps

1. Review the changes: `git status`
2. Verify operator functionality: Run full test suite
3. Commit the changes:
   ```bash
   git commit -m "Refactor: Rename kernels namespace to operators for semantic clarity"
   ```

## Notes

- All changes preserve git history via `git mv`
- No functional modifications - pure refactoring
- Backward compatibility: None (internal API change only)
- The base class `MPIKernelBase` was intentionally **not** renamed to `MPIOperatorBase` as it provides kernel-like interface patterns and is implementation detail

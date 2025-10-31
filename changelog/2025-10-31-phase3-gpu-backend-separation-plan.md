# Phase 3: GPU Backend Separation - Solution Design

**Date**: October 31, 2025  
**Session**: 8 (Phase 3 Planning)  
**Author**: David Sanftenberg  
**Status**: Planning Phase  

---

## Problem Statement

**Blocker**: CUDA and ROCm headers cannot coexist in the same compilation unit due to ~100+ type redefinitions.

**Conflicting Types** (both backends define):
- Vector types: `dim3`, `float1-4`, `double1-4`, `int1-4`, `uint1-4`, etc.
- CUDA headers: `/usr/local/cuda/targets/x86_64-linux/include/vector_types.h`
- ROCm headers: `/opt/rocm/include/hip/amd_detail/amd_hip_vector_types.h`

**Current State**:
- Both backends disabled (`HAVE_CUDA=OFF`, `HAVE_ROCM=OFF`)
- MPIStager written with conditional compilation (`#ifdef HAVE_CUDA / HAVE_ROCM`)
- GQAAttention integrated with staging infrastructure
- 8/8 MPIStaging tests passing (CPU-only)

---

## Solution Options Analysis

### Option 1: Separate Compilation Units ✅ **RECOMMENDED**

**Approach**: Create backend-specific `.cu` and `.cpp` files with mutually exclusive compilation.

**Architecture**:
```
backends/
├── BackendInterface.h       # Pure virtual interface (no GPU headers)
├── CPUBackend.cpp            # CPU-only implementation
├── cuda/
│   ├── CUDABackend.cu        # CUDA-specific (includes cuda_runtime.h)
│   ├── CUDABackend.h         # CUDA public API
│   └── CUDAKernels.cu        # CUDA kernel implementations
└── rocm/
    ├── ROCmBackend.cpp       # ROCm-specific (includes hip/hip_runtime.h)
    ├── ROCmBackend.h         # ROCm public API
    └── ROCmKernels.cpp       # HIP kernel implementations
```

**Key Benefits**:
- ✅ **Clean separation**: Headers never coexist in same compilation unit
- ✅ **Well-tested pattern**: Standard approach in multi-GPU projects (PyTorch, TensorFlow)
- ✅ **Type-safe**: Virtual interface prevents mixing GPU types
- ✅ **Easy to maintain**: Clear backend boundaries

**Limitations**:
- ❌ **Mutually exclusive**: `-DHAVE_CUDA=ON` XOR `-DHAVE_ROCM=ON` (not both)
- ⚠️ **Runtime selection unavailable**: Must recompile to switch backends

**Implementation Steps** (2-3 days):

1. **Create backend interface** (`backends/BackendInterface.h`):
   ```cpp
   class IComputeBackend {
   public:
       virtual ~IComputeBackend() = default;
       
       // Memory operations (no GPU types exposed)
       virtual bool deviceToHost(const void* device_ptr, void* host_ptr, 
                                size_t bytes, int device_id) = 0;
       virtual bool hostToDevice(const void* host_ptr, void* device_ptr,
                                size_t bytes, int device_id) = 0;
       virtual bool synchronize(int device_id) = 0;
       
       // Query
       virtual std::string backend_name() const = 0;
       virtual int device_count() const = 0;
   };
   ```

2. **Implement CUDA backend** (`backends/cuda/CUDABackend.cu`):
   ```cpp
   #include "CUDABackend.h"
   #include <cuda_runtime.h>  // OK: .cu file
   
   bool CUDABackend::deviceToHost(const void* device_ptr, void* host_ptr,
                                  size_t bytes, int device_id) {
       cudaSetDevice(device_id);
       cudaError_t err = cudaMemcpy(host_ptr, device_ptr, bytes, 
                                    cudaMemcpyDeviceToHost);
       return (err == cudaSuccess);
   }
   ```

3. **Implement ROCm backend** (`backends/rocm/ROCmBackend.cpp`):
   ```cpp
   #include "ROCmBackend.h"
   #include <hip/hip_runtime.h>  // OK: .cpp file, no CUDA
   
   bool ROCmBackend::deviceToHost(const void* device_ptr, void* host_ptr,
                                  size_t bytes, int device_id) {
       hipSetDevice(device_id);
       hipError_t err = hipMemcpy(host_ptr, device_ptr, bytes,
                                  hipMemcpyDeviceToHost);
       return (err == hipSuccess);
   }
   ```

4. **Update MPIStager** (`src/v2/utils/MPIStager.cpp`):
   ```cpp
   #include "../../backends/BackendInterface.h"
   
   #ifdef HAVE_CUDA
   #include "../../backends/cuda/CUDABackend.h"
   static IComputeBackend* g_backend = new CUDABackend();
   #elif HAVE_ROCM
   #include "../../backends/rocm/ROCmBackend.h"
   static IComputeBackend* g_backend = new ROCmBackend();
   #else
   static IComputeBackend* g_backend = nullptr;  // CPU-only
   #endif
   
   void MPIStager::deviceToHost(const void* device_ptr, void* host_ptr,
                               size_t bytes, int device_id) {
       if (g_backend) {
           g_backend->deviceToHost(device_ptr, host_ptr, bytes, device_id);
       }
   }
   ```

5. **Update CMakeLists.txt** (mutually exclusive):
   ```cmake
   option(HAVE_CUDA "Enable CUDA backend" OFF)
   option(HAVE_ROCM "Enable ROCm backend" OFF)
   
   # Enforce mutual exclusion
   if(HAVE_CUDA AND HAVE_ROCM)
       message(FATAL_ERROR "Cannot enable both CUDA and ROCm backends. Choose one.")
   endif()
   
   if(HAVE_CUDA)
       enable_language(CUDA)
       add_library(cuda_backend STATIC backends/cuda/CUDABackend.cu)
       target_link_libraries(llaminar2_core cuda_backend CUDA::cudart)
   endif()
   
   if(HAVE_ROCM)
       find_package(hip REQUIRED)
       add_library(rocm_backend STATIC backends/rocm/ROCmBackend.cpp)
       target_link_libraries(llaminar2_core rocm_backend hip::host)
   endif()
   ```

6. **Fix NVML/ROCm API issues**:
   - NVML: `nvmlDeviceGetNumaNode()` → `nvmlDeviceGetNumaNodeId()` (correct API)
   - ROCm: `hipDeviceProp_tR0600.gcnArch` → Find correct property name

7. **Add GPU staging tests**:
   ```cpp
   TEST(MPIStagingGPU, CUDAMemcpyCorrectness) {
       // Requires real GPU (skip if device_count() == 0)
       TensorFactory factory(mpi_ctx);
       auto tensor = factory.createFP32({1024}, /*device_index=*/0);
       
       // Stage to host
       auto host_buffer = MPIStager::toHost(tensor.get());
       
       // Verify data
       for (size_t i = 0; i < 1024; ++i) {
           EXPECT_FLOAT_EQ(host_buffer[i], expected_values[i]);
       }
   }
   ```

**Estimated Effort**: 2-3 days
- Day 1: Backend interface + CUDA implementation + CMake
- Day 2: ROCm implementation + MPIStager refactor
- Day 3: GPU tests + API fixes + validation

---

### Option 2: Runtime Plugin Architecture

**Approach**: Compile CUDA and ROCm as separate shared libraries, load at runtime.

**Architecture**:
```
backends/
├── BackendPlugin.h           # Plugin interface
├── PluginLoader.cpp          # dlopen/dlsym wrapper
├── libcuda_backend.so        # CUDA plugin (separate build)
└── librocm_backend.so        # ROCm plugin (separate build)
```

**Key Benefits**:
- ✅ **Both backends available**: Single binary can use either GPU type
- ✅ **Runtime selection**: Choose backend via environment variable
- ✅ **Isolated compilation**: Plugins never see each other's headers

**Limitations**:
- ❌ **Complex build**: Need separate CMake targets for each plugin
- ❌ **Dynamic loading overhead**: Symbol resolution at runtime
- ❌ **ABI stability**: Plugins must match llaminar2_core version
- ❌ **Deployment complexity**: Must distribute 3 binaries

**Estimated Effort**: 4-5 days (higher complexity)

**Recommendation**: **NOT recommended** for this project. Plugin architecture is overkill when:
- We don't need both backends simultaneously
- Most users will only have one GPU type
- Complexity outweighs benefits

---

### Option 3: Namespace Isolation + Forward Declarations

**Approach**: Wrap CUDA/ROCm headers in namespaces to avoid collisions.

**Architecture**:
```cpp
namespace cuda {
#include <cuda_runtime.h>
}

namespace rocm {
#include <hip/hip_runtime.h>
}

// Use qualified names
cuda::cudaMemcpy(...);
rocm::hipMemcpy(...);
```

**Key Benefits**:
- ✅ **Both backends in one binary**: Theoretically possible
- ✅ **No build changes**: Single compilation unit

**Limitations**:
- ❌ **Vendor headers not designed for this**: May break with macro pollution
- ❌ **High risk**: Template instantiation leaks across namespaces
- ❌ **Maintenance burden**: Every CUDA/ROCm update risks breakage
- ❌ **Undefined behavior**: Wrapping system headers in namespaces is non-standard

**Estimated Effort**: 3-4 days (high risk of failure)

**Recommendation**: **NOT recommended**. Violates C++ standards and relies on undefined behavior.

---

## Recommended Solution: **Option 1 - Separate Compilation Units**

**Rationale**:
1. **Reliability**: Well-tested industry standard (PyTorch, TensorFlow, JAX all use this)
2. **Simplicity**: Clear backend boundaries, easy to reason about
3. **Safety**: Type system enforces separation via virtual interface
4. **Maintainability**: Each backend is self-contained, changes isolated
5. **Speed**: 2-3 day implementation vs 4-5 days (Option 2) or high-risk (Option 3)

**Trade-off**: Users must recompile to switch between CUDA and ROCm. This is acceptable because:
- Most systems have **either** NVIDIA or AMD GPUs, not both
- Recompilation is a one-time cost during setup
- Clear build-time errors prevent misconfiguration

---

## Phase 3 Implementation Plan (Option 1)

**Timeline**: 2-3 days

### Day 1: Backend Interface + CUDA Implementation (6-8 hours)

**Tasks**:
1. ✅ Create `backends/BackendInterface.h` (pure virtual interface)
2. ✅ Create `backends/cuda/CUDABackend.h` + `CUDABackend.cu`
3. ✅ Implement CUDA memory operations (deviceToHost, hostToDevice, synchronize)
4. ✅ Update CMakeLists.txt (add CUDA backend target)
5. ✅ Fix NVML API: `nvmlDeviceGetNumaNode()` → `nvmlDeviceGetNumaNodeId()`
6. ✅ Verify CUDA backend compiles (`-DHAVE_CUDA=ON`)

**Deliverables**:
- Clean CUDA compilation
- No header conflicts
- Basic CUDA backend functional

### Day 2: ROCm Implementation + MPIStager Refactor (6-8 hours)

**Tasks**:
1. ✅ Create `backends/rocm/ROCmBackend.h` + `ROCmBackend.cpp`
2. ✅ Implement ROCm memory operations
3. ✅ Fix ROCm API: `hipDeviceProp_tR0600.gcnArch` → correct property
4. ✅ Update CMakeLists.txt (add ROCm backend target, enforce mutual exclusion)
5. ✅ Refactor `MPIStager.cpp` to use `IComputeBackend` interface
6. ✅ Verify ROCm backend compiles (`-DHAVE_ROCM=ON`)

**Deliverables**:
- Clean ROCm compilation
- MPIStager backend-agnostic
- CPU/CUDA/ROCm builds all work

### Day 3: GPU Tests + Validation (6-8 hours)

**Tasks**:
1. ✅ Create `Test__CUDAStaging.cpp` (GPU memory tests)
2. ✅ Create `Test__ROCmStaging.cpp` (GPU memory tests)
3. ✅ Add GPU tensor creation to TensorFactory
4. ✅ Measure GPU staging performance (compare to CPU baseline)
5. ✅ Update GQAAttention tests to run on GPU
6. ✅ Full validation: CPU-only, CUDA-only, ROCm-only builds

**Deliverables**:
- GPU staging tests passing
- Performance metrics (GPU vs CPU transfer times)
- Documentation: Build instructions for each backend

---

## Testing Strategy

### CPU-Only Build (Baseline)
```bash
cmake -B build_cpu -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu --parallel
# All 73 unit tests + 8 integration tests should pass
```

### CUDA Build
```bash
cmake -B build_cuda -S src/v2 -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON
cmake --build build_cuda --parallel
# Requires NVIDIA GPU (skip tests on CPU-only systems)
```

### ROCm Build
```bash
cmake -B build_rocm -S src/v2 -DCMAKE_BUILD_TYPE=Release -DHAVE_ROCM=ON
cmake --build build_rocm --parallel
# Requires AMD GPU (skip tests on NVIDIA-only systems)
```

### Test Coverage

**Existing Tests** (continue passing):
- ✅ All 73 V2 unit tests (CPU-only)
- ✅ 8/8 MPIStaging tests (CPU tensors)
- ✅ GQAAttention integration tests

**New Tests** (GPU-specific):
- ✅ CUDAMemcpyCorrectness: Validate cudaMemcpy DeviceToHost/HostToDevice
- ✅ ROCmMemcpyCorrectness: Validate hipMemcpy DeviceToHost/HostToDevice
- ✅ GPUStagingPerformance: Measure transfer bandwidth (GB/s)
- ✅ GPUTensorFactoryCreation: Create tensors on specific device indices
- ✅ GPUAttentionExecution: Run GQAAttention with GPU tensors

---

## Success Criteria

**Phase 3 Complete** when:
1. ✅ CUDA backend compiles cleanly (`-DHAVE_CUDA=ON`)
2. ✅ ROCm backend compiles cleanly (`-DHAVE_ROCM=ON`)
3. ✅ CPU-only build still works (`-DHAVE_CUDA=OFF -DHAVE_ROCM=OFF`)
4. ✅ No header conflicts (0 type redefinition errors)
5. ✅ MPIStager works with all 3 backends
6. ✅ GPU staging tests passing (CUDA + ROCm)
7. ✅ Performance validated: GPU transfer <1ms for 1MB tensor
8. ✅ Documentation updated: Build instructions for each backend

---

## Risk Mitigation

**Risk 1**: NVML API changes in newer CUDA versions
- **Mitigation**: Check CUDA Toolkit version, use API guards
- **Fallback**: Stub out NVML calls if API not available

**Risk 2**: ROCm property names differ across versions
- **Mitigation**: Runtime property detection with fallback
- **Fallback**: Skip NUMA node detection on unsupported ROCm versions

**Risk 3**: GPU tests fail on CPU-only CI systems
- **Mitigation**: Skip GPU tests when `device_count() == 0`
- **Pattern**: `if (backend->device_count() > 0) { /* run GPU test */ }`

**Risk 4**: Build complexity increases
- **Mitigation**: Keep CPU-only build simple (default)
- **Documentation**: Clear build instructions for each backend

---

## Next Steps

**Immediate Actions** (after user approval):
1. Create `backends/BackendInterface.h` (virtual interface)
2. Create `backends/cuda/` directory structure
3. Implement `CUDABackend.cu` (Day 1 deliverable)
4. Update CMakeLists.txt (CUDA target + mutual exclusion check)

**User Decision Required**:
- ✅ Approve **Option 1** (Separate Compilation Units)
- ⚠️ Consider **Option 2** (Runtime Plugins) if both backends needed simultaneously
- ❌ Reject **Option 3** (Namespace Isolation) - too risky

---

## References

**Related Changelogs**:
- `2025-10-31-phase2-mpi-staging-infrastructure.md` - MPIStager implementation
- `2025-10-31-phase2-mpi-staging-tests-complete.md` - Test suite (8/8 passing)
- `2025-10-31-phase2-gqa-attention-integration.md` - GQAAttention integration

**Implementation Files**:
- `src/v2/utils/MPIStager.h` (147 lines) - Public API
- `src/v2/utils/MPIStager.cpp` (203 lines) - Implementation (will be refactored)
- `src/v2/pipelines/attention/GQAAttention.cpp` - Staging consumer
- `tests/v2/integration/Test__MPIStaging.cpp` (350+ lines) - Test suite

**Build System**:
- `src/v2/CMakeLists.txt` (lines 17-69) - CUDA/ROCm configuration (disabled)
- Current state: `option(HAVE_CUDA "Enable CUDA backend" OFF)`

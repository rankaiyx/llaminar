# Llaminar v2 - Clean greenfield implementation
cmake_minimum_required(VERSION 3.18)

# Backend options (default to OFF - enable explicitly when ready)
option(HAVE_CUDA "Enable CUDA backend" OFF)
option(HAVE_ROCM "Enable ROCm backend" OFF)
option(HAVE_VULKAN "Enable Vulkan backend" OFF)
option(HAVE_MKL "Enable Intel MKL" OFF)

# V2 Core Library
add_library(llaminar2_core STATIC
    backends/ComputeBackend.cpp
    loaders/ModelLoader.cpp
    loaders/ModelContext.cpp
    loaders/WeightManager.cpp
    loaders/WeightPlacementMap.cpp
    loaders/DeviceOrchestrator.cpp
    pipelines/PipelineBase.cpp
    pipelines/PipelineFactory.cpp
    pipelines/qwen/Qwen2Pipeline.cpp
    tensors/FP32Tensor.cpp
    tensors/FP16Tensor.cpp
    tensors/BF16Tensor.cpp
    tensors/TensorFactory.cpp
    tensors/IQ4_NLTensor.cpp
    # Simple quantization formats (Q8_0, Q4_0, Q4_1)
    tensors/Q8_0Tensor.cpp
    tensors/Q4_0Tensor.cpp
    tensors/Q4_1Tensor.cpp
    # K-quant formats (Q6_K, Q2_K, Q5_K, Q3_K, Q4_K, Q8_K)
    tensors/Q6_KTensor.cpp
    tensors/Q2_KTensor.cpp
    tensors/Q5_KTensor.cpp
    tensors/Q3_KTensor.cpp
    tensors/Q4_KTensor.cpp
    tensors/Q8_KTensor.cpp
    # IQ formats (IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S, IQ3_S, IQ1_S, IQ1_M)
    tensors/IQ4_XSTensor.cpp
    tensors/IQ2_XXSTensor.cpp
    tensors/IQ2_XSTensor.cpp
    tensors/IQ3_XXSTensor.cpp
    tensors/IQ2_STensor.cpp
    tensors/IQ3_STensor.cpp
    tensors/IQ1_STensor.cpp
    tensors/IQ1_MTensor.cpp
    kernels/cpu/QuantizedGemm.cpp
    kernels/cpu/FP32GemmKernel.cpp
    kernels/cpu/CPURoPEKernel.cpp
    kernels/cpu/CPURMSNormKernel.cpp
    kernels/cpu/CPUSoftmaxKernel.cpp
    kernels/cpu/CPUSwiGLUKernel.cpp
    utils/ArgParser.cpp
)

target_include_directories(llaminar2_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compile features
target_compile_features(llaminar2_core PUBLIC cxx_std_17)

# Dependencies
find_package(MPI REQUIRED)

# Find OpenBLAS (required for GEMM operations)
# When built standalone (not from parent CMakeLists.txt)
if(NOT DEFINED OPENBLAS_LIBRARIES)
    find_package(BLAS REQUIRED)
    find_path(CBLAS_INCLUDE_DIR cblas.h PATHS /usr/include /usr/include/openblas /usr/local/include)
    if(CBLAS_INCLUDE_DIR)
        message(STATUS "V2: Found CBLAS include: ${CBLAS_INCLUDE_DIR}")
        target_include_directories(llaminar2_core PUBLIC ${CBLAS_INCLUDE_DIR})
    else()
        message(WARNING "V2: CBLAS header not found - using fallback")
    endif()
    set(OPENBLAS_LIBRARIES ${BLAS_LIBRARIES})
    message(STATUS "V2: Using system BLAS: ${BLAS_LIBRARIES}")
endif()

# Find Intel MKL if enabled (optional, for BF16 GEMM acceleration)
if(HAVE_MKL)
    if(NOT DEFINED MKL_LIBRARIES)
        find_package(MKL CONFIG QUIET)
        if(MKL_FOUND)
            message(STATUS "V2: Intel MKL found - enabling MKL backend")
            set(MKL_INCLUDE_DIRS ${MKL_INCLUDE})
            set(MKL_LIBRARIES MKL::MKL)
            target_include_directories(llaminar2_core PUBLIC ${MKL_INCLUDE_DIRS})
        else()
            message(WARNING "V2: Intel MKL requested (HAVE_MKL=ON) but not found")
            message(WARNING "V2: Install with: sudo apt install intel-oneapi-mkl-devel")
            set(HAVE_MKL OFF)
        endif()
    else()
        message(STATUS "V2: Using MKL from parent: ${MKL_LIBRARIES}")
    endif()
endif()

# Try to find libnuma (optional, for NUMA-aware allocation)
find_library(NUMA_LIBRARY NAMES numa)
if(NUMA_LIBRARY)
    message(STATUS "Found libnuma: ${NUMA_LIBRARY}")
    target_compile_definitions(llaminar2_core PUBLIC HAVE_NUMA)
    target_link_libraries(llaminar2_core PUBLIC ${NUMA_LIBRARY})
else()
    message(STATUS "libnuma not found - NUMA-aware allocation disabled")
endif()

target_link_libraries(llaminar2_core PUBLIC
    MPI::MPI_CXX
    ${OPENBLAS_LIBRARIES}
)

# Link MKL if available
if(HAVE_MKL AND DEFINED MKL_LIBRARIES)
    target_link_libraries(llaminar2_core PUBLIC ${MKL_LIBRARIES})
endif()

# Add conditional backend support
if(HAVE_CUDA)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(llaminar2_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(HAVE_ROCM)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_ROCM)
    # ROCm libraries will be found by parent CMakeLists.txt
    if(HIP_LIBRARY AND HIPBLAS_LIBRARY)
        target_link_libraries(llaminar2_core PUBLIC ${HIP_LIBRARY} ${HIPBLAS_LIBRARY})
    endif()
endif()

if(HAVE_VULKAN)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(llaminar2_core PUBLIC Vulkan::Vulkan)
endif()

if(HAVE_MKL)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_MKL)
endif()

# Main executable
add_executable(llaminar2 Main.cpp)
target_link_libraries(llaminar2 llaminar2_core)

# Enable testing for V2
enable_testing()

# V2 Tests
# Include test subdirectory (unit, integration, e2e, pytorch_parity)
# Note: The source tree is at ../../tests/v2 relative to src/v2
add_subdirectory(../../tests/v2 ${CMAKE_BINARY_DIR}/tests/v2)

# TODO: GPU backends (CUDA, ROCm, Vulkan) - not yet implemented
# TODO: Intel MKL backend - not yet implemented

# Install targets
install(TARGETS llaminar2
    RUNTIME DESTINATION bin
)

install(FILES
    utils/MPIContext.h
    DESTINATION include/llaminar2/utils
)

install(FILES
    backends/ComputeBackend.h
    DESTINATION include/llaminar2/backends
)

install(FILES
    tensors/Tensors.h
    tensors/TensorKernels.h
    DESTINATION include/llaminar2/tensors
)

install(FILES
    pipelines/QwenPipeline.h
    DESTINATION include/llaminar2/pipelines
)

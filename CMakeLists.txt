cmake_minimum_required(VERSION 3.16)
project(llaminar LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Check for NUMA support
find_path(NUMA_INCLUDE_DIR numa.h)
find_library(NUMA_LIBRARY numa)

if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
    set(NUMA_FOUND TRUE)
    message(STATUS "NUMA support found")
    add_definitions(-DHAVE_NUMA)
else()
    set(NUMA_FOUND FALSE)
    message(STATUS "NUMA support not found - continuing without it")
endif()

# Check for CUDA support (optional)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(ENABLE_CUDA TRUE)
    message(STATUS "CUDA support found - enabling GPU detection")
    add_definitions(-DENABLE_CUDA)
else()
    set(ENABLE_CUDA FALSE)
    message(STATUS "CUDA support not found - GPU detection will be limited")
endif()

# Check for ROCm support (optional)
find_path(ROCM_INCLUDE_DIR hip/hip_runtime.h PATHS /opt/rocm/include)
find_library(HIP_LIBRARY hip PATHS /opt/rocm/lib64)
if(ROCM_INCLUDE_DIR AND HIP_LIBRARY)
    set(ENABLE_ROCM TRUE)
    message(STATUS "ROCm support found - enabling AMD GPU detection")
    add_definitions(-DENABLE_ROCM)
else()
    set(ENABLE_ROCM FALSE)
    message(STATUS "ROCm support not found - AMD GPU detection disabled")
endif()

# Configure COSMA submodule
message(STATUS "Configuring COSMA submodule...")
set(COSMA_BLAS "CUSTOM" CACHE STRING "BLAS backend for COSMA")
set(COSMA_SCALAPACK "CUSTOM" CACHE STRING "ScaLAPACK backend for COSMA")
set(COSMA_WITH_TESTS OFF CACHE BOOL "Build COSMA tests")
set(COSMA_WITH_APPS OFF CACHE BOOL "Build COSMA applications")
set(COSMA_WITH_BENCHMARKS OFF CACHE BOOL "Build COSMA benchmarks")
set(COSMA_WITH_INSTALL OFF CACHE BOOL "Install COSMA")
set(COSMA_WITH_PROFILING OFF CACHE BOOL "Enable COSMA profiling")
add_subdirectory(COSMA)

# Configure llama.cpp submodule
message(STATUS "Configuring llama.cpp submodule...")
set(GGML_STATIC ON CACHE BOOL "Build GGML static library")
set(GGML_OPENMP ON CACHE BOOL "Enable OpenMP in GGML")
set(GGML_MPI ON CACHE BOOL "Enable MPI in GGML")
set(GGML_CUDA OFF CACHE BOOL "Disable CUDA in GGML")
set(GGML_VULKAN OFF CACHE BOOL "Disable Vulkan in GGML")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples")
add_subdirectory(llama.cpp)

# Source files for the core library
set(LLAMINAR_CORE_SOURCES
    src/common.cpp
    src/argument_parser.cpp
    src/topology_manager.cpp
    src/graph_compute.cpp
    src/kernel_manager.cpp
    src/kernels/mul_mat.cpp
    src/model_loader.cpp
    src/repacker.cpp
)

# Create the core library
add_library(llaminar_core STATIC ${LLAMINAR_CORE_SOURCES})

# Link libraries to core
target_link_libraries(llaminar_core
    PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cosma
    ggml
)

# Add NUMA if available
if(NUMA_FOUND)
    target_include_directories(llaminar_core PRIVATE ${NUMA_INCLUDE_DIR})
    target_link_libraries(llaminar_core PRIVATE ${NUMA_LIBRARY})
endif()

# Add CUDA if available
if(ENABLE_CUDA)
    target_include_directories(llaminar_core PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(llaminar_core PRIVATE ${CUDA_LIBRARIES} ${CUDA_NVML_LIBRARY})
endif()

# Add ROCm if available
if(ENABLE_ROCM)
    target_include_directories(llaminar_core PRIVATE ${ROCM_INCLUDE_DIR})
    target_link_libraries(llaminar_core PRIVATE ${HIP_LIBRARY})
endif()

# Set compile features
target_compile_features(llaminar_core PUBLIC cxx_std_17)

# Include directories for core library
target_include_directories(llaminar_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/COSMA/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
)

# Main executable
add_executable(llaminar src/main.cpp)
target_link_libraries(llaminar llaminar_core)

# Enable testing
enable_testing()

# Test sources and executables
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic llaminar_core)
add_test(NAME BasicTest COMMAND test_basic)

add_executable(test_numa tests/test_numa.cpp)
target_link_libraries(test_numa llaminar_core)
add_test(NAME NumaTest COMMAND test_numa)

add_executable(test_cosma tests/test_cosma.cpp)
target_link_libraries(test_cosma llaminar_core)
add_test(NAME CosmaTest COMMAND test_cosma)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph llaminar_core)
add_test(NAME GraphTest COMMAND test_graph)

# Set output directories
set_target_properties(llaminar
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(test_basic test_numa test_cosma test_graph
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Custom test target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_basic test_numa test_cosma test_graph
    COMMENT "Running all tests"
)

# Print configuration summary
message(STATUS "=== Llaminar Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C standard: ${CMAKE_C_STANDARD}")
message(STATUS "MPI found: ${MPI_FOUND}")
message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI libraries: ${MPI_LIBRARIES}")
message(STATUS "OpenMP found: ${OpenMP_FOUND}")
message(STATUS "NUMA found: ${NUMA_FOUND}")
message(STATUS "COSMA BLAS backend: ${COSMA_BLAS}")
message(STATUS "COSMA ScaLAPACK backend: ${COSMA_SCALAPACK}")
message(STATUS "GGML static: ${GGML_STATIC}")
message(STATUS "GGML OpenMP: ${GGML_OPENMP}")
message(STATUS "GGML MPI: ${GGML_MPI}")
message(STATUS "===============================")

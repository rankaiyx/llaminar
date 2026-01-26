cmake_minimum_required(VERSION 3.18)
project(cuda_hip_context_probe CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create diagnostic executable
add_executable(cuda_hip_context_probe cuda_hip_context_probe.cu)

# Link CUDA driver API and threading
target_link_libraries(cuda_hip_context_probe
    PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    dl
    pthread
)

# Set CUDA architectures (sm_86 only - this system's GPUs)
set_target_properties(cuda_hip_context_probe PROPERTIES
    CUDA_ARCHITECTURES "86"
)

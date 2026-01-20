# Llaminar - High-Performance LLM Inference Engine
# Root CMakeLists.txt - Delegates to V2 architecture
#
# The V1 architecture has been removed. All development is now on V2.
# See src/v2/CMakeLists.txt for the full build configuration.
#
# Build Instructions:
#   cmake -B build -S src/v2 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --parallel
#
# Or use the convenience wrapper (this file):
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --parallel

cmake_minimum_required(VERSION 3.18)
project(llaminar LANGUAGES C CXX)

# V2 is the only supported architecture
message(STATUS "Llaminar: Delegating to V2 architecture (src/v2/)")
add_subdirectory(src/v2)

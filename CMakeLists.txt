cmake_minimum_required(VERSION 3.18)

# Project configuration
project(Llaminar 
    VERSION 1.0.0
    DESCRIPTION "An LLM inferencing engine using COSMA / Open MPI for hyper-scalability"
    LANGUAGES CXX C
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C standard
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)

# Include directories
include_directories(${MPI_INCLUDE_PATH})
include_directories(src)

# Source files
set(SOURCES
    src/main.cpp
    # Add more source files here
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${MPI_LIBRARIES}
)

# MPI compile flags
if(MPI_COMPILE_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

# MPI link flags
if(MPI_LINK_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

# Enable compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# Create src directory and example files if they don't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# Print configuration summary
message(STATUS "=== Llaminar Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C standard: ${CMAKE_C_STANDARD}")
message(STATUS "MPI found: ${MPI_FOUND}")
message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI libraries: ${MPI_LIBRARIES}")
message(STATUS "===============================")
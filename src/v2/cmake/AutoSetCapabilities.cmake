# AutoSetCapabilities.cmake
# =============================================================================
# Automatic Linux capability assignment for built executables
# =============================================================================
#
# This module provides automatic CAP_SYS_ADMIN capability assignment to 
# executables after building. This is required for PCIe BAR direct P2P
# transfers between CUDA and ROCm GPUs.
#
# Why CAP_SYS_ADMIN?
# ------------------
# CUDA's cuMemHostRegister with CU_MEMHOSTREGISTER_IOMEMORY flag requires
# CAP_SYS_ADMIN to map PCIe BAR memory regions. This enables direct GPU-to-GPU
# transfers through AMD's exposed VRAM without host memory staging.
#
# Usage:
# ------
# 1. include(cmake/AutoSetCapabilities.cmake) in your CMakeLists.txt
# 2. Capabilities will be auto-set on all executables (if AUTO_SET_CAPS=ON)
# 3. Or manually: llaminar_set_capabilities(target_name)
#
# Requirements:
# -------------
# - Linux with libcap-bin installed (provides setcap)
# - Passwordless sudo for setcap (configured in devcontainer)
# - Privileged container (for capability inheritance)
#
# =============================================================================

# Option to control automatic capability assignment
# Default ON in devcontainer, can be disabled for CI/production builds
option(AUTO_SET_CAPABILITIES "Automatically set CAP_SYS_ADMIN on built executables" ON)

# Find setcap
find_program(SETCAP_EXECUTABLE setcap PATHS /usr/sbin /sbin /usr/bin /bin)

if(NOT SETCAP_EXECUTABLE)
    message(STATUS "setcap not found - automatic capability assignment disabled")
    set(AUTO_SET_CAPABILITIES OFF CACHE BOOL "" FORCE)
endif()

# Check if we can use sudo setcap without password
if(AUTO_SET_CAPABILITIES AND SETCAP_EXECUTABLE)
    # Create a temporary test file to verify setcap works
    file(WRITE "${CMAKE_BINARY_DIR}/setcap_test.sh" "#!/bin/bash\nexit 0\n")
    execute_process(
        COMMAND chmod +x "${CMAKE_BINARY_DIR}/setcap_test.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    execute_process(
        COMMAND sudo -n ${SETCAP_EXECUTABLE} cap_sys_admin+ep "${CMAKE_BINARY_DIR}/setcap_test.sh"
        RESULT_VARIABLE SETCAP_SUDO_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    file(REMOVE "${CMAKE_BINARY_DIR}/setcap_test.sh")
    
    if(NOT SETCAP_SUDO_RESULT EQUAL 0)
        message(WARNING "Passwordless sudo for setcap not available - capability assignment may require manual setup")
    else()
        message(STATUS "Passwordless sudo setcap verified - will auto-set CAP_SYS_ADMIN on executables")
    endif()
endif()

# Create wrapper script for setcap (handles output redirection properly)
if(AUTO_SET_CAPABILITIES AND SETCAP_EXECUTABLE)
    set(SETCAP_WRAPPER_SCRIPT "${CMAKE_BINARY_DIR}/setcap_wrapper.sh")
    file(WRITE ${SETCAP_WRAPPER_SCRIPT} 
"#!/bin/bash
# Wrapper script for setcap - silently sets capabilities, never fails
sudo -n ${SETCAP_EXECUTABLE} cap_sys_admin+ep \"$1\" >/dev/null 2>&1 || true
")
    execute_process(COMMAND chmod +x ${SETCAP_WRAPPER_SCRIPT})
endif()

# Function to add capability assignment to a target
# Usage: llaminar_set_capabilities(target_name)
function(llaminar_set_capabilities TARGET_NAME)
    if(NOT AUTO_SET_CAPABILITIES)
        return()
    endif()
    
    if(NOT SETCAP_EXECUTABLE)
        return()
    endif()
    
    # Add post-build command to set capabilities using wrapper script
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${SETCAP_WRAPPER_SCRIPT} $<TARGET_FILE:${TARGET_NAME}>
        COMMENT "Setting CAP_SYS_ADMIN on ${TARGET_NAME} for PCIe BAR P2P"
        VERBATIM
    )
endfunction()

# Macro to override add_executable and auto-add capabilities
# This is the magic that makes ALL executables get capabilities automatically
#
# We save the original add_executable, create our wrapper, and forward all calls
# After adding the executable, we automatically set capabilities on it
if(AUTO_SET_CAPABILITIES AND SETCAP_EXECUTABLE)
    # Store original add_executable
    function(_original_add_executable)
        _add_executable(${ARGN})
    endfunction()
    
    # Create wrapper that adds capability post-build
    macro(add_executable TARGET_NAME)
        _add_executable(${TARGET_NAME} ${ARGN})
        
        # Only set capabilities on actual executables (not IMPORTED, not ALIAS)
        get_target_property(_target_type ${TARGET_NAME} TYPE)
        get_target_property(_is_imported ${TARGET_NAME} IMPORTED)
        get_target_property(_is_alias ${TARGET_NAME} ALIASED_TARGET)
        
        if(_target_type STREQUAL "EXECUTABLE" AND NOT _is_imported AND NOT _is_alias)
            llaminar_set_capabilities(${TARGET_NAME})
        endif()
    endmacro()
    
    message(STATUS "Auto-capability assignment enabled for all executables")
endif()

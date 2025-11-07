#!/bin/bash
#
# Build script for E2E Release configuration
# This builds V2 with Release optimizations (-O3) while keeping
# ENABLE_PIPELINE_SNAPSHOTS enabled for E2E parity testing.
#
# Usage:
#   ./build_e2e_release.sh           # Configure and build
#   ./build_e2e_release.sh test      # Run E2E parity tests after build
#   ./build_e2e_release.sh clean     # Clean and rebuild from scratch

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_v2_e2e_release"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clean_build() {
    echo -e "${YELLOW}Cleaning E2E Release build...${NC}"
    rm -rf "$BUILD_DIR"
}

configure_build() {
    echo -e "${BLUE}Configuring E2E Release build...${NC}"
    cmake -B "$BUILD_DIR" -S src/v2 \
        -DCMAKE_BUILD_TYPE=E2ERelease \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    echo -e "${GREEN}✓ Configuration complete${NC}"
    echo -e "  Build directory: ${BUILD_DIR}"
    echo -e "  Build type: E2ERelease (Release optimizations + snapshots enabled)"
}

build() {
    echo -e "${BLUE}Building E2E Release...${NC}"
    cmake --build "$BUILD_DIR" --parallel
    echo -e "${GREEN}✓ Build complete${NC}"
}

run_tests() {
    echo -e "${BLUE}Running E2E parity tests...${NC}"
    cd "$BUILD_DIR"
    ctest -R "V2_E2E" --output-on-failure --verbose
    cd "$SCRIPT_DIR"
    echo -e "${GREEN}✓ Tests complete${NC}"
}

# Main script logic
case "${1:-build}" in
    clean)
        clean_build
        configure_build
        build
        ;;
    test)
        if [ ! -d "$BUILD_DIR" ]; then
            echo -e "${YELLOW}Build directory not found. Configuring first...${NC}"
            configure_build
        fi
        build
        run_tests
        ;;
    config)
        configure_build
        ;;
    build|*)
        if [ ! -d "$BUILD_DIR" ]; then
            configure_build
        fi
        build
        ;;
esac

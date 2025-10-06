#!/bin/bash
#
# Run PyTorch parity test with proper environment configuration
#
# This script sets up the environment variables needed for the
# DistributedPipelineVsPyTorchReference test and executes it.
#
# @author David Sanftenberg

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== PyTorch Parity Test Setup ===${NC}"
echo

# Check if snapshots are extracted
if [ ! -d "pytorch_snapshots" ]; then
    echo -e "${YELLOW}⚠ PyTorch snapshots directory not found${NC}"
    echo "Extracting pytorch_layer_captures.npz..."
    python3 tests/npz_to_npy.py pytorch_layer_captures.npz pytorch_snapshots/ --verbose
    echo
fi

# Check if snapshots are mapped to Llaminar naming convention
if [ ! -d "pytorch_snapshots_mapped" ]; then
    echo -e "${YELLOW}⚠ Mapped snapshots directory not found${NC}"
    echo "Mapping PyTorch snapshots to Llaminar naming convention..."
    python3 tests/map_pytorch_snapshots.py pytorch_snapshots/
    echo
fi

# Verify mapping
SNAPSHOT_COUNT=$(ls -1 pytorch_snapshots_mapped/*.npy 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Found $SNAPSHOT_COUNT mapped snapshot files${NC}"
echo

# Token sequence used to generate PyTorch snapshots
# These are the default tokens from capture_pytorch_layers.py: [1639, 266, 285, 17, 10, 17, 30]
# Meaning: "1+1=1." in the tokenizer
TOKEN_SEQUENCE="1639,266,285,17,10,17,30"

echo "Configuration:"
echo "  PYTORCH_SNAPSHOT_DIR: pytorch_snapshots_mapped"
echo "  PYTORCH_SNAPSHOT_TOKENS: $TOKEN_SEQUENCE"
echo "  Token count: 7 tokens"
echo

# Export environment variables
export PYTORCH_SNAPSHOT_DIR="pytorch_snapshots_mapped"
export PYTORCH_SNAPSHOT_TOKENS="$TOKEN_SEQUENCE"
export LLAMINAR_PARITY_CAPTURE=1  # Enable snapshot capture in Llaminar

# Check if test binary exists
if [ ! -f "build/test_parity_framework" ]; then
    echo -e "${YELLOW}⚠ Test binary not found. Building...${NC}"
    cmake --build build --target test_parity_framework --parallel
    echo
fi

echo -e "${BLUE}=== Running PyTorch Reference Test ===${NC}"
echo

# Run the specific test
# Use timeout to prevent hangs (60 seconds should be enough)
timeout 60 ./build/test_parity_framework \
    --gtest_filter=ParityFramework.DistributedPipelineVsPyTorchReference \
    2>&1 | tee pytorch_parity_test.log

EXIT_CODE=${PIPESTATUS[0]}

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch parity test PASSED${NC}"
else
    echo -e "${YELLOW}⚠ PyTorch parity test had issues (exit code: $EXIT_CODE)${NC}"
    echo "  See pytorch_parity_test.log for details"
fi

exit $EXIT_CODE

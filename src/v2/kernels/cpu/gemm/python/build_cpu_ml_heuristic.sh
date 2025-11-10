#!/bin/bash
# Complete CPU GEMM ML Heuristic Pipeline
#
# Runs full workflow:
#   1. Build benchmark test
#   2. Collect training data (15 shapes × 1225 variants = ~18K points)
#   3. Train neural network model
#   4. Export to C++ header
#   5. Validate integration
#
# Usage:
#   ./build_cpu_ml_heuristic.sh [--skip-data-collection]
#
# Author: David Sanftenberg
# Date: November 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_PATH="models/qwen2.5-0.5b-instruct-iq4_nl.gguf"
DATA_CSV="cpu_gemm_benchmark_data.csv"
TRAINED_MODEL="cpu_gemm_heuristic.onnx"
SCALER_FILE="cpu_gemm_scaler.npz"
OUTPUT_HEADER="src/v2/kernels/cpu/CpuGemmHeuristicWeights.h"

SKIP_DATA_COLLECTION=false
if [[ "$1" == "--skip-data-collection" ]]; then
    SKIP_DATA_COLLECTION=true
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}CPU GEMM ML Heuristic Builder${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# Step 1: Check Python dependencies
# ============================================================================
echo -e "${YELLOW}[1/6] Checking Python dependencies...${NC}"

if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${RED}✗ PyTorch not installed${NC}"
    echo "Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

if ! python3 -c "import onnx" 2>/dev/null; then
    echo -e "${RED}✗ ONNX not installed${NC}"
    echo "Installing ONNX..."
    pip install onnx
fi

if ! python3 -c "import sklearn" 2>/dev/null; then
    echo -e "${RED}✗ scikit-learn not installed${NC}"
    echo "Installing scikit-learn..."
    pip install scikit-learn
fi

if ! python3 -c "import pandas" 2>/dev/null; then
    echo -e "${RED}✗ pandas not installed${NC}"
    echo "Installing pandas..."
    pip install pandas
fi

echo -e "${GREEN}✓ Python dependencies satisfied${NC}"
echo ""

# ============================================================================
# Step 2: Build benchmark test (if needed)
# ============================================================================
echo -e "${YELLOW}[2/6] Building benchmark test...${NC}"

if [[ ! -f "build_v2_release/tests/v2/v2_perf_iq4nl_gemm" ]]; then
    echo "Building Release build..."
    cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-march=native -mtune=native"
    cmake --build build_v2_release --target v2_perf_iq4nl_gemm --parallel
else
    echo -e "${GREEN}✓ Benchmark test already built${NC}"
fi

if [[ ! -f "build_v2_release/tests/v2/v2_perf_iq4nl_gemm" ]]; then
    echo -e "${RED}✗ Failed to build benchmark test${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Benchmark test ready${NC}"
echo ""

# ============================================================================
# Step 3: Collect training data
# ============================================================================
if [[ "$SKIP_DATA_COLLECTION" == true ]]; then
    echo -e "${YELLOW}[3/6] Skipping data collection (--skip-data-collection)${NC}"
    
    if [[ ! -f "$DATA_CSV" ]]; then
        echo -e "${RED}✗ No data file found: $DATA_CSV${NC}"
        echo "Cannot skip data collection without existing data."
        exit 1
    fi
    
    echo -e "${GREEN}✓ Using existing data: $DATA_CSV${NC}"
else
    echo -e "${YELLOW}[3/6] Collecting training data...${NC}"
    echo "This will take ~6-8 hours (15 shapes × 1225 variants)"
    echo ""
    
    START_TIME=$(date +%s)
    
    ./benchmark_cpu_gemm.py --model "$MODEL_PATH" --output "$DATA_CSV"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_HOURS=$((ELAPSED / 3600))
    ELAPSED_MINS=$(( (ELAPSED % 3600) / 60 ))
    
    echo ""
    echo -e "${GREEN}✓ Data collection complete${NC}"
    echo "  Time: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
    echo "  Output: $DATA_CSV"
fi
echo ""

# ============================================================================
# Step 4: Train neural network
# ============================================================================
echo -e "${YELLOW}[4/6] Training neural network...${NC}"
echo "This may take 15-30 minutes"
echo ""

./train_cpu_gemm_heuristic.py \
    --data "$DATA_CSV" \
    --output "$TRAINED_MODEL" \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.001

if [[ ! -f "$TRAINED_MODEL" ]] || [[ ! -f "$SCALER_FILE" ]]; then
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Model trained${NC}"
echo "  Model: $TRAINED_MODEL"
echo "  Scaler: $SCALER_FILE"
echo ""

# ============================================================================
# Step 5: Export to C++ header
# ============================================================================
echo -e "${YELLOW}[5/6] Exporting to C++ header...${NC}"

./export_cpu_heuristic.py \
    --model "$TRAINED_MODEL" \
    --scaler "$SCALER_FILE" \
    --output "$OUTPUT_HEADER"

if [[ ! -f "$OUTPUT_HEADER" ]]; then
    echo -e "${RED}✗ Export failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ C++ header generated${NC}"
echo "  Header: $OUTPUT_HEADER"
echo ""

# ============================================================================
# Step 6: Validate header compiles
# ============================================================================
echo -e "${YELLOW}[6/6] Validating C++ header...${NC}"

# Create test program
cat > /tmp/test_cpu_heuristic.cpp << 'EOF'
#include "../src/v2/kernels/cpu/CpuGemmHeuristicWeights.h"
#include <iostream>

int main() {
    // Test prediction
    double gflops = llaminar::cpu::CpuGemmHeuristic::predict(
        2048, 896, 896,   // m, n, k
        4, 2,              // tile_m, tile_n
        16, 5,             // unroll_k, prefetch_dist
        true               // is_avx512
    );
    
    std::cout << "Predicted GFLOPS: " << gflops << std::endl;
    return 0;
}
EOF

g++ -std=c++17 -I. -o /tmp/test_cpu_heuristic /tmp/test_cpu_heuristic.cpp
/tmp/test_cpu_heuristic

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Header compiles and runs successfully${NC}"
else
    echo -e "${RED}✗ Header compilation failed${NC}"
    exit 1
fi

rm -f /tmp/test_cpu_heuristic /tmp/test_cpu_heuristic.cpp

echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}ML Heuristic Build Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Generated files:"
echo "  ✓ $DATA_CSV"
echo "  ✓ $TRAINED_MODEL"
echo "  ✓ $SCALER_FILE"
echo "  ✓ $OUTPUT_HEADER"
echo ""
echo "Next steps:"
echo ""
echo "1. Integrate into SmartGemmSearch.cpp:"
echo "   - #include \"CpuGemmHeuristicWeights.h\""
echo "   - Replace scorePerformanceModel() calls with:"
echo "     CpuGemmHeuristic::predict(m, n, k, tile_m, tile_n, unroll_k, prefetch, is_avx512)"
echo ""
echo "2. Add environment flag control:"
echo "   - LLAMINAR_USE_ML_HEURISTIC=1 (default)"
echo "   - LLAMINAR_USE_ML_HEURISTIC=0 (fallback to manual)"
echo ""
echo "3. Rebuild and test:"
echo "   cmake --build build_v2_release --target v2_perf_iq4nl_gemm --parallel"
echo "   ./run_benchmark.sh v2_perf_iq4nl_gemm"
echo ""
echo "4. Validate performance recovery:"
echo "   - Run: ./quick_profile_iq4nl.sh"
echo "   - Expected L1 miss rate: < 5% (was 31.84%)"
echo "   - Expected throughput: > 1000 GFLOPS (was 392)"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

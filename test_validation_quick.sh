#!/bin/bash
# Quick validation test on 3 unseen shapes

echo "========================================" echo "Quick Validation: Testing on Unseen Shapes"
echo "========================================"

# Use existing trained model
MODEL="cuda_heuristic_nn.onnx"
SCALER="feature_scaler.bin"

if [ ! -f "$MODEL" ] || [ ! -f "$SCALER" ]; then
    echo "ERROR: Trained model not found. Run training first."
    exit 1
fi

# Test on 3 unseen Qwen models
echo "Testing on: Qwen 1.5B, 14B, 32B (unseen during training)"

# First, backup the old benchmark CSV if it exists
if [ -f "cuda_gemm_benchmark_data.csv" ]; then
    mv cuda_gemm_benchmark_data.csv cuda_gemm_benchmark_data_old.csv
    echo "Backed up old benchmark CSV"
fi

# Run benchmarks on unseen shapes (CTest uses -E for test name, not --gtest_filter)
cd build_v2_release
./performance/v2_perf_cuda_heuristic_validation \
    --gtest_filter="*Qwen_1_5B_SingleToken_QKV:*Qwen_14B_SingleToken_QKV:*Qwen_32B_SingleToken_QKV"

# Move CSV to workspace root
if [ -f "cuda_gemm_benchmark_data.csv" ]; then
    mv cuda_gemm_benchmark_data.csv ../cuda_gemm_validation_data.csv
    echo "Moved validation CSV to workspace root"
fi

cd ..

# Check if validation CSV was created
if [ -f "cuda_gemm_validation_data.csv" ]; then
    # Run validation script
    python python/validate_heuristic.py \
        --model "$MODEL" \
        --scaler "$SCALER" \
        --benchmark cuda_gemm_validation_data.csv \
        --output validation_quick_results.json \
        --top-n 1 5 10 30
    
    echo ""
    echo "Validation complete! Check validation_quick_results.json"
else
    echo "ERROR: Benchmark CSV not found"
    exit 1
fi

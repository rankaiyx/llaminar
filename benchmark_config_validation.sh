#!/bin/bash
# Validation Configuration - Test on UNSEEN shapes to validate generalization
# Training data: Qwen 0.5B, 4B, 7B (9 tests)
# Validation data: Everything else (44 tests)

# Test unseen Qwen model sizes + different architectures
BENCHMARK_CTEST_NAME="V2_Perf_CudaHeuristicValidation"
BENCHMARK_GTEST_FILTER="*Qwen_1_5B*:*Qwen_14B*:*Qwen_32B*:*Qwen_72B*:*DeepSeek_671B*:*OddBatch*:*OddDim*"

# Profile only top-5 and bottom-5 (less data needed for validation)
PROFILE_TOP_N=5

# Reduce epochs for faster validation
NN_EPOCHS=50

echo "Validation Mode: Testing on UNSEEN shapes"
echo "  Training shapes: Qwen 0.5B, 4B, 7B (9 tests)"
echo "  Validation shapes: Qwen 1.5B, 14B, 32B, 72B, DeepSeek 671B, Odd dims (18 tests)"
echo "  Expected configs: ~70,000 benchmarks"
echo "  Expected profiles: ~180 (top-5 + bottom-5 per test)"

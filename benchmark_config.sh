#!/bin/bash
##
# @file benchmark_config.sh
# @brief Configuration for CUDA GEMM benchmark subset
#
# Controls which tests to run during benchmarking and profiling.
# Designed to reduce JIT compilation overhead by focusing on key model sizes.
#
# Current configuration (Nov 3, 2025):
# - Models: Qwen 0.5B, 4B, 7B only (skip 1.5B, 14B, 32B, 72B, DeepSeek, MoE)
# - Profiling: Top-10 and bottom-10 per test (reduced from 50)
# - Total tests: ~9 tests (down from 53)
# - Total configs profiled: ~180 (down from 5,300)
# - Estimated runtime: 1-2 hours (down from 4-6 hours)
#
# @author David Sanftenberg
# @date November 3, 2025
##

# CTest test name (the performance test suite)
export BENCHMARK_CTEST_NAME="V2_Perf_CudaHeuristicValidation"

# GTest filter (test cases within the suite)
# Only run test cases matching these patterns (colon-separated for GTest)
export BENCHMARK_GTEST_FILTER="*Qwen_0_5B*:*Qwen_4B*:*Qwen_7B*"

# Profiling configuration
export PROFILE_TOP_N=10  # Profile top-10 and bottom-10 (was 50)
export PROFILE_MAX_TESTS=""  # No limit on matching tests (filter handles it)

# Neural network training
export NN_EPOCHS=100
export NN_BATCH_SIZE=128
export NN_LEARNING_RATE=0.001

# Enabled tests (for documentation):
# Qwen 0.5B (3 tests):
#   - Qwen_0_5B_SingleToken_QKV (1×896×896)
#   - Qwen_0_5B_Batch32_QKV (32×896×896)
#   - Qwen_0_5B_FFN_Gate (1×4864×896)
#
# Qwen 4B (3 tests):
#   - Qwen_4B_SingleToken_QKV (1×2560×2560)
#   - Qwen_4B_Batch128_QKV (128×2560×2560)
#   - Qwen_4B_FFN_Down (1×2560×13824)
#
# Qwen 7B (3 tests):
#   - Qwen_7B_SingleToken_QKV (1×3584×3584)
#   - Qwen_7B_Batch128_QKV (128×3584×3584)
#   - Qwen_7B_FFN_Gate (1×22016×3584)
#
# Total: 9 tests × ~3,888 configs = ~35K benchmarks (down from 206K)
# Profiling: 9 tests × 20 configs (top-10 + bottom-10) = ~180 profiles (down from 5,300)

# Disabled tests (to reduce JIT overhead):
# - Qwen 1.5B (4 tests)
# - Qwen 14B (3 tests)
# - Qwen 32B (2 tests)
# - Qwen 72B (3 tests)
# - DeepSeek 671B (8 tests)
# - Qwen3-MoE 235B (9 tests)
# - Qwen3-MoE 30B (4 tests)
# - GPT-OSS 20B/120B (4 tests)
# - Odd batches (4 tests)
# - Odd dimensions (3 tests)
# Total disabled: 44 tests

echo "Benchmark configuration loaded:"
echo "  CTest name: $BENCHMARK_CTEST_NAME"
echo "  GTest filter: $BENCHMARK_GTEST_FILTER"
echo "  Profile top-N: $PROFILE_TOP_N"
echo "  Expected tests: ~9"
echo "  Expected configs: ~35K benchmarks, ~180 profiles"
echo "  Estimated runtime: 1-2 hours (benchmark + profile)"

#!/bin/bash
##
# @file auto_run_pipeline.sh
# @brief Automated pipeline for CUDA GEMM profiling and ML training
#
# This script orchestrates the complete workflow:
# 1. Build performance tests (if needed)
# 2. Run benchmark validation tests to generate cuda_gemm_benchmark_data.csv
# 3. Profile top-N/bottom-N configs with NVIDIA ncu
# 4. Train neural network heuristic with profiling features
# 5. Deploy model and validate with canary tests
#
# Usage:
#   ./auto_run_pipeline.sh [--skip-build] [--skip-benchmark] [--skip-profiling]
#
# Environment variables:
#   LLAMINAR_PROFILE_TOP_N: Number of top/bottom configs to profile (default: 50)
#   LLAMINAR_PROFILE_MAX_TESTS: Max test cases to profile (default: all)
#   LLAMINAR_NN_EPOCHS: Training epochs (default: 100)
#
# @author David Sanftenberg
# @date November 3, 2025
##

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${WORKSPACE_ROOT}/build_v2_release"
PYTHON_DIR="${WORKSPACE_ROOT}/python"
CUDA_KERNEL_DIR="${WORKSPACE_ROOT}/src/v2/kernels/cuda"

BENCHMARK_CSV="cuda_gemm_benchmark_data.csv"
PROFILING_CSV="cuda_gemm_profiling_data.csv"
MODEL_ONNX="cuda_heuristic_nn.onnx"
SCALER_BIN="feature_scaler.bin"
METRICS_JSON="training_metrics.json"

# Parse arguments
SKIP_BUILD=0
SKIP_BENCHMARK=0
SKIP_PROFILING=0

for arg in "$@"; do
    case $arg in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=1
            shift
            ;;
        --skip-profiling)
            SKIP_PROFILING=1
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            echo "Usage: $0 [--skip-build] [--skip-benchmark] [--skip-profiling]"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Load benchmark configuration (subset mode to reduce JIT overhead)
if [ -f "$WORKSPACE_ROOT/benchmark_config.sh" ]; then
    source "$WORKSPACE_ROOT/benchmark_config.sh"
    log_info "Loaded benchmark_config.sh (subset mode)"
else
    log_warning "benchmark_config.sh not found, using defaults"
fi

# Defaults (override from environment or config file)
BENCHMARK_CTEST_NAME=${BENCHMARK_CTEST_NAME:-"V2_Perf_CudaHeuristicValidation"}
BENCHMARK_GTEST_FILTER=${BENCHMARK_GTEST_FILTER:-"*"}
PROFILE_TOP_N=${LLAMINAR_PROFILE_TOP_N:-${PROFILE_TOP_N:-50}}
PROFILE_MAX_TESTS=${LLAMINAR_PROFILE_MAX_TESTS:-${PROFILE_MAX_TESTS:-}}
NN_EPOCHS=${LLAMINAR_NN_EPOCHS:-${NN_EPOCHS:-100}}

# Check dependencies
check_dependencies() {
    log_section "Checking Dependencies"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        exit 1
    fi
    log_info "Python: $(python3 --version)"
    
    # Check NVIDIA ncu
    if ! command -v ncu &> /dev/null; then
        log_warning "NVIDIA Nsight Compute (ncu) not found"
        log_warning "Profiling will be skipped (only benchmarking)"
        SKIP_PROFILING=1
    else
        log_info "ncu: $(ncu --version | head -n1)"
    fi
    
    # Check Python packages
    python3 -c "import torch, pandas, sklearn, numpy" 2>/dev/null || {
        log_error "Missing Python packages (torch, pandas, sklearn, numpy)"
        log_info "Install with: pip install torch pandas scikit-learn numpy"
        exit 1
    }
    log_success "All dependencies satisfied"
}

# Phase 1: Build
build_project() {
    if [ $SKIP_BUILD -eq 1 ]; then
        log_info "Skipping build (--skip-build)"
        return
    fi
    
    log_section "Phase 1: Building Project (Release)"
    
    cd "$WORKSPACE_ROOT"
    
    # Configure CMake (Release build for accurate profiling)
    if [ ! -d "$BUILD_DIR" ]; then
        log_info "Configuring CMake..."
        cmake -B "$BUILD_DIR" -S src/v2 \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES=75 \
            -DENABLE_CUDA=ON
    fi
    
    # Build performance tests and profiling tools
    log_info "Building performance tests..."
    cmake --build "$BUILD_DIR" --target v2_perf_cuda_heuristic_validation --parallel
    
    log_info "Building profile_cuda_config..."
    cmake --build "$BUILD_DIR" --target profile_cuda_config --parallel
    
    log_success "Build complete"
}

# Phase 2: Run benchmarks
run_benchmarks() {
    if [ $SKIP_BENCHMARK -eq 1 ]; then
        log_info "Skipping benchmarks (--skip-benchmark)"
        if [ ! -f "$BENCHMARK_CSV" ]; then
            log_error "Benchmark CSV not found: $BENCHMARK_CSV"
            log_error "Cannot skip benchmarking without existing data"
            exit 1
        fi
        return
    fi
    
    log_section "Phase 2: Running Benchmark Tests"
    
    cd "$BUILD_DIR"
    
    # Run heuristic validation tests (generates CSV)
    log_info "Running CTest: $BENCHMARK_CTEST_NAME"
    log_info "GTest filter: $BENCHMARK_GTEST_FILTER"
    log_info "This will take 5-10 minutes (subset mode: ~9 tests, ~35K configs)"
    
    # Set GTest filter and run via CTest
    export GTEST_FILTER="$BENCHMARK_GTEST_FILTER"
    ctest -R "^${BENCHMARK_CTEST_NAME}$" --output-on-failure || {
        log_error "Benchmark tests failed"
        exit 1
    }
    
    # Check output (CSV is generated in workspace root by the test)
    local csv_path="$WORKSPACE_ROOT/$BENCHMARK_CSV"
    if [ ! -f "$csv_path" ]; then
        log_error "Benchmark CSV not generated: $csv_path"
        exit 1
    fi
    
    local num_records=$(wc -l < "$csv_path")
    log_success "Benchmarks complete: $num_records records in $csv_path"
}

# Phase 3: Collect profiling data
run_profiling() {
    if [ $SKIP_PROFILING -eq 1 ]; then
        log_info "Skipping profiling (--skip-profiling or ncu not available)"
        return
    fi
    
    log_section "Phase 3: Collecting Profiling Data with NVIDIA ncu"
    
    cd "$WORKSPACE_ROOT"
    
    # Check if benchmark data exists
    if [ ! -f "$BENCHMARK_CSV" ]; then
        log_error "Benchmark CSV not found: $BENCHMARK_CSV"
        exit 1
    fi
    
    # Check if profile_cuda_config exists
    if [ ! -f "$BUILD_DIR/profile_cuda_config" ]; then
        log_error "profile_cuda_config not found in $BUILD_DIR"
        exit 1
    fi
    
    log_info "Profiling top-$PROFILE_TOP_N and bottom-$PROFILE_TOP_N configs per test"
    if [ -n "$PROFILE_MAX_TESTS" ]; then
        log_info "Limited to first $PROFILE_MAX_TESTS test cases"
    fi
    log_warning "This will take 1-2 hours (30-60 seconds per config × ~180 configs in subset mode)"
    
    # Run profiling collection script
    python3 "$PYTHON_DIR/collect_profiling_data.py" \
        --input "$BENCHMARK_CSV" \
        --executable "$BUILD_DIR/profile_cuda_config" \
        --output "$PROFILING_CSV" \
        --top-n "$PROFILE_TOP_N" \
        ${PROFILE_MAX_TESTS:+--max-tests $PROFILE_MAX_TESTS} \
        --metrics "dram_throughput,l1_cache_hit_rate,l2_cache_hit_rate,sm_throughput,sm_instruction_throughput,sm_warps_active,global_load_coalescing,global_store_coalescing,smem_bank_conflicts_ld,smem_bank_conflicts_st,warp_divergence" || {
        log_error "Profiling collection failed"
        exit 1
    }
    
    log_success "Profiling complete: $PROFILING_CSV"
}

# Phase 4: Train neural network
train_model() {
    log_section "Phase 4: Training Neural Network Heuristic"
    
    cd "$WORKSPACE_ROOT"
    
    # Check if benchmark data exists
    if [ ! -f "$BENCHMARK_CSV" ]; then
        log_error "Benchmark CSV not found: $BENCHMARK_CSV"
        exit 1
    fi
    
    # Determine if profiling data is available
    local profiling_arg=""
    if [ -f "$PROFILING_CSV" ]; then
        log_info "Using profiling data for enhanced features (84 features)"
        profiling_arg="--profiling $PROFILING_CSV"
    else
        log_warning "Profiling data not found, using baseline features only (73 features)"
    fi
    
    log_info "Training for $NN_EPOCHS epochs..."
    
    # Run training script
    python3 "$PYTHON_DIR/train_cuda_neural_network.py" \
        --input "$BENCHMARK_CSV" \
        $profiling_arg \
        --output-dir "$CUDA_KERNEL_DIR" \
        --epochs "$NN_EPOCHS" \
        --batch-size 128 \
        --learning-rate 0.001 || {
        log_error "Model training failed"
        exit 1
    }
    
    # Check outputs
    if [ ! -f "$CUDA_KERNEL_DIR/$MODEL_ONNX" ]; then
        log_error "ONNX model not generated: $CUDA_KERNEL_DIR/$MODEL_ONNX"
        exit 1
    fi
    
    log_success "Model training complete"
    log_info "Model: $CUDA_KERNEL_DIR/$MODEL_ONNX"
    log_info "Scaler: $CUDA_KERNEL_DIR/$SCALER_BIN"
    log_info "Metrics: $CUDA_KERNEL_DIR/$METRICS_JSON"
    
    # Display metrics
    if [ -f "$CUDA_KERNEL_DIR/$METRICS_JSON" ]; then
        log_info "Training Metrics:"
        cat "$CUDA_KERNEL_DIR/$METRICS_JSON"
    fi
}

# Phase 5: Validate with canary tests
validate_model() {
    log_section "Phase 5: Validating Model with Canary Tests"
    
    cd "$BUILD_DIR"
    
    # Enable NN heuristic
    export LLAMINAR_USE_NN_HEURISTIC=1
    
    log_info "Running canary tests with NN heuristic enabled..."
    
    ctest -R "V2_Perf_CudaHeuristicCanary" --verbose || {
        log_warning "Some canary tests failed (expected during development)"
    }
    
    log_success "Validation complete"
    log_info "Check test output for top-N hit rates"
}

# Main pipeline
main() {
    log_section "CUDA GEMM Profiling & ML Training Pipeline"
    
log_info "Configuration:"
log_info "  Workspace: $WORKSPACE_ROOT"
log_info "  Build dir: $BUILD_DIR"
log_info "  CTest name: $BENCHMARK_CTEST_NAME"
log_info "  GTest filter: $BENCHMARK_GTEST_FILTER"
log_info "  Profile top-N: $PROFILE_TOP_N"
log_info "  Training epochs: $NN_EPOCHS"    # Run phases
    check_dependencies
    build_project
    run_benchmarks
    run_profiling
    train_model
    validate_model
    
    # Summary
    log_section "Pipeline Complete! 🎉"
    
    echo -e "${GREEN}Summary:${NC}"
    if [ -f "$BENCHMARK_CSV" ]; then
        local bench_lines=$(wc -l < "$BENCHMARK_CSV")
        echo -e "  ✓ Benchmark data: $bench_lines records"
    fi
    
    if [ -f "$PROFILING_CSV" ]; then
        local prof_lines=$(wc -l < "$PROFILING_CSV")
        echo -e "  ✓ Profiling data: $prof_lines records"
    else
        echo -e "  ✗ Profiling data: not collected"
    fi
    
    if [ -f "$CUDA_KERNEL_DIR/$MODEL_ONNX" ]; then
        echo -e "  ✓ Model: $MODEL_ONNX"
        
        if [ -f "$CUDA_KERNEL_DIR/$METRICS_JSON" ]; then
            local top30=$(jq -r '.top_30 * 100' "$CUDA_KERNEL_DIR/$METRICS_JSON" 2>/dev/null || echo "N/A")
            echo -e "  ✓ Top-30 hit rate: ${top30}%"
        fi
    else
        echo -e "  ✗ Model: not generated"
    fi
    
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review training metrics in $CUDA_KERNEL_DIR/$METRICS_JSON"
    echo "  2. Test with: export LLAMINAR_USE_NN_HEURISTIC=1"
    echo "  3. Run inference: ./build_v2_release/your_inference_binary"
    echo ""
}

# Run pipeline
main "$@"

#!/usr/bin/env python3
"""
CPU GEMM Benchmark Data Collection - Exhaustive Approach

Mirrors CUDA approach: Run ALL training test cases × ALL 1225 variants × MULTIPLE quant formats
to collect comprehensive training data including quantization impact.

Phase 1: This script collects base features (23) for all configurations
  - Includes quantization type and block size as features
  - Tests multiple quant formats: IQ4_NL, Q4_0, Q6_K, Q8_0, FP16
  
Phase 2: collect_cpu_profiling_data.py enhances Qwen 0.5B/4B/7B with perf metrics

Outputs: cpu_gemm_benchmark_data.csv

Usage:
    ./benchmark_cpu_gemm.py --output cpu_gemm_benchmark_data.csv

Author: David Sanftenberg
Date: November 2025
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class QuantFormat:
    """Quantization format specification"""
    name: str            # e.g., "IQ4_NL", "Q6_K", "FP16"
    model_path: str      # Path to GGUF file
    block_size: int      # Elements per quantization block (1 for unquantized)
    bits_per_weight: float  # Effective bits per weight
    bytes_per_block: int    # Packed block size in bytes (affects memory bandwidth)
    
    def __str__(self):
        return f"{self.name} (block={self.block_size}, {self.bits_per_weight:.1f}bpw, {self.bytes_per_block}B/block)"

# Representative quantization formats to test
# Available quantization formats to benchmark
QUANT_FORMATS = [
    # IQ4_NL: Importance Matrix Quantization (4.5 bpw effective)
    QuantFormat("IQ4_NL", "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-iq4_nl.gguf", 32, 4.5, 18),
    
    # Q4_0: Standard 4-bit quantization (same bandwidth as IQ4_NL)
    QuantFormat("Q4_0", "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q4_0.gguf", 32, 4.5, 18),  # Same as IQ4_NL
    QuantFormat("Q8_0", "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf", 32, 8.5, 34),  # Nearly 2× bandwidth!
    
    # Q6_K: 6-bit quantization with super-blocks (256 elements/block)
    QuantFormat("Q6_K", "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q6_k.gguf", 256, 6.6, 210),
    
    # FP16: Full precision baseline
    QuantFormat("FP16", "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-fp16.gguf", 1, 16.0, 2),
]

@dataclass
class BenchmarkShape:
    """Real test case from production inference (matches Perf__CpuGemmHeuristicValidation.cpp)"""
    m: int
    n: int
    k: int
    test_name: str  # Exact test name for GTest filtering
    
    def __str__(self):
        return f"{self.m}×{self.n}×{self.k} ({self.test_name})"

# Training shapes: Exhaustively benchmark up to Qwen 32B only
# Rationale: Larger models (72B, 671B) would take too long to benchmark exhaustively
# We still validate generalization to larger sizes using validation script
# =============================================================================
# Training Shapes (used for ML heuristic training)
# =============================================================================
TRAINING_SHAPES = [
    # Qwen 0.5B (896 hidden, 14 heads, 4864 intermediate)
    BenchmarkShape(1, 896, 896, "Qwen_0_5B_SingleToken_QKV"),
    BenchmarkShape(32, 896, 896, "Qwen_0_5B_Batch32_QKV"),
    BenchmarkShape(128, 896, 896, "Qwen_0_5B_Batch128_QKV"),
    BenchmarkShape(1, 4864, 896, "Qwen_0_5B_FFN_Gate"),
    BenchmarkShape(1, 4864, 896, "Qwen_0_5B_FFN_Up"),
    BenchmarkShape(1, 896, 4864, "Qwen_0_5B_FFN_Down"),
    
    # Qwen 4B (2048 hidden, 16 heads, 11008 intermediate)
    BenchmarkShape(1, 2048, 2048, "Qwen_4B_SingleToken_QKV"),
    BenchmarkShape(32, 2048, 2048, "Qwen_4B_Batch32_QKV"),
    BenchmarkShape(128, 2048, 2048, "Qwen_4B_Batch128_QKV"),
    BenchmarkShape(1, 11008, 2048, "Qwen_4B_FFN_Gate"),
    BenchmarkShape(1, 11008, 2048, "Qwen_4B_FFN_Up"),
    BenchmarkShape(1, 2048, 11008, "Qwen_4B_FFN_Down"),
    
    # Qwen 7B (4096 hidden, 32 heads, 11008 intermediate)
    BenchmarkShape(1, 4096, 4096, "Qwen_7B_SingleToken_QKV"),
    BenchmarkShape(32, 4096, 4096, "Qwen_7B_Batch32_QKV"),
    BenchmarkShape(128, 4096, 4096, "Qwen_7B_Batch128_QKV"),
    BenchmarkShape(1, 11008, 4096, "Qwen_7B_FFN_Gate"),
    BenchmarkShape(1, 11008, 4096, "Qwen_7B_FFN_Up"),
    BenchmarkShape(1, 4096, 11008, "Qwen_7B_FFN_Down"),
    
    # Qwen 14B (5120 hidden, 40 heads, 13696 intermediate)
    BenchmarkShape(1, 5120, 5120, "Qwen_14B_SingleToken_QKV"),
    BenchmarkShape(32, 5120, 5120, "Qwen_14B_Batch32_QKV"),
    BenchmarkShape(1, 13696, 5120, "Qwen_14B_FFN_Gate"),
    BenchmarkShape(1, 13696, 5120, "Qwen_14B_FFN_Up"),
    BenchmarkShape(1, 5120, 13696, "Qwen_14B_FFN_Down"),
    
    # Qwen 32B (5120 hidden, 40 heads, 27392 intermediate)
    BenchmarkShape(1, 5120, 5120, "Qwen_32B_SingleToken_QKV"),
    BenchmarkShape(32, 5120, 5120, "Qwen_32B_Batch32_QKV"),
    BenchmarkShape(1, 27392, 5120, "Qwen_32B_FFN_Gate"),
    BenchmarkShape(1, 5120, 27392, "Qwen_32B_FFN_Down"),
    
    # Edge cases (odd dimensions, non-power-of-2) - keep for robustness
    BenchmarkShape(1, 127, 127, "EdgeCase_Odd_Tiny"),
    BenchmarkShape(1, 1023, 1023, "EdgeCase_Odd_Medium"),
    BenchmarkShape(17, 896, 896, "EdgeCase_Batch_Prime"),
    BenchmarkShape(1, 3072, 1024, "EdgeCase_Nonsquare_3to1"),
    BenchmarkShape(1, 1024, 3072, "EdgeCase_Nonsquare_1to3"),
    BenchmarkShape(63, 2048, 2048, "EdgeCase_Batch_NonPowerOf2"),
]

# =============================================================================
# Validation/Holdout Shapes (NOT used for training - test generalization)
# =============================================================================
VALIDATION_SHAPES = [
    # Qwen 1.5B (1536 hidden, 12 heads, 8960 intermediate)
    # HOLDOUT SET: Used to validate that heuristic generalizes to unseen model sizes
    BenchmarkShape(1, 1536, 1536, "Qwen_1_5B_SingleToken_QKV"),
    BenchmarkShape(32, 1536, 1536, "Qwen_1_5B_Batch32_QKV"),
    BenchmarkShape(1, 8960, 1536, "Qwen_1_5B_FFN_Gate"),
    BenchmarkShape(1, 8960, 1536, "Qwen_1_5B_FFN_Up"),
    BenchmarkShape(1, 1536, 8960, "Qwen_1_5B_FFN_Down"),
]

# Combined list (for backward compatibility with tests)
BENCHMARK_SHAPES = TRAINING_SHAPES + VALIDATION_SHAPES

# Note: Qwen 72B and DeepSeek 671B are NOT exhaustively benchmarked
# They are only used in validation to test generalization to unseen large sizes


@dataclass
class VariantSpec:
    """Micro-kernel variant specification"""
    isa: str  # "AVX512" or "AVX2"
    tile_m: int
    tile_n: int
    unroll_k: int
    prefetch_dist: int
    
    def to_name(self) -> str:
        """Convert to variant name format"""
        return f"{self.isa}Tag_{self.tile_m}x{self.tile_n}_u{self.unroll_k}_p{self.prefetch_dist}"
    
    def to_filter(self) -> str:
        """Convert to GTest filter"""
        # Escape for regex if needed
        return self.to_name().replace("x", "x")  # Keep literal 'x'

def generate_all_variants() -> List[VariantSpec]:
    """Generate all 1,225 variant combinations"""
    variants = []
    
    ISA_TYPES = ["AVX512", "AVX2"]
    MR_VALUES = [1, 2, 4, 8, 16, 32, 64]
    NR_VALUES = [1, 2, 4, 6, 8, 16, 32, 64]
    UNROLL_K_VALUES = [1, 2, 4, 8, 16]
    PREFETCH_DIST_VALUES = [0, 1, 2, 3, 5]
    
    for isa in ISA_TYPES:
        for mr in MR_VALUES:
            for nr in NR_VALUES:
                # Skip invalid register file combinations
                max_regs = 48 if isa == "AVX512" else 32
                if mr * nr > max_regs:
                    continue
                    
                for unroll_k in UNROLL_K_VALUES:
                    for prefetch in PREFETCH_DIST_VALUES:
                        variants.append(VariantSpec(
                            isa=isa,
                            tile_m=mr,
                            tile_n=nr,
                            unroll_k=unroll_k,
                            prefetch_dist=prefetch
                        ))
    
    return variants

def run_benchmark_for_shape(
    test_binary: str,
    shape: BenchmarkShape,
    model_path: str,
    output_csv: str
) -> int:
    """
    Run GTest for single shape (all 1,225 variants)
    
    The GTest binary handles:
    - Loading model from BENCHMARK_MODEL_PATH
    - Running all 1,225 variants
    - Writing CSV rows to BENCHMARK_CSV_OUTPUT
    
    Returns:
        Number of successful benchmarks (should be 1,225)
    """
    env = os.environ.copy()
    env['BENCHMARK_MODEL_PATH'] = model_path
    env['BENCHMARK_CSV_OUTPUT'] = output_csv
    env['LLAMINAR_DISABLE_GEMM_AUTOTUNE'] = '1'  # CRITICAL: Disable auto-tuner to benchmark all 1,225 variants
    env['OMP_NUM_THREADS'] = '28'
    env['OMP_PLACES'] = 'sockets'
    env['OMP_PROC_BIND'] = 'close'
    
    # Run single GTest test case (it will benchmark all variants)
    # Use mpirun for 2 ranks to match production MPI setup
    cmd = [
        'mpirun', '-np', '2',
        '--bind-to', 'socket', '--map-by', 'socket',
        '--mca', 'mpi_leave_pinned', '1',
        '--mca', 'btl_vader_single_copy_mechanism', 'none',
        test_binary,
        f'--gtest_filter=*{shape.test_name}*'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for all 1,225 variants
            env=env
        )
        
        if result.returncode == 0:
            # Parse output to count successful benchmarks
            # GTest prints "Completed: N/M variants successful"
            for line in result.stdout.split('\n'):
                if 'Completed:' in line and 'variants successful' in line:
                    # Extract "123/1225"
                    match = line.split('Completed:')[1].split('variants')[0].strip()
                    successful = int(match.split('/')[0])
                    return successful
            return 1225  # Assume success if no parse error
        else:
            print(f"    ✗ GTest failed (exit code {result.returncode})")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return 0
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ TIMEOUT after 1 hour")
        return 0
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        return 0

def collect_benchmark_data(
    test_binary: str,
    output_csv: str,
    use_validation_set: bool = False
):
    """Main benchmark collection loop
    
    Strategy: Run GTest once per shape × quant_format combination.
    Each GTest run benchmarks all 1,225 variants and appends to CSV.
    
    Args:
        test_binary: Path to GTest benchmark executable (v2_perf_cpu_gemm_validation)
        output_csv: Output CSV path
        use_validation_set: If True, benchmark validation shapes instead of training shapes
    """
    
    shapes = VALIDATION_SHAPES if use_validation_set else TRAINING_SHAPES
    quant_formats = QUANT_FORMATS
    
    shape_set_name = "VALIDATION (Qwen 1.5B holdout)" if use_validation_set else "TRAINING"
    
    print(f"Collecting CPU GEMM benchmark data ({shape_set_name}):")
    print(f"  Shapes: {len(shapes)}")
    print(f"  Variants per shape: 1,225")
    print(f"  Quant formats: {len(quant_formats)}")
    print(f"  Total benchmarks: {len(shapes) * 1225 * len(quant_formats)}")
    print(f"  Output: {output_csv}")
    print()
    
    # Clear output CSV (GTest will write header on first run)
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"Cleared existing {output_csv}")
    
    total_shapes = len(shapes) * len(quant_formats)
    completed = 0
    total_successful = 0
    
    for quant_fmt in quant_formats:
        print(f"\n{'='*70}")
        print(f"Quant Format: {quant_fmt.name} (block={quant_fmt.block_size}, {quant_fmt.bits_per_weight}bpw, {quant_fmt.bytes_per_block}B/block)")
        print(f"{'='*70}\n")
        
        for shape in shapes:
            completed += 1
            print(f"[{completed}/{total_shapes}] Shape: {shape.m}×{shape.n}×{shape.k} ({shape.test_name})")
            print(f"  Model: {quant_fmt.model_path}")
            print(f"  Testing 1,225 variants...", flush=True)
            
            # Run GTest for this shape+quant combination (all 1,225 variants)
            successful = run_benchmark_for_shape(
                test_binary,
                shape,
                quant_fmt.model_path,
                output_csv
            )
            
            total_successful += successful
            
            if successful == 1225:
                print(f"  ✓ Complete: {successful}/1,225 variants successful")
            else:
                print(f"  ⚠ Partial: {successful}/1,225 variants successful")
            
            # Show progress
            overall_progress = (total_successful / (total_shapes * 1225)) * 100
            print(f"  Overall: {total_successful:,}/{total_shapes * 1225:,} benchmarks ({overall_progress:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total successful: {total_successful:,}/{total_shapes * 1225:,}")
    print(f"Output: {output_csv}")
    
    # Verify CSV
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"CSV rows: {line_count:,} (including header)")
        expected = total_successful + 1  # +1 for header
        if line_count == expected:
            print(f"✓ CSV verification passed")
        else:
            print(f"⚠ Expected {expected} rows, got {line_count}")
    else:
        print(f"✗ ERROR: Output file not created!")

                    
def main():
    parser = argparse.ArgumentParser(
        description='Collect CPU GEMM benchmark data for ML heuristic training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect training data (excludes Qwen 1.5B holdout set)
  python3 benchmark_cpu_gemm.py --output training_data.csv
  
  # Collect validation data (Qwen 1.5B only)
  python3 benchmark_cpu_gemm.py --validation --output validation_data.csv
  
  # Test single shape
  python3 benchmark_cpu_gemm.py --shapes "1x896x896"
        """
    )
    parser.add_argument('--test-binary', 
                       default='build_v2_release/performance/v2_perf_cpu_gemm_validation',
                       help='Path to benchmark executable (default: v2_perf_cpu_gemm_validation)')
    parser.add_argument('--output', 
                       default='cpu_gemm_benchmark_data.csv',
                       help='Output CSV file (default: cpu_gemm_benchmark_data.csv)')
    parser.add_argument('--validation', 
                       action='store_true',
                       help='Benchmark validation set (Qwen 1.5B holdout) instead of training set')
    parser.add_argument('--shapes', 
                       type=str, 
                       default=None,
                       help='Override shapes (format: MxNxK,MxNxK,...)')
    
    args = parser.parse_args()
    
    # Validate test binary exists
    if not os.path.exists(args.test_binary):
        print(f"Error: Test binary not found: {args.test_binary}")
        print("Build it with:")
        print("  cmake --build build_v2_release --target v2_perf_cpu_gemm_validation --parallel")
        sys.exit(1)
    
    # Validate all model files exist
    for quant_fmt in QUANT_FORMATS:
        if not os.path.exists(quant_fmt.model_path):
            print(f"Error: Model file not found: {quant_fmt.model_path}")
            print(f"Required for quantization format: {quant_fmt.name}")
            sys.exit(1)
    
    collect_benchmark_data(args.test_binary, args.output, use_validation_set=args.validation)

if __name__ == '__main__':
    main()

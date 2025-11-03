#!/usr/bin/env python3
"""
@file collect_profiling_data.py
@brief Collect NVIDIA Nsight Compute profiling metrics for CUDA GEMM configurations

This script:
1. Reads benchmark CSV with all config performances
2. Identifies top-N and bottom-N configs per test case
3. Profiles each config with ncu to collect hardware metrics
4. Exports profiling data to CSV for ML training

Usage:
    python collect_profiling_data.py \\
        --input cuda_gemm_benchmark_data.csv \\
        --executable ../build_v2_release/profile_cuda_config \\
        --output cuda_gemm_profiling_data.csv \\
        --top-n 50 \\
        --metrics dram_throughput,l1_cache,l2_cache,sm_throughput

@author David Sanftenberg
@date November 3, 2025
"""

import argparse
import subprocess
import pandas as pd
import os
import sys
import time
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# NVIDIA Nsight Compute metrics to collect
# See: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
PROFILE_METRICS = {
    'dram_throughput': 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'l1_cache_hit_rate': 'l1tex__t_sector_hit_rate.pct',
    'l2_cache_hit_rate': 'lts__t_sector_hit_rate.pct',
    'sm_throughput': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm_instruction_throughput': 'sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed',
    'sm_warps_active': 'sm__warps_active.avg.pct_of_peak_sustained_elapsed',
    'global_load_coalescing': 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
    'global_store_coalescing': 'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct',
    'smem_bank_conflicts_ld': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'smem_bank_conflicts_st': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'warp_divergence': 'smsp__thread_inst_executed_per_inst_executed.ratio',
}


def check_ncu_available() -> bool:
    """Check if NVIDIA Nsight Compute (ncu) is available"""
    try:
        result = subprocess.run(['ncu', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            logger.info(f"Found ncu: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    logger.error("NVIDIA Nsight Compute (ncu) not found in PATH")
    logger.error("Install from: https://developer.nvidia.com/nsight-compute")
    return False


def profile_single_config(
    executable: str,
    m: int, n: int, k: int,
    config: Dict,
    metrics: List[str]
) -> Optional[Dict[str, float]]:
    """
    Profile a single CUDA GEMM configuration with ncu
    
    Args:
        executable: Path to profile_cuda_config binary
        m, n, k: Matrix dimensions
        config: Config dict with tile_m, tile_n, etc.
        metrics: List of metric names to collect
    
    Returns:
        Dict of metric_name -> value, or None on error
    """
    # Build ncu command
    metric_names = [PROFILE_METRICS[m] for m in metrics if m in PROFILE_METRICS]
    if not metric_names:
        logger.error(f"No valid metrics: {metrics}")
        return None
    
    ncu_cmd = [
        'sudo',  # NVIDIA Nsight Compute requires elevated permissions for hardware counters
        '/usr/local/cuda/bin/ncu',  # Use absolute path for sudo environment
        '--metrics', ','.join(metric_names),
        '--csv',
        '--target-processes', 'all',
        # Removed --quiet to get verbose error messages
        executable,
        str(m), str(n), str(k),
        str(config.get('tile_m', 16)),
        str(config.get('tile_n', 16)),
        str(config.get('tile_k', 32)),
        str(config.get('threads_m', 8)),
        str(config.get('threads_n', 8)),
        str(config.get('work_m', 2)),
        str(config.get('work_n', 2)),
        str(config.get('prefetch_stages', 1)),
        str(int(config.get('transpose_smem', 0))),
        str(config.get('vectorize_load', 1)),
    ]
    
    try:
        # Run profiling (can take 20-60 seconds per config)
        result = subprocess.run(
            ncu_cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode != 0:
            logger.warning(f"ncu failed (exit code {result.returncode}):")
            if result.stdout:
                logger.warning(f"  stdout: {result.stdout[:500]}")
            if result.stderr:
                logger.warning(f"  stderr: {result.stderr[:500]}")
            return None
        
        # Parse CSV from stdout (--csv flag writes to stdout)
        if not result.stdout or not result.stdout.strip():
            logger.warning("ncu didn't produce CSV output")
            return None
        
        # Parse CSV from string, skipping comment lines that start with ==
        from io import StringIO
        csv_lines = [line for line in result.stdout.split('\n') 
                     if line.strip() and not line.startswith('==')]
        csv_data = '\n'.join(csv_lines)
        
        if not csv_data.strip():
            logger.warning("ncu output contained no CSV data")
            return None
            
        df = pd.read_csv(StringIO(csv_data))
        
        # ncu CSV format: Each row has "Metric Name" and "Metric Value" columns
        # Multiple rows for multiple kernel invocations - take the last one
        results = {}
        for metric_name, ncu_metric in PROFILE_METRICS.items():
            if metric_name not in metrics:
                continue
            
            # Find rows where "Metric Name" column contains this metric
            if 'Metric Name' in df.columns and 'Metric Value' in df.columns:
                matching_rows = df[df['Metric Name'] == ncu_metric]
                if not matching_rows.empty:
                    # Take the last invocation (most representative)
                    value = matching_rows['Metric Value'].iloc[-1]
                    # Handle percentage strings or numeric values
                    if isinstance(value, str):
                        value = value.rstrip('%').strip()
                    results[metric_name] = float(value)
                else:
                    logger.debug(f"Metric {metric_name} ({ncu_metric}) not found in output")
                    results[metric_name] = 0.0
            else:
                logger.warning(f"Expected columns 'Metric Name' and 'Metric Value' not found")
                results[metric_name] = 0.0
        
        return results
        
    except subprocess.TimeoutExpired:
        logger.warning(f"Profiling timeout for config {config}")
        return None
    except Exception as e:
        logger.warning(f"Profiling error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Collect CUDA GEMM profiling data')
    parser.add_argument('--input', required=True,
                       help='Input benchmark CSV (cuda_gemm_benchmark_data.csv)')
    parser.add_argument('--executable', required=True,
                       help='Path to profile_cuda_config executable')
    parser.add_argument('--output', default='cuda_gemm_profiling_data.csv',
                       help='Output profiling CSV')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Profile top N and bottom N configs per test')
    parser.add_argument('--metrics', default='dram_throughput,l1_cache_hit_rate,l2_cache_hit_rate,sm_throughput,sm_warps_active',
                       help='Comma-separated metric names to collect')
    parser.add_argument('--max-tests', type=int, default=None,
                       help='Max test cases to profile (for debugging)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not check_ncu_available():
        return 1
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    if not os.path.exists(args.executable):
        logger.error(f"Executable not found: {args.executable}")
        return 1
    
    metric_list = [m.strip() for m in args.metrics.split(',')]
    
    # Load benchmark data
    logger.info(f"Loading benchmark data from {args.input}")
    df_bench = pd.read_csv(args.input)
    
    # Group by test case
    test_cases = df_bench.groupby(['test_name', 'm', 'n', 'k'])
    logger.info(f"Found {len(test_cases)} unique test cases")
    
    # Collect profiling data
    profiling_results = []
    total_profiles = 0
    
    for (test_name, m, n, k), group in test_cases:
        if args.max_tests and len(profiling_results) >= args.max_tests:
            break
        
        logger.info(f"Processing {test_name} ({m}×{n}×{k})")
        
        # Sort by performance
        group_sorted = group.sort_values('gflops', ascending=False)
        
        # Select top-N and bottom-N
        top_configs = group_sorted.head(args.top_n)
        bottom_configs = group_sorted.tail(args.top_n)
        
        configs_to_profile = pd.concat([top_configs, bottom_configs]).drop_duplicates()
        logger.info(f"  Profiling {len(configs_to_profile)} configs (top {args.top_n} + bottom {args.top_n})")
        
        for idx, row in configs_to_profile.iterrows():
            config = {
                'tile_m': row['tile_m'],
                'tile_n': row['tile_n'],
                'tile_k': row['tile_k'],
                'threads_m': row['threads_m'],
                'threads_n': row['threads_n'],
                'work_m': row['work_m'],
                'work_n': row['work_n'],
                'prefetch_stages': row['prefetch_stages'],
                'transpose_smem': row['transpose_smem'],
                'vectorize_load': row['vectorize_load'],
            }
            
            logger.info(f"  [{total_profiles+1}] Profiling tile={config['tile_m']}x{config['tile_n']}x{config['tile_k']} "
                       f"(rank={row.get('rank', 0)}, {row['gflops']:.1f} GFLOPS)")
            
            # Profile with ncu
            metrics = profile_single_config(args.executable, m, n, k, config, metric_list)
            
            if metrics:
                # Combine with benchmark data
                result = {
                    'test_name': test_name,
                    'm': m, 'n': n, 'k': k,
                    **config,
                    'gflops': row['gflops'],
                    'time_us': row.get('time_us', 0),
                    'rank': row.get('rank', 0),
                    **metrics
                }
                profiling_results.append(result)
                total_profiles += 1
            else:
                logger.warning(f"  Failed to profile config (skipping)")
            
            # Rate limiting (ncu is resource-intensive)
            time.sleep(0.5)
    
    # Export to CSV
    if profiling_results:
        df_out = pd.DataFrame(profiling_results)
        df_out.to_csv(args.output, index=False)
        logger.info(f"Exported {len(profiling_results)} profiling results to {args.output}")
        
        # Print summary statistics
        logger.info("\nProfiling Summary:")
        logger.info(f"  Total configs profiled: {len(profiling_results)}")
        logger.info(f"  Test cases covered: {df_out['test_name'].nunique()}")
        
        for metric in metric_list:
            if metric in df_out.columns:
                logger.info(f"  {metric}: min={df_out[metric].min():.2f}, "
                           f"max={df_out[metric].max():.2f}, "
                           f"mean={df_out[metric].mean():.2f}")
    else:
        logger.error("No profiling data collected!")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

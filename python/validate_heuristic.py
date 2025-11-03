#!/usr/bin/env python3
"""
@file validate_heuristic.py
@brief Validate trained CUDA heuristic on unseen test shapes

This script:
1. Loads the trained ONNX model
2. Loads benchmark data from unseen shapes
3. Predicts GFLOPS for each config
4. Compares predicted top-N configs vs actual top-N
5. Reports hit rates and ranking accuracy

Usage:
    python validate_heuristic.py \\
        --model cuda_heuristic_nn.onnx \\
        --scaler feature_scaler.bin \\
        --benchmark cuda_gemm_validation_data.csv \\
        --output validation_results.json

@author David Sanftenberg
@date November 3, 2025
"""

import argparse
import pandas as pd
import numpy as np
import json
import pickle
import logging
import os
from typing import List, Dict, Tuple
import sys

# ML libraries
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, scaler_path: str) -> Tuple[ort.InferenceSession, StandardScaler]:
    """Load ONNX model and feature scaler"""
    logger.info(f"Loading model from {model_path}")
    session = ort.InferenceSession(model_path)
    
    logger.info(f"Loading scaler from {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler_dict = pickle.load(f)
    
    # Reconstruct StandardScaler from dict
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_dict['mean'])
    scaler.scale_ = np.array(scaler_dict['std'])
    scaler.n_features_in_ = len(scaler_dict['mean'])
    
    return session, scaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features (same as training script)
    Copy-paste from train_cuda_neural_network.py to ensure consistency
    """
    features = df.copy()
    
    # Ratios
    features['m_div_n'] = features['m'] / (features['n'] + 1e-6)
    features['n_div_k'] = features['n'] / (features['k'] + 1e-6)
    features['k_div_m'] = features['k'] / (features['m'] + 1e-6)
    features['tile_m_div_tile_n'] = features['tile_m'] / (features['tile_n'] + 1e-6)
    features['tile_n_div_tile_k'] = features['tile_n'] / (features['tile_k'] + 1e-6)
    features['threads_m_div_threads_n'] = features['threads_m'] / (features['threads_n'] + 1e-6)
    features['work_m_div_work_n'] = features['work_m'] / (features['work_n'] + 1e-6)
    features['m_div_tile_m'] = features['m'] / (features['tile_m'] + 1e-6)
    features['n_div_tile_n'] = features['n'] / (features['tile_n'] + 1e-6)
    
    # Work metrics
    features['total_work'] = features['m'] * features['n'] * features['k']
    features['work_per_thread'] = features['work_m'] * features['work_n']
    features['threads_per_block'] = features['threads_m'] * features['threads_n']
    features['tile_area_mk'] = features['tile_m'] * features['tile_k']
    features['tile_area_nk'] = features['tile_n'] * features['tile_k']
    features['tile_volume'] = features['tile_m'] * features['tile_n'] * features['tile_k']
    
    # Resource usage
    features['smem_usage_est'] = (
        features['tile_m'] * features['tile_k'] + 
        features['tile_k'] * features['tile_n']
    ) * 4  # 4 bytes per float
    
    features['reg_pressure_est'] = (
        features['work_m'] * features['work_n'] * features['tile_k']
    )
    
    # Alignment
    features['m_mod_tile_m'] = features['m'] % features['tile_m']
    features['n_mod_tile_n'] = features['n'] % features['tile_n']
    features['k_mod_tile_k'] = features['k'] % features['tile_k']
    
    # Workload distribution
    features['m_tiles'] = np.ceil(features['m'] / features['tile_m'])
    features['n_tiles'] = np.ceil(features['n'] / features['tile_n'])
    features['k_tiles'] = np.ceil(features['k'] / features['tile_k'])
    features['total_tiles'] = features['m_tiles'] * features['n_tiles']
    
    # Thread utilization
    features['elems_per_thread_m'] = features['tile_m'] / (features['threads_m'] + 1e-6)
    features['elems_per_thread_n'] = features['tile_n'] / (features['threads_n'] + 1e-6)
    features['compute_per_thread'] = (
        features['work_m'] * features['work_n'] * features['tile_k'] * 2
    )  # 2 ops per MAC
    
    # Vectorization efficiency
    features['vectorize_factor'] = features['vectorize_load']
    features['effective_bandwidth'] = features['vectorize_load'] * 4  # bytes
    
    # Prefetching
    features['prefetch_factor'] = features['prefetch_stages']
    
    # Memory access patterns
    features['smem_transpose_penalty'] = features['transpose_smem'].astype(float)
    
    # Problem shape characteristics
    features['is_square'] = (features['m'] == features['n']).astype(float)
    features['is_tall'] = (features['m'] > features['n']).astype(float)
    features['is_wide'] = (features['n'] > features['m']).astype(float)
    
    # Batch characteristics
    features['is_batch'] = (features['m'] > 1).astype(float)
    features['batch_size'] = features['m']
    
    # Tile shape characteristics
    features['tile_is_square'] = (features['tile_m'] == features['tile_n']).astype(float)
    features['tile_aspect_ratio'] = features['tile_m'] / (features['tile_n'] + 1e-6)
    
    # Thread block characteristics
    features['threads_is_square'] = (features['threads_m'] == features['threads_n']).astype(float)
    features['threads_aspect_ratio'] = features['threads_m'] / (features['threads_n'] + 1e-6)
    
    # Work distribution characteristics
    features['work_is_balanced'] = (features['work_m'] == features['work_n']).astype(float)
    features['work_aspect_ratio'] = features['work_m'] / (features['work_n'] + 1e-6)
    
    # Occupancy estimates
    features['warps_per_block'] = np.ceil(features['threads_per_block'] / 32.0)
    features['blocks_per_sm_smem'] = np.floor(49152 / (features['smem_usage_est'] + 1))  # 48KB shared memory
    features['blocks_per_sm_threads'] = np.floor(2048 / (features['threads_per_block'] + 1))  # Max threads per SM
    features['occupancy_est'] = np.minimum(
        features['blocks_per_sm_smem'],
        features['blocks_per_sm_threads']
    )
    
    # Arithmetic intensity
    features['arithmetic_intensity'] = (
        2.0 * features['m'] * features['n'] * features['k']
    ) / (
        features['m'] * features['k'] + features['k'] * features['n'] + features['m'] * features['n']
    )
    
    # Tile efficiency
    features['tile_efficiency_m'] = 1.0 - (features['m_mod_tile_m'] / (features['tile_m'] + 1e-6))
    features['tile_efficiency_n'] = 1.0 - (features['n_mod_tile_n'] / (features['tile_n'] + 1e-6))
    features['tile_efficiency_k'] = 1.0 - (features['k_mod_tile_k'] / (features['tile_k'] + 1e-6))
    
    # Computational complexity
    features['flops_total'] = 2.0 * features['m'] * features['n'] * features['k']  # 2 ops per MAC
    features['flops_per_tile'] = 2.0 * features['tile_m'] * features['tile_n'] * features['tile_k']
    features['flops_per_thread'] = features['flops_per_tile'] / (features['threads_per_block'] + 1e-6)
    
    # Memory footprint
    features['bytes_A'] = features['m'] * features['k'] * 4
    features['bytes_B'] = features['k'] * features['n'] * 4
    features['bytes_C'] = features['m'] * features['n'] * 4
    features['total_memory'] = features['bytes_A'] + features['bytes_B'] + features['bytes_C']
    
    # Cache-level metrics
    features['l2_footprint'] = np.minimum(features['total_memory'], 6291456)  # 6MB L2 cache
    features['l1_footprint'] = np.minimum(features['smem_usage_est'], 196608)  # 192KB L1 cache
    
    # Warp-level metrics
    features['warps_per_tile_m'] = np.ceil(features['tile_m'] / 32.0)
    features['warps_per_tile_n'] = np.ceil(features['tile_n'] / 32.0)
    
    # Power-of-2 alignment
    features['tile_m_is_pow2'] = ((features['tile_m'] & (features['tile_m'] - 1)) == 0).astype(float)
    features['tile_n_is_pow2'] = ((features['tile_n'] & (features['tile_n'] - 1)) == 0).astype(float)
    features['tile_k_is_pow2'] = ((features['tile_k'] & (features['tile_k'] - 1)) == 0).astype(float)
    
    # Thread grid characteristics
    features['grid_size_x'] = np.ceil(features['m'] / features['tile_m'])
    features['grid_size_y'] = np.ceil(features['n'] / features['tile_n'])
    features['total_thread_blocks'] = features['grid_size_x'] * features['grid_size_y']
    
    # Load balance
    features['load_imbalance_m'] = features['m_mod_tile_m'] / (features['tile_m'] + 1e-6)
    features['load_imbalance_n'] = features['n_mod_tile_n'] / (features['tile_n'] + 1e-6)
    
    # Replace inf/NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)
    
    return features


def predict_gflops(
    session: ort.InferenceSession,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """Predict GFLOPS using ONNX model"""
    
    # Extract features
    X = df[feature_cols].values
    
    # Check if we have fewer features than scaler expects (validation without profiling)
    n_features = X.shape[1]
    expected_features = scaler.n_features_in_
    
    if n_features < expected_features:
        # Pad with zeros for missing profiling features
        logger.info(f"Padding {expected_features - n_features} missing profiling features with zeros")
        padding = np.zeros((X.shape[0], expected_features - n_features), dtype=X.dtype)
        X = np.hstack([X, padding])
    elif n_features > expected_features:
        raise ValueError(f"Feature count mismatch: got {n_features}, expected {expected_features}")
    
    # Normalize
    X_scaled = scaler.transform(X)
    
    # Predict
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
    
    return predictions.flatten()


def evaluate_ranking(
    df: pd.DataFrame,
    test_cases: List[str],
    top_ns: List[int] = [1, 5, 10, 30]
) -> Dict:
    """
    Evaluate ranking accuracy by comparing predicted top-N vs actual top-N
    """
    results = {f'top_{n}': [] for n in top_ns}
    results['kendall_tau'] = []
    results['spearman_rho'] = []
    
    for test_name in test_cases:
        test_df = df[df['test_name'] == test_name].copy()
        
        if len(test_df) == 0:
            logger.warning(f"No data for {test_name}")
            continue
        
        # Sort by actual GFLOPS (ground truth)
        test_df = test_df.sort_values('gflops', ascending=False).reset_index(drop=True)
        actual_top_indices = test_df.index.tolist()
        
        # Sort by predicted GFLOPS
        test_df = test_df.sort_values('predicted_gflops', ascending=False).reset_index(drop=True)
        predicted_top_indices = test_df.index.tolist()
        
        # Compute hit rates for different top-N
        for n in top_ns:
            n_configs = min(n, len(test_df))
            actual_top_n = set(actual_top_indices[:n_configs])
            predicted_top_n = set(predicted_top_indices[:n_configs])
            
            hit_rate = len(actual_top_n & predicted_top_n) / n_configs
            results[f'top_{n}'].append(hit_rate)
        
        # Compute rank correlation (Kendall's tau)
        from scipy.stats import kendalltau, spearmanr
        tau, _ = kendalltau(test_df['gflops'], test_df['predicted_gflops'])
        rho, _ = spearmanr(test_df['gflops'], test_df['predicted_gflops'])
        
        results['kendall_tau'].append(tau)
        results['spearman_rho'].append(rho)
    
    # Compute means
    summary = {}
    for key, values in results.items():
        if values:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Validate CUDA heuristic on unseen shapes')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--scaler', required=True, help='Path to feature scaler')
    parser.add_argument('--benchmark', required=True, help='Path to validation benchmark CSV')
    parser.add_argument('--output', default='validation_results.json', help='Output JSON file')
    parser.add_argument('--top-n', type=int, nargs='+', default=[1, 5, 10, 30],
                       help='Top-N values to evaluate')
    
    args = parser.parse_args()
    
    # Load model
    session, scaler = load_model(args.model, args.scaler)
    
    # Load benchmark data
    logger.info(f"Loading validation data from {args.benchmark}")
    df = pd.read_csv(args.benchmark)
    logger.info(f"Loaded {len(df)} validation records")
    
    # Get unique test cases
    test_cases = df['test_name'].unique().tolist()
    logger.info(f"Found {len(test_cases)} unique test cases")
    
    # Engineer features
    logger.info("Engineering features...")
    df = engineer_features(df)
    
    # Get feature columns (same as training)
    base_features = ['m', 'n', 'k', 'tile_m', 'tile_n', 'tile_k',
                    'threads_m', 'threads_n', 'work_m', 'work_n',
                    'prefetch_stages', 'transpose_smem', 'vectorize_load']
    
    all_feature_cols = [col for col in df.columns if col not in 
                       ['test_name', 'gflops', 'time_us', 'rank']]
    
    logger.info(f"Using {len(all_feature_cols)} features")
    
    # Predict GFLOPS
    logger.info("Predicting GFLOPS...")
    predictions = predict_gflops(session, scaler, df, all_feature_cols)
    df['predicted_gflops'] = predictions
    
    # Evaluate ranking accuracy
    logger.info("Evaluating ranking accuracy...")
    ranking_results = evaluate_ranking(df, test_cases, args.top_n)
    
    # Compute regression metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    
    r2 = r2_score(df['gflops'], df['predicted_gflops'])
    mae = mean_absolute_error(df['gflops'], df['predicted_gflops'])
    mape = np.mean(np.abs((df['gflops'] - df['predicted_gflops']) / df['gflops'])) * 100
    
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"MAE: {mae:.2f} GFLOPS")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Log ranking results
    for key, stats in ranking_results.items():
        logger.info(f"{key}: mean={stats['mean']:.2%}, std={stats['std']:.2%}")
    
    # Export results
    results = {
        'regression_metrics': {
            'r2_score': r2,
            'mae': mae,
            'mape': mape
        },
        'ranking_metrics': ranking_results,
        'test_cases': test_cases,
        'num_configs': len(df)
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Exported validation results to {args.output}")
    
    return 0 if r2 > 0.95 else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
@file train_cuda_neural_network.py
@brief Train a neural network to predict optimal CUDA GEMM configurations

This script:
1. Loads benchmark data (cuda_gemm_benchmark_data.csv)
2. Optionally loads profiling data (cuda_gemm_profiling_data.csv)
3. Engineers features from config parameters + problem dimensions
4. Trains a neural network to predict GFLOPS
5. Exports model to ONNX for C++ inference

Features engineered (73 base + 11 profiling = 84 total):
- Problem dimensions: m, n, k
- Config parameters: tile sizes, thread counts, work per thread, etc.
- Derived: ratios, alignment, work distribution, resource usage
- Profiling (if available): cache hit rates, occupancy, memory coalescing

Model architecture:
- Input: 84 features (normalized)
- Hidden: 256 -> 128 -> 64 neurons (ReLU + Dropout)
- Output: 1 (predicted GFLOPS)

Training:
- Loss: MSE on GFLOPS
- Optimizer: Adam (lr=0.001)
- Validation: 20% holdout
- Epochs: 100 (early stopping)

Exports:
- cuda_heuristic_nn.onnx: ONNX model for C++ inference
- feature_scaler.bin: Feature normalization parameters
- training_metrics.json: Validation metrics (R^2, MAE, top-30 hit rate)

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
from typing import List, Dict, Tuple, Optional
import sys

# ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CudaHeuristicNN(nn.Module):
    """Neural network for CUDA GEMM performance prediction"""
    
    def __init__(self, input_dim: int = 84):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)  # Predict GFLOPS
        )
    
    def forward(self, x):
        return self.net(x)


def engineer_features(df: pd.DataFrame, include_profiling: bool = False) -> pd.DataFrame:
    """
    Engineer features from raw benchmark data
    
    Features:
    - Problem dimensions (3): m, n, k
    - Config parameters (13): tile sizes, threads, work, prefetch, etc.
    - Derived features (57): ratios, work metrics, resource usage, alignment
    - Profiling features (11, optional): cache hits, occupancy, coalescing
    
    Total: 73 base features + 11 profiling = 84
    """
    logger.info("Engineering features...")
    
    features = df.copy()
    
    # 1. Problem dimensions (already present)
    # m, n, k
    
    # 2. Ratios (9)
    features['m_div_n'] = features['m'] / (features['n'] + 1e-6)
    features['n_div_k'] = features['n'] / (features['k'] + 1e-6)
    features['k_div_m'] = features['k'] / (features['m'] + 1e-6)
    features['tile_m_div_tile_n'] = features['tile_m'] / (features['tile_n'] + 1e-6)
    features['tile_n_div_tile_k'] = features['tile_n'] / (features['tile_k'] + 1e-6)
    features['threads_m_div_threads_n'] = features['threads_m'] / (features['threads_n'] + 1e-6)
    features['work_m_div_work_n'] = features['work_m'] / (features['work_n'] + 1e-6)
    features['m_div_tile_m'] = features['m'] / (features['tile_m'] + 1e-6)
    features['n_div_tile_n'] = features['n'] / (features['tile_n'] + 1e-6)
    
    # 3. Work metrics (10)
    features['total_work'] = features['m'] * features['n'] * features['k']
    features['work_per_thread'] = features['work_m'] * features['work_n']
    features['total_threads'] = features['threads_m'] * features['threads_n']
    features['work_per_block'] = features['tile_m'] * features['tile_n']
    features['blocks_needed_m'] = np.ceil(features['m'] / features['tile_m'])
    features['blocks_needed_n'] = np.ceil(features['n'] / features['tile_n'])
    features['total_blocks'] = features['blocks_needed_m'] * features['blocks_needed_n']
    features['parallelism'] = features['total_blocks'] * features['total_threads']
    features['k_iterations'] = np.ceil(features['k'] / features['tile_k'])
    features['flops_per_block'] = 2 * features['tile_m'] * features['tile_n'] * features['tile_k']
    
    # 4. Resource usage (8)
    features['smem_bytes_A'] = features['tile_m'] * features['tile_k'] * 4  # FP32
    features['smem_bytes_B'] = features['tile_n'] * features['tile_k'] * 4
    features['smem_total'] = features['smem_bytes_A'] + features['smem_bytes_B']
    features['register_pressure'] = features['work_per_thread'] * 4  # Estimate
    features['warps_per_block'] = np.ceil(features['total_threads'] / 32)
    features['threads_per_warp_m'] = np.ceil(features['threads_m'] / 32)
    features['threads_per_warp_n'] = np.ceil(features['threads_n'] / 32)
    features['estimated_occupancy'] = np.minimum(
        1.0,
        (features['total_threads'] * features['total_blocks']) / (80 * 1024)  # Assuming 80 SMs
    )
    
    # 5. Alignment and efficiency (12)
    features['tile_m_aligned'] = (features['tile_m'] % 16 == 0).astype(int)
    features['tile_n_aligned'] = (features['tile_n'] % 16 == 0).astype(int)
    features['tile_k_aligned'] = (features['tile_k'] % 32 == 0).astype(int)
    features['m_aligned_to_tile'] = (features['m'] % features['tile_m'] == 0).astype(int)
    features['n_aligned_to_tile'] = (features['n'] % features['tile_n'] == 0).astype(int)
    features['k_aligned_to_tile'] = (features['k'] % features['tile_k'] == 0).astype(int)
    features['threads_power_of_2'] = ((features['total_threads'] & (features['total_threads'] - 1)) == 0).astype(int)
    features['tile_m_div_threads_m'] = features['tile_m'] / features['threads_m']
    features['tile_n_div_threads_n'] = features['tile_n'] / features['threads_n']
    features['work_balance'] = np.abs(features['work_m'] - features['work_n'])
    features['tile_aspect_ratio'] = features['tile_m'] / (features['tile_n'] + 1e-6)
    features['thread_aspect_ratio'] = features['threads_m'] / (features['threads_n'] + 1e-6)
    
    # 6. Cache behavior predictions (6)
    features['l1_reuse_factor'] = features['tile_k'] / 32  # 32-byte cache lines
    features['l2_reuse_factor'] = features['k_iterations']
    features['expected_l1_hits'] = np.minimum(0.95, features['l1_reuse_factor'] / 10)
    features['expected_l2_hits'] = np.minimum(0.80, features['l2_reuse_factor'] / 5)
    features['memory_intensity'] = features['total_work'] / (
        features['m'] * features['k'] + features['k'] * features['n']
    )
    features['arithmetic_intensity'] = features['total_work'] / (
        features['smem_total'] * features['k_iterations']
    )
    
    # 7. Prefetch and vectorization (4)
    features['prefetch_enabled'] = (features['prefetch_stages'] > 0).astype(int)
    features['prefetch_benefit'] = features['prefetch_stages'] * features['k_iterations']
    features['vectorize_benefit'] = features['vectorize_load'] * features['tile_k']
    features['transpose_cost'] = features['transpose_smem'].astype(int) * features['tile_m'] * features['tile_k']
    
    # 8. Problem size categories (8)
    features['is_single_token'] = (features['m'] == 1).astype(int)
    features['is_small_batch'] = ((features['m'] > 1) & (features['m'] <= 32)).astype(int)
    features['is_large_batch'] = (features['m'] > 32).astype(int)
    features['is_small_k'] = (features['k'] < 1024).astype(int)
    features['is_medium_k'] = ((features['k'] >= 1024) & (features['k'] < 4096)).astype(int)
    features['is_large_k'] = (features['k'] >= 4096).astype(int)
    features['is_narrow_n'] = (features['n'] < 2048).astype(int)
    features['is_wide_n'] = (features['n'] >= 4096).astype(int)
    
    # Count features
    base_features = [
        # Problem dimensions
        'm', 'n', 'k',
        # Config parameters
        'tile_m', 'tile_n', 'tile_k', 'threads_m', 'threads_n',
        'work_m', 'work_n', 'prefetch_stages', 'transpose_smem', 'vectorize_load',
        # Derived features (all computed above)
    ] + [col for col in features.columns if col not in ['test_name', 'gflops', 'time_us', 'rank']]
    
    logger.info(f"Engineered {len(base_features)} features")
    
    # Add profiling features if available
    profiling_features = [
        'dram_throughput_pct', 'l1_cache_hit_rate', 'l2_cache_hit_rate',
        'sm_throughput_pct', 'sm_instruction_throughput_pct', 'sm_warps_active_pct',
        'global_load_coalescing_pct', 'global_store_coalescing_pct',
        'smem_bank_conflicts_ld', 'smem_bank_conflicts_st', 'warp_divergence_ratio'
    ]
    
    if include_profiling:
        available_prof = [f for f in profiling_features if f in features.columns]
        logger.info(f"Including {len(available_prof)} profiling features")
        base_features.extend(available_prof)
    
    # Replace inf and NaN values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)
    
    return features, base_features


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Prepare features and target for training"""
    
    # Extract features and target
    X = df[feature_cols].values
    y = df['gflops'].values.reshape(-1, 1)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Data shape: X={X_scaled.shape}, y={y.shape}")
    logger.info(f"Target range: [{y.min():.1f}, {y.max():.1f}] GFLOPS")
    
    return X_scaled, y, scaler


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.0001  # Reduced from 0.001 to prevent NaN
) -> CudaHeuristicNN:
    """Train neural network"""
    
    logger.info("Training neural network...")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = CudaHeuristicNN(input_dim)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return model


def evaluate_model(
    model: CudaHeuristicNN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame
) -> Dict:
    """Evaluate model performance"""
    
    logger.info("Evaluating model...")
    
    # Predictions
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        y_pred = model(X_test_t).numpy()
    
    # Regression metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"MAE: {mae:.2f} GFLOPS")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Top-N hit rate (per test case)
    df_test_copy = df_test.copy()
    df_test_copy['predicted_gflops'] = y_pred
    
    hit_rates = {}
    for top_n in [1, 10, 30]:
        hits = 0
        total = 0
        
        for (test_name, m, n, k), group in df_test_copy.groupby(['test_name', 'm', 'n', 'k']):
            # Sort by actual performance
            group_sorted = group.sort_values('gflops', ascending=False)
            top_actual = set(group_sorted.head(top_n).index)
            
            # Sort by predicted performance
            group_pred = group.sort_values('predicted_gflops', ascending=False)
            top_predicted = set(group_pred.head(top_n).index)
            
            # Check overlap
            overlap = len(top_actual & top_predicted)
            hits += overlap
            total += top_n
        
        hit_rate = hits / total if total > 0 else 0
        hit_rates[f'top_{top_n}'] = hit_rate
        logger.info(f"Top-{top_n} hit rate: {hit_rate*100:.1f}%")
    
    return {
        'r2_score': float(r2),
        'mae': float(mae),
        'mape': float(mape),
        **hit_rates
    }


def export_model(
    model: CudaHeuristicNN,
    scaler: StandardScaler,
    feature_cols: List[str],
    metrics: Dict,
    output_dir: str = '.'
):
    """Export model to ONNX and scaler to binary"""
    
    logger.info("Exporting model...")
    
    # Export to ONNX
    dummy_input = torch.randn(1, len(feature_cols))
    onnx_path = f"{output_dir}/cuda_heuristic_nn.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['features'],
        output_names=['gflops'],
        dynamic_axes={'features': {0: 'batch_size'}},
        opset_version=11
    )
    logger.info(f"Exported ONNX model to {onnx_path}")
    
    # Export scaler
    scaler_path = f"{output_dir}/feature_scaler.bin"
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'feature_names': feature_cols
        }, f)
    logger.info(f"Exported feature scaler to {scaler_path}")
    
    # Export metrics
    metrics_path = f"{output_dir}/training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Exported training metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CUDA GEMM neural network heuristic')
    parser.add_argument('--input', required=True,
                       help='Input benchmark CSV (cuda_gemm_benchmark_data.csv)')
    parser.add_argument('--profiling', default=None,
                       help='Optional profiling CSV (cuda_gemm_profiling_data.csv)')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for model files')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Load benchmark data
    logger.info(f"Loading benchmark data from {args.input}")
    df_bench = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df_bench)} benchmark records")
    
    # Merge with profiling data if available
    if args.profiling and os.path.exists(args.profiling):
        logger.info(f"Loading profiling data from {args.profiling}")
        df_prof = pd.read_csv(args.profiling)
        
        # Merge on config + dimensions
        merge_cols = ['test_name', 'm', 'n', 'k', 'tile_m', 'tile_n', 'tile_k',
                     'threads_m', 'threads_n', 'work_m', 'work_n',
                     'prefetch_stages', 'transpose_smem', 'vectorize_load']
        df = df_bench.merge(df_prof, on=merge_cols, how='left', suffixes=('', '_prof'))
        logger.info(f"Merged data: {len(df)} records with profiling")
        include_profiling = True
    else:
        df = df_bench
        include_profiling = False
    
    # Engineer features
    df_features, feature_cols = engineer_features(df, include_profiling)
    
    # Prepare data
    X, y, scaler = prepare_data(df_features, feature_cols)
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split train into train/val (80/20 of training set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    logger.info(f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Train model
    model = train_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate
    test_indices = df_features.index[len(X_train) + len(X_val):]
    df_test = df_features.loc[test_indices]
    metrics = evaluate_model(model, X_test, y_test, df_test)
    
    # Export
    export_model(model, scaler, feature_cols, metrics, args.output_dir)
    
    logger.info("Training complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

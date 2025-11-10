#!/usr/bin/env python3
"""
CPU GEMM ML Heuristic Trainer

Trains an ONNX neural network to predict optimal micro-kernel variant
for given matrix dimensions, replacing manual heuristics.

Input: cpu_gemm_benchmark_data.csv (from benchmark_cpu_gemm.py)
Output: cpu_gemm_heuristic.onnx

Usage:
    ./train_cpu_gemm_heuristic.py --data cpu_gemm_benchmark_data.csv

Author: David Sanftenberg
Date: November 2025
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
import sys

class CPUGemmDataset(Dataset):
    """PyTorch dataset for CPU GEMM benchmark data"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CPUGemmPredictor(nn.Module):
    """Neural network to predict GFLOPS for CPU GEMM variant"""
    
    def __init__(self, input_size=18, hidden_sizes=[128, 64, 32]):
        super(CPUGemmPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer (predict GFLOPS)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_and_prepare_data(csv_path, train_test_strategy='size_based'):
    """Load CSV and prepare features/targets
    
    Args:
        csv_path: Path to benchmark CSV
        train_test_strategy: 'size_based' (default) or 'random'
            - 'size_based': Train on Qwen ≤32B, validate on 72B + DeepSeek 671B
            - 'random': 80/20 random split (not recommended - doesn't test generalization)
    """
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out failed benchmarks
    df = df[df['success'] == 1]
    print(f"Loaded {len(df)} successful benchmark results")
    
    # Categorize by model size based on test_name
    def get_model_size(test_name):
        """Extract model size from test name for train/test split"""
        if 'Qwen_0_5B' in test_name or 'Qwen_1_5B' in test_name:
            return 'small'  # ≤2B
        elif 'Qwen_4B' in test_name or 'Qwen_7B' in test_name:
            return 'medium'  # 4-7B
        elif 'Qwen_14B' in test_name or 'Qwen_32B' in test_name:
            return 'large'  # 14-32B
        elif 'Qwen_72B' in test_name:
            return 'xlarge'  # 72B (validation/generalization)
        elif 'DeepSeek_671B' in test_name:
            return 'xxlarge'  # 671B (validation/generalization)
        elif 'EdgeCase' in test_name:
            return 'edge'  # Edge cases (validation)
        else:
            return 'unknown'
    
    df['model_size'] = df['test_name'].apply(get_model_size)
    
    # Report distribution
    print(f"\nModel size distribution:")
    for size in ['small', 'medium', 'large', 'xlarge', 'xxlarge', 'edge']:
        count = len(df[df['model_size'] == size])
        if count > 0:
            tests = df[df['model_size'] == size]['test_name'].nunique()
            print(f"  {size:10s}: {count:6d} data points ({tests} test cases)")
    
    if train_test_strategy == 'size_based':
        # Train on ≤32B, validate on 72B + DeepSeek + edge cases
        train_mask = df['model_size'].isin(['small', 'medium', 'large'])
        val_mask = df['model_size'].isin(['xlarge', 'xxlarge', 'edge'])
        
        df_train_all = df[train_mask].copy()
        df_val_unseen = df[val_mask].copy()  # Unseen sizes for generalization
        
        # Further split training data 80/20 for validation during training
        train_test_names = df_train_all['test_name'].unique()
        n_train = int(0.8 * len(train_test_names))
        train_names = np.random.choice(train_test_names, n_train, replace=False)
        
        df_train = df_train_all[df_train_all['test_name'].isin(train_names)]
        df_val_seen = df_train_all[~df_train_all['test_name'].isin(train_names)]
        
        print(f"\nSize-based split:")
        print(f"  Training (≤32B, 80%): {len(df_train)} points from {len(train_names)} test cases")
        print(f"  Validation (≤32B, 20%): {len(df_val_seen)} points (seen sizes)")
        print(f"  Validation (>32B): {len(df_val_unseen)} points (unseen sizes - generalization test)")
        print(f"  Total validation: {len(df_val_seen) + len(df_val_unseen)} points")
        
        return df, df_train, df_val_seen, df_val_unseen
    
    else:  # random split (not recommended)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"\nRandom 80/20 split:")
        print(f"  Training: {len(train_df)} points")
        print(f"  Validation: {len(val_df)} points")
        
        return df, train_df, val_df, pd.DataFrame()  # Empty unseen validation set
    
    # Feature columns (27 total: 24 base + 3 log-scale)
    feature_cols = [
        # Matrix dimensions
        'm', 'n', 'k',
        'problem_size',
        
        # Quantization parameters (NEW - 4 features)
        'quant_block_size',   # Logical: 32, 256, 1
        'bits_per_weight',    # Effective: 4.5, 6.6, 8.5, 16.0
        'bytes_per_block',    # Physical: 18, 34, 210, 2 (CRITICAL for bandwidth!)
        'quant_alignment',    # K % block_size
        
        # Variant parameters
        'is_avx512',
        'tile_m', 'tile_n',
        'tile_area',
        'unroll_k',
        'prefetch_dist',
        
        # Derived cache/alignment features
        'l1_fit_ratio',
        'l2_fit_ratio',
        'm_alignment',
        'n_alignment',
        'm_n_ratio',
        
        # Additional derived features
        'tile_bytes',
        'working_set_bytes',
    ]
    
    # Add log-scale features (neural nets like log-scaled inputs)
    df['log_m'] = np.log10(df['m'] + 1)
    df['log_n'] = np.log10(df['n'] + 1)
    df['log_k'] = np.log10(df['k'] + 1)
    df['log_problem_size'] = np.log10(df['problem_size'] + 1)
    
    feature_cols.extend(['log_m', 'log_n', 'log_k'])
    
    X = df[feature_cols].values
    y = df['gflops'].values.reshape(-1, 1)
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    # Statistics
    print(f"\nGFLOPS statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    
    return X, y, feature_cols

def train_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=64, lr=0.001):
    """Train neural network model"""
    
    print(f"\nTraining model...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    # Create datasets
    train_dataset = CPUGemmDataset(X_train, y_train)
    val_dataset = CPUGemmDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    input_size = X_train.shape[1]
    model = CPUGemmPredictor(input_size=input_size, hidden_sizes=[128, 64, 32])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            rmse_train = np.sqrt(train_loss)
            rmse_val = np.sqrt(val_loss)
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} (RMSE: {rmse_train:.2f}) "
                  f"Val Loss: {val_loss:.4f} (RMSE: {rmse_val:.2f})")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.4f} (RMSE: {np.sqrt(best_val_loss):.2f} GFLOPS)")
    
    return model

def export_to_onnx(model, input_size, output_path='cpu_gemm_heuristic.onnx'):
    """Export trained model to ONNX format"""
    
    print(f"\nExporting model to ONNX: {output_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['gflops'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'gflops': {0: 'batch_size'}
        }
    )
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ ONNX model exported and verified")
    print(f"  Input shape: (batch_size, {input_size})")
    print(f"  Output shape: (batch_size, 1)")

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance on test set"""
    
    print(f"\nEvaluating on test set...")
    
    model.eval()
    with torch.no_grad():
        features = torch.FloatTensor(X_test)
        predictions = model(features).numpy()
    
    # Compute metrics
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse:.2f} GFLOPS")
    print(f"  MAE: {mae:.2f} GFLOPS")
    print(f"  R²: {r2:.4f}")
    
    # Analyze prediction accuracy
    errors = np.abs(predictions - y_test)
    print(f"\nPrediction Error Distribution:")
    print(f"  < 10 GFLOPS: {(errors < 10).sum() / len(errors) * 100:.1f}%")
    print(f"  < 20 GFLOPS: {(errors < 20).sum() / len(errors) * 100:.1f}%")
    print(f"  < 50 GFLOPS: {(errors < 50).sum() / len(errors) * 100:.1f}%")
    
    return rmse, mae, r2

def extract_features(df, feature_cols):
    """Extract feature matrix and target vector from dataframe"""
    
    # Add log-scale features (neural nets like log-scaled inputs)
    df = df.copy()
    df['log_m'] = np.log10(df['m'] + 1)
    df['log_n'] = np.log10(df['n'] + 1)
    df['log_k'] = np.log10(df['k'] + 1)
    
    feature_cols_with_log = feature_cols + ['log_m', 'log_n', 'log_k']
    
    X = df[feature_cols_with_log].values
    y = df['gflops'].values.reshape(-1, 1)
    
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Train CPU GEMM ML heuristic')
    parser.add_argument('--data', required=True, help='Path to benchmark CSV')
    parser.add_argument('--output', default='cpu_gemm_heuristic.onnx', help='Output ONNX file')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--strategy', default='size_based', 
                       choices=['size_based', 'random'],
                       help='Train/test split strategy')
    
    args = parser.parse_args()
    
    # Load data with size-based split
    df_all, df_train, df_val_seen, df_val_unseen = load_and_prepare_data(
        args.data, train_test_strategy=args.strategy)
    
    # Feature columns (23 base features + 3 log-scale = 26 total)
    feature_cols = [
        # Matrix dimensions
        'm', 'n', 'k', 'problem_size',
        
        # Quantization features (NEW)
        'quant_block_size',    # 32, 256, or 1
        'bits_per_weight',     # 4.5, 6.6, 8.5, 16.0
        
        # Variant parameters
        'is_avx512', 'tile_m', 'tile_n', 'tile_area',
        'unroll_k', 'prefetch_dist',
        
        # Derived cache/alignment features
        'l1_fit_ratio', 'l2_fit_ratio',
        'm_alignment', 'n_alignment', 'm_n_ratio',
        'quant_alignment',     # k % quant_block_size (NEW)
        
        # Additional derived features
        'tile_bytes', 'working_set_bytes',
    ]
    
    # Extract features
    X_train, y_train = extract_features(df_train, feature_cols)
    X_val_seen, y_val_seen = extract_features(df_val_seen, feature_cols)
    X_val_unseen, y_val_unseen = extract_features(df_val_unseen, feature_cols)
    
    # Combine validation sets for training monitoring
    X_val = np.vstack([X_val_seen, X_val_unseen])
    y_val = np.vstack([y_val_seen, y_val_unseen])
    
    print(f"\nFinal dataset shapes:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation (seen): {X_val_seen.shape}")
    print(f"  Validation (unseen): {X_val_unseen.shape}")
    print(f"  Total validation: {X_val.shape}")
    
    # GFLOPS statistics
    print(f"\nGFLOPS statistics:")
    print(f"  Training - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, "
          f"Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
    print(f"  Validation (seen) - Mean: {y_val_seen.mean():.2f}, Std: {y_val_seen.std():.2f}")
    print(f"  Validation (unseen) - Mean: {y_val_unseen.mean():.2f}, Std: {y_val_unseen.std():.2f}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_val_seen = scaler.transform(X_val_seen)
    X_val_unseen = scaler.transform(X_val_unseen)
    
    # Save scaler parameters for C++ inference
    np.savez('cpu_gemm_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    print(f"\n✓ Saved scaler parameters to cpu_gemm_scaler.npz")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, 
                       epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    
    # Evaluate on both validation sets
    print(f"\n{'='*70}")
    print(f"EVALUATION ON SEEN SIZES (Qwen ≤32B, 20% held-out)")
    print(f"{'='*70}")
    evaluate_model(model, X_val_seen, y_val_seen, scaler)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION ON UNSEEN SIZES (Qwen 72B, DeepSeek 671B, Edge Cases)")
    print(f"{'='*70}")
    evaluate_model(model, X_val_unseen, y_val_unseen, scaler)
    
    # Export to ONNX
    export_to_onnx(model, X_train.shape[1], args.output)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Model: {args.output}")
    print(f"  Scaler: cpu_gemm_scaler.npz")
    print(f"\nNext steps:")
    print(f"  1. Run: ./export_cpu_heuristic.py --model {args.output}")
    print(f"  2. Run: ./validate_cpu_heuristic.py --model {args.output} --data {args.data}")
    print(f"  3. Integrate CpuGemmHeuristicWeights.h into SmartGemmSearch.cpp")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()


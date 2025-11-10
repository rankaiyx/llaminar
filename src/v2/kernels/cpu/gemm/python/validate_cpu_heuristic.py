#!/usr/bin/env python3
"""
CPU GEMM ML Heuristic Validator

Validates that the trained ML model:
1. Selects variants in top-5 GFLOPS performers (per shape)
2. Generalizes to unseen model sizes (Qwen 72B, DeepSeek 671B)

Metrics:
- Top-1 accuracy: Best predicted = best actual
- Top-5 accuracy: Best predicted in top-5 actual
- Regret: (Best actual GFLOPS - Selected GFLOPS) / Best actual GFLOPS

Usage:
    ./validate_cpu_heuristic.py --model cpu_gemm_heuristic.onnx --data cpu_gemm_benchmark_data.csv

Author: David Sanftenberg
Date: November 2025
"""

import argparse
import pandas as pd
import numpy as np
import torch
import onnxruntime as ort
from collections import defaultdict

def load_model(onnx_path, scaler_path='cpu_gemm_scaler.npz'):
    """Load ONNX model and scaler"""
    
    print(f"Loading model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    
    print(f"Loading scaler from {scaler_path}...")
    scaler_data = np.load(scaler_path)
    mean = scaler_data['mean']
    scale = scaler_data['scale']
    
    return session, mean, scale

def extract_features(df, feature_cols):
    """Extract feature matrix from dataframe"""
    
    df = df.copy()
    
    # Add log-scale features
    df['log_m'] = np.log10(df['m'] + 1)
    df['log_n'] = np.log10(df['n'] + 1)
    df['log_k'] = np.log10(df['k'] + 1)
    
    feature_cols_with_log = feature_cols + ['log_m', 'log_n', 'log_k']
    
    X = df[feature_cols_with_log].values
    
    return X

def normalize_features(X, mean, scale):
    """Normalize features using scaler parameters"""
    return (X - mean) / scale

def predict_gflops(session, X):
    """Predict GFLOPS using ONNX model"""
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    predictions = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    
    return predictions.flatten()

def validate_per_shape(df, session, mean, scale, feature_cols, top_k=5):
    """Validate model selections per shape
    
    For each test shape:
    1. Find actual best variant (ground truth)
    2. Predict GFLOPS for all variants
    3. Select best predicted variant
    4. Check if selected is in top-K actual performers
    
    Returns:
        dict with metrics: top1_acc, top5_acc, mean_regret, etc.
    """
    
    results = {
        'shapes': [],
        'top1_matches': 0,
        'top5_matches': 0,
        'total_shapes': 0,
        'regrets': [],
        'selected_ranks': [],
    }
    
    # Group by test shape
    grouped = df.groupby('test_name')
    
    for test_name, group in grouped:
        results['total_shapes'] += 1
        
        # Extract features for all variants of this shape
        X = extract_features(group, feature_cols)
        X_norm = normalize_features(X, mean, scale)
        
        # Predict GFLOPS
        predicted_gflops = predict_gflops(session, X_norm)
        
        # Actual GFLOPS
        actual_gflops = group['gflops'].values
        
        # Best predicted variant
        best_pred_idx = np.argmax(predicted_gflops)
        selected_actual_gflops = actual_gflops[best_pred_idx]
        selected_variant = group.iloc[best_pred_idx]['variant_name']
        
        # Best actual variant (ground truth)
        best_actual_idx = np.argmax(actual_gflops)
        best_actual_gflops = actual_gflops[best_actual_idx]
        best_actual_variant = group.iloc[best_actual_idx]['variant_name']
        
        # Top-K actual performers
        top_k_indices = np.argsort(actual_gflops)[-top_k:]
        
        # Metrics
        is_top1 = (best_pred_idx == best_actual_idx)
        is_top5 = (best_pred_idx in top_k_indices)
        regret = (best_actual_gflops - selected_actual_gflops) / best_actual_gflops
        
        # Rank of selected variant (1 = best, N = worst)
        selected_rank = len(actual_gflops) - np.where(np.argsort(actual_gflops) == best_pred_idx)[0][0]
        
        results['top1_matches'] += int(is_top1)
        results['top5_matches'] += int(is_top5)
        results['regrets'].append(regret)
        results['selected_ranks'].append(selected_rank)
        
        # Store per-shape results
        shape_result = {
            'test_name': test_name,
            'm': group.iloc[0]['m'],
            'n': group.iloc[0]['n'],
            'k': group.iloc[0]['k'],
            'selected_variant': selected_variant,
            'selected_gflops': selected_actual_gflops,
            'best_variant': best_actual_variant,
            'best_gflops': best_actual_gflops,
            'regret': regret,
            'rank': selected_rank,
            'is_top1': is_top1,
            'is_top5': is_top5,
        }
        results['shapes'].append(shape_result)
    
    # Aggregate metrics
    results['top1_accuracy'] = results['top1_matches'] / results['total_shapes']
    results['top5_accuracy'] = results['top5_matches'] / results['total_shapes']
    results['mean_regret'] = np.mean(results['regrets'])
    results['median_regret'] = np.median(results['regrets'])
    results['max_regret'] = np.max(results['regrets'])
    results['mean_rank'] = np.mean(results['selected_ranks'])
    
    return results

def categorize_by_size(test_name):
    """Categorize test cases by model size"""
    if 'Qwen_0_5B' in test_name or 'Qwen_1_5B' in test_name:
        return 'small'
    elif 'Qwen_4B' in test_name or 'Qwen_7B' in test_name:
        return 'medium'
    elif 'Qwen_14B' in test_name or 'Qwen_32B' in test_name:
        return 'large'
    elif 'Qwen_72B' in test_name:
        return 'xlarge'
    elif 'DeepSeek_671B' in test_name:
        return 'xxlarge'
    elif 'EdgeCase' in test_name:
        return 'edge'
    return 'unknown'

def print_results(results, category_name="All"):
    """Pretty print validation results"""
    
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS: {category_name}")
    print(f"{'='*70}")
    print(f"Total shapes: {results['total_shapes']}")
    print(f"\nAccuracy Metrics:")
    print(f"  Top-1 accuracy: {results['top1_accuracy']:.2%} ({results['top1_matches']}/{results['total_shapes']})")
    print(f"  Top-5 accuracy: {results['top5_accuracy']:.2%} ({results['top5_matches']}/{results['total_shapes']})")
    print(f"\nSelection Quality:")
    print(f"  Mean regret: {results['mean_regret']:.2%}")
    print(f"  Median regret: {results['median_regret']:.2%}")
    print(f"  Max regret: {results['max_regret']:.2%}")
    print(f"  Mean rank: {results['mean_rank']:.1f}")
    
    # Show worst cases (highest regret)
    sorted_shapes = sorted(results['shapes'], key=lambda x: x['regret'], reverse=True)
    print(f"\nWorst 5 Selections (highest regret):")
    for i, shape in enumerate(sorted_shapes[:5]):
        print(f"  {i+1}. {shape['test_name']}")
        print(f"     Selected: {shape['selected_variant']} → {shape['selected_gflops']:.2f} GFLOPS")
        print(f"     Best: {shape['best_variant']} → {shape['best_gflops']:.2f} GFLOPS")
        print(f"     Regret: {shape['regret']:.2%}, Rank: {shape['rank']}")
    
    # Show best cases (top-1 matches)
    top1_shapes = [s for s in results['shapes'] if s['is_top1']]
    if top1_shapes:
        print(f"\nPerfect Selections (Top-1 matches): {len(top1_shapes)}")
        for shape in top1_shapes[:5]:  # Show first 5
            print(f"  ✓ {shape['test_name']}: {shape['selected_variant']} → {shape['selected_gflops']:.2f} GFLOPS")

def main():
    parser = argparse.ArgumentParser(description='Validate CPU GEMM ML heuristic')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--data', required=True, help='Path to benchmark CSV')
    parser.add_argument('--scaler', default='cpu_gemm_scaler.npz', help='Path to scaler parameters')
    parser.add_argument('--top-k', type=int, default=5, help='Top-K accuracy threshold')
    
    args = parser.parse_args()
    
    # Load model
    session, mean, scale = load_model(args.model, args.scaler)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)
    df = df[df['success'] == 1]  # Only successful benchmarks
    print(f"Loaded {len(df)} successful benchmark results")
    
    # Categorize by size
    df['model_size'] = df['test_name'].apply(categorize_by_size)
    
    # Feature columns (must match training)
    feature_cols = [
        'm', 'n', 'k', 'problem_size',
        'is_avx512', 'tile_m', 'tile_n', 'tile_area',
        'unroll_k', 'prefetch_dist',
        'l1_fit_ratio', 'l2_fit_ratio',
        'm_alignment', 'n_alignment', 'm_n_ratio',
        'tile_bytes', 'working_set_bytes',
    ]
    
    # Validate on all shapes
    print(f"\nValidating on all {df['test_name'].nunique()} test shapes...")
    all_results = validate_per_shape(df, session, mean, scale, feature_cols, top_k=args.top_k)
    print_results(all_results, "All Shapes")
    
    # Validate by category
    for category in ['small', 'medium', 'large', 'xlarge', 'xxlarge', 'edge']:
        df_cat = df[df['model_size'] == category]
        if len(df_cat) == 0:
            continue
        
        cat_results = validate_per_shape(df_cat, session, mean, scale, feature_cols, top_k=args.top_k)
        
        category_names = {
            'small': 'SEEN: Qwen ≤2B',
            'medium': 'SEEN: Qwen 4-7B',
            'large': 'SEEN: Qwen 14-32B',
            'xlarge': 'UNSEEN: Qwen 72B',
            'xxlarge': 'UNSEEN: DeepSeek 671B',
            'edge': 'UNSEEN: Edge Cases',
        }
        
        print_results(cat_results, category_names.get(category, category))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    # Generalization test: Unseen vs Seen
    df_seen = df[df['model_size'].isin(['small', 'medium', 'large'])]
    df_unseen = df[df['model_size'].isin(['xlarge', 'xxlarge', 'edge'])]
    
    if len(df_seen) > 0:
        seen_results = validate_per_shape(df_seen, session, mean, scale, feature_cols, top_k=args.top_k)
        print(f"\nSeen sizes (Qwen ≤32B):")
        print(f"  Top-1: {seen_results['top1_accuracy']:.2%}")
        print(f"  Top-5: {seen_results['top5_accuracy']:.2%}")
        print(f"  Mean regret: {seen_results['mean_regret']:.2%}")
    
    if len(df_unseen) > 0:
        unseen_results = validate_per_shape(df_unseen, session, mean, scale, feature_cols, top_k=args.top_k)
        print(f"\nUnseen sizes (Qwen 72B, DeepSeek 671B, Edge Cases):")
        print(f"  Top-1: {unseen_results['top1_accuracy']:.2%}")
        print(f"  Top-5: {unseen_results['top5_accuracy']:.2%}")
        print(f"  Mean regret: {unseen_results['mean_regret']:.2%}")
        
        # Generalization gap
        if len(df_seen) > 0:
            gap_top1 = seen_results['top1_accuracy'] - unseen_results['top1_accuracy']
            gap_top5 = seen_results['top5_accuracy'] - unseen_results['top5_accuracy']
            print(f"\nGeneralization gap (seen - unseen):")
            print(f"  Top-1: {gap_top1:+.2%}")
            print(f"  Top-5: {gap_top5:+.2%}")
            
            if abs(gap_top1) < 0.1:  # Less than 10% gap
                print(f"  ✓ Good generalization!")
            else:
                print(f"  ⚠ Generalization may need improvement")
    
    print(f"\n{'='*70}")
    print(f"Validation complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

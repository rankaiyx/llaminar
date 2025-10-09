#!/usr/bin/env python3
"""
Analyze Q_PROJECTION error distribution to check if max_abs=0.11 is hiding systematic errors.

This script loads PyTorch and C++ snapshots and performs detailed statistical analysis
of the Q_PROJECTION differences to identify:
1. Whether errors are uniform or concentrated in specific regions
2. Per-head error statistics
3. Per-dimension error patterns
4. Outlier detection and localization
"""

import numpy as np
import sys
from pathlib import Path

def load_snapshots(pytorch_path="pytorch_layer_captures.npz", cpp_path="snapshots.npz"):
    """Load both PyTorch and C++ snapshots."""
    try:
        pytorch = np.load(pytorch_path)
        cpp = np.load(cpp_path)
        return pytorch, cpp
    except FileNotFoundError as e:
        print(f"Error: Could not find snapshot file: {e}")
        sys.exit(1)

def analyze_projection(pytorch_data, cpp_data, proj_name, n_heads, head_dim):
    """
    Detailed analysis of projection errors.
    
    Args:
        pytorch_data: PyTorch reference tensor [batch, seq_len, hidden_dim]
        cpp_data: C++ implementation tensor [batch, seq_len, hidden_dim]
        proj_name: Name of projection (Q/K/V)
        n_heads: Number of heads
        head_dim: Dimension per head
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {proj_name}_PROJECTION Error Distribution")
    print(f"{'='*80}")
    
    # Basic shape validation
    if pytorch_data.shape != cpp_data.shape:
        print(f"ERROR: Shape mismatch! PyTorch: {pytorch_data.shape}, C++: {cpp_data.shape}")
        return
    
    batch, seq_len, hidden_dim = pytorch_data.shape
    print(f"Shape: {pytorch_data.shape} (batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim})")
    print(f"Expected heads: {n_heads}, head_dim: {head_dim}, total: {n_heads * head_dim}")
    
    # Calculate differences
    diff = cpp_data - pytorch_data
    abs_diff = np.abs(diff)
    
    # Overall statistics
    print(f"\n--- Overall Statistics ---")
    print(f"Max absolute error: {np.max(abs_diff):.6f}")
    print(f"Mean absolute error: {np.mean(abs_diff):.6f}")
    print(f"Std absolute error: {np.std(abs_diff):.6f}")
    print(f"Median absolute error: {np.median(abs_diff):.6f}")
    
    # Percentile analysis
    print(f"\n--- Error Percentiles ---")
    for p in [50, 75, 90, 95, 99, 99.9]:
        val = np.percentile(abs_diff, p)
        print(f"  {p:5.1f}th percentile: {val:.6f}")
    
    # Find outliers (errors > mean + 3*std)
    threshold = np.mean(abs_diff) + 3 * np.std(abs_diff)
    outliers = abs_diff > threshold
    n_outliers = np.sum(outliers)
    outlier_pct = 100.0 * n_outliers / abs_diff.size
    
    print(f"\n--- Outlier Analysis (threshold={threshold:.6f}) ---")
    print(f"Number of outliers: {n_outliers} / {abs_diff.size} ({outlier_pct:.2f}%)")
    
    if n_outliers > 0:
        # Find top 10 outliers
        flat_abs_diff = abs_diff.flatten()
        flat_pytorch = pytorch_data.flatten()
        flat_cpp = cpp_data.flatten()
        
        outlier_indices = np.argsort(flat_abs_diff)[-min(10, n_outliers):][::-1]
        
        print(f"\n--- Top {len(outlier_indices)} Errors ---")
        print(f"{'Rank':<6} {'Index':<12} {'PyTorch':<12} {'C++':<12} {'Abs Diff':<12} {'Rel %':<10}")
        print("-" * 70)
        
        for rank, idx in enumerate(outlier_indices, 1):
            pt_val = flat_pytorch[idx]
            cpp_val = flat_cpp[idx]
            error = flat_abs_diff[idx]
            rel_pct = 100.0 * error / max(abs(pt_val), 1e-10)
            
            # Convert flat index to 3D coordinates
            b = idx // (seq_len * hidden_dim)
            remainder = idx % (seq_len * hidden_dim)
            s = remainder // hidden_dim
            h = remainder % hidden_dim
            
            print(f"{rank:<6} [{b},{s},{h}]  {pt_val:>11.6f} {cpp_val:>11.6f} {error:>11.6f} {rel_pct:>9.2f}%")
    
    # Per-head analysis (if tensor can be reshaped to heads)
    if hidden_dim == n_heads * head_dim:
        print(f"\n--- Per-Head Error Analysis ---")
        print(f"Reshaping to [batch, seq_len, n_heads={n_heads}, head_dim={head_dim}]")
        
        pt_heads = pytorch_data.reshape(batch, seq_len, n_heads, head_dim)
        cpp_heads = cpp_data.reshape(batch, seq_len, n_heads, head_dim)
        
        print(f"{'Head':<6} {'Max Abs':<12} {'Mean Abs':<12} {'Std Abs':<12} {'Median Abs':<12}")
        print("-" * 60)
        
        head_max_errors = []
        for h in range(n_heads):
            head_diff = np.abs(cpp_heads[:, :, h, :] - pt_heads[:, :, h, :])
            max_err = np.max(head_diff)
            mean_err = np.mean(head_diff)
            std_err = np.std(head_diff)
            median_err = np.median(head_diff)
            
            head_max_errors.append((h, max_err))
            print(f"{h:<6} {max_err:>11.6f} {mean_err:>11.6f} {std_err:>11.6f} {median_err:>11.6f}")
        
        # Find heads with highest errors
        head_max_errors.sort(key=lambda x: x[1], reverse=True)
        print(f"\nHeads sorted by max error:")
        for rank, (head_idx, max_err) in enumerate(head_max_errors[:5], 1):
            print(f"  {rank}. Head {head_idx}: max_abs={max_err:.6f}")
    
    # Per-dimension analysis (averaged across heads)
    if hidden_dim == n_heads * head_dim:
        print(f"\n--- Per-Dimension Error Analysis (within head_dim={head_dim}) ---")
        
        pt_heads = pytorch_data.reshape(batch, seq_len, n_heads, head_dim)
        cpp_heads = cpp_data.reshape(batch, seq_len, n_heads, head_dim)
        
        dim_errors = []
        for d in range(head_dim):
            dim_diff = np.abs(cpp_heads[:, :, :, d] - pt_heads[:, :, :, d])
            max_err = np.max(dim_diff)
            mean_err = np.mean(dim_diff)
            dim_errors.append((d, max_err, mean_err))
        
        dim_errors.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top 10 dimensions by max error:")
        print(f"{'Rank':<6} {'Dim':<6} {'Max Abs':<12} {'Mean Abs':<12}")
        print("-" * 40)
        for rank, (dim, max_err, mean_err) in enumerate(dim_errors[:10], 1):
            print(f"{rank:<6} {dim:<6} {max_err:>11.6f} {mean_err:>11.6f}")
    
    # Check for systematic patterns in sequence positions
    print(f"\n--- Per-Sequence-Position Error Analysis ---")
    seq_errors = []
    for s in range(seq_len):
        pos_diff = np.abs(cpp_data[:, s, :] - pytorch_data[:, s, :])
        max_err = np.max(pos_diff)
        mean_err = np.mean(pos_diff)
        seq_errors.append((s, max_err, mean_err))
    
    seq_errors.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 sequence positions by max error:")
    print(f"{'Rank':<6} {'Pos':<6} {'Max Abs':<12} {'Mean Abs':<12}")
    print("-" * 40)
    for rank, (pos, max_err, mean_err) in enumerate(seq_errors[:10], 1):
        print(f"{rank:<6} {pos:<6} {max_err:>11.6f} {mean_err:>11.6f}")
    
    # Histogram of error magnitudes
    print(f"\n--- Error Distribution Histogram ---")
    bins = [0, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, np.inf]
    hist, _ = np.histogram(abs_diff.flatten(), bins=bins)
    total = abs_diff.size
    
    print(f"{'Range':<20} {'Count':<12} {'Percentage':<12}")
    print("-" * 44)
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        count = hist[i]
        pct = 100.0 * count / total
        range_str = f"[{lower:.3f}, {upper:.3f})"
        print(f"{range_str:<20} {count:<12} {pct:>10.2f}%")

def main():
    print("Q_PROJECTION Error Distribution Analysis")
    print("=" * 80)
    
    # Load snapshots
    pytorch, cpp = load_snapshots()
    
    # Check available keys
    print("\nAvailable PyTorch snapshots:", list(pytorch.keys()))
    print("Available C++ snapshots:", list(cpp.keys()))
    
    # Analyze Q_PROJECTION at layer 0
    stage_name = "Q_PROJECTION_layer0"
    
    if stage_name not in pytorch:
        print(f"\nERROR: {stage_name} not found in PyTorch snapshots!")
        return
    
    if stage_name not in cpp:
        print(f"\nERROR: {stage_name} not found in C++ snapshots!")
        return
    
    # Qwen-0.5B configuration
    # 14 attention heads, 64 head_dim = 896 total
    n_heads = 14
    head_dim = 64
    
    pytorch_q = pytorch[stage_name]
    cpp_q = cpp[stage_name]
    
    analyze_projection(pytorch_q, cpp_q, "Q", n_heads, head_dim)
    
    # Also analyze K_PROJECTION if available
    k_stage = "K_PROJECTION_layer0"
    if k_stage in pytorch and k_stage in cpp:
        n_kv_heads = 2  # Qwen-0.5B uses GQA with 2 KV heads
        kv_head_dim = 64
        
        analyze_projection(pytorch[k_stage], cpp[k_stage], "K", n_kv_heads, kv_head_dim)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export feature scaler to C++-readable text format

The scaler is saved as a dict with 'mean' and 'std' arrays (101 values each).
This script exports them as a simple text file that C++ can read.
"""

import pickle
import sys

def main():
    # Load scaler
    with open('feature_scaler.bin', 'rb') as f:
        scaler_dict = pickle.load(f)
    
    mean = scaler_dict['mean']
    std = scaler_dict['std']
    feature_names = scaler_dict.get('feature_names', [])
    
    print(f"Loaded scaler with {len(mean)} features")
    print(f"Feature names: {len(feature_names)}")
    
    # Export to text format
    with open('src/v2/kernels/cuda/cuda_heuristic_scaler.txt', 'w') as f:
        # Header
        f.write(f"# Feature scaler parameters (StandardScaler)\n")
        f.write(f"# Format: num_features mean[0] mean[1] ... std[0] std[1] ...\n")
        f.write(f"{len(mean)}\n")
        
        # Mean values
        f.write("# Mean values\n")
        for val in mean:
            f.write(f"{val}\n")
        
        # Std values
        f.write("# Std (scale) values\n")
        for val in std:
            f.write(f"{val}\n")
    
    print(f"Exported scaler to src/v2/kernels/cuda/cuda_heuristic_scaler.txt")
    print(f"Format: {len(mean)} features, mean + std arrays")

if __name__ == '__main__':
    main()

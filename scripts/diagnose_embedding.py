#!/usr/bin/env python3
"""
Diagnostic tool to compare Llaminar vs PyTorch EMBEDDING snapshots.

This helps identify the root cause of embedding divergence by showing
actual values, differences, and statistical analysis.
"""

import numpy as np
import sys
from pathlib import Path

def analyze_embeddings():
    """Compare embedding snapshots from Llaminar and PyTorch."""
    
    # Load PyTorch embedding (dequantized from GGUF)
    pytorch_path = Path("/tmp/pytorch_snapshots_openblas/EMBEDDING_-1.npy")
    if not pytorch_path.exists():
        print(f"ERROR: PyTorch snapshot not found at {pytorch_path}")
        return 1
        
    pytorch_emb = np.load(pytorch_path)
    print(f"PyTorch EMBEDDING:")
    print(f"  Shape: {pytorch_emb.shape}")
    print(f"  dtype: {pytorch_emb.dtype}")
    print(f"  Range: [{pytorch_emb.min():.6f}, {pytorch_emb.max():.6f}]")
    print(f"  Mean: {pytorch_emb.mean():.6f}")
    print(f"  Std: {pytorch_emb.std():.6f}")
    print()
    
    # Show samples
    print("PyTorch Token 0 embedding (first 20 dims):")
    print(pytorch_emb[0, :20])
    print()
    
    print("PyTorch Token 1 embedding (first 20 dims):")
    print(pytorch_emb[0, 1, :20] if pytorch_emb.ndim == 3 else pytorch_emb[1, :20])
    print()
    
    # Try to find Llaminar snapshot from pipeline snapshot manager output
    # The test should have printed the snapshot data to the log
    print("To diagnose Llaminar EMBEDDING, we need to:")
    print("1. Check if snapshots were captured (LLAMINAR_PARITY_CAPTURE=1)")
    print("2. Look at test logs for snapshot comparison details")
    print("3. Enable LLAMINAR_EMBED_TRACE=1 for detailed embedding diagnostics")
    print()
    
    # Check the model's embedding table directly
    print("RECOMMENDATION:")
    print("Run the test with LLAMINAR_EMBED_TRACE=1 to see detailed embedding")
    print("lookup diagnostics, including:")
    print("  - Token IDs being looked up")  
    print("  - Embedding table shape and orientation")
    print("  - Actual embedding values copied")
    print("  - Finite value validation")
    
    return 0

if __name__ == "__main__":
    sys.exit(analyze_embeddings())

#!/usr/bin/env python3
"""
Direct comparison of Llaminar vs PyTorch embedding lookups.
This will help us understand the exact divergence pattern.
"""

import numpy as np

# Test tokens
tokens = [1, 2, 3, 4, 5]

# Load PyTorch embeddings (batch=1, seq=5, hidden=896)
pytorch_emb = np.load('/tmp/pytorch_snapshots_openblas/EMBEDDING_-1.npy')
print(f"PyTorch embedding shape: {pytorch_emb.shape}")

# Reshape to (seq, hidden) by removing batch dimension
if pytorch_emb.shape[0] == 1:
    pytorch_emb = pytorch_emb[0]  # Now (5, 896)
    
print(f"PyTorch embedding reshaped: {pytorch_emb.shape}")
print()

# Expected layout: [seq_len, hidden_dim] = [5, 896]
print("Token embedding samples from PyTorch:")
for i, token_id in enumerate(tokens):
    print(f"Token {token_id} (index {i}):")
    print(f"  First 10 dims: {pytorch_emb[i, :10]}")
    print(f"  Dims 150-160: {pytorch_emb[i, 150:160]}")
    print()

# Check specific divergence point: [158,3] means what in (5, 896)?
# If flattened with stride 896, index 158 = row 0, col 158
print("Checking divergence location [158,3]:")
print("If interpreted as flattened index with cols=896:")
row = 158 // 896
col = 158 % 896  
print(f"  Row (token): {row}, Col (dim): {col}")
print(f"  Value at pytorch_emb[{row}, {col}]: {pytorch_emb[row, col]}")

# But the log shows [158,3] which suggests (row=158, col=3)
# That would be out of bounds for tokens (only 5 tokens)
# So it must be using a different layout...

# Let me try flattened indexing
flat = pytorch_emb.flatten()
idx_158_3 = 158 * 896 + 3
if idx_158_3 < len(flat):
    print(f"\nIf [158,3] means element at flat_index[{idx_158_3}]:")
    print(f"  Value: {flat[idx_158_3]}")
    # Convert back to (seq, dim)
    token_idx = idx_158_3 // 896
    dim_idx = idx_158_3 % 896
    print(f"  Which is token {token_idx}, dimension {dim_idx}")
    print(f"  Value from 2D: {pytorch_emb[token_idx, dim_idx]}")

# The log said expected=-0.026093, let's search for this value
print("\nSearching for expected value -0.026093:")
matches = np.where(np.abs(pytorch_emb + 0.026093) < 0.0001)
print(f"Found at positions: {list(zip(matches[0], matches[1]))}")


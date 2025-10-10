#!/usr/bin/env python3
"""
Generate PyTorch decode reference snapshots for parity testing.

This module generates stage-by-stage snapshots for incremental decode steps,
allowing Llaminar to validate decode path correctness against PyTorch.

@author David Sanftenberg
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

from reference import ModelRegistry
from reference.generate_test_snapshots import PipelineStageCapture


def generate_decode_reference(
    model_path: str,
    prefill_tokens: List[int],
    num_decode_steps: int,
    verbose: bool = False
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Generate PyTorch reference snapshots for decode steps.
    
    This runs:
    1. Prefill with the given tokens (warming up KV cache)
    2. N decode steps, capturing all pipeline stages for each step
    
    Args:
        model_path: Path to GGUF model file
        prefill_tokens: Token IDs for prefill phase (e.g., [1,2,3,4,5])
        num_decode_steps: Number of decode steps to generate
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping step_idx -> {stage_name: numpy_array}
        
    Example:
        snapshots = generate_decode_reference(
            "models/qwen.gguf",
            prefill_tokens=[1,2,3,4,5],
            num_decode_steps=3
        )
        # snapshots[0] = snapshots for decode step 0
        # snapshots[0]['EMBEDDING'] = embedding output for first decode token
        # snapshots[0]['ATTENTION_OUTPUT_0'] = layer 0 attention output
    """
    if verbose:
        print(f"Generating decode reference snapshots...")
        print(f"  Model: {model_path}")
        print(f"  Prefill tokens: {prefill_tokens}")
        print(f"  Decode steps: {num_decode_steps}")
    
    # Load model
    capturer = PipelineStageCapture(model_path, verbose=verbose)
    capturer.load_model()
    
    # Step 1: Run prefill to warm up KV cache
    if verbose:
        print(f"\nStep 1: Running prefill with {len(prefill_tokens)} tokens...")
    
    # For prefill, we need to run the model and get logits to sample next token
    prefill_snapshots = capturer.capture_stages(prefill_tokens)
    
    # Get logits from prefill (last token position)
    if 'LM_HEAD' not in prefill_snapshots:
        raise RuntimeError("Prefill did not produce LM_HEAD logits")
    
    prefill_logits = prefill_snapshots['LM_HEAD']
    # Shape: (1, seq_len, vocab_size) - take last position for next token
    last_token_logits = prefill_logits[0, -1, :]  # (vocab_size,)
    
    # Greedy sampling: pick token with highest logit
    next_token = int(np.argmax(last_token_logits))
    
    if verbose:
        print(f"  ✓ Prefill complete, sampled next token: {next_token}")
    
    # Step 2: Run decode steps
    decode_step_snapshots = {}
    current_tokens = prefill_tokens.copy()
    
    for step_idx in range(num_decode_steps):
        if verbose:
            print(f"\nStep {step_idx + 2}: Decode step {step_idx} (token {next_token})...")
        
        # Append token to context
        current_tokens.append(next_token)
        
        # Run model with full context (PyTorch doesn't have KV cache abstraction in our wrapper)
        # The model will recompute everything, but we only care about the last position
        step_snapshots_full = capturer.capture_stages(current_tokens)
        
        # Extract only the last position's activations for decode validation
        # For decode, Llaminar processes 1 token at a time, so we extract position -1
        decode_snapshots = {}
        
        for stage_name, tensor in step_snapshots_full.items():
            # Tensor shapes:
            # - EMBEDDING: (1, seq_len, hidden_dim)
            # - ATTENTION_*: (1, seq_len, hidden_dim)
            # - LM_HEAD: (1, seq_len, vocab_size)
            
            # Extract last position (the new decode token)
            if len(tensor.shape) == 3:  # (batch, seq, dim)
                decode_snapshots[stage_name] = tensor[:, -1:, :]  # Keep as (1, 1, dim)
            elif len(tensor.shape) == 2:  # (batch, dim) - shouldn't happen but handle it
                decode_snapshots[stage_name] = tensor
            else:
                # Unknown shape, keep as-is
                decode_snapshots[stage_name] = tensor
        
        decode_step_snapshots[step_idx] = decode_snapshots
        
        # Sample next token for next iteration
        if step_idx < num_decode_steps - 1:
            decode_logits = decode_snapshots['LM_HEAD']
            # Shape after extraction: (1, 1, vocab_size)
            last_logits = decode_logits[0, 0, :]  # (vocab_size,)
            next_token = int(np.argmax(last_logits))
            
            if verbose:
                print(f"  ✓ Decode step {step_idx} complete, sampled next token: {next_token}")
        else:
            if verbose:
                print(f"  ✓ Decode step {step_idx} complete (final step)")
    
    if verbose:
        print(f"\n✓ Generated snapshots for {num_decode_steps} decode steps")
    
    # Cleanup
    capturer.cleanup()
    
    return decode_step_snapshots


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate PyTorch decode reference snapshots"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--prefill-tokens",
        default="1,2,3,4,5",
        help="Comma-separated prefill token IDs (default: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Number of decode steps to generate (default: 3)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="pytorch_snapshots_mapped",
        help="Output directory (default: pytorch_snapshots_mapped)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse prefill tokens
    prefill_tokens = [int(t.strip()) for t in args.prefill_tokens.split(',')]
    
    # Generate snapshots
    snapshots = generate_decode_reference(
        args.model,
        prefill_tokens,
        args.num_steps,
        verbose=args.verbose
    )
    
    # Save to disk
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving snapshots to {output_dir}...")
    for step_idx, step_snapshots in snapshots.items():
        step_dir = output_dir / f"decode_step_{step_idx}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        for stage_name, tensor in step_snapshots.items():
            output_path = step_dir / f"{stage_name}.npy"
            np.save(output_path, tensor)
        
        if args.verbose:
            print(f"  ✓ Saved {len(step_snapshots)} snapshots to decode_step_{step_idx}/")
    
    print(f"\n✓ Complete! Generated snapshots for {len(snapshots)} decode steps")
